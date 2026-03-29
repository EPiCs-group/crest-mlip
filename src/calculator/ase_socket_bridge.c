/*
 * This file is part of crest.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * Modifications for MLIP support:
 * Copyright (C) 2024-2026 Alexander Kolganov
 *
 * crest is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * crest is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with crest.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * ase_socket_bridge.c — TCP socket client for ASE calculator interface.
 *
 * Protocol: length-prefixed JSON over TCP.
 * Each message: 4-byte uint32 (network byte order) payload length, then JSON.
 * All coordinates in Bohr, energies in Hartree (CREST native units).
 * The Python server handles ASE unit conversion internally.
 */

#include "ase_socket_bridge.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <errno.h>

/* ------------------------------------------------------------------ */
/* Internal context                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    int sockfd;
    pthread_mutex_t mutex;
} ASESocketContext;

/* ------------------------------------------------------------------ */
/* Helper: send exactly n bytes                                        */
/* ------------------------------------------------------------------ */

static int send_all(int fd, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = send(fd, p, remaining, 0);
        if (n <= 0) return -1;
        p += n;
        remaining -= (size_t)n;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Helper: receive exactly n bytes                                     */
/* ------------------------------------------------------------------ */

static int recv_all(int fd, void* buf, size_t len) {
    char* p = (char*)buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = recv(fd, p, remaining, 0);
        if (n <= 0) return -1;
        p += n;
        remaining -= (size_t)n;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Helper: send length-prefixed JSON message                           */
/* ------------------------------------------------------------------ */

static int send_msg(int fd, const char* json, size_t json_len) {
    uint32_t net_len = htonl((uint32_t)json_len);
    if (send_all(fd, &net_len, 4) != 0) return -1;
    if (send_all(fd, json, json_len) != 0) return -1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Helper: receive length-prefixed JSON message                        */
/* Returns malloc'd buffer (caller must free). Sets *out_len.          */
/* ------------------------------------------------------------------ */

static char* recv_msg(int fd, size_t* out_len) {
    uint32_t net_len;
    if (recv_all(fd, &net_len, 4) != 0) return NULL;

    uint32_t payload_len = ntohl(net_len);
    if (payload_len == 0 || payload_len > 256 * 1024 * 1024) return NULL;

    char* buf = (char*)malloc(payload_len + 1);
    if (!buf) return NULL;

    if (recv_all(fd, buf, payload_len) != 0) {
        free(buf);
        return NULL;
    }
    buf[payload_len] = '\0';
    if (out_len) *out_len = payload_len;
    return buf;
}

/* ------------------------------------------------------------------ */
/* Simple JSON helpers (for predictable protocol format)               */
/* ------------------------------------------------------------------ */

/* Find "key": and return pointer to the value start */
static const char* json_find_key(const char* json, const char* key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char* p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    return p;
}

/* Extract a string value (returns pointer into json, writes to out) */
static int json_get_string(const char* json, const char* key,
                           char* out, size_t out_len) {
    const char* p = json_find_key(json, key);
    if (!p || *p != '"') return -1;
    p++; /* skip opening quote */
    size_t i = 0;
    while (*p && *p != '"' && i < out_len - 1) {
        out[i++] = *p++;
    }
    out[i] = '\0';
    return 0;
}

/* Extract a double value */
static int json_get_double(const char* json, const char* key, double* out) {
    const char* p = json_find_key(json, key);
    if (!p) return -1;
    char* end;
    *out = strtod(p, &end);
    if (end == p) return -1;
    return 0;
}

/* Extract an array of doubles: "key":[1.0,2.0,...] */
static int json_get_double_array(const char* json, const char* key,
                                 double* out, int count) {
    const char* p = json_find_key(json, key);
    if (!p || *p != '[') return -1;
    p++; /* skip '[' */

    for (int i = 0; i < count; i++) {
        while (*p == ' ' || *p == '\t' || *p == '\n') p++;
        char* end;
        out[i] = strtod(p, &end);
        if (end == p) return -1;
        p = end;
        while (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n') p++;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* ase_socket_connect                                                  */
/* ------------------------------------------------------------------ */

ase_socket_handle_t ase_socket_connect(const char* host, int port,
                                        char* err_msg, int err_len) {
    int sockfd = -1;
    struct addrinfo hints, *res = NULL, *rp;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;     /* IPv4 or IPv6 */
    hints.ai_socktype = SOCK_STREAM; /* TCP */

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    int gai_err = getaddrinfo(host, port_str, &hints, &res);
    if (gai_err != 0) {
        snprintf(err_msg, err_len, "Failed to resolve host '%s': %s",
                 host, gai_strerror(gai_err));
        return NULL;
    }

    /* Try each address until one succeeds */
    for (rp = res; rp != NULL; rp = rp->ai_next) {
        sockfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sockfd < 0) continue;

        if (connect(sockfd, rp->ai_addr, rp->ai_addrlen) == 0) break;

        close(sockfd);
        sockfd = -1;
    }
    freeaddrinfo(res);

    if (sockfd < 0) {
        snprintf(err_msg, err_len,
                 "Failed to connect to %s:%d: %s", host, port, strerror(errno));
        return NULL;
    }

    /* Disable Nagle's algorithm for lower latency */
    int flag = 1;
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    /* Send init message */
    const char* init_msg = "{\"type\":\"init\",\"version\":1}";
    if (send_msg(sockfd, init_msg, strlen(init_msg)) != 0) {
        snprintf(err_msg, err_len, "Failed to send init message");
        close(sockfd);
        return NULL;
    }

    /* Receive ready response */
    size_t resp_len;
    char* resp = recv_msg(sockfd, &resp_len);
    if (!resp) {
        snprintf(err_msg, err_len, "Failed to receive ready response from server");
        close(sockfd);
        return NULL;
    }

    char msg_type[32];
    if (json_get_string(resp, "type", msg_type, sizeof(msg_type)) != 0 ||
        strcmp(msg_type, "ready") != 0) {
        snprintf(err_msg, err_len, "Unexpected server response: %.*s",
                 (int)(resp_len < 200 ? resp_len : 200), resp);
        free(resp);
        close(sockfd);
        return NULL;
    }
    free(resp);

    /* Create context */
    ASESocketContext* ctx = (ASESocketContext*)malloc(sizeof(ASESocketContext));
    if (!ctx) {
        snprintf(err_msg, err_len, "Memory allocation failed");
        close(sockfd);
        return NULL;
    }
    ctx->sockfd = sockfd;
    pthread_mutex_init(&ctx->mutex, NULL);

    return (ase_socket_handle_t)ctx;
}

/* ------------------------------------------------------------------ */
/* ase_socket_engrad                                                   */
/* ------------------------------------------------------------------ */

int ase_socket_engrad(ase_socket_handle_t handle,
                      int nat,
                      const double* positions_bohr,
                      const int* atomic_numbers,
                      int charge, int uhf,
                      double* energy_out,
                      double* gradient_out,
                      char* err_msg, int err_len) {
    if (!handle) {
        snprintf(err_msg, err_len, "NULL socket handle");
        return 1;
    }

    ASESocketContext* ctx = (ASESocketContext*)handle;
    int rc = 0;

    pthread_mutex_lock(&ctx->mutex);

    /* Build JSON request.
     * Estimate buffer: header ~100 + atomic_numbers ~12*nat + positions ~25*3*nat
     * Plus gradient array in response: ~25*3*nat */
    size_t buf_size = 256 + (size_t)nat * 100;
    char* buf = (char*)malloc(buf_size);
    if (!buf) {
        snprintf(err_msg, err_len, "Memory allocation failed (buf_size=%zu)", buf_size);
        pthread_mutex_unlock(&ctx->mutex);
        return 1;
    }

    /* Build JSON: header */
    int pos = snprintf(buf, buf_size,
        "{\"type\":\"engrad\",\"nat\":%d,\"charge\":%d,\"uhf\":%d,"
        "\"atomic_numbers\":[", nat, charge, uhf);

    /* Atomic numbers array */
    for (int i = 0; i < nat; i++) {
        if (i > 0) pos += snprintf(buf + pos, buf_size - pos, ",");
        pos += snprintf(buf + pos, buf_size - pos, "%d", atomic_numbers[i]);
    }
    pos += snprintf(buf + pos, buf_size - pos, "],\"positions_bohr\":[");

    /* Positions array (flat: x1,y1,z1,x2,...) */
    for (int i = 0; i < 3 * nat; i++) {
        if (i > 0) pos += snprintf(buf + pos, buf_size - pos, ",");
        pos += snprintf(buf + pos, buf_size - pos, "%.17g", positions_bohr[i]);
    }
    pos += snprintf(buf + pos, buf_size - pos, "]}");

    /* Send request */
    if (send_msg(ctx->sockfd, buf, (size_t)pos) != 0) {
        snprintf(err_msg, err_len, "Failed to send engrad request");
        free(buf);
        pthread_mutex_unlock(&ctx->mutex);
        return 1;
    }
    free(buf);

    /* Receive response */
    size_t resp_len;
    char* resp = recv_msg(ctx->sockfd, &resp_len);
    if (!resp) {
        snprintf(err_msg, err_len, "Failed to receive engrad response");
        pthread_mutex_unlock(&ctx->mutex);
        return 1;
    }

    pthread_mutex_unlock(&ctx->mutex);

    /* Check for error response */
    char msg_type[32];
    if (json_get_string(resp, "type", msg_type, sizeof(msg_type)) != 0) {
        snprintf(err_msg, err_len, "Invalid response: missing type field");
        free(resp);
        return 1;
    }

    if (strcmp(msg_type, "error") == 0) {
        json_get_string(resp, "message", err_msg, err_len);
        free(resp);
        return 1;
    }

    if (strcmp(msg_type, "result") != 0) {
        snprintf(err_msg, err_len, "Unexpected response type: %s", msg_type);
        free(resp);
        return 1;
    }

    /* Parse energy */
    if (json_get_double(resp, "energy_hartree", energy_out) != 0) {
        snprintf(err_msg, err_len, "Failed to parse energy_hartree from response");
        free(resp);
        return 1;
    }

    /* Parse gradient array */
    if (json_get_double_array(resp, "gradient_hartree_bohr",
                              gradient_out, 3 * nat) != 0) {
        snprintf(err_msg, err_len, "Failed to parse gradient_hartree_bohr from response");
        free(resp);
        return 1;
    }

    free(resp);
    return 0;
}

/* ------------------------------------------------------------------ */
/* ase_socket_disconnect                                               */
/* ------------------------------------------------------------------ */

void ase_socket_disconnect(ase_socket_handle_t handle) {
    if (!handle) return;

    ASESocketContext* ctx = (ASESocketContext*)handle;

    /* Send exit message (best effort) */
    const char* exit_msg = "{\"type\":\"exit\"}";
    send_msg(ctx->sockfd, exit_msg, strlen(exit_msg));

    /* Try to receive bye (non-blocking, ignore errors) */
    size_t resp_len;
    char* resp = recv_msg(ctx->sockfd, &resp_len);
    if (resp) free(resp);

    close(ctx->sockfd);
    pthread_mutex_destroy(&ctx->mutex);
    free(ctx);
}
