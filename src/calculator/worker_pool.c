/*
 * worker_pool.c — Persistent worker pool for CREST MLIP parallelism.
 *
 * Creates T worker processes connected via anonymous pipes.
 * Workers load the ML model once, then accept MD/optimization tasks
 * without reinitializing Python or reloading model weights.
 *
 * Wire protocol (both directions):
 *   [4 bytes: total payload length]
 *   [4 bytes: task_type or status]
 *   [remaining bytes: file path string, no null terminator]
 *
 * A payload length of 0 is never sent; SHUTDOWN is task_type=0.
 */

#include "worker_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifdef _WIN32
/* Windows: not supported */
int pool_create(int n, const char* p) { return -1; }
int pool_is_active(void) { return 0; }
int pool_get_n_workers(void) { return 0; }
int pool_send_task(int w, int t, const char* p) { return -1; }
int pool_recv_result(int w, int* s, char* p, int l) { return -1; }
int pool_destroy(void) { return -1; }
int pool_worker_recv_task(int f, int* t, char* p, int l) { return -1; }
int pool_worker_send_result(int f, int s, const char* p) { return -1; }
#else

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>

/* ------------------------------------------------------------------ */
/* Internal data structures                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    pid_t pid;
    int to_worker_fd;    /* parent writes tasks here */
    int from_worker_fd;  /* parent reads results here */
    int alive;           /* 1=running, 0=dead */
} pool_worker_t;

typedef struct {
    pool_worker_t* workers;
    int n_workers;
    int active;
} worker_pool_t;

/* Singleton pool */
static worker_pool_t g_pool = { NULL, 0, 0 };

/* ------------------------------------------------------------------ */
/* Pipe I/O helpers                                                    */
/* ------------------------------------------------------------------ */

static int write_all(int fd, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = write(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += n;
        remaining -= (size_t)n;
    }
    return 0;
}

static int read_all(int fd, void* buf, size_t len) {
    char* p = (char*)buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = read(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;  /* EOF = broken pipe */
        p += n;
        remaining -= (size_t)n;
    }
    return 0;
}

/* Send a message: [4B payload_len][4B int_field][string_data] */
static int send_msg(int fd, int int_field, const char* str) {
    int str_len = str ? (int)strlen(str) : 0;
    int payload_len = 4 + str_len;  /* int_field + string */

    if (write_all(fd, &payload_len, 4) != 0) return -1;
    if (write_all(fd, &int_field, 4) != 0) return -1;
    if (str_len > 0) {
        if (write_all(fd, str, (size_t)str_len) != 0) return -1;
    }
    return 0;
}

/* Receive a message: [4B payload_len][4B int_field][string_data] */
static int recv_msg(int fd, int* int_field, char* str_buf, int str_buf_len) {
    int payload_len;
    if (read_all(fd, &payload_len, 4) != 0) return -1;
    if (payload_len < 4) return -1;

    if (read_all(fd, int_field, 4) != 0) return -1;

    int str_len = payload_len - 4;
    if (str_len > 0) {
        if (str_len >= str_buf_len) str_len = str_buf_len - 1;
        if (read_all(fd, str_buf, (size_t)str_len) != 0) return -1;
        str_buf[str_len] = '\0';

        /* Drain any excess bytes we couldn't fit */
        int excess = (payload_len - 4) - str_len;
        if (excess > 0) {
            char tmp[256];
            while (excess > 0) {
                int chunk = excess > 256 ? 256 : excess;
                if (read_all(fd, tmp, (size_t)chunk) != 0) return -1;
                excess -= chunk;
            }
        }
    } else {
        str_buf[0] = '\0';
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Parent-side pool management                                         */
/* ------------------------------------------------------------------ */

int pool_create(int n_workers, const char* progname)
{
    if (g_pool.active) {
        fprintf(stderr, "[worker_pool] Pool already active\n");
        return -1;
    }
    if (n_workers <= 0 || !progname) return -1;

    g_pool.workers = (pool_worker_t*)calloc((size_t)n_workers,
                                             sizeof(pool_worker_t));
    if (!g_pool.workers) return -1;
    g_pool.n_workers = n_workers;

    for (int i = 0; i < n_workers; i++) {
        int parent_to_child[2];  /* parent writes [1], child reads [0] */
        int child_to_parent[2];  /* child writes [1], parent reads [0] */

        if (pipe(parent_to_child) != 0 || pipe(child_to_parent) != 0) {
            fprintf(stderr, "[worker_pool] pipe() failed: %s\n",
                    strerror(errno));
            /* Clean up already-created workers */
            g_pool.n_workers = i;
            pool_destroy();
            return -1;
        }

        pid_t pid = fork();
        if (pid < 0) {
            fprintf(stderr, "[worker_pool] fork() failed: %s\n",
                    strerror(errno));
            close(parent_to_child[0]); close(parent_to_child[1]);
            close(child_to_parent[0]); close(child_to_parent[1]);
            g_pool.n_workers = i;
            pool_destroy();
            return -1;
        }

        if (pid == 0) {
            /* ---- CHILD PROCESS ---- */
            /* Close parent's ends of pipes */
            close(parent_to_child[1]);
            close(child_to_parent[0]);

            /* Close all other workers' pipe fds (inherited from parent) */
            for (int j = 0; j < i; j++) {
                close(g_pool.workers[j].to_worker_fd);
                close(g_pool.workers[j].from_worker_fd);
            }

            /* Build argv for exec */
            char fd_r_str[16], fd_w_str[16], idx_str[16];
            snprintf(fd_r_str, sizeof(fd_r_str), "%d", parent_to_child[0]);
            snprintf(fd_w_str, sizeof(fd_w_str), "%d", child_to_parent[1]);
            snprintf(idx_str, sizeof(idx_str), "%d", i);

            execl(progname, progname,
                  "--worker-pool", fd_r_str, fd_w_str, idx_str,
                  (char*)NULL);

            /* exec failed */
            fprintf(stderr, "[worker_pool] execl(%s) failed: %s\n",
                    progname, strerror(errno));
            _exit(127);
        }

        /* ---- PARENT PROCESS ---- */
        close(parent_to_child[0]);  /* parent doesn't read from this pipe */
        close(child_to_parent[1]);  /* parent doesn't write to this pipe */

        g_pool.workers[i].pid = pid;
        g_pool.workers[i].to_worker_fd = parent_to_child[1];
        g_pool.workers[i].from_worker_fd = child_to_parent[0];
        g_pool.workers[i].alive = 1;
    }

    g_pool.active = 1;
    return 0;
}


int pool_is_active(void)
{
    return g_pool.active;
}


int pool_get_n_workers(void)
{
    return g_pool.active ? g_pool.n_workers : 0;
}


int pool_send_task(int worker_index, int task_type, const char* config_path)
{
    if (!g_pool.active || worker_index < 0 ||
        worker_index >= g_pool.n_workers) return -1;
    if (!g_pool.workers[worker_index].alive) return -1;

    int fd = g_pool.workers[worker_index].to_worker_fd;
    return send_msg(fd, task_type, config_path);
}


int pool_recv_result(int worker_index, int* status_out,
                     char* result_path, int path_len)
{
    if (!g_pool.active || worker_index < 0 ||
        worker_index >= g_pool.n_workers) return -1;

    int fd = g_pool.workers[worker_index].from_worker_fd;
    int rc = recv_msg(fd, status_out, result_path, path_len);

    if (rc != 0) {
        /* Worker died or pipe broken */
        g_pool.workers[worker_index].alive = 0;
    }
    return rc;
}


int pool_destroy(void)
{
    if (!g_pool.workers) return -1;

    /* Send shutdown to all living workers */
    for (int i = 0; i < g_pool.n_workers; i++) {
        if (g_pool.workers[i].alive) {
            /* Send shutdown task (ignore errors — worker may already be dead) */
            send_msg(g_pool.workers[i].to_worker_fd,
                     POOL_TASK_SHUTDOWN, "");
        }
    }

    /* Wait for all workers and close fds */
    for (int i = 0; i < g_pool.n_workers; i++) {
        if (g_pool.workers[i].to_worker_fd >= 0)
            close(g_pool.workers[i].to_worker_fd);
        if (g_pool.workers[i].from_worker_fd >= 0)
            close(g_pool.workers[i].from_worker_fd);

        if (g_pool.workers[i].pid > 0) {
            int status;
            waitpid(g_pool.workers[i].pid, &status, 0);
        }
    }

    free(g_pool.workers);
    g_pool.workers = NULL;
    g_pool.n_workers = 0;
    g_pool.active = 0;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Worker-side functions                                                */
/* ------------------------------------------------------------------ */

int pool_worker_recv_task(int fd_read, int* task_type_out,
                          char* path_buf, int buf_len)
{
    return recv_msg(fd_read, task_type_out, path_buf, buf_len);
}


int pool_worker_send_result(int fd_write, int status,
                            const char* result_path)
{
    return send_msg(fd_write, status, result_path ? result_path : "");
}

#endif /* _WIN32 */
