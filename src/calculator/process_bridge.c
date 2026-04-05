/*
 * process_bridge.c — Process spawning and waiting for CREST worker processes.
 *
 * Provides fork/exec/wait wrappers callable from Fortran via iso_c_binding.
 * Used to launch parallel MD worker processes, each with its own Python
 * interpreter, bypassing the GIL serialization of OpenMP threads.
 */

#include "process_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
/* Windows: use CreateProcess (not implemented yet) */
int spawn_process(const char* cmd) { return -1; }
int wait_for_process(int pid) { return -1; }
int wait_for_any(int* exit_status) { *exit_status = -1; return -1; }
#else
/* POSIX: fork/exec/wait */
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>

int spawn_process(const char* cmd)
{
    if (!cmd || !cmd[0]) return -1;

    pid_t pid = fork();
    if (pid < 0) {
        /* fork failed */
        return -1;
    }
    if (pid == 0) {
        /* Child process: exec the command via shell */
        execl("/bin/sh", "sh", "-c", cmd, (char*)NULL);
        /* If execl returns, it failed */
        _exit(127);
    }
    /* Parent: return child PID */
    return (int)pid;
}

int wait_for_process(int pid)
{
    if (pid <= 0) return -1;

    int status;
    pid_t result = waitpid((pid_t)pid, &status, 0);
    if (result < 0) return -1;

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return -1;
}

int wait_for_any(int* exit_status)
{
    int status;
    pid_t pid = wait(&status);
    if (pid < 0) {
        *exit_status = -1;
        return -1;
    }

    if (WIFEXITED(status)) {
        *exit_status = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        *exit_status = 128 + WTERMSIG(status);
    } else {
        *exit_status = -1;
    }
    return (int)pid;
}

#endif /* _WIN32 */
