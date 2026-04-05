/*
 * process_bridge.h — C-linkage interface for process spawning and waiting.
 * Used by CREST to launch worker processes for GPU-parallel MD
 * (bypassing Python GIL limitation with separate interpreters per process).
 */

#ifndef PROCESS_BRIDGE_H
#define PROCESS_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Spawn a child process to execute a shell command.
 * Returns PID of child on success, -1 on error.
 */
int spawn_process(const char* cmd);

/*
 * Wait for a specific child process to finish.
 * Returns exit status (0 = success), -1 on error.
 */
int wait_for_process(int pid);

/*
 * Wait for any child process to finish.
 * Sets *exit_status to the child's exit code.
 * Returns PID of finished child, -1 on error.
 */
int wait_for_any(int* exit_status);

#ifdef __cplusplus
}
#endif

#endif /* PROCESS_BRIDGE_H */
