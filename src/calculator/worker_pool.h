/*
 * worker_pool.h — Persistent worker pool for CREST MLIP parallelism.
 *
 * Workers are spawned once, load the ML model once, then accept
 * multiple task packets (MD or optimization) over anonymous pipes.
 * This eliminates the 30-60s model reload penalty between workflow steps.
 */

#ifndef WORKER_POOL_H
#define WORKER_POOL_H

#ifdef __cplusplus
extern "C" {
#endif

/* Task type constants */
#define POOL_TASK_SHUTDOWN  0
#define POOL_TASK_MD        1
#define POOL_TASK_OPT       2

/*
 * Create a persistent worker pool with n_workers processes.
 * Each worker is started via: progname --worker-pool <fd_r> <fd_w> <index>
 * Returns 0 on success, -1 on error.
 */
int pool_create(int n_workers, const char* progname);

/*
 * Check if the pool is currently active.
 * Returns 1 if active, 0 if not.
 */
int pool_is_active(void);

/*
 * Get number of workers in the pool.
 */
int pool_get_n_workers(void);

/*
 * Send a task to a specific worker.
 * task_type: POOL_TASK_MD, POOL_TASK_OPT
 * config_path: path to binary config file (null-terminated)
 * Returns 0 on success, -1 on error.
 */
int pool_send_task(int worker_index, int task_type, const char* config_path);

/*
 * Receive a result from a specific worker (blocks until worker responds).
 * status_out: worker's exit status (0=success)
 * result_path: buffer to receive result file path
 * path_len: size of result_path buffer
 * Returns 0 on success, -1 on error (broken pipe, timeout, etc.)
 */
int pool_recv_result(int worker_index, int* status_out,
                     char* result_path, int path_len);

/*
 * Shut down all workers and destroy the pool.
 * Sends SHUTDOWN to each worker, waits for exit, closes pipes.
 * Returns 0 on success, -1 on error.
 */
int pool_destroy(void);

/* ---- Worker-side functions (called from child processes) ---- */

/*
 * Receive a task from the parent process.
 * fd_read: file descriptor to read from (inherited pipe)
 * task_type_out: receives POOL_TASK_MD, POOL_TASK_OPT, or POOL_TASK_SHUTDOWN
 * path_buf: buffer to receive config file path
 * buf_len: size of path_buf
 * Returns 0 on success, -1 on error.
 */
int pool_worker_recv_task(int fd_read, int* task_type_out,
                          char* path_buf, int buf_len);

/*
 * Send a result back to the parent process.
 * fd_write: file descriptor to write to (inherited pipe)
 * status: 0=success, nonzero=error
 * result_path: path to result file (or empty string for MD)
 * Returns 0 on success, -1 on error.
 */
int pool_worker_send_result(int fd_write, int status, const char* result_path);

#ifdef __cplusplus
}
#endif

#endif /* WORKER_POOL_H */
