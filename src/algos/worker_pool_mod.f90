!================================================================================!
! Fortran bindings for the persistent worker pool (worker_pool.c).
!
! Provides iso_c_binding interfaces to the C pool manager functions,
! plus convenience wrappers that handle Fortran string conversion.
!================================================================================!
module worker_pool_module
  use iso_c_binding
  implicit none
  private

  !> Task type constants (must match worker_pool.h)
  integer, parameter, public :: POOL_TASK_SHUTDOWN = 0
  integer, parameter, public :: POOL_TASK_MD       = 1
  integer, parameter, public :: POOL_TASK_OPT      = 2

  !> Public interface
  public :: pool_create_f, pool_destroy_f
  public :: pool_is_active_f, pool_get_n_workers_f
  public :: pool_send_task_f, pool_recv_result_f
  public :: pool_worker_recv_task_f, pool_worker_send_result_f

  !> C function interfaces
  interface
    function c_pool_create(n_workers, progname) &
        bind(C, name='pool_create')
      import :: c_int, c_char
      integer(c_int), value, intent(in) :: n_workers
      character(kind=c_char), intent(in) :: progname(*)
      integer(c_int) :: c_pool_create
    end function

    function c_pool_is_active() bind(C, name='pool_is_active')
      import :: c_int
      integer(c_int) :: c_pool_is_active
    end function

    function c_pool_get_n_workers() bind(C, name='pool_get_n_workers')
      import :: c_int
      integer(c_int) :: c_pool_get_n_workers
    end function

    function c_pool_send_task(worker_index, task_type, config_path) &
        bind(C, name='pool_send_task')
      import :: c_int, c_char
      integer(c_int), value, intent(in) :: worker_index
      integer(c_int), value, intent(in) :: task_type
      character(kind=c_char), intent(in) :: config_path(*)
      integer(c_int) :: c_pool_send_task
    end function

    function c_pool_recv_result(worker_index, status_out, &
        result_path, path_len) bind(C, name='pool_recv_result')
      import :: c_int, c_char
      integer(c_int), value, intent(in) :: worker_index
      integer(c_int), intent(out) :: status_out
      character(kind=c_char), intent(out) :: result_path(*)
      integer(c_int), value, intent(in) :: path_len
      integer(c_int) :: c_pool_recv_result
    end function

    function c_pool_destroy() bind(C, name='pool_destroy')
      import :: c_int
      integer(c_int) :: c_pool_destroy
    end function

    function c_pool_worker_recv_task(fd_read, task_type_out, &
        path_buf, buf_len) bind(C, name='pool_worker_recv_task')
      import :: c_int, c_char
      integer(c_int), value, intent(in) :: fd_read
      integer(c_int), intent(out) :: task_type_out
      character(kind=c_char), intent(out) :: path_buf(*)
      integer(c_int), value, intent(in) :: buf_len
      integer(c_int) :: c_pool_worker_recv_task
    end function

    function c_pool_worker_send_result(fd_write, status, result_path) &
        bind(C, name='pool_worker_send_result')
      import :: c_int, c_char
      integer(c_int), value, intent(in) :: fd_write
      integer(c_int), value, intent(in) :: status
      character(kind=c_char), intent(in) :: result_path(*)
      integer(c_int) :: c_pool_worker_send_result
    end function
  end interface

contains

  !> Create worker pool. progname is a Fortran string.
  subroutine pool_create_f(n_workers, progname, iostat)
    integer, intent(in) :: n_workers
    character(len=*), intent(in) :: progname
    integer, intent(out) :: iostat
    iostat = int(c_pool_create(int(n_workers, c_int), &
                               trim(progname) // c_null_char))
  end subroutine

  !> Check if pool is active.
  function pool_is_active_f() result(active)
    logical :: active
    active = (c_pool_is_active() /= 0)
  end function

  !> Get number of workers.
  function pool_get_n_workers_f() result(n)
    integer :: n
    n = int(c_pool_get_n_workers())
  end function

  !> Send task to a worker (0-based index). config_path is Fortran string.
  subroutine pool_send_task_f(worker_index, task_type, config_path, iostat)
    integer, intent(in) :: worker_index, task_type
    character(len=*), intent(in) :: config_path
    integer, intent(out) :: iostat
    iostat = int(c_pool_send_task(int(worker_index, c_int), &
                                   int(task_type, c_int), &
                                   trim(config_path) // c_null_char))
  end subroutine

  !> Receive result from a worker (0-based index). Blocks until ready.
  subroutine pool_recv_result_f(worker_index, status, result_path, iostat)
    integer, intent(in) :: worker_index
    integer, intent(out) :: status, iostat
    character(len=*), intent(out) :: result_path
    integer, parameter :: PATH_BUF_LEN = 1024
    character(len=PATH_BUF_LEN) :: buf
    integer(c_int) :: status_c, rc

    buf = ' '
    rc = c_pool_recv_result(int(worker_index, c_int), status_c, &
                             buf, int(PATH_BUF_LEN, c_int))
    status = int(status_c)
    iostat = int(rc)
    !> Copy C string to Fortran string (stop at null terminator)
    result_path = ' '
    block
      integer :: k
      do k = 1, min(len(result_path), PATH_BUF_LEN)
        if (buf(k:k) == c_null_char .or. buf(k:k) == char(0)) exit
        result_path(k:k) = buf(k:k)
      end do
    end block
  end subroutine

  !> Destroy the pool.
  subroutine pool_destroy_f(iostat)
    integer, intent(out) :: iostat
    iostat = int(c_pool_destroy())
  end subroutine

  !> Worker-side: receive a task from the parent.
  subroutine pool_worker_recv_task_f(fd_read, task_type, config_path, iostat)
    integer, intent(in) :: fd_read
    integer, intent(out) :: task_type, iostat
    character(len=*), intent(out) :: config_path
    integer, parameter :: PATH_BUF_LEN = 1024
    character(len=PATH_BUF_LEN) :: buf
    integer(c_int) :: task_c, rc

    buf = ' '
    rc = c_pool_worker_recv_task(int(fd_read, c_int), task_c, &
                                  buf, int(PATH_BUF_LEN, c_int))
    task_type = int(task_c)
    iostat = int(rc)
    config_path = ' '
    block
      integer :: k
      do k = 1, min(len(config_path), PATH_BUF_LEN)
        if (buf(k:k) == c_null_char .or. buf(k:k) == char(0)) exit
        config_path(k:k) = buf(k:k)
      end do
    end block
  end subroutine

  !> Worker-side: send a result back to the parent.
  subroutine pool_worker_send_result_f(fd_write, status, result_path, iostat)
    integer, intent(in) :: fd_write, status
    character(len=*), intent(in) :: result_path
    integer, intent(out) :: iostat
    iostat = int(c_pool_worker_send_result(int(fd_write, c_int), &
                                            int(status, c_int), &
                                            trim(result_path) // c_null_char))
  end subroutine

end module worker_pool_module
