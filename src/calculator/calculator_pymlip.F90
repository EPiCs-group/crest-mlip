!================================================================================!
! This file is part of crest.
! SPDX-License-Identifier: LGPL-3.0-or-later
!
! Modifications for MLIP support:
! Copyright (C) 2024-2026 Alexander Kolganov
!
! crest is free software: you can redistribute it and/or modify it under
! the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! crest is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Lesser General Public License for more details.
!
! You should have received a copy of the GNU Lesser General Public License
! along with crest.  If not, see <https://www.gnu.org/licenses/>.
!================================================================================!

!> MLIP inference via embedded Python (UMA, MACE).
!> Replaces TCP socket overhead with in-process CPython calls.
!>
!> The Python interpreter is embedded via CPython C API and holds
!> model instances directly in memory. GIL is acquired per-call.
!>
!> Model state is stored per calculation_settings instance for thread safety.
!> Each OpenMP thread creates its own Python calculator handle (lazy init).
!>
!> Requires compilation with -DWITH_PYMLIP (cmake -DWITH_PYMLIP=true)
!> Without it, stub routines are provided that return an error.
module calc_pymlip
  use iso_fortran_env, only: wp => real64, stdout => output_unit
  use iso_c_binding
  implicit none
  private

  integer, parameter :: PYMLIP_ERR_LEN = 4096  !< error message buffer length

  public :: pymlip_engrad, pymlip_engrad_batch_f
  public :: pymlip_init, pymlip_cleanup, pymlip_finalize
  public :: pymlip_get_gpu_memory_f

#ifdef WITH_PYMLIP

  !> Threadprivate buffer for atomic number kind conversion
  integer(c_int), allocatable, save, target :: at_buf(:)
  integer(c_int), save :: buf_nat = -1
  !$omp threadprivate(at_buf, buf_nat)

  !> C interface to pymlip_bridge.c
  interface
    function c_pymlip_init_python(err_msg, err_len) &
        bind(C, name="pymlip_init_python")
      import :: c_int, c_char
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      integer(c_int) :: c_pymlip_init_python
    end function c_pymlip_init_python

    function c_pymlip_create(model_type, model_path, device, task, &
        atom_refs, charge, spin, compile_mode, dtype, turbo, &
        err_msg, err_len) &
        bind(C, name="pymlip_create")
      import :: c_ptr, c_char, c_int
      character(c_char), intent(in) :: model_type(*)
      character(c_char), intent(in) :: model_path(*)
      character(c_char), intent(in) :: device(*)
      character(c_char), intent(in) :: task(*)
      character(c_char), intent(in) :: atom_refs(*)
      integer(c_int), value :: charge
      integer(c_int), value :: spin
      character(c_char), intent(in) :: compile_mode(*)
      character(c_char), intent(in) :: dtype(*)
      integer(c_int), value :: turbo
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      type(c_ptr) :: c_pymlip_create
    end function c_pymlip_create

    function c_pymlip_engrad(handle, nat, positions, atomic_numbers, &
        energy, gradient, err_msg, err_len) &
        bind(C, name="pymlip_engrad")
      import :: c_ptr, c_int, c_double, c_char
      type(c_ptr), value :: handle
      integer(c_int), value :: nat
      real(c_double), intent(in) :: positions(*)
      integer(c_int), intent(in) :: atomic_numbers(*)
      real(c_double), intent(out) :: energy
      real(c_double), intent(out) :: gradient(*)
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      integer(c_int) :: c_pymlip_engrad
    end function c_pymlip_engrad

    function c_pymlip_engrad_batch(handle, batch_size, nat, &
        positions_batch, atomic_numbers, energies, gradients, &
        err_msg, err_len) &
        bind(C, name="pymlip_engrad_batch")
      import :: c_ptr, c_int, c_double, c_char
      type(c_ptr), value :: handle
      integer(c_int), value :: batch_size
      integer(c_int), value :: nat
      real(c_double), intent(in) :: positions_batch(*)
      integer(c_int), intent(in) :: atomic_numbers(*)
      real(c_double), intent(out) :: energies(*)
      real(c_double), intent(out) :: gradients(*)
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      integer(c_int) :: c_pymlip_engrad_batch
    end function c_pymlip_engrad_batch

    subroutine c_pymlip_free(handle) bind(C, name="pymlip_free")
      import :: c_ptr
      type(c_ptr), value :: handle
    end subroutine c_pymlip_free

    subroutine c_pymlip_finalize() bind(C, name="pymlip_finalize_python")
    end subroutine c_pymlip_finalize

    function c_pymlip_get_gpu_memory(total_bytes, free_bytes) &
        bind(C, name="pymlip_get_gpu_memory")
      import :: c_int, c_long_long
      integer(c_long_long), intent(out) :: total_bytes
      integer(c_long_long), intent(out) :: free_bytes
      integer(c_int) :: c_pymlip_get_gpu_memory
    end function c_pymlip_get_gpu_memory
  end interface

#endif

contains

  !> Initialize Python interpreter and create calculator
  subroutine pymlip_init(calc, iostat)
    use calc_type, only: calculation_settings
    type(calculation_settings), intent(inout) :: calc
    integer, intent(out) :: iostat

#ifdef WITH_PYMLIP
    character(len=PYMLIP_ERR_LEN, kind=c_char) :: err_msg
    character(len=512) :: model_type_c, model_path_c, device_c, task_c, refs_c
    character(len=512) :: compile_mode_c, dtype_c
    integer(c_int) :: rc, turbo_c

    iostat = 0

    !> Already loaded
    if (c_associated(calc%pymlip_handle)) return

    !> Check model_type is set
    if (.not.allocated(calc%pymlip_model_type) .or. &
        len_trim(calc%pymlip_model_type) == 0) then
      write(stdout,'(a)') '**ERROR** pymlip: no model_type specified'
      write(stdout,'(a)') '  Use method = ''uma'' or method = ''mace'' for automatic setup'
      write(stdout,'(a)') '  Or set method = ''pymlip'' with model_type = ''uma''/''mace'''
      iostat = 1
      return
    end if

    !> Check model path is set
    if (.not.allocated(calc%pymlip_model_path) .or. &
        len_trim(calc%pymlip_model_path) == 0) then
      write(stdout,'(a)') '**ERROR** pymlip: no model specified'
      write(stdout,'(a)') '  Use model_path for local files: model_path = ''/path/to/model.pt'''
      write(stdout,'(a)') '  Or a registry name:             model_path = ''uma-s-1p1'''
      write(stdout,'(a)') '  Shorthand: method = ''uma'' (auto-downloads uma-s-1p1 from HuggingFace)'
      iostat = 1
      return
    end if

    !> Initialize Python interpreter (idempotent)
    err_msg = ' '
    rc = c_pymlip_init_python(err_msg, int(PYMLIP_ERR_LEN, c_int))
    if (rc /= 0) then
      write(stdout,'(a)') '**ERROR** pymlip: failed to initialize Python'
      write(stdout,'(a)') '  ' // trim(err_msg)
      iostat = 1
      return
    end if

    !> Prepare null-terminated C strings
    model_type_c = trim(calc%pymlip_model_type) // c_null_char
    model_path_c = trim(calc%pymlip_model_path) // c_null_char
    device_c = trim(calc%pymlip_device) // c_null_char
    task_c = trim(calc%pymlip_task) // c_null_char
    refs_c = trim(calc%pymlip_atom_refs) // c_null_char
    if (allocated(calc%pymlip_compile_mode)) then
      compile_mode_c = trim(calc%pymlip_compile_mode) // c_null_char
    else
      compile_mode_c = c_null_char
    end if
    if (allocated(calc%pymlip_dtype)) then
      dtype_c = trim(calc%pymlip_dtype) // c_null_char
    else
      dtype_c = 'float64' // c_null_char
    end if
    turbo_c = 0
    if (calc%pymlip_turbo) turbo_c = 1

    if (calc%pymlip_debug) then
      write(stdout,'(a,a)') ' [pymlip] Model type:    ', trim(calc%pymlip_model_type)
      write(stdout,'(a,a)') ' [pymlip] Model path:    ', trim(calc%pymlip_model_path)
      write(stdout,'(a,a)') ' [pymlip] Device:        ', trim(calc%pymlip_device)
      write(stdout,'(a,a)') ' [pymlip] Task:          ', trim(calc%pymlip_task)
      write(stdout,'(a,i0)') ' [pymlip] Charge:        ', calc%chrg
      write(stdout,'(a,i0)') ' [pymlip] UHF/Spin:      ', calc%uhf
      if (allocated(calc%pymlip_compile_mode) .and. &
          len_trim(calc%pymlip_compile_mode) > 0) then
        write(stdout,'(a,a)') ' [pymlip] Compile mode:  ', trim(calc%pymlip_compile_mode)
      end if
      if (allocated(calc%pymlip_dtype)) then
        write(stdout,'(a,a)') ' [pymlip] Dtype:         ', trim(calc%pymlip_dtype)
      end if
      if (calc%pymlip_turbo) then
        write(stdout,'(a)') ' [pymlip] Turbo:         enabled'
      end if
    end if

    !> Create Python calculator
    err_msg = ' '
    calc%pymlip_handle = c_pymlip_create( &
      model_type_c, model_path_c, device_c, task_c, refs_c, &
      int(calc%chrg, c_int), int(calc%uhf, c_int), &
      compile_mode_c, dtype_c, turbo_c, &
      err_msg, int(PYMLIP_ERR_LEN, c_int))

    if (.not. c_associated(calc%pymlip_handle)) then
      write(stdout,'(a)') '**ERROR** pymlip: failed to create calculator'
      write(stdout,'(a)') '  ' // trim(err_msg)
      iostat = 1
      return
    end if

    if (calc%pymlip_debug) then
      write(stdout,'(a)') ' [pymlip] Calculator created successfully'
    end if

#else
    write(stdout,'(a)') '**ERROR** pymlip support not compiled. ' // &
      'Rebuild with -DWITH_PYMLIP=true'
    iostat = 1
#endif
  end subroutine pymlip_init


  !> Release Python calculator and reset state
  subroutine pymlip_cleanup(calc)
    use calc_type, only: calculation_settings
    type(calculation_settings), intent(inout) :: calc

#ifdef WITH_PYMLIP
    if (calc%pymlip_debug .and. calc%pymlip_call_count > 0) then
      write(stdout,'(a)') ' [pymlip] Session summary:'
      write(stdout,'(a,i0)')    '   Total calls:  ', calc%pymlip_call_count
      write(stdout,'(a,f10.2,a)') '   Total time:   ', calc%pymlip_total_time, 's'
      if (calc%pymlip_call_count > 0 .and. calc%pymlip_total_time > 0.0d0) then
        write(stdout,'(a,f10.2,a)') '   Avg per call: ', &
          calc%pymlip_total_time / real(calc%pymlip_call_count, wp) * 1000.0_wp, 'ms'
        write(stdout,'(a,f10.1,a)') '   Throughput:   ', &
          real(calc%pymlip_call_count, wp) / calc%pymlip_total_time, ' calls/s'
      end if
    end if

    if (c_associated(calc%pymlip_handle)) then
      call c_pymlip_free(calc%pymlip_handle)
      calc%pymlip_handle = c_null_ptr
    end if
    calc%pymlip_call_count = 0
    calc%pymlip_total_time = 0.0d0
#endif
  end subroutine pymlip_cleanup


  !> Finalize Python interpreter — call once at program exit,
  !> after all pymlip_cleanup calls are done.
  subroutine pymlip_finalize()
#ifdef WITH_PYMLIP
    call c_pymlip_finalize()
#endif
  end subroutine pymlip_finalize


  !> Calculate energy and gradient using embedded Python MLIP
  subroutine pymlip_engrad(mol, calc, energy, gradient, iostat)
    use strucrd, only: coord
    use calc_type, only: calculation_settings
    implicit none

    type(coord), intent(in) :: mol
    type(calculation_settings), intent(inout) :: calc
    real(wp), intent(out) :: energy
    real(wp), intent(out) :: gradient(3, mol%nat)
    integer, intent(out) :: iostat

#ifdef WITH_PYMLIP
    integer(c_int) :: status, nat_c
    integer :: i
    real(c_double) :: energy_c
    character(len=PYMLIP_ERR_LEN, kind=c_char) :: err_msg

    !> Debug timing
    integer(8) :: t_start, t_done, t_count_rate
    real(wp) :: dt_total

    iostat = 0
    nat_c = int(mol%nat, c_int)

    if (calc%pymlip_debug) call system_clock(t_start, t_count_rate)

    !> Lazy initialization: create calculator on first call
    if (.not. c_associated(calc%pymlip_handle)) then
      call pymlip_init(calc, iostat)
      if (iostat /= 0) then
        energy = 0.0_wp
        gradient = 0.0_wp
        return
      end if
    end if

    !> (Re)allocate threadprivate atomic number buffer if atom count changes
    if (nat_c /= buf_nat) then
      if (allocated(at_buf)) deallocate(at_buf)
      allocate(at_buf(nat_c))
      buf_nat = nat_c
    end if

    !> Convert atomic numbers to c_int
    do i = 1, nat_c
      at_buf(i) = int(mol%at(i), c_int)
    end do

    !> Call embedded Python bridge
    !> mol%xyz(3, nat) is column-major = C row-major [nat][3]
    !> gradient(3, nat) has same layout
    err_msg = ' '
    status = c_pymlip_engrad( &
      calc%pymlip_handle, &
      nat_c, &
      mol%xyz(1, 1), &       !> positions_bohr: flat [3*nat]
      at_buf(1), &            !> atomic_numbers: [nat]
      energy_c, &             !> energy output (Hartree)
      gradient(1, 1), &       !> gradient output (Hartree/Bohr)
      err_msg, &
      int(PYMLIP_ERR_LEN, c_int))

    if (status /= 0) then
      write(stdout,'(a,i0,a)') '**ERROR** pymlip inference failed (status=', status, ')'
      write(stdout,'(a)') '  ' // trim(err_msg)
      iostat = 1
      energy = 0.0_wp
      gradient = 0.0_wp
      return
    end if

    energy = real(energy_c, wp)

    !> Debug timing
    if (calc%pymlip_debug) then
      call system_clock(t_done)
      dt_total = real(t_done - t_start, wp) / real(t_count_rate, wp)
      calc%pymlip_call_count = calc%pymlip_call_count + 1
      calc%pymlip_total_time = calc%pymlip_total_time + dt_total
      if (mod(calc%pymlip_call_count, 50) == 0 .or. &
          calc%pymlip_call_count == 1) then
        write(stdout,'(a,i0,a,f8.2,a,f10.2,a)') &
          ' [pymlip #', calc%pymlip_call_count, '] total=', &
          dt_total*1000.0_wp, 'ms  cumul=', calc%pymlip_total_time, 's'
      end if
    end if

#else
    energy = 0.0_wp
    gradient = 0.0_wp
    write(stdout,'(a)') '**ERROR** pymlip support not compiled. ' // &
      'Rebuild with -DWITH_PYMLIP=true'
    iostat = 1
#endif

  end subroutine pymlip_engrad


  !> Batched energy+gradient: process multiple structures with one GIL acquisition.
  !> All structures must have the same number of atoms.
  subroutine pymlip_engrad_batch_f(calc, batch_size, nat, at, &
      pos_batch, energies, grad_batch, iostat)
    use calc_type, only: calculation_settings
    implicit none

    type(calculation_settings), intent(inout) :: calc
    integer, intent(in) :: batch_size
    integer, intent(in) :: nat
    integer, intent(in) :: at(nat)
    real(wp), intent(in) :: pos_batch(3*nat*batch_size)
    real(wp), intent(out) :: energies(batch_size)
    real(wp), intent(out) :: grad_batch(3*nat*batch_size)
    integer, intent(out) :: iostat

#ifdef WITH_PYMLIP
    integer(c_int) :: status, nat_c, bs_c
    integer :: i
    integer(c_int), allocatable :: at_c(:)
    character(len=PYMLIP_ERR_LEN, kind=c_char) :: err_msg

    !> Debug timing
    integer(8) :: t_start, t_done, t_count_rate
    real(wp) :: dt_total

    iostat = 0
    nat_c = int(nat, c_int)
    bs_c = int(batch_size, c_int)

    if (calc%pymlip_debug) call system_clock(t_start, t_count_rate)

    !> Lazy initialization (fallback if not pre-loaded in parallel.f90)
    if (.not. c_associated(calc%pymlip_handle)) then
      call pymlip_init(calc, iostat)
      if (iostat /= 0) return
    end if

    !> Convert atomic numbers to c_int
    allocate(at_c(nat_c))
    do i = 1, nat_c
      at_c(i) = int(at(i), c_int)
    end do

    !> Call batched C bridge
    err_msg = ' '
    status = c_pymlip_engrad_batch( &
      calc%pymlip_handle, &
      bs_c, nat_c, &
      pos_batch(1), &
      at_c(1), &
      energies(1), &
      grad_batch(1), &
      err_msg, &
      int(PYMLIP_ERR_LEN, c_int))

    deallocate(at_c)

    if (status /= 0) then
      write(stdout,'(a,i0,a)') '**ERROR** pymlip batch failed (status=', status, ')'
      write(stdout,'(a)') '  ' // trim(err_msg)
      iostat = 1
      energies = 0.0_wp
      grad_batch = 0.0_wp
      return
    end if

    !> Debug timing
    if (calc%pymlip_debug) then
      call system_clock(t_done)
      dt_total = real(t_done - t_start, wp) / real(t_count_rate, wp)
      calc%pymlip_call_count = calc%pymlip_call_count + batch_size
      calc%pymlip_total_time = calc%pymlip_total_time + dt_total
      write(stdout,'(a,i0,a,f8.2,a,f8.2,a)') &
        ' [pymlip batch] ', batch_size, ' structures in ', &
        dt_total*1000.0_wp, 'ms (', &
        dt_total*1000.0_wp/real(batch_size,wp), 'ms/struct)'
    end if

#else
    energies = 0.0_wp
    grad_batch = 0.0_wp
    write(stdout,'(a)') '**ERROR** pymlip support not compiled. ' // &
      'Rebuild with -DWITH_PYMLIP=true'
    iostat = 1
#endif

  end subroutine pymlip_engrad_batch_f


  !> Query GPU memory via PyTorch (torch.cuda.mem_get_info).
  !> Returns total and free GPU memory in bytes.
  !> iostat=0 on success, 1 if CUDA unavailable or not compiled.
  subroutine pymlip_get_gpu_memory_f(total_bytes, free_bytes, iostat)
    implicit none
    integer(8), intent(out) :: total_bytes, free_bytes
    integer, intent(out) :: iostat
#ifdef WITH_PYMLIP
    integer(c_int) :: rc
    integer(c_long_long) :: total_c, free_c
    rc = c_pymlip_get_gpu_memory(total_c, free_c)
    total_bytes = int(total_c, 8)
    free_bytes = int(free_c, 8)
    iostat = int(rc)
#else
    total_bytes = 0
    free_bytes = 0
    iostat = 1
#endif
  end subroutine pymlip_get_gpu_memory_f

end module calc_pymlip
