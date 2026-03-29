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

!> Direct MLIP inference via libtorch (TorchScript models).
!> Replaces the Python server + TCP socket path with in-process C++ calls.
!>
!> The TorchScript model (exported by scripts/export_model.py) embeds:
!>   - Graph construction (neighbor list)
!>   - Model forward pass
!>   - Unit conversions (Bohr<->Angstrom, eV<->Hartree)
!>
!> Model state is stored per calculation_settings instance for thread safety.
!> Each OpenMP thread loads its own model copy (lazy init on first call).
!>
!> Requires compilation with -DWITH_LIBTORCH (cmake -DWITH_LIBTORCH=true)
!> Without it, stub routines are provided that return an error.
module calc_libtorch
  use iso_fortran_env, only: wp => real64, stdout => output_unit
  use iso_c_binding
  implicit none
  private

  public :: libtorch_engrad, libtorch_init, libtorch_cleanup
  public :: libtorch_set_threads, libtorch_init_shared
  public :: libtorch_engrad_batch_f, libtorch_shared_cleanup
  public :: libtorch_engrad_batch_pipeline_f
  public :: libtorch_engrad_batch_multigpu_f
  public :: libtorch_load_shared_on_device_f
  public :: libtorch_get_cuda_device_count_f

#ifdef WITH_LIBTORCH

  !> Threadprivate buffer for atomic number kind conversion (int -> c_int)
  integer(c_int), allocatable, save, target :: at_buf(:)
  integer(c_int), save :: buf_nat = -1
  !$omp threadprivate(at_buf, buf_nat)

  !> C interface to libtorch_bridge.cpp
  interface
    function c_libtorch_load(model_path, device, model_format, cutoff, &
        err_msg, err_len) &
        bind(C, name="libtorch_load")
      import :: c_ptr, c_char, c_int, c_double
      character(c_char), intent(in) :: model_path(*)
      integer(c_int), value :: device
      integer(c_int), value :: model_format
      real(c_double), value :: cutoff
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      type(c_ptr) :: c_libtorch_load
    end function c_libtorch_load

    function c_libtorch_load_shared(model_path, device, model_format, cutoff, &
        err_msg, err_len) &
        bind(C, name="libtorch_load_shared")
      import :: c_ptr, c_char, c_int, c_double
      character(c_char), intent(in) :: model_path(*)
      integer(c_int), value :: device
      integer(c_int), value :: model_format
      real(c_double), value :: cutoff
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      type(c_ptr) :: c_libtorch_load_shared
    end function c_libtorch_load_shared

    function c_libtorch_engrad(handle, nat, positions, atomic_numbers, &
        energy, gradient, err_msg, err_len) &
        bind(C, name="libtorch_engrad")
      import :: c_ptr, c_int, c_double, c_char
      type(c_ptr), value :: handle
      integer(c_int), value :: nat
      real(c_double), intent(in) :: positions(*)
      integer(c_int), intent(in) :: atomic_numbers(*)
      real(c_double), intent(out) :: energy
      real(c_double), intent(out) :: gradient(*)
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      integer(c_int) :: c_libtorch_engrad
    end function c_libtorch_engrad

    function c_libtorch_engrad_batch(handle, batch_size, nat, positions, &
        atomic_numbers, energies, gradients, err_msg, err_len) &
        bind(C, name="libtorch_engrad_batch")
      import :: c_ptr, c_int, c_double, c_char
      type(c_ptr), value :: handle
      integer(c_int), value :: batch_size
      integer(c_int), value :: nat
      real(c_double), intent(in) :: positions(*)
      integer(c_int), intent(in) :: atomic_numbers(*)
      real(c_double), intent(out) :: energies(*)
      real(c_double), intent(out) :: gradients(*)
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      integer(c_int) :: c_libtorch_engrad_batch
    end function c_libtorch_engrad_batch

    subroutine c_libtorch_free(handle) bind(C, name="libtorch_free")
      import :: c_ptr
      type(c_ptr), value :: handle
    end subroutine c_libtorch_free

    subroutine c_libtorch_shared_free_all() &
        bind(C, name="libtorch_shared_free_all")
    end subroutine c_libtorch_shared_free_all

    subroutine c_libtorch_set_num_threads(aten_threads) &
        bind(C, name="libtorch_set_num_threads")
      import :: c_int
      integer(c_int), value :: aten_threads
    end subroutine c_libtorch_set_num_threads

    function c_libtorch_engrad_batch_pipeline(handle, total_structures, nat, &
        all_positions, atomic_numbers, batch_size, energies, gradients, &
        err_msg, err_len) &
        bind(C, name="libtorch_engrad_batch_pipeline")
      import :: c_ptr, c_int, c_double, c_char
      type(c_ptr), value :: handle
      integer(c_int), value :: total_structures
      integer(c_int), value :: nat
      real(c_double), intent(in) :: all_positions(*)
      integer(c_int), intent(in) :: atomic_numbers(*)
      integer(c_int), value :: batch_size
      real(c_double), intent(out) :: energies(*)
      real(c_double), intent(out) :: gradients(*)
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      integer(c_int) :: c_libtorch_engrad_batch_pipeline
    end function c_libtorch_engrad_batch_pipeline

    function c_libtorch_engrad_batch_multigpu(handles, ngpus, &
        total_structures, nat, all_positions, atomic_numbers, batch_size, &
        energies, gradients, err_msg, err_len) &
        bind(C, name="libtorch_engrad_batch_multigpu")
      import :: c_ptr, c_int, c_double, c_char
      type(c_ptr), intent(in) :: handles(*)
      integer(c_int), value :: ngpus
      integer(c_int), value :: total_structures
      integer(c_int), value :: nat
      real(c_double), intent(in) :: all_positions(*)
      integer(c_int), intent(in) :: atomic_numbers(*)
      integer(c_int), value :: batch_size
      real(c_double), intent(out) :: energies(*)
      real(c_double), intent(out) :: gradients(*)
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      integer(c_int) :: c_libtorch_engrad_batch_multigpu
    end function c_libtorch_engrad_batch_multigpu

    function c_libtorch_load_shared_on_device(model_path, cuda_device_index, &
        model_format, cutoff, err_msg, err_len) &
        bind(C, name="libtorch_load_shared_on_device")
      import :: c_ptr, c_char, c_int, c_double
      character(c_char), intent(in) :: model_path(*)
      integer(c_int), value :: cuda_device_index
      integer(c_int), value :: model_format
      real(c_double), value :: cutoff
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      type(c_ptr) :: c_libtorch_load_shared_on_device
    end function c_libtorch_load_shared_on_device

    function c_libtorch_get_cuda_device_count() &
        bind(C, name="libtorch_get_cuda_device_count")
      import :: c_int
      integer(c_int) :: c_libtorch_get_cuda_device_count
    end function c_libtorch_get_cuda_device_count
  end interface

#endif

contains

  !> Load TorchScript model from file
  subroutine libtorch_init(calc, iostat)
    use calc_type, only: calculation_settings
    type(calculation_settings), intent(inout) :: calc
    integer, intent(out) :: iostat

#ifdef WITH_LIBTORCH
    character(len=4096, kind=c_char) :: err_msg
    character(len=512) :: model_path_c

    iostat = 0

    !> Already loaded
    if (c_associated(calc%libtorch_handle)) return

    !> Check model path is set
    if (len_trim(calc%libtorch_model_path) == 0) then
      write(stdout,'(a)') '**ERROR** libtorch: no model_path specified'
      iostat = 1
      return
    end if

    !> Convert Fortran string to null-terminated C string
    model_path_c = trim(calc%libtorch_model_path) // c_null_char

    !> Load model
    if (calc%libtorch_debug) then
      write(stdout,'(a,a)') ' [libtorch] Loading model: ', &
        trim(calc%libtorch_model_path)
      write(stdout,'(a,i0)') ' [libtorch] Device: ', calc%libtorch_device_id
      write(stdout,'(a,i0)') ' [libtorch] Format: ', calc%libtorch_model_format
      write(stdout,'(a,f6.2)') ' [libtorch] Cutoff: ', calc%libtorch_cutoff
    end if

    err_msg = ' '
    calc%libtorch_handle = c_libtorch_load( &
      model_path_c, &
      int(calc%libtorch_device_id, c_int), &
      int(calc%libtorch_model_format, c_int), &
      real(calc%libtorch_cutoff, c_double), &
      err_msg, &
      int(4096, c_int))

    if (.not. c_associated(calc%libtorch_handle)) then
      write(stdout,'(a)') '**ERROR** libtorch: failed to load model'
      write(stdout,'(a)') '  ' // trim(err_msg)
      iostat = 1
      return
    end if

    if (calc%libtorch_debug) then
      write(stdout,'(a)') ' [libtorch] Model loaded successfully'
    end if

#else
    write(stdout,'(a)') '**ERROR** libtorch support not compiled. ' // &
      'Rebuild with -DWITH_LIBTORCH=true'
    iostat = 1
#endif
  end subroutine libtorch_init


  !> Release model and reset state
  subroutine libtorch_cleanup(calc)
    use calc_type, only: calculation_settings
    type(calculation_settings), intent(inout) :: calc

#ifdef WITH_LIBTORCH
    if (calc%libtorch_debug .and. calc%libtorch_call_count > 0) then
      write(stdout,'(a)') ' [libtorch] Session summary:'
      write(stdout,'(a,i0)')    '   Total calls:  ', calc%libtorch_call_count
      write(stdout,'(a,f10.2,a)') '   Total time:   ', calc%libtorch_total_time, 's'
      if (calc%libtorch_call_count > 0 .and. calc%libtorch_total_time > 0.0d0) then
        write(stdout,'(a,f10.2,a)') '   Avg per call: ', &
          calc%libtorch_total_time / real(calc%libtorch_call_count, wp) * 1000.0_wp, 'ms'
        write(stdout,'(a,f10.1,a)') '   Throughput:   ', &
          real(calc%libtorch_call_count, wp) / calc%libtorch_total_time, ' calls/s'
      end if
    end if

    if (c_associated(calc%libtorch_handle)) then
      call c_libtorch_free(calc%libtorch_handle)
      calc%libtorch_handle = c_null_ptr
    end if
    calc%libtorch_call_count = 0
    calc%libtorch_total_time = 0.0d0
#endif
  end subroutine libtorch_cleanup


  !> Calculate energy and gradient using libtorch TorchScript model
  subroutine libtorch_engrad(mol, calc, energy, gradient, iostat)
    use strucrd, only: coord
    use calc_type, only: calculation_settings
    implicit none

    type(coord), intent(in) :: mol
    type(calculation_settings), intent(inout) :: calc
    real(wp), intent(out) :: energy
    real(wp), intent(out) :: gradient(3, mol%nat)
    integer, intent(out) :: iostat

#ifdef WITH_LIBTORCH
    integer(c_int) :: status, nat_c
    integer :: i
    real(c_double) :: energy_c
    character(len=4096, kind=c_char) :: err_msg

    !> Debug timing
    integer(8) :: t_start, t_done, t_count_rate
    real(wp) :: dt_total

    iostat = 0
    nat_c = int(mol%nat, c_int)

    if (calc%libtorch_debug) call system_clock(t_start, t_count_rate)

    !> Lazy initialization: load model on first call
    if (.not. c_associated(calc%libtorch_handle)) then
      call libtorch_init(calc, iostat)
      if (iostat /= 0) return
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

    !> Call libtorch C++ bridge
    !> mol%xyz(3, nat) is column-major = C row-major [nat][3]
    !> gradient(3, nat) has same layout — written directly by C
    err_msg = ' '
    status = c_libtorch_engrad( &
      calc%libtorch_handle, &
      nat_c, &
      mol%xyz(1, 1), &       !> positions_bohr: flat [3*nat]
      at_buf(1), &            !> atomic_numbers: [nat]
      energy_c, &             !> energy output (Hartree)
      gradient(1, 1), &       !> gradient output (Hartree/Bohr), flat [3*nat]
      err_msg, &
      int(4096, c_int))

    if (status /= 0) then
      write(stdout,'(a,i0)') '**ERROR** libtorch inference failed (status=', status
      write(stdout,'(a)') '  ' // trim(err_msg)
      iostat = 1
      energy = 0.0_wp
      gradient = 0.0_wp
      return
    end if

    energy = real(energy_c, wp)

    !> Debug timing
    if (calc%libtorch_debug) then
      call system_clock(t_done)
      dt_total = real(t_done - t_start, wp) / real(t_count_rate, wp)
      calc%libtorch_call_count = calc%libtorch_call_count + 1
      calc%libtorch_total_time = calc%libtorch_total_time + dt_total
      if (mod(calc%libtorch_call_count, 50) == 0 .or. &
          calc%libtorch_call_count == 1) then
        write(stdout,'(a,i0,a,f8.2,a,f10.2,a)') &
          ' [libtorch #', calc%libtorch_call_count, '] total=', &
          dt_total*1000.0_wp, 'ms  cumul=', calc%libtorch_total_time, 's'
      end if
    end if

#else
    energy = 0.0_wp
    gradient = 0.0_wp
    write(stdout,'(a)') '**ERROR** libtorch support not compiled. ' // &
      'Rebuild with -DWITH_LIBTORCH=true'
    iostat = 1
#endif

  end subroutine libtorch_engrad

  !> Set ATen internal thread count — call once before parallel region
  subroutine libtorch_set_threads(aten_threads)
    integer, intent(in) :: aten_threads
#ifdef WITH_LIBTORCH
    call c_libtorch_set_num_threads(int(aten_threads, c_int))
#endif
  end subroutine libtorch_set_threads


  !> Load or retrieve shared model (for GPU batched mode)
  subroutine libtorch_init_shared(calc, iostat)
    use calc_type, only: calculation_settings
    type(calculation_settings), intent(inout) :: calc
    integer, intent(out) :: iostat

#ifdef WITH_LIBTORCH
    character(len=4096, kind=c_char) :: err_msg
    character(len=512) :: model_path_c

    iostat = 0

    !> Already loaded
    if (c_associated(calc%libtorch_handle)) return

    if (len_trim(calc%libtorch_model_path) == 0) then
      write(stdout,'(a)') '**ERROR** libtorch shared: no model_path specified'
      iostat = 1
      return
    end if

    model_path_c = trim(calc%libtorch_model_path) // c_null_char

    if (calc%libtorch_debug) then
      write(stdout,'(a,a)') ' [libtorch] Loading shared model: ', &
        trim(calc%libtorch_model_path)
    end if

    err_msg = ' '
    calc%libtorch_handle = c_libtorch_load_shared( &
      model_path_c, &
      int(calc%libtorch_device_id, c_int), &
      int(calc%libtorch_model_format, c_int), &
      real(calc%libtorch_cutoff, c_double), &
      err_msg, &
      int(4096, c_int))

    if (.not. c_associated(calc%libtorch_handle)) then
      write(stdout,'(a)') '**ERROR** libtorch: failed to load shared model'
      write(stdout,'(a)') '  ' // trim(err_msg)
      iostat = 1
      return
    end if

    if (calc%libtorch_debug) then
      write(stdout,'(a)') ' [libtorch] Shared model loaded successfully'
    end if

#else
    write(stdout,'(a)') '**ERROR** libtorch support not compiled.'
    iostat = 1
#endif
  end subroutine libtorch_init_shared


  !> Batched energy+gradient for multiple structures with same nat and Z
  subroutine libtorch_engrad_batch_f(calc, batch_size, nat, at, &
      pos_batch, energies, grad_batch, iostat)
    use calc_type, only: calculation_settings
    type(calculation_settings), intent(inout) :: calc
    integer, intent(in) :: batch_size
    integer, intent(in) :: nat
    integer, intent(in) :: at(nat)
    real(wp), intent(in) :: pos_batch(3*nat*batch_size)
    real(wp), intent(out) :: energies(batch_size)
    real(wp), intent(out) :: grad_batch(3*nat*batch_size)
    integer, intent(out) :: iostat

#ifdef WITH_LIBTORCH
    integer(c_int) :: status
    integer :: i
    integer(c_int), allocatable :: at_c(:)
    character(len=4096, kind=c_char) :: err_msg
    integer(8) :: t_start, t_done, t_count_rate
    real(wp) :: dt_total

    iostat = 0

    if (calc%libtorch_debug) call system_clock(t_start, t_count_rate)

    !> Lazy initialization
    if (.not. c_associated(calc%libtorch_handle)) then
      if (calc%libtorch_shared_model) then
        call libtorch_init_shared(calc, iostat)
      else
        call libtorch_init(calc, iostat)
      end if
      if (iostat /= 0) return
    end if

    !> Convert atomic numbers
    allocate(at_c(nat))
    do i = 1, nat
      at_c(i) = int(at(i), c_int)
    end do

    !> Call C++ batched inference
    err_msg = ' '
    status = c_libtorch_engrad_batch( &
      calc%libtorch_handle, &
      int(batch_size, c_int), &
      int(nat, c_int), &
      pos_batch(1), &
      at_c(1), &
      energies(1), &
      grad_batch(1), &
      err_msg, &
      int(4096, c_int))

    deallocate(at_c)

    if (status /= 0) then
      write(stdout,'(a,i0)') '**ERROR** libtorch batch inference failed (status=', status
      write(stdout,'(a)') '  ' // trim(err_msg)
      iostat = 1
      return
    end if

    !> Debug timing
    if (calc%libtorch_debug) then
      call system_clock(t_done)
      dt_total = real(t_done - t_start, wp) / real(t_count_rate, wp)
      calc%libtorch_call_count = calc%libtorch_call_count + batch_size
      calc%libtorch_total_time = calc%libtorch_total_time + dt_total
      write(stdout,'(a,i0,a,f8.2,a,f6.1,a)') &
        ' [libtorch batch] ', batch_size, ' structures in ', &
        dt_total*1000.0_wp, 'ms (', &
        dt_total*1000.0_wp / real(batch_size, wp), 'ms/struct)'
    end if

#else
    energies = 0.0_wp
    grad_batch = 0.0_wp
    write(stdout,'(a)') '**ERROR** libtorch support not compiled.'
    iostat = 1
#endif
  end subroutine libtorch_engrad_batch_f


  !> Release all shared models — call once at program exit
  subroutine libtorch_shared_cleanup()
#ifdef WITH_LIBTORCH
    call c_libtorch_shared_free_all()
#endif
  end subroutine libtorch_shared_cleanup


  !> Pipelined batch inference (single GPU, double-buffered).
  !> Processes all structures in one call with internal pipelining.
  subroutine libtorch_engrad_batch_pipeline_f(calc, total_structures, nat, &
      at, all_positions, energies, gradients, batch_size, iostat)
    use calc_type, only: calculation_settings
    type(calculation_settings), intent(inout) :: calc
    integer, intent(in) :: total_structures
    integer, intent(in) :: nat
    integer, intent(in) :: at(nat)
    real(wp), intent(in) :: all_positions(3*nat*total_structures)
    real(wp), intent(out) :: energies(total_structures)
    real(wp), intent(out) :: gradients(3*nat*total_structures)
    integer, intent(in) :: batch_size
    integer, intent(out) :: iostat

#ifdef WITH_LIBTORCH
    integer(c_int) :: status
    integer :: i
    integer(c_int), allocatable :: at_c(:)
    character(len=4096, kind=c_char) :: err_msg

    iostat = 0

    !> Convert atomic numbers
    allocate(at_c(nat))
    do i = 1, nat
      at_c(i) = int(at(i), c_int)
    end do

    err_msg = ' '
    status = c_libtorch_engrad_batch_pipeline( &
      calc%libtorch_handle, &
      int(total_structures, c_int), &
      int(nat, c_int), &
      all_positions(1), &
      at_c(1), &
      int(batch_size, c_int), &
      energies(1), &
      gradients(1), &
      err_msg, &
      int(4096, c_int))

    deallocate(at_c)

    if (status /= 0) then
      write(stdout,'(a,i0)') '**ERROR** libtorch pipeline failed (status=', status
      write(stdout,'(a)') '  ' // trim(err_msg)
      iostat = 1
    end if

#else
    energies = 0.0_wp
    gradients = 0.0_wp
    write(stdout,'(a)') '**ERROR** libtorch support not compiled.'
    iostat = 1
#endif
  end subroutine libtorch_engrad_batch_pipeline_f


  !> Multi-GPU pipelined batch inference.
  subroutine libtorch_engrad_batch_multigpu_f(handles, ngpus, &
      total_structures, nat, at, all_positions, energies, gradients, &
      batch_size, iostat)
    use calc_type, only: calculation_settings
    type(c_ptr), intent(in) :: handles(ngpus)
    integer, intent(in) :: ngpus
    integer, intent(in) :: total_structures
    integer, intent(in) :: nat
    integer, intent(in) :: at(nat)
    real(wp), intent(in) :: all_positions(3*nat*total_structures)
    real(wp), intent(out) :: energies(total_structures)
    real(wp), intent(out) :: gradients(3*nat*total_structures)
    integer, intent(in) :: batch_size
    integer, intent(out) :: iostat

#ifdef WITH_LIBTORCH
    integer(c_int) :: status
    integer :: i
    integer(c_int), allocatable :: at_c(:)
    character(len=4096, kind=c_char) :: err_msg

    iostat = 0

    allocate(at_c(nat))
    do i = 1, nat
      at_c(i) = int(at(i), c_int)
    end do

    err_msg = ' '
    status = c_libtorch_engrad_batch_multigpu( &
      handles(1), &
      int(ngpus, c_int), &
      int(total_structures, c_int), &
      int(nat, c_int), &
      all_positions(1), &
      at_c(1), &
      int(batch_size, c_int), &
      energies(1), &
      gradients(1), &
      err_msg, &
      int(4096, c_int))

    deallocate(at_c)

    if (status /= 0) then
      write(stdout,'(a,i0)') '**ERROR** libtorch multigpu failed (status=', status
      write(stdout,'(a)') '  ' // trim(err_msg)
      iostat = 1
    end if

#else
    energies = 0.0_wp
    gradients = 0.0_wp
    write(stdout,'(a)') '**ERROR** libtorch support not compiled.'
    iostat = 1
#endif
  end subroutine libtorch_engrad_batch_multigpu_f


  !> Load shared model on specific CUDA device.
  subroutine libtorch_load_shared_on_device_f(calc, cuda_device_index, &
      handle_out, iostat)
    use calc_type, only: calculation_settings
    type(calculation_settings), intent(in) :: calc
    integer, intent(in) :: cuda_device_index
    type(c_ptr), intent(out) :: handle_out
    integer, intent(out) :: iostat

#ifdef WITH_LIBTORCH
    character(len=4096, kind=c_char) :: err_msg
    character(len=512) :: model_path_c

    iostat = 0
    handle_out = c_null_ptr

    model_path_c = trim(calc%libtorch_model_path) // c_null_char
    err_msg = ' '

    handle_out = c_libtorch_load_shared_on_device( &
      model_path_c, &
      int(cuda_device_index, c_int), &
      int(calc%libtorch_model_format, c_int), &
      real(calc%libtorch_cutoff, c_double), &
      err_msg, &
      int(4096, c_int))

    if (.not. c_associated(handle_out)) then
      write(stdout,'(a,i0)') '**ERROR** libtorch: failed to load model on CUDA:', &
        cuda_device_index
      write(stdout,'(a)') '  ' // trim(err_msg)
      iostat = 1
    end if

#else
    handle_out = c_null_ptr
    write(stdout,'(a)') '**ERROR** libtorch support not compiled.'
    iostat = 1
#endif
  end subroutine libtorch_load_shared_on_device_f


  !> Query number of CUDA devices available.
  function libtorch_get_cuda_device_count_f() result(ndevices)
    integer :: ndevices
#ifdef WITH_LIBTORCH
    ndevices = int(c_libtorch_get_cuda_device_count())
#else
    ndevices = 0
#endif
  end function libtorch_get_cuda_device_count_f


end module calc_libtorch
