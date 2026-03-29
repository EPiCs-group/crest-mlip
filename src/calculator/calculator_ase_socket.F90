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

!> ASE calculator via TCP socket.
!> Connects to an external Python server wrapping any ASE calculator.
!>
!> The server must implement the CREST ASE socket protocol:
!> length-prefixed JSON messages over TCP (see crest_ase_server.py).
!>
!> No conditional compilation needed — POSIX sockets are always available.
module calc_ase_socket
  use iso_fortran_env, only: wp => real64, stdout => output_unit
  use iso_c_binding
  implicit none
  private

  public :: ase_socket_engrad, ase_socket_init, ase_socket_cleanup

  !> C interface to ase_socket_bridge.c
  interface
    function c_ase_socket_connect(host, port, err_msg, err_len) &
        bind(C, name="ase_socket_connect")
      import :: c_ptr, c_char, c_int
      character(c_char), intent(in) :: host(*)
      integer(c_int), value :: port
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      type(c_ptr) :: c_ase_socket_connect
    end function c_ase_socket_connect

    function c_ase_socket_engrad(handle, nat, positions_bohr, &
        atomic_numbers, charge, uhf, energy_out, gradient_out, &
        err_msg, err_len) &
        bind(C, name="ase_socket_engrad")
      import :: c_ptr, c_int, c_double, c_char
      type(c_ptr), value :: handle
      integer(c_int), value :: nat
      real(c_double), intent(in) :: positions_bohr(*)
      integer(c_int), intent(in) :: atomic_numbers(*)
      integer(c_int), value :: charge
      integer(c_int), value :: uhf
      real(c_double), intent(out) :: energy_out
      real(c_double), intent(out) :: gradient_out(*)
      character(c_char) :: err_msg(*)
      integer(c_int), value :: err_len
      integer(c_int) :: c_ase_socket_engrad
    end function c_ase_socket_engrad

    subroutine c_ase_socket_disconnect(handle) &
        bind(C, name="ase_socket_disconnect")
      import :: c_ptr
      type(c_ptr), value :: handle
    end subroutine c_ase_socket_disconnect
  end interface

contains

  !> Connect to the ASE socket server (lazy, called on first engrad)
  subroutine ase_socket_init(calc, iostat)
    use calc_type, only: calculation_settings
    type(calculation_settings), intent(inout) :: calc
    integer, intent(out) :: iostat

    character(len=4096, kind=c_char) :: err_msg
    character(len=256) :: host_c

    iostat = 0

    !> Already connected
    if (c_associated(calc%socket_handle)) return

    !> Apply defaults
    if (.not. allocated(calc%socket_host) .or. &
        len_trim(calc%socket_host) == 0) then
      calc%socket_host = 'localhost'
    end if
    if (calc%socket_port <= 0) then
      calc%socket_port = 6789
    end if

    write(stdout,'(a,a,a,i0)') &
      ' [ase-socket] Connecting to ', trim(calc%socket_host), ':', calc%socket_port

    !> Prepare null-terminated C string
    host_c = trim(calc%socket_host) // c_null_char
    err_msg = ' '

    calc%socket_handle = c_ase_socket_connect( &
      host_c, int(calc%socket_port, c_int), &
      err_msg, int(4096, c_int))

    if (.not. c_associated(calc%socket_handle)) then
      write(stdout,'(a)') '**ERROR** ase-socket: connection failed'
      write(stdout,'(a)') '  ' // trim(err_msg)
      write(stdout,'(a)') '  Make sure the Python ASE socket server is running:'
      write(stdout,'(a,i0)') '    python crest_ase_server.py --port ', calc%socket_port
      iostat = 1
      return
    end if

    write(stdout,'(a)') ' [ase-socket] Connected successfully'
  end subroutine ase_socket_init


  !> Calculate energy and gradient via socket server
  subroutine ase_socket_engrad(mol, calc, energy, gradient, iostat)
    use strucrd, only: coord
    use calc_type, only: calculation_settings
    implicit none

    type(coord), intent(in) :: mol
    type(calculation_settings), intent(inout) :: calc
    real(wp), intent(out) :: energy
    real(wp), intent(out) :: gradient(3, mol%nat)
    integer, intent(out) :: iostat

    integer(c_int) :: status, nat_c
    real(c_double) :: energy_c
    character(len=4096, kind=c_char) :: err_msg
    integer(c_int), allocatable :: at_buf(:)
    integer :: i

    !> Debug timing
    integer(8) :: t_start, t_done, t_count_rate
    real(wp) :: dt_total

    iostat = 0
    nat_c = int(mol%nat, c_int)

    if (calc%socket_debug) call system_clock(t_start, t_count_rate)

    !> Lazy initialization: connect on first call
    if (.not. c_associated(calc%socket_handle)) then
      call ase_socket_init(calc, iostat)
      if (iostat /= 0) return
    end if

    !> Convert atomic numbers to c_int
    allocate(at_buf(nat_c))
    do i = 1, nat_c
      at_buf(i) = int(mol%at(i), c_int)
    end do

    !> Call socket bridge
    !> mol%xyz(3, nat) is column-major = C row-major [nat][3]
    err_msg = ' '
    status = c_ase_socket_engrad( &
      calc%socket_handle, nat_c, &
      mol%xyz(1,1), at_buf(1), &
      int(calc%chrg, c_int), int(calc%uhf, c_int), &
      energy_c, gradient(1,1), err_msg, int(4096, c_int))

    deallocate(at_buf)

    if (status /= 0) then
      write(stdout,'(a)') '**ERROR** ase-socket engrad failed'
      write(stdout,'(a)') '  ' // trim(err_msg)
      iostat = 1
      energy = 0.0_wp
      gradient = 0.0_wp
      return
    end if

    energy = real(energy_c, wp)

    !> Debug timing
    if (calc%socket_debug) then
      call system_clock(t_done)
      dt_total = real(t_done - t_start, wp) / real(t_count_rate, wp)
      calc%socket_call_count = calc%socket_call_count + 1
      calc%socket_total_time = calc%socket_total_time + dt_total
      if (mod(calc%socket_call_count, 50) == 0 .or. &
          calc%socket_call_count == 1) then
        write(stdout,'(a,i0,a,f8.2,a,f10.2,a)') &
          ' [ase-socket #', calc%socket_call_count, '] dt=', &
          dt_total*1000.0_wp, 'ms  cumul=', calc%socket_total_time, 's'
      end if
    end if

  end subroutine ase_socket_engrad


  !> Disconnect from server and reset state
  subroutine ase_socket_cleanup(calc)
    use calc_type, only: calculation_settings
    type(calculation_settings), intent(inout) :: calc

    if (calc%socket_debug .and. calc%socket_call_count > 0) then
      write(stdout,'(a)')      ' [ase-socket] Session summary:'
      write(stdout,'(a,i0)')   '   Total calls:  ', calc%socket_call_count
      write(stdout,'(a,f10.2,a)') '   Total time:   ', calc%socket_total_time, 's'
      if (calc%socket_call_count > 0 .and. calc%socket_total_time > 0.0d0) then
        write(stdout,'(a,f10.2,a)') '   Avg per call: ', &
          calc%socket_total_time / real(calc%socket_call_count, wp) * 1000.0_wp, 'ms'
        write(stdout,'(a,f10.1,a)') '   Throughput:   ', &
          real(calc%socket_call_count, wp) / calc%socket_total_time, ' calls/s'
      end if
    end if

    if (c_associated(calc%socket_handle)) then
      call c_ase_socket_disconnect(calc%socket_handle)
      calc%socket_handle = c_null_ptr
    end if
    calc%socket_call_count = 0
    calc%socket_total_time = 0.0d0

  end subroutine ase_socket_cleanup

end module calc_ase_socket
