!================================================================================!
! This file is part of crest.
!
! Copyright (C) 2023 Philipp Pracht
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

!> module subprocess_engrad
!> RE-EXPORTS of subprocess engrad routines

!=========================================================================================!

module subprocess_engrad
  use generic_sc
  use xtb_sc
  use turbom_sc
  use orca_sc
  use calc_libtorch, only: libtorch_engrad, libtorch_cleanup, &
    libtorch_set_threads, libtorch_init_shared, &
    libtorch_engrad_batch_f, libtorch_shared_cleanup, &
    libtorch_engrad_batch_pipeline_f, &
    libtorch_engrad_batch_multigpu_f, &
    libtorch_load_shared_on_device_f, &
    libtorch_get_cuda_device_count_f
  use calc_pymlip, only: pymlip_engrad, pymlip_engrad_batch_f, &
    pymlip_init, pymlip_cleanup, pymlip_finalize
  use calc_ase_socket, only: ase_socket_engrad, ase_socket_init, &
    ase_socket_cleanup
  implicit none
  !>--- private module variables and parameters
  private

  !>--- generic subrpocess (run.sh)
  public :: generic_engrad

  !>--- xtb subprocess
  public :: xtb_engrad

  !>--- Turbomole-style subprocesses
  public :: turbom_engrad

  !>--- ORCA subprocesses
  public :: ORCA_engrad

  !>--- libtorch direct MLIP inference
  public :: libtorch_engrad
  public :: libtorch_cleanup
  public :: libtorch_set_threads
  public :: libtorch_init_shared
  public :: libtorch_engrad_batch_f
  public :: libtorch_shared_cleanup
  public :: libtorch_engrad_batch_pipeline_f
  public :: libtorch_engrad_batch_multigpu_f
  public :: libtorch_load_shared_on_device_f
  public :: libtorch_get_cuda_device_count_f

  !>--- embedded Python MLIP inference
  public :: pymlip_engrad
  public :: pymlip_engrad_batch_f
  public :: pymlip_init
  public :: pymlip_cleanup
  public :: pymlip_finalize

  !>--- ASE socket calculator
  public :: ase_socket_engrad
  public :: ase_socket_init
  public :: ase_socket_cleanup

!=========================================================================================!
!=========================================================================================!
contains    !> MODULE PROCEDURES START HERE
!=========================================================================================!
!=========================================================================================!

end module subprocess_engrad
