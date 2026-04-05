!================================================================================!
! This file is part of crest.
! SPDX-Identifier: LGPL-3.0-or-later
!
! Worker process entry point for process-based parallel MD.
! Each worker is a separate CREST process that:
!   1. Reads molecule + MD config from a binary file
!   2. Initializes its own MLIP model (fresh Python interpreter)
!   3. Runs dynamics() for the assigned MTD simulation
!   4. Writes trajectory and exits
!
! This bypasses the Python GIL limitation by using separate processes
! instead of OpenMP threads.
!================================================================================!
module worker_md_module
  implicit none
  private
  public :: crest_worker_run
contains

!========================================================================================!
subroutine crest_worker_run(env)
!*********************************************************************
!* Main entry point for a worker process launched with --worker flag.
!* Reads config, initializes calculator, runs MD, cleans up, exits.
!*********************************************************************
  use crest_parameters, only: wp, stdout
  use crest_data
  use crest_calculator
  use strucrd, only: coord
  use dynamics_module
  use worker_io_module, only: read_worker_config
  use iomod, only: makedir, directory_exist
  implicit none
  type(systemdata), intent(inout) :: env

  type(coord) :: mol
  type(mddata) :: mddat
  type(calcdata) :: calc
  integer :: worker_index, io, j
  real(wp) :: etmp
  real(wp), allocatable :: grdtmp(:,:)
  logical :: pr, ex
  integer :: term

  !>--- Read binary config file
  if (.not. allocated(env%worker_config)) then
    write(stdout, '(a)') '**ERROR** Worker: no config file specified'
    call exit(1)
  end if

  call read_worker_config(env%worker_config, mol, mddat, calc, worker_index, io)
  if (io /= 0) then
    write(stdout, '(a,i0)') '**ERROR** Worker: failed to read config, status=', io
    call exit(1)
  end if

  write(stdout, '(a,i0,a,i0,a)') &
    'Worker process ', worker_index, ' started (PID ', getpid(), ')'
  write(stdout, '(a,a)') '  Config: ', trim(env%worker_config)
  write(stdout, '(a,a)') '  Trajectory: ', trim(mddat%trajectoryfile)
  write(stdout, '(a,i0,a)') '  MD steps: ', mddat%length_steps, ''

  !>--- Create working directories for each calculation level
  do j = 1, calc%ncalculations
    if (allocated(calc%calcs(j)%calcspace)) then
      ex = directory_exist(calc%calcs(j)%calcspace)
      if (.not. ex) then
        io = makedir(trim(calc%calcs(j)%calcspace))
      end if
    end if
  end do

  !>--- Initialize the calculator (loads fresh Python interpreter + model)
  do j = 1, calc%ncalculations
    if (calc%calcs(j)%id == jobtype%pymlip) then
      call pymlip_init(calc%calcs(j), io)
      if (io /= 0) then
        write(stdout, '(a,i0)') '**ERROR** Worker: pymlip_init failed for level ', j
        call exit(1)
      end if
    end if
    if (calc%calcs(j)%id == jobtype%libtorch) then
      call libtorch_init_shared(calc%calcs(j), io)
      if (io /= 0) then
        write(stdout, '(a,i0)') '**ERROR** Worker: libtorch_init failed for level ', j
        call exit(1)
      end if
    end if
  end do

  !>--- Warm-up engrad call (initializes internal state)
  allocate(grdtmp(3, mol%nat), source=0.0_wp)
  call engrad(mol, calc, etmp, grdtmp, io)
  deallocate(grdtmp)

  !>--- Run dynamics
  pr = .false.  ! suppress per-step printout (parent collects trajectories)
  call dynamics(mol, mddat, calc, pr, term)

  write(stdout, '(a,i0,a,i0)') 'Worker ', worker_index, ' finished, status=', term

  !>--- Cleanup MLIP handles
  do j = 1, calc%ncalculations
    if (calc%calcs(j)%id == jobtype%pymlip) then
      call pymlip_cleanup(calc%calcs(j))
    end if
    if (calc%calcs(j)%id == jobtype%libtorch) then
      call libtorch_cleanup(calc%calcs(j))
    end if
  end do
  call pymlip_finalize()

  !>--- Exit with appropriate status
  if (term == 0) then
    call exit(0)
  else
    call exit(1)
  end if

end subroutine crest_worker_run

end module worker_md_module
