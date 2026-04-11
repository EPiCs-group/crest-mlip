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
  public :: crest_worker_opt_run
  public :: crest_worker_pool_run
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
  call mlip_cleanup_all(calc)
  call pymlip_finalize()

  !>--- Exit with appropriate status
  if (term == 0) then
    call exit(0)
  else
    call exit(1)
  end if

end subroutine crest_worker_run

!========================================================================================!
subroutine crest_worker_opt_run(env)
!*********************************************************************
!* Worker entry point for process-based parallel optimization.
!* Reads a chunk of structures, initializes MLIP, optimizes each,
!* writes results, and exits.
!*********************************************************************
  use crest_parameters, only: wp, stdout
  use crest_data
  use crest_calculator
  use strucrd, only: coord
  use optimize_module, only: optimize_geometry
  use worker_io_module, only: read_worker_opt_config, write_worker_opt_results
  use iomod, only: makedir, directory_exist
  implicit none
  type(systemdata), intent(inout) :: env

  integer :: nat, nstructs, worker_index, io, j, i
  integer, allocatable :: at(:)
  real(wp), allocatable :: xyz(:,:,:), xyz_out(:,:,:)
  real(wp), allocatable :: energies(:), grd(:,:)
  integer, allocatable :: opt_status(:)
  type(calcdata) :: calc
  type(coord) :: mol, molnew
  logical :: pr, wr, ex
  character(len=512) :: outfile

  !>--- Read binary config file
  if (.not. allocated(env%worker_config)) then
    write(stdout, '(a)') '**ERROR** Opt worker: no config file specified'
    call exit(1)
  end if

  call read_worker_opt_config(env%worker_config, nat, nstructs, at, xyz, &
                               calc, worker_index, io)
  if (io /= 0) then
    write(stdout, '(a,i0)') '**ERROR** Opt worker: config read failed, status=', io
    call exit(1)
  end if

  write(stdout, '(a,i0,a,i0,a)') &
    'Opt worker ', worker_index, ': ', nstructs, ' structures to optimize'

  !>--- Create working directories
  do j = 1, calc%ncalculations
    if (allocated(calc%calcs(j)%calcspace)) then
      ex = directory_exist(calc%calcs(j)%calcspace)
      if (.not. ex) io = makedir(trim(calc%calcs(j)%calcspace))
    end if
  end do

  !>--- Initialize MLIP model
  do j = 1, calc%ncalculations
    if (calc%calcs(j)%id == jobtype%pymlip) then
      call pymlip_init(calc%calcs(j), io)
      if (io /= 0) then
        write(stdout, '(a,i0)') '**ERROR** Opt worker: pymlip_init failed, level=', j
        call exit(1)
      end if
    end if
    if (calc%calcs(j)%id == jobtype%libtorch) then
      call libtorch_init_shared(calc%calcs(j), io)
      if (io /= 0) then
        write(stdout, '(a,i0)') '**ERROR** Opt worker: libtorch_init failed, level=', j
        call exit(1)
      end if
    end if
  end do

  !>--- Allocate working and output arrays
  allocate(mol%at(nat), mol%xyz(3, nat))
  allocate(molnew%at(nat), molnew%xyz(3, nat))
  allocate(grd(3, nat), source=0.0_wp)
  allocate(xyz_out(3, nat, nstructs))
  allocate(energies(nstructs), source=0.0_wp)
  allocate(opt_status(nstructs), source=-1)

  !>--- Warm-up engrad call
  mol%nat = nat
  mol%at(:) = at(:)
  mol%xyz(:,:) = xyz(:,:,1)
  call engrad(mol, calc, energies(1), grd, io)
  energies(1) = 0.0_wp  ! reset after warmup

  !>--- Optimize each structure in the chunk
  pr = .false.
  wr = .false.
  do i = 1, nstructs
    mol%nat = nat
    mol%at(:) = at(:)
    mol%xyz(:,:) = xyz(:,:,i)
    molnew%nat = nat
    molnew%at(:) = at(:)
    molnew%xyz(:,:) = xyz(:,:,i)
    grd(:,:) = 0.0_wp

    call optimize_geometry(mol, molnew, calc, energies(i), grd, pr, wr, io)

    if (io == 0) then
      xyz_out(:,:,i) = molnew%xyz(:,:)
      opt_status(i) = 0
    else if (io == calc%maxcycle .and. calc%anopt) then
      xyz_out(:,:,i) = molnew%xyz(:,:)
      opt_status(i) = 0  ! partial optimization accepted
    else
      xyz_out(:,:,i) = xyz(:,:,i)  ! keep original
      energies(i) = 1.0_wp         ! flag as failed
      opt_status(i) = io
    end if
  end do

  write(stdout, '(a,i0,a)') 'Opt worker ', worker_index, ' finished'

  !>--- Write results
  write(outfile, '(a,a)') trim(env%worker_config), '.out'
  call write_worker_opt_results(trim(outfile), nat, nstructs, xyz_out, &
                                 energies, opt_status)

  !>--- Cleanup
  call mlip_cleanup_all(calc)
  call pymlip_finalize()

  call exit(0)

end subroutine crest_worker_opt_run

!========================================================================================!
subroutine crest_worker_pool_run(env, fd_read, fd_write, worker_idx)
!*********************************************************************
!* Persistent worker process for the worker pool.
!* Loads the ML model ONCE on the first task, then loops accepting
!* MD or optimization tasks over a pipe without reinitializing.
!*********************************************************************
  use crest_parameters, only: wp, stdout
  use iso_c_binding, only: c_null_ptr, c_associated
  use crest_data
  use crest_calculator
  use strucrd, only: coord
  use dynamics_module
  use optimize_module, only: optimize_geometry
  use worker_io_module, only: read_worker_config, read_worker_opt_config, &
                               write_worker_opt_results
  use worker_pool_module
  use iomod, only: makedir, directory_exist
  implicit none
  type(systemdata), intent(inout) :: env
  integer, intent(in) :: fd_read, fd_write, worker_idx

  !>--- Persistent state (survives across tasks)
  type(calcdata) :: calc
  logical :: model_loaded
  integer :: j, io

  !>--- Task-local variables
  integer :: task_type
  character(len=1024) :: config_path, result_path
  type(coord) :: mol
  type(mddata) :: mddat
  integer :: task_worker_idx
  real(wp) :: etmp
  real(wp), allocatable :: grdtmp(:,:)
  logical :: pr, wr, ex
  integer :: term

  !>--- Optimization task variables
  integer :: nat, nstructs, i
  integer, allocatable :: at(:)
  real(wp), allocatable :: xyz(:,:,:), xyz_out(:,:,:)
  real(wp), allocatable :: energies(:), grd(:,:)
  integer, allocatable :: opt_status(:)
  type(coord) :: molnew
  type(calcdata) :: task_calc

  model_loaded = .false.

  write(stdout, '(a,i0,a)') '[pool] Worker ', worker_idx, ' started, waiting for tasks...'

  !>=========================================================================
  !> EVENT LOOP: receive tasks, execute, send results
  !>=========================================================================
  do while (.true.)

    !>--- Receive next task from parent
    call pool_worker_recv_task_f(fd_read, task_type, config_path, io)
    if (io /= 0) then
      write(stdout, '(a,i0,a)') '[pool] Worker ', worker_idx, &
        ': pipe read failed, exiting'
      exit
    end if

    !>--- Check for shutdown
    if (task_type == POOL_TASK_SHUTDOWN) then
      write(stdout, '(a,i0,a)') '[pool] Worker ', worker_idx, &
        ': received shutdown signal'
      exit
    end if

    !>=========================================================================
    !> MD TASK
    !>=========================================================================
    if (task_type == POOL_TASK_MD) then

      !>--- Read config file (geometry, MD settings, calc settings)
      call read_worker_config(trim(config_path), mol, mddat, &
                               task_calc, task_worker_idx, io)
      if (io /= 0) then
        write(stdout, '(a,i0,a)') '[pool] Worker ', worker_idx, &
          ': MD config read failed'
        call pool_worker_send_result_f(fd_write, 1, '', io)
        cycle
      end if

      !>--- First task: initialize model
      if (.not. model_loaded) then
        calc = task_calc
        call init_mlip_model(calc, worker_idx, io)
        if (io /= 0) then
          call pool_worker_send_result_f(fd_write, 1, '', io)
          cycle
        end if
        !>--- Warmup inference
        allocate(grdtmp(3, mol%nat), source=0.0_wp)
        call engrad(mol, calc, etmp, grdtmp, io)
        deallocate(grdtmp)
        model_loaded = .true.
      else
        !>--- Subsequent tasks: splice live MLIP handles into new config
        call splice_mlip_handles(calc, task_calc)
      end if

      !>--- Create working directories
      do j = 1, calc%ncalculations
        if (allocated(calc%calcs(j)%calcspace)) then
          ex = directory_exist(calc%calcs(j)%calcspace)
          if (.not. ex) io = makedir(trim(calc%calcs(j)%calcspace))
        end if
      end do

      !>--- Run dynamics
      pr = .false.
      call dynamics(mol, mddat, calc, pr, term)

      write(stdout, '(a,i0,a,i0)') '[pool] Worker ', worker_idx, &
        ': MD finished, status=', term

      !>--- Send result (trajectory file is written by dynamics())
      if (term == 0) then
        call pool_worker_send_result_f(fd_write, 0, '', io)
      else
        call pool_worker_send_result_f(fd_write, 1, '', io)
      end if

    !>=========================================================================
    !> OPTIMIZATION TASK
    !>=========================================================================
    else if (task_type == POOL_TASK_OPT) then

      !>--- Read config file (chunk of structures + calc settings)
      call read_worker_opt_config(trim(config_path), nat, nstructs, at, xyz, &
                                   task_calc, task_worker_idx, io)
      if (io /= 0) then
        write(stdout, '(a,i0,a)') '[pool] Worker ', worker_idx, &
          ': OPT config read failed'
        call pool_worker_send_result_f(fd_write, 1, '', io)
        cycle
      end if

      !>--- First task: initialize model
      if (.not. model_loaded) then
        calc = task_calc
        call init_mlip_model(calc, worker_idx, io)
        if (io /= 0) then
          call pool_worker_send_result_f(fd_write, 1, '', io)
          if (allocated(at)) deallocate(at)
          if (allocated(xyz)) deallocate(xyz)
          cycle
        end if
        !>--- Warmup with first structure
        allocate(grdtmp(3, nat), source=0.0_wp)
        allocate(mol%at(nat), mol%xyz(3, nat))
        mol%nat = nat
        mol%at(:) = at(:)
        mol%xyz(:,:) = xyz(:,:,1)
        call engrad(mol, calc, etmp, grdtmp, io)
        deallocate(grdtmp)
        deallocate(mol%at, mol%xyz)
        model_loaded = .true.
      else
        !>--- Splice live handles
        call splice_mlip_handles(calc, task_calc)
      end if

      !>--- Create working directories
      do j = 1, calc%ncalculations
        if (allocated(calc%calcs(j)%calcspace)) then
          ex = directory_exist(calc%calcs(j)%calcspace)
          if (.not. ex) io = makedir(trim(calc%calcs(j)%calcspace))
        end if
      end do

      !>--- Allocate working arrays
      allocate(mol%at(nat), mol%xyz(3, nat))
      allocate(molnew%at(nat), molnew%xyz(3, nat))
      allocate(grd(3, nat), source=0.0_wp)
      allocate(xyz_out(3, nat, nstructs))
      allocate(energies(nstructs), source=0.0_wp)
      allocate(opt_status(nstructs), source=-1)

      !>--- Optimize each structure
      pr = .false.
      wr = .false.
      do i = 1, nstructs
        mol%nat = nat
        mol%at(:) = at(:)
        mol%xyz(:,:) = xyz(:,:,i)
        molnew%nat = nat
        molnew%at(:) = at(:)
        molnew%xyz(:,:) = xyz(:,:,i)
        grd(:,:) = 0.0_wp

        call optimize_geometry(mol, molnew, calc, energies(i), grd, pr, wr, io)

        if (io == 0) then
          xyz_out(:,:,i) = molnew%xyz(:,:)
          opt_status(i) = 0
        else if (io == calc%maxcycle .and. calc%anopt) then
          xyz_out(:,:,i) = molnew%xyz(:,:)
          opt_status(i) = 0
        else
          xyz_out(:,:,i) = xyz(:,:,i)
          energies(i) = 1.0_wp
          opt_status(i) = io
        end if
      end do

      write(stdout, '(a,i0,a,i0,a)') '[pool] Worker ', worker_idx, &
        ': optimized ', nstructs, ' structures'

      !>--- Write results to file
      write(result_path, '(a,a)') trim(config_path), '.out'
      call write_worker_opt_results(trim(result_path), nat, nstructs, &
                                     xyz_out, energies, opt_status)

      !>--- Send result
      call pool_worker_send_result_f(fd_write, 0, trim(result_path), io)

      !>--- Deallocate task-local arrays
      deallocate(mol%at, mol%xyz, molnew%at, molnew%xyz)
      deallocate(grd, xyz_out, energies, opt_status)
      if (allocated(at)) deallocate(at)
      if (allocated(xyz)) deallocate(xyz)

    else
      write(stdout, '(a,i0,a,i0)') '[pool] Worker ', worker_idx, &
        ': unknown task type ', task_type
      call pool_worker_send_result_f(fd_write, 1, '', io)
    end if

  end do  !> event loop

  !>--- Final cleanup
  write(stdout, '(a,i0,a)') '[pool] Worker ', worker_idx, ': shutting down'
  if (model_loaded) then
    call mlip_cleanup_all(calc)
  end if
  call pymlip_finalize()
  call exit(0)

contains

  !> Initialize MLIP model for all calculation levels
  subroutine init_mlip_model(calc, widx, iostat)
    type(calcdata), intent(inout) :: calc
    integer, intent(in) :: widx
    integer, intent(out) :: iostat
    integer :: j, io2
    logical :: ex
    iostat = 0
    do j = 1, calc%ncalculations
      if (allocated(calc%calcs(j)%calcspace)) then
        ex = directory_exist(calc%calcs(j)%calcspace)
        if (.not. ex) io2 = makedir(trim(calc%calcs(j)%calcspace))
      end if
      if (calc%calcs(j)%id == jobtype%pymlip) then
        call pymlip_init(calc%calcs(j), io2)
        if (io2 /= 0) then
          write(stdout, '(a,i0,a)') '[pool] Worker ', widx, &
            ': pymlip_init FAILED'
          iostat = 1
          return
        end if
      end if
      if (calc%calcs(j)%id == jobtype%libtorch) then
        call libtorch_init_shared(calc%calcs(j), io2)
        if (io2 /= 0) then
          write(stdout, '(a,i0,a)') '[pool] Worker ', widx, &
            ': libtorch_init FAILED'
          iostat = 1
          return
        end if
      end if
    end do
  end subroutine

  !> Copy live MLIP handles from persistent calc into task calc,
  !> then copy non-MLIP settings from task_calc back to calc
  subroutine splice_mlip_handles(calc, task_calc)
    type(calcdata), intent(inout) :: calc
    type(calcdata), intent(in) :: task_calc
    integer :: j
    !> Preserve MLIP handles but update optimization/other settings
    calc%optlev = task_calc%optlev
    calc%micro_opt = task_calc%micro_opt
    calc%maxcycle = task_calc%maxcycle
    calc%maxdispl_opt = task_calc%maxdispl_opt
    calc%ethr_opt = task_calc%ethr_opt
    calc%gthr_opt = task_calc%gthr_opt
    calc%hlow_opt = task_calc%hlow_opt
    calc%hmax_opt = task_calc%hmax_opt
    calc%acc_opt = task_calc%acc_opt
    calc%maxerise = task_calc%maxerise
    calc%exact_rf = task_calc%exact_rf
    calc%average_conv = task_calc%average_conv
    calc%tsopt = task_calc%tsopt
    calc%iupdat = task_calc%iupdat
    calc%opt_engine = task_calc%opt_engine
    calc%anopt = task_calc%anopt
    !> Update calcspace paths from task_calc (different per-worker dirs)
    do j = 1, min(calc%ncalculations, task_calc%ncalculations)
      if (allocated(task_calc%calcs(j)%calcspace)) then
        calc%calcs(j)%calcspace = task_calc%calcs(j)%calcspace
      end if
    end do
  end subroutine

end subroutine crest_worker_pool_run

end module worker_md_module
