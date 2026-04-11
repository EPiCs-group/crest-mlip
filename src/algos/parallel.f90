!================================================================================!
! This file is part of crest.
!
! Copyright (C) 2022-2023  Philipp Pracht
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

!> A collection of routines to set up OMP-parallel runs of MDs and optimizations.

!========================================================================================!
!========================================================================================!
!> Interfaces to handle optional arguments
!========================================================================================!
!========================================================================================!
module parallel_interface
!*******************************************************
!* module to load an interface to the parallel routines
!* mandatory to handle any optional input arguments
!*******************************************************
  implicit none
  interface
    subroutine crest_sploop(env,nat,nall,at,xyz,eread)
      use crest_parameters,only:wp,stdout,sep
      use crest_calculator
      use omp_lib
      use crest_data
      use strucrd
      use optimize_module
      use iomod,only:makedir,directory_exist,remove
      implicit none
      type(systemdata),intent(inout) :: env
      real(wp),intent(inout) :: xyz(3,nat,nall)
      integer,intent(in)  :: at(nat)
      real(wp),intent(inout) :: eread(nall)
      integer,intent(in) :: nat,nall
    end subroutine crest_sploop
  end interface

  interface
    subroutine crest_oloop(env,nat,nall,at,xyz,eread,dump,customcalc)
      use crest_parameters,only:wp,stdout,sep
      use crest_calculator
      use omp_lib
      use crest_data
      use strucrd
      use optimize_module
      use iomod,only:makedir,directory_exist,remove
      use crest_restartlog,only:trackrestart,restart_write_dummy
      implicit none
      type(systemdata),target,intent(inout) :: env
      real(wp),intent(inout) :: xyz(3,nat,nall)
      integer,intent(in)  :: at(nat)
      real(wp),intent(inout) :: eread(nall)
      integer,intent(in) :: nat,nall
      logical,intent(in) :: dump
      type(calcdata),intent(in),target,optional :: customcalc
    end subroutine crest_oloop
  end interface
end module parallel_interface

!========================================================================================!
!========================================================================================!
!> Routines for concurrent singlepoint evaluations
!========================================================================================!
!========================================================================================!
subroutine crest_sploop(env,nat,nall,at,xyz,eread)
!***************************************************************
!* subroutine crest_sploop
!* This subroutine performs concurrent singlepoint evaluations
!* for the given ensemble. Input eread is overwritten
!***************************************************************
  use crest_parameters,only:wp,stdout,sep
  use crest_calculator
  use iso_c_binding,only:c_null_ptr
  use omp_lib
  use crest_data
  use strucrd
  use optimize_module
  use iomod,only:makedir,directory_exist,remove
  implicit none
  type(systemdata),intent(inout) :: env
  real(wp),intent(inout) :: xyz(3,nat,nall)
  integer,intent(in)  :: at(nat)
  real(wp),intent(inout) :: eread(nall)
  integer,intent(in) :: nat,nall

  type(coord),allocatable :: mols(:)
  integer :: i,j,k,l,io,ich,ich2,c,z,job_id,zcopy
  logical :: pr,wr,ex
  type(calcdata),allocatable :: calculations(:)
  real(wp) :: energy,gnorm
  real(wp),allocatable :: grad(:,:),grads(:,:,:)
  integer :: thread_id,vz,job
  character(len=80) :: atmp
  real(wp) :: percent,runtime

  type(timer) :: profiler
  integer :: T,Tn  !> threads and threads per core
  logical :: nested

!>--- check if we have any calculation settings allocated
  if (env%calc%ncalculations < 1) then
    write (stdout,*) 'no calculations allocated'
    return
  end if

!>--- Prepare per-thread calculation objects for parallel singlepoints.
!>    Each OpenMP thread gets its own copy of the calculator settings,
!>    its own working directory, and its own mol object.  MLIP handles
!>    are reset to null here and populated in the shared model setup
!>    below (one model loaded → handle broadcast to all threads).
  call new_ompautoset(env,'auto_nested',nall,T,Tn)
  nested = env%omp_allow_nested

  T = env%threads
  allocate (calculations(T),source=env%calc)
  allocate (mols(T))
  do i = 1,T
    do j = 1,env%calc%ncalculations
      calculations(i)%calcs(j) = env%calc%calcs(j)
      !>--- per-thread working directories
      ex = directory_exist(env%calc%calcs(j)%calcspace)
      if (.not.ex) then
        io = makedir(trim(env%calc%calcs(j)%calcspace))
      end if
      write (atmp,'(a,"_",i0)') sep,i
      calculations(i)%calcs(j)%calcspace = env%calc%calcs(j)%calcspace//trim(atmp)
      if(allocated(calculations(i)%calcs(j)%calcfile)) deallocate(calculations(i)%calcs(j)%calcfile)
      if(allocated(calculations(i)%calcs(j)%systemcall)) deallocate(calculations(i)%calcs(j)%systemcall)
      !>--- libtorch: null out handles — shared model set after this loop
      if (calculations(i)%calcs(j)%id == jobtype%libtorch) then
        calculations(i)%calcs(j)%libtorch_handle = c_null_ptr
        calculations(i)%calcs(j)%libtorch_call_count = 0
        calculations(i)%calcs(j)%libtorch_total_time = 0.0d0
      end if
      !>--- pymlip: null out handles — shared model set after this loop
      if (calculations(i)%calcs(j)%id == jobtype%pymlip) then
        calculations(i)%calcs(j)%pymlip_handle = c_null_ptr
        calculations(i)%calcs(j)%pymlip_call_count = 0
        calculations(i)%calcs(j)%pymlip_total_time = 0.0d0
      end if
      call calculations(i)%calcs(j)%printid(i,j)
    end do
    calculations(i)%pr_energies = .false.
    allocate (mols(i)%at(nat),mols(i)%xyz(3,nat))
  end do

!>--- libtorch: load shared model ONCE, propagate handle to all threads
  do j = 1,env%calc%ncalculations
    if (env%calc%calcs(j)%id == jobtype%libtorch) then
      call libtorch_init_shared(env%calc%calcs(j), io)
      if (io /= 0) then
        write(stdout,'(a)') '**ERROR** Failed to load shared libtorch model'
        return
      end if
      do i = 1,T
        calculations(i)%calcs(j)%libtorch_handle = env%calc%calcs(j)%libtorch_handle
      end do
    end if
  end do

!>--- pymlip: load shared model ONCE, propagate handle to all threads
!>    GIL serializes Python calls, so sharing one model is safe and
!>    avoids loading T copies of model weights into GPU memory.
  do j = 1,env%calc%ncalculations
    if (env%calc%calcs(j)%id == jobtype%pymlip) then
      call pymlip_init(env%calc%calcs(j), io)
      if (io /= 0) then
        write(stdout,'(a)') '**ERROR** Failed to load shared pymlip model'
        return
      end if
      do i = 1,T
        calculations(i)%calcs(j)%pymlip_handle = env%calc%calcs(j)%pymlip_handle
      end do
    end if
  end do

!>--- printout directions and timer initialization
  pr = .false. !> stdout printout
  wr = .false. !> write crestopt.log
  call profiler%init(1)
  call profiler%start(1)

!>--- Configure ATen (PyTorch internal) thread count for libtorch.
!>    ATen controls intra-op parallelism inside each forward pass.
!>    - GPU mode: ATen threads=1 (GPU handles SIMD parallelism internally;
!>      extra CPU threads just waste resources and cause contention)
!>    - CPU mode with shared model: ATen threads=T (forward passes are
!>      serialized by mutex, so each pass can use all available cores)
!>    - User override via mlip_aten_threads takes priority
  do j = 1,env%calc%ncalculations
    if (env%calc%calcs(j)%id == jobtype%libtorch) then
      if (env%calc%calcs(j)%mlip_aten_threads > 0) then
        call libtorch_set_threads(env%calc%calcs(j)%mlip_aten_threads)
      else if (env%calc%calcs(j)%libtorch_device_id > 0) then
        !> GPU: ATen threads=1 (GPU handles parallelism)
        call libtorch_set_threads(1)
      else
        !> CPU shared model: let ATen use all threads (only 1 forward at a time)
        call libtorch_set_threads(T)
      end if
      exit
    end if
  end do

!>=========================================================================
!> GPU BATCHED INFERENCE PATH
!>=========================================================================
!> When an MLIP runs on GPU, the standard OpenMP task loop is inefficient:
!> each thread calls engrad() sequentially (serialized by mutex/GIL),
!> underutilizing the GPU which excels at processing many structures at once.
!>
!> Instead, we pack ALL structures into a contiguous buffer and send them
!> to the C++/Python layer in batches.  The GPU processes each batch in a
!> single forward pass, amortizing kernel launch overhead and maximizing
!> throughput.  This bypasses the OpenMP loop entirely.
!>
!> Two sub-paths exist:
!>   1. libtorch GPU:  Pipelined batching in C++ (single or multi-GPU)
!>   2. pymlip GPU:    Sequential batching via Python (single GPU + GIL)
!>
!> Batch size is auto-tuned from atom count if not set by user:
!>   nat < 30  → batch=64  (small molecules, GPU can handle many)
!>   nat < 100 → batch=16  (medium, balance memory vs throughput)
!>   nat >= 100 → batch=4  (large, conserve GPU memory)
!>
!> If the GPU path fails, execution falls through to label 100 (the
!> standard per-thread CPU path) as a graceful fallback.
!>=========================================================================
  block
    use iso_c_binding, only: c_ptr, c_null_ptr, c_associated
    logical :: use_batch_gpu
    integer :: batch_sz, ngpus, ig
    real(wp), allocatable :: all_pos(:), all_grad(:)
    type(c_ptr), allocatable :: gpu_handles(:)

    use_batch_gpu = .false.
    batch_sz = 0
    ngpus = 0
    do j = 1,env%calc%ncalculations
      if (env%calc%calcs(j)%id == jobtype%libtorch .and. &
          env%calc%calcs(j)%libtorch_device_id > 0) then
        use_batch_gpu = .true.
        batch_sz = env%calc%calcs(j)%mlip_batch_size
        ngpus = env%calc%calcs(j)%mlip_ngpus
        if (batch_sz <= 0) batch_sz = mlip_auto_batch_size(nat)
        exit
      end if
    end do

    if (use_batch_gpu .and. env%calc%ncalculations == 1) then
      !> ============================================================
      !> GPU BATCHED PATH: pipelined + optional multi-GPU
      !> ============================================================

      !> Determine number of GPUs
      if (ngpus <= 0) then
        ngpus = libtorch_get_cuda_device_count_f()
        if (ngpus > 2) ngpus = 2  !> cap at 2 for safety
      end if
      if (ngpus < 1) ngpus = 1

      if (env%calc%calcs(1)%libtorch_debug) then
        write(stdout,'(a,i0,a,i0,a,i0,a)') &
          ' [libtorch] GPU pipelined mode: ', nall, &
          ' structures, batch_size=', batch_sz, ', ngpus=', ngpus, ''
      end if

      !> Pack ALL positions into contiguous buffer for C++
      allocate(all_pos(3*nat*nall))
      allocate(all_grad(3*nat*nall))
      all_pos(1:3*nat*nall) = reshape(xyz(:,:,1:nall), [3*nat*nall])

      !> First progress printout
      call crest_oloop_pr_progress(env,nall,0)
      eread(:) = 0.0_wp

      if (ngpus == 1) then
        !> --- Single GPU: pipelined batch inference ---
        call libtorch_init_shared(env%calc%calcs(1), io)
        if (io /= 0) then
          write(stdout,'(a)') '**ERROR** libtorch: shared model init failed'
          deallocate(all_pos, all_grad)
          goto 100
        end if

        call libtorch_engrad_batch_pipeline_f(env%calc%calcs(1), &
          nall, nat, at, all_pos, eread, all_grad, batch_sz, io)

        call libtorch_shared_cleanup()
        env%calc%calcs(1)%libtorch_handle = c_null_ptr

        if (io == 0) then
          c = nall
        else
          c = 0
          write(stdout,'(a)') '**ERROR** libtorch pipeline failed'
        end if

      else
        !> --- Multi-GPU: interleaved pipelined batch inference ---
        allocate(gpu_handles(ngpus))
        do ig = 1, ngpus
          call libtorch_load_shared_on_device_f(env%calc%calcs(1), &
            ig-1, gpu_handles(ig), io)
          if (io /= 0) then
            write(stdout,'(a,i0)') '**ERROR** libtorch: failed to load on CUDA:', ig-1
            !> Cleanup already loaded handles
            call libtorch_shared_cleanup()
            env%calc%calcs(1)%libtorch_handle = c_null_ptr
            deallocate(gpu_handles, all_pos, all_grad)
            goto 100
          end if
        end do

        call libtorch_engrad_batch_multigpu_f(gpu_handles, ngpus, &
          nall, nat, at, all_pos, eread, all_grad, batch_sz, io)

        call libtorch_shared_cleanup()
        env%calc%calcs(1)%libtorch_handle = c_null_ptr
        deallocate(gpu_handles)

        if (io == 0) then
          c = nall
        else
          c = 0
          write(stdout,'(a)') '**ERROR** libtorch multi-GPU pipeline failed'
        end if
      end if

      deallocate(all_grad, all_pos)

      !> Finalize progress
      call crest_oloop_pr_progress(env,nall,-1)

      !> Stop timer and print summary
      call profiler%stop(1)
      percent = float(c)/float(nall)*100.0_wp
      write (atmp,'(f5.1,a)') percent,'% success)'
      write (stdout,'(">",1x,i0,a,i0,a,a)') c,' of ',nall, &
        ' structures successfully evaluated (', trim(adjustl(atmp))
      write (atmp,'(">",1x,a,i0,a)') 'Total runtime for ',nall, &
        ' singlepoint calculations:'
      call profiler%write_timing(stdout,1,trim(atmp),.true.)
      runtime = profiler%get(1)
      write (atmp,'(f16.3,a)') runtime/real(nall,wp),' sec'
      write (stdout,'(a,a,a)') '> Corresponding to approximately ', &
        trim(adjustl(atmp)), ' per processed structure'
      write (stdout,'(">",1x,a,i0)') 'Total number of energy+grad calls: ',c
      call profiler%clear()
      deallocate (calculations)
      if (allocated(mols)) deallocate (mols)
      return

    end if
  end block

!>--- PyMLIP GPU batched path (sequential batch, single GIL)
!>    Similar to libtorch but simpler: Python GIL means only one forward
!>    pass runs at a time anyway, so we just loop over batches sequentially.
!>    No multi-GPU support for pymlip (would need multiple interpreters).
  block
    use iso_c_binding, only: c_ptr, c_null_ptr, c_associated
    logical :: use_pymlip_batch
    integer :: batch_sz
    real(wp), allocatable :: all_pos(:), all_grad(:)

    use_pymlip_batch = .false.
    batch_sz = 0
    do j = 1,env%calc%ncalculations
      if (env%calc%calcs(j)%id == jobtype%pymlip .and. &
          allocated(env%calc%calcs(j)%pymlip_device)) then
        if (env%calc%calcs(j)%pymlip_device /= 'cpu') then
          use_pymlip_batch = .true.
          batch_sz = env%calc%calcs(j)%mlip_batch_size
          if (batch_sz <= 0) batch_sz = mlip_auto_batch_size(nat)
          exit
        end if
      end if
    end do

    if (use_pymlip_batch .and. env%calc%ncalculations == 1) then
      !> ============================================================
      !> PyMLIP GPU BATCHED PATH: sequential batch with single GIL
      !> ============================================================

      if (env%calc%calcs(1)%pymlip_debug) then
        write(stdout,'(a,i0,a,i0)') &
          ' [pymlip] GPU batch mode: ', nall, &
          ' structures, batch_size=', batch_sz
      end if

      !> Pack all positions
      allocate(all_pos(3*nat*nall))
      allocate(all_grad(3*nat*nall))
      !$omp parallel do private(i) schedule(static)
      do i = 1, nall
        all_pos((i-1)*nat*3+1 : i*nat*3) = reshape(xyz(:,:,i), [nat*3])
      end do
      !$omp end parallel do

      call crest_oloop_pr_progress(env,nall,0)
      eread(:) = 0.0_wp

      !> Process in batches
      c = 0
      do i = 1, nall, batch_sz
        k = min(batch_sz, nall - i + 1)
        call pymlip_engrad_batch_f(env%calc%calcs(1), &
          k, nat, at, &
          all_pos((i-1)*nat*3+1), &
          eread(i), &
          all_grad((i-1)*nat*3+1), &
          io)
        if (io /= 0) then
          write(stdout,'(a,i0,a,i0)') &
            '**ERROR** pymlip batch failed at structure ', i, '-', i+k-1
          exit
        end if
        c = c + k
        call crest_oloop_pr_progress(env,nall,c)
      end do

      deallocate(all_grad, all_pos)

      call crest_oloop_pr_progress(env,nall,-1)
      call profiler%stop(1)
      percent = float(c)/float(nall)*100.0_wp
      write (atmp,'(f5.1,a)') percent,'% success)'
      write (stdout,'(">",1x,i0,a,i0,a,a)') c,' of ',nall, &
        ' structures successfully evaluated (', trim(adjustl(atmp))
      write (atmp,'(">",1x,a,i0,a)') 'Total runtime for ',nall, &
        ' singlepoint calculations:'
      call profiler%write_timing(stdout,1,trim(atmp),.true.)
      runtime = profiler%get(1)
      write (atmp,'(f16.3,a)') runtime/real(nall,wp),' sec'
      write (stdout,'(a,a,a)') '> Corresponding to approximately ', &
        trim(adjustl(atmp)), ' per processed structure'
      write (stdout,'(">",1x,a,i0)') 'Total number of energy+grad calls: ',c
      call profiler%clear()
      !>--- Cleanup shared MLIP model
      call mlip_cleanup_all(env%calc)
      deallocate (calculations)
      if (allocated(mols)) deallocate (mols)
      return
    end if
  end block

!>--- Standard per-thread OpenMP path (CPU mode, or GPU fallback on error)
!>    Each thread processes structures one-at-a-time via engrad().
!>    For xTB/GFN-FF this is the normal path.  For MLIP on CPU this works
!>    but is slow — the warnings below inform the user.
100 continue

!>--- Warn if running MLIP on CPU (GPU is 10-100x faster for inference)
  do j = 1,env%calc%ncalculations
    if (env%calc%calcs(j)%id == jobtype%libtorch .and. &
        env%calc%calcs(j)%libtorch_device_id == 0) then
      write(stdout,'(a)') ' [libtorch] NOTE: Running MLIP on CPU. '// &
        'For production throughput, set device=''cuda''. '// &
        'For CPU-only work, consider method=''gfn2''.'
      exit
    end if
    !>--- Warn if libtorch GPU fell through to the serialized OpenMP path
    if (env%calc%calcs(j)%id == jobtype%libtorch .and. &
        env%calc%calcs(j)%libtorch_device_id > 0 .and. T > 1) then
      write(stdout,'(a)') ' [libtorch] WARNING: GPU MLIP fell through to '// &
        'per-thread OpenMP path. All forward passes are serialized by '// &
        'forward_mutex — this is SLOWER than single-threaded.'
      write(stdout,'(a)') '   Prefer the batched GPU path (ensure '// &
        'ncalculations=1) or set threads=1.'
      exit
    end if
  end do

!>--- Warn if running pymlip on CPU or serialized GPU path
  do j = 1,env%calc%ncalculations
    if (env%calc%calcs(j)%id == jobtype%pymlip) then
      if (.not.allocated(env%calc%calcs(j)%pymlip_device) .or. &
          env%calc%calcs(j)%pymlip_device == 'cpu') then
        write(stdout,'(a)') ' [pymlip] NOTE: Running MLIP on CPU. '// &
          'For faster inference, set device=''cuda'' in the input.'
      else
        if (T > 1) then
          write(stdout,'(a,i0,a)') ' [pymlip] WARNING: GPU mode with ', T, &
            ' threads, but Python GIL serializes all forward passes — '// &
            'this is SLOWER than single-threaded. '// &
            'Prefer the batched path (ncalculations=1) or set threads=1.'
        end if
      end if
      exit
    end if
  end do

!>--- first progress printout (initializes progress variables)
  call crest_oloop_pr_progress(env,nall,0)

!>--- shared variables
  allocate (grads(3,nat,T),source=0.0_wp)
  c = 0  !> counter of successfull optimizations
  k = 0  !> counter of total optimization (fail+success)
  z = 0  !> counter to perform optimization in right order (1...nall)
  eread(:) = 0.0_wp
  grads(:,:,:) = 0.0_wp
!>--- loop over ensemble
  !$omp parallel &
  !$omp shared(env,calculations,nat,nall,at,xyz,eread,grads,c,k,z,pr,wr) &
  !$omp shared(ich,ich2,mols, nested,Tn)
  !$omp single
  do i = 1,nall

    call initsignal()
    vz = i
    !$omp task firstprivate( vz ) private(i,j,job,energy,io,thread_id,zcopy)
    call initsignal()

    !>--- OpenMP nested region threads
    if (nested) call ompmklset(Tn)

    thread_id = OMP_GET_THREAD_NUM()
    job = thread_id+1
    !>--- modify calculation spaces
    !$omp critical
    z = z+1
    zcopy = z
    mols(job)%nat = nat
    mols(job)%at(:) = at(:)
    mols(job)%xyz(:,:) = xyz(:,:,z)
    !$omp end critical

    !>-- engery+gradient call
    call engrad(mols(job),calculations(job),energy,grads(:,:,job),io)

    !$omp critical
    if (io == 0) then
      !>--- successful optimization (io==0)
      c = c+1
      eread(zcopy) = energy
    else
      eread(zcopy) = 0.0_wp
    end if
    k = k+1
    !>--- print progress
    call crest_oloop_pr_progress(env,nall,k)
    !$omp end critical
    !$omp end task
  end do
  !$omp taskwait
  !$omp end single
  !$omp end parallel

!>--- finalize progress printout
  call crest_oloop_pr_progress(env,nall,-1)

!>--- stop timer
  call profiler%stop(1)

!>--- prepare some summary printout
  percent = float(c)/float(nall)*100.0_wp
  write (atmp,'(f5.1,a)') percent,'% success)'
  write (stdout,'(">",1x,i0,a,i0,a,a)') c,' of ',nall,' structures successfully evaluated (', &
  &     trim(adjustl(atmp))
  write (atmp,'(">",1x,a,i0,a)') 'Total runtime for ',nall,' singlepoint calculations:'
  call profiler%write_timing(stdout,1,trim(atmp),.true.)
  runtime = profiler%get(1)
  write (atmp,'(f16.3,a)') runtime/real(nall,wp),' sec'
  write (stdout,'(a,a,a)') '> Corresponding to approximately ',trim(adjustl(atmp)), &
  &                       ' per processed structure'

  deallocate (grads)
  call profiler%clear()
  !>--- MLIP per-thread cleanup.  In shared-model mode, all T threads
  !>    point to the SAME C/C++ handle (loaded once on master thread).
  !>    We nullify the handle before calling cleanup so that cleanup only
  !>    prints accumulated stats (call count, total time) without freeing
  !>    the shared model — the actual model is freed by libtorch_shared_cleanup()
  !>    or mlip_cleanup_all() on the master's env%calc.
  do i = 1,T
    do j = 1,env%calc%ncalculations
      if (calculations(i)%calcs(j)%id == jobtype%libtorch) &
        calculations(i)%calcs(j)%libtorch_handle = c_null_ptr
      if (calculations(i)%calcs(j)%id == jobtype%pymlip) &
        calculations(i)%calcs(j)%pymlip_handle = c_null_ptr
      if (calculations(i)%calcs(j)%id == jobtype%ase_socket) &
        calculations(i)%calcs(j)%socket_handle = c_null_ptr
    end do
    call mlip_cleanup_all(calculations(i))
  end do
  !>--- Free shared models (skipped if mlip_keep_loaded for persistence)
  if (.not. env%calc%mlip_keep_loaded) then
    call libtorch_shared_cleanup()
    do j = 1,env%calc%ncalculations
      if (env%calc%calcs(j)%id == jobtype%libtorch) &
        env%calc%calcs(j)%libtorch_handle = c_null_ptr
    end do
  end if
  call mlip_cleanup_all(env%calc)
  deallocate (calculations)
  if (allocated(mols)) deallocate (mols)
  return
end subroutine crest_sploop

!========================================================================================!
!========================================================================================!
!> Routines for concurrent geometry optimization
!========================================================================================!
!========================================================================================!
subroutine crest_oloop(env,nat,nall,at,xyz,eread,dump,customcalc)
!*******************************************************************************
!* subroutine crest_oloop
!* This subroutine performs concurrent geometry optimizations
!* for the given ensemble. Inputs xyz and eread are overwritten
!* env        - contains parallelization and other program settings
!* dump       - decides on whether to dump an ensemble file
!*              WARNING: the ensemble file will NOT be in the same order
!*              as the input xyz array. However, the overwritten xyz will be!
!* customcalc - customized (optional) calculation level data
!*
!* IMPORTANT: xyz should be in Bohr(!) for this routine
!******************************************************************************
  use crest_parameters,only:wp,stdout,sep
  use crest_calculator
  use iso_c_binding,only:c_null_ptr,c_int,c_char,c_null_char
  use omp_lib
  use crest_data
  use strucrd
  use optimize_module
  use iomod,only:makedir,directory_exist,remove
  use crest_restartlog,only:trackrestart,restart_write_dummy
  use worker_io_module,only:write_worker_opt_config,read_worker_opt_results
  use worker_pool_module
  implicit none
  type(systemdata),target,intent(inout) :: env
  real(wp),intent(inout) :: xyz(3,nat,nall)
  integer,intent(in)  :: at(nat)
  real(wp),intent(inout) :: eread(nall)
  integer,intent(in) :: nat,nall
  logical,intent(in) :: dump
  type(calcdata),intent(in),target,optional :: customcalc

  type(coord),allocatable :: mols(:)
  type(coord),allocatable :: molsnew(:)
  integer :: i,j,k,l,io,ich,ich2,c,z,job_id,zcopy
  logical :: pr,wr,ex
  type(calcdata),allocatable :: calculations(:)
  real(wp) :: energy,gnorm
  real(wp),allocatable :: grads(:,:,:)
  integer :: thread_id,vz,job
  character(len=80) :: atmp
  real(wp) :: percent,runtime
  type(calcdata),pointer :: mycalc
  type(timer) :: profiler
  integer :: T,Tn  !> threads and threads per core
  logical :: nested

  !>--- process-based optimization variables
  logical :: use_process_parallel
  integer :: n_workers, chunk_size, idx_start, idx_end, n_chunk, n_failed
  integer,allocatable :: pids(:), exit_codes(:), opt_stat_chunk(:)
  real(wp),allocatable :: e_chunk(:), xyz_chunk(:,:,:)
  character(len=512) :: progname, config_file, cmd_str, outfile
  type(calcdata) :: worker_calc
  integer :: exitstat

  !>--- C interface for process bridge
  interface
    function c_spawn_process(cmd) bind(C, name='spawn_process')
      import :: c_int, c_char
      character(kind=c_char), intent(in) :: cmd(*)
      integer(c_int) :: c_spawn_process
    end function
    function c_wait_for_process(pid) bind(C, name='wait_for_process')
      import :: c_int
      integer(c_int), value, intent(in) :: pid
      integer(c_int) :: c_wait_for_process
    end function
  end interface

!>--- decide wether to skip this call
  if (trackrestart(env)) then
    call restart_write_dummy(ensemblefile)
    return
  end if

!>--- check which calc to use
  if(present(customcalc))then
    mycalc => customcalc
  else
    mycalc => env%calc
  endif

!>--- check if we have any calculation settings allocated
  if (mycalc%ncalculations < 1) then
    write (stdout,*) 'no calculations allocated'
    return
  end if

!>--- prepare calculation objects for parallelization (one per thread)
  call new_ompautoset(env,'auto_nested',nall,T,Tn)
  nested = env%omp_allow_nested

!==========================================================================!
!> PROCESS-BASED OPTIMIZATION (pymlip/libtorch GPU — bypasses GIL/mutex)
!==========================================================================!
  use_process_parallel = mlip_needs_process_parallel(mycalc)

  if (use_process_parallel .and. T > 1 .and. nall > 1) then

    call get_command_argument(0, progname)

    !>--- Try persistent worker pool first (model already loaded from MD step)
    if (.not. pool_is_active_f()) then
      n_workers = min(T, nall)
      write(stdout,'(/,1x,a,i0,a)') &
        'Creating persistent worker pool with ', n_workers, ' workers'
      call pool_create_f(n_workers, trim(progname), io)
      if (io /= 0) then
        write(stdout,'(a)') '  Pool creation failed, falling back to spawn/wait'
        goto 300  !> fall through to legacy path
      end if
    end if

    if (pool_is_active_f()) then
      !> ============================================================
      !> POOL OPTIMIZATION PATH: reuse persistent workers
      !> ============================================================
      n_workers = pool_get_n_workers_f()
      chunk_size = (nall + n_workers - 1) / n_workers
      n_workers = (nall + chunk_size - 1) / chunk_size

      write(stdout,'(/,1x,a)') 'Using persistent worker pool for MLIP optimization'
      write(stdout,'(1x,a,i0,a,i0,a,i0,a)') &
        'Splitting ', nall, ' structures into ', n_workers, &
        ' chunks (~', chunk_size, ' each)'

      call print_opt_data(mycalc,stdout)
      call profiler%init(1)
      call profiler%start(1)

      !>--- Write configs and send OPT tasks to pool workers
      ex = directory_exist('OPTFILES')
      if (.not.ex) exitstat = makedir('OPTFILES')

      do i = 1, n_workers
        idx_start = (i-1)*chunk_size + 1
        idx_end = min(i*chunk_size, nall)
        n_chunk = idx_end - idx_start + 1

        worker_calc = mycalc
        do j = 1, worker_calc%ncalculations
          if (allocated(worker_calc%calcs(j)%calcspace)) then
            ex = directory_exist(worker_calc%calcs(j)%calcspace)
            if (.not.ex) exitstat = makedir(trim(worker_calc%calcs(j)%calcspace))
            write(atmp,'(a,"_wopt",i0)') sep, i
            worker_calc%calcs(j)%calcspace = &
              mycalc%calcs(j)%calcspace // trim(atmp)
          end if
          worker_calc%calcs(j)%pymlip_handle = c_null_ptr
          worker_calc%calcs(j)%libtorch_handle = c_null_ptr
          worker_calc%calcs(j)%socket_handle = c_null_ptr
        end do

        write(config_file,'(a,a,i0,a)') 'OPTFILES', sep, i, '.bin'
        call write_worker_opt_config(trim(config_file), nat, n_chunk, at, &
                                      xyz(:,:,idx_start:idx_end), worker_calc, i)

        call pool_send_task_f(i-1, POOL_TASK_OPT, trim(config_file), io)
        if (io /= 0) then
          write(stdout,'(a,i0)') '**WARNING** Failed to send OPT task to worker ', i
        end if
      end do

      !>--- Wait for all workers and collect results
      eread(:) = 0.0_wp
      c = 0
      n_failed = 0
      do i = 1, n_workers
        idx_start = (i-1)*chunk_size + 1
        idx_end = min(i*chunk_size, nall)
        n_chunk = idx_end - idx_start + 1

        call pool_recv_result_f(i-1, exitstat, outfile, io)

        if (io /= 0 .or. exitstat /= 0) then
          write(stdout,'(a,i0,a)') '**WARNING** OPT task ', i, ' failed'
          eread(idx_start:idx_end) = 1.0_wp
          n_failed = n_failed + 1
          cycle
        end if

        !>--- Read results from worker output file
        allocate(e_chunk(n_chunk), opt_stat_chunk(n_chunk))
        allocate(xyz_chunk(3, nat, n_chunk))
        call read_worker_opt_results(trim(outfile), nat, n_chunk, &
                                      xyz_chunk, e_chunk, opt_stat_chunk, io)
        if (io == 0) then
          do j = 1, n_chunk
            k = idx_start + j - 1
            eread(k) = e_chunk(j)
            xyz(:,:,k) = xyz_chunk(:,:,j)
            if (opt_stat_chunk(j) == 0) c = c + 1
          end do
        else
          write(stdout,'(a,i0)') '**WARNING** Failed to read results from worker ', i
          eread(idx_start:idx_end) = 1.0_wp
          n_failed = n_failed + 1
        end if
        deallocate(e_chunk, opt_stat_chunk, xyz_chunk)
      end do

      call profiler%stop(1)

      percent = float(c)/float(nall)*100.0_wp
      write (atmp,'(f5.1,a)') percent,'% success)'
      write (stdout,'(">",1x,i0,a,i0,a,a)') c,' of ',nall, &
        ' structures successfully optimized (', trim(adjustl(atmp))
      write (atmp,'(">",1x,a,i0,a)') 'Total runtime for ',nall, &
        ' optimizations:'
      call profiler%write_timing(stdout,1,trim(atmp),.true.)

      if (dump) then
        open (newunit=ich,file=ensemblefile)
        open (newunit=ich2,file=ensembleelog)
        do i = 1,nall
          if (eread(i) < 0.5_wp .and. eread(i) /= 0.0_wp) then
            write (ich,'(2x,i0)') nat
            write (ich,'(2x,f25.15)') eread(i)
            do j = 1,nat
              write (ich,'(a2,3F24.14)') i2e(at(j),'nc'), &
                xyz(1,j,i),xyz(2,j,i),xyz(3,j,i)
            end do
          end if
        end do
        close (ich)
        close (ich2)
      end if

      call profiler%clear()
      return

    end if

!==========================================================================!
!> LEGACY SPAWN/WAIT OPTIMIZATION (fallback)
!==========================================================================!
300 continue

    write(stdout,'(/,1x,a)') 'Using process-based parallelism for MLIP optimization'

    n_workers = min(T, nall)
    chunk_size = (nall + n_workers - 1) / n_workers  ! ceiling division
    n_workers = (nall + chunk_size - 1) / chunk_size  ! actual workers needed
    allocate(pids(n_workers), source=0)
    allocate(exit_codes(n_workers), source=0)

    write(stdout,'(1x,a,i0,a,i0,a,i0,a)') &
      'Splitting ', nall, ' structures into ', n_workers, &
      ' chunks (~', chunk_size, ' each)'

    !>--- print optimization settings
    call print_opt_data(mycalc,stdout)

    call profiler%init(1)
    call profiler%start(1)

    !>--- Write configs and spawn all workers
    n_failed = 0
    do i = 1, n_workers
      idx_start = (i-1)*chunk_size + 1
      idx_end = min(i*chunk_size, nall)
      n_chunk = idx_end - idx_start + 1

      !>--- Prepare per-worker calculator with unique calcspace
      worker_calc = mycalc
      do j = 1, worker_calc%ncalculations
        if (allocated(worker_calc%calcs(j)%calcspace)) then
          ex = directory_exist(worker_calc%calcs(j)%calcspace)
          if (.not.ex) exitstat = makedir(trim(worker_calc%calcs(j)%calcspace))
          write(atmp,'(a,"_wopt",i0)') sep, i
          worker_calc%calcs(j)%calcspace = &
            mycalc%calcs(j)%calcspace // trim(atmp)
        end if
        worker_calc%calcs(j)%pymlip_handle = c_null_ptr
        worker_calc%calcs(j)%libtorch_handle = c_null_ptr
        worker_calc%calcs(j)%socket_handle = c_null_ptr
      end do

      !>--- Write binary config
      write(config_file,'(a,a,i0,a)') 'OPTFILES', sep, i, '.bin'
      ex = directory_exist('OPTFILES')
      if (.not.ex) exitstat = makedir('OPTFILES')
      call write_worker_opt_config(trim(config_file), nat, n_chunk, at, &
                                    xyz(:,:,idx_start:idx_end), worker_calc, i)

      !>--- Spawn worker
      write(cmd_str,'(a,a,a,a,i0)') &
        trim(progname), ' --worker-opt ', trim(config_file), ' ', i
      pids(i) = int(c_spawn_process(trim(cmd_str) // c_null_char))
      if (pids(i) <= 0) then
        write(stdout,'(a,i0)') '**WARNING** Failed to spawn opt worker ', i
        n_failed = n_failed + 1
      end if
    end do

    !>--- Wait for all workers and collect results
    eread(:) = 0.0_wp
    c = 0
    do i = 1, n_workers
      if (pids(i) <= 0) cycle
      exit_codes(i) = int(c_wait_for_process(int(pids(i), c_int)))

      idx_start = (i-1)*chunk_size + 1
      idx_end = min(i*chunk_size, nall)
      n_chunk = idx_end - idx_start + 1

      if (exit_codes(i) /= 0) then
        write(stdout,'(a,i0,a,i0)') &
          '**WARNING** Opt worker ', i, ' exited with status ', exit_codes(i)
        n_failed = n_failed + 1
        eread(idx_start:idx_end) = 1.0_wp
        cycle
      end if

      !>--- Read results from worker output
      write(outfile,'(a,a,i0,a)') 'OPTFILES', sep, i, '.bin.out'
      allocate(e_chunk(n_chunk), opt_stat_chunk(n_chunk))
      allocate(xyz_chunk(3, nat, n_chunk))
      call read_worker_opt_results(trim(outfile), nat, n_chunk, &
                                    xyz_chunk, e_chunk, opt_stat_chunk, io)
      if (io == 0) then
        do j = 1, n_chunk
          k = idx_start + j - 1
          eread(k) = e_chunk(j)
          xyz(:,:,k) = xyz_chunk(:,:,j)
          if (opt_stat_chunk(j) == 0) c = c + 1
        end do
      else
        write(stdout,'(a,i0)') '**WARNING** Failed to read results from opt worker ', i
        eread(idx_start:idx_end) = 1.0_wp
        n_failed = n_failed + 1
      end if
      deallocate(e_chunk, opt_stat_chunk, xyz_chunk)
    end do

    call profiler%stop(1)

    !>--- Summary
    if (n_failed > 0) then
      write(stdout,'(/,a,i0,a)') '**WARNING** ', n_failed, ' opt worker(s) had issues'
    end if
    percent = float(c)/float(nall)*100.0_wp
    write (atmp,'(f5.1,a)') percent,'% success)'
    write (stdout,'(">",1x,i0,a,i0,a,a)') c,' of ',nall, &
      ' structures successfully optimized (', trim(adjustl(atmp))
    write (atmp,'(">",1x,a,i0,a)') 'Total runtime for ',nall,' optimizations:'
    call profiler%write_timing(stdout,1,trim(atmp),.true.)
    runtime = profiler%get(1)
    write (atmp,'(f16.3,a)') runtime/real(nall,wp),' sec'
    write (stdout,'(a,a,a)') '> Corresponding to approximately ', &
      trim(adjustl(atmp)), ' per processed structure'
    call profiler%clear()

    !>--- Dump ensemble file if requested
    if (dump) then
      open(newunit=ich, file=ensemblefile)
      open(newunit=ich2, file=ensembleelog)
      do i = 1, nall
        if (eread(i) < 0.5_wp) then  ! valid energy (failed = 1.0)
          block
            type(coord) :: tmpmol
            allocate(tmpmol%at(nat), tmpmol%xyz(3,nat))
            tmpmol%nat = nat; tmpmol%at = at; tmpmol%xyz = xyz(:,:,i)
            write(atmp,'(1x,"Etot=",f16.10)') eread(i)
            tmpmol%comment = trim(atmp)
            call tmpmol%append(ich)
            deallocate(tmpmol%at, tmpmol%xyz)
          end block
        end if
      end do
      close(ich)
      close(ich2)
    end if

    deallocate(pids, exit_codes)
    return
  end if

!==========================================================================!
!> OPENMP THREAD-PARALLEL PATH (non-MLIP or single thread fallback)
!==========================================================================!

!>--- prepare objects for parallelization
  allocate (calculations(T),source=mycalc)
  allocate (mols(T),molsnew(T))
  do i = 1,T
    do j = 1,mycalc%ncalculations
      calculations(i)%calcs(j) = mycalc%calcs(j)
      !>--- directories and io preparation
      ex = directory_exist(mycalc%calcs(j)%calcspace)
      if (.not.ex) then
        io = makedir(trim(mycalc%calcs(j)%calcspace))
      end if
      if(calculations(i)%calcs(j)%id == jobtype%tblite)then
         calculations(i)%optnewinit=.true.
      endif
      write (atmp,'(a,"_",i0)') sep,i
      calculations(i)%calcs(j)%calcspace = mycalc%calcs(j)%calcspace//trim(atmp)
      if(allocated(calculations(i)%calcs(j)%calcfile)) deallocate(calculations(i)%calcs(j)%calcfile)
      if(allocated(calculations(i)%calcs(j)%systemcall)) deallocate(calculations(i)%calcs(j)%systemcall)
      !>--- libtorch: reset counters (shared model handle set after loop)
      if (calculations(i)%calcs(j)%id == jobtype%libtorch) then
        calculations(i)%calcs(j)%libtorch_handle = c_null_ptr
        calculations(i)%calcs(j)%libtorch_call_count = 0
        calculations(i)%calcs(j)%libtorch_total_time = 0.0d0
      end if
      !>--- pymlip: reset handle (shared model handle set after loop)
      if (calculations(i)%calcs(j)%id == jobtype%pymlip) then
        calculations(i)%calcs(j)%pymlip_handle = c_null_ptr
        calculations(i)%calcs(j)%pymlip_call_count = 0
        calculations(i)%calcs(j)%pymlip_total_time = 0.0d0
      end if
      call calculations(i)%calcs(j)%printid(i,j)
    end do
    calculations(i)%pr_energies = .false.
    allocate (mols(i)%at(nat),mols(i)%xyz(3,nat))
    allocate (molsnew(i)%at(nat),molsnew(i)%xyz(3,nat))
  end do

!>--- libtorch: load shared model ONCE, propagate handle to all threads
  do j = 1,mycalc%ncalculations
    if (mycalc%calcs(j)%id == jobtype%libtorch) then
      call libtorch_init_shared(mycalc%calcs(j), io)
      if (io /= 0) then
        write(stdout,'(a)') '**ERROR** Failed to load shared libtorch model'
        return
      end if
      do i = 1,T
        calculations(i)%calcs(j)%libtorch_handle = mycalc%calcs(j)%libtorch_handle
      end do
    end if
  end do

!>--- pymlip: load shared model ONCE, propagate handle to all threads
  do j = 1,mycalc%ncalculations
    if (mycalc%calcs(j)%id == jobtype%pymlip) then
      call pymlip_init(mycalc%calcs(j), io)
      if (io /= 0) then
        write(stdout,'(a)') '**ERROR** Failed to load shared pymlip model'
        return
      end if
      do i = 1,T
        calculations(i)%calcs(j)%pymlip_handle = mycalc%calcs(j)%pymlip_handle
      end do
    end if
  end do

!>--- Warn if MLIP GPU is using serialized OpenMP path
  if (T > 1) then
    do j = 1,mycalc%ncalculations
      if (mycalc%calcs(j)%id == jobtype%libtorch .and. &
          mycalc%calcs(j)%libtorch_device_id > 0) then
        write(stdout,'(a)') ' [libtorch] WARNING: GPU optimization using '// &
          'per-thread OpenMP path (forward_mutex serializes all passes). '// &
          'This is SLOWER than single-threaded. Set threads=1 to avoid overhead.'
        exit
      end if
      if (mycalc%calcs(j)%id == jobtype%pymlip) then
        if (allocated(mycalc%calcs(j)%pymlip_device)) then
          if (mycalc%calcs(j)%pymlip_device /= 'cpu') then
            write(stdout,'(a)') ' [pymlip] WARNING: GPU optimization using '// &
              'per-thread OpenMP path (GIL serializes all passes). '// &
              'This is SLOWER than single-threaded. Set threads=1 to avoid overhead.'
            exit
          end if
        end if
      end if
    end do
  end if

!>--- printout directions and timer initialization
  pr = .false. !> stdout printout
  wr = .false. !> write crestopt.log
  if (dump) then
    open (newunit=ich,file=ensemblefile)
    open (newunit=ich2,file=ensembleelog)
  end if
  call profiler%init(1)
  call profiler%start(1)

!>--- first progress printout (initializes progress variables)
  call crest_oloop_pr_progress(env,nall,0)

!>--- shared variables
  allocate (grads(3,nat,T),source=0.0_wp)
  c = 0  !> counter of successfull optimizations
  k = 0  !> counter of total optimization (fail+success)
  z = 0  !> counter to perform optimization in right order (1...nall)
  eread(:) = 0.0_wp
  grads(:,:,:) = 0.0_wp
!>--- loop over ensemble
  !$omp parallel &
  !$omp shared(env,calculations,nat,nall,at,xyz,eread,grads,c,k,z,pr,wr,dump) &
  !$omp shared(ich,ich2,mols,molsnew, nested,Tn)
  !$omp single
  do i = 1,nall

    call initsignal()
    vz = i
    !$omp task firstprivate( vz ) private(j,job,energy,io,atmp,gnorm,thread_id,zcopy)
    call initsignal()

    !>--- OpenMP nested region threads
    if (nested) call ompmklset(Tn)

    thread_id = OMP_GET_THREAD_NUM()
    job = thread_id+1
    !>--- modify calculation spaces
    !$omp critical
    z = z+1
    zcopy = z
    mols(job)%nat = nat
    mols(job)%at(:) = at(:)
    mols(job)%xyz(:,:) = xyz(:,:,z)

    molsnew(job)%nat = nat
    molsnew(job)%at(:) = at(:)
    molsnew(job)%xyz(:,:) = xyz(:,:,z)
    !$omp end critical

    !>-- geometry optimization
    call optimize_geometry(mols(job),molsnew(job),calculations(job),energy,grads(:,:,job),pr,wr,io)

    !$omp critical
    if (io == 0) then
      !>--- successful optimization (io==0)
      c = c+1
      if (dump) then
        gnorm = norm2(grads(:,:,job))
        write (atmp,'(1x,"Etot=",f16.10,1x,"g norm=",f12.8)') energy,gnorm
        molsnew(job)%comment = trim(atmp)
        call molsnew(job)%append(ich)
        call calc_eprint(calculations(job),energy,calculations(job)%etmp,gnorm,ich2)
      end if
      eread(zcopy) = energy
      xyz(:,:,zcopy) = molsnew(job)%xyz(:,:)
    else if(io==calculations(job)%maxcycle .and. calculations(job)%anopt) then
      !>--- allow partial optimization?
      c = c+1
      eread(zcopy) = energy
      xyz(:,:,zcopy) = molsnew(job)%xyz(:,:)
    else
      eread(zcopy) = 1.0_wp
    end if
    k = k+1
    !>--- print progress
    call crest_oloop_pr_progress(env,nall,k)
    !$omp end critical
    !$omp end task
  end do
  !$omp taskwait
  !$omp end single
  !$omp end parallel

!>--- finalize progress printout
  call crest_oloop_pr_progress(env,nall,-1)

!>--- stop timer
  call profiler%stop(1)

!>--- prepare some summary printout
  percent = float(c)/float(nall)*100.0_wp
  write (atmp,'(f5.1,a)') percent,'% success)'
  write (stdout,'(">",1x,i0,a,i0,a,a)') c,' of ',nall,' structures successfully optimized (', &
  &     trim(adjustl(atmp))
  write (atmp,'(">",1x,a,i0,a)') 'Total runtime for ',nall,' optimizations:'
  call profiler%write_timing(stdout,1,trim(atmp),.true.)
  runtime = profiler%get(1)
  write (atmp,'(f16.3,a)') runtime/real(nall,wp),' sec'
  write (stdout,'(a,a,a)') '> Corresponding to approximately ',trim(adjustl(atmp)), &
  &                       ' per processed structure'

!>--- close files (if they are open)
  if (dump) then
    close (ich)
    close (ich2)
  end if

  deallocate (grads)
  call profiler%clear()
  !>--- Nullify per-thread shared handles, then cleanup stats
  do i = 1,T
    do j = 1,mycalc%ncalculations
      if (calculations(i)%calcs(j)%id == jobtype%libtorch) &
        calculations(i)%calcs(j)%libtorch_handle = c_null_ptr
      if (calculations(i)%calcs(j)%id == jobtype%pymlip) &
        calculations(i)%calcs(j)%pymlip_handle = c_null_ptr
      if (calculations(i)%calcs(j)%id == jobtype%ase_socket) &
        calculations(i)%calcs(j)%socket_handle = c_null_ptr
    end do
    call mlip_cleanup_all(calculations(i))
  end do
  !>--- Free shared models (skipped if mlip_keep_loaded for persistence)
  if (.not. mycalc%mlip_keep_loaded) then
    call libtorch_shared_cleanup()
    do j = 1,mycalc%ncalculations
      if (mycalc%calcs(j)%id == jobtype%libtorch) &
        mycalc%calcs(j)%libtorch_handle = c_null_ptr
    end do
  end if
  call mlip_cleanup_all(mycalc)
  deallocate (calculations)
  if (allocated(mols)) deallocate (mols)
  if (allocated(molsnew)) deallocate (molsnew)
  return
end subroutine crest_oloop

!========================================================================================!
subroutine crest_oloop_pr_progress(env,total,current)
!*********************************************
!* subroutine crest_oloop_pr_progress
!* A subroutine to print and track progress of
!* concurrent geometry optimizations
!*********************************************
  use crest_parameters,only:wp,stdout
  use crest_data
  use iomod,only:to_str
  implicit none
  type(systemdata),intent(inout) :: env
  integer,intent(in) :: total,current
  real(wp) :: percent
  character(len=5) :: atmp
  real(wp),save :: increment
  real(wp),save :: progressbarrier

  percent = float(current)/float(total)*100.0_wp
  if (current == 0) then !> as a wrapper to start the printout
    progressbarrier = 0.0_wp
    if (env%niceprint) then
      percent = 0.0_wp
      call printprogbar(percent)
    end if
    increment = 10.0_wp
    if (total > 1000) increment = 7.5_wp
    if (total > 5000) increment = 5.0_wp
    if (total > 10000) increment = 2.5_wp
    if (total > 20000) increment = 1.0_wp

  else if (current <= total.and.current > 0) then !> the regular printout case
    if (env%niceprint) then
      call printprogbar(percent)

    else if (.not.env%legacy) then
      if (percent >= progressbarrier) then
        write (atmp,'(f5.1)') percent
        write (stdout,'(1x,a)',advance='no') '|>'//trim(adjustl(atmp))//'%'
        progressbarrier = progressbarrier+increment
        progressbarrier = min(progressbarrier,100.0_wp)
        flush (stdout)
      end if
    else
      write (stdout,'(1x,i0)',advance='no') current
      flush (stdout)
    end if

  else !> as a wrapper to finalize the printout
    if (.not.env%niceprint) then
      write (stdout,'(/,1x,a)') 'done.'
    else
      write (stdout,*)
    end if
  end if

end subroutine crest_oloop_pr_progress

!========================================================================================!
!========================================================================================!
!> Routines for parallel MDs
!========================================================================================!
!========================================================================================!

subroutine crest_search_multimd(env,mol,mddats,nsim)
!*****************************************************
!* subroutine crest_search_multimd
!* this runs #nsim MDs on the same structure (mol)
!*
!* For pymlip calculators: uses process-based parallelism
!* (separate CREST worker processes) to bypass Python GIL.
!* For all others: uses OpenMP thread parallelism.
!*****************************************************
  use crest_parameters,only:wp,stdout,sep
  use iso_c_binding,only:c_null_ptr,c_int,c_char,c_null_char
  use crest_data
  use crest_calculator
  use strucrd
  use dynamics_module
  use iomod,only:makedir,directory_exist,remove
  use omp_lib
  use crest_restartlog,only:trackrestart,restart_write_dummy
  use worker_io_module,only:write_worker_config
  use worker_pool_module
  implicit none
  type(systemdata),intent(inout) :: env
  type(mddata) :: mddats(nsim)
  integer :: nsim
  type(coord) :: mol
  type(coord),allocatable :: moltmps(:)
  integer :: i,j,io,ich
  logical :: pr,ex,nested
  integer :: T,Tn
  real(wp) :: percent
  character(len=80) :: atmp
  character(len=*),parameter :: mdir = 'MDFILES'

  type(calcdata),allocatable :: calculations(:)
  integer :: vz,job,thread_id
  real(wp) :: etmp
  real(wp),allocatable :: grdtmp(:,:)
  type(timer) :: profiler

  !>--- process-based parallelism variables
  logical :: use_process_parallel
  integer :: n_workers, batch_start, batch_end, n_batch
  integer,allocatable :: pids(:), exit_codes(:)
  integer :: pid, exitstat, n_failed
  character(len=512) :: progname, config_file, cmd_str
  type(calcdata) :: worker_calc

  !>--- C interface for process bridge
  interface
    function c_spawn_process(cmd) bind(C, name='spawn_process')
      import :: c_int, c_char
      character(kind=c_char), intent(in) :: cmd(*)
      integer(c_int) :: c_spawn_process
    end function
    function c_wait_for_process(pid) bind(C, name='wait_for_process')
      import :: c_int
      integer(c_int), value, intent(in) :: pid
      integer(c_int) :: c_wait_for_process
    end function
  end interface

!===========================================================!
!>--- decide wether to skip this call
  if (trackrestart(env)) then
    call restart_write_dummy('crest_dynamics.trj')
    return
  end if

!>--- check if we have any MD & calculation settings allocated
  if (.not.env%mddat%requested) then
    write (stdout,*) 'MD requested, but no MD settings present.'
    return
  else if (env%calc%ncalculations < 1) then
    write (stdout,*) 'MD requested, but no calculation settings present.'
    return
  end if

!>--- detect if MLIP GPU is used (needs process-based parallelism)
  use_process_parallel = mlip_needs_process_parallel(env%calc)

!>--- determine thread/worker count
  call new_ompautoset(env,'auto_nested',nsim,T,Tn)
  nested = env%omp_allow_nested

!==========================================================================!
!> PERSISTENT WORKER POOL PATH (pymlip/libtorch GPU)
!> Workers load the model once and accept multiple MD/OPT tasks.
!==========================================================================!
  if (use_process_parallel .and. T > 1) then

    !>--- Create pool if not already active (first use in this run)
    call get_command_argument(0, progname)
    if (.not. pool_is_active_f()) then
      n_workers = min(T, nsim)
      write(stdout,'(/,1x,a,i0,a)') &
        'Creating persistent worker pool with ', n_workers, ' workers'
      call pool_create_f(n_workers, trim(progname), io)
      if (io /= 0) then
        write(stdout,'(a)') '  Pool creation failed, falling back to spawn/wait'
        goto 200  !> fall through to legacy spawn/wait path
      end if
    end if

    if (pool_is_active_f()) then
      !> ============================================================
      !> POOL MD PATH: send tasks over pipes to persistent workers
      !> ============================================================
      n_workers = pool_get_n_workers_f()

      write(stdout,'(1x,a,i0,a)') &
        'Using persistent worker pool for ', nsim, ' MTDs'

      !>--- Send MD tasks in batches of n_workers
      n_failed = 0
      do batch_start = 1, nsim, n_workers
        batch_end = min(batch_start + n_workers - 1, nsim)
        n_batch = batch_end - batch_start + 1

        !>--- Write config files and send tasks to workers
        do i = batch_start, batch_end
          j = i - batch_start + 1  ! 1-based local index

          !>--- prepare per-worker calculator with unique calcspace
          worker_calc = env%calc
          do io = 1, worker_calc%ncalculations
            if (allocated(worker_calc%calcs(io)%calcspace)) then
              ex = directory_exist(worker_calc%calcs(io)%calcspace)
              if (.not.ex) exitstat = makedir(trim(worker_calc%calcs(io)%calcspace))
              write(atmp,'(a,"_w",i0)') sep, i
              worker_calc%calcs(io)%calcspace = &
                env%calc%calcs(io)%calcspace // trim(atmp)
            end if
            worker_calc%calcs(io)%pymlip_handle = c_null_ptr
            worker_calc%calcs(io)%libtorch_handle = c_null_ptr
            worker_calc%calcs(io)%socket_handle = c_null_ptr
          end do

          write(config_file,'(a,a,a,i0,a)') mdir, sep, '.worker_', i, '.bin'
          call write_worker_config(trim(config_file), mol, mddats(i), &
                                   worker_calc, i)

          call pool_send_task_f(j-1, POOL_TASK_MD, trim(config_file), io)
          if (io /= 0) then
            write(stdout,'(a,i0)') '**WARNING** Failed to send MD task ', i
            n_failed = n_failed + 1
          end if
        end do

        !>--- Wait for all workers in batch to finish
        do i = batch_start, batch_end
          j = i - batch_start + 1
          call pool_recv_result_f(j-1, exitstat, config_file, io)
          if (io /= 0 .or. exitstat /= 0) then
            write(stdout,'(a,i0,a)') '**WARNING** MD task ', i, ' failed'
            n_failed = n_failed + 1
          else
            write(stdout,'(1x,a,i0,a)') 'MTD ', i, ' completed successfully'
          end if
        end do
      end do

      if (n_failed > 0) then
        write(stdout,'(/,a,i0,a)') '**WARNING** ', n_failed, ' MD task(s) failed'
      end if

      !>--- Collect trajectories (same as legacy path)
      call collect(nsim, mddats)
      return
    end if

!==========================================================================!
!> LEGACY SPAWN/WAIT PATH (fallback if pool creation failed)
!==========================================================================!
200 continue

    write(stdout,'(/,1x,a)') 'Using process-based parallelism for MLIP (bypassing GIL/mutex)'
    write(stdout,'(1x,a,i0,a,i0,a)') 'Spawning up to ', T, ' worker processes for ', nsim, ' MTDs'

    !>--- prepare per-worker calc with unique working directories
    n_workers = min(T, nsim)
    allocate(pids(n_workers), source=0)
    allocate(exit_codes(n_workers), source=0)

    !>--- spawn workers in batches of n_workers
    n_failed = 0
    do batch_start = 1, nsim, n_workers
      batch_end = min(batch_start + n_workers - 1, nsim)
      n_batch = batch_end - batch_start + 1

      write(stdout,'(1x,a,i0,a,i0)') 'Launching batch: MTD ', batch_start, ' to ', batch_end

      !>--- write config files and spawn worker processes
      do i = batch_start, batch_end
        j = i - batch_start + 1  ! local index within batch

        !>--- prepare per-worker calculator with unique calcspace
        worker_calc = env%calc
        do io = 1, worker_calc%ncalculations
          if (allocated(worker_calc%calcs(io)%calcspace)) then
            ex = directory_exist(worker_calc%calcs(io)%calcspace)
            if (.not.ex) then
              exitstat = makedir(trim(worker_calc%calcs(io)%calcspace))
            end if
            write(atmp,'(a,"_w",i0)') sep, i
            worker_calc%calcs(io)%calcspace = &
              env%calc%calcs(io)%calcspace // trim(atmp)
          end if
          !>--- null out all handles (worker will init its own)
          worker_calc%calcs(io)%pymlip_handle = c_null_ptr
          worker_calc%calcs(io)%libtorch_handle = c_null_ptr
          worker_calc%calcs(io)%socket_handle = c_null_ptr
        end do

        !>--- write binary config file
        write(config_file,'(a,a,a,i0,a)') mdir, sep, '.worker_', i, '.bin'
        call write_worker_config(trim(config_file), mol, mddats(i), &
                                 worker_calc, i)

        !>--- build command: crest --worker <config_file> <index>
        write(cmd_str,'(a,a,a,a,i0)') &
          trim(progname), ' --worker ', trim(config_file), ' ', i

        !>--- spawn the worker process
        pids(j) = int(c_spawn_process(trim(cmd_str) // c_null_char))
        if (pids(j) <= 0) then
          write(stdout,'(a,i0)') '**WARNING** Failed to spawn worker for MTD ', i
          n_failed = n_failed + 1
        else
          write(stdout,'(1x,a,i0,a,i0)') 'Worker for MTD ', i, ' spawned (PID ', pids(j), ')'
        end if
      end do

      !>--- wait for all workers in this batch to finish
      do i = batch_start, batch_end
        j = i - batch_start + 1
        if (pids(j) > 0) then
          exit_codes(j) = int(c_wait_for_process(int(pids(j), c_int)))
          if (exit_codes(j) /= 0) then
            write(stdout,'(a,i0,a,i0)') &
              '**WARNING** Worker for MTD ', i, ' exited with status ', exit_codes(j)
            n_failed = n_failed + 1
          else
            write(stdout,'(1x,a,i0,a)') 'Worker for MTD ', i, ' completed successfully'
          end if
        end if
      end do
    end do

    if (n_failed > 0) then
      write(stdout,'(/,a,i0,a)') '**WARNING** ', n_failed, ' worker(s) failed'
    end if
    write(stdout,'(1x,a)') 'All worker processes finished.'

    deallocate(pids, exit_codes)

    !>--- collect trajectories into one (same as OpenMP path)
    call collect(nsim, mddats)
    return

  end if

!==========================================================================!
!> OPENMP THREAD-PARALLEL PATH (libtorch, xtb, gfn-ff, etc.)
!==========================================================================!

!>--- prepare calculation containers for parallelization (one per thread)
  allocate (calculations(T),source=env%calc)
  allocate (moltmps(T),source=mol)
  allocate (grdtmp(3,mol%nat),source=0.0_wp)
  do i = 1,T
    moltmps(i)%nat = mol%nat
    moltmps(i)%at = mol%at
    moltmps(i)%xyz = mol%xyz
    do j = 1,env%calc%ncalculations
      calculations(i)%calcs(j) = env%calc%calcs(j)
      !>--- directories and io preparation
      ex = directory_exist(env%calc%calcs(j)%calcspace)
      if (.not.ex) then
        io = makedir(trim(env%calc%calcs(j)%calcspace))
      end if
      write (atmp,'(a,"_",i0)') sep,i
      calculations(i)%calcs(j)%calcspace = env%calc%calcs(j)%calcspace//trim(atmp)
      if(allocated(calculations(i)%calcs(j)%calcfile)) deallocate(calculations(i)%calcs(j)%calcfile)
      if(allocated(calculations(i)%calcs(j)%systemcall)) deallocate(calculations(i)%calcs(j)%systemcall)
      !>--- libtorch: reset counters (shared model handle set after loop)
      if (calculations(i)%calcs(j)%id == jobtype%libtorch) then
        calculations(i)%calcs(j)%libtorch_handle = c_null_ptr
        calculations(i)%calcs(j)%libtorch_call_count = 0
        calculations(i)%calcs(j)%libtorch_total_time = 0.0d0
      end if
      !>--- pymlip: reset handle (shared model handle set after loop)
      if (calculations(i)%calcs(j)%id == jobtype%pymlip) then
        calculations(i)%calcs(j)%pymlip_handle = c_null_ptr
        calculations(i)%calcs(j)%pymlip_call_count = 0
        calculations(i)%calcs(j)%pymlip_total_time = 0.0d0
      end if
      call calculations(i)%calcs(j)%printid(i,j)
    end do
    calculations(i)%pr_energies = .false.
  end do

!>--- libtorch: load shared model ONCE, propagate handle to all threads
  do j = 1,env%calc%ncalculations
    if (env%calc%calcs(j)%id == jobtype%libtorch) then
      call libtorch_init_shared(env%calc%calcs(j), io)
      if (io /= 0) then
        write(stdout,'(a)') '**ERROR** Failed to load shared libtorch model'
        return
      end if
      do i = 1,T
        calculations(i)%calcs(j)%libtorch_handle = env%calc%calcs(j)%libtorch_handle
      end do
    end if
  end do

!>--- pymlip: load shared model ONCE, propagate handle to all threads.
!>    Python GIL serializes all calls -- multiple handles don't help.
!>    True parallelism requires separate processes (not OpenMP threads).
!>    This path is only reached when T==1 (single thread fallback).
  do j = 1,env%calc%ncalculations
    if (env%calc%calcs(j)%id == jobtype%pymlip) then
      call pymlip_init(env%calc%calcs(j), io)
      if (io /= 0) then
        write(stdout,'(a)') '**ERROR** Failed to load shared pymlip model'
        return
      end if
      do i = 1,T
        calculations(i)%calcs(j)%pymlip_handle = env%calc%calcs(j)%pymlip_handle
      end do
    end if
  end do

!>--- Warn if MLIP GPU ended up in serialized OpenMP path
  if (T > 1) then
    do j = 1,env%calc%ncalculations
      if (env%calc%calcs(j)%id == jobtype%libtorch .and. &
          env%calc%calcs(j)%libtorch_device_id > 0) then
        write(stdout,'(a)') ' [libtorch] WARNING: GPU MD using per-thread '// &
          'OpenMP path (forward_mutex serializes all passes). '// &
          'This is SLOWER than single-threaded. Set threads=1 to avoid overhead.'
        exit
      end if
      if (env%calc%calcs(j)%id == jobtype%pymlip) then
        if (allocated(env%calc%calcs(j)%pymlip_device)) then
          if (env%calc%calcs(j)%pymlip_device /= 'cpu') then
            write(stdout,'(a)') ' [pymlip] WARNING: GPU MD using per-thread '// &
              'OpenMP path (GIL serializes all passes). '// &
              'This is SLOWER than single-threaded. Set threads=1 to avoid overhead.'
            exit
          end if
        end if
      end if
    end do
  end if

  !>--- initialize the calculations (one per thread, not per simulation)
  do i = 1,T
    call engrad(moltmps(i),calculations(i),etmp,grdtmp,io)
  end do

  !>--- other settings
  pr = .false.
  call profiler%init(nsim)

  !>--- run the MDs
  !$omp parallel &
  !$omp shared(env,calculations,mddats,mol,pr,percent,ich, nsim, moltmps, nested,Tn) &
  !!$omp single
  !$omp private(vz,i,job,thread_id,io,ex)
  !$omp do
  do i = 1,nsim

    call initsignal()
    vz = i

    !>--- OpenMP nested region threads
    if (nested) call ompmklset(Tn)

    !!$omp task firstprivate( vz ) private( job,thread_id,io,ex )
    call initsignal()

    thread_id = OMP_GET_THREAD_NUM()
    job = thread_id+1
    !$omp critical
    moltmps(job)%nat = mol%nat
    moltmps(job)%at = mol%at
    moltmps(job)%xyz = mol%xyz
    !$omp end critical
    !>--- startup printout (thread safe)
    call parallel_md_block_printout(mddats(vz),vz)

    !>--- the acutal MD call with timing
    call profiler%start(vz)
    call dynamics(moltmps(job),mddats(vz),calculations(job),pr,io)
    call profiler%stop(vz)

    !>--- finish printout (thread safe)
    call parallel_md_finish_printout(mddats(vz),vz,io,profiler)
    !!$omp end task
  end do
  !!$omp taskwait
  !$omp end parallel

  !>--- collect trajectories into one
  call collect(nsim,mddats)

  call profiler%clear()
  !>--- Nullify per-thread shared handles, then cleanup stats
  do i = 1,T
    do j = 1,env%calc%ncalculations
      if (calculations(i)%calcs(j)%id == jobtype%libtorch) &
        calculations(i)%calcs(j)%libtorch_handle = c_null_ptr
      if (calculations(i)%calcs(j)%id == jobtype%pymlip) &
        calculations(i)%calcs(j)%pymlip_handle = c_null_ptr
      if (calculations(i)%calcs(j)%id == jobtype%ase_socket) &
        calculations(i)%calcs(j)%socket_handle = c_null_ptr
    end do
    call mlip_cleanup_all(calculations(i))
  end do
  !>--- Free shared models (skipped if mlip_keep_loaded for persistence)
  if (.not. env%calc%mlip_keep_loaded) then
    call libtorch_shared_cleanup()
    do j = 1,env%calc%ncalculations
      if (env%calc%calcs(j)%id == jobtype%libtorch) &
        env%calc%calcs(j)%libtorch_handle = c_null_ptr
    end do
  end if
  call mlip_cleanup_all(env%calc)
  deallocate (calculations)
  if (allocated(moltmps)) deallocate (moltmps)
  return
contains
  subroutine collect(n,mddats)
    implicit none
    integer :: n
    type(mddata) :: mddats(n)
    logical :: ex
    integer :: i,io,ich,ich2
    character(len=:),allocatable :: atmp
    character(len=256) :: btmp
    open (newunit=ich,file='crest_dynamics.trj')
    do i = 1,n
      atmp = mddats(i)%trajectoryfile
      inquire (file=atmp,exist=ex)
      if (ex) then
        open (newunit=ich2,file=atmp)
        io = 0
        do while (io == 0)
          read (ich2,'(a)',iostat=io) btmp
          if (io == 0) then
            write (ich,'(a)') trim(btmp)
          end if
        end do
        close (ich2)
      end if
    end do
    close (ich)
    return
  end subroutine collect
end subroutine crest_search_multimd

!========================================================================================!
subroutine crest_search_multimd_init(env,mol,mddat,nsim)
!*******************************************************
!* subroutine crest_search_multimd_init
!* This routine will initialize a copy of env%mddat
!* and save it to the local mddat. If we are about to
!* run RMSD metadynamics, the required number of
!* simulations (#nsim) is returned
!*******************************************************
  use crest_parameters,only:wp,stdout
  use crest_data
  use crest_calculator
  use strucrd
  use dynamics_module
  use iomod,only:makedir,directory_exist,remove
  use omp_lib
  implicit none
  type(systemdata),intent(inout) :: env
  type(mddata) :: mddat
  type(coord) :: mol
  integer,intent(inout) :: nsim
  integer :: i,io
  logical :: pr
!=======================================================!
  type(calcdata),target :: calc
  type(shakedata) :: shk

  real(wp) :: energy
  real(wp),allocatable :: grad(:,:)
  character(len=*),parameter :: mdir = 'MDFILES'
!======================================================!

  !>--- check if we have any MD & calculation settings allocated
  mddat = env%mddat
  if (.not.mddat%requested) then
    write (stdout,*) 'MD requested, but no MD settings present.'
    return
  else if (env%calc%ncalculations < 1) then
    write (stdout,*) 'MD requested, but no calculation settings present.'
    return
  end if

  !>--- init SHAKE?
  if (mddat%shake) then
    if (allocated(env%ref%wbo)) then
      shk%wbo = env%ref%wbo
    else
      calc = env%calc
      calc%calcs(1)%rdwbo = .true.
      allocate (grad(3,mol%nat),source=0.0_wp)
      call engrad(mol,calc,energy,grad,io)
      deallocate (grad)
      calc%calcs(1)%rdwbo = .false.
      !>--- MLIP cleanup after WBO calculation
      call mlip_cleanup_all(calc)

      shk%shake_mode = env%mddat%shk%shake_mode
      if (allocated(calc%calcs(1)%wbo)) then
        call move_alloc(calc%calcs(1)%wbo,shk%wbo)
      end if
    end if

    if (calc%nfreeze > 0) then
      shk%freezeptr => calc%freezelist
    else
      nullify (shk%freezeptr)
    end if

    shk%shake_mode = env%shake
    mddat%shk = shk
    call init_shake(mol%nat,mol%at,mol%xyz,mddat%shk,pr)
    mddat%nshake = mddat%shk%ncons
  end if
  !>--- complete real-time settings to steps
  call mdautoset(mddat,io)

  !>--- (optional)  MTD initialization
  if (nsim < 0) then
    mddat%simtype = type_mtd  !>-- set runtype to MTD

    call defaultGF(env)
    write (stdout,*) 'list of applied metadynamics Vbias parameters:'
    do i = 1,env%nmetadyn
      write (stdout,'(''$metadyn '',f10.5,f8.3,i5)') env%metadfac(i),env%metadexp(i)
    end do
    write (stdout,*)

    !>--- how many simulations
    nsim = env%nmetadyn
  end if

  return
end subroutine crest_search_multimd_init

!========================================================================================!
subroutine crest_search_multimd_init2(env,mddats,nsim)
  use crest_parameters,only:wp,stdout,sep
  use crest_data
  use crest_calculator
  use strucrd
  use dynamics_module
  use iomod,only:makedir,directory_exist,remove
  use omp_lib
  implicit none
  type(systemdata),intent(inout) :: env
  type(mddata) :: mddats(nsim)
  integer :: nsim
  integer :: i,io,j
  logical :: ex
!========================================================!
  type(mtdpot),allocatable :: mtds(:)

  character(len=80) :: atmp
  character(len=*),parameter :: mdir = 'MDFILES'

  !>--- parallel MD setup
  ex = directory_exist(mdir)
  if (ex) then
    call rmrf(mdir)
  end if
  io = makedir(mdir)
  do i = 1,nsim
    mddats(i)%md_index = i
    write (atmp,'(a,i0,a)') 'crest_',i,'.trj'
    mddats(i)%trajectoryfile = mdir//sep//trim(atmp)
    write (atmp,'(a,i0,a)') 'crest_',i,'.mdrestart'
    mddats(i)%restartfile = mdir//sep//trim(atmp)
  end do

  allocate (mtds(nsim))
  do i = 1,nsim
    if (mddats(i)%simtype == type_mtd) then
      mtds(i)%kpush = env%metadfac(i)
      mtds(i)%alpha = env%metadexp(i)
      mtds(i)%cvdump_fs = float(env%mddump)
      mtds(i)%mtdtype = cv_rmsd

      mddats(i)%npot = 1
      allocate (mddats(i)%mtd(1),source=mtds(i))
      allocate (mddats(i)%cvtype(1),source=cv_rmsd)
      !> if necessary exclude atoms from RMSD bias
      if (sum(env%includeRMSD) /= env%ref%nat) then
        if (.not.allocated(mddats(i)%mtd(1)%atinclude)) &
        & allocate (mddats(i)%mtd(1)%atinclude(env%ref%nat),source=.true.)
        do j = 1,env%ref%nat
          if (env%includeRMSD(j) .ne. 1) mddats(i)%mtd(1)%atinclude(j) = .false.
        end do
      end if
    end if
  end do
  if (allocated(mtds)) deallocate (mtds)

  return
end subroutine crest_search_multimd_init2

!========================================================================================!
subroutine crest_search_multimd2(env,mols,mddats,nsim)
!*******************************************************************
!* subroutine crest_search_multimd2
!* this runs #nsim MDs on #nsim selected different structures (mols)
!*******************************************************************
  use crest_parameters,only:wp,stdout,sep
  use iso_c_binding,only:c_null_ptr
  use crest_data
  use crest_calculator
  use strucrd
  use dynamics_module
  use shake_module
  use iomod,only:makedir,directory_exist,remove
  use omp_lib
  use crest_restartlog,only:trackrestart,restart_write_dummy
  implicit none
  !> INPUT
  type(systemdata),intent(inout) :: env
  type(mddata) :: mddats(nsim)
  integer :: nsim
  type(coord) :: mols(nsim)
  type(coord),allocatable :: moltmps(:)
  integer :: i,j,io,ich
  logical :: pr,ex,nested
  integer :: T,Tn
  real(wp) :: percent
  character(len=80) :: atmp
  character(len=*),parameter :: mdir = 'MDFILES'

  type(calcdata),allocatable :: calculations(:)
  integer :: vz,job,thread_id
  type(timer) :: profiler
!===========================================================!
!>--- decide wether to skip this call
  if (trackrestart(env)) then
    call restart_write_dummy('crest_dynamics.trj')
    return
  end if

!>--- check if we have any MD & calculation settings allocated
  if (.not.env%mddat%requested) then
    write (stdout,*) 'MD requested, but no MD settings present.'
    return
  else if (env%calc%ncalculations < 1) then
    write (stdout,*) 'MD requested, but no calculation settings present.'
    return
  end if

!>--- prepare calculation objects for parallelization (one per thread)
  call new_ompautoset(env,'auto_nested',nsim,T,Tn)
  nested = env%omp_allow_nested

  allocate (calculations(T),source=env%calc)
  allocate (moltmps(T),source=mols(1))
  do i = 1,T
    do j = 1,env%calc%ncalculations
      calculations(i)%calcs(j) = env%calc%calcs(j)
      !>--- directories and io preparation
      ex = directory_exist(env%calc%calcs(j)%calcspace)
      if (.not.ex) then
        io = makedir(trim(env%calc%calcs(j)%calcspace))
      end if
      write (atmp,'(a,"_",i0)') sep,i
      calculations(i)%calcs(j)%calcspace = env%calc%calcs(j)%calcspace//trim(atmp)
      if(allocated(calculations(i)%calcs(j)%calcfile)) deallocate(calculations(i)%calcs(j)%calcfile)
      if(allocated(calculations(i)%calcs(j)%systemcall)) deallocate(calculations(i)%calcs(j)%systemcall)
      !>--- libtorch: reset counters (shared model handle set after loop)
      if (calculations(i)%calcs(j)%id == jobtype%libtorch) then
        calculations(i)%calcs(j)%libtorch_handle = c_null_ptr
        calculations(i)%calcs(j)%libtorch_call_count = 0
        calculations(i)%calcs(j)%libtorch_total_time = 0.0d0
      end if
      !>--- pymlip: reset handle (shared model handle set after loop)
      if (calculations(i)%calcs(j)%id == jobtype%pymlip) then
        calculations(i)%calcs(j)%pymlip_handle = c_null_ptr
        calculations(i)%calcs(j)%pymlip_call_count = 0
        calculations(i)%calcs(j)%pymlip_total_time = 0.0d0
      end if
      call calculations(i)%calcs(j)%printid(i,j)
    end do
    calculations(i)%pr_energies = .false.
  end do

!>--- libtorch: load shared model ONCE, propagate handle to all threads
  do j = 1,env%calc%ncalculations
    if (env%calc%calcs(j)%id == jobtype%libtorch) then
      call libtorch_init_shared(env%calc%calcs(j), io)
      if (io /= 0) then
        write(stdout,'(a)') '**ERROR** Failed to load shared libtorch model'
        return
      end if
      do i = 1,T
        calculations(i)%calcs(j)%libtorch_handle = env%calc%calcs(j)%libtorch_handle
      end do
    end if
  end do

!>--- pymlip: load shared model ONCE, propagate handle to all threads
  do j = 1,env%calc%ncalculations
    if (env%calc%calcs(j)%id == jobtype%pymlip) then
      call pymlip_init(env%calc%calcs(j), io)
      if (io /= 0) then
        write(stdout,'(a)') '**ERROR** Failed to load shared pymlip model'
        return
      end if
      do i = 1,T
        calculations(i)%calcs(j)%pymlip_handle = env%calc%calcs(j)%pymlip_handle
      end do
    end if
  end do

!>--- other settings
  pr = .false.
  call profiler%init(nsim)

!>--- run the MDs
  !$omp parallel &
  !$omp shared(env,calculations,mddats,mols,pr,percent,ich, moltmps,profiler, nested,Tn)
  !$omp single
  do i = 1,nsim

    call initsignal()
    vz = i

    !>--- OpenMP nested region threads
    if (nested) call ompmklset(Tn)

    !$omp task firstprivate( vz ) private( job,thread_id,io,ex )
    call initsignal()

    thread_id = OMP_GET_THREAD_NUM()
    job = thread_id+1
    !$omp critical
    moltmps(job)%nat = mols(vz)%nat
    moltmps(job)%at = mols(vz)%at
    moltmps(job)%xyz = mols(vz)%xyz
    !$omp end critical
    !>--- startup printout (thread safe)
    call parallel_md_block_printout(mddats(vz),vz)

    !>--- the acutal MD call with timing
    call profiler%start(vz)
    call dynamics(moltmps(job),mddats(vz),calculations(job),pr,io)
    call profiler%stop(vz)

    !>--- finish printout (thread safe)
    call parallel_md_finish_printout(mddats(vz),vz,io,profiler)
    !$omp end task
  end do
  !$omp taskwait
  !$omp end single
  !$omp end parallel

!>--- collect trajectories into one
  call collect(nsim,mddats)

  call profiler%clear()
  !>--- Nullify per-thread shared handles, then cleanup stats
  do i = 1,T
    do j = 1,env%calc%ncalculations
      if (calculations(i)%calcs(j)%id == jobtype%libtorch) &
        calculations(i)%calcs(j)%libtorch_handle = c_null_ptr
      if (calculations(i)%calcs(j)%id == jobtype%pymlip) &
        calculations(i)%calcs(j)%pymlip_handle = c_null_ptr
      if (calculations(i)%calcs(j)%id == jobtype%ase_socket) &
        calculations(i)%calcs(j)%socket_handle = c_null_ptr
    end do
    call mlip_cleanup_all(calculations(i))
  end do
  !>--- Free shared models (skipped if mlip_keep_loaded for persistence)
  if (.not. env%calc%mlip_keep_loaded) then
    call libtorch_shared_cleanup()
    do j = 1,env%calc%ncalculations
      if (env%calc%calcs(j)%id == jobtype%libtorch) &
        env%calc%calcs(j)%libtorch_handle = c_null_ptr
    end do
  end if
  call mlip_cleanup_all(env%calc)
  deallocate (calculations)
  if (allocated(moltmps)) deallocate (moltmps)
  return
contains
  subroutine collect(n,mddats)
    implicit none
    integer :: n
    type(mddata) :: mddats(n)
    logical :: ex
    integer :: i,io,ich,ich2
    character(len=:),allocatable :: atmp
    character(len=256) :: btmp
    open (newunit=ich,file='crest_dynamics.trj')
    do i = 1,n
      atmp = mddats(i)%trajectoryfile
      inquire (file=atmp,exist=ex)
      if (ex) then
        open (newunit=ich2,file=atmp)
        io = 0
        do while (io == 0)
          read (ich2,'(a)',iostat=io) btmp
          if (io == 0) then
            write (ich,'(a)') trim(btmp)
          end if
        end do
        close (ich2)
      end if
    end do
    close (ich)
    return
  end subroutine collect
end subroutine crest_search_multimd2

!========================================================================================!
subroutine parallel_md_block_printout(MD,vz)
!***********************************************
!* subroutine parallel_md_block_printout
!* This will print information about the MD/MTD
!* simulation. The execution is omp threadsave
!***********************************************
  use crest_parameters,only:wp,stdout,sep
  use crest_data
  use crest_calculator
  use strucrd
  use dynamics_module
  use shake_module
  use iomod,only:to_str
  implicit none
  type(mddata),intent(in) :: MD
  integer,intent(in) :: vz
  character(len=40) :: atmp
  integer :: il
  !$omp critical

  if (MD%simtype == type_md) then
    write (atmp,'(a,1x,i3)') 'starting MD',vz
  else if (MD%simtype == type_mtd) then
    if (MD%cvtype(1) == cv_rmsd_static) then
      write (atmp,'(a,1x,i3)') 'starting static MTD',vz
    else
      write (atmp,'(a,1x,i4)') 'starting MTD',vz
    end if
  end if
  il = (44-len_trim(atmp))/2
  write (stdout,'(2x,a,1x,a,1x,a)') repeat(':',il),trim(atmp),repeat(':',il)

  write (stdout,'(2x,"|   MD simulation time   :",f8.1," ps       |")') MD%length_ps
  write (stdout,'(2x,"|   target T             :",f8.1," K        |")') MD%tsoll
  write (stdout,'(2x,"|   timestep dt          :",f8.1," fs       |")') MD%tstep
  write (stdout,'(2x,"|   dump interval(trj)   :",f8.1," fs       |")') MD%dumpstep
  if (MD%shake.and.MD%shk%shake_mode > 0) then
    if (MD%shk%shake_mode == 2) then
      write (stdout,'(2x,"|   SHAKE algorithm      :",a5," (all bonds) |")') to_str(MD%shake)
    else
      write (stdout,'(2x,"|   SHAKE algorithm      :",a5," (H only) |")') to_str(MD%shake)
    end if
  end if
  if (allocated(MD%active_potentials)) then
    write (stdout,'(2x,"|   active potentials    :",i4," potential    |")') size(MD%active_potentials,1)
  end if
  if (MD%simtype == type_mtd) then
    if (MD%cvtype(1) == cv_rmsd) then
      write (stdout,'(2x,"|   dump interval(Vbias) :",f8.2," ps       |")') &
          & MD%mtd(1)%cvdump_fs/1000.0_wp
    end if
    write (stdout,'(2x,"|   Vbias prefactor (k)  :",f8.4," Eh       |")') &
      &  MD%mtd(1)%kpush
    if (MD%cvtype(1) == cv_rmsd.or.MD%cvtype(1) == cv_rmsd_static) then
      write (stdout,'(2x,"|   Vbias exponent (α)   :",f8.4," bohr⁻²   |")') MD%mtd(1)%alpha
    else
      write (stdout,'(2x,"|   Vbias exponent (α)   :",f8.4,"          |")') MD%mtd(1)%alpha
    end if
  end if

  !$omp end critical

end subroutine parallel_md_block_printout

subroutine parallel_md_finish_printout(MD,vz,io,profiler)
!*******************************************
!* subroutine parallel_md_finish_printout
!* This will print information termination
!* info about the MD/MTD simulation
!*******************************************
  use crest_parameters,only:wp,stdout,sep
  use crest_data
  use crest_calculator
  use strucrd
  use dynamics_module
  use shake_module
  implicit none
  type(mddata),intent(in) :: MD
  integer,intent(in) :: vz,io
  type(timer),intent(inout) :: profiler
  character(len=40) :: atmp
  character(len=80) :: btmp

  !$omp critical

  if (MD%simtype == type_mtd) then
    if (MD%cvtype(1) == cv_rmsd_static) then
      write (atmp,'(a)') '*sMTD'
    else
      write (atmp,'(a)') '*MTD'
    end if
  else
    write (atmp,'(a)') '*MD'
  end if
  if (io == 0) then
    write (btmp,'(a,1x,i3,a)') trim(atmp),vz,' completed successfully'
  else
    write (btmp,'(a,1x,i3,a)') trim(atmp),vz,' terminated EARLY'
  end if
  call profiler%write_timing(stdout,vz,trim(btmp))

  !$omp end critical

end subroutine parallel_md_finish_printout
!========================================================================================!

!========================================================================================!
subroutine load_parallel_pymlip_models(env, calculations, T, nsim, jcalc, iostat)
!*************************************************************
!* Load N parallel pymlip model copies for GPU-parallel MD.
!* Measures GPU memory footprint of the first model, then
!* loads as many additional copies as GPU memory allows.
!*************************************************************
  use crest_parameters, only: wp, stdout
  use iso_c_binding, only: c_null_ptr
  use crest_data
  use crest_calculator
  use calc_pymlip, only: pymlip_init, pymlip_get_gpu_memory_f
  implicit none
  type(systemdata), intent(inout) :: env
  integer, intent(in) :: T, nsim, jcalc
  type(calcdata), intent(inout) :: calculations(T)
  integer, intent(out) :: iostat

  integer :: n_par, i, k, io, mem_io
  integer(8) :: gpu_total, gpu_free_before, gpu_free_after, footprint

  iostat = 0

  !>--- Measure GPU memory before loading
  call pymlip_get_gpu_memory_f(gpu_total, gpu_free_before, mem_io)
  write(stdout,'(1x,a,i0,a,i0,a,i0)') &
    'GPU mem query (before model): io=',mem_io, &
    ' total_MB=',gpu_total/(1024*1024),' free_MB=',gpu_free_before/(1024*1024)

  !>--- Load first model on env%calc (master copy)
  call pymlip_init(env%calc%calcs(jcalc), io)
  if (io /= 0) then
    write(stdout,'(a)') '**ERROR** Failed to load pymlip model'
    iostat = 1
    return
  end if

  !>--- Measure GPU memory after loading
  call pymlip_get_gpu_memory_f(gpu_total, gpu_free_after, mem_io)
  write(stdout,'(1x,a,i0,a,i0)') &
    'GPU mem query (after model): io=',mem_io,' free_MB=',gpu_free_after/(1024*1024)

  !>--- Estimate per-model footprint.
  !>    Use delta if available; otherwise estimate from total reserved
  !>    (gpu_total - gpu_free gives all PyTorch-reserved memory = ~1 model).
  footprint = 0
  if (mem_io == 0) then
    if (gpu_free_before > gpu_free_after) then
      footprint = gpu_free_before - gpu_free_after
    else
      footprint = gpu_total - gpu_free_after  !> total reserved ≈ 1 model
    end if
    env%calc%calcs(jcalc)%mlip_model_footprint = int(footprint / (1024*1024))
  end if

  !>--- Determine number of parallel model copies
  n_par = env%calc%calcs(jcalc)%mlip_n_parallel
  if (n_par <= 0 .and. footprint > 0 .and. gpu_free_after > 0) then
    !> Auto: how many more copies fit? (20% safety margin per copy)
    n_par = 1 + int(gpu_free_after / (footprint * 12 / 10))
    n_par = max(1, min(n_par, T, nsim))
  else if (n_par <= 0) then
    n_par = 1
  end if
  n_par = min(n_par, T)

  write(stdout,'(1x,a,i0,a,i0,a)') &
    'GPU parallel MD: n_par=',n_par,' footprint_MB=',footprint/(1024*1024)
  if (footprint > 0) then
    write(stdout,'(1x,a,i0,a)') &
      'GPU parallel MD: loading ',n_par,' MLIP model copy(ies)'
    write(stdout,'(1x,a,i0,a,i0,a,i0,a)') &
      '  GPU: ',gpu_total/(1024*1024),' MB total, ', &
      gpu_free_after/(1024*1024),' MB free, ', &
      footprint/(1024*1024),' MB/model'
  end if

  !>--- First thread gets the master handle (env owns it)
  calculations(1)%calcs(jcalc)%pymlip_handle = env%calc%calcs(jcalc)%pymlip_handle
  calculations(1)%calcs(jcalc)%mlip_is_owner = .false.

  !>--- Load independent copies for threads 2..n_par
  do i = 2, n_par
    call pymlip_init(calculations(i)%calcs(jcalc), io)
    if (io /= 0) then
      write(stdout,'(a,i0,a)') &
        '  WARNING: Model copy ',i,' failed, sharing handle instead'
      calculations(i)%calcs(jcalc)%pymlip_handle = &
        env%calc%calcs(jcalc)%pymlip_handle
      calculations(i)%calcs(jcalc)%mlip_is_owner = .false.
    else
      calculations(i)%calcs(jcalc)%mlip_is_owner = .true.
    end if
  end do

  !>--- Remaining threads share handles (round-robin)
  do i = n_par+1, T
    k = mod(i-1, n_par) + 1
    calculations(i)%calcs(jcalc)%pymlip_handle = &
      calculations(k)%calcs(jcalc)%pymlip_handle
    calculations(i)%calcs(jcalc)%mlip_is_owner = .false.
  end do

end subroutine load_parallel_pymlip_models
!========================================================================================!
