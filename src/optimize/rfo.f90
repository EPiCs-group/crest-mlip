!================================================================================!
! This file is part of crest.
!
! Copyright (C) 2021 - 2022 Philipp Pracht
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
!
! Routines were adapted from the xtb code (github.com/grimme-lab/xtb)
! under the Open-source software LGPL-3.0 Licencse.
!================================================================================!

!> This module implements a standard RFO algorithm (in Cart. coords)

module rfo_module
  use iso_fortran_env, only: wp=>real64, sp=>real32
  use crest_calculator
  use axis_module
  use strucrd
  use ls_rmsd

  use optimize_type
  use optimize_maths
  use modelhessian_module
  use hessupdate_module
  use optimize_utils
!$ use omp_lib
  implicit none
  private

  public :: rfopt
  public :: rfopt_batch

!========================================================================================!
!========================================================================================!
contains  !> MODULE PROCEDURES START HERE
!========================================================================================!
!========================================================================================!

  subroutine rfopt(mol,calc,etot,grd,pr,wr,iostatus)
!*************************************************************************
!> subroutine rfopt
!> Implementation of the standard rational function optimizer (RFO)
!>
!> Input/Output:
!>      mol  - object containing the molecule,
!>             Cartesian coordinates in Bohrs.
!>             will be overwritten on output
!>     calc  - object containing calculation settings
!>             and optimization thresholds (look for calc% )
!>     etot  - on input initial energy (do a singlepoint before ancopt)
!>             on output final energy
!>      grd  - Cartesian gradient
!>       pr  - printout bool
!>       wr  - logfile (crestopt.log) bool
!>  iostatus - return status of the routine
!>             (success=0, error<0, not converged>0)
!!***********************************************************************
    implicit none
    !> INPUT/OUTPUT
    type(coord),intent(inout) :: mol
    type(calcdata),intent(in) :: calc
    real(wp),intent(inout) :: etot
    real(wp),intent(inout) :: grd(3,mol%nat)
    logical,intent(in) :: pr
    logical,intent(in) :: wr
    integer,intent(out) :: iostatus
    !> LOCAL
    integer  :: tight
    real(wp) :: eel
    real(wp) :: et
    real(wp) :: egap
    logical :: fail
    !> Local objects
    type(coord)   :: molopt
    type(optimizer)  :: OPT
    type(mhparam) :: mhset

    real(wp) :: step,amu2au,au2cm,dumi,dumj,damp,hlow,edum,s6,thr
    real(wp) :: maxdispl,gthr,ethr,hmax,energy,rij(3),t1,t0,w1,w0
    real(wp) :: rot(3),gnorm
    integer :: n3,i,j,k,l,jjj,ic,jc,ia,ja,ii,jj,info,nat3
    integer :: nvar,iter,nread,maxcycle,maxmicro,itry,maxopt,iupdat,iii
    integer :: id,ihess,error
    integer :: ilog,imax(3)
    real(wp) :: depred,echng,alp,gnold,eold,gchng,dummy,dsnrm,maxd
    real(wp),allocatable :: h(:,:)
    real(wp),allocatable :: b(:,:)
    real(wp),allocatable :: fc(:)
    real(wp),allocatable :: eig(:)
    real(wp),allocatable :: aux(:)
    real(wp),allocatable :: hess(:)
    integer,allocatable :: iwork(:)
    integer,allocatable :: totsym(:)
    real(wp),allocatable :: pmode(:,:)
    real(wp),allocatable :: grmsd(:,:)
    real(wp),allocatable :: grd1(:)
    real(wp),allocatable :: gold(:)
    real(wp),allocatable :: displ(:)
    integer :: nvar1,npvar,npvar1
    real(sp),allocatable :: eaug(:)
    real(sp),allocatable :: Uaug(:,:)
    real(sp),allocatable :: Aaug(:)
    type(convergence_log),allocatable :: avconv
    real(wp) :: U(3,3),x_center(3),y_center(3),rmsdval
    integer :: modef
    logical :: ex,converged,linear,exact
    logical :: econverged,gconverged,lowered
    real(wp) :: estart,esave
    real(sp),parameter :: r4dum = 1.e-8
    !> LAPACK & BLAS
    external :: dgemv
    real(wp),external :: ddot
    real(sp),external :: sdot


    iostatus = 0
    fail = .false.
    converged = .false.
    if (mol%nat .eq. 1) return
!>  defaults
    tight = calc%optlev
    modef = 0
    call get_optthr(mol%nat,tight,calc,ethr,gthr)
    iupdat = calc%iupdat
    hlow = calc%hlow_opt !> 0.01 in ancopt, 0.002 too small
    hmax = calc%hmax_opt
    maxdispl = calc%maxdispl_opt 
    gnorm = 0.0_wp
    depred = 0.0_wp
    echng = 0.0_wp
    alp = 1.0_wp
    exact = calc%exact_rf

!> initial number of steps in relax() routine before
!> new ANC are made by model Hessian
!> increased during opt.
    maxmicro = 100
    maxcycle = calc%maxcycle
    if (maxcycle .lt. maxmicro) maxmicro = maxcycle

    !> check if the molecule is linear
    call axis(mol%nat,mol%at,mol%xyz,rot,dumi)
    linear = (rot(3) .lt. 1.d-10).or.(mol%nat == 2)

    !> set degrees of freedom
    nat3 = 3*mol%nat
    nvar = nat3-6
    if (linear) then
      nvar = nat3-5
    end if
    if (calc%nfreeze .gt. 0) then ! exact fixing
      nvar = nat3-3*calc%nfreeze-3
      if (nvar .le. 0) nvar = 1
    end if

    !$omp critical
    allocate (pmode(nat3,1),grmsd(3,mol%nat)) ! dummy allocated
    !$omp end critical

!>--- print a summary of settings, if desired
    if (pr) then
      call print_optsummary(calc,tight,nvar,maxcycle,maxmicro, &
      &                       ethr,gthr,linear,wr)
    end if

!>--- initialize OPT object
    !$omp critical
    allocate (h(nat3,nat3),hess(nat3*(nat3+1)/2),eig(nat3))
    call OPT%allocate2(mol%nat) !> NOTE: OPT%nvar will be nat*3 !!!
    allocate (molopt%at(mol%nat),molopt%xyz(3,mol%nat))
    nvar1 = OPT%nvar+1         !> dimension of RF calculation
    npvar = OPT%nvar*(nvar1)/2 !> packed size of Hessian (note the abuse of nvar1!)
    npvar1 = nvar1*(nvar1+1)/2 !> packed size of augmented Hessian
    allocate (Uaug(nvar1,1),eaug(nvar1),Aaug(npvar1))
    allocate ( gold(OPT%nvar),displ(OPT%nvar),grd1(OPT%nvar),source=0.0_wp)
    !$omp end critical

!>------------------------------------------------------------------------
!>--- put the Hessian guess into the type
!>------------------------------------------------------------------------
   k = 0
   do i = 1,nat3
     do j = 1,i
       k = k+1
       if( i /= j )then
         OPT%hess(k) = 0.0_wp
       else
         OPT%hess(k) = calc%hguess
       endif
     end do
   end do

!>--- backup coordinates, and starting energy
    molopt%nat = mol%nat
    molopt%at = mol%at
    molopt%xyz = mol%xyz
    estart = etot

!>--- initialize .log file, if desired
    ilog = 942
    if (wr) then
      open (newunit=ilog,file='crestopt.log')
    end if

!>--- The ANCOPT iteration loop. "iter" is updated in relax() subroutine
    iter = 0
!>--- start with a printout of the preceeding single point
    if (pr) call print_optiter(iter)
    gnorm = norm2(grd)
    if (pr) then
      write (*,'(" * total energy  :",f14.7,1x,"Eh")',advance='no') etot
      write (*,'(5x,"change ΔE",e18.7,1x,"Eh")') 0.0_wp
      write (*,'(3x,"gradient norm :",f14.7,1x,"Eh/a0")',advance='no') gnorm
      write (*,'(2x,"predicted",e18.7)',advance='no') 0.0_wp
      write (*,'(1x,"("f7.2"%)")')-0.0_wp
    end if

!>======================================================================
    RFO_iter: do while (iter < maxcycle.and..not.converged)
!>======================================================================
!>--- count the step and print out
      iter = iter+1
      if (pr) call print_optiter(iter)
      gold = reshape(grd, [nat3])
      gnold = gnorm
      eold = energy

!>--- project trans. and rot. from Hessian
!      if (.not.linear) then
!        if (calc%nfreeze == 0) then
!          call trproj(molopt%nat,nat3,molopt%xyz,OPT%hess,.false.,0,pmode,1)  !> normal
!        else
!          call trproj(molopt%nat,nat3,molopt%xyz,OPT%hess,.false.,calc%freezelist) !> fozen atoms
!        end if
!      end if

!>--- calc predicted energy change based on E = E0 + delta * G + delta^2 * H
      if (iter > 1) then
        call prdechng(OPT%nvar,gold,displ,OPT%hess,depred)
      end if

!>------------------------------------------------------------------------
!>--- SINGLEPOINT CALCULATION
!>------------------------------------------------------------------------
      grd = 0.0_wp
      call engrad(molopt,calc,energy,grd,iostatus)
      if (iostatus .ne. 0) then
        fail = .true.
        exit RFO_iter
      end if
      gnorm = norm2(grd)
      grd1 = reshape(grd, [nat3])

!>--- dump to .log file
      if (wr) then
        call molopt%appendlog(ilog,energy)
      end if

      if (gnorm .gt. 500.0_wp) then
        if (pr) write (*,*) '|grad| > 500, something is totally wrong!'
        fail = .true.
        iostatus = -1
        exit RFO_iter
      end if

!>--- check for convergence
      gchng = gnorm-gnold
      echng = energy-eold
      econverged = abs(echng) .lt. ethr
      gconverged = gnorm .lt. gthr
      lowered = echng .lt. 0.0_wp

!>--- optimization step printout
      if (pr) then
        write (*,'(" * total energy  :",f14.7,1x,"Eh")',advance='no') energy
        write (*,'(5x,"change ΔE",e18.7,1x,"Eh")') echng
        write (*,'(3x,"gradient norm :",f14.7,1x,"Eh/a0")',advance='no') gnorm
        write (*,'(2x,"predicted",e18.7)',advance='no') depred
        if (iter > 1) then
          dummy = (depred-echng)/echng*100.0_wp
          if (abs(dummy) < 1000.0_wp) then
            write (*,'(1x,"("f7.2"%)")') dummy
          else
            write (*,'(1x,"(*******%)")')
          end if
        else
          write (*,'(1x,"("f7.2"%)")')-100.0_wp
        end if
      end if

!>--- dynamic scaling in dependence of grad norm
!>--- if we are close to convergence we can take larger steps
      alp = 1.0d-1
      if (gnorm .lt. 0.002) then ! 0.002
        alp = 1.5d-1 ! 1.5
      endif
      if (gnorm .lt. 0.0006) then
        alp = 2.0d-1 ! 2
      endif
      if (gnorm .lt. 0.0003) then
        alp = 3.0d-1 ! 3
      endif

!>------------------------------------------------------------------------
!> Update the Hessian
!>------------------------------------------------------------------------
      if (iter .gt. 1) then
!>--- Hessian update, but only after first iteration (iter > 1)
        select case (iupdat)
        case (0)
          call bfgs(OPT%nvar,gnorm,grd1,gold,displ,OPT%hess)
        case (1)
          call powell(OPT%nvar,gnorm,grd1,gold,displ,OPT%hess)
        case (2)
          call sr1(OPT%nvar,gnorm,grd1,gold,displ,OPT%hess)
        case (3)
          call bofill(OPT%nvar,gnorm,grd1,gold,displ,OPT%hess)
        case (4)
          call schlegel(OPT%nvar,gnorm,grd1,gold,displ,OPT%hess)
        case default
          write (*,*) 'invalid hessian update selection'
          stop
        end select
      end if

!>------------------------------------------------------------------------
!>  rational function (RF) method
!>------------------------------------------------------------------------
!>  To get initial guess for the displacement solve this:
!>  (Note: SINGLE PRECISION accuracy, there are some typecasts!)
!>
!>   ⎛ H  g ⎞ ⎛ dx ⎞     ⎛ dx ⎞
!>   ⎝ g  0 ⎠ ⎝  1 ⎠ = λ ⎝  1 ⎠
!>     Aaug    Uaug       Uaug

!>--- first, augment Hessian by gradient, everything packed, no blowup
      Aaug(1:npvar) = real(OPT%hess(1:npvar),sp)
      Aaug(npvar+1:npvar1-1) = real(grd1(1:OPT%nvar),sp)
      Aaug(npvar1) = 0.0_sp

!>--- choose solver for the RF eigenvalue problem
      if (exact.or.nvar1 .lt. 50) then
        call solver_sspevx(nvar1,r4dum,Aaug,Uaug,eaug,fail)
      else
        !>--- steepest decent guess for displacement
        if (iter .eq. 1) then
          Uaug(:,1) = [-real(grd1(1:OPT%nvar),sp),1.0_sp]
          dsnrm = sqrt(sdot(nvar1,Uaug,1,Uaug,1))
          Uaug = Uaug/real(dsnrm,sp)
        end if
        call solver_sdavidson(nvar1,r4dum,Aaug,Uaug,eaug,fail,.false.)
        !>--- if that failed, retry with better solver
        if (fail) then
          call solver_sspevx(nvar1,r4dum,Aaug,Uaug,eaug,fail)
        end if
      end if

!>--- divide by last element(=λ) to get the displacement vector dx
      if (fail.or.abs(Uaug(nvar1,1)) .lt. 1.e-10) then
        if (pr) write (*,*) "internal rational function error"
        iostatus = -1
        exit RFO_iter
      end if
      displ(1:OPT%nvar) = Uaug(1:OPT%nvar,1)/Uaug(nvar1,1)

!>--- rescale displacementaif necessary
      maxd = alp*sqrt(ddot(OPT%nvar,displ,1,displ,1))
      if (maxd > maxdispl) then
        if (pr) write (*,'(" * rescaling step by",f14.7)') maxdispl/maxd
        displ = maxdispl*displ/maxd
      end if

!>--- now some output
      dsnrm = sqrt(ddot(OPT%nvar,displ,1,displ,1))
      if (pr) then
        !> this array is currently not used and will be overwritten in next step
        gold = abs(displ)
        imax(1) = maxloc(gold,1); gold(imax(1)) = 0.0_wp
        imax(2) = maxloc(gold,1); gold(imax(2)) = 0.0_wp
        imax(3) = maxloc(gold,1)
        write (*,'(3x,"displ. norm   :",f14.7,1x,"a.u.")',advance='no') &
          dsnrm*alp
        write (*,'(6x,"lambda   ",e18.7)') eaug(1)
        write (*,'(3x,"maximum displ.:",f14.7,1x,"a.u.")',advance='no') &
          abs(displ(imax(1)))*alp
        write (*,'(6x,"in coords ",3("#",i0,", "),"...")') imax
      end if

!>------------------------------------------------------------------------
!>--- new coordinates
!>------------------------------------------------------------------------
      molopt%xyz = molopt%xyz+reshape(displ, [3,molopt%nat])*alp

!>--- converged ?
      econverged = abs(echng) .lt. ethr
      gconverged = gnorm .lt. gthr
      lowered = echng .lt. 0.0_wp
      converged = econverged.and.gconverged.and.lowered
      if (pr) then
        call print_convd(econverged,gconverged)
      end if
      if (converged) then
        converged = .true.
        etot = energy
        exit RFO_iter
      end if

!>======================================================================
    end do RFO_iter
!>======================================================================

!>--- close .log file
    if (wr) then
      close (ilog)
    end if

    if (converged) then
!>--- if the relaxation converged properly do this
      iostatus = 0
      if (pr) then
        call rmsd(mol%nat,mol%xyz,molopt%xyz,1,U,x_center,y_center,rmsdval,.false.,grmsd)
        write (*,'(/,3x,"***",1x,a,1x,i0,1x,a,1x,"***",/)') &
          "GEOMETRY OPTIMIZATION CONVERGED AFTER",iter,"ITERATIONS"
        write (*,'(72("-"))')
        write (*,'(1x,"total energy gain   :",F18.7,1x,"Eh",F14.4,1x,"kcal/mol")') &
          etot-estart, (etot-estart)*autokcal
        write (*,'(1x,"total RMSD          :",F18.7,1x,"a0",F14.4,1x,"Å")') &
          rmsdval,rmsdval*autoaa
        write (*,'(72("-"))')
      end if
    else if (iostatus .ne. 0) then
!>--- if iostatus =/= 0, something went wrong in the relaxation
      if (pr) then
        write (*,'(/,3x,"***",1x,a,1x,"***",/)') &
          "GEOMETRY RELAXATION FAILED"
      end if
    else
!>--- not converging in the given cycles is considered a FAILURE
      !> some iostatus>0 is selected to signal this
      iostatus = iter
      if (pr) then
        write (*,'(/,3x,"***",1x,a,1x,i0,1x,a,1x,"***",/)') &
          "FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN",iter,"ITERATIONS"
      end if
    end if

!>--- overwrite input structure with optimized one
    mol%nat = molopt%nat
    mol%at = molopt%at
    mol%xyz = molopt%xyz

!> deallocate data
    !$omp critical
    if (allocated(gold)) deallocate(gold)
    if (allocated(displ)) deallocate(displ)
    if (allocated(grd1)) deallocate(grd1)
    if (allocated(Uaug)) deallocate (Uaug)
    if (allocated(eaug)) deallocate (eaug)
    if (allocated(Aaug)) deallocate (Aaug)
    if (allocated(grmsd)) deallocate (grmsd)
    if (allocated(pmode)) deallocate (pmode)
    if (allocated(h)) deallocate (h)
    if (allocated(hess)) deallocate (hess)
    if (allocated(molopt%at)) deallocate (molopt%at)
    if (allocated(molopt%xyz)) deallocate (molopt%xyz)
    call OPT%deallocate
    !$omp end critical

    return
  end subroutine rfopt

!========================================================================================!
!========================================================================================!

  subroutine rfopt_batch(nall, nat, at, xyz_all, calc, energies_out, &
                         status_out, pr)
!***********************************************************************
!> Batched RFO optimizer: optimizes nall structures simultaneously
!> using lockstep iteration with batched energy+gradient GPU calls.
!>
!> All structures advance through the RFO algorithm together.
!> At each step, a single batched GPU call evaluates energies and
!> gradients for all active (not yet converged) structures.
!> CPU work (Hessian update, RF solve) is parallelized with OpenMP.
!>
!> Input/Output:
!>   nall         - number of structures
!>   nat          - atoms per structure (all same)
!>   at(nat)      - atomic numbers
!>   xyz_all(3,nat,nall) - coordinates in Bohr, overwritten with optimized
!>   calc         - calculation settings (thresholds, MLIP handles)
!>   pr           - print progress
!> Output:
!>   energies_out(nall) - final energies
!>   status_out(nall)   - 0=converged, >0=maxcycle reached, <0=error
!***********************************************************************
    implicit none
    !> Arguments
    integer, intent(in) :: nall, nat
    integer, intent(in) :: at(nat)
    real(wp), intent(inout) :: xyz_all(3, nat, nall)
    type(calcdata), intent(inout) :: calc
    real(wp), intent(out) :: energies_out(nall)
    integer, intent(out) :: status_out(nall)
    logical, intent(in) :: pr

    !> Dimensions
    integer :: nat3, npvar, nvar1, npvar1
    integer :: maxcycle, iupdat
    real(wp) :: ethr, gthr, hguess, maxdispl
    logical :: exact

    !> Per-structure state (struct of arrays)
    real(wp), allocatable :: xyz(:,:)      ! (nat3, nall)
    real(wp), allocatable :: grd(:,:)      ! (nat3, nall)
    real(wp), allocatable :: gold(:,:)     ! (nat3, nall)
    real(wp), allocatable :: displ(:,:)    ! (nat3, nall)
    real(wp), allocatable :: hess(:,:)     ! (npvar, nall)
    real(wp), allocatable :: energy(:)     ! (nall)
    real(wp), allocatable :: eold(:)       ! (nall)
    real(wp), allocatable :: gnorm(:)      ! (nall)
    real(wp), allocatable :: gnold(:)      ! (nall)
    integer, allocatable  :: siter(:)      ! (nall) step counter
    integer, allocatable  :: sstatus(:)    ! (nall) 0=active,1=converged,-1=failed

    !> Active set
    integer :: nactive
    integer, allocatable :: amap(:)        ! active index -> global index

    !> Batch I/O buffers
    real(wp), allocatable :: pos_batch(:)  ! (nat3 * nactive)
    real(wp), allocatable :: e_batch(:)    ! (nactive)
    real(wp), allocatable :: g_batch(:)    ! (nat3 * nactive)

    !> Loop variables
    integer :: step, ia, ig, i, j, k, io, nconv, nfail
    real(wp) :: echng, alp, depred, maxd, dsnrm, gnrm
    logical :: econverged, gconverged, lowered
    integer :: gpu_batch_sz

    !> Thread count (workspace allocated inside OMP parallel region)
    integer :: nthreads
    real(sp), parameter :: r4dum = 1.e-8_sp

    !> BLAS externals
    real(wp), external :: ddot
    real(sp), external :: sdot

    !> Timing
    integer(8) :: t0, t1, trate
    real(wp) :: dt_gpu, dt_cpu

    !>=========================================================================
    !> SETUP
    !>=========================================================================
    nat3 = 3 * nat
    npvar = nat3 * (nat3 + 1) / 2
    nvar1 = nat3 + 1
    npvar1 = nvar1 * (nvar1 + 1) / 2

    maxcycle = calc%maxcycle
    iupdat = calc%iupdat
    hguess = calc%hguess
    maxdispl = calc%maxdispl_opt
    exact = calc%exact_rf
    call get_optthr(nat, calc%optlev, calc, ethr, gthr)

    gpu_batch_sz = mlip_auto_batch_size(nat)

    !> Allocate per-structure state
    allocate(xyz(nat3, nall), grd(nat3, nall), gold(nat3, nall))
    allocate(displ(nat3, nall), source=0.0_wp)
    allocate(hess(npvar, nall))
    allocate(energy(nall), eold(nall), gnorm(nall), gnold(nall))
    allocate(siter(nall), source=0)
    allocate(sstatus(nall), source=0)  ! 0 = active
    allocate(amap(nall))

    !> Copy input coordinates to flat layout
    do i = 1, nall
      xyz(:, i) = reshape(xyz_all(:, :, i), [nat3])
    end do

    !> Initialize Hessians to hguess * I (diagonal)
    hess = 0.0_wp
    do i = 1, nall
      k = 0
      do j = 1, nat3
        k = k + j  ! index of diagonal element (j,j)
        hess(k, i) = hguess
      end do
    end do

    !> Initialize active set (all structures)
    nactive = nall
    do i = 1, nall
      amap(i) = i
    end do

    nthreads = 1
    !$ nthreads = omp_get_max_threads()

    if (pr) then
      write(stdout, '(/,1x,a,i0,a)') &
        'Batched RFO optimization of ', nall, ' structures'
      write(stdout, '(1x,a,i0,a,e10.3,a,e10.3)') &
        'maxcycle=', maxcycle, '  ethr=', ethr, '  gthr=', gthr
      write(stdout, '(1x,a,i0,a,i0)') &
        'GPU batch size=', gpu_batch_sz, '  OMP threads=', nthreads
    end if

    !>=========================================================================
    !> INITIAL BATCH ENGRAD (step 0)
    !>=========================================================================
    allocate(pos_batch(nat3 * nactive))
    allocate(e_batch(nactive), g_batch(nat3 * nactive))

    !> Collect all positions
    do ia = 1, nactive
      ig = amap(ia)
      pos_batch((ia-1)*nat3+1 : ia*nat3) = xyz(:, ig)
    end do

    !> Batched GPU call (sub-batched if needed)
    call batch_engrad_sub(calc, nactive, nat, nat3, at, pos_batch, &
                          e_batch, g_batch, gpu_batch_sz, io)
    if (io /= 0) then
      if (pr) write(stdout, '(a)') '**ERROR** Initial batch engrad failed'
      status_out = -1
      energies_out = 0.0_wp
      return
    end if

    !> Distribute results
    do ia = 1, nactive
      ig = amap(ia)
      energy(ig) = e_batch(ia)
      grd(:, ig) = g_batch((ia-1)*nat3+1 : ia*nat3)
      gnorm(ig) = norm2(grd(:, ig))
    end do

    deallocate(pos_batch, e_batch, g_batch)

    !>=========================================================================
    !> MAIN OPTIMIZATION LOOP
    !>=========================================================================
    do step = 1, maxcycle
      if (nactive == 0) exit

      call system_clock(t0, trate)

      !>--- Save old state for all active structures
      do ia = 1, nactive
        ig = amap(ia)
        gold(:, ig) = grd(:, ig)
        gnold(ig) = gnorm(ig)
        eold(ig) = energy(ig)
        siter(ig) = siter(ig) + 1
      end do

      !>--- Per-structure CPU work: Hessian update + RF solve + coordinate step
      !>    (before the engrad call — we update coords first, then evaluate)
      !$omp parallel do default(none) schedule(dynamic) &
      !$omp shared(nactive, amap, nat3, npvar, nvar1, npvar1, &
      !$omp        hess, grd, gold, displ, xyz, gnorm, energy, eold, &
      !$omp        siter, sstatus, iupdat, exact, maxdispl, ethr, gthr) &
      !$omp private(ia, ig, echng, alp, maxd, dsnrm, gnrm, &
      !$omp         econverged, gconverged, lowered)
      do ia = 1, nactive
        ig = amap(ia)
        if (sstatus(ig) /= 0) cycle

        !> Check convergence from previous step's engrad
        echng = energy(ig) - eold(ig)
        econverged = abs(echng) < ethr
        gconverged = gnorm(ig) < gthr
        lowered = echng < 0.0_wp
        if (siter(ig) > 1 .and. econverged .and. gconverged .and. lowered) then
          sstatus(ig) = 1  ! converged
          cycle
        end if

        !> Guard against divergence
        if (gnorm(ig) > 500.0_wp) then
          sstatus(ig) = -1
          cycle
        end if

        !> Hessian update (BFGS default)
        if (siter(ig) > 1) then
          select case (iupdat)
          case (0)
            call bfgs(nat3, gnorm(ig), grd(:,ig), gold(:,ig), displ(:,ig), hess(:,ig))
          case (1)
            call powell(nat3, gnorm(ig), grd(:,ig), gold(:,ig), displ(:,ig), hess(:,ig))
          case (2)
            call sr1(nat3, gnorm(ig), grd(:,ig), gold(:,ig), displ(:,ig), hess(:,ig))
          case (3)
            call bofill(nat3, gnorm(ig), grd(:,ig), gold(:,ig), displ(:,ig), hess(:,ig))
          case (4)
            call schlegel(nat3, gnorm(ig), grd(:,ig), gold(:,ig), displ(:,ig), hess(:,ig))
          end select
        end if

        !> Dynamic step scaling based on gradient norm
        alp = 1.0d-1
        gnrm = gnorm(ig)
        if (gnrm < 0.002_wp) alp = 1.5d-1
        if (gnrm < 0.0006_wp) alp = 2.0d-1
        if (gnrm < 0.0003_wp) alp = 3.0d-1

        !> Solve RF eigenvalue problem (thread-local workspace)
        block
          real(sp), allocatable :: Aaug(:), Uaug(:,:), eaug(:)
          logical :: rf_fail

          allocate(Aaug(npvar1), Uaug(nvar1, 1), eaug(nvar1))
          rf_fail = .false.

          !> Build augmented Hessian [H g; g 0] in packed format
          Aaug(1:npvar) = real(hess(1:npvar, ig), sp)
          Aaug(npvar+1:npvar1-1) = real(grd(1:nat3, ig), sp)
          Aaug(npvar1) = 0.0_sp

          !> Steepest descent guess for first iteration
          if (siter(ig) == 1) then
            Uaug(1:nat3, 1) = -real(grd(1:nat3, ig), sp)
            Uaug(nvar1, 1) = 1.0_sp
            dsnrm = sqrt(sdot(nvar1, Uaug(1,1), 1, Uaug(1,1), 1))
            if (dsnrm > 0.0_sp) Uaug(:,1) = Uaug(:,1) / real(dsnrm, sp)
          else
            !> Use previous displacement as guess for Davidson
            Uaug(1:nat3, 1) = real(displ(1:nat3, ig), sp)
            Uaug(nvar1, 1) = 1.0_sp
            dsnrm = sqrt(sdot(nvar1, Uaug(1,1), 1, Uaug(1,1), 1))
            if (dsnrm > 0.0_sp) Uaug(:,1) = Uaug(:,1) / real(dsnrm, sp)
          end if

          !> Solve
          if (exact .or. nvar1 < 50) then
            call solver_sspevx(nvar1, r4dum, Aaug, Uaug, eaug, rf_fail)
          else
            call solver_sdavidson(nvar1, r4dum, Aaug, Uaug, eaug, &
                                  rf_fail, .false.)
            if (rf_fail) then
              call solver_sspevx(nvar1, r4dum, Aaug, Uaug, eaug, rf_fail)
            end if
          end if

          if (rf_fail .or. abs(Uaug(nvar1, 1)) < 1.e-10) then
            sstatus(ig) = -1
            deallocate(Aaug, Uaug, eaug)
            cycle
          end if

          !> Extract displacement
          displ(1:nat3, ig) = real(Uaug(1:nat3, 1), wp) / &
                               real(Uaug(nvar1, 1), wp)
          deallocate(Aaug, Uaug, eaug)
        end block

        !> Rescale if too large
        maxd = alp * sqrt(ddot(nat3, displ(1,ig), 1, displ(1,ig), 1))
        if (maxd > maxdispl) then
          displ(:, ig) = maxdispl * displ(:, ig) / maxd
        end if

        !> Update coordinates
        xyz(:, ig) = xyz(:, ig) + displ(:, ig) * alp
      end do
      !$omp end parallel do

      !>--- Rebuild active set
      nconv = 0
      nfail = 0
      nactive = 0
      do i = 1, nall
        if (sstatus(i) == 0) then
          nactive = nactive + 1
          amap(nactive) = i
        else if (sstatus(i) == 1) then
          nconv = nconv + 1
        else
          nfail = nfail + 1
        end if
      end do

      call system_clock(t1)
      dt_cpu = real(t1 - t0, wp) / real(trate, wp)

      !>--- Progress output
      if (pr .and. (mod(step, 10) == 0 .or. step == 1 .or. nactive == 0)) then
        write(stdout, '(1x,a,i4,a,i0,a,i0,a,i0,a,f6.1,a)') &
          'Step ', step, ': ', nactive, ' active, ', nconv, &
          ' converged, ', nfail, ' failed  (', dt_cpu*1000.0_wp, ' ms)'
      end if

      if (nactive == 0) exit

      !>--- Batch engrad for all remaining active structures
      call system_clock(t0)
      allocate(pos_batch(nat3 * nactive))
      allocate(e_batch(nactive), g_batch(nat3 * nactive))

      do ia = 1, nactive
        ig = amap(ia)
        pos_batch((ia-1)*nat3+1 : ia*nat3) = xyz(:, ig)
      end do

      call batch_engrad_sub(calc, nactive, nat, nat3, at, pos_batch, &
                            e_batch, g_batch, gpu_batch_sz, io)

      if (io /= 0) then
        if (pr) write(stdout, '(a,i0)') '**ERROR** Batch engrad failed at step ', step
        !> Mark all remaining as failed
        do ia = 1, nactive
          sstatus(amap(ia)) = -1
        end do
        deallocate(pos_batch, e_batch, g_batch)
        exit
      end if

      do ia = 1, nactive
        ig = amap(ia)
        energy(ig) = e_batch(ia)
        grd(:, ig) = g_batch((ia-1)*nat3+1 : ia*nat3)
        gnorm(ig) = norm2(grd(:, ig))
      end do

      deallocate(pos_batch, e_batch, g_batch)

      call system_clock(t1)
      dt_gpu = real(t1 - t0, wp) / real(trate, wp)

    end do  !> step loop

    !>=========================================================================
    !> WRITE BACK RESULTS
    !>=========================================================================
    do i = 1, nall
      xyz_all(:, :, i) = reshape(xyz(:, i), [3, nat])
      energies_out(i) = energy(i)
      if (sstatus(i) == 1) then
        status_out(i) = 0  ! converged
      else if (sstatus(i) == -1) then
        status_out(i) = -1  ! failed
      else
        !> Not converged within maxcycle
        if (calc%anopt) then
          status_out(i) = 0  ! accept partial optimization
        else
          status_out(i) = siter(i)  ! return step count as status
        end if
      end if
    end do

    nconv = count(status_out == 0)
    nfail = count(status_out < 0)
    if (pr) then
      write(stdout, '(/,1x,a,i0,a,i0,a,i0,a,i0,a)') &
        'Batched RFO complete: ', nconv, ' converged, ', &
        nall - nconv - nfail, ' not converged, ', nfail, ' failed (of ', nall, ')'
    end if

    !> Cleanup
    deallocate(xyz, grd, gold, displ, hess)
    deallocate(energy, eold, gnorm, gnold, siter, sstatus, amap)

  contains

    !> Sub-batched engrad dispatch: handles GPU batch size limits
    subroutine batch_engrad_sub(calc, nstructs, nat, nat3, at, &
                                pos, energies, gradients, bsz, iostat)
      type(calcdata), intent(inout) :: calc
      integer, intent(in) :: nstructs, nat, nat3, bsz
      integer, intent(in) :: at(nat)
      real(wp), intent(in) :: pos(nat3 * nstructs)
      real(wp), intent(out) :: energies(nstructs)
      real(wp), intent(out) :: gradients(nat3 * nstructs)
      integer, intent(out) :: iostat
      integer :: offset, chunk, io2

      iostat = 0
      do offset = 1, nstructs, bsz
        chunk = min(bsz, nstructs - offset + 1)

        if (calc%calcs(1)%id == jobtype%pymlip) then
          call pymlip_engrad_batch_f(calc%calcs(1), chunk, nat, at, &
            pos((offset-1)*nat3+1), energies(offset), &
            gradients((offset-1)*nat3+1), io2)
        else if (calc%calcs(1)%id == jobtype%libtorch) then
          call libtorch_engrad_batch_pipeline_f(calc%calcs(1), &
            chunk, nat, at, pos((offset-1)*nat3+1), energies(offset), &
            gradients((offset-1)*nat3+1), bsz, io2)
        else
          !> Fallback: serial engrad calls
          block
            type(coord) :: tmol
            real(wp) :: egrd(3, nat)
            integer :: ii
            allocate(tmol%at(nat), tmol%xyz(3, nat))
            tmol%nat = nat
            tmol%at = at
            io2 = 0
            do ii = 1, chunk
              tmol%xyz = reshape(pos((offset-1+ii-1)*nat3+1 : (offset-1+ii)*nat3), [3, nat])
              call engrad(tmol, calc, energies(offset+ii-1), egrd, io2)
              gradients((offset-1+ii-1)*nat3+1 : (offset-1+ii)*nat3) = reshape(egrd, [nat3])
            end do
            deallocate(tmol%at, tmol%xyz)
          end block
        end if

        if (io2 /= 0) then
          iostat = io2
          return
        end if
      end do
    end subroutine

  end subroutine rfopt_batch

!========================================================================================!
!========================================================================================!
end module rfo_module
