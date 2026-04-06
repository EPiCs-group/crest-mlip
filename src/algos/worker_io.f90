!================================================================================!
! This file is part of crest.
! SPDX-Identifier: LGPL-3.0-or-later
!
! Binary config serialization for worker processes.
! Each worker process reads its molecule, MD settings, and calculator
! configuration from a binary file written by the parent.
!================================================================================!
module worker_io_module
  use crest_parameters, only: wp, stdout
  use strucrd, only: coord
  use dynamics_module, only: mddata, mtdpot, type_md, type_mtd
  use crest_calculator, only: calcdata, calculation_settings, jobtype
  use shake_module, only: shakedata
  implicit none
  private

  public :: write_worker_config, read_worker_config
  public :: write_worker_opt_config, read_worker_opt_config
  public :: write_worker_opt_results, read_worker_opt_results

contains

!========================================================================================!
subroutine write_worker_config(filename, mol, mddat, calc, worker_index)
!*********************************************************************
!* Write molecule + MD settings + calculator config to a binary file.
!* Only scalars and arrays are written -- NO C pointers, NO file units.
!* The worker will reconstruct handles by re-initializing the model.
!*********************************************************************
  implicit none
  character(len=*), intent(in) :: filename
  type(coord), intent(in) :: mol
  type(mddata), intent(in) :: mddat
  type(calcdata), intent(in) :: calc
  integer, intent(in) :: worker_index
  integer :: u, i, j, slen

  open(newunit=u, file=filename, form='unformatted', status='replace', &
       access='stream', action='write')

  !>--- Magic number and version for validation
  write(u) 'CWRK'
  write(u) 1  ! format version

  !>--- Worker index
  write(u) worker_index

  !>--- Molecule data
  write(u) mol%nat
  write(u) mol%chrg
  write(u) mol%uhf
  write(u) mol%at(1:mol%nat)
  write(u) mol%xyz(1:3, 1:mol%nat)

  !>--- MD settings (scalars)
  write(u) mddat%simtype
  write(u) mddat%length_ps
  write(u) mddat%length_steps
  write(u) mddat%tstep
  write(u) mddat%dumpstep
  write(u) mddat%sdump
  write(u) mddat%printstep
  write(u) mddat%md_hmass
  write(u) mddat%shake
  write(u) mddat%tsoll
  write(u) mddat%thermostat
  write(u) mddat%thermo_damp
  write(u) mddat%md_index

  !>--- MD file paths (variable-length strings)
  call write_string(u, mddat%trajectoryfile)
  call write_string(u, mddat%restartfile)

  !>--- SHAKE constraint data (preserve parent's mode 2 constraints from GFN-FF WBOs)
  write(u) mddat%shk%shake_mode
  write(u) mddat%shk%ncons
  if (mddat%shk%ncons > 0) then
    write(u) mddat%shk%conslist(1:2, 1:mddat%shk%ncons)
    write(u) mddat%shk%distcons(1:mddat%shk%ncons)
  end if

  !>--- MTD potentials
  write(u) mddat%npot
  if (mddat%npot > 0) then
    do i = 1, mddat%npot
      write(u) mddat%mtd(i)%mtdtype
      write(u) mddat%mtd(i)%kpush
      write(u) mddat%mtd(i)%alpha
      write(u) mddat%mtd(i)%cvdump_fs
      write(u) mddat%mtd(i)%maxsave
      !>--- atinclude mask
      if (allocated(mddat%mtd(i)%atinclude)) then
        write(u) .true.
        write(u) size(mddat%mtd(i)%atinclude)
        write(u) mddat%mtd(i)%atinclude
      else
        write(u) .false.
      end if
    end do
    if (allocated(mddat%cvtype)) then
      write(u) .true.
      write(u) mddat%cvtype(1:mddat%npot)
    else
      write(u) .false.
    end if
  end if

  !>--- Calculator settings
  write(u) calc%ncalculations
  write(u) calc%id
  write(u) calc%optlev
  if (calc%nfreeze > 0 .and. allocated(calc%freezelist)) then
    write(u) calc%nfreeze
    write(u) calc%freezelist
  else
    write(u) 0
  end if

  !>--- Constraints
  write(u) calc%nconstraints

  !>--- Per-level calculation settings (only what worker needs)
  do j = 1, calc%ncalculations
    call write_calc_level(u, calc%calcs(j))
  end do

  close(u)

end subroutine write_worker_config

!========================================================================================!
subroutine read_worker_config(filename, mol, mddat, calc, worker_index, iostat)
!*********************************************************************
!* Read molecule + MD settings + calculator config from binary file.
!* Reconstructs all data structures; handles are left as c_null_ptr.
!*********************************************************************
  use iso_c_binding, only: c_null_ptr
  implicit none
  character(len=*), intent(in) :: filename
  type(coord), intent(out) :: mol
  type(mddata), intent(out) :: mddat
  type(calcdata), intent(out) :: calc
  integer, intent(out) :: worker_index
  integer, intent(out) :: iostat
  integer :: u, i, j, n, version
  character(len=4) :: magic
  logical :: has_data

  iostat = 0

  open(newunit=u, file=filename, form='unformatted', status='old', &
       access='stream', action='read', iostat=iostat)
  if (iostat /= 0) then
    write(stdout, '(a,a)') '**ERROR** Cannot open worker config: ', trim(filename)
    return
  end if

  !>--- Validate magic number
  read(u, iostat=iostat) magic
  if (iostat /= 0 .or. magic /= 'CWRK') then
    write(stdout, '(a)') '**ERROR** Invalid worker config file (bad magic)'
    iostat = -1
    close(u)
    return
  end if
  read(u) version
  if (version /= 1) then
    write(stdout, '(a,i0)') '**ERROR** Unsupported worker config version: ', version
    iostat = -2
    close(u)
    return
  end if

  !>--- Worker index
  read(u) worker_index

  !>--- Molecule
  read(u) mol%nat
  read(u) mol%chrg
  read(u) mol%uhf
  allocate(mol%at(mol%nat))
  allocate(mol%xyz(3, mol%nat))
  read(u) mol%at(1:mol%nat)
  read(u) mol%xyz(1:3, 1:mol%nat)

  !>--- MD settings
  read(u) mddat%simtype
  read(u) mddat%length_ps
  read(u) mddat%length_steps
  read(u) mddat%tstep
  read(u) mddat%dumpstep
  read(u) mddat%sdump
  read(u) mddat%printstep
  read(u) mddat%md_hmass
  read(u) mddat%shake
  read(u) mddat%tsoll
  read(u) mddat%thermostat
  read(u) mddat%thermo_damp
  read(u) mddat%md_index
  mddat%requested = .true.

  !>--- MD file paths
  call read_string(u, mddat%trajectoryfile)
  call read_string(u, mddat%restartfile)

  !>--- SHAKE constraint data (preserves parent's mode 2 from GFN-FF WBOs)
  read(u) mddat%shk%shake_mode
  read(u) mddat%shk%ncons
  mddat%nshake = mddat%shk%ncons
  if (mddat%shk%ncons > 0) then
    allocate(mddat%shk%conslist(2, mddat%shk%ncons))
    allocate(mddat%shk%distcons(mddat%shk%ncons))
    read(u) mddat%shk%conslist(1:2, 1:mddat%shk%ncons)
    read(u) mddat%shk%distcons(1:mddat%shk%ncons)
    !>--- allocate workspace arrays (normally done by init_shake)
    allocate(mddat%shk%dro(3, mddat%shk%ncons), source=0.0_wp)
    allocate(mddat%shk%dr(4, mddat%shk%ncons), source=0.0_wp)
  end if
  !>--- mark as initialized so dynamics() -> init_shake() skips re-init.
  !>    freezeptr is left unset here; dynamics() handles it at line 170-172
  !>    BEFORE init_shake is called (which returns immediately).
  mddat%shk%initialized = .true.
  nullify(mddat%shk%freezeptr)  ! safety: explicit null until dynamics() sets it

  !>--- MTD potentials
  read(u) mddat%npot
  if (mddat%npot > 0) then
    mddat%simtype = type_mtd
    allocate(mddat%mtd(mddat%npot))
    do i = 1, mddat%npot
      read(u) mddat%mtd(i)%mtdtype
      read(u) mddat%mtd(i)%kpush
      read(u) mddat%mtd(i)%alpha
      read(u) mddat%mtd(i)%cvdump_fs
      read(u) mddat%mtd(i)%maxsave
      read(u) has_data
      if (has_data) then
        read(u) n
        allocate(mddat%mtd(i)%atinclude(n))
        read(u) mddat%mtd(i)%atinclude
      end if
    end do
    read(u) has_data
    if (has_data) then
      allocate(mddat%cvtype(mddat%npot))
      read(u) mddat%cvtype(1:mddat%npot)
    end if
  end if

  !>--- Calculator settings
  read(u) calc%ncalculations
  read(u) calc%id
  read(u) calc%optlev
  read(u) calc%nfreeze
  if (calc%nfreeze > 0) then
    allocate(calc%freezelist(mol%nat))
    read(u) calc%freezelist
  end if

  !>--- Constraints
  read(u) calc%nconstraints

  !>--- Per-level calculation settings
  allocate(calc%calcs(calc%ncalculations))
  do j = 1, calc%ncalculations
    call read_calc_level(u, calc%calcs(j))
  end do

  !>--- Allocate working arrays
  allocate(calc%etmp(calc%ncalculations), source=0.0_wp)
  allocate(calc%grdtmp(3, mol%nat, calc%ncalculations), source=0.0_wp)
  allocate(calc%eweight(calc%ncalculations), source=1.0_wp)

  close(u)

end subroutine read_worker_config

!========================================================================================!
!> Helper: write a single calculation_settings level
!========================================================================================!
subroutine write_calc_level(u, cs)
  implicit none
  integer, intent(in) :: u
  type(calculation_settings), intent(in) :: cs

  !>--- Job type and basic settings
  write(u) cs%id
  write(u) cs%chrg
  write(u) cs%uhf
  write(u) cs%weight
  write(u) cs%active

  !>--- Working directory
  call write_string(u, cs%calcspace)

  !>--- MLIP settings (pymlip)
  call write_string(u, cs%pymlip_model_type)
  call write_string(u, cs%pymlip_model_path)
  call write_string(u, cs%pymlip_device)
  call write_string(u, cs%pymlip_task)
  call write_string(u, cs%pymlip_atom_refs)
  call write_string(u, cs%pymlip_compile_mode)
  call write_string(u, cs%pymlip_dtype)
  write(u) cs%pymlip_turbo
  write(u) cs%pymlip_debug

  !>--- MLIP settings (libtorch)
  call write_string(u, cs%libtorch_model_path)
  write(u) cs%libtorch_device_id
  write(u) cs%libtorch_model_format
  write(u) cs%libtorch_cutoff
  write(u) cs%libtorch_debug

  !>--- API settings (for tblite/gfn-ff fallbacks)
  write(u) cs%tblitelvl
  write(u) cs%etemp
  write(u) cs%accuracy
  write(u) cs%maxscc

end subroutine write_calc_level

!========================================================================================!
!> Helper: read a single calculation_settings level
!========================================================================================!
subroutine read_calc_level(u, cs)
  use iso_c_binding, only: c_null_ptr
  implicit none
  integer, intent(in) :: u
  type(calculation_settings), intent(out) :: cs

  !>--- Job type and basic settings
  read(u) cs%id
  read(u) cs%chrg
  read(u) cs%uhf
  read(u) cs%weight
  read(u) cs%active

  !>--- Working directory
  call read_string(u, cs%calcspace)

  !>--- MLIP settings (pymlip)
  call read_string(u, cs%pymlip_model_type)
  call read_string(u, cs%pymlip_model_path)
  call read_string(u, cs%pymlip_device)
  call read_string(u, cs%pymlip_task)
  call read_string(u, cs%pymlip_atom_refs)
  call read_string(u, cs%pymlip_compile_mode)
  call read_string(u, cs%pymlip_dtype)
  read(u) cs%pymlip_turbo
  read(u) cs%pymlip_debug

  !>--- MLIP settings (libtorch)
  call read_string(u, cs%libtorch_model_path)
  read(u) cs%libtorch_device_id
  read(u) cs%libtorch_model_format
  read(u) cs%libtorch_cutoff
  read(u) cs%libtorch_debug

  !>--- API settings
  read(u) cs%tblitelvl
  read(u) cs%etemp
  read(u) cs%accuracy
  read(u) cs%maxscc

  !>--- Ensure handles are null (worker will initialize its own)
  cs%pymlip_handle = c_null_ptr
  cs%libtorch_handle = c_null_ptr
  cs%socket_handle = c_null_ptr
  cs%pymlip_call_count = 0
  cs%pymlip_total_time = 0.0d0
  cs%libtorch_call_count = 0
  cs%libtorch_total_time = 0.0d0

end subroutine read_calc_level

!========================================================================================!
!========================================================================================!
!> Optimization worker config serialization
!========================================================================================!
!========================================================================================!

subroutine write_worker_opt_config(filename, nat, nstructs, at, xyz, calc, worker_index)
!*********************************************************************
!* Write a chunk of structures + calculator/optimization settings
!* for an optimization worker process.
!*********************************************************************
  implicit none
  character(len=*), intent(in) :: filename
  integer, intent(in) :: nat, nstructs, worker_index
  integer, intent(in) :: at(nat)
  real(wp), intent(in) :: xyz(3, nat, nstructs)
  type(calcdata), intent(in) :: calc
  integer :: u, j

  open(newunit=u, file=filename, form='unformatted', status='replace', &
       access='stream', action='write')

  write(u) 'COPT'   ! magic for optimization config
  write(u) 1        ! format version
  write(u) worker_index
  write(u) nat
  write(u) nstructs

  !>--- Atom types (shared across all structures)
  write(u) at(1:nat)

  !>--- All structures (contiguous)
  write(u) xyz(1:3, 1:nat, 1:nstructs)

  !>--- Calculator settings
  write(u) calc%ncalculations
  write(u) calc%id
  if (calc%nfreeze > 0 .and. allocated(calc%freezelist)) then
    write(u) calc%nfreeze
    write(u) calc%freezelist
  else
    write(u) 0
  end if
  write(u) calc%nconstraints

  !>--- Optimization settings
  write(u) calc%optlev
  write(u) calc%maxcycle
  write(u) calc%opt_engine
  write(u) calc%ethr_opt
  write(u) calc%gthr_opt
  write(u) calc%hlow_opt
  write(u) calc%hmax_opt
  write(u) calc%acc_opt
  write(u) calc%maxdispl_opt
  write(u) calc%maxerise
  write(u) calc%hguess
  write(u) calc%exact_rf
  write(u) calc%average_conv
  write(u) calc%iupdat
  write(u) calc%anopt
  write(u) calc%micro_opt

  !>--- Per-level calculation settings
  do j = 1, calc%ncalculations
    call write_calc_level(u, calc%calcs(j))
  end do

  close(u)
end subroutine write_worker_opt_config

!========================================================================================!
subroutine read_worker_opt_config(filename, nat, nstructs, at, xyz, calc, &
                                   worker_index, iostat)
!*********************************************************************
!* Read optimization worker config from binary file.
!*********************************************************************
  use iso_c_binding, only: c_null_ptr
  implicit none
  character(len=*), intent(in) :: filename
  integer, intent(out) :: nat, nstructs, worker_index, iostat
  integer, allocatable, intent(out) :: at(:)
  real(wp), allocatable, intent(out) :: xyz(:,:,:)
  type(calcdata), intent(out) :: calc
  integer :: u, j, version
  character(len=4) :: magic

  iostat = 0
  open(newunit=u, file=filename, form='unformatted', status='old', &
       access='stream', action='read', iostat=iostat)
  if (iostat /= 0) then
    write(stdout, '(a,a)') '**ERROR** Cannot open opt config: ', trim(filename)
    return
  end if

  read(u, iostat=iostat) magic
  if (iostat /= 0 .or. magic /= 'COPT') then
    write(stdout, '(a)') '**ERROR** Invalid opt config (bad magic)'
    iostat = -1; close(u); return
  end if
  read(u) version
  if (version /= 1) then
    write(stdout, '(a,i0)') '**ERROR** Unsupported opt config version: ', version
    iostat = -2; close(u); return
  end if

  read(u) worker_index
  read(u) nat
  read(u) nstructs

  allocate(at(nat))
  allocate(xyz(3, nat, nstructs))
  read(u) at(1:nat)
  read(u) xyz(1:3, 1:nat, 1:nstructs)

  !>--- Calculator settings
  read(u) calc%ncalculations
  read(u) calc%id
  read(u) calc%nfreeze
  if (calc%nfreeze > 0) then
    allocate(calc%freezelist(nat))
    read(u) calc%freezelist
  end if
  read(u) calc%nconstraints

  !>--- Optimization settings
  read(u) calc%optlev
  read(u) calc%maxcycle
  read(u) calc%opt_engine
  read(u) calc%ethr_opt
  read(u) calc%gthr_opt
  read(u) calc%hlow_opt
  read(u) calc%hmax_opt
  read(u) calc%acc_opt
  read(u) calc%maxdispl_opt
  read(u) calc%maxerise
  read(u) calc%hguess
  read(u) calc%exact_rf
  read(u) calc%average_conv
  read(u) calc%iupdat
  read(u) calc%anopt
  read(u) calc%micro_opt

  !>--- Per-level calculation settings
  allocate(calc%calcs(calc%ncalculations))
  do j = 1, calc%ncalculations
    call read_calc_level(u, calc%calcs(j))
  end do

  !>--- Allocate working arrays
  allocate(calc%etmp(calc%ncalculations), source=0.0_wp)
  allocate(calc%grdtmp(3, nat, calc%ncalculations), source=0.0_wp)
  allocate(calc%eweight(calc%ncalculations), source=1.0_wp)

  close(u)
end subroutine read_worker_opt_config

!========================================================================================!
subroutine write_worker_opt_results(filename, nat, nstructs, xyz, energies, status)
!*********************************************************************
!* Write optimization results: optimized coords + energies + status
!*********************************************************************
  implicit none
  character(len=*), intent(in) :: filename
  integer, intent(in) :: nat, nstructs
  real(wp), intent(in) :: xyz(3, nat, nstructs)
  real(wp), intent(in) :: energies(nstructs)
  integer, intent(in) :: status(nstructs)
  integer :: u

  open(newunit=u, file=filename, form='unformatted', status='replace', &
       access='stream', action='write')
  write(u) 'CRES'
  write(u) nat
  write(u) nstructs
  write(u) energies(1:nstructs)
  write(u) status(1:nstructs)
  write(u) xyz(1:3, 1:nat, 1:nstructs)
  close(u)
end subroutine write_worker_opt_results

!========================================================================================!
subroutine read_worker_opt_results(filename, nat, nstructs, xyz, energies, status, iostat)
!*********************************************************************
!* Read optimization results from worker output file.
!*********************************************************************
  implicit none
  character(len=*), intent(in) :: filename
  integer, intent(in) :: nat, nstructs
  real(wp), intent(out) :: xyz(3, nat, nstructs)
  real(wp), intent(out) :: energies(nstructs)
  integer, intent(out) :: status(nstructs)
  integer, intent(out) :: iostat
  integer :: u
  character(len=4) :: magic

  iostat = 0
  open(newunit=u, file=filename, form='unformatted', status='old', &
       access='stream', action='read', iostat=iostat)
  if (iostat /= 0) return

  read(u, iostat=iostat) magic
  if (iostat /= 0 .or. magic /= 'CRES') then
    iostat = -1; close(u); return
  end if
  read(u) nat  ! redundant but validates
  read(u) nstructs
  read(u) energies(1:nstructs)
  read(u) status(1:nstructs)
  read(u) xyz(1:3, 1:nat, 1:nstructs)
  close(u)
end subroutine read_worker_opt_results

!========================================================================================!
!> Helper: write allocatable string (length-prefixed)
!========================================================================================!
subroutine write_string(u, str)
  implicit none
  integer, intent(in) :: u
  character(len=:), allocatable, intent(in) :: str
  integer :: slen

  if (allocated(str)) then
    slen = len_trim(str)
    write(u) slen
    if (slen > 0) write(u) str(1:slen)
  else
    write(u) 0
  end if
end subroutine write_string

!========================================================================================!
!> Helper: read allocatable string (length-prefixed)
!========================================================================================!
subroutine read_string(u, str)
  implicit none
  integer, intent(in) :: u
  character(len=:), allocatable, intent(out) :: str
  integer :: slen

  read(u) slen
  if (slen > 0) then
    allocate(character(len=slen) :: str)
    read(u) str
  else
    str = ''
  end if
end subroutine read_string

end module worker_io_module
