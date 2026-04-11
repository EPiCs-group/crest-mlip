# This file is part of crest.
# SPDX-Identifier: LGPL-3.0-or-later
#
# crest is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# crest is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with crest.  If not, see <https://www.gnu.org/licenses/>.

if get_option('WITH_LIBTORCH')
  srcs += files('libtorch_bridge.cpp')
endif

if get_option('WITH_PYMLIP')
  srcs += files('pymlip_bridge.c')
endif

srcs += files('ase_socket_bridge.c')
srcs += files('process_bridge.c')
srcs += files('worker_pool.c')

srcs += files(
  'calculator_libtorch.F90',
  'calculator_pymlip.F90',
  'calculator_ase_socket.F90',
  'calculator.F90',
  'calc_type.f90',
  'printouts.F90',
  'constraints.f90',
  'gfn0_api.F90',
  'gfnff_api.F90',
  'lj.f90',
  'lwoniom.F90', 
  'oniom_hessian.F90',
  'nonadiabatic.f90',
  'tblite_api.F90',
  'api_helpers.F90',
  'api_engrad.f90', 
  'gradreader.f90',
  'libpvol.F90',
  'xtb_sc.f90',
  'subprocess_types.f90',
  'orca_sc.f90',
  'generic_sc.f90',
  'turbom_sc.f90',
  'subprocess_engrad.f90',
)
