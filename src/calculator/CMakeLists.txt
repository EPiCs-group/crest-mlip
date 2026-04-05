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

set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

if(WITH_LIBTORCH)
  list(APPEND srcs
    "${dir}/libtorch_bridge.cpp"
  )
endif()

if(WITH_PYMLIP)
  list(APPEND srcs
    "${dir}/pymlip_bridge.c"
  )
endif()

list(APPEND srcs
  "${dir}/ase_socket_bridge.c"
  "${dir}/process_bridge.c"
)

list(APPEND srcs
  "${dir}/calc_type.f90"
  "${dir}/calculator_libtorch.F90"
  "${dir}/calculator_pymlip.F90"
  "${dir}/calculator_ase_socket.F90"
  "${dir}/calculator.F90"
  "${dir}/printouts.F90"
  "${dir}/constraints.f90"
  "${dir}/gfn0_api.F90"
  "${dir}/gfnff_api.F90"
  "${dir}/lj.f90"
  "${dir}/lwoniom.F90"
  "${dir}/oniom_hessian.F90"
  "${dir}/nonadiabatic.f90"
  "${dir}/tblite_api.F90"
  "${dir}/api_helpers.F90"
  "${dir}/api_engrad.f90"
  "${dir}/gradreader.f90"
  "${dir}/libpvol.F90"
  "${dir}/subprocess_types.f90"
  "${dir}/xtb_sc.f90"
  "${dir}/orca_sc.f90"
  "${dir}/generic_sc.f90"
  "${dir}/turbom_sc.f90"
  "${dir}/subprocess_engrad.f90"
)

set(srcs ${srcs} PARENT_SCOPE)







