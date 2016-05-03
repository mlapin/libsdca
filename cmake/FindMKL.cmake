#.rst:
# FindMKL
# -------
#
# Find module for Intel MKL
#
# Finds Intel MKL and Intel Composer XE installations
#
# The following variables and pre-defined locations are checked:
# for Intel MKL,
#     - MKL_ROOT_DIR            # CMake variable
#     - MKLROOT                 # environment variable
#     - /opt/intel/mkl          # default location
# (note that the MKL root dir contains the 'include' and 'lib' subdirs).
#
# for Intel OMP library,
#     - INTEL_OMP_DIR           # CMake variable
#     - /opt/intel/lib          # default location

# Copyright 2015-2016 Maksim Lapin.

include(FindPackageHandleStandardArgs)

set(MKL_ROOT_DIR ${MKL_ROOT_DIR} CACHE PATH
    "Intel MKL installation root path.")

if(MKL_ROOT_DIR)
  file(TO_CMAKE_PATH ${MKL_ROOT_DIR} MKL_ROOT_DIR)
elseif($ENV{MKLROOT})
  file(TO_CMAKE_PATH $ENV{MKLROOT} MKL_ROOT_DIR)
else()
  file(TO_CMAKE_PATH "/opt/intel/mkl" MKL_ROOT_DIR)
endif()

set(INTEL_OMP_DIR ${INTEL_OMP_DIR} CACHE PATH
    "Directory containing the Intel OMP library (libiomp5).")

if(INTEL_OMP_DIR)
  file(TO_CMAKE_PATH ${INTEL_OMP_DIR} INTEL_OMP_DIR)
else()
  file(TO_CMAKE_PATH "/opt/intel/lib" INTEL_OMP_DIR)
endif()

find_path(
  MKL_INCLUDE_DIR
  mkl.h
  PATHS "${MKL_ROOT_DIR}/include"
  )

find_library(
  MKL_LP64_LIBRARY
  NAMES libmkl_intel_lp64.a mkl_intel_lp64
  PATHS ${MKL_ROOT_DIR}/lib/intel64
  )
find_library(
  MKL_ILP64_LIBRARY
  NAMES libmkl_intel_ilp64.a mkl_intel_ilp64
  PATHS ${MKL_ROOT_DIR}/lib/intel64
  )
find_library(
  MKL_CORE_LIBRARY
  NAMES libmkl_core.a mkl_core
  PATHS ${MKL_ROOT_DIR}/lib/intel64
  )
find_library(
  MKL_INTEL_THREAD_LIBRARY
  NAMES libmkl_intel_thread.a mkl_intel_thread
  PATHS ${MKL_ROOT_DIR}/lib/intel64
  )
find_library(
  MKL_GNU_THREAD_LIBRARY
  NAMES libmkl_gnu_thread.a mkl_gnu_thread
  PATHS ${MKL_ROOT_DIR}/lib/intel64
  )
find_library(
  MKL_SEQUENTIAL_LIBRARY
  NAMES libmkl_sequential.a mkl_sequential
  PATHS ${MKL_ROOT_DIR}/lib/intel64
  )

find_library(
  INTEL_OMP_LIBRARY
  NAMES libiomp5 iomp5
  PATHS ${INTEL_OMP_DIR}/intel64
  )

set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
set(MKL_LIBRARIES ${MKL_CORE_LIBRARY})
set(MKL_DEFINITIONS)

find_package_handle_standard_args(
  MKL
  DEFAULT_MSG
  MKL_INCLUDE_DIR
  MKL_LP64_LIBRARY
  MKL_ILP64_LIBRARY
  MKL_CORE_LIBRARY
  MKL_INTEL_THREAD_LIBRARY
  MKL_GNU_THREAD_LIBRARY
  MKL_SEQUENTIAL_LIBRARY
  INTEL_OMP_LIBRARY
  )

mark_as_advanced(
  MKL_INCLUDE_DIR
  MKL_LP64_LIBRARY
  MKL_ILP64_LIBRARY
  MKL_CORE_LIBRARY
  MKL_INTEL_THREAD_LIBRARY
  MKL_GNU_THREAD_LIBRARY
  MKL_SEQUENTIAL_LIBRARY
  INTEL_OMP_LIBRARY
  )
