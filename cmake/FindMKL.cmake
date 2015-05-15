#.rst:
# FindMKL
# --------
#
# Find Intel MKL library (version 11.1)
#

# Copyright 2015 Maksim Lapin.

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

find_path(
  MKL_INCLUDE_DIR
  mkl.h
  PATHS "${MKL_ROOT_DIR}/include"
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

set(MKL_INCLUDE_DIRS MKL_INCLUDE_DIR)
set(MKL_LIBRARIES MKL_ILP64_LIBRARY MKL_CORE_LIBRARY)
set(MKL_DEFINITIONS "-DMKL_ILP64")

find_package_handle_standard_args(
  MKL
  DEFAULT_MSG
  MKL_INCLUDE_DIR
  MKL_ILP64_LIBRARY
  MKL_CORE_LIBRARY
  MKL_INTEL_THREAD_LIBRARY
  MKL_GNU_THREAD_LIBRARY
  MKL_SEQUENTIAL_LIBRARY
  )

mark_as_advanced(
  MKL_INCLUDE_DIR
  MKL_ILP64_LIBRARY
  MKL_CORE_LIBRARY
  MKL_INTEL_THREAD_LIBRARY
  MKL_GNU_THREAD_LIBRARY
  MKL_SEQUENTIAL_LIBRARY
  )
