#.rst:
# SelectBLAS
# ----------
#
# Select module for BLAS
#
# Selects a BLAS library and sets up C and C++ to use it.
#
# Use
#     find_package(BLAS)
#     find_package(MKL)
#     find_package(Matlab COMPONENTS MX_LIBRARY BLAS_LIBRARY IOMP_LIBRARY)
# before
#     include(SelectBLAS)
#
# BLAS libraries have the following precedence:
#     Intel MKL
#     Accelerate Framework
#     BLAS shipped with Matlab
#     BLAS found by FindBLAS
#
# Intel MKL setup depends on the flags
#     USE_ILP64
#     USE_SEQUENTIAL
#
# The following variables are set by this script
#     BLAS_FOUND
#     BLAS_LIBRARIES
#     BLAS_LINKER_FLAGS

# Copyright 2015 Maksim Lapin.

if(MKL_FOUND)

  set(BLAS_LIBRARIES)
  set(BLAS_LINKER_FLAGS)
  list(APPEND BLAS_LINKER_FLAGS "-Wl,--start-group")

  if(USE_ILP64)
    add_definitions(-DMKL_ILP64)
    list(APPEND BLAS_LINKER_FLAGS ${MKL_ILP64_LIBRARY})
  else()
    list(APPEND BLAS_LINKER_FLAGS ${MKL_LP64_LIBRARY})
  endif()

  list(APPEND BLAS_LINKER_FLAGS ${MKL_CORE_LIBRARY})

  if(USE_SEQUENTIAL)
    list(APPEND BLAS_LINKER_FLAGS ${MKL_SEQUENTIAL_LIBRARY})
  else()
    if(INTEL_OMP_LIBRARY)
      list(APPEND BLAS_LINKER_FLAGS ${MKL_INTEL_THREAD_LIBRARY})
      list(APPEND BLAS_LINKER_FLAGS ${INTEL_OMP_LIBRARY})
    elseif(Matlab_IOMP_LIBRARY)
      list(APPEND BLAS_LINKER_FLAGS ${MKL_INTEL_THREAD_LIBRARY})
      list(APPEND BLAS_LINKER_FLAGS ${Matlab_IOMP_LIBRARY})
    else()
      list(APPEND BLAS_LINKER_FLAGS ${MKL_GNU_THREAD_LIBRARY})
      add_definitions(-fopenmp)
    endif()
  endif()

  list(APPEND BLAS_LINKER_FLAGS "-Wl,--end-group")

  include_directories("${MKL_INCLUDE_DIRS}")
  add_definitions(-DBLAS_MKL)

elseif(BLAS_Accelerate_LIBRARY)

  add_definitions(-DBLAS_ACCELERATE)

elseif(Matlab_BLAS_LIBRARY)

  add_definitions(-DBLAS_MATLAB)
  set(BLAS_LIBRARIES "${Matlab_BLAS_LIBRARY}")

elseif(BLAS_FOUND)

  find_file(CBLAS_HEADER cblas.h)
  if(CBLAS_HEADER)
    add_definitions(-DBLAS_DEFAULT)
  else()
    add_definitions(-DBLAS_DEFAULT_LOCAL_HEADER)
  endif()

endif()

if((BLAS_LIBRARIES OR BLAS_LINKER_FLAGS) AND Threads_FOUND)

  set(BLAS_FOUND TRUE)
  list(APPEND BLAS_LIBRARIES "-ldl" "-lm")
  list(APPEND BLAS_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
  message(STATUS "Selected BLAS libraries: ${BLAS_LIBRARIES}")

else()

  set(BLAS_FOUND FALSE)
  set(BLAS_LIBRARIES)
  set(BLAS_LINKER_FLAGS)
  message(FATAL_ERROR "BLAS or Threads not found.")

endif()
