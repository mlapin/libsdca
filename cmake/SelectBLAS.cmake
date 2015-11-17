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
#     find_package(MKL) # optional
# before
#     include(SelectBLAS)
#
# If Intel MKL is found, it will be used instead of the BLAS found by FindBLAS.
# Intel MKL setup depends on the flags
#     USE_ILP64
#     USE_SEQUENTIAL
#
# Note: the following flags are overwritten in any case
#     BLAS_FOUND
#     BLAS_LIBRARIES

# Copyright 2015 Maksim Lapin.

if(MKL_FOUND)

  set(BLAS_LIBRARIES)
  list(APPEND BLAS_LIBRARIES "-Wl,--start-group")

  if(USE_ILP64)
    add_definitions(-DMKL_ILP64)
    list(APPEND BLAS_LIBRARIES ${MKL_ILP64_LIBRARY})
  else()
    list(APPEND BLAS_LIBRARIES ${MKL_LP64_LIBRARY})
  endif()

  list(APPEND BLAS_LIBRARIES ${MKL_CORE_LIBRARY})

  if(USE_SEQUENTIAL)
    list(APPEND BLAS_LIBRARIES ${MKL_SEQUENTIAL_LIBRARY})
  else()
    if(INTEL_OMP_LIBRARY)
      list(APPEND BLAS_LIBRARIES ${MKL_INTEL_THREAD_LIBRARY})
      list(APPEND BLAS_LIBRARIES ${INTEL_OMP_LIBRARY})
    else()
      list(APPEND BLAS_LIBRARIES ${MKL_GNU_THREAD_LIBRARY})
      add_definitions(-fopenmp)
    endif()
  endif()

  list(APPEND BLAS_LIBRARIES "-Wl,--end-group")

  include_directories("${MKL_INCLUDE_DIRS}")
  add_definitions(-DBLAS_MKL)

elseif(BLAS_Accelerate_LIBRARY)

  add_definitions(-DBLAS_ACCELERATE)

elseif(BLAS_FOUND)

  find_file(CBLAS_HEADER cblas.h)
  if(CBLAS_HEADER)
    add_definitions(-DBLAS_DEFAULT)
  else()
    add_definitions(-DBLAS_DEFAULT_LOCAL_HEADER)
  endif()

endif()

if(BLAS_LIBRARIES AND Threads_FOUND)

  set(BLAS_FOUND TRUE)
  list(APPEND BLAS_LIBRARIES "-ldl" "-lm")
  list(APPEND BLAS_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
  message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")

else()

  set(BLAS_FOUND FALSE)
  set(BLAS_LIBRARIES)
  message(FATAL_ERROR "BLAS or Threads not found.")

endif()
