# UseBLAS
# -------
#
# Use Module for BLAS
#
# Selects a BLAS library from several options and sets up C and C++ to use it.
# It is assumed that
#   FindThreads.cmake
# and at least one of the following have already been loaded:
#   FindBLAS.cmake (loads FindThreads.cmake)
#   FindMatlab.cmake
#   FindMKL.cmake
# Note that BLAS_* variables are overwritten.
# Priority:
# 1. Intel MKL
# 2. Matlab BLAS
# 3. FindBLAS

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
    elseif(NOT USE_MATLAB)
      list(APPEND BLAS_LIBRARIES ${MKL_GNU_THREAD_LIBRARY})
      add_definitions(-fopenmp)
    else()
      message(FATAL_ERROR "Matlab is not compatible with Intel MKL + GNU OMP.")
    endif()
  endif()

  list(APPEND BLAS_LIBRARIES "-Wl,--end-group")

  include_directories("${MKL_INCLUDE_DIRS}")
  add_definitions(-DBLAS_MKL)

#elseif(Matlab_BLAS_LIBRARY_FOUND)

#  list(APPEND BLAS_LIBRARIES ${Matlab_BLAS_LIBRARY})

#  include_directories(Matlab_INCLUDE_DIRS)
#  add_definitions(-DBLAS_MATLAB)

elseif(BLAS_FOUND)

  add_definitions(-DBLAS_DEFAULT)

endif()

if(BLAS_LIBRARIES AND Threads_FOUND)

  set(BLAS_FOUND TRUE)
  list(APPEND BLAS_LIBRARIES "-Wl,--as-needed" "-ldl" "-lm")
  list(APPEND BLAS_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
  message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")

else()

  set(BLAS_FOUND FALSE)
  set(BLAS_LIBRARIES)
  message(FATAL_ERROR "BLAS or Threads not found.")

endif()



# # Multi-threaded, libiomp5
# # Compiler:
# -DMKL_ILP64 -m64 -I${MKLROOT}/include
# # Linker:
# -Wl,--start-group
# ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a
# ${MKLROOT}/lib/intel64/libmkl_core.a
# ${MKLROOT}/lib/intel64/libmkl_intel_thread.a
# -Wl,--end-group
# -liomp5 -ldl -lpthread -lm
#
# # Multi-threaded, libgomp
# # Compiler:
# -fopenmp -DMKL_ILP64 -m64 -I${MKLROOT}/include
# # Linker:
# -Wl,--start-group
# ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a
# ${MKLROOT}/lib/intel64/libmkl_core.a
# ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a
# -Wl,--end-group
# -ldl -lpthread -lm
#
# # Sequential
# # Compiler:
# -DMKL_ILP64 -m64 -I${MKLROOT}/include
# # Linker:
# -Wl,--start-group
# ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a
# ${MKLROOT}/lib/intel64/libmkl_core.a
# ${MKLROOT}/lib/intel64/libmkl_sequential.a
# -Wl,--end-group
# -lpthread -lm
