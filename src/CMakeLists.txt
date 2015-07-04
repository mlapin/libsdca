add_subdirectory(prox)
include_directories(.)

# Utility code
set(LIB_UTIL "libutil")
include_directories(util)
add_subdirectory(util)

# Projectors lib
set(LIB_PROJECTORS "libprojectors")
include_directories(projectors)
add_subdirectory(projectors)

# Solvers lib
set(LIB_SOLVERS "libsolvers")
include_directories(solvers)
add_subdirectory(solvers)

# Matlab interface
if(${Matlab_FOUND})
  add_subdirectory(matlab)
else()
  message(STATUS "Matlab interface will not be compiled.")
endif()

# A playground for quick checks and small tests
add_subdirectory(playground)
