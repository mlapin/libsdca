cmake_minimum_required(VERSION 2.8)
project(libsdca)

###############################################################################
#
# Build Configuration
#
###############################################################################

# Whether to use 32 or 64 bit integers
# ILP64 pro: address more than 2^31-1 arrays
# ILP64 con: default cblas.h uses 32 bit ints
set(USE_ILP64 TRUE)

# Whether to use multi-threaded or sequential Intel MKL layer
set(USE_SEQUENTIAL FALSE)


# CXX compiler flags, see also:
# https://www.gnu.org/software/gsl/manual/html_node/GCC-warning-options-for-numerical-programs.html
#
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Werror -W -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wcast-qual -Wcast-align")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdouble-promotion -Wconversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith -Wwrite-strings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -fshort-enums -fno-common")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -fopenmp -fstack-protector")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -freg-struct-return")

set(LTO_FLAGS "-flto flto-partition=none -fno-fat-lto-objects")
set(LTO_FLAGS "${LTO_FLAGS} -fuse-linker-plugin -funroll-loops")

# Debug / Release specific flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${LTO_FLAGS}")

# Install prefix
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH
    "Install directory used by install." FORCE)
endif()


###############################################################################
#
# Code Dependencies and Third Party Libraries
#
###############################################################################

# Custom Cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Matlab (do not use their BLAS lib for now as it does not provide CBLAS API)
#find_package(Matlab COMPONENTS BLAS_LIBRARY)
find_package(Matlab)

# Intel MKL
find_package(MKL)

# BLAS libraries from various vendors (e.g. Atlas BLAS)
find_package(BLAS)

# Helper script to select a BLAS lib and set linker flags
include(UseBLAS)


###############################################################################
#
# SDCA Library Code
#
###############################################################################

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
if(Matlab_FOUND)
  add_subdirectory(matlab)
else(Matlab_FOUND)
  message(WARNING "Matlab not found. Matlab interface will not be compiled.")
endif(Matlab_FOUND)


###############################################################################
#
# Testing
#
###############################################################################

# A main function for small tests
add_subdirectory(main)
