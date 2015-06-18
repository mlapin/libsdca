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
set(USE_SEQUENTIAL TRUE)

# Whether to compile Matlab interface (if Matlab is found)
set(USE_MATLAB TRUE)

# If set, solver outputs parameters / solution details on begin and end
add_definitions(-DVERBOSE_BEGIN_END)

# If set, solver outputs some statistics on each duality gap computation
add_definitions(-DVERBOSE_DUALITY_GAP)

# CXX compiler flags
set(WARN_FLAGS "-pedantic -Werror -W -Wall -Wextra -Wwrite-strings")
set(WARN_FLAGS "${WARN_FLAGS} -Wcast-qual -Wcast-align -Wshadow")
set(WARN_FLAGS "${WARN_FLAGS} -Wconversion -Wpointer-arith")
if(CMAKE_COMPILER_IS_GNUCXX)
  set(WARN_FLAGS "${WARN_FLAGS} -Wdouble-promotion")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -m64 ${WARN_FLAGS}")

# Debug / Release specific flags -Wno-unused-variable
set(DEBUG_FLAGS "-O0 -DDEBUG -fstack-protector-all")

set(RELEASE_FLAGS "-O3 -DNDEBUG -fno-stack-protector -fno-exceptions")
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(RELEASE_FLAGS "${RELEASE_FLAGS} -flto -flto-partition=none")
  set(RELEASE_FLAGS "${RELEASE_FLAGS} -fno-fat-lto-objects")
  set(RELEASE_FLAGS "${RELEASE_FLAGS} -fuse-linker-plugin")
endif()

# Profiling (add debug symbols)
#set(RELEASE_FLAGS "${RELEASE_FLAGS} -g")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")


# Install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/bin" CACHE PATH
    "Install directory used by install." FORCE)
endif()

# Avoid a warning due to CMP0042
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()

###############################################################################
#
# Code Dependencies and Third Party Libraries
#
###############################################################################

# Custom Cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Matlab
if(USE_MATLAB)
  # Honor the visibility properties set for targets of type OBJECT
  # (visibility is set to 'hidden' for mex files, see FindMatlab.cmake)
  if(${CMAKE_VERSION} VERSION_GREATER 2.9)
    cmake_policy(SET CMP0063 NEW)
  endif()
  find_package(Matlab COMPONENTS MX_LIBRARY)
  set(USE_MATLAB ${Matlab_FOUND})
endif()

# Intel MKL
find_package(MKL)

# BLAS libraries from various vendors (e.g. Atlas BLAS)
find_package(BLAS)

# Helper script to select a BLAS lib and set compiler/linker flags
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
if(USE_MATLAB)
  add_subdirectory(matlab)
else()
  message(STATUS "Matlab interface will not be compiled.")
endif()


###############################################################################
#
# Testing
#
###############################################################################

# A main function for small tests
add_subdirectory(main)
