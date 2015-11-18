cmake_minimum_required(VERSION 2.8.9)
project(libsdca)

set(libsdca_MAJOR_VERSION 0)
set(libsdca_MINOR_VERSION 1)
set(libsdca_PATCH_VERSION 0)
set(libsdca_VERSION
  ${libsdca_MAJOR_VERSION}.${libsdca_MINOR_VERSION}.${libsdca_PATCH_VERSION})
add_definitions(-DLIBSDCA_VERSION="${libsdca_VERSION}")

###############################################################################
#
# Build Configuration
#
###############################################################################

#------------------------------------------------------------------------------
# Intel MKL configuration (not applicable if another BLAS is used)
#------------------------------------------------------------------------------
# Whether to use 64 or 32 bit integers (default: TRUE)
set(USE_ILP64 TRUE)

# Whether to use sequential or multi-threaded Intel MKL layer (default: FALSE)
set(USE_SEQUENTIAL FALSE)

# Whether to add a suffix (e.g. st/mt) for single-/multi-threaded versions
set(USE_THREADING_SUFFIX FALSE)

# Whether to use the fast math functions from https://github.com/herumi/fmath
add_definitions(-DUSE_FMATH_HERUMI=1)

# RPATH settings
set(CMAKE_SKIP_RPATH FALSE)
SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
SET(CMAKE_INSTALL_RPATH "\\\$ORIGIN")

#------------------------------------------------------------------------------
# Compiler flags
#------------------------------------------------------------------------------
# Select a build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif()

# Warning flags
set(WARN_FLAGS "-pedantic -Werror -W -Wall -Wextra -Wwrite-strings")
set(WARN_FLAGS "${WARN_FLAGS} -Wcast-qual -Wcast-align -Wshadow")
set(WARN_FLAGS "${WARN_FLAGS} -Wconversion -Wpointer-arith")
if(CMAKE_COMPILER_IS_GNUCXX)
  set(WARN_FLAGS "${WARN_FLAGS} -Wdouble-promotion")
endif()

# Debug flags
set(DEBUG_FLAGS "-O0 -DDEBUG -fstack-protector-all")

# Release flags
set(RELEASE_FLAGS "-O3 -DNDEBUG -march=native")
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(RELEASE_FLAGS "${RELEASE_FLAGS} -flto -flto-partition=none")
  set(RELEASE_FLAGS "${RELEASE_FLAGS} -fno-fat-lto-objects")
  set(RELEASE_FLAGS "${RELEASE_FLAGS} -fuse-linker-plugin")
endif()

# Build specific flags
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -std=c++11 ${WARN_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${RELEASE_FLAGS}")

#------------------------------------------------------------------------------
# Auxiliary Cmake configuration
#------------------------------------------------------------------------------
# Custom Cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH
    "Installation directory." FORCE)
endif()

# Avoid a warning due to CMP0042
if(POLICY CMP0042 AND APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()


###############################################################################
#
# Code Dependencies and Third Party Libraries
#
###############################################################################

#------------------------------------------------------------------------------
# Matlab interface
#------------------------------------------------------------------------------
# Honor the visibility properties set for targets of type OBJECT
# (visibility is set to 'hidden' for mex files, see FindMatlab.cmake)
if(POLICY CMP0063)
  cmake_policy(SET CMP0063 NEW)
endif()
find_package(Matlab COMPONENTS MX_LIBRARY BLAS_LIBRARY IOMP_LIBRARY)

# This is appended to the install prefix defined above
set(MATLAB_API_INSTALL_DIR matlab)

#------------------------------------------------------------------------------
# Find a BLAS library
#------------------------------------------------------------------------------
# BLAS libraries from various vendors (e.g. Atlas or Accelerate)
find_package(BLAS)

# Intel MKL
#find_package(MKL)

# Select a BLAS library and set the corresponding compiler/linker flags
include(SelectBLAS)


###############################################################################
#
# SDCA Library Code
#
###############################################################################

add_subdirectory(src)
