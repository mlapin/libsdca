cmake_minimum_required(VERSION 2.8.9)
project(libsdca)

set(libsdca_MAJOR_VERSION 0)
set(libsdca_MINOR_VERSION 3)
set(libsdca_PATCH_VERSION 0)
set(libsdca_VERSION
  ${libsdca_MAJOR_VERSION}.${libsdca_MINOR_VERSION}.${libsdca_PATCH_VERSION}
  )
add_definitions(-DSDCA_VERSION="${libsdca_VERSION}")


###############################################################################
#
# Build Configuration
#
###############################################################################

#------------------------------------------------------------------------------
# Compiler Flags
#------------------------------------------------------------------------------
# Select Release as the default build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Warning flags
set(WARN_FLAGS "-pedantic -Werror -W -Wall -Wextra -Wwrite-strings")
set(WARN_FLAGS "${WARN_FLAGS} -Wcast-qual -Wcast-align -Wshadow")
#set(WARN_FLAGS "${WARN_FLAGS} -Wconversion") # google test fails with this
set(WARN_FLAGS "${WARN_FLAGS} -Wpointer-arith")
if(CMAKE_COMPILER_IS_GNUCXX)
  set(WARN_FLAGS "${WARN_FLAGS} -Wdouble-promotion")
endif()

# Debug flags
set(DEBUG_FLAGS "-fstack-protector-all")

# Release flags
set(RELEASE_FLAGS "-march=native")
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
# Intel MKL configuration (not applicable if another BLAS is used)
#------------------------------------------------------------------------------
# Whether to use 64 or 32 bit integers (default: TRUE)
set(USE_ILP64 TRUE)

# Whether to use sequential or multi-threaded Intel MKL layer (default: FALSE)
set(USE_SEQUENTIAL FALSE)

# Whether to add a suffix (e.g. st/mt) for single-/multi-threaded versions
set(USE_THREADING_SUFFIX FALSE)

#------------------------------------------------------------------------------
# Auxiliary Cmake Configuration
#------------------------------------------------------------------------------
# Custom Cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
    "${CMAKE_SOURCE_DIR}" CACHE PATH "Installation directory." FORCE
    )
endif()

# Avoid a warning due to CMP0042
if(POLICY CMP0042 AND APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()

# RPATH settings
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "\\\$ORIGIN")


###############################################################################
#
# Code Dependencies and Third Party Libraries
#
###############################################################################

#------------------------------------------------------------------------------
# BLAS Libraries
#------------------------------------------------------------------------------
# BLAS libraries from various vendors (e.g. Atlas or Accelerate)
find_package(BLAS)

# Intel MKL
find_package(MKL)

# Select a BLAS library and set the corresponding compiler/linker flags
include(SelectBLAS)

set(CMAKE_SHARED_LINKER_FLAGS
  "${CMAKE_SHARED_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}"
  )

#------------------------------------------------------------------------------
# Google Test Framework
#------------------------------------------------------------------------------
set(GOOGLETEST_PATH
  ${CMAKE_SOURCE_DIR}/vendor/googletest/googletest
  )

include_directories(
  ${GOOGLETEST_PATH}
  ${GOOGLETEST_PATH}/include
  )

add_library(googletest ${GOOGLETEST_PATH}/src/gtest-all.cc)


###############################################################################
#
# SDCA Library
#
###############################################################################

include_directories(include)

set(libsdca_INCLUDE_PATH
  ${CMAKE_SOURCE_DIR}/include/sdca
  )

set(libsdca_MATH_SOURCES
  ${libsdca_INCLUDE_PATH}/math/blas.h
  ${libsdca_INCLUDE_PATH}/math/lambert.h
  ${libsdca_INCLUDE_PATH}/math/log_sum_exp.h
  )

set(libsdca_PROX_SOURCES
  ${libsdca_INCLUDE_PATH}/prox/prox.h
  ${libsdca_INCLUDE_PATH}/prox/proxdef.h
  ${libsdca_INCLUDE_PATH}/prox/entropy.h
  ${libsdca_INCLUDE_PATH}/prox/knapsack_eq.h
  ${libsdca_INCLUDE_PATH}/prox/knapsack_le.h
  ${libsdca_INCLUDE_PATH}/prox/knapsack_le_biased.h
  ${libsdca_INCLUDE_PATH}/prox/topk_cone.h
  ${libsdca_INCLUDE_PATH}/prox/topk_cone_biased.h
  ${libsdca_INCLUDE_PATH}/prox/topk_entropy.h
  ${libsdca_INCLUDE_PATH}/prox/topk_entropy_biased.h
  ${libsdca_INCLUDE_PATH}/prox/topk_simplex.h
  ${libsdca_INCLUDE_PATH}/prox/topk_simplex_biased.h
)

set(libsdca_UTIL_SOURCES
  ${libsdca_INCLUDE_PATH}/util/logging.h
  ${libsdca_INCLUDE_PATH}/util/stopwatch.h
  )

#------------------------------------------------------------------------------
# Unit tests
#------------------------------------------------------------------------------
add_subdirectory(test)

#------------------------------------------------------------------------------
# Matlab Interface
#------------------------------------------------------------------------------
# Honor the visibility properties set for targets of type OBJECT
# (visibility is set to 'hidden' for mex files, see FindMatlab.cmake)
if(POLICY CMP0063)
  cmake_policy(SET CMP0063 NEW)
endif()
#find_package(Matlab COMPONENTS MX_LIBRARY BLAS_LIBRARY IOMP_LIBRARY)

# Whether to use the std::chrono namespace in the stopwatch
#add_definitions(-DUSE_CHRONO=1)
