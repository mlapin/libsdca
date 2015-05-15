cmake_minimum_required(VERSION 2.8)
project(libsdca)

# see also
# https://www.gnu.org/software/gsl/manual/html_node/GCC-warning-options-for-numerical-programs.html
#
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Werror -W -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wcast-qual -Wcast-align")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdouble-promotion -Wconversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith -Wwrite-strings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -fshort-enums -fno-common")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -m64")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -DDEBUG")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH
    "Install directory used by install." FORCE)
endif()

# Search for dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
find_package(Matlab COMPONENTS BLAS_LIBRARY)
find_package(BLAS)
find_package(MKL)
include(UseBLAS)

# Math API
include_directories(math)

# Projections lib
set(LIB_PROJECTORS projectors)
include_directories(projectors)
add_subdirectory(projectors)

# Matlab interface
if(Matlab_FOUND)
  add_subdirectory(matlab)
else(Matlab_FOUND)
  message(WARNING "Matlab not found. Matlab interface will not be compiled.")
endif(Matlab_FOUND)

# Test
# add_subdirectory(main)
