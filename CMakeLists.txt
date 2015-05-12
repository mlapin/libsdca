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

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -DDEBUG")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH
    "Install directory used by install." FORCE)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Projections lib
set(LIB_PROJECTIONS projections)
include_directories(projections)
add_subdirectory(projections)

# Matlab interface
set(MATLAB_ROOT "/usr/lib/matlab-8.4")
find_package(Matlab)
if(MATLAB_FOUND)
  add_subdirectory(matlab)
else(MATLAB_FOUND)
  message(WARNING "MATLAB_ROOT is not set or invalid. "
    "Matlab interface will not be compiled.")
endif(MATLAB_FOUND)

# Test
add_subdirectory(main)
