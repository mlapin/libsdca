cmake_minimum_required(VERSION 3.12)
project(libsdca)

set(libsdca_MAJOR_VERSION 0)
set(libsdca_MINOR_VERSION 3)
set(libsdca_PATCH_VERSION 1)
set(libsdca_VERSION
  ${libsdca_MAJOR_VERSION}.${libsdca_MINOR_VERSION}.${libsdca_PATCH_VERSION}
  )
add_definitions(-DSDCA_VERSION="${libsdca_VERSION}")

# Whether to use <chrono>
#   TRUE    use std::chrono::high_resolution_clock to measure wall-clock time
#   FALSE   use gettimeofday from <sys/time.h>
option(USE_CHRONO "Use the standard C++ clock instead of POSIX gettimeofday" ON)
option(BUILD_TESTING "Build the bundled GoogleTest unit suites" OFF)
option(SDCA_BUILD_MATLAB "Build MATLAB MEX interfaces" OFF)
option(SDCA_WARNINGS_AS_ERRORS "Treat project compiler warnings as errors" OFF)
option(SDCA_BUILD_C_API "Build and install the C ABI shared library" OFF)
set(SDCA_BLAS_BACKEND "AUTO" CACHE STRING "BLAS backend: AUTO, EXTERNAL, or BUILTIN")
set_property(CACHE SDCA_BLAS_BACKEND PROPERTY STRINGS AUTO EXTERNAL BUILTIN)

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
set(WARN_FLAGS "-pedantic -W -Wall -Wextra")
if(SDCA_WARNINGS_AS_ERRORS)
  set(WARN_FLAGS "${WARN_FLAGS} -Werror")
endif()
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  # A fixed set rather than -Weverything, whose contents change with every
  # release and would break -Werror builds on newer compilers.
  set(WARN_FLAGS "${WARN_FLAGS} -Wshadow")
  set(WARN_FLAGS "${WARN_FLAGS} -Wpointer-arith")
  set(WARN_FLAGS "${WARN_FLAGS} -Wcast-qual")
  set(WARN_FLAGS "${WARN_FLAGS} -Wwrite-strings")
  set(WARN_FLAGS "${WARN_FLAGS} -Wdouble-promotion")
  set(WARN_FLAGS "${WARN_FLAGS} -Wreserved-identifier")
  set(WARN_FLAGS "${WARN_FLAGS} -Wunreachable-code")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(WARN_FLAGS "${WARN_FLAGS} -Wparentheses")
  set(WARN_FLAGS "${WARN_FLAGS} -Wshadow")
  set(WARN_FLAGS "${WARN_FLAGS} -Wpointer-arith")
  set(WARN_FLAGS "${WARN_FLAGS} -Wcast-qual")
  set(WARN_FLAGS "${WARN_FLAGS} -Wcast-align")
  set(WARN_FLAGS "${WARN_FLAGS} -Wwrite-strings")
  set(WARN_FLAGS "${WARN_FLAGS} -Wmissing-declarations")
  set(WARN_FLAGS "${WARN_FLAGS} -Wredundant-decls")
  set(WARN_FLAGS "${WARN_FLAGS} -Wstack-protector")
  set(WARN_FLAGS "${WARN_FLAGS} -Wuseless-cast")
  set(WARN_FLAGS "${WARN_FLAGS} -Wdouble-promotion")
  set(WARN_FLAGS "${WARN_FLAGS} -Wvector-operation-performance")
  set(WARN_FLAGS "${WARN_FLAGS} -Wlogical-op")
endif()
# The tests mix integer and floating-point types freely.
set(WARN_FLAGS "${WARN_FLAGS} -Wno-conversion")

# Debug flags
set(DEBUG_FLAGS "-fstack-protector-all")

# Release flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(RELEASE_FLAGS "-mtune=generic")
  set(RELEASE_FLAGS "${RELEASE_FLAGS} -funsafe-loop-optimizations")
  set(RELEASE_FLAGS "${RELEASE_FLAGS} -flto -flto-partition=none")
  set(RELEASE_FLAGS "${RELEASE_FLAGS} -fno-fat-lto-objects")
  set(RELEASE_FLAGS "${RELEASE_FLAGS} -fuse-linker-plugin")
endif()

#------------------------------------------------------------------------------
# Intel MKL configuration (not applicable if another BLAS is used)
#------------------------------------------------------------------------------
# Whether to use 64 or 32 bit integers
option(USE_ILP64 "Use 64-bit integers with MKL" ON)

# Whether to use the sequential or multi-threaded Intel MKL layer
option(USE_SEQUENTIAL "Use sequential MKL instead of its threaded layer" ON)

# Whether to add a suffix (e.g. st/mt) for single-/multi-threaded versions
option(USE_THREADING_SUFFIX "Append the MKL threading mode to binary names" OFF)


#------------------------------------------------------------------------------
# Auxiliary Cmake Configuration
#------------------------------------------------------------------------------
# Custom Cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

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
# Matlab Libraries
#------------------------------------------------------------------------------
# Honor the visibility properties set for targets of type OBJECT
# (visibility is set to 'hidden' for mex files, see FindMatlab.cmake)
if(POLICY CMP0063)
  cmake_policy(SET CMP0063 NEW)
endif()
if(SDCA_BUILD_MATLAB)
  find_package(Matlab REQUIRED COMPONENTS MX_LIBRARY)
endif()

#------------------------------------------------------------------------------
# BLAS Libraries
#------------------------------------------------------------------------------
include(SelectBLAS)
add_compile_definitions(${SDCA_BLAS_DEFINITIONS})
include_directories(${SDCA_BLAS_INCLUDE_DIRS})

set(CMAKE_SHARED_LINKER_FLAGS
  "${CMAKE_SHARED_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}"
  )

#------------------------------------------------------------------------------
# Google Test Framework
#------------------------------------------------------------------------------

# At this point, we only need googletest, not googlemock
if(BUILD_TESTING)
set(GOOGLETEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vendor/googletest/googletest)

include(ExternalProject)
find_package(Threads REQUIRED)
ExternalProject_Add(googletest
  SOURCE_DIR "${GOOGLETEST_PATH}"
  BINARY_DIR "${CMAKE_BINARY_DIR}/googletest-build"
  PREFIX "${CMAKE_BINARY_DIR}/googletest"
  INSTALL_COMMAND   ""
  # Ninja needs to know which step produces the imported library below.
  BUILD_BYPRODUCTS "${CMAKE_BINARY_DIR}/googletest-build/libgtest.a"
  )

ExternalProject_Get_Property(googletest source_dir binary_dir)
add_library(gtest UNKNOWN IMPORTED)
set_target_properties(gtest PROPERTIES
  IMPORTED_LOCATION ${binary_dir}/libgtest.a
  IMPORTED_LINK_INTERFACE_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}"
  INTERFACE_INCLUDE_DIRECTORIES "${source_dir}/include"
)

add_dependencies(gtest googletest)
endif()


###############################################################################
#
# SDCA Library
#
###############################################################################

if(${USE_CHRONO})
  # If defined, std::chrono::high_resolution_clock is used in stopwatch
  add_definitions(-DSDCA_USE_CHRONO)
else()
  add_definitions(-DSDCA_USE_POSIX_CLOCK)
endif()


# Set SDCA specific flags after 3rd party code is built
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -std=c++11 ${WARN_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${RELEASE_FLAGS}")

include_directories(include)

set(libsdca_INCLUDE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/include/sdca
  )

set(libsdca_MATH_SOURCES
  ${libsdca_INCLUDE_PATH}/math/entropy.h
  ${libsdca_INCLUDE_PATH}/math/blas.h
  ${libsdca_INCLUDE_PATH}/math/blas_builtin.h
  ${libsdca_INCLUDE_PATH}/math/compensated.h
  ${libsdca_INCLUDE_PATH}/math/functor.h
  ${libsdca_INCLUDE_PATH}/math/lambert.h
  ${libsdca_INCLUDE_PATH}/math/log_exp.h
  )

set(libsdca_PROX_SOURCES
  ${libsdca_INCLUDE_PATH}/prox.h
  ${libsdca_INCLUDE_PATH}/prox/proxdef.h
  ${libsdca_INCLUDE_PATH}/prox/entropy.h
  ${libsdca_INCLUDE_PATH}/prox/entropy_norm.h
  ${libsdca_INCLUDE_PATH}/prox/knapsack_eq.h
  ${libsdca_INCLUDE_PATH}/prox/knapsack_le.h
  ${libsdca_INCLUDE_PATH}/prox/knapsack_le_biased.h
  ${libsdca_INCLUDE_PATH}/prox/topk_cone.h
  ${libsdca_INCLUDE_PATH}/prox/topk_cone_biased.h
  ${libsdca_INCLUDE_PATH}/prox/topk_entropy.h
  ${libsdca_INCLUDE_PATH}/prox/topk_entropy_biased.h
  ${libsdca_INCLUDE_PATH}/prox/topk_simplex.h
  ${libsdca_INCLUDE_PATH}/prox/topk_simplex_biased.h
  ${libsdca_INCLUDE_PATH}/prox/two_entropy.h
  ${libsdca_INCLUDE_PATH}/prox/two_simplex.h
  ${libsdca_INCLUDE_PATH}/prox/two_simplex_sort.h
)

set(libsdca_SOLVER_SOURCES
  ${libsdca_INCLUDE_PATH}/loss.h
  ${libsdca_INCLUDE_PATH}/solver.h
  ${libsdca_INCLUDE_PATH}/solver/solverdef.h
  ${libsdca_INCLUDE_PATH}/solver/context.h
  ${libsdca_INCLUDE_PATH}/solver/data/dataset.h
  ${libsdca_INCLUDE_PATH}/solver/data/input.h
  ${libsdca_INCLUDE_PATH}/solver/data/output.h
  ${libsdca_INCLUDE_PATH}/solver/data/scratch.h
  ${libsdca_INCLUDE_PATH}/solver/data.h
  ${libsdca_INCLUDE_PATH}/solver/eval/accumulate.h
  ${libsdca_INCLUDE_PATH}/solver/eval/core.h
  ${libsdca_INCLUDE_PATH}/solver/eval/dual.h
  ${libsdca_INCLUDE_PATH}/solver/eval/evaluate.h
  ${libsdca_INCLUDE_PATH}/solver/eval/primal.h
  ${libsdca_INCLUDE_PATH}/solver/eval/scores.h
  ${libsdca_INCLUDE_PATH}/solver/eval/types.h
  ${libsdca_INCLUDE_PATH}/solver/eval.h
  ${libsdca_INCLUDE_PATH}/solver/objective/feasible_dual.h
  ${libsdca_INCLUDE_PATH}/solver/objective/l2_entropy.h
  ${libsdca_INCLUDE_PATH}/solver/objective/l2_entropy_topk.h
  ${libsdca_INCLUDE_PATH}/solver/objective/l2_hinge_topk.h
  ${libsdca_INCLUDE_PATH}/solver/objective/l2_multilabel_entropy.h
  ${libsdca_INCLUDE_PATH}/solver/objective/l2_multilabel_hinge.h
  ${libsdca_INCLUDE_PATH}/solver/objective/l2_topk_hinge.h
  ${libsdca_INCLUDE_PATH}/solver/objective/objective_base.h
  ${libsdca_INCLUDE_PATH}/solver/objective.h
  ${libsdca_INCLUDE_PATH}/solver/reporting.h
  ${libsdca_INCLUDE_PATH}/solver/update.h
  )

set(libsdca_UTILITY_SOURCES
  ${libsdca_INCLUDE_PATH}/utility/fpenv.h
  ${libsdca_INCLUDE_PATH}/utility/logging.h
  ${libsdca_INCLUDE_PATH}/utility/parallel.h
  ${libsdca_INCLUDE_PATH}/utility/scratch.h
  ${libsdca_INCLUDE_PATH}/utility/stopwatch.h
  ${libsdca_INCLUDE_PATH}/utility/types.h
  )


#------------------------------------------------------------------------------
# Interfaces
#------------------------------------------------------------------------------
set(SDCA_BINARY_SUFFIX)
if(${USE_THREADING_SUFFIX})
  if(${USE_SEQUENTIAL})
    set(SDCA_BINARY_SUFFIX _st)
  else()
    set(SDCA_BINARY_SUFFIX _mt)
  endif()
endif()

set(SDCA_MATLAB_PREFIX matsdca_)
set(SDCA_MATLAB_INSTALL_DIR matlab)

# The solver and loss rows run on std::thread workers (sdca/utility/parallel.h).
find_package(Threads REQUIRED)

add_subdirectory(src)
install(DIRECTORY include/ DESTINATION include)
install(FILES LICENSE DESTINATION share/licenses/libsdca)

# Consumers can link this header-only target instead of reproducing the
# selected backend, include directories, and numerical compile settings.
add_library(libsdca INTERFACE)
add_library(libsdca::libsdca ALIAS libsdca)
target_compile_features(libsdca INTERFACE cxx_std_11)
target_include_directories(libsdca INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include> ${SDCA_BLAS_INCLUDE_DIRS})
target_compile_definitions(libsdca INTERFACE ${SDCA_BLAS_DEFINITIONS}
  SDCA_VERSION="${libsdca_VERSION}")
if(USE_CHRONO)
  target_compile_definitions(libsdca INTERFACE SDCA_USE_CHRONO)
else()
  target_compile_definitions(libsdca INTERFACE SDCA_USE_POSIX_CLOCK)
endif()
# IEEE semantics: fast math would fold the finiteness checks of the inputs.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(libsdca INTERFACE -fno-fast-math -ffp-contract=off)
endif()
target_link_libraries(libsdca INTERFACE ${BLAS_LIBRARIES} Threads::Threads)
install(TARGETS libsdca EXPORT libsdcaTargets)

# The C ABI (include/sdca/c_api.h) as libsdca_c, for C and other languages.
if(SDCA_BUILD_C_API)
  add_library(sdca_c SHARED src/c_api.cpp)
  add_library(libsdca::sdca_c ALIAS sdca_c)
  target_link_libraries(sdca_c PRIVATE libsdca Threads::Threads)
  target_include_directories(sdca_c INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
  # The shared library version follows SDCA_ABI_VERSION.
  set_target_properties(sdca_c PROPERTIES OUTPUT_NAME sdca_c VERSION 5 SOVERSION 5
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
  install(TARGETS sdca_c EXPORT libsdcaTargets
    LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
endif()
install(EXPORT libsdcaTargets NAMESPACE libsdca:: DESTINATION lib/cmake/libsdca)
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/libsdcaConfigVersion.cmake"
  VERSION ${libsdca_VERSION} COMPATIBILITY SameMinorVersion)
install(FILES cmake/libsdcaConfig.cmake "${CMAKE_CURRENT_BINARY_DIR}/libsdcaConfigVersion.cmake"
  DESTINATION lib/cmake/libsdca)


#------------------------------------------------------------------------------
# Unit tests
#------------------------------------------------------------------------------
if(BUILD_TESTING)
enable_testing()
add_subdirectory(tests/cpp)
endif()
