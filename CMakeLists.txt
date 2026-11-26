cmake_minimum_required(VERSION 3.12)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" libsdca_VERSION LIMIT_COUNT 1)
project(libsdca VERSION ${libsdca_VERSION})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS VERSION)
add_definitions(-DSDCA_VERSION="${libsdca_VERSION}")

# Whether to use <chrono>
#   TRUE    use std::chrono::high_resolution_clock to measure wall-clock time
#   FALSE   use gettimeofday from <sys/time.h>
option(USE_CHRONO "Use the standard C++ clock instead of POSIX gettimeofday" ON)
option(BUILD_TESTING "Build the bundled GoogleTest unit suites" OFF)
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
  # Generic SIMD lanes intentionally split on narrower targets. GCC LTO emits
  # its optional vector-performance diagnostic even for code guarded by isa.h.
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
# BLAS Libraries
#------------------------------------------------------------------------------
include(SelectBLAS)

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
  # Our compiler, without our warning flags or CXXFLAGS: -w silences the vendored code
  # (GCC 13 -Wmaybe-uninitialized in gtest-death-test.cc), so -Werror builds stay ours.
  CMAKE_ARGS
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_CXX_FLAGS=-w"
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

#------------------------------------------------------------------------------
# Interfaces
#------------------------------------------------------------------------------
# The solver and loss rows run on std::thread workers (sdca/utility/parallel.h).
find_package(Threads REQUIRED)

install(DIRECTORY include/ DESTINATION include)
install(FILES LICENSE DESTINATION share/licenses/libsdca)

# Consumers can link this header-only target instead of reproducing the
# selected backend, include directories, and numerical compile settings.
add_library(libsdca INTERFACE)
add_library(libsdca::libsdca ALIAS libsdca)
target_compile_features(libsdca INTERFACE cxx_std_11)
target_include_directories(libsdca INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_definitions(libsdca INTERFACE ${SDCA_BLAS_DEFINITIONS})
if(USE_CHRONO)
  target_compile_definitions(libsdca INTERFACE SDCA_USE_CHRONO)
else()
  target_compile_definitions(libsdca INTERFACE SDCA_USE_POSIX_CLOCK)
endif()
# IEEE semantics (the flags of src/native_build.txt that consumers keep): fast math would fold
# the finiteness checks of the inputs.
set(SDCA_IEEE_FLAGS ${SDCA_COMPILE_FLAGS})
list(FILTER SDCA_IEEE_FLAGS INCLUDE REGEX "fast-math|fp-contract")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(libsdca INTERFACE ${SDCA_IEEE_FLAGS})
endif()
target_link_libraries(libsdca INTERFACE ${BLAS_LIBRARIES} Threads::Threads)
install(TARGETS libsdca EXPORT libsdcaTargets)

# The C ABI (include/sdca/c_api.h) as libsdca_c, for C and other languages. One translation
# unit: sdca/kernel.h compiles every instruction set of the target under target attributes
# and picks one at run time.
if(SDCA_BUILD_C_API)
  add_library(sdca_c_objects OBJECT src/c_api.cpp)
  target_link_libraries(sdca_c_objects PRIVATE libsdca)
  target_compile_options(sdca_c_objects PRIVATE ${SDCA_COMPILE_FLAGS})
  set_target_properties(sdca_c_objects PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endif()
if(SDCA_BUILD_C_API)
  add_library(sdca_c SHARED $<TARGET_OBJECTS:sdca_c_objects>)
  add_library(libsdca::sdca_c ALIAS sdca_c)
  target_link_libraries(sdca_c PRIVATE libsdca Threads::Threads)
  target_include_directories(sdca_c INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
  # The shared library version follows SDCA_ABI_VERSION.
  set_target_properties(sdca_c PROPERTIES OUTPUT_NAME sdca_c VERSION 7 SOVERSION 7
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
