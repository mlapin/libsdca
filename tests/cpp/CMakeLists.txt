# Update build flags for unit tests
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-global-constructors")
endif()

include_directories(.)

add_executable(
  testsolver
  test_main.cpp
  test_util.h
  solver/context.cpp
  solver/blas.cpp
  solver/dataset.cpp
  solver/fpenv.cpp
  solver/objective.cpp
  solver/parallel.cpp
  solver/reporting.cpp
  solver/solver.cpp
  solver/surface.cpp
  solver/validation.cpp
  stress.cpp
  stress_projections.cpp
  )

target_link_libraries(testsolver gtest libsdca)

# The same suite on the built-in products, whose results are the same bits for every thread
# count and instruction set (an external BLAS rounds as it will: the suite above compares
# such runs to a tolerance).
get_target_property(SDCA_TESTSOLVER_SOURCES testsolver SOURCES)
add_executable(testsolver_builtin ${SDCA_TESTSOLVER_SOURCES})
target_include_directories(testsolver_builtin PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(testsolver_builtin PRIVATE BLAS_BUILTIN)
target_compile_options(testsolver_builtin PRIVATE ${SDCA_IEEE_FLAGS})
target_link_libraries(testsolver_builtin gtest Threads::Threads)

# The numerical regressions of the losses, projections and objectives (regression.cpp and
# regression_solver.h), through the public row losses (loss_util.h).
add_executable(testregression test_main.cpp regression.cpp)
target_link_libraries(testregression gtest libsdca)
add_test(NAME test_solver COMMAND testsolver --gtest_filter=-ParallelEnvironmentTest.*)
add_test(NAME test_regression COMMAND testregression)
add_test(NAME test_solver_builtin
  COMMAND testsolver_builtin --gtest_filter=-ParallelEnvironmentTest.*)
foreach(setting default 1 3 0 many)
  add_test(NAME test_threads_env_${setting}
    COMMAND testsolver --gtest_filter=ParallelEnvironmentTest.*)
  if(setting STREQUAL "default")
    set_property(TEST test_threads_env_${setting} PROPERTY ENVIRONMENT "SDCA_NUM_THREADS=")
  else()
    set_property(TEST test_threads_env_${setting} PROPERTY ENVIRONMENT "SDCA_NUM_THREADS=${setting}")
  endif()
endforeach()
# Automatic calls capped by OMP_NUM_THREADS=1 stay on their caller.
add_test(NAME test_threads_env_omp1 COMMAND testsolver --gtest_filter=ParallelEnvironmentTest.*)
set_property(TEST test_threads_env_omp1 PROPERTY ENVIRONMENT "SDCA_NUM_THREADS=;OMP_NUM_THREADS=1")
# Thread-placement tests time their first row: keep them off a machine busy with the shards.
set_tests_properties(test_solver test_solver_builtin test_threads_env_default test_threads_env_1 test_threads_env_3
  test_threads_env_0 test_threads_env_many test_threads_env_omp1 PROPERTIES RUN_SERIAL TRUE)

if(SDCA_BUILD_C_API)
  # Masked classes (scores of -inf) through the C++ loss functions and the C ABI.
  add_executable(test_masked_classes masked_classes.cpp)
  target_link_libraries(test_masked_classes PRIVATE sdca_c libsdca)
  add_test(NAME test_masked_classes COMMAND test_masked_classes)
  # Every entry point of the C ABI from C.
  add_executable(test_c_api_entry_points c_api_entry_points.c)
  target_link_libraries(test_c_api_entry_points PRIVATE sdca_c libsdca)
  if(NOT MSVC)
    target_link_libraries(test_c_api_entry_points PRIVATE m)
  endif()
  add_test(NAME test_c_api_entry_points COMMAND test_c_api_entry_points)
endif()

# Self-checking kernel tests against naive long double references: the lanes and the losses,
# the projections, the limits and the contract of the row kernels.
foreach(smoke ce contract distill float_rows limits projections softmax topk)
  add_executable(test_smoke_${smoke} rewrite_smoke_${smoke}.cpp)
  target_link_libraries(test_smoke_${smoke} PRIVATE libsdca)
  target_compile_options(test_smoke_${smoke} PRIVATE ${SDCA_IEEE_FLAGS})
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # The references promote to long double on purpose (80-bit on x86-64 Linux).
    target_compile_options(test_smoke_${smoke} PRIVATE -Wno-double-promotion)
  endif()
  add_test(NAME test_smoke_${smoke} COMMAND test_smoke_${smoke})
endforeach()
