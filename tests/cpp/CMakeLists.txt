# Update build flags for unit tests
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-global-constructors")
endif()

include_directories(.)

add_executable(
  testprox
  test_main.cpp
  test_util.h
  prox/lambert.cpp
  prox/log_exp.cpp
  prox/partition.cpp
  prox/entropy.cpp
  prox/entropy_norm.cpp
  prox/knapsack_eq.cpp
  prox/knapsack_le.cpp
  prox/knapsack_le_biased.cpp
  prox/topk_cone.cpp
  prox/topk_cone_biased.cpp
  prox/topk_entropy.cpp
  prox/topk_entropy_biased.cpp
  prox/topk_simplex.cpp
  prox/topk_simplex_biased.cpp
  prox/two_entropy.cpp
  prox/two_simplex.cpp
  prox/two_simplex_sort.cpp
  ${libsdca_MATH_SOURCES}
  ${libsdca_PROX_SOURCES}
  ${libsdca_UTILITY_SOURCES}
  )

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
  solver/validation.cpp
  regression.cpp
  stress.cpp
  stress_projections.cpp
  ${libsdca_MATH_SOURCES}
  ${libsdca_PROX_SOURCES}
  ${libsdca_SOLVER_SOURCES}
  ${libsdca_UTILITY_SOURCES}
  )

target_link_libraries(testprox gtest)
target_link_libraries(testsolver gtest ${BLAS_LIBRARIES})
# GoogleTest shards of the projection tests, which ctest --parallel runs at once.
foreach(shard RANGE 15)
  add_test(NAME test_prox_${shard} COMMAND testprox)
  set_property(TEST test_prox_${shard}
    PROPERTY ENVIRONMENT "GTEST_TOTAL_SHARDS=16;GTEST_SHARD_INDEX=${shard}")
endforeach()
add_test(NAME test_solver COMMAND testsolver --gtest_filter=-ParallelEnvironmentTest.*)
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
set_tests_properties(test_solver test_threads_env_default test_threads_env_1 test_threads_env_3
  test_threads_env_0 test_threads_env_many test_threads_env_omp1 PROPERTIES RUN_SERIAL TRUE)

if(SDCA_BUILD_C_API)
  # A C caller with flush-to-zero set.
  add_executable(test_c_api_fpenv c_api_fpenv.c)
  target_link_libraries(test_c_api_fpenv PRIVATE sdca_c $<$<BOOL:${UNIX}>:m>)
  add_test(NAME test_c_api_fpenv COMMAND test_c_api_fpenv)
  add_executable(test_c_api_validation c_api_validation.c)
  target_link_libraries(test_c_api_validation PRIVATE sdca_c)
  add_test(NAME test_c_api_validation COMMAND test_c_api_validation)
  # The entry points on threads; built from the source, whose thread settings it sets.
  add_executable(test_c_api_threads c_api_threads.cpp ${PROJECT_SOURCE_DIR}/src/c_api.cpp)
  target_link_libraries(test_c_api_threads PRIVATE libsdca Threads::Threads)
  add_test(NAME test_c_api_threads COMMAND test_c_api_threads)
endif()
