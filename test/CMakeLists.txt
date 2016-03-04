# Update build flags for unit tests
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-global-constructors")
endif()

include_directories(.)

add_executable(
  test_libsdca_prox
  test_main.cpp
  test_util.h
  prox/lambert.cpp
  prox/log_exp.cpp
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
  prox/two_simplex.cpp
  ${libsdca_MATH_SOURCES}
  ${libsdca_PROX_SOURCES}
  ${libsdca_UTILITY_SOURCES}
  )

add_executable(
  test_libsdca_solver
  test_main.cpp
  test_util.h
  solver/objective.cpp
  solver/dataset.cpp
  solver/context.cpp
#  solver/dual_solver.cpp
  ${libsdca_MATH_SOURCES}
  ${libsdca_PROX_SOURCES}
  ${libsdca_SOLVER_SOURCES}
  ${libsdca_UTILITY_SOURCES}
  )

target_link_libraries(test_libsdca_prox gtest)
target_link_libraries(test_libsdca_solver gtest ${BLAS_LIBRARIES})

enable_testing()

add_test(test_libsdca_prox ${PROJECT_BINARY_DIR}/test_libsdca_prox)
add_test(test_libsdca_solver ${PROJECT_BINARY_DIR}/test_libsdca_solver)
