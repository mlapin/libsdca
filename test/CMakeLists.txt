include_directories(.)

add_executable(
  test_libsdca
  ${GOOGLETEST_PATH}/src/gtest_main.cc
  test_util.h
  test_lambert.cpp
  test_log_sum_exp.cpp
  test_prox_knapsack_eq.cpp
  test_prox_knapsack_le.cpp
  test_prox_knapsack_le_biased.cpp
  test_prox_topk_cone.cpp
  test_prox_topk_cone_biased.cpp
  test_prox_topk_simplex.cpp
  test_prox_topk_simplex_biased.cpp
  ${libsdca_MATH_SOURCES}
  ${libsdca_PROX_SOURCES}
  ${libsdca_UTIL_SOURCES}
  )

target_link_libraries(
  test_libsdca
  googletest
  )

enable_testing()

add_test(test_libsdca ${PROJECT_BINARY_DIR}/test_libsdca)
