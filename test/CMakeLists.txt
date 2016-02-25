include_directories(.)

#add_executable(
#  test_libsdca_prox
#  ${GOOGLETEST_PATH}/src/gtest_main.cc
#  test_util.h
#  prox/lambert.cpp
#  prox/log_exp.cpp
#  prox/entropy.cpp
#  prox/entropy_norm.cpp
#  prox/knapsack_eq.cpp
#  prox/knapsack_le.cpp
#  prox/knapsack_le_biased.cpp
#  prox/topk_cone.cpp
#  prox/topk_cone_biased.cpp
#  prox/topk_entropy.cpp
#  prox/topk_entropy_biased.cpp
#  prox/topk_simplex.cpp
#  prox/topk_simplex_biased.cpp
#  ${libsdca_MATH_SOURCES}
#  ${libsdca_PROX_SOURCES}
#  ${libsdca_UTILITY_SOURCES}
#  )

add_executable(
  test_libsdca_solver
  ${GOOGLETEST_PATH}/src/gtest_main.cc
  test_util.h
  solver/dataset.cpp
  solver/context.cpp
#  solver/dual_solver.cpp
  ${libsdca_MATH_SOURCES}
  ${libsdca_PROX_SOURCES}
  ${libsdca_SOLVER_SOURCES}
  ${libsdca_UTILITY_SOURCES}
  )

#target_link_libraries(test_libsdca_prox googletest)
target_link_libraries(test_libsdca_solver googletest ${BLAS_LIBRARIES})

enable_testing()

#add_test(test_libsdca_prox ${PROJECT_BINARY_DIR}/test_libsdca_prox)
add_test(test_libsdca_solver ${PROJECT_BINARY_DIR}/test_libsdca_solver)
