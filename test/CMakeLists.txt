include_directories(.)

add_executable(
  test_libsdca
  ${GOOGLETEST_PATH}/src/gtest_main.cc
  test_util.h
  test_lambert.cpp
  test_log_sum_exp.cpp
  ${libsdca_MATH_SOURCES}
  )

target_link_libraries(
  test_libsdca
  googletest
  )

enable_testing()

add_test(test_libsdca ${PROJECT_BINARY_DIR}/test_libsdca)
