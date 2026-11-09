#!/usr/bin/env bash
# Build and run the C++ suites directly: run_cpp_tests.sh [--sanitize]
# (--sanitize runs the regression tests under ASan and UBSan, the threaded ones under TSan).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build/cpp-tests
blas="${SDCA_BLAS_LIBRARY:--l:libblas.so.3}"
if [[ "$blas" == /* ]]; then
  export LD_LIBRARY_PATH="$(dirname "$blas")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
compiler_flags=(-std=c++11 -DBLAS_DEFAULT_LOCAL_HEADER -DSDCA_VERSION='"0.3.0-test"'
  -Iinclude -Itests/cpp -Ivendor/googletest/googletest/include -Ivendor/googletest/googletest)
if [[ "${1:-}" == "--sanitize" ]]; then
  g++ "${compiler_flags[@]}" -O1 -g -fno-omit-frame-pointer \
    -fsanitize=address,undefined,float-cast-overflow -fno-sanitize-recover=all \
    tests/cpp/test_main.cpp tests/cpp/regression.cpp vendor/googletest/googletest/src/gtest-all.cc \
    -pthread "$blas" -o build/cpp-tests/test_sanitize
  ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" OPENBLAS_NUM_THREADS=1 build/cpp-tests/test_sanitize
  gcc -std=c99 -Iinclude -fsanitize=address,undefined -fno-sanitize-recover=all \
    -c tests/cpp/c_api_validation.c -o build/cpp-tests/c_api_validation.o
  g++ "${compiler_flags[@]}" -O1 -g -fno-omit-frame-pointer \
    -fsanitize=address,undefined,float-cast-overflow -fno-sanitize-recover=all \
    src/c_api.cpp build/cpp-tests/c_api_validation.o \
    -pthread "$blas" -o build/cpp-tests/test_c_api_sanitize
  ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" OPENBLAS_NUM_THREADS=1 build/cpp-tests/test_c_api_sanitize
  # ThreadSanitizer by Clang (SDCA_TSAN_CXX): GCC 11's runtime does not intercept the
  # pthread_cond_clockwait of condition_variable::wait_for and reports false races.
  tsan="${SDCA_TSAN_CXX:-clang++}"
  tsan_flags=(-O1 -g -fno-omit-frame-pointer -fsanitize=thread)
  "$tsan" "${compiler_flags[@]}" "${tsan_flags[@]}" \
    tests/cpp/test_main.cpp tests/cpp/regression.cpp tests/cpp/solver/*.cpp \
    vendor/googletest/googletest/src/gtest-all.cc -pthread "$blas" -o build/cpp-tests/test_tsan
  "$tsan" "${compiler_flags[@]}" "${tsan_flags[@]}" tests/cpp/c_api_threads.cpp src/c_api.cpp \
    -pthread "$blas" -o build/cpp-tests/test_c_api_tsan
  export TSAN_OPTIONS="${TSAN_OPTIONS:-halt_on_error=1}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
  build/cpp-tests/test_tsan --gtest_filter=-ParallelEnvironmentTest.*
  for setting in "" 1 3 0 many; do
    SDCA_NUM_THREADS="$setting" build/cpp-tests/test_tsan --gtest_filter=ParallelEnvironmentTest.*
  done
  SDCA_NUM_THREADS= OMP_NUM_THREADS=1 build/cpp-tests/test_tsan --gtest_filter=ParallelEnvironmentTest.*
  build/cpp-tests/test_c_api_tsan
else
  g++ "${compiler_flags[@]}" -O2 -fno-fast-math \
    tests/cpp/test_main.cpp tests/cpp/regression.cpp tests/cpp/prox/*.cpp tests/cpp/solver/*.cpp \
    vendor/googletest/googletest/src/gtest-all.cc \
    -pthread "$blas" -o build/cpp-tests/test_all
  OPENBLAS_NUM_THREADS=1 build/cpp-tests/test_all --gtest_filter=-ParallelEnvironmentTest.*
fi
