#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "math_util.hpp"
#include <mkl_blas.h>

using namespace sdca;


template <typename T = double>
void display_vector(const std::vector<T> &v) {
  std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

int main() {
  std::mt19937 gen(0);
  std::normal_distribution<> d(0,1);

  IndexType m = 400;
  IndexType n = 2000;
  std::vector<double> A, Ki, scores;
  A.reserve(m*n);
  for (IndexType i = 0; i < m*n; ++i) {
    A.push_back(d(gen));
  }
  Ki.reserve(n);
  for (IndexType i = 0; i < n; ++i) {
    Ki.push_back(d(gen));
  }
  scores.resize(m);

  double elapsedTime1(0), elapsedTime2(0);
  const char *noTrans = "N";
  const char *trans = "T";
  double alpha = 1;
  double beta = 0;
  const IndexType kIndexOne = static_cast<IndexType>(1);

  for (IndexType j = 0; j < 20; ++j) {

    std::clock_t start = std::clock();

    for (IndexType i = 0; i < n; ++i) {
      dgemv(noTrans, &m, &n, &alpha, &A[0], &m, &Ki[0], &kIndexOne,
           &beta, &scores[0], &kIndexOne);
    }

    elapsedTime1 += static_cast<double>(std::clock() - start) /
      CLOCKS_PER_SEC;

    start = std::clock();

    for (IndexType i = 0; i < n; ++i) {
      dgemv(trans, &n, &m, &alpha, &A[0], &n, &Ki[0], &kIndexOne,
           &beta, &scores[0], &kIndexOne);
    }

    elapsedTime2 += static_cast<double>(std::clock() - start) /
      CLOCKS_PER_SEC;
  }

  std::cout << "time (no trans) = " << elapsedTime1 << std::endl;
  std::cout << "time (trans) = " << elapsedTime1 << std::endl;
  std::cout << "sizeof(int) = " << sizeof(int) << std::endl;
  std::cout << "sizeof(long int) = " << sizeof(long int) << std::endl;
  std::cout << "sizeof(long unsigned int) = " << sizeof(long unsigned int) << std::endl;
  std::cout << "sizeof(std::size_t) = " << sizeof(std::size_t) << std::endl;
  std::cout << "sizeof(std::vector<double>::size_type) = " << sizeof(std::vector<double>::size_type) << std::endl;
  std::cout << "sizeof(std::vector<float>::size_type) = " << sizeof(std::vector<float>::size_type) << std::endl;
  std::cout << "sizeof(IndexType) = " << sizeof(IndexType) << std::endl;
#ifdef MKL_INT
  std::cout << "sizeof(MKL_INT) = " << sizeof(MKL_INT) << std::endl;
#endif

  return 0;
}
