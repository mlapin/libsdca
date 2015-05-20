#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "math_util.hpp"

using namespace sdca;


template <typename T = double>
void display_vector(const std::vector<T> &v) {
  std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

int main() {
  std::mt19937 gen(0);
  std::normal_distribution<> d(0,1);

  IndexType n = 400;
  std::vector<double> x, y;
  x.reserve(n);
  y.reserve(n);
  for (IndexType i = 0; i < n; ++i) {
    x.push_back(d(gen));
    y.push_back(d(gen));
  }

  std::clock_t start = std::clock();

  double dot = sdca_blas_dot(n, &x[0], &y[0]);

  double elapsedTime = static_cast<double>(std::clock() - start) /
    CLOCKS_PER_SEC;

  std::cout << "time = " << elapsedTime << std::endl;
  std::cout << "dot = " << dot << std::endl;
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
