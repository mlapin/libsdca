#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "linalg/numeric.h"

using namespace sdca;

template <typename T>
void display(const std::vector<T> &v) {
  if (v.size() < 100) {
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
  }
}

int main() {
  using RealType = float;
  using ResultType = float;
  using IndexType = std::vector<RealType>::difference_type;
  using SizeType = std::vector<RealType>::size_type;

  SizeType dim = 100000;
  SizeType repeat = 1000;

  std::mt19937 gen(0);
  std::uniform_int_distribution<SizeType> dice(1, 3);
  std::normal_distribution<RealType> dist1(0, static_cast<RealType>(1e-5));
  std::normal_distribution<RealType> dist2(0, static_cast<RealType>(1e+5));

  std::vector<RealType> v;
  v.reserve(dim);
  for (SizeType i = 0; i < dim; ++i) {
    if (dice(gen) > 1) {
      v.push_back(dist1(gen));
    } else {
      v.push_back(dist2(gen));
    }
  }

  display(v);

  ResultType sum1(0), sum2(0);
  double sum3(0), sum4(0);

  double et1(0), et2(0), et3(0), et4(0);
  std::clock_t start;

  for (SizeType i = 0; i < repeat; ++i) {
    // Work 1
    start = std::clock();
    sum1 = std::accumulate(v.begin(), v.end(), sum1);
    et1 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

    // Work 2
    start = std::clock();
    sum2 = kahan_accumulate(v.begin(), v.end(), sum2);
    et2 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

    // Work 3
    start = std::clock();
    sum3 = std::accumulate(v.begin(), v.end(), sum3);
    et3 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

    // Work 4
    start = std::clock();
    sum4 = kahan_accumulate(v.begin(), v.end(), sum4);
    et4 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;
  }

  std::cout << std::scientific << std::setprecision(15);
  std::cout << "sum1 = " << sum1 << std::endl;
  std::cout << "sum2 = " << sum2 << std::endl;
  std::cout << "sum3 = " << sum3 << std::endl;
  std::cout << "sum4 = " << sum4 << std::endl;

  std::cout << "time (work 1) = " << et1 << std::endl;
  std::cout << "time (work 2) = " << et2 << std::endl;
  std::cout << "time (work 3) = " << et3 << std::endl;
  std::cout << "time (work 4) = " << et4 << std::endl;

//  std::cout << "sizeof(int) = " << sizeof(int) << std::endl;
//  std::cout << "sizeof(long int) = " << sizeof(long int) << std::endl;
//  std::cout << "sizeof(long unsigned int) = " << sizeof(long unsigned int) << std::endl;
//  std::cout << "sizeof(std::size_t) = " << sizeof(std::size_t) << std::endl;
//  std::cout << "sizeof(std::vector<double>::size_type) = " << sizeof(std::vector<double>::size_type) << std::endl;
//  std::cout << "sizeof(std::vector<float>::size_type) = " << sizeof(std::vector<float>::size_type) << std::endl;
//  std::cout << "sizeof(IndexType) = " << sizeof(IndexType) << std::endl;
//#ifdef MKL_INT
//  std::cout << "sizeof(MKL_INT) = " << sizeof(MKL_INT) << std::endl;
//#endif

//  long double s, t = 3.0;
//  s = 1.0 - (4.0/t - 1.0)*t;
//  std::cout << "s = " << s << std::endl;

  return 0;
}
