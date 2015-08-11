#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "util/lambert.h"
#include "util/numeric.h"
#include "util/fmath.h"

using namespace sdca;

template <typename T>
void display(const std::vector<T> &v) {
  if (v.size() < 100) {
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
  }
}

int main() {
  using RealType = double;
  using ResultType = double;
  using IndexType = std::vector<RealType>::difference_type;
  using SizeType = std::vector<RealType>::size_type;
  using IteratorType = std::vector<RealType>::iterator;

  SizeType dim = 1000000;
  SizeType repeat = 10;

  std::mt19937 gen(0);
  std::uniform_int_distribution<SizeType> dice(1, 8);
  std::uniform_real_distribution<RealType> dist1(-1e15, -700);
  std::uniform_real_distribution<RealType> dist2(-700, -36);
  std::uniform_real_distribution<RealType> dist3(-36, -20);
  std::uniform_real_distribution<RealType> dist4(-20, -1);
  std::uniform_real_distribution<RealType> dist5(-1, 0.5);
  std::uniform_real_distribution<RealType> dist6(0.5, 2);
  std::uniform_real_distribution<RealType> dist7(2, 5.7647e+17);
  std::uniform_real_distribution<RealType> dist8(5.7647e+17, 1e+24);
//  std::normal_distribution<RealType> dist1(0, static_cast<RealType>(1e-6));
//  std::normal_distribution<RealType> dist2(0, static_cast<RealType>(1e+0));
//  std::normal_distribution<RealType> dist3(0, static_cast<RealType>(1e+6));

  std::vector<RealType> v;
  v.reserve(dim);
  for (SizeType i = 0; i < dim; ++i) {
    SizeType outcome = dice(gen);
    switch (outcome) {
      case 1: v.push_back(dist1(gen)); break;
      case 2: v.push_back(dist2(gen)); break;
      case 3: v.push_back(dist3(gen)); break;
      case 4: v.push_back(dist4(gen)); break;
      case 5: v.push_back(dist5(gen)); break;
      case 6: v.push_back(dist6(gen)); break;
      case 7: v.push_back(dist7(gen)); break;
      case 8: v.push_back(dist8(gen)); break;
    }
  }

//  display(v);

  ResultType sum1(0), sum2(0), sum3(0), sum4(0), sum5(0);//, sum6(0);
  double et1(0), et2(0), et3(0), et4(0), et5(0);//, et6(0);
  std::clock_t start;

//  std_sum<IteratorType, double> stdsum;
//  kahan_sum<IteratorType, double> kahansum;

  for (SizeType i = 0; i < repeat; ++i) {
    // Work 1
    start = std::clock();
    std::for_each(v.begin(), v.end(), [&](const RealType x){ sum1 += lambert_w_exp(x); });
    et1 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

    // Work 2
    start = std::clock();
    std::for_each(v.begin(), v.end(), [&](const RealType x){ sum2 += std::log(x); });
    et2 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

    // Work 3
    start = std::clock();
    std::for_each(v.begin(), v.end(), [&](const RealType x){ sum3 += std::exp(x); });
    et3 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

    // Work 4
    start = std::clock();
    std::for_each(v.begin(), v.end(), [&](const RealType x){ sum4 += fmath::expd(x); });
    et4 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

    // Work 5
    start = std::clock();
    std::for_each(v.begin(), v.end(), [&](const RealType x){ sum5 += exp_approx(x); });
    et5 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

//    // Work 1
//    start = std::clock();
//    sum1 = std::accumulate(v.begin(), v.end(), sum1);
//    et1 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

//    // Work 2
//    start = std::clock();
//    sum2 = kahan_accumulate(v.begin(), v.end(), sum2);
//    et2 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

//    // Work 3
//    start = std::clock();
//    sum3 = std::accumulate(v.begin(), v.end(), sum3);
//    et3 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

//    // Work 4
//    start = std::clock();
//    sum4 = kahan_accumulate(v.begin(), v.end(), sum4);
//    et4 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

//    // Work 5
//    start = std::clock();
//    sum5 = stdsum(v.begin(), v.end(), sum5);
//    et5 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

//    // Work 6
//    start = std::clock();
//    sum6 = kahansum(v.begin(), v.end(), sum6);
//    et6 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;
  }

  std::cout << std::scientific << std::setprecision(16);
  std::cout << "sum1 = " << sum1 << std::endl;
  std::cout << "sum2 = " << sum2 << std::endl;
  std::cout << "sum3 = " << sum3 << std::endl;
  std::cout << "sum4 = " << sum4 << std::endl;
  std::cout << "sum5 = " << sum5 << std::endl;
//  std::cout << "sum6 = " << sum6 << std::endl;

  std::cout << "time (work 1) = " << et1 << std::endl;
  std::cout << "time (work 2) = " << et2 << std::endl;
  std::cout << "time (work 3) = " << et3 << std::endl;
  std::cout << "time (work 4) = " << et4 << std::endl;
  std::cout << "time (work 5) = " << et5 << std::endl;
//  std::cout << "time (work 6) = " << et6 << std::endl;

//  std::cout << std::exp(709.0) << std::endl;
//  std::cout << std::exp(710.0) << std::endl;
//  std::cout << std::exp(-708.0) << std::endl;
//  std::cout << std::exp(-709.0) << std::endl;

//  std::cout << std::exp(88.0f) << std::endl;
//  std::cout << std::exp(89.0f) << std::endl;
//  std::cout << std::exp(-88.0f) << std::endl;
//  std::cout << std::exp(-89.0f) << std::endl;

  std::cout << std::scientific << std::setprecision(16);
  using Type = double;
  Type err = 0;
  for (Type x = -300; x <= 300; x += 1) {
    Type w = lambert_w_exp(x);
    if (x >= 0) {
      err = (w + std::log(w) - x) / std::max(1.0, x);
    } else {
      err = w - std::exp(x - w);
    }
    if (std::abs(err) > std::numeric_limits<Type>::epsilon()) {
      std::cout << x << ", " << w << ", " << std::log(w) << ", " << err << std::endl;
    }
//    std::cout << x << ", " << w << std::endl;
  }

  for (auto x : v) {
    Type w = lambert_w_exp(x);
    if (x >= 0) {
      err = (w + std::log(w) - x) / std::max(1.0, x);
    } else {
      err = w - std::exp(x - w);
    }
    if (std::abs(err) > 2*std::numeric_limits<Type>::epsilon()) {
      std::cout << x << ", " << w << ", " << std::log(w) << ", " << err << std::endl;
    }
//    std::cout << x << ", " << w << std::endl;
  }

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
