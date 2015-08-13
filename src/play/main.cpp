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

void test_lambert_float(std::size_t dim, std::size_t repeat) {
  using RealType = float;
  using ResultType = RealType;
  using IndexType = std::vector<RealType>::difference_type;
  using SizeType = std::vector<RealType>::size_type;
  using IteratorType = std::vector<RealType>::iterator;

  std::mt19937 gen(0);
  std::uniform_int_distribution<SizeType> dice(1, 6);
  std::uniform_real_distribution<RealType> dist1(-10000, -91);
  std::uniform_real_distribution<RealType> dist2(-91, -18);
  std::uniform_real_distribution<RealType> dist3(-18, -1);
  std::uniform_real_distribution<RealType> dist4(-1, 8);
  std::uniform_real_distribution<RealType> dist5(8, 536870912);
  std::uniform_real_distribution<RealType> dist6(536870912.0f, 537870912.0f);

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
    }
  }

  ResultType sum1(0), sum2(0), sum3(0), sum4(0), sum5(0);//, sum6(0);
  double et1(0), et2(0), et3(0), et4(0), et5(0);//, et6(0);
  std::clock_t start;

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
    std::for_each(v.begin(), v.end(), [&](const RealType x){ sum4 += fmath::exp(x); });
    et4 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

    // Work 5
    start = std::clock();
    std::for_each(v.begin(), v.end(), [&](const RealType x){ sum5 += exp_approx(x); });
    et5 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;
  }

  std::cout << std::scientific << std::setprecision(16);
  std::cout << "sum1 = " << sum1 << std::endl;
  std::cout << "sum2 = " << sum2 << std::endl;
  std::cout << "sum3 = " << sum3 << std::endl;
  std::cout << "sum4 = " << sum4 << std::endl;
  std::cout << "sum5 = " << sum5 << std::endl;

  std::cout << "time (work 1) = " << et1 << std::endl;
  std::cout << "time (work 2) = " << et2 << std::endl;
  std::cout << "time (work 3) = " << et3 << std::endl;
  std::cout << "time (work 4) = " << et4 << std::endl;
  std::cout << "time (work 5) = " << et5 << std::endl;

  using Type = ResultType;
  std::cout << std::scientific << std::setprecision(16);
  std::cout << "eps = " << std::numeric_limits<Type>::epsilon() << std::endl;
  Type err = 0;
  for (Type x = -20; x <= 100; x += 0.00001f) {
    Type w = lambert_w_exp(x);
    if (x >= 0) {
      err = (w + std::log(w) - x) / std::max(static_cast<Type>(1), x);
    } else {
      err = w - std::exp(x - w);
    }
    if (std::abs(err) >= 4*std::numeric_limits<Type>::epsilon()) {
      std::cout << x << ", " << w << ", " << std::log(w) << ", " << err << std::endl;
    }
  }

  for (auto x : v) {
    Type w = lambert_w_exp(x);
    if (x >= 0) {
      err = (w + std::log(w) - x) / std::max(1.0f, x);
    } else {
      err = w - std::exp(x - w);
    }
    if (std::abs(err) >= 4*std::numeric_limits<Type>::epsilon()) {
      std::cout << x << ", " << w << ", " << std::log(w) << ", " << err << std::endl;
    }
  }
}

void test_lambert_double(std::size_t dim, std::size_t repeat) {
  using RealType = double;
  using ResultType = RealType;
  using IndexType = std::vector<RealType>::difference_type;
  using SizeType = std::vector<RealType>::size_type;
  using IteratorType = std::vector<RealType>::iterator;

  std::mt19937 gen(0);
  std::uniform_int_distribution<SizeType> dice(1, 7);
  std::uniform_real_distribution<RealType> dist1(-10000, -715);
  std::uniform_real_distribution<RealType> dist2(-715, -36);
  std::uniform_real_distribution<RealType> dist3(-36, -20);
  std::uniform_real_distribution<RealType> dist4(-20, 0);
  std::uniform_real_distribution<RealType> dist5(0, 4);
  std::uniform_real_distribution<RealType> dist6(4, 576460752303423488.0);
  std::uniform_real_distribution<RealType> dist7(576460752303423488.0, 576460752303523488.0);

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
    }
  }

  ResultType sum1(0), sum2(0), sum3(0), sum4(0), sum5(0);//, sum6(0);
  double et1(0), et2(0), et3(0), et4(0), et5(0);//, et6(0);
  std::clock_t start;

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
  }

  std::cout << std::scientific << std::setprecision(16);
  std::cout << "sum1 = " << sum1 << std::endl;
  std::cout << "sum2 = " << sum2 << std::endl;
  std::cout << "sum3 = " << sum3 << std::endl;
  std::cout << "sum4 = " << sum4 << std::endl;
  std::cout << "sum5 = " << sum5 << std::endl;

  std::cout << "time (work 1) = " << et1 << std::endl;
  std::cout << "time (work 2) = " << et2 << std::endl;
  std::cout << "time (work 3) = " << et3 << std::endl;
  std::cout << "time (work 4) = " << et4 << std::endl;
  std::cout << "time (work 5) = " << et5 << std::endl;

  using Type = ResultType;
  std::cout << std::scientific << std::setprecision(16);
  std::cout << "eps = " << std::numeric_limits<Type>::epsilon() << std::endl;
  Type err = 0;
  for (Type x = -20; x <= 100; x += 0.00001) {
    Type w = lambert_w_exp(x);
    if (x >= 0) {
      err = (w + std::log(w) - x) / std::max(static_cast<Type>(1), x);
    } else {
      err = w - std::exp(x - w);
    }
    if (std::abs(err) >= 4*std::numeric_limits<Type>::epsilon()) {
      std::cout << x << ", " << w << ", " << std::log(w) << ", " << err << std::endl;
    }
  }

  for (auto x : v) {
    Type w = lambert_w_exp(x);
    if (x >= 0) {
      err = (w + std::log(w) - x) / std::max(1.0, x);
    } else {
      err = w - std::exp(x - w);
    }
    if (std::abs(err) >= 4*std::numeric_limits<Type>::epsilon()) {
      std::cout << x << ", " << w << ", " << std::log(w) << ", " << err << std::endl;
    }
  }
}

int main() {

  std::cout << "test_lambert_float" << std::endl;
//  test_lambert_float(1000000, 10000);
  test_lambert_float(100000, 100);
  std::cout << "test_lambert_double" << std::endl;
//  test_lambert_double(1000000, 10000);
  test_lambert_double(100000, 100);

//  using RealType = float;
//  using ResultType = RealType;
//  using IndexType = std::vector<RealType>::difference_type;
//  using SizeType = std::vector<RealType>::size_type;
//  using IteratorType = std::vector<RealType>::iterator;

//  SizeType dim = 100000;
//  SizeType repeat = 10;

//  std::mt19937 gen(0);
//  std::uniform_int_distribution<SizeType> dice(1, 8);
////  std::uniform_real_distribution<RealType> dist1(-1e15, -715);
////  std::uniform_real_distribution<RealType> dist2(-715, -36);
////  std::uniform_real_distribution<RealType> dist3(-36, -20);
////  std::uniform_real_distribution<RealType> dist4(-20, -1);
////  std::uniform_real_distribution<RealType> dist5(-1, 0.5);
////  std::uniform_real_distribution<RealType> dist6(0.5, 2);
////  std::uniform_real_distribution<RealType> dist7(2, 576460752303423488);
////  std::uniform_real_distribution<RealType> dist8(576460752303423488, 1e+24);
//  std::uniform_real_distribution<RealType> dist1(-1000, -91);
//  std::uniform_real_distribution<RealType> dist2(-91, -18);
//  std::uniform_real_distribution<RealType> dist3(-18, -1);
//  std::uniform_real_distribution<RealType> dist4(-18, 0.5);
//  std::uniform_real_distribution<RealType> dist5(-1, 0.5);
//  std::uniform_real_distribution<RealType> dist6(0.5, 2);
//  std::uniform_real_distribution<RealType> dist7(2, 536870912);
//  std::uniform_real_distribution<RealType> dist8(536870912, 1e+10f);
////  std::normal_distribution<RealType> dist1(0, static_cast<RealType>(1e-6));
////  std::normal_distribution<RealType> dist2(0, static_cast<RealType>(1e+0));
////  std::normal_distribution<RealType> dist3(0, static_cast<RealType>(1e+6));

//  std::vector<RealType> v;
//  v.reserve(dim);
//  for (SizeType i = 0; i < dim; ++i) {
//    SizeType outcome = dice(gen);
//    switch (outcome) {
//      case 1: v.push_back(dist1(gen)); break;
//      case 2: v.push_back(dist2(gen)); break;
//      case 3: v.push_back(dist3(gen)); break;
//      case 4: v.push_back(dist4(gen)); break;
//      case 5: v.push_back(dist5(gen)); break;
//      case 6: v.push_back(dist6(gen)); break;
//      case 7: v.push_back(dist7(gen)); break;
//      case 8: v.push_back(dist8(gen)); break;
//    }
//  }

////  display(v);

//  ResultType sum1(0), sum2(0), sum3(0), sum4(0), sum5(0);//, sum6(0);
//  double et1(0), et2(0), et3(0), et4(0), et5(0);//, et6(0);
//  std::clock_t start;

////  std_sum<IteratorType, double> stdsum;
////  kahan_sum<IteratorType, double> kahansum;

//  for (SizeType i = 0; i < repeat; ++i) {
//    // Work 1
//    start = std::clock();
//    std::for_each(v.begin(), v.end(), [&](const RealType x){ sum1 += lambert_w_exp(x); });
//    et1 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

//    // Work 2
//    start = std::clock();
//    std::for_each(v.begin(), v.end(), [&](const RealType x){ sum2 += std::log(x); });
//    et2 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

//    // Work 3
//    start = std::clock();
//    std::for_each(v.begin(), v.end(), [&](const RealType x){ sum3 += std::exp(x); });
//    et3 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

//    // Work 4
//    start = std::clock();
//    std::for_each(v.begin(), v.end(), [&](const RealType x){ sum4 += fmath::exp(x); });
//    et4 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

//    // Work 5
//    start = std::clock();
//    std::for_each(v.begin(), v.end(), [&](const RealType x){ sum5 += exp_approx(x); });
//    et5 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

////    // Work 1
////    start = std::clock();
////    sum1 = std::accumulate(v.begin(), v.end(), sum1);
////    et1 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

////    // Work 2
////    start = std::clock();
////    sum2 = kahan_accumulate(v.begin(), v.end(), sum2);
////    et2 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

////    // Work 3
////    start = std::clock();
////    sum3 = std::accumulate(v.begin(), v.end(), sum3);
////    et3 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

////    // Work 4
////    start = std::clock();
////    sum4 = kahan_accumulate(v.begin(), v.end(), sum4);
////    et4 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

////    // Work 5
////    start = std::clock();
////    sum5 = stdsum(v.begin(), v.end(), sum5);
////    et5 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;

////    // Work 6
////    start = std::clock();
////    sum6 = kahansum(v.begin(), v.end(), sum6);
////    et6 += static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;
//  }

//  std::cout << std::scientific << std::setprecision(16);
//  std::cout << "sum1 = " << sum1 << std::endl;
//  std::cout << "sum2 = " << sum2 << std::endl;
//  std::cout << "sum3 = " << sum3 << std::endl;
//  std::cout << "sum4 = " << sum4 << std::endl;
//  std::cout << "sum5 = " << sum5 << std::endl;
////  std::cout << "sum6 = " << sum6 << std::endl;

//  std::cout << "time (work 1) = " << et1 << std::endl;
//  std::cout << "time (work 2) = " << et2 << std::endl;
//  std::cout << "time (work 3) = " << et3 << std::endl;
//  std::cout << "time (work 4) = " << et4 << std::endl;
//  std::cout << "time (work 5) = " << et5 << std::endl;
////  std::cout << "time (work 6) = " << et6 << std::endl;

////  std::cout << std::exp(709.0) << std::endl;
////  std::cout << std::exp(710.0) << std::endl;
////  std::cout << std::exp(-708.0) << std::endl;
////  std::cout << std::exp(-709.0) << std::endl;

////  std::cout << std::exp(88.0f) << std::endl;
////  std::cout << std::exp(89.0f) << std::endl;
////  std::cout << std::exp(-88.0f) << std::endl;
////  std::cout << std::exp(-89.0f) << std::endl;

//  using Type = ResultType;
//  std::cout << std::scientific << std::setprecision(16);
//  std::cout << "eps = " << std::numeric_limits<Type>::epsilon() << std::endl;
//  Type err = 0;
//  for (Type x = -1; x <= 10; x += 0.000001f) {
//    Type w = lambert_w_exp(x);
//    if (x >= 0) {
//      err = (w + std::log(w) - x) / std::max(static_cast<Type>(1), x);
//    } else {
//      err = w - std::exp(x - w);
//    }
//    if (std::abs(err) > 4*std::numeric_limits<Type>::epsilon()) {
//      std::cout << x << ", " << w << ", " << std::log(w) << ", " << err << std::endl;
//    }
////    std::cout << x << ", " << w << std::endl;
//  }

////  for (auto x : v) {
////    Type w = lambert_w_exp(x);
////    if (x >= 0) {
////      err = (w + std::log(w) - x) / std::max(1.0f, x);
////    } else {
////      err = w - std::exp(x - w);
////    }
////    if (std::abs(err) > 2*std::numeric_limits<Type>::epsilon()) {
////      std::cout << x << ", " << w << ", " << std::log(w) << ", " << err << std::endl;
////    }
//////    std::cout << x << ", " << w << std::endl;
////  }

////  std::cout << "sizeof(int) = " << sizeof(int) << std::endl;
////  std::cout << "sizeof(long int) = " << sizeof(long int) << std::endl;
////  std::cout << "sizeof(long unsigned int) = " << sizeof(long unsigned int) << std::endl;
////  std::cout << "sizeof(std::size_t) = " << sizeof(std::size_t) << std::endl;
////  std::cout << "sizeof(std::vector<double>::size_type) = " << sizeof(std::vector<double>::size_type) << std::endl;
////  std::cout << "sizeof(std::vector<float>::size_type) = " << sizeof(std::vector<float>::size_type) << std::endl;
////  std::cout << "sizeof(IndexType) = " << sizeof(IndexType) << std::endl;
////#ifdef MKL_INT
////  std::cout << "sizeof(MKL_INT) = " << sizeof(MKL_INT) << std::endl;
////#endif

////  long double s, t = 3.0;
////  s = 1.0 - (4.0/t - 1.0)*t;
////  std::cout << "s = " << s << std::endl;

  return 0;
}
