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
#include "prox/prox.h"

using namespace sdca;

template <typename T>
void display(const std::vector<T> &v) {
  if (v.size() < 100) {
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
  }
}

void test_thresholds_topk_entropy_biased(std::size_t dim) {
  using RealType = double;
  using IndexType = std::vector<RealType>::difference_type;
  using SizeType = std::vector<RealType>::size_type;
  using IteratorType = std::vector<RealType>::iterator;

  std::mt19937 gen(0);
//  std::uniform_real_distribution<RealType> dist1(1, 1);
  std::normal_distribution<RealType> dist1(-1e+12, 1);
  std::vector<RealType> v;
  v.reserve(dim);
  for (SizeType i = 0; i < dim; ++i) {
    v.push_back(dist1(gen));
  }

  IndexType k = 1;
  RealType alpha = 1;
  std::cout << "d = " << dim << ", k = " << k << ", alpha = " << alpha << std::endl;
  std::cout << "min = " << *std::min_element(v.begin(), v.end()) << std::endl;
  std::cout << "max = " << *std::max_element(v.begin(), v.end()) << std::endl;
  auto t = thresholds_topk_entropy_biased(v.begin(), v.end(), k, alpha);
//  std::cout << "before:" << std::endl;
  display(v);
  prox(t, v.begin(), v.end());
//  std::cout << "after:" << std::endl;
  display(v);
  RealType s1 = std::accumulate(v.begin(), v.end(), 0.0);
  std::cout << "sum = " << s1 << std::endl;
  std::cout << "hi  = " << s1/static_cast<RealType>(k) << std::endl;
  std::cout << "max = " << *std::max_element(v.begin(), v.end()) << std::endl;
  std::cout << "min = " << *std::min_element(v.begin(), v.end()) << std::endl;

}

void test_solve_sum_w_exp(std::size_t dim, std::size_t repeat = 1) {
  using RealType = double;
  using ResultType = RealType;
  using IndexType = std::vector<RealType>::difference_type;
  using SizeType = std::vector<RealType>::size_type;
  using IteratorType = std::vector<RealType>::iterator;

  std::mt19937 gen(0);
//  std::uniform_real_distribution<RealType> dist1(1, 10);

  for (std::size_t iter = 0; iter < repeat; ++iter) {

    for (RealType mu = 1e-10; mu <= 1e+10; mu *= 10) {
    for (RealType sigma = 1e-10; sigma <= 1e+10; sigma *= 10) {
    for (int sgn = -1; sgn <= 1; sgn += 2) {

    std::normal_distribution<RealType> dist1(sgn * mu, sigma);

    std::vector<RealType> v;
    v.reserve(dim);
    for (SizeType i = 0; i < dim; ++i) {
      v.push_back(dist1(gen));
    }

    {
    ResultType rhs = 1;
    ResultType t = solve_sum_w_exp(v.begin(), v.end(), rhs);
    ResultType f = std::accumulate(v.begin(), v.end(), 0.0,
      [=](const ResultType a, const ResultType b)
      { return a + lambert_w_exp(b - t); });
    ResultType err = f - rhs;
    ResultType rel = err / std::max(static_cast<ResultType>(1), std::max(rhs, std::abs(t)));
    if (rel > 10*std::numeric_limits<ResultType>::epsilon()) std::cout << "rhs: " << rhs << ", t: " << t << ", f: " << f << ", err: " << err << ", rel: " << rel << std::endl << std::endl;
    }
    {
    ResultType rhs = 1e3;
    ResultType t = solve_sum_w_exp(v.begin(), v.end(), rhs);
    ResultType f = std::accumulate(v.begin(), v.end(), 0.0,
      [=](const ResultType a, const ResultType b)
      { return a + lambert_w_exp(b - t); });
    ResultType err = f - rhs;
    ResultType rel = err / std::max(static_cast<ResultType>(1), std::max(rhs, std::abs(t)));
    if (rel > 10*std::numeric_limits<ResultType>::epsilon()) std::cout << "rhs: " << rhs << ", t: " << t << ", f: " << f << ", err: " << err << ", rel: " << rel << std::endl << std::endl;
    }
    {
    ResultType rhs = 1e-3;
    ResultType t = solve_sum_w_exp(v.begin(), v.end(), rhs);
    ResultType f = std::accumulate(v.begin(), v.end(), 0.0,
      [=](const ResultType a, const ResultType b)
      { return a + lambert_w_exp(b - t); });
    ResultType err = f - rhs;
    ResultType rel = err / std::max(static_cast<ResultType>(1), std::max(rhs, std::abs(t)));
    if (rel > 10*std::numeric_limits<ResultType>::epsilon()) std::cout << "rhs: " << rhs << ", t: " << t << ", f: " << f << ", err: " << err << ", rel: " << rel << std::endl << std::endl;
    }
    {
    ResultType rhs = 1e6;
    ResultType t = solve_sum_w_exp(v.begin(), v.end(), rhs);
    ResultType f = std::accumulate(v.begin(), v.end(), 0.0,
      [=](const ResultType a, const ResultType b)
      { return a + lambert_w_exp(b - t); });
    ResultType err = f - rhs;
    ResultType rel = err / std::max(static_cast<ResultType>(1), std::max(rhs, std::abs(t)));
    if (rel > 10*std::numeric_limits<ResultType>::epsilon()) std::cout << "rhs: " << rhs << ", t: " << t << ", f: " << f << ", err: " << err << ", rel: " << rel << std::endl << std::endl;
    }
    {
    ResultType rhs = 1e-6;
    ResultType t = solve_sum_w_exp(v.begin(), v.end(), rhs);
    ResultType f = std::accumulate(v.begin(), v.end(), 0.0,
      [=](const ResultType a, const ResultType b)
      { return a + lambert_w_exp(b - t); });
    ResultType err = f - rhs;
    ResultType rel = err / std::max(static_cast<ResultType>(1), std::max(rhs, std::abs(t)));
    if (rel > 10*std::numeric_limits<ResultType>::epsilon()) std::cout << "rhs: " << rhs << ", t: " << t << ", f: " << f << ", err: " << err << ", rel: " << rel << std::endl << std::endl;
    }
    {
    ResultType rhs = 1e9;
    ResultType t = solve_sum_w_exp(v.begin(), v.end(), rhs);
    ResultType f = std::accumulate(v.begin(), v.end(), 0.0,
      [=](const ResultType a, const ResultType b)
      { return a + lambert_w_exp(b - t); });
    ResultType err = f - rhs;
    ResultType rel = err / std::max(static_cast<ResultType>(1), std::max(rhs, std::abs(t)));
    if (rel > 10*std::numeric_limits<ResultType>::epsilon()) std::cout << "rhs: " << rhs << ", t: " << t << ", f: " << f << ", err: " << err << ", rel: " << rel << std::endl << std::endl;
    }
    {
    ResultType rhs = 1e-9;
    ResultType t = solve_sum_w_exp(v.begin(), v.end(), rhs);
    ResultType f = std::accumulate(v.begin(), v.end(), 0.0,
      [=](const ResultType a, const ResultType b)
      { return a + lambert_w_exp(b - t); });
    ResultType err = f - rhs;
    ResultType rel = err / std::max(static_cast<ResultType>(1), std::max(rhs, std::abs(t)));
    if (rel > 10*std::numeric_limits<ResultType>::epsilon()) std::cout << "rhs: " << rhs << ", t: " << t << ", f: " << f << ", err: " << err << ", rel: " << rel << std::endl << std::endl;
    }
    {
    ResultType rhs = 1e12;
    ResultType t = solve_sum_w_exp(v.begin(), v.end(), rhs);
    ResultType f = std::accumulate(v.begin(), v.end(), 0.0,
      [=](const ResultType a, const ResultType b)
      { return a + lambert_w_exp(b - t); });
    ResultType err = f - rhs;
    ResultType rel = err / std::max(static_cast<ResultType>(1), std::max(rhs, std::abs(t)));
    if (rel > 10*std::numeric_limits<ResultType>::epsilon()) std::cout << "rhs: " << rhs << ", t: " << t << ", f: " << f << ", err: " << err << ", rel: " << rel << std::endl << std::endl;
    }
    {
    ResultType rhs = 1e-12;
    ResultType t = solve_sum_w_exp(v.begin(), v.end(), rhs);
    ResultType f = std::accumulate(v.begin(), v.end(), 0.0,
      [=](const ResultType a, const ResultType b)
      { return a + lambert_w_exp(b - t); });
    ResultType err = f - rhs;
    ResultType rel = err / std::max(static_cast<ResultType>(1), std::max(rhs, std::abs(t)));
    if (rel > 10*std::numeric_limits<ResultType>::epsilon()) std::cout << "rhs: " << rhs << ", t: " << t << ", f: " << f << ", err: " << err << ", rel: " << rel << std::endl << std::endl;
    }

    }
    }
    }
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
  std::cout << std::setprecision(15) << std::scientific;

  test_thresholds_topk_entropy_biased(10);
//  test_solve_sum_w_exp(10000);

//  std::cout << "test_lambert_float" << std::endl;
////  test_lambert_float(1000000, 10000);
//  test_lambert_float(100000, 1);
//  std::cout << "test_lambert_double" << std::endl;
//  test_lambert_double(1000000, 10000);
//  test_lambert_double(100000, 1);

//  using Type = double;
//  std::cout << std::numeric_limits<Type>::epsilon() << std::endl;
//  std::cout << std::numeric_limits<Type>::infinity() << std::endl;
//  std::cout << std::numeric_limits<Type>::denorm_min() << std::endl;
//  std::cout << std::numeric_limits<Type>::min() << std::endl;
//  std::cout << std::numeric_limits<Type>::max() << std::endl;
//  std::cout << std::numeric_limits<Type>::lowest() << std::endl;

//  std::cout << std::endl;
//  std::cout << std::numeric_limits<Type>::digits << std::endl;
//  std::cout << std::numeric_limits<Type>::digits10 << std::endl;
//  std::cout << std::numeric_limits<Type>::max_digits10 << std::endl;
//  std::cout << std::numeric_limits<Type>::max_exponent << std::endl;
//  std::cout << std::numeric_limits<Type>::max_exponent10 << std::endl;
//  std::cout << std::numeric_limits<Type>::min_exponent << std::endl;
//  std::cout << std::numeric_limits<Type>::min_exponent10 << std::endl;

  return 0;
}
