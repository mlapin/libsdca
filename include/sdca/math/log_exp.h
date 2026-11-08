#ifndef SDCA_MATH_LOG_EXP_H
#define SDCA_MATH_LOG_EXP_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace sdca {

template <typename Type>
inline Type softplus(const Type x) {
  return (x > 0) ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}

template <typename Type>
struct log_traits {
  /**
   * The minimal argument of the given Type such that
   * the logarithm value is finite and normal.
   **/
  static constexpr Type min_arg() { return std::numeric_limits<Type>::min(); }

  /**
   * The maximal argument of the given Type such that
   * the logarithm value is finite and normal.
   **/
  static constexpr Type max_arg() { return std::numeric_limits<Type>::max(); }
};

// The range of exp as constants (std::log is not constexpr): the arguments
// nearest to the logarithms of the smallest normal and the largest finite
// value at which exp is still finite and normal.
template <typename Type>
struct exp_traits;

template <>
struct exp_traits<float> {
  /**
   * The minimal argument of the given Type such that
   * the exponent value is finite and normal (and > 0).
   **/
  static constexpr float min_arg() { return -8.733654022216796875000000e+01f; }

  /**
   * The maximal argument of the given Type such that
   * the exponent value is finite and normal.
   **/
  static constexpr float max_arg() { return 8.87228317260742187500e+01f; }
};

template <>
struct exp_traits<double> {
  static constexpr double min_arg() { return -7.083964185322641e+02; }
  static constexpr double max_arg() { return 7.09782712893384e+02; }
};

// The x87 (and binary128) exponent range; where long double is double
// (Apple arm64, MSVC), the range of double.
template <>
struct exp_traits<long double> {
  static constexpr bool wide =
      std::numeric_limits<long double>::max_exponent > std::numeric_limits<double>::max_exponent;
  static constexpr long double min_arg() {
    return wide ? -1.135513711193302405799699e+04L
                : static_cast<long double>(exp_traits<double>::min_arg());
  }
  static constexpr long double max_arg() {
    return wide ? 1.135652340629414394879149e+04L
                : static_cast<long double>(exp_traits<double>::max_arg());
  }
};

/**
 * Computes
 *    x * exp(x)
 **/
template <typename Type>
inline Type x_exp_x(const Type x) {
  return x * std::exp(x);
}

/**
 * Computes
 *    x * log(x)
 **/
template <typename Type>
inline Type x_log_x(const Type x) {
  return (x > 0) ? x * std::log(x) : 0;
}

/**
 * Computes
 *    log(sum_i exp(a_i))
 * where a_i are elements in the range [first, last)
 * and max is an iterator to the maximum in the given range.
 **/
template <typename Result = double, typename Iterator>
inline Result log_sum_exp(Iterator first, Iterator last, Iterator max) {
  Result s(0), m(static_cast<Result>(*max));
  for (; first != max; ++first) {
    s += std::exp(static_cast<Result>(*first) - m);
  }
  for (first = max + 1; first != last; ++first) {
    s += std::exp(static_cast<Result>(*first) - m);
  }
  return m + std::log1p(s);
}

/**
 * Computes
 *    log(sum_i exp(a_i))
 * where a_i are elements in the range [first, last).
 * Returns 0 if the range is empty (first == last).
 **/
template <typename Result = double, typename Iterator>
inline Result log_sum_exp(Iterator first, Iterator last) {
  if (first == last) return 0;
  Iterator max = std::max_element(first, last);
  return log_sum_exp<Result>(first, last, max);
}

/**
 * Computes
 *    log(sum_i exp(a_i))
 * where a_i are elements in the range [first, last)
 * and max is an iterator to the maximum in the given range.
 * Additionally, the variable s is set to
 *    s = sum_i exp(a_i - *max)
 * where the sum is over all the elements in the range,
 * except the maximum element.
 **/
template <typename Result, typename Iterator>
inline Result log_sum_exp(Iterator first, Iterator last, Iterator max, Result& s) {
  s = 0;
  Result m(static_cast<Result>(*max));
  for (; first != max; ++first) {
    s += std::exp(static_cast<Result>(*first) - m);
  }
  for (first = max + 1; first != last; ++first) {
    s += std::exp(static_cast<Result>(*first) - m);
  }
  return m + std::log1p(s);
}

/**
 * Computes
 *    log(sum_i exp(a_i))
 * where a_i are elements in the range [first, last).
 * Additionally, the variable s is set to
 *    s = sum_i exp(a_i - *max)
 * where the sum is over all the elements in the range,
 * except the maximum element.
 * Returns 0 if the range is empty (first == last).
 **/
template <typename Result, typename Iterator>
inline Result log_sum_exp(Iterator first, Iterator last, Result& s) {
  if (first == last) return 0;
  Iterator max = std::max_element(first, last);
  return log_sum_exp(first, last, max, s);
}

/**
 * Computes
 *    log(1 + sum_i exp(a_i))
 * where a_i are elements in the range [first, last)
 * and max is an iterator to the maximum in the given range.
 **/
template <typename Result = double, typename Iterator>
inline Result log_1_sum_exp(Iterator first, Iterator last, Iterator max) {
  return softplus(log_sum_exp<Result>(first, last, max));
}

/**
 * Computes
 *    log(1 + sum_i exp(a_i))
 * where a_i are elements in the range [first, last).
 * Returns 0 if the range is empty (first == last).
 **/
template <typename Result = double, typename Iterator>
inline Result log_1_sum_exp(Iterator first, Iterator last) {
  if (first == last) return 0;
  Iterator max = std::max_element(first, last);
  return log_1_sum_exp<Result>(first, last, max);
}

/**
 * Computes
 *    log(1 + sum_i exp(a_i))
 * where a_i are elements in the range [first, last)
 * and max is an iterator to the maximum in the given range.
 * Additionally, the variable s is set to
 *    s = exp(-*max) + sum_i exp(a_i - *max)
 * where the sum is over all the elements in the range,
 * except the maximum element.
 **/
template <typename Result, typename Iterator>
inline Result log_1_sum_exp(Iterator first, Iterator last, Iterator max, Result& s) {
  Result m(static_cast<Result>(*max));
  s = 0;
  for (; first != max; ++first) {
    s += std::exp(static_cast<Result>(*first) - m);
  }
  for (first = max + 1; first != last; ++first) {
    s += std::exp(static_cast<Result>(*first) - m);
  }
  Result value = softplus(m + std::log1p(s));
  s += std::exp(-m);  // documented auxiliary sum may overflow; value need not
  return value;
}

/**
 * Computes
 *    log(1 + sum_i exp(a_i))
 * where a_i are elements in the range [first, last).
 * Additionally, the variable s is set to
 *    s = exp(-*max) + sum_i exp(a_i - *max)
 * where the sum is over all the elements in the range,
 * except the maximum element.
 * Returns 0 if the range is empty (first == last).
 **/
template <typename Result, typename Iterator>
inline Result log_1_sum_exp(Iterator first, Iterator last, Result& s) {
  if (first == last) return 0;
  Iterator max = std::max_element(first, last);
  return log_1_sum_exp(first, last, max, s);
}

/**
 * Computes both
 *    lse  = log(sum_i exp(a_i))
 *    lse1 = log(1 + sum_i exp(a_i))
 * in a single pass over the data
 * where a_i are elements in the range [first, last)
 * and max is an iterator to the maximum in the given range.
 * Returns
 *    sum_i exp(a_i - *max)
 * where the sum is over all the elements in the range,
 * except the maximum element.
 **/
template <typename Result, typename Iterator>
inline Result log_sum_exp(Iterator first, Iterator last, Iterator max, Result& lse, Result& lse1) {
  Result s(0), m(static_cast<Result>(*max));
  for (; first != max; ++first) {
    s += std::exp(static_cast<Result>(*first) - m);
  }
  for (first = max + 1; first != last; ++first) {
    s += std::exp(static_cast<Result>(*first) - m);
  }
  lse = m + std::log1p(s);
  lse1 = softplus(lse);
  return s;
}

/**
 * Computes both
 *    lse  = log(sum_i exp(a_i))
 *    lse1 = log(1 + sum_i exp(a_i))
 * in a single pass over the data
 * where a_i are elements in the range [first, last)
 * and max is an iterator to the maximum in the given range.
 * Returns
 *    sum_i exp(a_i - *max)
 * where the sum is over all the elements in the range,
 * except the maximum element.
 * Returns 0 and sets lse = lse1 = 0 if the range is empty (first == last).
 **/
template <typename Result, typename Iterator>
inline Result log_sum_exp(Iterator first, Iterator last, Result& lse, Result& lse1) {
  if (first == last) {
    lse = 0;
    lse1 = 0;
    return 0;
  }
  Iterator max = std::max_element(first, last);
  return log_sum_exp(first, last, max, lse, lse1);
}

}  // namespace sdca

#endif
