#ifndef SDCA_MATH_LOG_SUM_EXP_H
#define SDCA_MATH_LOG_SUM_EXP_H

#include <algorithm>
#include <cmath>
#include <numeric>

namespace sdca {

/**
 * Computes
 *    log(\sum_i exp(a_i))
 * where a_i are elements in the range [first, last)
 * and max is an iterator to the maximum in the given range.
 **/
template <typename Result = double,
          typename Iterator>
inline Result
log_sum_exp(
    Iterator first,
    Iterator last,
    Iterator max
  ) {
  Result s(0);
  for (; first != max; ++first) {
    s += std::exp(static_cast<Result>(*first - *max));
  }
  for (first = max + 1; first != last; ++first) {
    s += std::exp(static_cast<Result>(*first - *max));
  }
  return static_cast<Result>(*max) + std::log1p(s);
}

/**
 * Computes
 *    log(\sum_i exp(a_i))
 * where a_i are elements in the range [first, last).
 * Returns 0 if the range is empty (first == last).
 **/
template <typename Result = double,
          typename Iterator>
inline Result
log_sum_exp(
    Iterator first,
    Iterator last
  ) {
  if (first == last) return 0;
  Iterator max = std::max_element(first, last);
  return log_sum_exp<Result>(first, last, max);
}

/**
 * Computes
 *    log(\sum_i exp(a_i))
 * where a_i are elements in the range [first, last)
 * and max is an iterator to the maximum in the given range.
 * Additionally, the variable s is set to
 *    s = \sum_i exp(a_i - *max)
 * where the sum is over all the elements in the range,
 * except the maximum element.
 **/
template <typename Result,
          typename Iterator>
inline Result
log_sum_exp(
    Iterator first,
    Iterator last,
    Iterator max,
    Result& s
  ) {
  s = 0;
  for (; first != max; ++first) {
    s += std::exp(static_cast<Result>(*first - *max));
  }
  for (first = max + 1; first != last; ++first) {
    s += std::exp(static_cast<Result>(*first - *max));
  }
  return static_cast<Result>(*max) + std::log1p(s);
}

/**
 * Computes
 *    log(\sum_i exp(a_i))
 * where a_i are elements in the range [first, last).
 * Additionally, the variable s is set to
 *    s = \sum_i exp(a_i - *max)
 * where the sum is over all the elements in the range,
 * except the maximum element.
 * Returns 0 if the range is empty (first == last).
 **/
template <typename Result,
          typename Iterator>
inline Result
log_sum_exp(
    Iterator first,
    Iterator last,
    Result& s
  ) {
  if (first == last) return 0;
  Iterator max = std::max_element(first, last);
  return log_sum_exp(first, last, max, s);
}

/**
 * Computes
 *    log(1 + \sum_i exp(a_i))
 * where a_i are elements in the range [first, last)
 * and max is an iterator to the maximum in the given range.
 **/
template <typename Result = double,
          typename Iterator>
inline Result
log_1_sum_exp(
    Iterator first,
    Iterator last,
    Iterator max
  ) {
  Result s(std::exp(static_cast<Result>(-*max)));
  if (!std::isfinite(s)) return 0;
  for (; first != max; ++first) {
    s += std::exp(static_cast<Result>(*first - *max));
  }
  for (first = max + 1; first != last; ++first) {
    s += std::exp(static_cast<Result>(*first - *max));
  }
  return static_cast<Result>(*max) + std::log1p(s);
}

/**
 * Computes
 *    log(1 + \sum_i exp(a_i))
 * where a_i are elements in the range [first, last).
 * Returns 0 if the range is empty (first == last).
 **/
template <typename Result = double,
          typename Iterator>
inline Result
log_1_sum_exp(
    Iterator first,
    Iterator last
  ) {
  if (first == last) return 0;
  Iterator max = std::max_element(first, last);
  return log_1_sum_exp<Result>(first, last, max);
}

/**
 * Computes
 *    log(1 + \sum_i exp(a_i))
 * where a_i are elements in the range [first, last)
 * and max is an iterator to the maximum in the given range.
 * Additionally, the variable s is set to
 *    s = exp(-*max) + \sum_i exp(a_i - *max)
 * where the sum is over all the elements in the range,
 * except the maximum element.
 **/
template <typename Result,
          typename Iterator>
inline Result
log_1_sum_exp(
    Iterator first,
    Iterator last,
    Iterator max,
    Result& s
  ) {
  s = std::exp(static_cast<Result>(-*max));
  if (!std::isfinite(s)) return 0;
  for (; first != max; ++first) {
    s += std::exp(static_cast<Result>(*first - *max));
  }
  for (first = max + 1; first != last; ++first) {
    s += std::exp(static_cast<Result>(*first - *max));
  }
  return static_cast<Result>(*max) + std::log1p(s);
}

/**
 * Computes
 *    log(1 + \sum_i exp(a_i))
 * where a_i are elements in the range [first, last).
 * Additionally, the variable s is set to
 *    s = exp(-*max) + \sum_i exp(a_i - *max)
 * where the sum is over all the elements in the range,
 * except the maximum element.
 * Returns 0 if the range is empty (first == last).
 **/
template <typename Result,
          typename Iterator>
inline Result
log_1_sum_exp(
    Iterator first,
    Iterator last,
    Result& s
  ) {
  if (first == last) return 0;
  Iterator max = std::max_element(first, last);
  return log_1_sum_exp(first, last, max, s);
}

/**
 * Computes both
 *    lse  = log(\sum_i exp(a_i))
 *    lse1 = log(1 + \sum_i exp(a_i))
 * in a single pass over the data
 * where a_i are elements in the range [first, last)
 * and max is an iterator to the maximum in the given range.
 * Returns
 *    \sum_i exp(a_i - *max)
 * where the sum is over all the elements in the range,
 * except the maximum element.
 **/
template <typename Result,
          typename Iterator>
inline Result
log_sum_exp(
    Iterator first,
    Iterator last,
    Iterator max,
    Result& lse,
    Result& lse1
  ) {
  Result s(0);
  for (; first != max; ++first) {
    s += std::exp(static_cast<Result>(*first - *max));
  }
  for (first = max + 1; first != last; ++first) {
    s += std::exp(static_cast<Result>(*first - *max));
  }
  lse = static_cast<Result>(*max) + std::log1p(s);
  lse1 = std::exp(static_cast<Result>(-*max));
  lse1 = std::isfinite(lse1) ?
    static_cast<Result>(*max) + std::log1p(s + lse1) : 0;
  return s;
}

/**
 * Computes both
 *    lse  = log(\sum_i exp(a_i))
 *    lse1 = log(1 + \sum_i exp(a_i))
 * in a single pass over the data
 * where a_i are elements in the range [first, last)
 * and max is an iterator to the maximum in the given range.
 * Returns
 *    \sum_i exp(a_i - *max)
 * where the sum is over all the elements in the range,
 * except the maximum element.
 * Returns 0 and sets lse = lse1 = 0 if the range is empty (first == last).
 **/
template <typename Result,
          typename Iterator>
inline Result
log_sum_exp(
    Iterator first,
    Iterator last,
    Result& lse,
    Result& lse1
  ) {
  if (first == last) {
    lse = 0; lse1 = 0;
    return 0;
  }
  Iterator max = std::max_element(first, last);
  return log_sum_exp(first, last, max, lse, lse1);
}

}

#endif
