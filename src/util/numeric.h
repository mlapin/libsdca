#ifndef SDCA_UTIL_NUMERIC_H
#define SDCA_UTIL_NUMERIC_H

#include <algorithm>
#include <cmath>
#include <numeric>

namespace sdca {

template <typename Type>
struct type_traits {
  static constexpr const char*
  name() { return "unknown type"; }
};

template <>
struct type_traits<float> {
  static constexpr const char*
  name() { return "float"; }

  static constexpr float
  min_exp_arg() { return -103.0f; }
};

template <>
struct type_traits<double> {
  static constexpr const char*
  name() { return "double"; }

  static constexpr double
  min_exp_arg() { return -745.0; }
};

template <>
struct type_traits<long double> {
  static constexpr const char*
  name() { return "long double"; }

  static constexpr long double
  min_exp_arg() { return -11399.0L; }
};

template <typename Data,
          typename Result>
inline void
kahan_add(
    const Data& value,
    Result& sum,
    Result& compensation
  ) {
  Result y = static_cast<Result>(value) - compensation;
  Result t = sum + y;
  compensation = (t - sum) - y;
  sum = t;
}

template <typename Iterator,
          typename Result>
inline Result
kahan_accumulate(
    Iterator first,
    Iterator last,
    Result init,
    Result c
  ) {
  for (; first != last; ++first) {
    Result y = static_cast<Result>(*first) - c;
    Result t = init + y;
    c = (t - init) - y;
    init = t;
  }
  return init;
}

template <typename Iterator,
          typename Result>
inline Result
kahan_accumulate(
    Iterator first,
    Iterator last,
    Result init
  ) {
  return kahan_accumulate(first, last, init, static_cast<Result>(0));
}

template <typename Iterator,
          typename Result>
struct std_sum {
  constexpr const char*
  name() const { return "standard"; }

  inline Result
  operator()(Iterator first, Iterator last, Result init) const {
    return std::accumulate(first, last, init);
  }

  inline Result
  operator()(Iterator first, Iterator last, Result init, Result) const {
    return std::accumulate(first, last, init);
  }

  inline void
  add(const Result& value, Result& sum, Result&) const {
    sum += value;
  }
};

template <typename Iterator,
          typename Result>
struct kahan_sum {
  constexpr const char*
  name() const { return "kahan"; }

  inline Result
  operator()(Iterator first, Iterator last, Result init) const {
    return kahan_accumulate(first, last, init);
  }

  inline Result
  operator()(Iterator first, Iterator last, Result init, Result c) const {
    return kahan_accumulate(first, last, init, c);
  }

  inline void
  add(const Result& value, Result& sum, Result& compensation) const {
    kahan_add(value, sum, compensation);
  }
};

/**
 * Compute log(\sum_i exp(a_i))
 * given a max, an iterator to the maximum over a_i.
 **/
template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline Result
log_sum_exp(
    Iterator first,
    Iterator last,
    Iterator max,
    Summation sum = Summation()
  ) {
  Result s(0), c(0);
  for (; first != max; ++first) {
    sum.add(std::exp(static_cast<Result>(*first - *max)), s, c);
  }
  for (first = max + 1; first != last; ++first) {
    sum.add(std::exp(static_cast<Result>(*first - *max)), s, c);
  }
  return static_cast<Result>(*max) + std::log1p(s);
}

/**
 * Compute log(\sum_i exp(a_i)).
 **/
template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline Result
log_sum_exp(
    Iterator first,
    Iterator last,
    Summation sum = Summation()
  ) {
  if (first == last) return 0;
  auto max = std::max_element(first, last);
  return log_sum_exp<Iterator, Result>(first, last, max, sum);
}

/**
 * Compute log(\sum_i exp(a_i))
 * given a max, an iterator to the maximum over a_i.
 * Let s = \sum_i exp(a_i - *max) over i \neq *max.
 **/
template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline Result
log_sum_exp(
    Iterator first,
    Iterator last,
    Iterator max,
    Result& s,
    Summation sum = Summation()
  ) {
  s = 0;
  Result c(0);
  for (; first != max; ++first) {
    sum.add(std::exp(static_cast<Result>(*first - *max)), s, c);
  }
  for (first = max + 1; first != last; ++first) {
    sum.add(std::exp(static_cast<Result>(*first - *max)), s, c);
  }
  return static_cast<Result>(*max) + std::log1p(s);
}

/**
 * Compute log(\sum_i exp(a_i)).
 * Let s = \sum_i exp(a_i - *max) over i \neq *max.
 **/
template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline Result
log_sum_exp(
    Iterator first,
    Iterator last,
    Result& s,
    Summation sum = Summation()
  ) {
  if (first == last) return 0;
  auto max = std::max_element(first, last);
  return log_sum_exp(first, last, max, s, sum);
}

/**
 * Compute log(1 + \sum_i exp(a_i))
 * given a max, an iterator to the maximum over a_i.
 **/
template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline Result
log_1_sum_exp(
    Iterator first,
    Iterator last,
    Iterator max,
    Summation sum = Summation()
  ) {
  Result s(std::exp(static_cast<Result>(-*max))), c(0);
  if (!std::isfinite(s)) return 0;
  for (; first != max; ++first) {
    sum.add(std::exp(static_cast<Result>(*first - *max)), s, c);
  }
  for (first = max + 1; first != last; ++first) {
    sum.add(std::exp(static_cast<Result>(*first - *max)), s, c);
  }
  return static_cast<Result>(*max) + std::log1p(s);
}

/**
 * Compute log(1 + \sum_i exp(a_i)).
 **/
template <typename Iterator,
          typename Result = double,
          typename Summation = std_sum<Iterator, Result>>
inline Result
log_1_sum_exp(
    Iterator first,
    Iterator last,
    Summation sum = Summation()
  ) {
  if (first == last) return 0;
  auto max = std::max_element(first, last);
  return log_1_sum_exp<Iterator, Result>(first, last, max, sum);
}

/**
 * Compute log(1 + \sum_i exp(a_i))
 * given a max, an iterator to the maximum over a_i.
 * Let s = exp(-*max) + \sum_i exp(a_i - *max) over i \neq *max.
 **/
template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline Result
log_1_sum_exp(
    Iterator first,
    Iterator last,
    Iterator max,
    Result& s,
    Summation sum = Summation()
  ) {
  s = std::exp(static_cast<Result>(-*max));
  if (!std::isfinite(s)) return 0;
  Result c(0);
  for (; first != max; ++first) {
    sum.add(std::exp(static_cast<Result>(*first - *max)), s, c);
  }
  for (first = max + 1; first != last; ++first) {
    sum.add(std::exp(static_cast<Result>(*first - *max)), s, c);
  }
  return static_cast<Result>(*max) + std::log1p(s);
}

/**
 * Compute log(1 + \sum_i exp(a_i)).
 * Let s = exp(-*max) + \sum_i exp(a_i - *max) over i \neq *max.
 **/
template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline Result
log_1_sum_exp(
    Iterator first,
    Iterator last,
    Result& s,
    Summation sum = Summation()
  ) {
  if (first == last) return 0;
  auto max = std::max_element(first, last);
  return log_1_sum_exp(first, last, max, s, sum);
}

/**
 * Compute both
 *    lse = log(\sum_i exp(a_i))
 * and
 *    lse1 = log(1 + \sum_i exp(a_i))
 * in a single pass over a_i and
 * given a max, an iterator to the maximum over a_i.
 * Return \sum_i exp(a_i - *max) over i \neq *max.
 **/
template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline Result
log_sum_exp_log_1_sum_exp(
    Iterator first,
    Iterator last,
    Iterator max,
    Result& lse,
    Result& lse1,
    Summation sum = Summation()
  ) {
  Result s(0), c(0);
  for (; first != max; ++first) {
    sum.add(std::exp(static_cast<Result>(*first - *max)), s, c);
  }
  for (first = max + 1; first != last; ++first) {
    sum.add(std::exp(static_cast<Result>(*first - *max)), s, c);
  }
  lse = static_cast<Result>(*max) + std::log1p(s);
  c = std::exp(static_cast<Result>(-*max)) - c;
  lse1 = std::isfinite(c) ? static_cast<Result>(*max) + std::log1p(s + c) : 0;
  return s;
}

/**
 * Compute both
 *    lse = log(\sum_i exp(a_i))
 * and
 *    lse1 = log(1 + \sum_i exp(a_i))
 * in a single pass over a_i.
 * Return \sum_i exp(a_i - *max) over i \neq *max.
 **/
template <typename Iterator,
          typename Result,
          typename Summation = std_sum<Iterator, Result>>
inline Result
log_sum_exp_log_1_sum_exp(
    Iterator first,
    Iterator last,
    Result& lse,
    Result& lse1,
    Summation sum = Summation()
  ) {
  if (first == last) {
    lse = 0; lse1 = 0;
    return 0;
  }
  auto max = std::max_element(first, last);
  return log_sum_exp_log_1_sum_exp(first, last, max, lse, lse1, sum);
}

}

#endif
