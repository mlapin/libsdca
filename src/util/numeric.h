#ifndef SDCA_UTIL_NUMERIC_H
#define SDCA_UTIL_NUMERIC_H

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
};

template <>
struct type_traits<double> {
  static constexpr const char*
  name() { return "double"; }
};

template <>
struct type_traits<long double> {
  static constexpr const char*
  name() { return "long double"; }
};

template <typename Data, typename Result>
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

template <typename Iterator, typename Result>
inline Result
kahan_accumulate(
    Iterator first,
    Iterator last,
    Result init
  ) {
  Result c = 0;
  for (; first != last; ++first) {
    Result y = static_cast<Result>(*first) - c;
    Result t = init + y;
    c = (t - init) - y;
    init = t;
  }
  return init;
}

template <typename Iterator, typename Result>
struct std_sum {
  constexpr const char*
  name() const { return "standard"; }

  inline Result
  operator()(Iterator first, Iterator last, Result init) const {
    return std::accumulate(first, last, init);
  }

  inline void
  add(const Result& value, Result& sum, Result&) const {
    sum += value;
  }
};

template <typename Iterator, typename Result>
struct kahan_sum {
  constexpr const char*
  name() const { return "kahan"; }

  inline Result
  operator()(Iterator first, Iterator last, Result init) const {
    return kahan_accumulate(first, last, init);
  }

  inline void
  add(const Result& value, Result& sum, Result& compensation) const {
    kahan_add(value, sum, compensation);
  }
};

}

#endif
