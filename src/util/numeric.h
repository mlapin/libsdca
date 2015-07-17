#ifndef SDCA_UTIL_NUMERIC_H
#define SDCA_UTIL_NUMERIC_H

#include <numeric>

namespace sdca {

template <typename Data, typename Result>
inline
void
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
inline
Result
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
  std_sum() { std::cout << "std sum" << std::endl; }
  inline Result operator()(Iterator first, Iterator last, Result init) {
    return std::accumulate(first, last, init);
  }
};

template <typename Iterator, typename Result>
struct kahan_sum {
//  kahan_sum() { std::cout << "kahan sum" << std::endl; }
  inline Result operator()(Iterator first, Iterator last, Result init) {
    return kahan_accumulate(first, last, init);
  }
};

}

#endif
