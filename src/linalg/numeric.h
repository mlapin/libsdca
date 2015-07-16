#ifndef SDCA_LINALG_NUMERIC_H
#define SDCA_LINALG_NUMERIC_H

namespace sdca {

template <typename Data, typename Result>
inline
void
kahan_add(
    const Data& x,
    Result& sum,
    Result& c
  ) {
  Result y = static_cast<Result>(x) - c;
  Result t = sum + y;
  c = (t - sum) - y;
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
//  std_sum() { std::cout << "std sum" << std::endl; }
  inline Result operator()(Iterator first, Iterator last, Result init) {
    return std::accumulate(first, last, init);
  }
};

template <typename Iterator, typename Result>
struct kahan_sum {
//  std_sum() { std::cout << "kahan sum" << std::endl; }
  inline Result operator()(Iterator first, Iterator last, Result init) {
    return kahan_accumulate(first, last, init);
  }
};

}

#endif
