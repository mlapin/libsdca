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

}

#endif
