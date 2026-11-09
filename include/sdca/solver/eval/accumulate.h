#ifndef SDCA_SOLVER_EVAL_ACCUMULATE_H
#define SDCA_SOLVER_EVAL_ACCUMULATE_H

#include <algorithm>
#include <cmath>

#include "sdca/utility/types.h"

namespace sdca {

// Neumaier's compensated sum, with the magnitude sum |x| of its error bound.
template <typename T>
struct compensated_sum {
  T sum = T(0), correction = T(0), magnitude = T(0);

  inline void add(const T x) {
    const T t = sum + x;
    correction += (std::abs(sum) >= std::abs(x)) ? (sum - t) + x : (x - t) + sum;
    sum = t;
    magnitude += std::abs(x);
  }

  inline T value() const { return sum + correction; }
};

// Long sums go in chunks of about sqrt(n) terms: their rounding then grows
// with chunk_steps(n), not with n.
inline size_type eval_chunk(const size_type n) {
  size_type chunk = 8;
  while (chunk * chunk < n) chunk *= 2;
  return chunk;
}

// Roundings per term of a chunked sum (or a dot product) of n terms.
inline double chunk_steps(const size_type n) {
  const size_type chunk = eval_chunk(n);
  return static_cast<double>(std::min(n, chunk) + n / chunk + 2);
}

// sum_j x_j y_j in T, in chunks. Float products are exact in double.
template <typename T, typename X, typename Y>
inline T dot_chunked(const size_type n, const X* x, const Y* y) {
  const size_type chunk = eval_chunk(n);
  T total(0);
  for (size_type j = 0; j < n; j += chunk) {
    const size_type end = std::min(n, j + chunk);
    T part(0);
    for (size_type l = j; l < end; ++l) part += static_cast<T>(x[l]) * static_cast<T>(y[l]);
    total += part;
  }
  return total;
}

// dot_chunked of four pairs (x[q], y[q]) of length n at once: each sum adds
// its terms in the order of dot_chunked, so the results are the same bits;
// the four independent sums keep the adder busy where one waits on each add.
template <typename T, typename X, typename Y>
inline void dot_chunked4(const size_type n, const X* const x[4], const Y* const y[4], T* out) {
  const size_type chunk = eval_chunk(n);
  T total[4] = {T(0), T(0), T(0), T(0)};
  for (size_type j = 0; j < n; j += chunk) {
    const size_type end = std::min(n, j + chunk);
    T part[4] = {T(0), T(0), T(0), T(0)};
    for (size_type l = j; l < end; ++l)
      for (int q = 0; q < 4; ++q) part[q] += static_cast<T>(x[q][l]) * static_cast<T>(y[q][l]);
    for (int q = 0; q < 4; ++q) total[q] += part[q];
  }
  for (int q = 0; q < 4; ++q) out[q] = total[q];
}

}  // namespace sdca

#endif
