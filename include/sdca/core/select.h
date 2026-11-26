#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_SELECT_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_SELECT_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_SELECT_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_SELECT_H3 1
#else
#define SDCA_CORE_SELECT_H4 1
#endif

// Top-k selection of a row: the k largest entries in descending order.
#include <algorithm>
#include <cstddef>
#include <limits>

#include "lanes.h"
#include "reduce.h"
#include "storage.h"

SDCA_ISA_BEGIN
namespace core {

// The k largest of [first, last) (0 < k <= last - first) in descending order at the front:
// partial_sort for a small k, else nth_element and a sort of the k, O(n + k log k). The rest
// is left in an unspecified order.
template <typename T>
inline void sort_largest(T* first, std::size_t k, T* last) {
  if (k <= 16) {
    std::partial_sort(first, first + k, last, ::sdca::detail::Greater());
    return;
  }
  std::nth_element(first, first + (k - 1), last, ::sdca::detail::Greater());
  std::sort(first, first + k, ::sdca::detail::Greater());
}

// The k largest entries of x[0, n) over j != skip in descending order into top (k slots,
// -inf where fewer), read as lanes A in place: an insertion into top for each entry above
// its k-th, whole vectors at or below it skipped (one compare per vector once top fills).
// For small k (up to about 64); larger k take sort_largest on a copy. valid turns false on NaN
// or +inf, by the lane sums of max_except (sdca/core/reduce.h).
template <typename A, typename S>
inline void top_k(const S* x, std::size_t n, std::size_t skip, std::size_t k,
                  typename A::value_type* top, bool& valid) {
  typedef typename A::value_type T;
  const T inf = std::numeric_limits<T>::infinity();
  for (std::size_t i = 0; i < k; ++i) top[i] = -inf;
  T last = -inf;
  auto insert = [&](T v) {
    std::size_t p = k - 1;
    for (; p > 0 && v > top[p - 1]; --p) top[p] = top[p - 1];
    top[p] = v;
    last = top[k - 1];
  };
  A total = A::zero();
  const std::size_t tail = n % A::L, full = n - tail;
  auto take = [&](const A& v, std::size_t j) {
    for (int i = 0; i < A::L; ++i)
      if (v[i] > last && j + static_cast<std::size_t>(i) != skip) insert(v[i]);
  };
  // Four vectors per test once top fills: their maximum against the k-th largest.
  const std::size_t grouped = full - full % (4 * A::L);
  std::size_t j = 0;
  for (; j < grouped; j += 4 * A::L) {
    const A v0 = load<A>(x + j), v1 = load<A>(x + j + A::L);
    const A v2 = load<A>(x + j + 2 * A::L), v3 = load<A>(x + j + 3 * A::L);
    total += (v0 + v1) + (v2 + v3);
    if (!any(max(max(v0, v1), max(v2, v3)) > last)) continue;
    take(v0, j);
    take(v1, j + A::L);
    take(v2, j + 2 * A::L);
    take(v3, j + 3 * A::L);
  }
  for (; j < full; j += A::L) {
    const A v = load<A>(x + j);
    total += v;
    if (any(v > last)) take(v, j);
  }
  bool below = all(total < inf);
  for (std::size_t i = 0; i < tail; ++i) {
    const std::size_t at = full + i;
    const T v = to_work<T>(x[at]);
    below = below && v < inf;
    if (v > last && at != skip) insert(v);
  }
  if (!below) valid = valid && below_infinity<A>(x, n);
}
template <typename A, typename S>
inline void top_k(const S* x, std::size_t n, std::size_t skip, std::size_t k,
                  typename A::value_type* top) {
  bool valid = true;
  top_k<A>(x, n, skip, k, top, valid);
}

// The index of the nth (from 1) entry of x[0, n) over j != skip equal to value (n if none):
// with nth the count of value among the k largest, the cut of a top-k selection.
template <typename A, typename S>
inline std::size_t index_of(const S* x, std::size_t n, std::size_t skip,
                            typename A::value_type value, std::size_t nth = 1) {
  std::size_t j = 0;
  for (; j + A::L <= n; j += A::L)
    if (any(load<A>(x + j) == value)) break;
  for (; j < n; ++j)
    if (j != skip && to_work<typename A::value_type>(x[j]) == value && --nth == 0) return j;
  return n;
}

// How many of top[0, k) equal value.
template <typename T>
inline std::size_t count_of(const T* top, std::size_t k, T value) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < k; ++i) count += top[i] == value;
  return count;
}

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_SELECT_H
