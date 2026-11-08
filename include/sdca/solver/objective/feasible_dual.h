#ifndef SDCA_SOLVER_OBJECTIVE_FEASIBLE_DUAL_H
#define SDCA_SOLVER_OBJECTIVE_FEASIBLE_DUAL_H

#include <algorithm>
#include <cstddef>
#include <limits>

namespace sdca {

// The evaluation's repair of a stored dual block (labels first), in a wider
// type: storage rounding can break its balance and bounds by an ulp. Feasible
// blocks only get their balancing first entry recomputed.
namespace feasible {

// b_j = -a_j in [0, cap] for j >= 1, sum b <= total, and a_0 = sum b.
template <typename Int, typename T>
inline void capped(const Int n, T* a, const T cap, const T total) {
  T sum(0);
  for (Int j = 1; j < n; ++j) {
    a[j] = -std::min(std::max(-a[j], T(0)), cap);
    sum -= a[j];
  }
  if (sum > total) {
    const T scale = total / sum;
    sum = 0;
    for (Int j = 1; j < n; ++j) {
      a[j] *= scale;
      sum -= a[j];
    }
  }
  a[0] = sum;
}

// The top-k simplex: b_j in [0, sum b / k], sum b <= total, a_0 = sum b.
template <typename Int, typename T>
inline void topk_simplex(const Int n, T* a, const std::size_t k, const T total) {
  capped(n, a, std::numeric_limits<T>::infinity(), total);
  // The entries J above the cap all take t = rest / (k - |J|), the cap of
  // the clipped block; J grows (and t falls) until no other entry exceeds t.
  T t = a[0] / T(k);
  for (Int pass = 0; pass < n; ++pass) {
    std::size_t count = 0;
    T rest(0);
    for (Int j = 1; j < n; ++j) {
      if (-a[j] > t)
        ++count;
      else
        rest -= a[j];
    }
    if (count == 0 || count >= k) break;
    const T next = rest / T(k - count);
    if (!(next < t)) break;
    t = next;
  }
  T sum(0);
  for (Int j = 1; j < n; ++j) {
    a[j] = std::max(a[j], -t);
    sum -= a[j];
  }
  a[0] = sum;
}

// q - a / c on the probability simplex, q uniform on the p labels:
// a_j <= c / p on the labels, a_j <= 0 elsewhere, sum a = 0.
template <typename Int, typename T>
inline void label_distribution(const Int n, const Int p, T* a, const T c) {
  const T hi = c / T(p);
  T sum(0);
  for (Int j = 0; j < n; ++j) {
    a[j] = std::min(a[j], j < p ? hi : T(0));
    sum += a[j];
  }
  if (sum > 0) {
    a[0] -= sum;
    return;
  }
  // Raise entries below their bounds by the deficit.
  T deficit = -sum;
  for (Int j = 0; j < n && deficit > 0; ++j) {
    const T room = std::min((j < p ? hi : T(0)) - a[j], deficit);
    a[j] += room;
    deficit -= room;
  }
}

// a_j >= 0 on the p labels, a_j <= 0 elsewhere, both sums equal and <= c.
template <typename Int, typename T>
inline void two_simplex(const Int n, const Int p, T* a, const T c) {
  T pos(0), neg(0);
  for (Int j = 0; j < p; ++j) pos += a[j] = std::max(a[j], T(0));
  for (Int j = p; j < n; ++j) neg -= a[j] = std::min(a[j], T(0));
  const T target = std::min(std::min(pos, neg), c);
  T scaled(0);
  for (Int j = p; j < n; ++j) {
    if (neg > target) a[j] *= target / neg;
    scaled -= a[j];
  }
  if (pos > 0) {
    // Scale the labels to the other side's sum, so that the block sums to zero.
    Int top = 0;
    T sum(0);
    for (Int j = 0; j < p; ++j) {
      a[j] *= scaled / pos;
      sum += a[j];
      if (a[j] > a[top]) top = j;
    }
    a[top] = std::max(T(0), a[top] + (scaled - sum));
  } else if (scaled > 0) {
    for (Int j = p; j < n; ++j) a[j] = 0;
  }
}

}  // namespace feasible
}  // namespace sdca

#endif
