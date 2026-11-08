#ifndef SDCA_PROX_TOPK_ENTROPY_H
#define SDCA_PROX_TOPK_ENTROPY_H

#include "sdca/math/functor.h"
#include "sdca/math/entropy.h"
#include "sdca/math/log_exp.h"
#include "sdca/prox/proxdef.h"

namespace sdca {

namespace detail {

// The number u of capped scores among the n scores v (partially sorted here), the
// sum of the capped ones relative to wmax, and the log-sum-exp of the others
// relative to the largest of them, v[u]. Only the k largest can be capped: the
// rest are summed once, relative to v[k - 1], and the sums of the scores after
// each of the k largest relative to it follow from one backward recurrence,
//   rest[a] = exp(v[a + 1] - v[a]) (1 + rest[a + 1]),
// whose factors are at most one: one pass instead of one per candidate.
template <typename Work>
inline std::size_t topk_entropy_active_set(Work* v, std::size_t n, std::size_t k, Work wmax,
                                           Work& sum, Work& lse) {
  const Work K = Work(k), eps = Work(32) * std::numeric_limits<Work>::epsilon();
  detail::sort_largest(v, k, v + n, std::greater<Work>());
  Work tail_k(0);
  const Work vk = v[k - 1];
  for (std::size_t j = k; j < n; ++j) tail_k += std::exp(v[j] - vk);
  entropy_buffer<Work> rest_buffer(k);
  Work* rest = rest_buffer.data();
  rest[k - 1] = tail_k;
  for (std::size_t a = k - 1; a > 1; --a) rest[a - 1] = std::exp(v[a] - v[a - 1]) * (1 + rest[a]);
  sum = v[0] - wmax;
  for (std::size_t active = 1; active < k; ++active) {
    // The active-set test relative to va (as torch._entropy_topk).
    const Work threshold = std::log1p(rest[active]) - std::log(K - Work(active));
    if (Work(0) <= threshold + eps * std::max(Work(1), std::abs(threshold))) {
      // The log-sum-exp of the free scores from a direct sum.
      const Work va = v[active], scale = std::exp(vk - va);
      Work direct(0);
      if (scale > 0 && std::isfinite(tail_k)) {
        for (std::size_t j = active + 1; j < k; ++j) direct += std::exp(v[j] - va);
        direct += scale * tail_k;
      } else {
        for (std::size_t j = active + 1; j < n; ++j) direct += std::exp(v[j] - va);
      }
      lse = std::log1p(direct);
      return active;
    }
    sum += v[active] - wmax;
  }
  throw std::runtime_error("Top-k entropy active set did not converge");
}

}  // namespace detail

/**
 * Solve
 *    min_{x,s} <x, log(x)> + (1 - s) * log(1 - s) - <a, x>
 *              <1, x> = s
 *              s <= 1
 *              0 <= x_i <= s / k
 *
 * The solution is
 *    x = max(0, min(exp(a - t), hi))
 * with t the largest uncapped score, so differences among the uncapped ones
 * survive a far larger capped one.
 **/
template <typename Result = double, typename Iterator>
inline auto thresholds_topk_entropy_offset(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result offset = 0)
    -> generalized_thresholds<
        typename std::common_type<Result, typename std::iterator_traits<Iterator>::value_type,
                                  double>::type,
        Iterator,
        capped_entropy_map<typename std::common_type<
            Result, typename std::iterator_traits<Iterator>::value_type, double>::type>> {
  using Data = typename std::iterator_traits<Iterator>::value_type;
  using Work = typename std::common_type<Result, Data, double>::type;
  const auto n = std::distance(first, last);
  if (k < 1 || k > n || !std::isfinite(offset)) {
    throw std::invalid_argument("Invalid top-k projection dimensions");
  }
  for (auto it = first; it != last; ++it) {
    if (!std::isfinite(*it)) {
      throw std::invalid_argument("Top-k entropy scores must be finite");
    }
  }
  if (k == n) {
    const Work mean =
        Work(detail::entropy_mean_offset(n, [&](decltype(n) j) { return *(first + j); }, offset));
    const Work cap = std::exp(-softplus(-(mean + std::log(Work(k))))) / Work(k);
    return make_thresholds(Work(0), cap, cap, last, last,
                           capped_entropy_map<Work>(Work(0), cap, false));
  }
  auto max_it = std::max_element(first, last);
  const Work wmax = Work(*max_it), origin = wmax + Work(offset), K = Work(k);
  const Work eps = Work(32) * std::numeric_limits<Work>::epsilon();

  // Fast check for active == 0 (always holds for k == 1) without sorting or allocating.
  Work tail0(0);
  for (auto it = first; it != last; ++it) {
    if (it != max_it) tail0 += std::exp(Work(*it) - wmax);
  }
  Work lse0 = std::log1p(tail0);
  Work thresh0 = lse0 - std::log(K);
  if (k == 1 || Work(0) <= thresh0 + eps * std::max(Work(1), std::abs(thresh0))) {
    Work odds = origin + thresh0 + std::log(K);
    Work logs = -softplus(-odds), cap = std::exp(logs) / K;
    Work shift = logs - lse0;
    return make_thresholds(wmax, Work(0), cap, first, last,
                           capped_entropy_map<Work>(shift, cap, false));
  }

  const std::size_t un = static_cast<std::size_t>(n), uk = static_cast<std::size_t>(k);
  detail::entropy_buffer<Work> v_buf(un);
  Work* v = v_buf.data();
  std::size_t idx = 0;
  // The scores themselves, compared with each other rather than with wmax: free
  // scores far below a capped one keep their differences.
  for (auto it = first; it != last; ++it) v[idx++] = Work(*it);
  Work sum(0), lse(0);
  const std::size_t ua = detail::topk_entropy_active_set(v, un, uk, wmax, sum, lse);
  const auto active = static_cast<typename std::iterator_traits<Iterator>::difference_type>(ua);
  const Work remaining = K - Work(ua);
  // The first free score is the map's origin, so prox keeps its differences too.
  const Work origin_free = v[ua];
  Work odds = origin + (sum + remaining * ((origin_free - wmax) + lse - std::log(remaining))) / K +
              std::log(K);
  Work logs = -softplus(-odds), cap = std::exp(logs) / K;
  Work shift = logs + std::log(remaining / K) - lse;
  auto middle = first + active;
  if (active > 0) std::nth_element(first, middle, last, std::greater<Data>());
  return make_thresholds(origin_free, Work(0), cap, middle, last,
                         capped_entropy_map<Work>(shift, cap, false));
}

template <typename Result = double, typename Iterator>
inline auto thresholds_topk_entropy(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1)
    -> decltype(thresholds_topk_entropy_offset<Result>(first, last, k, Result(0))) {
  return thresholds_topk_entropy_offset<Result>(first, last, k, Result(0));
}

// [first, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_topk_entropy(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1) {
  prox(first, last,
       thresholds_topk_entropy<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       k);
}

// [first, last) in place; aux (as long) is scratch.
template <typename Result = double, typename Iterator>
inline void prox_topk_entropy(
    Iterator first, Iterator last, Iterator aux,
    const typename std::iterator_traits<Iterator>::difference_type k = 1) {
  prox(first, last, aux, thresholds_topk_entropy<Result, Iterator>, k);
}

// Each block of dim consecutive entries of [first, last) in place; aux holds dim
// entries.
template <typename Result = double, typename Iterator>
inline void prox_topk_entropy(
    const typename std::iterator_traits<Iterator>::difference_type dim, Iterator first,
    Iterator last, Iterator aux,
    const typename std::iterator_traits<Iterator>::difference_type k = 1) {
  prox(dim, first, last, aux, thresholds_topk_entropy<Result, Iterator>, k);
}

}  // namespace sdca

#endif
