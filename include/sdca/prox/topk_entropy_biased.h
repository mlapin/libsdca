#ifndef SDCA_PROX_TOPK_ENTROPY_BIASED_H
#define SDCA_PROX_TOPK_ENTROPY_BIASED_H

#include "sdca/math/entropy.h"
#include "sdca/math/functor.h"
#include "sdca/math/log_exp.h"
#include "sdca/prox/proxdef.h"

namespace sdca {

/**
 * Solve
 *    min_{x,s} 0.5 * alpha * (<x, x> + s * s)
 *              + <x, log(x)> + (1 - s) * log(1 - s) - <a, x>
 *    s.t.      <1, x> = s
 *              s <= 1
 *              0 <= x_i <= s / k
 *
 * The solution is
 *    x = max(0, min(lambert_w_exp(a - t) / alpha, hi))
 **/
template <typename Result = double, typename Iterator>
inline generalized_thresholds<
    typename std::common_type<Result, typename std::iterator_traits<Iterator>::value_type,
                              double>::type,
    Iterator,
    scaled_lambert_w_exp_map<typename std::common_type<
        Result, typename std::iterator_traits<Iterator>::value_type, double>::type>>
thresholds_topk_entropy_biased(Iterator first, Iterator last,
                               const typename std::iterator_traits<Iterator>::difference_type k = 1,
                               const Result alpha = 1) {
  typedef typename std::iterator_traits<Iterator>::value_type Data;
  typedef typename std::common_type<Result, Data, double>::type Work;
  if (!(alpha > 0) || !std::isfinite(alpha) || k < 1 || k > std::distance(first, last))
    throw std::invalid_argument("Invalid top-k entropy dimensions or scale");

  // Auxiliary variables
  const Work a = Work(alpha), lo(0);
  Work s(0), t(0);

  // For k = 1 no entry can reach the bound s / k: this is the entropy block
  // of a target with score 0 and the inputs as the other scores,
  //    x_i = W(exp(a_i - t)) / alpha,   s = sum_i x_i,
  // solved in residual coordinates so that s stays accurate for large alpha.
  const std::size_t n = static_cast<std::size_t>(std::distance(first, last));
  if (k == 1) {
    detail::entropy_buffer<Work> v(n + 1), r(n + 1);
    v.data()[0] = Work(0);
    r.data()[0] = Work(0);
    for (std::size_t j = 0; j < n; ++j)
      v.data()[j + 1] = Work(*(first + static_cast<std::ptrdiff_t>(j)));
    entropy_prox_dual_ptr(n + 1, std::size_t(1), Work(1), a, v.data(), r.data(),
                          static_cast<std::vector<Work>*>(nullptr), &t);
    s = r.data()[0];
    t -= std::log(a);
    return make_thresholds(Work(0), lo, s, first, last, scaled_lambert_w_exp_map<Work>(a, -t));
  }

  // For k > 1 solve the block in the logit of the total mass s
  // (topk_entropy_block), which reports the normalizer and cap of
  //    x_i = min(V(log(alpha) + a_i - t) / alpha, s / k).
  detail::entropy_buffer<Work> v(n), x(n);
  for (std::size_t j = 0; j < n; ++j) v.data()[j] = Work(*(first + static_cast<std::ptrdiff_t>(j)));
  Work cap(0);
  topk_entropy_free<Work> largest_free;
  topk_entropy_block(n, v.data(), static_cast<std::size_t>(k), a, x.data(), &t, &cap, Work(0),
                     Work(1), &largest_free);
  if (largest_free.index == n)
    return make_thresholds(Work(0), lo, cap, first, last,
                           scaled_lambert_w_exp_map<Work>(a, std::log(a) - t));
  // The map takes the scores relative to the largest free one, which keeps their
  // differences beside a far larger capped score.
  return make_thresholds(v.data()[largest_free.index], lo, cap, first, last,
                         scaled_lambert_w_exp_map<Work>(a, std::log(a) - largest_free.normalizer));
}

// [first, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_topk_entropy_biased(
    Iterator first, Iterator last,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result alpha = 1) {
  prox(first, last,
       thresholds_topk_entropy_biased<
           Result,
           typename std::vector<typename std::iterator_traits<Iterator>::value_type>::iterator>,
       k, alpha);
}

// [first, last) in place; aux (as long) is scratch.
template <typename Result = double, typename Iterator>
inline void prox_topk_entropy_biased(
    Iterator first, Iterator last, Iterator aux,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result alpha = 1) {
  prox(first, last, aux, thresholds_topk_entropy_biased<Result, Iterator>, k, alpha);
}

// Each block of dim consecutive entries of [first, last) in place; aux holds dim
// entries.
template <typename Result = double, typename Iterator>
inline void prox_topk_entropy_biased(
    const typename std::iterator_traits<Iterator>::difference_type dim, Iterator first,
    Iterator last, Iterator aux,
    const typename std::iterator_traits<Iterator>::difference_type k = 1, const Result alpha = 1) {
  prox(dim, first, last, aux, thresholds_topk_entropy_biased<Result, Iterator>, k, alpha);
}

}  // namespace sdca

#endif
