#ifndef SDCA_PROX_TWO_ENTROPY_H
#define SDCA_PROX_TWO_ENTROPY_H

#include "sdca/math/functor.h"
#include "sdca/math/entropy.h"
#include "sdca/math/log_exp.h"
#include "sdca/prox/proxdef.h"

namespace sdca {

/**
 * Solve
 *    min_{x,y} 0.5 * alpha * ||x - a / alpha - 1 / p||^2 + <x, log(x)> +
 *              0.5 * alpha * ||y - b / alpha||^2 + <y, log(y)>
 *    s.t.      <1, x> + <1, y> = 1
 *              0 <= x_i,  0 <= y_j
 **/
template <typename Result = double, typename Iterator>
inline auto thresholds_two_entropy(Iterator a_first, Iterator a_last, Iterator b_first,
                                   Iterator b_last, const Result alpha = 1)
    -> std::pair<
        generalized_thresholds<
            typename std::common_type<Result, typename std::iterator_traits<Iterator>::value_type,
                                      double>::type,
            Iterator,
            scaled_lambert_w_exp_map<typename std::common_type<
                Result, typename std::iterator_traits<Iterator>::value_type, double>::type>>,
        generalized_thresholds<
            typename std::common_type<Result, typename std::iterator_traits<Iterator>::value_type,
                                      double>::type,
            Iterator,
            scaled_lambert_w_exp_map<typename std::common_type<
                Result, typename std::iterator_traits<Iterator>::value_type, double>::type>>> {
  using Data = typename std::iterator_traits<Iterator>::value_type;
  using Work = typename std::common_type<Result, Data, double>::type;
  const std::size_t p = static_cast<std::size_t>(std::distance(a_first, a_last));
  const std::size_t q_len = static_cast<std::size_t>(std::distance(b_first, b_last));
  const std::size_t n = p + q_len;
  if (p == 0 || q_len == 0) throw std::invalid_argument("Empty entropy block");
  detail::entropy_buffer<Work> v_buf(n), r_buf(n);
  Work* v = v_buf.data();
  for (std::size_t j = 0; j < p; ++j) v[j] = Work(*(a_first + static_cast<std::ptrdiff_t>(j)));
  for (std::size_t j = 0; j < q_len; ++j)
    v[p + j] = Work(*(b_first + static_cast<std::ptrdiff_t>(j)));
  // Centered at the largest score so that differences survive a far-off block.
  const Work origin = *std::max_element(v, v + n), q = Work(1) / Work(p);
  for (std::size_t j = 0; j < n; ++j) v[j] -= origin;
  Work normal;
  entropy_prox_dual_ptr(n, p, Work(1), Work(alpha), v, r_buf.data(),
                        static_cast<std::vector<Work>*>(nullptr), &normal);
  const Work shift = std::log(Work(alpha)) + std::log(q) - normal;
  return std::make_pair(
      make_thresholds(origin, Work(0), Work(1), a_first, a_last,
                      scaled_lambert_w_exp_map<Work>(Work(alpha), shift + Work(alpha) * q)),
      make_thresholds(origin, Work(0), Work(1), b_first, b_last,
                      scaled_lambert_w_exp_map<Work>(Work(alpha), shift)));
}

// [a_first, a_last) and [b_first, b_last) in place, through temporary copies.
template <typename Result = double, typename Iterator>
inline void prox_two_entropy(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,
                             const Result alpha = 1) {
  using Data = typename std::iterator_traits<Iterator>::value_type;
  using Work = typename std::common_type<Result, Data, double>::type;
  const std::size_t p = static_cast<std::size_t>(std::distance(a_first, a_last));
  const std::size_t q_len = static_cast<std::size_t>(std::distance(b_first, b_last));
  const std::size_t n = p + q_len;
  if (p == 0 || q_len == 0) throw std::invalid_argument("Empty entropy block");
  detail::entropy_buffer<Work> v_buf(n), r_buf(n);
  Work* v = v_buf.data();
  for (std::size_t j = 0; j < p; ++j) v[j] = Work(*(a_first + static_cast<std::ptrdiff_t>(j)));
  for (std::size_t j = 0; j < q_len; ++j)
    v[p + j] = Work(*(b_first + static_cast<std::ptrdiff_t>(j)));
  const Work origin = *std::max_element(v, v + n);
  for (std::size_t j = 0; j < n; ++j) v[j] -= origin;
  std::vector<Work> probabilities;
  entropy_prox_dual_ptr(n, p, Work(1), Work(alpha), v, r_buf.data(), &probabilities);
  for (std::size_t j = 0; j < p; ++j)
    *(a_first + static_cast<std::ptrdiff_t>(j)) = Data(probabilities[j]);
  for (std::size_t j = p; j < n; ++j)
    *(b_first + static_cast<std::ptrdiff_t>(j - p)) = Data(probabilities[j]);
}

// [a_first, a_last) and [b_first, b_last) in place; a_aux and b_aux are as long.
template <typename Result = double, typename Iterator>
inline void prox_two_entropy(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,
                             Iterator a_aux, Iterator b_aux, const Result alpha = 1) {
  (void)a_aux;
  (void)b_aux;
  prox_two_entropy<Result>(a_first, a_last, b_first, b_last, alpha);
}

// [first, middle) and [middle, last) in place, through a temporary copy.
template <typename Result = double, typename Iterator>
inline void prox_two_entropy(Iterator first, Iterator middle, Iterator last,
                             const Result alpha = 1) {
  prox_two_entropy<Result>(first, middle, middle, last, alpha);
}

// [first, last) in place; aux (as long) is scratch.
template <typename Result = double, typename Iterator>
inline void prox_two_entropy(std::ptrdiff_t p, Iterator first, Iterator last, Iterator aux,
                             const Result alpha = 1) {
  if (p <= 0 || p >= std::distance(first, last))
    throw std::invalid_argument("Invalid entropy block dimensions");
  prox_two_entropy<Result>(first, first + p, first + p, last, aux, aux + p, alpha);
}

}  // namespace sdca

#endif
