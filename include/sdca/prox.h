#ifndef SDCA_PROX_H
#define SDCA_PROX_H

// Projections and proximal operators (docs/CPP.md, Projections and proximal operators).

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "sdca/core/project.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN

using core::lambert_w_exp;

using core::generalized_thresholds;
using core::thresholds;

using core::thresholds_entropy;
using core::thresholds_entropy_norm;
using core::thresholds_knapsack_eq;
using core::thresholds_knapsack_le;
using core::thresholds_knapsack_le_biased;
using core::thresholds_topk_cone;
using core::thresholds_topk_cone_biased;
using core::thresholds_topk_entropy;
using core::thresholds_topk_entropy_biased;
using core::thresholds_topk_entropy_offset;
using core::thresholds_topk_simplex;
using core::thresholds_topk_simplex_biased;
using core::thresholds_two_entropy;
using core::thresholds_two_simplex;

// Writes the solution x of a projection's thresholds (thresholds_NAME) over [first, last).
template <typename Result, typename ThresholdIterator, typename Iterator>
inline void prox(const thresholds<Result, ThresholdIterator>& t, Iterator first, Iterator last) {
  core::apply(t, first, last);
}

template <typename Result, typename ThresholdIterator, typename Iterator, typename Mapping>
inline void prox(const generalized_thresholds<Result, ThresholdIterator, Mapping>& t,
                 Iterator first, Iterator last) {
  core::apply(t, first, last);
}

namespace prox_detail {

template <typename Difference>
inline void check_blocks(const Difference dim, const Difference size) {
  if (dim <= 0 || size % dim != 0)
    throw std::invalid_argument("Invalid projection batch dimensions");
}

template <typename Difference>
inline void check_two_blocks(const Difference dim, const Difference p, const Difference size) {
  if (dim <= 0 || p <= 0 || p >= dim || size % dim != 0)
    throw std::invalid_argument("Invalid two-block projection dimensions");
}

template <typename Iterator>
inline void check_block(const Iterator first, const Iterator last) {
  if (first == last) throw std::invalid_argument("Empty projection input");
}

template <typename Iterator>
inline void check_block(const Iterator a_first, const Iterator a_last, const Iterator b_first,
                        const Iterator b_last) {
  if (a_first == a_last || b_first == b_last) throw std::invalid_argument("Empty projection block");
}

template <typename Thresholds, typename Iterator>
inline void apply_pair(const Thresholds& t, Iterator a_first, Iterator a_last, Iterator b_first,
                       Iterator b_last) {
  core::apply(t.first, a_first, a_last);
  core::apply(t.second, b_first, b_last);
}

}  // namespace prox_detail

#define SDCA_UNPAREN(...) __VA_ARGS__

// prox_NAME over [first, last) through a temporary copy, through scratch aux (as long), and
// over each block of dim entries with scratch aux of dim entries.
#define SDCA_PROX_OVERLOADS(NAME, PARAMS, ARGS)                                                    \
  template <typename Result = double, typename Iterator>                                           \
  inline void prox_##NAME(Iterator first, Iterator last, SDCA_UNPAREN PARAMS) {                    \
    prox_detail::check_block(first, last);                                                         \
    std::vector<typename std::iterator_traits<Iterator>::value_type> aux(first, last);             \
    core::apply(core::thresholds_##NAME<Result>(aux.begin(), aux.end(), SDCA_UNPAREN ARGS), first, \
                last);                                                                             \
  }                                                                                                \
  template <typename Result = double, typename Iterator>                                           \
  inline void prox_##NAME(Iterator first, Iterator last, Iterator aux, SDCA_UNPAREN PARAMS) {      \
    prox_detail::check_block(first, last);                                                         \
    const Iterator aux_last = std::copy(first, last, aux);                                         \
    core::apply(core::thresholds_##NAME<Result>(aux, aux_last, SDCA_UNPAREN ARGS), first, last);   \
  }                                                                                                \
  template <typename Result = double, typename Iterator>                                           \
  inline void prox_##NAME(const typename std::iterator_traits<Iterator>::difference_type dim,      \
                          Iterator first, Iterator last, Iterator aux, SDCA_UNPAREN PARAMS) {      \
    prox_detail::check_blocks(dim, std::distance(first, last));                                    \
    for (; first != last; first += dim) {                                                          \
      const Iterator aux_last = std::copy(first, first + dim, aux);                                \
      core::apply(core::thresholds_##NAME<Result>(aux, aux_last, SDCA_UNPAREN ARGS), first,        \
                  first + dim);                                                                    \
    }                                                                                              \
  }

#define SDCA_K_PARAM const typename std::iterator_traits<Iterator>::difference_type k = 1

SDCA_PROX_OVERLOADS(knapsack_eq, (const Result lo = 0, const Result hi = 1, const Result rhs = 1),
                    (lo, hi, rhs))
SDCA_PROX_OVERLOADS(knapsack_le, (const Result lo = 0, const Result hi = 1, const Result rhs = 1),
                    (lo, hi, rhs))
SDCA_PROX_OVERLOADS(knapsack_le_biased,
                    (const Result lo = 0, const Result hi = 1, const Result rhs = 1,
                     const Result rho = 1),
                    (lo, hi, rhs, rho))
SDCA_PROX_OVERLOADS(topk_simplex, (SDCA_K_PARAM, const Result rhs = 1), (k, rhs))
SDCA_PROX_OVERLOADS(topk_simplex_biased, (SDCA_K_PARAM, const Result rhs = 1, const Result rho = 1),
                    (k, rhs, rho))
SDCA_PROX_OVERLOADS(topk_cone, (SDCA_K_PARAM), (k))
SDCA_PROX_OVERLOADS(topk_cone_biased, (SDCA_K_PARAM, const Result rho = 1), (k, rho))
SDCA_PROX_OVERLOADS(entropy, (const Result hi = 1, const Result rhs = 1), (hi, rhs))
SDCA_PROX_OVERLOADS(entropy_norm, (const Result hi = 1, const Result rhs = 1), (hi, rhs))
SDCA_PROX_OVERLOADS(topk_entropy, (SDCA_K_PARAM), (k))
SDCA_PROX_OVERLOADS(topk_entropy_biased, (SDCA_K_PARAM, const Result alpha = 1), (k, alpha))

// prox_NAME of two blocks: two ranges through temporary copies, two ranges with scratch of
// their lengths, [first, middle) and [middle, last), the first p entries and the rest, and
// each block of dim entries split at p.
#define SDCA_PROX_TWO_BLOCK_OVERLOADS(NAME, PARAMS, ARGS)                                         \
  template <typename Result = double, typename Iterator>                                          \
  inline void prox_##NAME(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,   \
                          SDCA_UNPAREN PARAMS) {                                                  \
    prox_detail::check_block(a_first, a_last, b_first, b_last);                                   \
    typedef std::vector<typename std::iterator_traits<Iterator>::value_type> Copy;                \
    Copy a(a_first, a_last), b(b_first, b_last);                                                  \
    prox_detail::apply_pair(core::thresholds_##NAME<Result>(a.begin(), a.end(), b.begin(),        \
                                                            b.end(), SDCA_UNPAREN ARGS),          \
                            a_first, a_last, b_first, b_last);                                    \
  }                                                                                               \
  template <typename Result = double, typename Iterator>                                          \
  inline void prox_##NAME(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,   \
                          Iterator a_aux, Iterator b_aux, SDCA_UNPAREN PARAMS) {                  \
    prox_detail::check_block(a_first, a_last, b_first, b_last);                                   \
    const Iterator a_aux_last = std::copy(a_first, a_last, a_aux);                                \
    const Iterator b_aux_last = std::copy(b_first, b_last, b_aux);                                \
    prox_detail::apply_pair(                                                                      \
        core::thresholds_##NAME<Result>(a_aux, a_aux_last, b_aux, b_aux_last, SDCA_UNPAREN ARGS), \
        a_first, a_last, b_first, b_last);                                                        \
  }                                                                                               \
  template <typename Result = double, typename Iterator>                                          \
  inline void prox_##NAME(Iterator first, Iterator middle, Iterator last, SDCA_UNPAREN PARAMS) {  \
    prox_##NAME<Result>(first, middle, middle, last, SDCA_UNPAREN ARGS);                          \
  }                                                                                               \
  template <typename Result = double, typename Iterator>                                          \
  inline void prox_##NAME(const typename std::iterator_traits<Iterator>::difference_type p,       \
                          Iterator first, Iterator last, Iterator aux, SDCA_UNPAREN PARAMS) {     \
    prox_##NAME<Result>(first, first + p, first + p, last, aux, aux + p, SDCA_UNPAREN ARGS);      \
  }                                                                                               \
  template <typename Result = double, typename Iterator>                                          \
  inline void prox_##NAME(const typename std::iterator_traits<Iterator>::difference_type dim,     \
                          const typename std::iterator_traits<Iterator>::difference_type p,       \
                          Iterator first, Iterator last, Iterator aux, SDCA_UNPAREN PARAMS) {     \
    prox_detail::check_two_blocks(dim, p, std::distance(first, last));                            \
    for (; first != last; first += dim)                                                           \
      prox_##NAME<Result>(first, first + p, first + p, first + dim, aux, aux + p,                 \
                          SDCA_UNPAREN ARGS);                                                     \
  }

SDCA_PROX_TWO_BLOCK_OVERLOADS(two_simplex, (const Result rhs = 1), (rhs))

#undef SDCA_PROX_TWO_BLOCK_OVERLOADS
#undef SDCA_PROX_OVERLOADS
#undef SDCA_K_PARAM
#undef SDCA_UNPAREN

// prox_two_entropy writes the solved probabilities (not a map of thresholds).
template <typename Result = double, typename Iterator>
inline void prox_two_entropy(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,
                             const Result alpha = 1) {
  typedef core::entropy::work<Result, Iterator> Work;
  prox_detail::check_block(a_first, a_last, b_first, b_last);
  const std::size_t p = std::size_t(std::distance(a_first, a_last));
  const std::size_t n = p + std::size_t(std::distance(b_first, b_last));
  core::row_buffer<Work> v(n);
  core::softmax_prox<Work> block(n, Work(alpha));
  core::solve_two_entropy(block, a_first, a_last, b_first, b_last, v.data());
  block.probabilities(v.data());
  std::copy(v.data(), v.data() + p, a_first);
  std::copy(v.data() + p, v.data() + n, b_first);
}

template <typename Result = double, typename Iterator>
inline void prox_two_entropy(Iterator a_first, Iterator a_last, Iterator b_first, Iterator b_last,
                             Iterator, Iterator, const Result alpha = 1) {
  prox_two_entropy<Result>(a_first, a_last, b_first, b_last, alpha);
}

template <typename Result = double, typename Iterator>
inline void prox_two_entropy(Iterator first, Iterator middle, Iterator last,
                             const Result alpha = 1) {
  prox_two_entropy<Result>(first, middle, middle, last, alpha);
}

template <typename Result = double, typename Iterator>
inline void prox_two_entropy(const typename std::iterator_traits<Iterator>::difference_type p,
                             Iterator first, Iterator last, Iterator, const Result alpha = 1) {
  if (p <= 0 || p >= std::distance(first, last))
    throw std::invalid_argument("Invalid entropy block dimensions");
  prox_two_entropy<Result>(first, first + p, first + p, last, alpha);
}

SDCA_ISA_END

// The public names in namespace sdca: those of the instruction set of the translation unit's
// flags. This header must not be first included inside an instruction-set pass of
// sdca/kernel.h.
#include "sdca/core/isa.h"
namespace sdca {
#define SDCA_EXPORT(name) using SDCA_ISA_NAMESPACE::name;
SDCA_EXPORT(lambert_w_exp)
SDCA_EXPORT(generalized_thresholds)
SDCA_EXPORT(thresholds)
SDCA_EXPORT(prox)
SDCA_EXPORT(thresholds_entropy)
SDCA_EXPORT(thresholds_entropy_norm)
SDCA_EXPORT(thresholds_knapsack_eq)
SDCA_EXPORT(thresholds_knapsack_le)
SDCA_EXPORT(thresholds_knapsack_le_biased)
SDCA_EXPORT(thresholds_topk_cone)
SDCA_EXPORT(thresholds_topk_cone_biased)
SDCA_EXPORT(thresholds_topk_entropy)
SDCA_EXPORT(thresholds_topk_entropy_biased)
SDCA_EXPORT(thresholds_topk_entropy_offset)
SDCA_EXPORT(thresholds_topk_simplex)
SDCA_EXPORT(thresholds_topk_simplex_biased)
SDCA_EXPORT(thresholds_two_entropy)
SDCA_EXPORT(thresholds_two_simplex)
SDCA_EXPORT(prox_entropy)
SDCA_EXPORT(prox_entropy_norm)
SDCA_EXPORT(prox_knapsack_eq)
SDCA_EXPORT(prox_knapsack_le)
SDCA_EXPORT(prox_knapsack_le_biased)
SDCA_EXPORT(prox_topk_cone)
SDCA_EXPORT(prox_topk_cone_biased)
SDCA_EXPORT(prox_topk_entropy)
SDCA_EXPORT(prox_topk_entropy_biased)
SDCA_EXPORT(prox_topk_simplex)
SDCA_EXPORT(prox_topk_simplex_biased)
SDCA_EXPORT(prox_two_entropy)
SDCA_EXPORT(prox_two_simplex)
#undef SDCA_EXPORT
}  // namespace sdca

#endif
