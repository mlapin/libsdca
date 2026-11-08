#ifndef SDCA_SOLVER_OBJECTIVE_L2_ENTROPY_TOPK_H
#define SDCA_SOLVER_OBJECTIVE_L2_ENTROPY_TOPK_H

#include "sdca/math/entropy.h"
#include "sdca/prox/topk_entropy.h"
#include "sdca/prox/topk_entropy_biased.h"
#include "sdca/solver/objective/objective_base.h"
#include "sdca/utility/types.h"

namespace sdca {

template <typename Data, typename Result>
struct l2_entropy_topk : public objective_base<Data, Result> {
  typedef Data data_type;
  typedef Result result_type;

  typedef objective_base<Data, Result> base;

  const Result c;
  const size_type k;

  l2_entropy_topk(const Result c_, const size_type k_) : base::objective_base(c_), c(c_), k(k_) {
    if (!(c > 0) || !std::isfinite(c)) throw std::invalid_argument("C must be finite and positive");
    if (k < 1) throw std::invalid_argument("k must be positive");
  }

  // The same objective in another precision, for the evaluation.
  template <typename D, typename R>
  explicit l2_entropy_topk(const l2_entropy_topk<D, R>& other)
      : l2_entropy_topk(static_cast<Result>(other.c), other.k) {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_entropy_topk (c: " << c << ", k: " << k << ", precision: " << type_name<Result>()
        << ")";
    return str.str();
  }

  template <typename Int>
  void update_dual_variables(const Int num_classes, const Data norm2, Data* variables, Data* scores,
                             const projection_precision = projection_precision::float64) const {
    detail::check_topk(k, num_classes);
    if (k == 1) {
      entropy_update(num_classes, Int(1), c, norm2, variables, scores);
      return;
    }
    using Work = entropy_block_type<Data, Result>;
    const std::size_t m = static_cast<std::size_t>(num_classes - 1);
    if (k == m) {
      // Average before shifting every score by the largest competitor.
      // Otherwise [1e16,1,-1e16] loses its middle score irreversibly.
      using Long = long double;
      Long score_low, dual_low(0), dual_mean;
      const Long score_sum = detail::entropy_sum_offset(
          m, [&](std::size_t j) { return scores[j + 1]; }, -Long(scores[0]), score_low);
      // Cold starts and every all-capped update have identical competitor
      // duals. Check that invariant before skipping the compensated pass;
      // arbitrary caller-supplied warm starts still use the full mean.
      bool uniform_dual = std::isfinite(variables[1]);
      for (std::size_t j = 1; uniform_dual && j < m; ++j)
        uniform_dual = variables[j + 1] == variables[1];
      Long dual_sum;
      if (uniform_dual) {
        Long gap_lo, sum_lo;
        dual_mean = detail::two_sum(Long(variables[1]), -Long(variables[0]), gap_lo);
        dual_sum = detail::split_product(Long(m), dual_mean, sum_lo);
        dual_low = sum_lo + Long(m) * gap_lo;
      } else {
        dual_sum = detail::entropy_sum_offset(
            m, [&](std::size_t j) { return variables[j + 1]; }, -Long(variables[0]), dual_low);
        dual_mean = (dual_sum + dual_low) / Long(m);
      }
      // Average scores and duals separately: score - norm2*dual can lose
      // the entire dual contribution beside cancelling huge scores.
      const Work plain = Work((score_sum + score_low) / Long(m) - Long(norm2) * dual_mean);
      // Exact sums minus an exact product keep a small residual between huge
      // cancelling score and dual sums; the plain mean stays within two ulps.
      Long product_lo, difference_lo;
      const Long product = detail::split_product(Long(norm2), dual_sum, product_lo);
      const Long difference = detail::two_sum(score_sum, -product, difference_lo);
      const Long exact =
          (difference + (difference_lo + (score_low - (product_lo + Long(norm2) * dual_low)))) /
          Long(m);
      const Long tol = 2 * Long(std::numeric_limits<Work>::epsilon()) * std::abs(Long(plain));
      const Work offset =
          std::abs(exact - Long(plain)) > tol && std::isfinite(exact) ? Work(exact) : plain;
      Work cap(0), scaled_cap(0), logs(0);
      detail::topk_entropy_all_capped(m, offset, Work(c) * Work(norm2), Work(c), cap, scaled_cap,
                                      logs);
      using Sum = typename std::common_type<Work, double>::type;  // see detail::duals
      Sum sum(0);
      for (Int j = 1; j < num_classes; ++j) {
        variables[j] = Data(-scaled_cap);
        sum -= Sum(variables[j]);
      }
      variables[0] = Data(sum);
      return;
    }
    detail::entropy_buffer<Work> v(m), probabilities(m);
    Work offset, offset_low(0);
    bool far = false;
    const Int r = static_cast<Int>(std::max_element(scores + 1, scores + num_classes) - scores);
    for (Int j = 1; j < num_classes; ++j) {
      const Work x =
          detail::warm_log_odds<Work>(scores[j], scores[r], variables[j], variables[r], norm2);
      v.data()[static_cast<std::size_t>(j - 1)] = x;
      far |= std::abs(x) > Work(64);
    }
    offset = detail::warm_log_odds<Work>(scores[r], scores[0], variables[r], variables[0], norm2);
    // Scores relative to the largest keep their distance from it to 53 bits; beyond 64
    // from it their exact low parts keep the free scores' differences in one solve.
    detail::entropy_buffer<Work> low(far ? m : 0);
    if (far) {
      for (Int j = 1; j < num_classes; ++j) {
        const std::size_t i = static_cast<std::size_t>(j - 1);
        low.data()[i] = Work(0);
        if (std::abs(v.data()[i]) > Work(64))
          v.data()[i] = detail::warm_log_odds_split<Work>(scores[j], scores[r], variables[j],
                                                          variables[r], norm2, low.data()[i]);
      }
    }
    if (std::abs(offset) > Work(64))
      offset = detail::warm_log_odds_split<Work>(scores[r], scores[0], variables[r], variables[0],
                                                 norm2, offset_low);
    topk_entropy_block(m, v.data(), k, Work(c) * Work(norm2), probabilities.data(),
                       static_cast<Work*>(nullptr), static_cast<Work*>(nullptr), offset, Work(c),
                       static_cast<topk_entropy_free<Work>*>(nullptr),
                       far ? low.data() : static_cast<Work*>(nullptr), offset_low);
    using Sum = typename std::common_type<Work, double>::type;
    Sum sum(0);
    for (Int j = 1; j < num_classes; ++j) {
      variables[j] = Data(-probabilities.data()[static_cast<std::size_t>(j - 1)]);
      sum -= Sum(variables[j]);
    }
    variables[0] = Data(sum);
  }

  template <typename Int>
  void zero_scores_dual_variables(const Int num_classes, Data* variables) const {
    // At zero scores the top-k caps are inactive: the softmax block.
    detail::zero_scores_dual(num_classes, Int(1), c * Result(num_classes - 1) / Result(num_classes),
                             variables);
  }

  template <typename Int>
  inline Result primal_loss(const Int num_tasks, Data* scores) const {
    detail::check_topk(k, num_tasks);
    using Work = typename std::common_type<Data, Result, double>::type;
    if (k == static_cast<size_type>(num_tasks - 1)) {
      const long double mean = detail::entropy_mean_offset(
          num_tasks - 1, [&](Int j) { return scores[j + 1]; },
          -static_cast<long double>(scores[0]));
      return Result(softplus(Work(mean) + std::log(Work(k))));
    }
    const std::size_t m = static_cast<std::size_t>(num_tasks - 1);
    detail::entropy_buffer<Work> buffer(m);
    Work* v = buffer.data();
    const Work s0 = Work(scores[0]);
    const Work smax = Work(*std::max_element(scores + 1, scores + num_tasks));
    for (Int j = 1; j < num_tasks; ++j) v[static_cast<std::size_t>(j - 1)] = Work(scores[j]);
    auto th = thresholds_topk_entropy_offset<Work>(v, v + m, static_cast<diff_type>(k), -s0);
    const Work u = static_cast<Work>(std::distance(v, th.first)), K = static_cast<Work>(k),
               t = Work(th.t);
    Work head(0), rest(0);
    for (Work* it = v; it != th.first; ++it) head += *it - smax;
    const Work* top = std::max_element(th.first, v + m);
    for (Work* it = th.first; it != v + m; ++it)
      if (it != top) rest += std::exp(*it - t);
    const Work lse = (t - smax) + std::log1p(rest);
    const Work log_odds =
        (smax - s0) + (head + (K - u) * (lse - std::log(K - u))) / K + std::log(K);
    return Result(softplus(log_odds));
  }

  template <typename Int, typename T>
  inline void feasible_dual(const Int num_classes, T* variables) const {
    feasible::topk_simplex(num_classes, variables, k, T(c));
  }

  template <typename Int>
  inline Result dual_loss(const Int num_classes, const Data* variables) const {
    return entropy_dual_loss(num_classes, Int(1), c, variables);
  }
};

}  // namespace sdca

#endif
