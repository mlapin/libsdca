#ifndef SDCA_SOLVER_OBJECTIVE_L2_DISTILLATION_H
#define SDCA_SOLVER_OBJECTIVE_L2_DISTILLATION_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include "sdca/math/compensated.h"
#include "sdca/math/distillation.h"
#include "sdca/math/entropy.h"
#include "sdca/prox/proxdef.h"
#include "sdca/solver/data/output.h"
#include "sdca/solver/objective/objective_base.h"
#include "sdca/utility/scratch.h"

namespace sdca {

template <typename Data, typename Result>
struct l2_distillation : public objective_base<Data, Result> {
  typedef Data data_type;
  typedef Result result_type;
  typedef objective_base<Data, Result> base;

  const Result c;

  explicit l2_distillation(const Result c_) : base::objective_base(c_), c(c_) {
    if (!(c > 0) || !std::isfinite(c)) throw std::invalid_argument("C must be finite and positive");
  }

  template <typename D, typename R>
  explicit l2_distillation(const l2_distillation<D, R>& other)
      : l2_distillation(static_cast<Result>(other.c)) {}

  inline std::string to_string() const {
    std::ostringstream str;
    str << "l2_distillation (c: " << c << ", precision: " << type_name<Result>() << ")";
    return str.str();
  }

  template <typename Int, typename Q>
  void update_dual_variables(const Int num_classes, const Q* q, const Data norm2, Data* variables,
                             Data* scores,
                             const projection_precision = projection_precision::float64) const {
    const std::size_t m = static_cast<std::size_t>(num_classes);
    const double cc = static_cast<double>(c), n2 = static_cast<double>(norm2), alpha = cc * n2;
    if (!(alpha > 0) || !std::isfinite(alpha))
      throw std::invalid_argument("Invalid distillation block scale");
    detail::entropy_buffer<double> work(14 * m + 5 * detail::row_lanes);
    detail::entropy_buffer<std::size_t> index(2 * m);
    double* v = work.data();
    std::size_t ref = 0;
    for (std::size_t j = 1; j < m; ++j)
      if (scores[j] > scores[ref]) ref = j;
    for (std::size_t j = 0; j < m; ++j)
      v[j] = detail::warm_log_odds<double>(scores[j], scores[ref], variables[j], variables[ref],
                                           norm2);
    double v_max = v[0];
    for (std::size_t j = 0; j < m; ++j) {
      // As check_entropy_block: an overflowed log-odds would turn v - v_max into NaN.
      if (!std::isfinite(v[j])) throw std::invalid_argument("Distillation scores must be finite");
      v_max = std::max(v_max, v[j]);
    }
    detail::compensated_sum<double> q_norm_sum;
    double* e_v = v + m;
    for (std::size_t j = 0; j < m; ++j) {
      e_v[j] = detail::exp_double(v[j] - v_max);
      q_norm_sum.add(e_v[j]);
    }
    const double q_norm = q_norm_sum.value(), log_q_norm = std::log(q_norm);
    double *p = v + 2 * m, *log_p = v + 3 * m, *q_prob = v + 4 * m;
    double *u = v + 5 * m, *ratio = v + 6 * m, *score = v + 7 * m;
    double *r = v + 8 * m, *slope = v + 9 * m;
    std::size_t pivot = 0;
    double top = -1.0;
    for (std::size_t j = 0; j < m; ++j) {
      const double pj = static_cast<double>(q[j]);
      const double qj = e_v[j] / q_norm;
      const double log_qj = (v[j] - v_max) - log_q_norm;
      p[j] = pj;
      log_p[j] = pj > 0 ? std::log(pj) : -std::numeric_limits<double>::infinity();
      q_prob[j] = qj;
      u[j] = qj - pj;
      ratio[j] = pj > 0 ? log_qj - log_p[j] : std::numeric_limits<double>::infinity();
      score[j] = log_qj + alpha * pj;
      if (pj + qj > top) {
        top = pj + qj;
        pivot = j;
      }
    }
    const detail::prox_solve solver = {m,
                                       alpha,
                                       std::log(alpha),
                                       ratio,
                                       u,
                                       log_p,
                                       p,
                                       q_prob,
                                       score,
                                       r,
                                       slope,
                                       v + 10 * m,
                                       v + 10 * m + 5 * detail::row_lanes,
                                       index.data()};
    solver.solve();
    detail::compensated_sum<double> sum;
    for (std::size_t j = 0; j < m; ++j) {
      variables[j] = static_cast<Data>(cc * r[j]);
      if (j != pivot) sum.add(static_cast<double>(variables[j]));
    }
    variables[pivot] = static_cast<Data>(-sum.value());
  }

  template <typename Int, typename Q>
  void zero_scores_dual_variables(const Int num_classes, const Q* q, Data* variables) const {
    const std::size_t m = static_cast<std::size_t>(num_classes);
    const double cc = static_cast<double>(c), unif = 1.0 / static_cast<double>(m);
    std::size_t pivot = 0;
    for (std::size_t j = 1; j < m; ++j)
      if (q[j] > q[pivot]) pivot = j;
    detail::compensated_sum<double> sum;
    for (std::size_t j = 0; j < m; ++j) {
      variables[j] = static_cast<Data>(cc * (static_cast<double>(q[j]) - unif));
      if (j != pivot) sum.add(static_cast<double>(variables[j]));
    }
    variables[pivot] = static_cast<Data>(-sum.value());
  }

  template <typename Int, typename Q>
  inline Result primal_loss(const Int num_classes, const Q* q, const Data* scores) const {
    const std::size_t m = static_cast<std::size_t>(num_classes);
    double z_max = static_cast<double>(scores[0]);
    std::size_t best = 0;
    for (std::size_t j = 1; j < m; ++j) {
      const double zj = static_cast<double>(scores[j]);
      if (zj > z_max) {
        z_max = zj;
        best = j;
      }
    }
    detail::compensated_sum<double> tail, cross;
    for (std::size_t j = 0; j < m; ++j) {
      const double d = static_cast<double>(scores[j]) - z_max;
      if (j != best) tail.add(std::exp(d));
    }
    const double lse = std::log1p(tail.value());
    for (std::size_t j = 0; j < m; ++j) {
      const double qj = static_cast<double>(q[j]);
      if (qj > 0) cross.add(qj * (lse - (static_cast<double>(scores[j]) - z_max)));
    }
    return static_cast<Result>(cross.value());
  }

  template <typename Int, typename Q, typename T>
  inline void feasible_dual(const Int num_classes, const Q* q, T* variables) const {
    const std::size_t m = static_cast<std::size_t>(num_classes);
    const T cc = static_cast<T>(c);
    T sum(0);
    for (std::size_t j = 0; j < m; ++j) {
      const T hi = cc * static_cast<T>(q[j]);
      const T lo = hi - cc;
      variables[j] = std::max(lo, std::min(hi, variables[j]));
      sum += variables[j];
    }
    if (!(sum > 0 || sum < 0)) return;
    // The residual goes first to the class of largest probability q_j - a_j / c, as in
    // update_dual_variables: at a class of probability near 0, where -pi log pi has an
    // infinite slope, it would change the dual by far more than its size.
    std::size_t pivot = 0;
    for (std::size_t j = 1; j < m; ++j)
      if (static_cast<T>(q[j]) - variables[j] / cc >
          static_cast<T>(q[pivot]) - variables[pivot] / cc)
        pivot = j;
    const bool excess = sum > 0;
    T rest = std::abs(sum);
    auto absorb = [&](std::size_t j) {
      const T hi = cc * static_cast<T>(q[j]);
      const T room = std::min(excess ? variables[j] - (hi - cc) : hi - variables[j], rest);
      variables[j] += excess ? -room : room;
      rest -= room;
    };
    absorb(pivot);
    for (std::size_t j = 0; j < m && rest > 0; ++j)
      if (j != pivot) absorb(j);
  }

  template <typename Int, typename Q>
  inline Result dual_loss(const Int num_classes, const Q* q, const Data* variables) const {
    using Work = typename std::common_type<Result, Data, double>::type;
    const std::size_t m = static_cast<std::size_t>(num_classes);
    const Work C = Work(c), eps = 64 * Work(std::numeric_limits<Data>::epsilon());
    Work balance(0), size(0);
    for (std::size_t j = 0; j < m; ++j) {
      const Work aj = Work(variables[j]), qj = Work(q[j]);
      if (!std::isfinite(aj) || aj > C * qj + eps * C || aj < C * (qj - Work(1)) - eps * C)
        return -std::numeric_limits<Result>::infinity();
      balance += aj;
      size += std::abs(aj);
    }
    if (std::abs(balance) > eps * std::max(size, C))
      return -std::numeric_limits<Result>::infinity();
    detail::compensated_sum<Work> loss;
    for (std::size_t j = 0; j < m; ++j) {
      const Work qj = Work(q[j]), r = Work(variables[j]) / C;
      const Work prob = qj - r;
      if (prob > 0) {
        const Work log_prob =
            qj > 0 && std::abs(r) <= qj / 2 ? std::log(qj) + std::log1p(-r / qj) : std::log(prob);
        loss.add(-C * prob * log_prob);
      }
    }
    return Result(loss.value());
  }
};

}  // namespace sdca

#endif
