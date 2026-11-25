#ifndef SDCA_SOLVER_OBJECTIVE_ENTROPY_H
#define SDCA_SOLVER_OBJECTIVE_ENTROPY_H

#include <string>

#include "sdca/solver/objective/objective_base.h"
#include "sdca/solver/objective/entropy_block.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN

namespace solver_detail {

// Warm log-odds (z - zr) - norm2 (a - ar) in T. Exact differences and product in W >= double
// keep a small residual between huge cancelling score and dual terms.
template <typename T, typename Data>
inline T warm_log_odds(const Data z, const Data zr, const Data a, const Data ar, const Data norm2) {
  const T dz = T(z) - T(zr), product = T(norm2) * (T(a) - T(ar)), plain = dz - product;
  // Without cancellation the plain rounding error is below 1.5 ulps.
  if (std::abs(dz) + 2 * std::abs(product) <= 2 * std::abs(plain)) return plain;
  typedef typename std::common_type<T, double>::type W;
  W dz_lo, da_lo, product_lo, sum_lo;
  const W wdz = core::sum_error(W(z), -W(zr), dz_lo), da = core::sum_error(W(a), -W(ar), da_lo);
  const W wproduct = core::split_product(W(norm2), da, product_lo);
  const W sum = core::sum_error(wdz, -wproduct, sum_lo);
  const W exact = sum + (sum_lo + (dz_lo - (product_lo + W(norm2) * da_lo)));
  // The plain value within two ulps stands (and where the split overflows).
  const W tol = 2 * W(std::numeric_limits<T>::epsilon()) * std::abs(W(plain));
  return std::abs(exact - W(plain)) > tol && std::isfinite(exact) ? T(exact) : plain;
}

// warm_log_odds as high + low (both T), exact to the rounding of low.
template <typename T, typename Data>
inline T warm_log_odds_split(const Data z, const Data zr, const Data a, const Data ar,
                             const Data norm2, T& low) {
  typedef typename std::common_type<T, double>::type W;
  W dz_lo, da_lo, product_lo, sum_lo;
  const W wdz = core::sum_error(W(z), -W(zr), dz_lo), da = core::sum_error(W(a), -W(ar), da_lo);
  const W wproduct = core::split_product(W(norm2), da, product_lo);
  const W sum = core::sum_error(wdz, -wproduct, sum_lo);
  const W rest = sum_lo + (dz_lo - (product_lo + W(norm2) * da_lo));
  const T high = T(sum + rest);
  low = T((sum - W(high)) + rest);
  if (std::isfinite(high) && std::isfinite(low)) return high;
  low = T(0);
  return warm_log_odds<T>(z, zr, a, ar, norm2);
}

// One label at index 0, uniform on the first p entries, and given probabilities.
struct first_label {
  double operator()(const std::size_t j) const { return j == 0 ? 1.0 : 0.0; }
};
struct first_labels {
  std::size_t p;
  double share;
  explicit first_labels(const std::size_t p_) : p(p_), share(1.0 / double(p_)) {}
  double operator()(const std::size_t j) const { return j < p ? share : 0.0; }
};
template <typename Q>
struct given_target {
  const Q* q;
  double operator()(const std::size_t j) const { return double(q[j]); }
};
template <typename Q>
inline given_target<Q> target_of(const Q* q) {
  given_target<Q> t = {q};
  return t;
}

// The labels of a target uniform on the first p entries, 0 for given probabilities.
inline std::size_t uniform_labels(const first_label&) { return 1; }
inline std::size_t uniform_labels(const first_labels& t) { return t.p; }
template <typename Q>
inline std::size_t uniform_labels(const given_target<Q>&) {
  return 0;
}

// The entropy objectives: the cross-entropy to a target distribution q of a block (one
// label, uniform on a label set, or given probabilities); target(j) gives q_j. The dual
// block is a = c (q - pi) for probabilities pi; a step is the softmax proximal step at the
// warm log-odds, from the current block. The steps solve in common_type<Data, Result>:
// <float, float> in float, <float, double> in double.
template <typename Data, typename Result>
struct entropy_objective : objective_base<Data, Result> {
  typedef objective_base<Data, Result> base;
  typedef typename std::common_type<Data, Result>::type step_type;

  const Result c;

  entropy_objective(const kernel::Kind kind, const Result c_, const size_type k = 0)
      : base(spec_of(kind, c_, k)), c(c_) {}
  static objective_spec spec_of(const kernel::Kind kind, const Result c_, const size_type k) {
    const objective_spec s = {kind, double(c_), k, 0.0};
    return s;
  }

  std::string describe(const char* name) const {
    std::ostringstream str;
    str << name << " (c: " << c << ", precision: " << type_name<Result>() << ")";
    return str.str();
  }

  template <typename Target>
  void step(const std::size_t n, const Data norm2, Data* a, const Data* z, Target target) const {
    typedef step_type T;
    const T cc = T(c), alpha = T(c) * T(norm2);
    if (!(cc > 0) || !std::isfinite(cc) || n < 2)
      throw std::invalid_argument("Invalid entropy dimensions or C");
    core::row_buffer<T> v(n), prior(n);
    // Centered before forming the score differences: [-1e16, 0, 1] would otherwise become
    // [0, 1e16, 1e16], losing a real margin.
    const std::size_t ref = std::size_t(std::max_element(z, z + n) - z);
    const bool zero = warm_point(n, ref, norm2, cc, z, a, v.data(), prior.data());
    // Labels uniform on p < n classes take the label-block step (entropy_block.h), and the
    // softmax proximal step where that does not converge.
    const std::size_t p = uniform_labels(target);
    if (p > 0 && p < n && std::all_of(v.data(), v.data() + n, [](T x) { return x - x == 0; })) {
      try {
        label_block::entropy_update(n, p, cc, alpha, norm2, a, v.data());
        return;
      } catch (const std::runtime_error&) {
      }
    }
    static thread_local core::softmax_prox<T> block(1, T(1));
    block.reset(n, alpha);
    block.set_scores(v.data(), target);
    block.solve(zero ? nullptr : prior.data());
    block.duals(cc, a);
  }

  // v = the warm log-odds about entry ref and prior = a / c; whether a is zero. The lanes
  // whose plain log-odds may cancel take warm_log_odds.
  static bool warm_point(const std::size_t n, const std::size_t ref, const Data norm2,
                         const step_type cc, const Data* z, const Data* a, step_type* v,
                         step_type* prior) {
    typedef step_type T;
    typedef point_lanes<T> A;
    const T zr = T(z[ref]), ar = T(a[ref]), scale = T(norm2);
    typename A::mask_type nonzero = A::mask_type::zero();
    std::size_t j = 0;
    for (; j + A::L <= n; j += A::L) {
      const A aj = core::load<A>(a + j), dz = core::load<A>(z + j) - zr,
              product = (aj - ar) * scale, plain = dz - product;
      const typename A::mask_type cancels =
          ~(core::abs(dz) + core::abs(product) * T(2) <= core::abs(plain) * T(2));
      A x = plain;
      if (core::any(cancels))
        for (int l = 0; l < A::L; ++l)
          if (cancels[l]) x.set(l, warm_log_odds<T>(z[j + l], z[ref], a[j + l], a[ref], norm2));
      x.store(v + j);
      (aj / cc).store(prior + j);
      nonzero = nonzero | (aj != T(0));
    }
    bool zero = !core::any(nonzero);
    for (; j < n; ++j) {
      v[j] = warm_log_odds<T>(z[j], z[ref], a[j], a[ref], norm2);
      prior[j] = T(a[j]) / cc;
      zero = zero && a[j] == 0;
    }
    return zero;
  }

  // c (q - 1/m), the gradient at zero scores; the entry of the largest target balances.
  template <typename Target>
  void zero_scores(const std::size_t n, Data* a, Target target) const {
    typedef typename std::common_type<Data, Result, double>::type W;
    const W cc = W(c), uniform = W(1) / W(n);
    std::size_t pivot = 0;
    for (std::size_t j = 1; j < n; ++j)
      if (W(target(j)) > W(target(pivot))) pivot = j;
    core::compensated_sum<W> sum;
    for (std::size_t j = 0; j < n; ++j) {
      a[j] = static_cast<Data>(cc * (W(target(j)) - uniform));
      if (!std::isfinite(a[j]))
        throw std::runtime_error("The zero-score dual block overflows the data type");
      if (j != pivot) sum.add(W(a[j]));
    }
    a[pivot] = static_cast<Data>(-sum.value());
  }

  // The bounds c (q_j - 1) <= a_j <= c q_j, and the zero sum restored first at the entry of
  // largest probability q_j - a_j / c: at a probability near 0, where -pi log pi has an
  // infinite slope, it would change the dual by far more than its size.
  template <typename T, typename Target>
  void repair(const std::size_t n, T* a, Target target) const {
    const T cc = T(c);
    T sum(0);
    for (std::size_t j = 0; j < n; ++j) {
      const T hi = cc * T(target(j)), lo = hi - cc;
      a[j] = std::max(lo, std::min(hi, a[j]));
      sum += a[j];
    }
    if (!(sum > 0 || sum < 0)) return;
    std::size_t pivot = 0;
    for (std::size_t j = 1; j < n; ++j)
      if (T(target(j)) - a[j] / cc > T(target(pivot)) - a[pivot] / cc) pivot = j;
    const bool excess = sum > 0;
    T rest = std::abs(sum);
    auto absorb = [&](const std::size_t j) {
      const T hi = cc * T(target(j));
      const T room = std::min(excess ? a[j] - (hi - cc) : hi - a[j], rest);
      a[j] += excess ? -room : room;
      rest -= room;
    };
    absorb(pivot);
    for (std::size_t j = 0; j < n && rest > 0; ++j)
      if (j != pivot) absorb(j);
  }

  // c H(q - a / c), or -inf where a is not feasible: outside the bounds by more than the
  // rounding of n entries in Data against a mass of c (gamma_n(Data) c), or off the zero sum
  // by more than that relative to its size.
  template <typename Target>
  Result dual(const std::size_t n, const Data* a, Target target) const {
    typedef typename std::common_type<Result, Data, double>::type W;
    const W C = W(c), eps = W(rounding_bound<Data>(n));
    W balance(0), size(0);
    for (std::size_t j = 0; j < n; ++j) {
      const W aj = W(a[j]), qj = W(target(j));
      if (!std::isfinite(aj) || aj > C * qj + eps * C || aj < C * (qj - 1) - eps * C)
        return -std::numeric_limits<Result>::infinity();
      balance += aj;
      size += std::abs(aj);
    }
    if (std::abs(balance) > eps * std::max(size, C))
      return -std::numeric_limits<Result>::infinity();
    core::compensated_sum<W> loss;
    for (std::size_t j = 0; j < n; ++j) {
      const W qj = W(target(j)), r = W(a[j]) / C, prob = qj - r;
      if (qj == 0) {
        // -C pi log pi with log c apart from log b, b = -a_j.
        const W b = -W(a[j]);
        if (b > 0) loss.add(-b * (std::log(b) - std::log(C)));
      } else if (prob > 0) {
        const W log_prob =
            std::abs(r) <= qj / 2 ? std::log(qj) + std::log1p(-r / qj) : std::log(prob);
        loss.add(-C * prob * log_prob);
      }
    }
    return Result(loss.value());
  }
};

}  // namespace solver_detail

// Cross-entropy of the label at index 0.
template <typename Data, typename Result>
struct l2_cross_entropy : solver_detail::entropy_objective<Data, Result> {
  typedef Data data_type;
  typedef Result result_type;
  typedef solver_detail::entropy_objective<Data, Result> base;

  explicit l2_cross_entropy(const Result c_) : base(kernel::cross_entropy, c_) {}
  template <typename D, typename R>
  explicit l2_cross_entropy(const l2_cross_entropy<D, R>& other)
      : base(kernel::cross_entropy, Result(other.c)) {}
  std::string to_string() const { return this->describe("l2_cross_entropy"); }

  template <typename Int>
  void update_dual_variables(const Int num_classes, const Data norm2, Data* variables,
                             Data* scores) const {
    this->step(std::size_t(num_classes), norm2, variables, scores, solver_detail::first_label());
  }
  template <typename Int>
  void zero_scores_dual_variables(const Int num_classes, Data* variables) const {
    this->zero_scores(std::size_t(num_classes), variables, solver_detail::first_label());
  }
  template <typename Int, typename T>
  void feasible_dual(const Int num_classes, T* variables) const {
    feasible::capped(num_classes, variables, T(this->c), T(this->c));
  }
  template <typename Int>
  Result dual_loss(const Int num_classes, const Data* variables) const {
    return this->dual(std::size_t(num_classes), variables, solver_detail::first_label());
  }
};

// Cross-entropy to the target uniform on the labels [0, num_labels).
template <typename Data, typename Result>
struct l2_multilabel_cross_entropy : solver_detail::entropy_objective<Data, Result> {
  typedef Data data_type;
  typedef Result result_type;
  typedef solver_detail::entropy_objective<Data, Result> base;

  explicit l2_multilabel_cross_entropy(const Result c_)
      : base(kernel::multilabel_cross_entropy, c_) {}
  template <typename D, typename R>
  explicit l2_multilabel_cross_entropy(const l2_multilabel_cross_entropy<D, R>& other)
      : base(kernel::multilabel_cross_entropy, Result(other.c)) {}
  std::string to_string() const { return this->describe("l2_multilabel_cross_entropy"); }

  template <typename Int>
  void update_dual_variables(const Int num_classes, const Int num_labels, const Data norm2,
                             Data* variables, Data* scores) const {
    this->step(std::size_t(num_classes), norm2, variables, scores,
               solver_detail::first_labels(std::size_t(num_labels)));
  }
  template <typename Int>
  void zero_scores_dual_variables(const Int num_classes, const Int num_labels,
                                  Data* variables) const {
    this->zero_scores(std::size_t(num_classes), variables,
                      solver_detail::first_labels(std::size_t(num_labels)));
  }
  template <typename Int, typename T>
  void feasible_dual(const Int num_classes, const Int num_labels, T* variables) const {
    feasible::label_distribution(num_classes, num_labels, variables, T(this->c));
  }
  template <typename Int>
  Result dual_loss(const Int num_classes, const Int num_labels, const Data* variables) const {
    return this->dual(std::size_t(num_classes), variables,
                      solver_detail::first_labels(std::size_t(num_labels)));
  }
};

// Cross-entropy to given probabilities q (rows summing to one, make_output_distribution).
template <typename Data, typename Result>
struct l2_distillation : solver_detail::entropy_objective<Data, Result> {
  typedef Data data_type;
  typedef Result result_type;
  typedef solver_detail::entropy_objective<Data, Result> base;

  explicit l2_distillation(const Result c_) : base(kernel::cross_entropy, c_) {}
  template <typename D, typename R>
  explicit l2_distillation(const l2_distillation<D, R>& other)
      : base(kernel::cross_entropy, Result(other.c)) {}
  std::string to_string() const { return this->describe("l2_distillation"); }

  template <typename Int, typename Q>
  void update_dual_variables(const Int num_classes, const Q* q, const Data norm2, Data* variables,
                             Data* scores) const {
    this->step(std::size_t(num_classes), norm2, variables, scores, solver_detail::target_of(q));
  }
  template <typename Int, typename Q>
  void zero_scores_dual_variables(const Int num_classes, const Q* q, Data* variables) const {
    this->zero_scores(std::size_t(num_classes), variables, solver_detail::target_of(q));
  }
  template <typename Int, typename Q, typename T>
  void feasible_dual(const Int num_classes, const Q* q, T* variables) const {
    this->repair(std::size_t(num_classes), variables, solver_detail::target_of(q));
  }
  template <typename Int, typename Q>
  Result dual_loss(const Int num_classes, const Q* q, const Data* variables) const {
    return this->dual(std::size_t(num_classes), variables, solver_detail::target_of(q));
  }
};

// Top-k cross-entropy of the label at index 0: the softmax step at k = 1, else the top-k
// entropy block of the competitors (all capped at k = m - 1).
template <typename Data, typename Result>
struct l2_topk_cross_entropy : solver_detail::entropy_objective<Data, Result> {
  typedef Data data_type;
  typedef Result result_type;
  typedef solver_detail::entropy_objective<Data, Result> base;

  const size_type k;

  l2_topk_cross_entropy(const Result c_, const size_type k_)
      : base(kernel::topk_cross_entropy, c_, k_), k(k_) {}
  template <typename D, typename R>
  explicit l2_topk_cross_entropy(const l2_topk_cross_entropy<D, R>& other)
      : l2_topk_cross_entropy(Result(other.c), other.k) {}
  std::string to_string() const {
    std::ostringstream str;
    str << "l2_topk_cross_entropy (c: " << this->c << ", k: " << k
        << ", precision: " << type_name<Result>() << ")";
    return str.str();
  }

  template <typename Int>
  void update_dual_variables(const Int num_classes, const Data norm2, Data* variables,
                             Data* scores) const {
    this->check_classes(num_classes);
    if (k == 1) {
      this->step(std::size_t(num_classes), norm2, variables, scores, solver_detail::first_label());
      return;
    }
    typedef typename base::step_type Work;
    typedef typename std::common_type<Work, double>::type Sum;
    const std::size_t m = std::size_t(num_classes - 1);
    if (k == m) {
      all_capped(m, norm2, variables, scores);
      return;
    }
    core::row_buffer<Work> v(m), probabilities(m);
    Work offset, offset_low(0);
    bool far = false;
    const Int r = Int(std::max_element(scores + 1, scores + num_classes) - scores);
    for (Int j = 1; j < num_classes; ++j) {
      const Work x = solver_detail::warm_log_odds<Work>(scores[j], scores[r], variables[j],
                                                        variables[r], norm2);
      v.data()[std::size_t(j - 1)] = x;
      far |= std::abs(x) > Work(64);
    }
    offset =
        solver_detail::warm_log_odds<Work>(scores[r], scores[0], variables[r], variables[0], norm2);
    // Scores relative to the largest keep their distance from it to 53 bits; beyond 64 from
    // it their exact low parts keep the free scores' differences in one solve.
    core::row_buffer<Work> low(far ? m : 0);
    if (far) {
      for (Int j = 1; j < num_classes; ++j) {
        const std::size_t i = std::size_t(j - 1);
        low.data()[i] = Work(0);
        if (std::abs(v.data()[i]) > Work(64))
          v.data()[i] = solver_detail::warm_log_odds_split<Work>(
              scores[j], scores[r], variables[j], variables[r], norm2, low.data()[i]);
      }
    }
    if (std::abs(offset) > Work(64))
      offset = solver_detail::warm_log_odds_split<Work>(scores[r], scores[0], variables[r],
                                                        variables[0], norm2, offset_low);
    core::topk_entropy_block(m, v.data(), std::size_t(k), Work(this->c) * Work(norm2),
                             probabilities.data(), static_cast<Work*>(nullptr),
                             static_cast<Work*>(nullptr), offset, Work(this->c),
                             static_cast<core::topk_entropy_free<Work>*>(nullptr),
                             far ? low.data() : static_cast<Work*>(nullptr), offset_low);
    Sum sum(0);
    for (Int j = 1; j < num_classes; ++j) {
      variables[j] = Data(-probabilities.data()[std::size_t(j - 1)]);
      sum -= Sum(variables[j]);
    }
    variables[0] = Data(sum);
  }

  template <typename Int>
  void zero_scores_dual_variables(const Int num_classes, Data* variables) const {
    // At zero scores the top-k caps are inactive: the softmax block.
    this->zero_scores(std::size_t(num_classes), variables, solver_detail::first_label());
  }
  template <typename Int, typename T>
  void feasible_dual(const Int num_classes, T* variables) const {
    feasible::topk_simplex(num_classes, variables, k, T(this->c));
  }
  template <typename Int>
  Result dual_loss(const Int num_classes, const Data* variables) const {
    return this->dual(std::size_t(num_classes), variables, solver_detail::first_label());
  }

 private:
  // Every competitor capped: the block depends on the mean of their warm log-odds alone,
  // averaged before shifting every score by the largest competitor (otherwise
  // [1e16, 1, -1e16] loses its middle score).
  void all_capped(const std::size_t m, const Data norm2, Data* variables,
                  const Data* scores) const {
    typedef typename base::step_type Work;
    typedef typename std::common_type<Work, double>::type Sum;
    typedef long double Long;
    Long score_low, dual_low(0), dual_mean;
    const Long score_sum = core::topk_entropy::entropy_sum_offset(
        m, [&](std::size_t j) { return scores[j + 1]; }, -Long(scores[0]), score_low);
    // Cold starts and every all-capped update have equal competitor duals: that invariant
    // spares the compensated pass; other warm starts take the full mean.
    bool uniform_dual = std::isfinite(variables[1]);
    for (std::size_t j = 1; uniform_dual && j < m; ++j)
      uniform_dual = variables[j + 1] == variables[1];
    Long dual_sum;
    if (uniform_dual) {
      Long gap_lo, sum_lo;
      dual_mean = core::sum_error(Long(variables[1]), -Long(variables[0]), gap_lo);
      dual_sum = core::split_product(Long(m), dual_mean, sum_lo);
      dual_low = sum_lo + Long(m) * gap_lo;
    } else {
      dual_sum = core::topk_entropy::entropy_sum_offset(
          m, [&](std::size_t j) { return variables[j + 1]; }, -Long(variables[0]), dual_low);
      dual_mean = (dual_sum + dual_low) / Long(m);
    }
    // Scores and duals averaged apart: score - norm2 dual can lose the whole dual part
    // beside cancelling huge scores.
    const Work plain = Work((score_sum + score_low) / Long(m) - Long(norm2) * dual_mean);
    // Exact sums less an exact product keep a small residual between huge cancelling score
    // and dual sums; the plain mean within two ulps stands.
    Long product_lo, difference_lo;
    const Long product = core::split_product(Long(norm2), dual_sum, product_lo);
    const Long difference = core::sum_error(score_sum, -product, difference_lo);
    const Long exact =
        (difference + (difference_lo + (score_low - (product_lo + Long(norm2) * dual_low)))) /
        Long(m);
    const Long tol = 2 * Long(std::numeric_limits<Work>::epsilon()) * std::abs(Long(plain));
    const Work offset =
        std::abs(exact - Long(plain)) > tol && std::isfinite(exact) ? Work(exact) : plain;
    Work cap(0), scaled_cap(0), logs(0);
    core::topk_entropy::topk_entropy_all_capped(m, offset, Work(this->c) * Work(norm2),
                                                Work(this->c), cap, scaled_cap, logs);
    Sum sum(0);
    for (std::size_t j = 1; j <= m; ++j) {
      variables[j] = Data(-scaled_cap);
      sum -= Sum(variables[j]);
    }
    variables[0] = Data(sum);
  }
};

SDCA_ISA_END

#endif
