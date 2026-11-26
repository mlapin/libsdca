#ifndef SDCA_SOLVER_OBJECTIVE_OBJECTIVE_BASE_H
#define SDCA_SOLVER_OBJECTIVE_OBJECTIVE_BASE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "sdca/kernel.h"
#include "sdca/core/limits.h"
#include "sdca/core/project.h"
#include "sdca/core/storage.h"
#include "sdca/math/blas.h"
#include "sdca/solver/objective/feasible_dual.h"
#include "sdca/utility/types.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN

// gamma_k = k u / (1 - k u), u the unit roundoff of T: the relative error of k roundings in
// T (a sum of k + 1 terms, a product of k factors), the unit of the solver's tolerances.
template <typename T>
inline double rounding_bound(const std::size_t k) {
  const double ku = static_cast<double>(k) * static_cast<double>(limits::roundoff<T>());
  return ku / (1 - ku);
}

namespace solver_detail {

// The working precision of the hinge steps: the wider of storage and result.
template <typename Data, typename Result>
using hinge_work = typename std::common_type<Data, Result>::type;

// A hinge margin below this fraction of its terms cancels and is refined.
template <typename T>
inline T margin_cancellation() {
  return T(1) / T(16);
}

// The lanes of the step points: their plain entries in lanes, the rare entries that cancel
// one by one.
template <typename T>
using point_lanes = typename core::Native<T>::type;

// beta v + a (z - z0): a scal by beta followed by an axpy, element by element.
template <typename T>
inline T hinge_scaled(const T a, const T beta, const T v, const T z, const T z0) {
  T y = beta * v;
  y += a * (z - z0);
  return y;
}

// A sum with its rounding errors, products included (TwoProduct).
template <typename T>
struct exact_sum : core::compensated_sum<T> {
  void add_product(const T x, const T y) {
    T e;
    this->add(core::two_product(x, y, e));
    this->lo += e;
  }
};

// beta v + a (z - c) with the rounding errors kept: the low part of z - c is the margin a
// large common offset of z and c would absorb.
template <typename T>
inline exact_sum<T> scaled_difference(const T a, const T beta, const T v, const T z, const T c) {
  exact_sum<T> s;
  T low;
  const T d = core::sum_error(z, -c, low);
  s.add_product(beta, v);
  s.add_product(a, d);
  s.lo += a * low;
  return s;
}

// The top-k hinge point beta (v - v0) + a (z - z0) + a with every rounding error kept.
template <typename T>
inline T topk_hinge_compensated(const T a, const T beta, const T v, const T z, const T z0,
                                const T v0) {
  exact_sum<T> s = scaled_difference(a, beta, v, z, z0);
  s.add(a);
  s.add_product(-beta, v0);
  return s.value();
}

// Whether a point entry is below 1/16 of its warm terms: the scaled old dual cancels
// a (z - z0) and may hold the residual in its rounding (warm cancellation).
template <typename T>
inline bool warm_cancels(const T entry, const T dual_term, const T offset) {
  return std::abs(entry) < margin_cancellation<T>() * (std::abs(dual_term) + offset);
}

// The precision of a warm recomputation: TwoSum/TwoProduct pairs in at least double.
template <typename Work>
using warm_work = typename std::common_type<Work, double>::type;

// A recomputed warm entry, or the plain one where they agree to two ulps.
template <typename Work, typename W>
inline Work warm_entry(const W exact, const Work plain) {
  const W tol = 2 * W(std::numeric_limits<Work>::epsilon()) * std::abs(W(plain));
  return std::isfinite(exact) && std::abs(exact - W(plain)) > tol ? static_cast<Work>(exact)
                                                                  : plain;
}

// A warm entry and the terms it cancelled (zero where the plain entry stands).
template <typename Work>
struct warm_result {
  Work value, terms;
};

// A top-k entry y recomputed where the scaled old dual cancels a (z - z0).
template <typename Work>
inline warm_result<Work> topk_hinge_warm(const Work a, const Work beta, const Work vj,
                                         const Work zj, const Work z0, const Work v0,
                                         const Work offset, const Work y) {
  const Work terms = std::abs(beta * vj) + offset;
  if (!warm_cancels(y, Work(0), terms)) return {y, Work(0)};
  typedef warm_work<Work> W;
  const Work warm =
      warm_entry(topk_hinge_compensated(W(a), W(beta), W(vj), W(zj), W(z0), W(v0)), y);
  return {warm, warm != y ? terms : Work(0)};
}

// Plain entries [j, n): hinge_scaled + shift, or shift - hinge_scaled (Minus).
template <bool Minus, typename Work, typename Data, typename Int>
inline void hinge_plain(Int j, const Int n, const Work a, const Work beta, const Work z0,
                        const Work shift, const Data* z, Data* v) {
  typedef point_lanes<Work> A;
  for (; j + A::L <= n; j += A::L) {
    const A u = core::load<A>(v + j) * beta + (core::load<A>(z + j) - z0) * a;
    core::store(v + j, Minus ? shift - u : u + shift);
  }
  for (; j < n; ++j) {
    const Work u = hinge_scaled(a, beta, Work(v[j]), Work(z[j]), z0);
    v[j] = static_cast<Data>(Minus ? shift - u : u + shift);
  }
}

// Plain top-k entries [j, n).
template <typename Work, typename Data, typename Int>
inline void topk_hinge_plain(Int j, const Int n, const Work a, const Work beta, const Work z0,
                             const Work shift, const Data* z, Data* v) {
  hinge_plain<false>(j, n, a, beta, z0, shift, z, v);
}

// Plain top-k entries of a warm block from j, up to the first entry that reaches bound or
// cancels (warm_cancels): its index, or n.
template <typename Work, typename Data, typename Int>
inline Int topk_hinge_lanes(Int j, const Int n, const Work a, const Work beta, const Work z0,
                            const Work shift, const Work offset, const Work bound, const Data* z,
                            Data* v) {
  typedef point_lanes<Work> A;
  const Work cancel = margin_cancellation<Work>();
  for (; j + A::L <= n; j += A::L) {
    const A dual = core::load<A>(v + j) * beta;
    const A y = (dual + (core::load<A>(z + j) - z0) * a) + shift;
    if (core::any((y >= bound) | (core::abs(y) < (core::abs(dual) + offset) * cancel))) break;
    core::store(v + j, y);
  }
  for (; j < n; ++j) {
    const Work vj = Work(v[j]), y = hinge_scaled(a, beta, vj, Work(z[j]), z0) + shift;
    if (y >= bound || warm_cancels(y, beta * vj, offset)) return j;
    v[j] = static_cast<Data>(y);
  }
  return n;
}

// Top-k entries of a warm block from its first cancelling one: plain into v, recomputed into
// z (scratch), until an entry reaches bound (no warm cancellation; returns the next index).
// Otherwise the recomputed entries replace the plain ones if the largest entry is below the
// cancelled terms too (warm cancellation); returns n.
template <typename Work, typename Data, typename Int>
inline Int topk_hinge_warm_tail(const Int first, const Int n, const Work a, const Work beta,
                                const Work z0, const Work v0, const Work shift, const Work bound,
                                Data* z, Data* v) {
  const Work offset = std::abs(shift);
  Work terms(0);
  for (Int j = first; j < n; ++j) {
    const Work zj = Work(z[j]), vj = Work(v[j]);
    const Work y = hinge_scaled(a, beta, vj, zj, z0) + shift;
    v[j] = static_cast<Data>(y);
    if (y >= bound) return j + 1;
    const warm_result<Work> w = topk_hinge_warm(a, beta, vj, zj, z0, v0, offset, y);
    terms = std::max(terms, w.terms);
    z[j] = static_cast<Data>(w.value);
  }
  if (!(terms > 0)) return n;
  Work largest = static_cast<Work>(*std::max_element(z + first, z + n));
  if (first > 1)
    largest = std::max(largest, static_cast<Work>(*std::max_element(v + 1, v + first)));
  if (warm_cancels(largest, Work(0), terms)) std::copy(z + first, z + n, v + first);
  return n;
}

// The vector a top-k hinge step projects, in one pass and in Work: u = beta v + a (z - z0)
// relative to the label's score z0, then v_0 = u_0 and v_j = u_j + (a - u_0). From an entry
// where the scaled old dual cancels on, see topk_hinge_warm_tail.
template <typename Work, typename Data, typename Int>
inline void topk_hinge_point(const Int n, const Work a, const Work beta, Data* z, Data* v) {
  const Work z0 = Work(z[0]), v0 = Work(v[0]), u0 = hinge_scaled(a, beta, v0, z0, z0),
             shift = a - u0;
  v[0] = static_cast<Data>(u0);
  // Warm cancellation needs |y| below (|beta v_j| + |shift|) / 16 <= 2 |shift| / 16 in a
  // feasible block, at every entry; a cold block (v0 = 0, so v = 0) has no warm terms.
  const Work offset = std::abs(shift), bound = 2 * margin_cancellation<Work>() * offset;
  Int j = 1;
  if (v0 != 0) j = topk_hinge_lanes(j, n, a, beta, z0, shift, offset, bound, z, v);
  if (v0 != 0 && j < n) {
    const Work y = hinge_scaled(a, beta, Work(v[j]), Work(z[j]), z0) + shift;
    if (y >= bound)
      v[j++] = static_cast<Data>(y);
    else
      j = topk_hinge_warm_tail(j, n, a, beta, z0, v0, shift, bound, z, v);
  }
  topk_hinge_plain(j, n, a, beta, z0, shift, z, v);
}

// A multilabel entry about label 0, -w at a label and w + a elsewhere with w = u - u0,
// recomputed where the scaled old dual cancels a (z - z0). Every plain entry carries u0, so
// the terms count even if w is exact.
template <bool label, typename Work>
inline warm_result<Work> multilabel_warm(const Work a, const Work beta, const Work vj,
                                         const Work zj, const Work z0, const Work v0, const Work u,
                                         const Work u0, const Work offset) {
  const Work plain = label ? -(u - u0) : (u - u0) + a, terms = std::abs(beta * vj) + offset;
  if (!warm_cancels(plain, Work(0), terms)) return {plain, Work(0)};
  typedef warm_work<Work> W;
  exact_sum<W> s = scaled_difference(W(a), W(beta), W(vj), W(zj), W(z0));
  s.add_product(-W(beta), W(v0));
  if (!label) s.add(W(a));
  return {warm_entry(label ? -s.value() : s.value(), plain), terms};
}

// Plain multilabel entries a / 2 - u (labels) or u + a / 2 on [j, end), up to the first
// whose entry about label 0 cancels: its index, or end. A non-label entry about label 0 at or
// above bound rules warm cancellation out: the rest is plain and the result end + 1.
template <bool label, typename Work, typename Data, typename Int>
inline Int multilabel_plain(Int j, const Int end, const Work a, const Work beta, const Work z0,
                            const Work u0, const Work offset, const Work bound, const Data* z,
                            Data* v) {
  typedef point_lanes<Work> A;
  const Work half = a / 2, cancel = margin_cancellation<Work>();
  for (; j + A::L <= end; j += A::L) {
    const A dual = core::load<A>(v + j) * beta;
    const A u = dual + (core::load<A>(z + j) - z0) * a;
    const A w = label ? -(u - u0) : (u - u0) + a;
    const typename A::mask_type cancels = core::abs(w) < (core::abs(dual) + offset) * cancel;
    if (core::any(label ? cancels : cancels | (w >= bound))) break;
    core::store(v + j, label ? half - u : u + half);
  }
  for (; j < end; ++j) {
    const Work vj = Work(v[j]), u = hinge_scaled(a, beta, vj, Work(z[j]), z0);
    const Work w = label ? -(u - u0) : (u - u0) + a;
    if (!label && w >= bound) {
      hinge_plain<false>(j, end, a, beta, z0, half, z, v);
      return end + 1;
    }
    if (warm_cancels(w, beta * vj, offset)) return j;
    v[j] = static_cast<Data>(label ? half - u : u + half);
  }
  return end;
}

// Multilabel entries [j, end) plain into v and about label 0 into z (scratch).
template <bool label, typename Work, typename Data, typename Int>
inline void multilabel_both(Int j, const Int end, const Work a, const Work beta, const Work z0,
                            const Work v0, const Work u0, const Work offset, Data* z, Data* v,
                            Work& terms) {
  const Work half = a / 2;
  for (; j < end; ++j) {
    const Work vj = Work(v[j]), zj = Work(z[j]), u = hinge_scaled(a, beta, vj, zj, z0);
    v[j] = static_cast<Data>(label ? half - u : u + half);
    const warm_result<Work> w = multilabel_warm<label>(a, beta, vj, zj, z0, v0, u, u0, offset);
    z[j] = static_cast<Data>(w.value);
    terms = std::max(terms, w.terms);
  }
}

// Multilabel entries [first, n) of a warm block from its first cancelling one, as in
// multilabel_both; if the largest pairwise margin also cancels, the point about label 0
// replaces the plain one (earlier entries shifted).
template <typename Work, typename Data, typename Int>
inline void multilabel_warm_tail(const Int first, const Int n, const Int p, const Work a,
                                 const Work beta, const Work z0, const Work v0, const Work u0,
                                 Data* z, Data* v) {
  const Work offset = std::abs(u0), half = a / 2;
  Work terms(0);
  multilabel_both<true>(first, p, a, beta, z0, v0, u0, offset, z, v, terms);
  multilabel_both<false>(std::max(first, p), n, a, beta, z0, v0, u0, offset, z, v, terms);
  if (!(terms > 0)) return;
  z[0] = static_cast<Data>(0);
  for (Int j = 1; j < first; ++j)
    z[j] = static_cast<Data>(static_cast<Work>(v[j]) - (j < p ? half - u0 : u0 - half));
  const Work largest = static_cast<Work>(*std::max_element(z, z + p)) +
                       static_cast<Work>(*std::max_element(z + p, z + n));
  if (warm_cancels(largest, Work(0), terms)) std::copy(z, z + n, v);
}

// The plain point of a warm multilabel block with 0 < p < n (false if cold); see
// multilabel_warm_tail.
template <typename Work, typename Data, typename Int>
inline bool multilabel_warm_point(const Int n, const Int p, const Work a, const Work beta, Data* z,
                                  Data* v) {
  // A cold block (no label mass, so v = 0 if feasible) takes the plain loops.
  Work mass(0);
  for (Int j = 0; j < p; ++j) mass += std::abs(static_cast<Work>(v[j]));
  if (!(mass > 0)) return false;
  const Work z0 = Work(z[0]), v0 = Work(v[0]), u0 = hinge_scaled(a, beta, v0, z0, z0),
             offset = std::abs(u0);
  // Warm cancellation needs every pairwise margin below (|beta v_j| + |u0|) / 16, and |v_j|
  // is at most the label mass in a feasible block: an entry about label 0 past bound at a
  // non-label (whose margin with label 0 exceeds it) rules it out.
  const Work bound = 2 * margin_cancellation<Work>() * (std::abs(beta) * mass + offset);
  v[0] = static_cast<Data>(a / 2 - u0);
  Int first = multilabel_plain<true>(Int(1), p, a, beta, z0, u0, offset, bound, z, v);
  if (first == p) first = multilabel_plain<false>(p, n, a, beta, z0, u0, offset, bound, z, v);
  if (first < n) multilabel_warm_tail(first, n, p, a, beta, z0, v0, u0, z, v);
  return true;
}

// The vector a multilabel hinge step projects, in Work: a / 2 - u at the p labels and
// u + a / 2 elsewhere (see multilabel_warm_point for warm blocks).
template <typename Work, typename Data, typename Int>
inline void multilabel_hinge_point(const Int n, const Int p, const Work a, const Work beta, Data* z,
                                   Data* v) {
  const Work z0 = Work(z[0]);
  if (p > 0 && p < n && multilabel_warm_point(n, p, a, beta, z, v)) return;
  const Work half = a / 2;
  hinge_plain<true>(Int(0), p, a, beta, z0, half, z, v);
  hinge_plain<false>(p, n, a, beta, z0, half, z, v);
}

// The dual block at zero scores, p labels first: s / p on every label and -s / (n - p) on
// every other class, the first entry balancing the rest.
template <typename Data, typename Result, typename Int>
inline void zero_scores_dual(const Int n, const Int p, const Result s, Data* variables) {
  typedef typename std::common_type<Data, Result, double>::type Work;
  const Work label = Work(s) / Work(p), other = -Work(s) / Work(n - p);
  // The block stays unchanged if it does not fit the storage type.
  if (!std::isfinite(static_cast<Data>(label)) || !std::isfinite(static_cast<Data>(other)) ||
      !std::isfinite(static_cast<Data>(-Work(s))))
    throw std::runtime_error("The zero-score dual block overflows the data type");
  Work sum(0);
  for (Int j = 1; j < n; ++j) {
    variables[j] = static_cast<Data>(j < p ? label : other);
    sum += Work(variables[j]);
  }
  variables[0] = static_cast<Data>(-sum);
}

// Returns sum plus the entries of [first, last), and negates the entries.
template <typename Result, typename Data>
inline Result negate_and_sum(Data* first, Data* last, Result sum) {
  for (; first != last; ++first) {
    sum = sum + static_cast<Result>(*first);
    *first = -*first;
  }
  return sum;
}

// The labels [0, num_labels) as int64_t.
inline const int64_t* label_indices(const std::size_t num_labels) {
  static thread_local std::vector<int64_t> labels;
  for (std::size_t i = labels.size(); i < num_labels; ++i) labels.push_back(int64_t(i));
  return labels.data();
}

}  // namespace solver_detail

// The dual block's screening rules (see screening.h) and the solver's traits of an
// objective: the hinges shrink, and the single-label hinges take part steps.
struct screening_rules {
  bool examples = false, classes = false, pairs = false;
};

// A training objective 0.5 ||W - W0||^2 + c sum_i loss_i by its loss's shared name
// (docs/API.md): k for the top-k losses (0 for the others) and gamma >= 0 smoothing the
// hinges (0: nonsmooth, and for the other losses).
struct objective_spec {
  kernel::Kind kind;
  double c;
  size_type k;
  double gamma;
};

namespace solver_detail {
// spec, or std::invalid_argument unless it is a training objective of valid parameters.
inline objective_spec checked(const objective_spec& spec) {
  kernel::check_objective(spec.kind);
  if (!(spec.c > 0) || !std::isfinite(spec.c))
    throw std::invalid_argument("C must be finite and positive");
  kernel::check_parameters(spec.kind, 0, spec.k, spec.gamma, 1.0);
  return spec;
}
}  // namespace solver_detail

template <typename Data, typename Result>
struct objective_base {
  typedef Data data_type;
  typedef Result result_type;

  static const bool shrinks = false;
  static const bool part_steps = false;

  const objective_spec spec;
  const Result coeff_primal_loss;

  explicit objective_base(const objective_spec& spec_)
      : spec(solver_detail::checked(spec_)), coeff_primal_loss(static_cast<Result>(spec_.c)) {}

  // The loss of the rows (sdca/kernel.h).
  kernel::Loss loss() const {
    const kernel::Loss f = {spec.kind, spec.k, spec.gamma, 1.0, 0.0};
    return f;
  }

  // std::invalid_argument unless the objective suits num_classes classes (k < num_classes).
  // The constructor checked the other parameters, so a coordinate step checks k alone.
  template <typename Int>
  void check_classes(const Int num_classes) const {
    const std::size_t classes = static_cast<std::size_t>(num_classes);
    if (classes == 0 || spec.k < classes) return;
    kernel::check_parameters(spec.kind, classes, spec.k, spec.gamma, 1.0);
  }

  // The loss of m scores in class order at targets t (sdca/kernel.h), in double.
  template <typename Targets>
  double row_loss(const std::size_t m, const Data* scores, const Targets& t) const {
    check_classes(m);
    core::row_buffer<double> z(m);
    std::copy(scores, scores + m, z.data());
    const kernel::Execution e = {kernel::precision_float64, 1, false};
    double value = 0;
    if (kernel::evaluate(loss(), e, 1, m, z.data(), t, &value, static_cast<double*>(nullptr))
            .status != kernel::status_ok)
      throw std::runtime_error("The loss kernel does not take the objective's targets");
    return value;
  }

  // The loss of the label at scores[0].
  template <typename Int>
  Result primal_loss(const Int num_classes, const Data* scores) const {
    const int64_t label = 0;
    const kernel::Labels t = {&label, nullptr, 0, false};
    return Result(row_loss(std::size_t(num_classes), scores, t));
  }
  // The loss of the labels at scores[0, num_labels).
  template <typename Int>
  Result primal_loss(const Int num_classes, const Int num_labels, const Data* scores) const {
    const int64_t offsets[2] = {0, int64_t(num_labels)};
    const kernel::LabelSets t = {solver_detail::label_indices(std::size_t(num_labels)), offsets};
    return Result(row_loss(std::size_t(num_classes), scores, t));
  }
  // The loss of the probabilities q (summing to one).
  template <typename Int, typename Q>
  Result primal_loss(const Int num_classes, const Q* q, const Data* scores) const {
    const std::size_t m = std::size_t(num_classes);
    core::row_buffer<double> p(m);
    std::copy(q, q + m, p.data());
    const kernel::Probabilities<double> t = {p.data(), nullptr};
    return Result(row_loss(m, scores, t));
  }

  screening_rules screening() const { return screening_rules(); }

  // The evaluation's repair of a stored block (in a wider type): rebalance it.
  template <typename Int, typename T>
  inline void feasible_dual(const Int num_classes, T* variables) const {
    T sum(0);
    for (Int j = 1; j < num_classes; ++j) sum -= variables[j];
    variables[0] = sum;
  }

  template <typename Int>
  inline Result dual_loss(const Int, const Data* variables) const {
    return static_cast<Result>(variables[0]);
  }

  inline void update_primal_loss(Result& primal_loss_) const { primal_loss_ *= coeff_primal_loss; }

  inline void update_all(Result& primal_, Result& dual_, Result& primal_loss_, Result& dual_loss_,
                         Result& primal_regularizer_, Result& dual_regularizer_) const {
    primal_loss_ *= coeff_primal_loss;
    primal_ = primal_loss_ + primal_regularizer_;
    dual_ = dual_loss_ - dual_regularizer_;
  }
};

template <typename Data, typename Result>
const bool objective_base<Data, Result>::shrinks;
template <typename Data, typename Result>
const bool objective_base<Data, Result>::part_steps;

SDCA_ISA_END

#endif
