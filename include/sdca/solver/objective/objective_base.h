#ifndef SDCA_SOLVER_OBJECTIVE_OBJECTIVE_BASE_H
#define SDCA_SOLVER_OBJECTIVE_OBJECTIVE_BASE_H

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "sdca/math/blas.h"
#include "sdca/math/compensated.h"
#include "sdca/solver/objective/feasible_dual.h"
#include "sdca/utility/types.h"

namespace sdca {

namespace detail {

// The working precision of the hinge steps: the wider of storage and result.
template <typename Data, typename Result>
using hinge_work = typename std::common_type<Data, Result>::type;

// beta v + a (z - z0): a scal by beta followed by an axpy, element by element.
template <typename T>
inline T hinge_scaled(const T a, const T beta, const T v, const T z, const T z0) {
  T y = beta * v;
  y += a * (z - z0);
  return y;
}

// The top-k hinge point beta (v - v0) + a (z - z0) + a with every rounding
// error kept (in twice the precision of T).
template <typename T>
inline T topk_hinge_compensated(const T a, const T beta, const T v, const T z, const T z0,
                                const T v0) {
  compensated_sum<T> s = scaled_difference(a, beta, v, z, z0);
  s.add(a);
  s.add_product(-beta, v0);
  return s.value();
}

// Whether a point entry is below 1/16 of its warm terms: the scaled old dual
// cancels a (z - z0) and may hold the residual in its rounding (warm cancellation).
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
  using W = warm_work<Work>;
  const Work warm =
      warm_entry(topk_hinge_compensated(W(a), W(beta), W(vj), W(zj), W(z0), W(v0)), y);
  return {warm, warm != y ? terms : Work(0)};
}

// Plain top-k entries [j, n).
template <typename Work, typename Data, typename Int>
inline void topk_hinge_plain(Int j, const Int n, const Work a, const Work beta, const Work z0,
                             const Work shift, const Data* z, Data* v) {
  for (; j < n; ++j)
    v[j] = static_cast<Data>(hinge_scaled(a, beta, Work(v[j]), Work(z[j]), z0) + shift);
}

// Top-k entries of a warm block from its first cancelling one: plain into v,
// recomputed into z (scratch), until an entry reaches 'bound' (no warm cancellation; returns
// the next index). Otherwise the recomputed entries replace the plain ones if
// the largest entry is below the cancelled terms too (warm cancellation); returns n.
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

// The vector a top-k hinge step projects, in one pass and in the working
// precision Work: u = beta v + a (z - z0) relative to the label's score z0,
// then v_0 = u_0 and v_j = u_j + (a - u_0). From an entry where the scaled
// old dual cancels on, see topk_hinge_warm_tail.
template <typename Work, typename Data, typename Int>
inline void topk_hinge_point(const Int n, const Work a, const Work beta, Data* z, Data* v) {
  const Work z0 = Work(z[0]), v0 = Work(v[0]), u0 = hinge_scaled(a, beta, v0, z0, z0),
             shift = a - u0;
  v[0] = static_cast<Data>(u0);
  // Warm cancellation needs |y| below (|beta v_j| + |shift|) / 16 <= 2 |shift| / 16 in a feasible
  // block, at every entry; a cold block (v0 = 0, so v = 0) has no warm terms.
  const Work offset = std::abs(shift), bound = 2 * margin_cancellation<Work>() * offset;
  Int j = 1;
  for (; v0 != 0 && j < n; ++j) {
    const Work zj = Work(z[j]), vj = Work(v[j]);
    const Work y = hinge_scaled(a, beta, vj, zj, z0) + shift;
    if (y >= bound) {
      v[j++] = static_cast<Data>(y);
      break;
    } else if (warm_cancels(y, beta * vj, offset)) {
      j = topk_hinge_warm_tail(j, n, a, beta, z0, v0, shift, bound, z, v);
      break;
    }
    v[j] = static_cast<Data>(y);
  }
  topk_hinge_plain(j, n, a, beta, z0, shift, z, v);
}

// A multilabel entry about label 0, -w at a label and w + a elsewhere with
// w = u - u0, recomputed where the scaled old dual cancels a (z - z0).
// Every plain entry carries u0, so the terms count even if w is exact.
template <bool label, typename Work>
inline warm_result<Work> multilabel_warm(const Work a, const Work beta, const Work vj,
                                         const Work zj, const Work z0, const Work v0, const Work u,
                                         const Work u0, const Work offset) {
  const Work plain = label ? -(u - u0) : (u - u0) + a, terms = std::abs(beta * vj) + offset;
  if (!warm_cancels(plain, Work(0), terms)) return {plain, Work(0)};
  using W = warm_work<Work>;
  compensated_sum<W> s = scaled_difference(W(a), W(beta), W(vj), W(zj), W(z0));
  s.add_product(-W(beta), W(v0));
  if (!label) s.add(W(a));
  return {warm_entry(label ? -s.value() : s.value(), plain), terms};
}

// Plain multilabel entries a / 2 - u (labels) or u + a / 2 on [j, end), up
// to the first whose entry about label 0 cancels: its index, or end. A
// non-label entry about label 0 at or above 'bound' rules warm cancellation out: the rest
// is plain and the result end + 1.
template <bool label, typename Work, typename Data, typename Int>
inline Int multilabel_plain(Int j, const Int end, const Work a, const Work beta, const Work z0,
                            const Work u0, const Work offset, const Work bound, const Data* z,
                            Data* v) {
  const Work half = a / 2;
  for (; j < end; ++j) {
    const Work vj = Work(v[j]), u = hinge_scaled(a, beta, vj, Work(z[j]), z0);
    const Work w = label ? -(u - u0) : (u - u0) + a;
    if (!label && w >= bound) {
      for (; j < end; ++j)
        v[j] = static_cast<Data>(hinge_scaled(a, beta, Work(v[j]), Work(z[j]), z0) + half);
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

// Multilabel entries [first, n) of a warm block from its first cancelling one,
// as in multilabel_both; if the largest pairwise margin also cancels, the
// point about label 0 replaces the plain one (earlier entries shifted).
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

// The plain point of a warm multilabel block with 0 < p < n (false if cold);
// see multilabel_warm_tail.
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
  // is at most the label mass in a feasible block: an entry about label 0
  // past 'bound' at a non-label (whose margin with label 0 exceeds it) rules it out.
  const Work bound = 2 * margin_cancellation<Work>() * (std::abs(beta) * mass + offset);
  v[0] = static_cast<Data>(a / 2 - u0);
  Int first = multilabel_plain<true>(Int(1), p, a, beta, z0, u0, offset, bound, z, v);
  if (first == p) first = multilabel_plain<false>(p, n, a, beta, z0, u0, offset, bound, z, v);
  if (first < n) multilabel_warm_tail(first, n, p, a, beta, z0, v0, u0, z, v);
  return true;
}

// The vector a multilabel hinge step projects, in Work: a / 2 - u at the p
// labels and u + a / 2 elsewhere (see multilabel_warm_point for warm blocks).
template <typename Work, typename Data, typename Int>
inline void multilabel_hinge_point(const Int n, const Int p, const Work a, const Work beta, Data* z,
                                   Data* v) {
  const Work z0 = Work(z[0]);
  if (p > 0 && p < n && multilabel_warm_point(n, p, a, beta, z, v)) return;
  const Work half = a / 2;
  for (Int j = 0; j < p; ++j)
    v[j] = static_cast<Data>(half - hinge_scaled(a, beta, Work(v[j]), Work(z[j]), z0));
  for (Int j = p; j < n; ++j)
    v[j] = static_cast<Data>(hinge_scaled(a, beta, Work(v[j]), Work(z[j]), z0) + half);
}

// The dual block at zero scores, p labels first: s / p on every label and
// -s / (n - p) on every other class, the first entry balancing the rest.
template <typename Data, typename Result, typename Int>
inline void zero_scores_dual(const Int n, const Int p, const Result s, Data* variables) {
  using Work = typename std::common_type<Data, Result, double>::type;
  const Work label = Work(s) / Work(p), other = -Work(s) / Work(n - p);
  // Leave the block unchanged if it does not fit the storage type.
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

// k of a top-k objective lies in [1, num_classes - 1]. solver_context checks
// it once; the steps and losses check it again for direct callers.
template <typename Int>
inline void check_topk(const size_type k, const Int num_classes) {
  if (k < 1 || k >= static_cast<size_type>(num_classes))
    throw std::invalid_argument("k must lie in [1, num_classes-1]");
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

}  // namespace detail

template <typename Data, typename Result>
struct objective_base {
  typedef Data data_type;
  typedef Result result_type;

  const Result coeff_primal_loss;

  explicit objective_base(const Result coeff_primal_loss_ = 1)
      : coeff_primal_loss(coeff_primal_loss_) {
    if (!std::isfinite(coeff_primal_loss_) || !(coeff_primal_loss_ > 0))
      throw std::invalid_argument("Objective loss scale must be finite and positive");
  }

  inline std::string to_string() const {
    std::ostringstream str;
    str << "objective (precision: " << type_name<Result>() << ")";
    return str.str();
  }

  // The evaluation's repair of a stored block (in a wider type): rebalance it.
  template <typename Int, typename T>
  inline void feasible_dual(const Int num_classes, T* variables) const {
    T sum(0);
    for (Int j = 1; j < num_classes; ++j) sum -= variables[j];
    variables[0] = sum;
  }

  template <typename Int>
  inline Result primal_loss(const Int, Data*) const {
    return Result(0);
  }

  template <typename Int>
  inline Result dual_loss(const Int,  // num_classes
                          const Data* variables) const {
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

}  // namespace sdca

#endif
