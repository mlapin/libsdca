#ifndef SDCA_SOLVER_EVAL_EVALUATE_H
#define SDCA_SOLVER_EVAL_EVALUATE_H

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "sdca/solver/data.h"
#include "sdca/solver/eval/accumulate.h"
#include "sdca/solver/eval/core.h"
#include "sdca/solver/eval/dual.h"
#include "sdca/solver/eval/primal.h"
#include "sdca/utility/parallel.h"

namespace sdca {

/*
 * The evaluation computes, in Work (at least double), the primal objective at
 * the stored model (the weights W, or the duals A of a kernel) and the dual
 * objective at the stored duals A, each block repaired by the objective's
 * feasible_dual. The dual regularizer comes from W0 + X A', recomputed from
 * A, never from the incrementally updated W. Each evaluation bounds its own
 * rounding (eval_train::gap_error), to first order in units of epsilon(Work):
 * a sum of magnitudes times the roundings per term. The objectives' losses
 * are 2c-Lipschitz in the max norm of the scores (their dual blocks have l1
 * norm at most 2c), the entropies less where they are small (loss_slope).
 *
 * The examples run on criteria.num_threads threads in fixed ranges whose
 * results add up in order: every count gives the same bits.
 */

// The objective in the evaluation's arithmetic.
template <typename Work, typename Data, typename Result,
          template <typename, typename> class Objective>
inline Objective<Work, Work> eval_objective(const Objective<Data, Result>& obj) {
  return Objective<Work, Work>(obj);
}

// Rounding of one primal loss: its score errors, and its own arithmetic on
// score differences (about 2 max|z| + m per unit of slope, see loss_slope).
template <typename Work, typename Objective>
inline Work primal_rounding(const Objective& obj, const size_type m, const Work score_error,
                            const Work max_score, const Work loss, const Work slope) {
  return slope * (score_error + 2 * max_score + static_cast<Work>(m)) +
         static_cast<Work>(m + 2) * static_cast<Work>(obj.coeff_primal_loss) * std::abs(loss);
}

// Rounding of one conjugate loss of a block with l1 norm block_l1 (the
// entropies take log c apart from log a_j).
template <typename Work, typename Objective>
inline Work dual_rounding(const Objective& obj, const size_type m, const Work loss,
                          const Work block_l1) {
  const Work log_c = std::abs(std::log(static_cast<Work>(obj.c)));
  return static_cast<Work>(m + 2) * std::abs(loss) + 2 * block_l1 * (1 + log_c);
}

template <typename Work>
inline Work max_abs(const size_type n, const Work* x) {
  Work max(0);
  for (size_type j = 0; j < n; ++j) max = std::max(max, std::abs(x[j]));
  return max;
}

// The largest column norm of a dim-by-m matrix.
template <typename Work, typename Data>
inline Work max_column_norm(const size_type dim, const size_type m, const Data* w) {
  Work max(0);
  for (size_type c = 0; c < m; ++c)
    max = std::max(max, std::sqrt(dot_chunked<Work>(dim, w + dim * c, w + dim * c)));
  return max;
}

// The norms of the n examples of a dim-by-n feature matrix.
template <typename Work, typename Data>
inline std::vector<Work> example_norms(const size_type dim, const size_type n, const Data* X,
                                       const std::size_t num_threads) {
  std::vector<Work> norms(n);
  detail::parallel_rows(n, dim, num_threads, [&](size_type first, size_type last) {
    size_type i = first;
    for (; i + 4 <= last; i += 4) {
      const Data* x[4] = {X + dim * i, X + dim * (i + 1), X + dim * (i + 2), X + dim * (i + 3)};
      Work sums[4];
      dot_chunked4<Work>(dim, x, x, sums);
      for (size_type q = 0; q < 4; ++q) norms[i + q] = std::sqrt(sums[q]);
    }
    for (; i < last; ++i) norms[i] = std::sqrt(dot_chunked<Work>(dim, X + dim * i, X + dim * i));
  });
  return norms;
}

// Primal losses at the scores that scores(i, z, scratch) writes and whose
// rounding it returns. Sets eval.primal_loss and the metrics; returns the
// rounding of the losses.
template <typename Work, typename Output, typename Objective, typename Evaluation, typename Scores>
inline Work eval_primal_losses(const size_type n, const Output& out, const Objective& obj,
                               const stopping_criteria& criteria, Evaluation& eval,
                               const Scores& scores) {
  const size_type m = out.num_classes;
  std::vector<Work> loss(n), bound(n), metric(n);
  detail::parallel_rows(n, scores.cost, criteria.num_threads, [&](size_type first, size_type last) {
    std::vector<Work> z(m), scratch(scores.scratch_size);
    for (size_type i = first; i < last; ++i) {
      const Work score_error = scores(i, z.data(), scratch.data());
      const Work max_score = max_abs(m, z.data());
      Work slope(0);
      loss[i] = eval_primal_loss(i, out, obj, criteria.projection, z.data(), slope, metric[i]);
      bound[i] = primal_rounding(obj, m, score_error, max_score, loss[i], slope);
    }
  });
  compensated_sum<Work> sum;
  Work total(0);
  for (size_type i = 0; i < n; ++i) {
    sum.add(loss[i]);
    total += bound[i];
    add_metric(out, metric[i], eval);
  }
  eval.primal_loss = sum.value();
  return total + static_cast<Work>(obj.coeff_primal_loss) * sum.magnitude;
}

// n entries in Work: the entries themselves, or their copy into wide.
template <typename Work>
inline const Work* widened(const Work* x, size_type, Work*) {
  return x;
}
template <typename Work, typename T>
inline const Work* widened(const T* x, const size_type n, Work* wide) {
  for (size_type j = 0; j < n; ++j) wide[j] = static_cast<Work>(x[j]);
  return wide;
}

// Scores W' x_i of weights (dim-by-m, the relative ones of a common center),
// and their rounding (Cauchy-Schwarz) from the example norms. Narrower
// weights and inputs are widened to Work once, not in every product (the
// same products).
template <typename Work, typename Weight, typename Data>
struct weight_scores {
  size_type dim, m, cost, scratch_size;
  std::vector<Work> wide;  // narrower weights in Work
  const Work* weights;
  const Data* inputs;  // dim-by-n
  const Work* norms;   // n
  Work steps, max_norm;

  weight_scores(size_type dim_, size_type m_, const Weight* weights_, const Data* inputs_,
                const std::vector<Work>& norms_, const double extra_steps = 0)
      : dim(dim_),
        m(m_),
        cost(dim_ * m_),
        scratch_size(std::is_same<Data, Work>::value ? 0 : dim_),
        wide(std::is_same<Weight, Work>::value ? 0 : dim_ * m_),
        weights(widened(weights_, dim_ * m_, wide.data())),
        inputs(inputs_),
        norms(norms_.data()),
        steps(static_cast<Work>(chunk_steps(dim_) + extra_steps)),
        max_norm(max_column_norm<Work>(dim_, m_, weights)) {}

  Work operator()(const size_type i, Work* z, Work* scratch) const {
    const Work* x = widened(inputs + dim * i, dim, scratch);
    const Work* const xs[4] = {x, x, x, x};
    size_type c = 0;
    for (; c + 4 <= m; c += 4) {
      const Work* const w[4] = {weights + dim * c, weights + dim * (c + 1), weights + dim * (c + 2),
                                weights + dim * (c + 3)};
      dot_chunked4<Work>(dim, w, xs, z + c);
    }
    for (; c < m; ++c) z[c] = dot_chunked<Work>(dim, weights + dim * c, x);
    return steps * norms[i] * max_norm;
  }
};

// Sums over the examples of a feature model's dual side.
template <typename Work>
struct feature_dual_sums {
  Work bound = 0, x_norm2 = 0, a_norm2 = 0, a_l1_x = 0;
};

// The repaired blocks of the stored duals (each loss to dual_loss) and
// H = X A' (dim-by-m, added to), in chunks of examples whose partial sums
// add to H in order; the entries of H run in parallel over the chunks of a
// span (blocks of at most 2^20 entries, or one chunk where X A' has under
// 2^24 products: a small pass is faster in calls that stay serial).
template <typename Work, typename Data, typename Output, typename Context, typename Objective>
inline feature_dual_sums<Work> eval_feature_duals(const Context& ctx, const size_type dim,
                                                  const Output& out, const Objective& obj,
                                                  const Data* X, const std::vector<Work>& norms,
                                                  compensated_sum<Work>& dual_loss,
                                                  std::vector<Work>& H) {
  const std::size_t threads = ctx.criteria.num_threads;
  const size_type m = out.num_classes, n = out.num_examples(), dm = dim * m;
  const size_type chunk = eval_chunk(n);
  const size_type span = static_cast<double>(n) * static_cast<double>(dm) < double(1 << 24)
                             ? chunk
                             : chunk * std::max<size_type>(1, (size_type(1) << 20) / (chunk * m));
  std::vector<Work> block(std::min(span, n) * m), part(dm, Work(0)), loss(std::min(span, n)),
      l1(std::min(span, n));
  feature_dual_sums<Work> sums;
  for (size_type i0 = 0; i0 < n; i0 += span) {
    const size_type count = std::min(span, n - i0);
    detail::parallel_rows(count, m, threads, [&](size_type first, size_type last) {
      for (size_type r = first; r < last; ++r) {
        Work* a = &block[m * r];
        loss[r] = eval_dual_block(i0 + r, out, obj, ctx.dual_variables + m * (i0 + r), a);
        l1[r] = 0;
        for (size_type c = 0; c < m; ++c) l1[r] += std::abs(a[c]);
      }
    });
    // Entries [first, last) of H: each example of a chunk adds to them in
    // turn (in order, as one serial pass would), then the chunk adds to H.
    detail::parallel_rows(dm, count, threads, [&](const size_type first, const size_type last) {
      for (size_type r0 = 0; r0 < count; r0 += chunk) {
        for (size_type r = r0; r < std::min(count, r0 + chunk); ++r) {
          const Data* x = X + dim * (i0 + r);
          for (size_type e = first; e < last;) {
            const size_type c = e / dim, j0 = e % dim, j1 = std::min(dim, j0 + (last - e));
            const Work a = block[m * r + c];
            Work* p = &part[e] - j0;
            if (a != 0)
              for (size_type j = j0; j < j1; ++j) p[j] += static_cast<Work>(x[j]) * a;
            e += j1 - j0;
          }
        }
        for (size_type e = first; e < last; ++e) {
          H[e] += part[e];
          part[e] = 0;
        }
      }
    });
    for (size_type r = 0; r < count; ++r) {
      const Work x_norm = norms[i0 + r];
      dual_loss.add(loss[r]);
      for (size_type c = 0; c < m; ++c) sums.a_norm2 += block[m * r + c] * block[m * r + c];
      sums.x_norm2 += x_norm * x_norm;
      sums.a_l1_x += l1[r] * x_norm;
      sums.bound += dual_rounding(obj, m, loss[r], l1[r]);
    }
  }
  return sums;
}

// Full centered objectives, without adding/subtracting |W0|^2/2:
//   P = |W-W0|^2/2 + losses(W)
//   D = conjugate_losses(A) - |H|^2/2 - <W0,H>, H = X A'.
// Interpret the rounded repaired block Ahat as an approximation to an exactly
// feasible block A*, as in the uncentered evaluator (Ahat itself need NOT sum
// to zero in exact arithmetic). Then H*=X A*' has zero class sum and
// <W0,H*> = <G0,H*>, where G0 subtracts W0's first class column. The repair
// term below bounds the effect of Hhat-H* using G0, not the common W0.
// This inherits the evaluator's first-order dual/conjugate repair bound;
// it does not assert exact feasibility of floating-point repaired entries.
// Losses are invariant to a score shared by every class: the primal scores
// subtract W's first column before the dot products, so a large common
// center enters neither the score arithmetic nor its error bound.
template <typename Work, typename Data, typename Output, typename Context, typename Objective>
inline void eval_centered_dataset(const Context& ctx,
                                  dataset<feature_input<Data>, Output, eval_train<Work, Output>>& d,
                                  const Objective& obj, eval_train<Work, Output>& eval) {
  const size_type m = d.num_classes(), n = d.num_examples(), dim = d.num_dimensions();
  const size_type dm = dim * m;
  const Data *X = d.in.features, *W0 = ctx.primal_initial;
  const std::vector<Work> norms = example_norms<Work>(dim, n, X, ctx.criteria.num_threads);
  std::vector<Work> H(dm, Work(0)), G0(dm);

  compensated_sum<Work> dual_loss;
  const feature_dual_sums<Work> s =
      eval_feature_duals(ctx, dim, d.out, obj, X, norms, dual_loss, H);
  eval.dual_loss = dual_loss.value();
  Work bound = s.bound;
  for (size_type c = 0; c < m; ++c)
    for (size_type j = 0; j < dim; ++j)
      G0[dim * c + j] = static_cast<Work>(W0[dim * c + j]) - static_cast<Work>(W0[j]);
  const Work hh = dot_chunked<Work>(dm, &H[0], &H[0]);
  const Work h_norm = std::sqrt(hh);
  const Work g_norm = std::sqrt(dot_chunked<Work>(dm, &G0[0], &G0[0]));
  const Work gh = dot_chunked<Work>(dm, &G0[0], &H[0]);
  eval.dual_regularizer = hh / 2 + gh;
  // dH from chunked products, dot products, G0 subtraction, dual repair,
  // and conjugate summation, all in units of epsilon(Work).
  bound +=
      static_cast<Work>(chunk_steps(n) + 2) * (h_norm + g_norm) * std::sqrt(s.x_norm2 * s.a_norm2) +
      static_cast<Work>(chunk_steps(dm) + 2) * (hh / 2 + g_norm * h_norm) +
      static_cast<Work>(m + 2) * s.a_l1_x *
          (max_column_norm<Work>(dim, m, &H[0]) + max_column_norm<Work>(dim, m, &G0[0])) +
      dual_loss.magnitude;

  // Retain the quadratic repair/accumulation term too: its linear part
  // vanishes when H and G0 vanish, but the true norm need not be exactly zero.
  const Work h_error_units =
      static_cast<Work>(chunk_steps(n) + 2) * std::sqrt(s.x_norm2 * s.a_norm2) +
      static_cast<Work>(m + 2) * s.a_l1_x;
  bound += (std::numeric_limits<Work>::epsilon() / 2 * h_error_units) * h_error_units;

  // Let W = W0 + X A' on request.
  if (ctx.criteria.recompute_primal)
    for (size_type j = 0; j < dm; ++j)
      ctx.primal_variables[j] = static_cast<Data>(static_cast<Work>(W0[j]) + H[j]);

  // Primal at the stored W; G0 and H are reused for W - W0 and its relative weights.
  const typename Context::variable_type* W = ctx.primal_variables;
  std::vector<Work>& E = G0;
  for (size_type j = 0; j < dm; ++j) E[j] = static_cast<Work>(W[j]) - static_cast<Work>(W0[j]);
  const Work ee = dot_chunked<Work>(dm, &E[0], &E[0]);
  eval.primal_regularizer = ee / 2;
  bound += static_cast<Work>(chunk_steps(dm) + 2) * ee / 2;
  std::vector<Work>& relative = H;
  for (size_type c = 0; c < m; ++c)
    for (size_type j = 0; j < dim; ++j)
      relative[dim * c + j] = static_cast<Work>(W[dim * c + j]) - static_cast<Work>(W[j]);
  bound += eval_primal_losses<Work>(
      n, d.out, obj, ctx.criteria, eval,
      weight_scores<Work, Work, Data>(dim, m, relative.data(), X, norms, 1));
  eval_end(m, n, ctx, obj, bound, eval);
}

// Training set on features: P(W) + D(A), with V = X A' for the dual.
template <typename Work, typename Data, typename Output, typename Context, typename Objective>
inline void eval_dataset(const Context& ctx,
                         dataset<feature_input<Data>, Output, eval_train<Work, Output>>& d,
                         const Objective& obj, eval_train<Work, Output>& eval) {
  if (ctx.primal_initial) {
    eval_centered_dataset(ctx, d, obj, eval);
    return;
  }
  const size_type m = d.num_classes(), n = d.num_examples(), dim = d.num_dimensions();
  const size_type dm = dim * m;
  const Data* X = d.in.features;
  const std::vector<Work> norms = example_norms<Work>(dim, n, X, ctx.criteria.num_threads);
  std::vector<Work> V(dm, Work(0));

  // Dual: chunks of examples bound the rounding of V's long sums.
  compensated_sum<Work> dual_loss;
  const feature_dual_sums<Work> s =
      eval_feature_duals(ctx, dim, d.out, obj, X, norms, dual_loss, V);
  eval.dual_loss = dual_loss.value();
  const Work vv = dot_chunked<Work>(dm, &V[0], &V[0]);
  eval.dual_regularizer = vv / 2;
  // |V||dV| with |dV| <= steps |X||A'|; the scores at V of each block's own
  // rounding; the conjugate losses' sum.
  Work bound =
      s.bound +
      static_cast<Work>(chunk_steps(n) + 2) * std::sqrt(vv) * std::sqrt(s.x_norm2 * s.a_norm2) +
      static_cast<Work>(chunk_steps(dm)) * vv / 2 +
      static_cast<Work>(m + 2) * s.a_l1_x * max_column_norm<Work>(dim, m, &V[0]) +
      dual_loss.magnitude;

  // Let W = X A' on request.
  if (ctx.criteria.recompute_primal)
    for (size_type j = 0; j < dm; ++j) ctx.primal_variables[j] = static_cast<Data>(V[j]);

  // Primal at the stored W.
  typedef typename Context::variable_type Weight;
  const Weight* W = ctx.primal_variables;
  const Work ww = dot_chunked<Work>(dm, W, W);
  eval.primal_regularizer = ww / 2;
  bound += static_cast<Work>(chunk_steps(dm) + 1) * ww / 2;
  bound += eval_primal_losses<Work>(n, d.out, obj, ctx.criteria, eval,
                                    weight_scores<Work, Weight, Data>(dim, m, W, X, norms));

  eval_end(m, n, ctx, obj, bound, eval);
}

template <typename Work, typename Data, typename Output, typename Context, typename Objective>
inline void eval_dataset(const Context& ctx,
                         dataset<feature_input<Data>, Output, eval_test<Work, Output>>& d,
                         const Objective& obj, eval_test<Work, Output>& eval) {
  const size_type m = d.num_classes(), n = d.num_examples(), dim = d.num_dimensions();
  const std::vector<Work> norms =
      example_norms<Work>(dim, n, d.in.features, ctx.criteria.num_threads);
  eval_primal_losses<Work>(n, d.out, obj, ctx.criteria, eval,
                           weight_scores<Work, typename Context::variable_type, Data>(
                               dim, m, ctx.primal_variables, d.in.features, norms));
  eval_end(m, n, obj, eval);
}

// Scores A' K_i of n-by-m duals (one or two matrices) at a kernel column, in
// chunks of the n rows; their rounding from the exact magnitudes
// sum_l |A_lc K_li|.
template <typename Work, typename Data>
struct kernel_scores {
  size_type n, m, cost, scratch_size;
  const Data* gram;  // n-by-examples
  Work steps;

  kernel_scores(size_type n_, size_type m_, const Data* gram_, const size_type matrices = 1)
      : n(n_),
        m(m_),
        cost(n_ * m_ * matrices),
        scratch_size(2 * m_ * matrices),
        gram(gram_),
        steps(static_cast<Work>(chunk_steps(n_) + 1)) {}

  // z = A' K_i and its rounding; magnitude and part are m-vectors of scratch.
  template <typename Dual>
  Work operator()(const size_type i, const Dual* A, Work* z, Work* magnitude, Work* part) const {
    const Data* k = gram + n * i;
    const size_type chunk = eval_chunk(n);
    std::fill(z, z + m, Work(0));
    std::fill(magnitude, magnitude + m, Work(0));
    for (size_type l0 = 0; l0 < n; l0 += chunk) {
      std::fill(part, part + m, Work(0));
      for (size_type l = l0; l < std::min(n, l0 + chunk); ++l) {
        const Work kl = static_cast<Work>(k[l]);
        if (kl == 0) continue;
        const Dual* a = A + m * l;
        for (size_type c = 0; c < m; ++c) {
          const Work product = static_cast<Work>(a[c]) * kl;
          part[c] += product;
          magnitude[c] += std::abs(product);
        }
      }
      for (size_type c = 0; c < m; ++c) z[c] += part[c];
    }
    return steps * *std::max_element(magnitude, magnitude + m);
  }
};

// The scores of the stored duals A of a kernel model.
template <typename Work, typename Data, typename Dual = Data>
struct stored_kernel_scores : kernel_scores<Work, Data> {
  const Dual* A;
  stored_kernel_scores(size_type n_, size_type m_, const Dual* A_, const Data* gram_)
      : kernel_scores<Work, Data>(n_, m_, gram_), A(A_) {}

  Work operator()(const size_type i, Work* z, Work* scratch) const {
    return kernel_scores<Work, Data>::operator()(i, A, z, scratch, scratch + this->m);
  }
};

// The repairs R = A_repaired - A of the stored duals A, row by row: their
// nonzeros (a feasible block changes in its label entry at most).
template <typename Work>
struct dual_repairs {
  std::vector<size_type> start, column;
  std::vector<Work> value;

  template <typename Data>
  dual_repairs(const size_type n, const size_type m, const Data* stored, const Work* repaired)
      : start(n + 1, 0) {
    for (size_type l = 0; l < n; ++l) {
      for (size_type c = 0; c < m; ++c) {
        const Work r = repaired[m * l + c] - static_cast<Work>(stored[m * l + c]);
        if (r == 0) continue;
        column.push_back(c);
        value.push_back(r);
      }
      start[l + 1] = column.size();
    }
  }
};

// The training scores of a kernel model at the stored duals A (the primal
// losses and regularizer) and at the repaired ones A + R (the dual
// regularizer), from the pass over the kernel and the few repairs. Writes
// the regularizer terms <a_i, A' K_i> of example i to primal[i] and dual[i],
// and their rounding to bound[i] (half of each: the regularizers enter the
// gap halved).
template <typename Work, typename Data, typename Dual = Data>
struct training_kernel_scores : kernel_scores<Work, Data> {
  const Dual* stored;
  const Work* repaired;
  const dual_repairs<Work>& repairs;
  Work *primal, *dual, *bound;

  training_kernel_scores(size_type n_, size_type m_, const Dual* stored_, const Work* repaired_,
                         const dual_repairs<Work>& repairs_, const Data* gram_, Work* primal_,
                         Work* dual_, Work* bound_)
      : kernel_scores<Work, Data>(n_, m_, gram_, 2),
        stored(stored_),
        repaired(repaired_),
        repairs(repairs_),
        primal(primal_),
        dual(dual_),
        bound(bound_) {
    this->cost = n_ * m_;
  }

  // <a, z> and its rounding from the rounding of z.
  Work inner(const Work* a, const Work* z, const Work error, Work& value) const {
    Work l1(0), magnitude(0);
    value = 0;
    for (size_type c = 0; c < this->m; ++c) {
      value += a[c] * z[c];
      l1 += std::abs(a[c]);
      magnitude += std::abs(a[c] * z[c]);
    }
    return l1 * error + static_cast<Work>(this->m + 1) * magnitude;
  }

  Work operator()(const size_type i, Work* z, Work* scratch) const {
    const size_type classes = this->m, rows = this->n;
    Work *magnitude = scratch, *part = scratch + classes, *y = scratch + 2 * classes,
         *a = scratch + 3 * classes;
    const Work error = kernel_scores<Work, Data>::operator()(i, stored, z, magnitude, part);
    // y = z + R' K_i, the n-term sums of R' K_i bounded plainly; part holds |R' K_i|.
    const Data* k = this->gram + rows * i;
    std::fill(y, y + classes, Work(0));
    std::fill(part, part + classes, Work(0));
    for (size_type l = 0; l < rows; ++l)
      for (size_type p = repairs.start[l]; p < repairs.start[l + 1]; ++p) {
        const Work product = repairs.value[p] * static_cast<Work>(k[l]);
        y[repairs.column[p]] += product;
        part[repairs.column[p]] += std::abs(product);
      }
    for (size_type c = 0; c < classes; ++c) y[c] += z[c];
    Work dual_error(0);
    for (size_type c = 0; c < classes; ++c)
      dual_error = std::max(dual_error, this->steps * magnitude[c] +
                                            static_cast<Work>(rows + 1) * part[c] + std::abs(y[c]));
    for (size_type c = 0; c < classes; ++c) a[c] = static_cast<Work>(stored[classes * i + c]);
    bound[i] =
        (inner(a, z, error, primal[i]) + inner(repaired + classes * i, y, dual_error, dual[i])) / 2;
    return error;
  }
};

// The repaired training duals (n-by-m) of a kernel model; returns the
// rounding of their conjugate losses, whose sum goes to dual_loss.
template <typename Work, typename Context, typename Objective>
inline Work eval_kernel_duals(const Context& ctx, const Objective& obj, std::vector<Work>& A,
                              Work& dual_loss) {
  const size_type m = ctx.train.num_classes(), n = ctx.train.num_examples();
  A.resize(n * m);
  std::vector<Work> loss(n), bound(n);
  detail::parallel_rows(n, m, ctx.criteria.num_threads, [&](size_type first, size_type last) {
    for (size_type i = first; i < last; ++i) {
      Work* a = &A[m * i];
      loss[i] = eval_dual_block(i, ctx.train.out, obj, ctx.dual_variables + m * i, a);
      Work l1(0);
      for (size_type c = 0; c < m; ++c) l1 += std::abs(a[c]);
      bound[i] = dual_rounding(obj, m, loss[i], l1);
    }
  });
  compensated_sum<Work> sum;
  Work total(0);
  for (size_type i = 0; i < n; ++i) {
    sum.add(loss[i]);
    total += bound[i];
  }
  dual_loss = sum.value();
  return total + sum.magnitude;
}

// Training set on a kernel: the primal at the stored duals A (the model:
// scores A' K_i and the regularizer sum_i <a_i, A' K_i> / 2), the dual at
// the repaired ones.
template <typename Work, typename Data, typename Output, typename Context, typename Objective>
inline void eval_dataset(const Context& ctx,
                         dataset<kernel_input<Data>, Output, eval_train<Work, Output>>& d,
                         const Objective& obj, eval_train<Work, Output>& eval) {
  const size_type m = d.num_classes(), n = d.num_examples();
  std::vector<Work> A, primal(n), dual(n), terms(n);
  Work bound = eval_kernel_duals(ctx, obj, A, eval.dual_loss);
  const dual_repairs<Work> repairs(n, m, ctx.dual_variables, A.data());
  bound +=
      eval_primal_losses<Work>(n, d.out, obj, ctx.criteria, eval,
                               training_kernel_scores<Work, Data, typename Context::variable_type>(
                                   n, m, ctx.dual_variables, A.data(), repairs, d.in.kernel,
                                   primal.data(), dual.data(), terms.data()));
  compensated_sum<Work> primal_regularizer, dual_regularizer;
  for (size_type i = 0; i < n; ++i) {
    primal_regularizer.add(primal[i]);
    dual_regularizer.add(dual[i]);
    bound += terms[i];
  }
  eval.primal_regularizer = primal_regularizer.value() / 2;
  eval.dual_regularizer = dual_regularizer.value() / 2;
  bound += (primal_regularizer.magnitude + dual_regularizer.magnitude) / 2;
  eval_end(m, n, ctx, obj, bound, eval);
}

// A test set on a kernel: the scores of the stored duals A.
template <typename Work, typename Data, typename Output, typename Context, typename Objective>
inline void eval_dataset(const Context& ctx,
                         dataset<kernel_input<Data>, Output, eval_test<Work, Output>>& d,
                         const Objective& obj, eval_test<Work, Output>& eval) {
  const size_type m = d.num_classes(), n = d.num_examples();
  eval_primal_losses<Work>(n, d.out, obj, ctx.criteria, eval,
                           stored_kernel_scores<Work, Data, typename Context::variable_type>(
                               d.in.num_train_examples, m, ctx.dual_variables, d.in.kernel));
  eval_end(m, n, obj, eval);
}

}  // namespace sdca

#endif
