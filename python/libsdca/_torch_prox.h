#ifndef SDCA_TORCH_PROX_H
#define SDCA_TORCH_PROX_H
#include "_cpu_lanes.h"

namespace {
namespace sdca_cpu {
// f (an ATen Vectorized function) on every lane of x.
template <typename T, typename V, typename F>
inline V prox_on_lanes(V x, F f) {
  using A = at::vec::Vectorized<T>;
  constexpr int L = sizeof(V) / sizeof(T);
  alignas(64) T buffer[L];
  std::memcpy(buffer, &x, sizeof x);
  if constexpr (A::size() >= L) {
    f(A::loadu(buffer, L)).store(buffer, L);
  } else {
    for (int i = 0; i < L; i += A::size()) f(A::loadu(buffer + i)).store(buffer + i);
  }
  std::memcpy(&x, buffer, sizeof x);
  return x;
}

inline dlanes::vd proximal_log(dlanes::vd x) {
  return prox_on_lanes<double>(x, [](auto a) { return a.log(); });
}
}  // namespace sdca_cpu
}  // namespace
#include "_cpu_prox.h"

namespace {
// proximal_cross_entropy of one row in T: work holds 4 c entries (scores, residuals, the
// weights u of proxlanes::Model and its residuals).
// Single-label tail with a proximal perturbation below working precision.
// Keep the exponent split until after summing, including aggregate subnormals.
template <typename T, typename G>
double prox_tiny_tail(int64_t c, const T* v, T other, G* gradient) {
  const double half = std::exp((double(other) - double(v[0])) / 2);
  double total = 0;
  for (int64_t j = 1; j < c; ++j) {
    const double term = std::exp(double(v[j]) - double(other));
    total += term;
    gradient[j] = static_cast<G>((half * term) * half);
  }
  const double mass = half * (half * total);
  gradient[0] = static_cast<G>(-mass);
  return mass;
}

template <typename Data, typename T, typename G>
double prox_entropy_row(int64_t c, int64_t p, const Data* z_perm, double alpha_d, T* work,
                        G* g_perm) {
  T *v = work, *r = work + c, *u = work + 2 * c, *expanded = work + 3 * c;
  if (std::numeric_limits<T>::digits < std::numeric_limits<double>::digits) {
    const double tiny = static_cast<double>(std::numeric_limits<T>::min());
    T lo_z = T(z_perm[0]), hi_z = T(z_perm[0]);
    for (int64_t j = 1; j < c; ++j) {
      lo_z = std::min(lo_z, T(z_perm[j]));
      hi_z = std::max(hi_z, T(z_perm[j]));
    }
    if (!(tiny <= alpha_d && alpha_d <= 1.0 / tiny && std::isfinite(hi_z - lo_z))) {
      std::vector<double> wd(4 * c);
      return prox_entropy_row<Data, double>(c, p, z_perm, alpha_d, wd.data(), g_perm);
    }
  }
  T origin = -std::numeric_limits<T>::infinity();
  for (int64_t j = 0; j < c; ++j) {
    const T zj = static_cast<T>(z_perm[j]);
    TORCH_CHECK_VALUE(std::isfinite(zj),
                      "proximal_cross_entropy needs finite scores, labels and a non-target class");
    if (zj > origin) origin = zj;
  }
  for (int64_t j = 0; j < c; ++j) {
    v[j] = static_cast<T>(z_perm[j]) - origin;
    TORCH_CHECK_VALUE(std::isfinite(v[j]),
                      "proximal_cross_entropy needs finite scores, labels and a non-target class");
  }
  const T alpha = static_cast<T>(alpha_d);
  const T q = T(1) / static_cast<T>(p);
  const T log_alpha = std::log(alpha), log_q = std::log(q);
  const T clip_th = alpha * (q - T(1)) + log_q;
  T v_max_label = v[0], v_max_other = v[p];
  for (int64_t j = 1; j < p; ++j)
    if (v[j] > v_max_label) v_max_label = v[j];
  for (int64_t j = p + 1; j < c; ++j)
    if (v[j] > v_max_other) v_max_other = v[j];
  const T gap = v_max_other - v_max_label;
  if (p == 1 && std::isfinite(gap) && gap < std::log(std::numeric_limits<T>::min())) {
    const T tail = gap + std::log(T(c - 1)), error = std::max(T(0), log_alpha) + tail;
    const T bound = std::log(std::numeric_limits<T>::epsilon() / T(8));
    if (error < bound || error + tail < std::log(std::numeric_limits<T>::min()) + bound) {
      return prox_tiny_tail(c, v, v_max_other, g_perm);
    }
  }
  const T v_ref = std::max(v_max_label, v_max_other);

  T lo =
      std::max((v_max_label - v_ref) - alpha * (T(1) - q), (v_max_other - v_ref) - alpha) + log_q;
  T hi = std::max((v_max_label - v_ref) + alpha * q, (v_max_other - v_ref)) + log_q +
         std::log(static_cast<T>(c));
  if (!(lo <= hi)) hi = lo;

  const bool tiny_tail = p == 1 && v_max_label == v_ref &&
                         (v_max_other - v_ref) + std::log(static_cast<T>(c)) <
                             std::log(std::numeric_limits<T>::epsilon()) / T(2);
  const T start_dt = tiny_tail ? log_q : proxlanes::prox_start(v, c, p, v_ref, q, log_q, alpha, r);
  T dt = std::max(lo, std::min(hi, start_dt));
  const T eps = std::numeric_limits<T>::epsilon(), tiny = std::numeric_limits<T>::min();
  T f(0), derivative(0), curvature(0), magnitude(0), other_mass(0);
  // Whether the last evaluation clipped a probability (no model about it).
  bool clipped = false;
  // f sums in double, as entropy_block_solver::evaluate.
  auto eval_dt = [&](T cur_dt) {
    double sum = 0.0;
    derivative = T(0);
    curvature = T(0);
    magnitude = T(0);
    other_mass = T(0);
    clipped = false;
    for (int64_t j = 0; j < p; ++j) {
      const T arg = (v_ref - v[j]) + cur_dt;
      const T rj = prox_residual(arg, alpha, q, log_alpha, log_q);
      r[j] = rj;
      sum += static_cast<double>(rj);
      magnitude += std::abs(rj);
      u[j] = T(0);
      if (arg >= clip_th) {
        const T prob = q - rj;
        u[j] = T(1) / (T(1) + alpha * prob);
        const T d1 = prob * u[j];
        derivative += d1;
        curvature -= (d1 * u[j]) * u[j];
      } else {
        clipped = true;
      }
    }
    clipped |= proxlanes::prox_others(v, p, c, v_ref, cur_dt, log_q, alpha, log_alpha, r, u, sum,
                                      derivative, curvature, magnitude, other_mass);
    f = static_cast<T>(sum);
  };
  // Near the root, the Taylor model about the last evaluation (proxlanes::Model) gives the
  // root and its residuals without another one: true with dt, r, f and magnitude there;
  // else false with its root in delta (when finite and inside [lo, hi]).
  using Model = proxlanes::Model<T>;
  auto model_root = [&](T& delta) {
    Model model;
    for (int64_t j = 0; j < p; ++j) {
      const double prob = static_cast<double>(q - r[j]), uj = u[j];
      double x = prob;
      for (int m = 1; m < proxlanes::kModelPowers; ++m) model.moment[m] += (x *= uj);
    }
    model.add(r, u, p, c, T(0));
    const double f0 = f;
    double d = delta;
    for (int it = 0; it < 8; ++it) {
      model.at(d);
      const double step = (f0 - model.change()) / model.slope();
      if (!std::isfinite(step)) return false;
      d -= step;
      if (std::abs(step) <= 1e-3 * eps * std::abs(d)) break;
    }
    for (int attempt = 0; attempt < 3; ++attempt) {
      const T next = dt + static_cast<T>(d);
      if (!(lo <= next && next <= hi)) return false;
      delta = next - dt;
      if (!(std::abs(static_cast<double>(delta)) <= Model::reach)) return false;
      d = delta;
      model.at(d);
      // The remainder relative to eps / 32 of the probability.
      const double bound = std::pow(std::abs(d) / Model::reach, proxlanes::kModelOrder + 1);
      double sum = 0.0;
      T size(0);
      for (int64_t j = 0; j < p; ++j) {
        const T prob = q - r[j];
        double sj = 0;
        for (int m = proxlanes::kModelPowers - 1; m > 0; --m) sj = (sj + model.b[m]) * u[j];
        const T change = prob * static_cast<T>(sj), rj = r[j] - change;
        // A label whose residual would lose its relative accuracy is evaluated exactly.
        const bool kept = std::abs(change) <= std::abs(r[j]) / T(2) &&
                          prob * bound <= std::abs(r[j]) && rj > q - T(1);
        expanded[j] = kept ? rj : prox_residual((v_ref - v[j]) + next, alpha, q, log_alpha, log_q);
        sum += static_cast<double>(expanded[j]);
        size += std::abs(expanded[j]);
      }
      model.expand(r, u, p, c, expanded, sum, size);
      if (std::abs(sum) <= double(T(8) * eps * std::max(size, tiny))) {
        std::copy(expanded, expanded + c, r);
        dt = next;
        f = static_cast<T>(sum);
        magnitude = size;
        return true;
      }
      d -= sum / model.slope();
    }
    delta = static_cast<T>(d);
    return false;
  };
  for (int it = 0; it < 200; ++it) {
    eval_dt(dt);
    if (std::abs(f) <= T(8) * eps * std::max(magnitude, tiny)) break;
    if (f < T(0))
      lo = dt;
    else
      hi = dt;
    T delta(0);
    if (derivative > T(0) && std::isfinite(derivative)) {
      const T newton = -f / derivative;
      if (std::isfinite(newton)) {
        T corr = (f * curvature) / (T(2) * derivative * derivative);
        delta = newton;
        if (std::isfinite(corr)) {
          corr = std::max(T(-0.5), std::min(T(0.5), corr));
          delta = newton / (T(1) - corr);
        }
        if (f < T(0) && other_mass > T(1)) {
          const T log_step =
              std::log(other_mass) + alpha * (other_mass - T(1)) / static_cast<T>(c - p);
          if (std::isfinite(log_step) && log_step > delta) delta = log_step;
        }
      }
    }
    if (!clipped && std::abs(delta) <= T(2 * Model::reach)) {
      T root = delta;
      if (model_root(root)) break;
      if (std::isfinite(root) && root != T(0)) delta = root;
    }
    T next = dt + delta;
    if (delta != T(0) && next == dt) {
      lo = hi = dt;
      break;
    }
    if (!(lo <= next && next <= hi && next != dt) || !std::isfinite(next))
      next = lo + (hi - lo) / T(2);
    if (next == dt) break;
    dt = next;
  }
  if (!std::isfinite(f) || (std::abs(f) > T(256) * eps * std::max(magnitude, tiny) &&
                            hi - lo > T(64) * eps * std::max({std::abs(lo), std::abs(hi), tiny}))) {
    throw std::runtime_error("Entropy normalization did not converge");
  }
  int64_t pivot = 0;
  if (p > 1) {
    bool labels_zero = true;
    for (int64_t j = 0; j < p; ++j) labels_zero = labels_zero && (r[j] == T(0));
    T mass(0);
    if (labels_zero) {
      for (int64_t j = p; j < c; ++j) mass += r[j];
    }
    if (mass != T(0)) {
      for (int64_t j = 1; j < p; ++j) r[j] = -mass / static_cast<T>(p);
    } else {
      for (int64_t j = 1; j < c; ++j) {
        if (std::abs(r[j]) > std::abs(r[pivot])) pivot = j;
      }
    }
  }
  const double others = proxlanes::lane_sum(r, 0, pivot) + proxlanes::lane_sum(r, pivot + 1, c);
  r[pivot] = static_cast<T>(-others);
  for (int64_t j = 0; j < c; ++j) g_perm[j] = static_cast<G>(-r[j]);

  auto label_log_p = [&](int64_t j) -> T {
    if (r[j] <= q / T(2)) return log_q + std::log1p(-r[j] / q);
    const T h = log_q + (alpha * q - ((v_ref - v[j]) + dt));
    return h - lambert_w_exp_t(h + log_alpha);
  };
  T cross(0);
  for (int64_t j = 0; j < p; ++j) cross -= label_log_p(j);
  const T penalty = proxlanes::lane_penalty(r, c, alpha);
  if (!std::isfinite(cross)) {
    cross = T(0);
    for (int64_t j = 0; j < p; ++j) cross -= label_log_p(j) / static_cast<T>(p);
    return static_cast<double>(cross + penalty / T(2));
  }
  return static_cast<double>(cross / static_cast<T>(p) + penalty / T(2));
}

// Rows per task of the proximal kernels: their root solves cost a few microseconds per row
// even at ten classes, so small calls split too (about 16 microseconds per task).
inline int64_t prox_grain(int64_t c) { return std::max<int64_t>(1, 2048 / (c + 32)); }

// The dtype of the proximal kernels' gradients: float32 for 16-bit logits, so that the caller
// scales them by the upstream derivative before their one rounding.
inline at::ScalarType gradient_type(const at::Tensor& z) { return value_type(z); }

// The positives of a mask row first, then its negatives, each in index order; returns the
// number of positives.
inline int64_t mask_order(const bool* mask, int64_t c, int64_t* order) {
  int64_t p = 0, q = c;
  for (int64_t j = 0; j < c; ++j) {
    if (mask[j]) order[p++] = j;
  }
  for (int64_t j = c; j-- > 0;) {
    if (!mask[j]) order[--q] = j;
  }
  return p;
}

// proximal_cross_entropy per row: (values (N,), gradient (N, C) in gradient_type).
std::tuple<at::Tensor, at::Tensor> prox_entropy_forward(const at::Tensor& z,
                                                        const at::Tensor& target, double alpha,
                                                        bool fp32_work) {
  const int64_t n = z.size(0), c = z.size(1);
  auto grad_t = at::empty(z.sizes(), z.options().dtype(gradient_type(z)));
  const bool indexed = target.scalar_type() == at::kLong;
  const int64_t* idx_ptr = indexed ? target.data_ptr<int64_t>() : nullptr;
  const bool* mask_ptr = indexed ? nullptr : target.data_ptr<bool>();
  std::vector<double> values(n);
  // Recorded and raised after the loop, in the order of the other CPU kernels and the
  // native library (nonfinite logits first), whichever rows fail on whichever threads.
  RowErrors errors;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, z.scalar_type(), "sdca_prox_entropy_forward", [&] {
        using G = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
        const scalar_t* z_ptr = z.data_ptr<scalar_t>();
        G* g_ptr = grad_t.data_ptr<G>();
        const int64_t grain = prox_grain(c);
        at::parallel_for(0, n, grain, [&](int64_t begin, int64_t end) {
          std::vector<scalar_t> z_perm(c);
          std::vector<G> g_perm(c);
          std::vector<int64_t> order(c);
          std::vector<double> wd(fp32_work ? 0 : 4 * c);
          std::vector<float> wf(fp32_work ? 4 * c : 0);
          const auto solve = [&](int64_t p, G* g) {
            return fp32_work
                       ? prox_entropy_row<scalar_t, float>(c, p, z_perm.data(), alpha, wf.data(), g)
                       : prox_entropy_row<scalar_t, double>(c, p, z_perm.data(), alpha, wd.data(),
                                                            g);
          };
          for (int64_t row = begin; row < end; ++row) {
            const scalar_t* zr = z_ptr + row * c;
            G* gr = g_ptr + row * c;
            if (!all_finite(zr, c)) {
              errors.finite = true;
              continue;
            }
            if (indexed) {
              const int64_t y = idx_ptr[row];
              if (y < 0 || y >= c) {
                errors.target = true;
                continue;
              }
              // The label first: the row with its first entry and the label's swapped.
              std::copy(zr, zr + c, z_perm.data());
              std::swap(z_perm[0], z_perm[y]);
              values[row] = solve(1, gr);
              std::swap(gr[0], gr[y]);
            } else {
              const int64_t p = mask_order(mask_ptr + row * c, c, order.data());
              if (p < 1 || p >= c) {
                (p < 1 ? errors.positive : errors.negative) = true;
                continue;
              }
              for (int64_t j = 0; j < c; ++j) z_perm[j] = zr[order[j]];
              values[row] = solve(p, g_perm.data());
              for (int64_t j = 0; j < c; ++j) gr[order[j]] = g_perm[j];
            }
          }
        });
      });
  errors.raise();
  return {at::tensor(values, z.options().dtype(at::kDouble)).to(value_type(z)), grad_t};
}

}  // namespace

#endif  // SDCA_TORCH_PROX_H
