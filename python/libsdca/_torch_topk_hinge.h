#ifndef SDCA_TORCH_TOPK_HINGE_H
#define SDCA_TORCH_TOPK_HINGE_H

namespace {

// k = 1 hinge losses of integer targets, as include/sdca/loss.h in double:
// max(0, max_j h_j) of the margins h_j = 1 + z_j - z_y, or (smooth) the
// projection a = clip((h - origin) - correction, 0, gamma) onto sum a <= gamma.

// The compensated margin 1 + x - y of include/sdca/math/compensated.h.
inline double hinge_margin(double x, double y) {
  const double d = x - y;
  const double h = d + 1.0;
  if (!(std::abs(h) < 0.0625 * std::abs(d))) return h;
  const double rd = d - x;
  double low = (x - (d - rd)) + (-y - rd);
  const double rh = h - d;
  low += (d - (h - rh)) + (1.0 - rh);
  return std::isfinite(low) ? h + low : h;
}

// Margins of one row into h (-inf at the target); false for nonfinite logits.
template <typename scalar_t>
bool hinge_margins(const scalar_t* z, int64_t c, int64_t target, double* h) {
  using namespace dlanes;
  convert_to_double(z, h, c);
  const double label = h[target];
  // Plain margins (d + 1) on lanes; lanes that cancel take the compensated margin.
  vd bad{};
  int64_t j = 0;
  for (; j + L <= c; j += L) {
    const vd x = load(h + j), d = x - label, m = d + 1.0;
    bad = bad + x * 0.0;
    store(h + j, m);
    if (any_lane(abs(m) < abs(d) * 0.0625))
      for (int i = 0; i < L; ++i) h[j + i] = hinge_margin(x[i], label);
  }
  bool finite = true;
  for (int i = 0; i < L; ++i) finite &= bad[i] == 0.0;
  for (; j < c; ++j) {
    finite &= std::isfinite(h[j]);
    h[j] = hinge_margin(h[j], label);
  }
  h[target] = -std::numeric_limits<double>::infinity();
  return finite;
}

// Float lanes of L logits at p, 16-bit ones widened in registers.
inline lanes::vf load_float(const float* p) { return lanes::load(p); }
inline lanes::vf load_float(const at::BFloat16* p) {
  lanes::vh h;
  std::memcpy(&h, p, sizeof h);
  return (lanes::vf)(__builtin_convertvector(h, lanes::vu) << 16);
}
inline lanes::vf load_float(const at::Half* p) {
#if defined(SDCA_SIMD)
  lanes::vf v;
  for (int i = 0; i < lanes::L; i += 8) {
    const __m256 f = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + i)));
    std::memcpy(reinterpret_cast<float*>(&v) + i, &f, sizeof f);
  }
  return v;
#else
  float wide[lanes::L];
  to_float(p, wide, lanes::L);
  return lanes::load(wide);
#endif
}

// lanes::max (a > b ? a : b, b where either is NaN) in one instruction where the build has
// AVX2: GCC compiles the select to a compare and a blend.
inline lanes::vf max_lanes(lanes::vf a, lanes::vf b) {
#if defined(SDCA_SIMD)
  lanes::vf r;
  for (int i = 0; i < lanes::L; i += 8) {
    __m256 x, y;
    std::memcpy(&x, reinterpret_cast<const float*>(&a) + i, sizeof x);
    std::memcpy(&y, reinterpret_cast<const float*>(&b) + i, sizeof y);
    const __m256 m = _mm256_max_ps(x, y);
    std::memcpy(reinterpret_cast<float*>(&r) + i, &m, sizeof m);
  }
  return r;
#else
  return lanes::max(a, b);
#endif
}

// lanes::min in one instruction, as max_lanes.
inline lanes::vf min_lanes(lanes::vf a, lanes::vf b) {
#if defined(SDCA_SIMD)
  lanes::vf r;
  for (int i = 0; i < lanes::L; i += 8) {
    __m256 x, y;
    std::memcpy(&x, reinterpret_cast<const float*>(&a) + i, sizeof x);
    std::memcpy(&y, reinterpret_cast<const float*>(&b) + i, sizeof y);
    const __m256 m = _mm256_min_ps(x, y);
    std::memcpy(reinterpret_cast<float*>(&r) + i, &m, sizeof m);
  }
  return r;
#else
  return lanes::min(a, b);
#endif
}

// max_except of a float or 16-bit row read in place: four independent vectors of maxima
// per step, so no step waits on the previous one (named vectors: GCC keeps arrays of them
// on the stack). Lane sums of the entries flag nonfinite ones (also large finite rows whose
// sums overflow: those are checked entry by entry).
template <typename T>
float max_except_rows(const T* x, int64_t n, int64_t skip, bool& finite) {
  using namespace lanes;
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr int64_t kStep = 4 * L;
  vf t0 = splat(-kInf), t1 = t0, t2 = t0, t3 = t0;
  vf s0 = splat(0.0f), s1 = s0, s2 = s0, s3 = s0;
  auto lane = [&](vf& top, vf& sum, int64_t at) {
    vf v = load_float(x + at);
    sum = sum + v;
    if (skip >= at && skip < at + L)
      v = select(iota() == static_cast<int32_t>(skip - at), splat(-kInf), v);
    top = max_lanes(top, v);
  };
  int64_t j = 0;
  for (; j + kStep <= n; j += kStep)
    lane(t0, s0, j), lane(t1, s1, j + L), lane(t2, s2, j + 2 * L), lane(t3, s3, j + 3 * L);
  const vf all = max_lanes(max_lanes(t0, t1), max_lanes(t2, t3));
  const vf total = (s0 + s1) + (s2 + s3);
  float result = -kInf;
  bool sums = true;
  for (int i = 0; i < L; ++i) result = std::max(result, all[i]), sums &= std::isfinite(total[i]);
  for (int64_t i = 0; !sums && i < j; ++i) finite &= std::isfinite(static_cast<float>(x[i]));
  for (; j < n; ++j) {
    const float v = static_cast<float>(x[j]);
    finite &= std::isfinite(v);
    if (j != skip) result = std::max(result, v);
  }
  return result;
}

// find_at_least of a float or 16-bit row read in place.
template <typename T>
int64_t find_at_least_rows(const T* x, int64_t from, int64_t n, float floor) {
  int64_t j = from;
  for (; j + lanes::L <= n; j += lanes::L)
    if (any_lane(load_float(x + j) >= floor)) break;
  for (; j < n; ++j)
    if (static_cast<float>(x[j]) >= floor) return j;
  return n;
}

// The nonsmooth k = 1 hinge of a row from its largest competitor alone: value and statistics
// as hinge_row of hinge_margins (the margin 1 + z_j - z_y rises with z_j, so the largest is
// the top competitor's and its first index is the first competitor within rounding of the
// top whose margin equals it); false for nonfinite logits. Float and 16-bit rows are read in
// place.
template <typename scalar_t>
bool hinge_top(const scalar_t* z, int64_t c, int64_t target, double* stats, double& value) {
  auto scan = [&](const auto* x) {
    bool finite = true;
    double top;
    if constexpr (std::is_same_v<scalar_t, double>) {
      top = max_except(x, c, target, finite);
    } else {
      top = max_except_rows(x, c, target, finite);
    }
    if (!finite) return false;
    const double label = static_cast<double>(x[target]), h = hinge_margin(top, label);
    const double eps = std::numeric_limits<double>::epsilon();
    const double floor = top - 4 * eps * (std::abs(top) + std::abs(label) + 1);
    auto find = [&](int64_t from) {
      if constexpr (std::is_same_v<scalar_t, double>) {
        return find_at_least(x, from, c, floor);
      } else {
        return find_at_least_rows(x, from, c, static_cast<float>(floor));
      }
    };
    int64_t index = 0;
    for (int64_t j = find(0); j < c; j = find(j + 1)) {
      if (j != target && hinge_margin(static_cast<double>(x[j]), label) == h) {
        index = j;
        break;
      }
    }
    value = std::max(h, 0.0);
    stats[0] = static_cast<double>(index);
    stats[1] = value > 0;
    return true;
  };
  return scan(z);
}

// Value and the row statistics (first largest index and activity, or origin and correction).
double hinge_row(const double* h, int64_t c, bool smooth, double gamma, double* stats) {
  double top = -std::numeric_limits<double>::infinity();
  int64_t index = 0;
  for (int64_t j = 0; j < c; ++j)
    if (h[j] > top) top = h[j], index = j;
  if (!smooth) {
    const double value = std::max(top, 0.0);
    stats[0] = static_cast<double>(index);
    stats[1] = value > 0;
    return value;
  }
  double box = 0;
  for (int64_t j = 0; j < c; ++j) box += std::min(std::max(h[j], 0.0), gamma);
  double origin = 0, correction = 0;
  if (box > gamma) {
    // Michelot from the superset {g > -gamma} of the gaps g = h - top (see _triton_hinge).
    // The set {g > level} only shrinks: a rounded s cannot re-admit a gap and cycle.
    double s = -gamma, level = -gamma;
    int64_t count = -1;
    for (;;) {
      int64_t inside = 0;
      double sum = 0;
      for (int64_t j = 0; j < c; ++j) {
        const double g = h[j] - top;
        if (g > level) ++inside, sum += g;
      }
      if (inside == count) break;
      count = inside;
      s = (sum - gamma) / static_cast<double>(count);
      level = std::max(level, s);
    }
    origin = top, correction = s;
  }
  stats[0] = origin, stats[1] = correction;
  double value = 0;
  constexpr double tiny =
      std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon();
  for (int64_t j = 0; j < c; ++j) {
    const double a = std::min(std::max((h[j] - origin) - correction, 0.0), gamma);
    if (!(a > 0)) continue;
    const double ha = h[j] * a, aa = a * a / 2, v = std::abs(ha - aa);
    value += v <= std::numeric_limits<double>::max() && v >= tiny ? (ha - aa) / gamma
                                                                  : (h[j] - a / 2) * (a / gamma);
  }
  return value;
}

std::tuple<at::Tensor, at::Tensor> hinge_forward(const at::Tensor& z, const at::Tensor& y,
                                                 int64_t reduction, bool smooth, double gamma) {
  const int64_t n = z.size(0), c = z.size(1);
  auto stats = at::empty({n, 2}, z.options().dtype(at::kDouble));
  std::vector<double> values(n);
  RowErrors errors;
  double* s = stats.data_ptr<double>();
  const int64_t* labels = y.data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, z.scalar_type(), "sdca_hinge_forward", [&] {
        const scalar_t* data = z.data_ptr<scalar_t>();
        const int64_t grain = (n * c >= 16384) ? std::max<int64_t>(1, 4096 / c) : (n + 1);
        at::parallel_for(0, n, grain, [&](int64_t begin, int64_t end) {
          if (!smooth) {
            for (int64_t row = begin; row < end; ++row) {
              const scalar_t* z_row = data + row * c;
              checked_row(z_row, c, labels[row], errors, [&] {
                return hinge_top(z_row, c, labels[row], s + 2 * row, values[row]);
              });
            }
            return;
          }
          std::vector<double> h(c);
          for (int64_t row = begin; row < end; ++row) {
            const scalar_t* z_row = data + row * c;
            if (checked_row(z_row, c, labels[row], errors,
                            [&] { return hinge_margins(z_row, c, labels[row], h.data()); }))
              values[row] = hinge_row(h.data(), c, smooth, gamma, s + 2 * row);
          }
        });
      });
  errors.raise();
  return {reduced_losses(values, reduction, z), stats};
}

at::Tensor hinge_backward(const at::Tensor& z, const at::Tensor& y, const at::Tensor& stats,
                          const at::Tensor& upstream, int64_t reduction, bool smooth,
                          double gamma) {
  const int64_t n = z.size(0), c = z.size(1);
  auto output = gradient_like(z);
  const std::vector<double> u = double_values(upstream);
  const double* weights = u.data();
  const double* s = stats.data_ptr<double>();
  const int64_t* labels = y.data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, z.scalar_type(), "sdca_hinge_backward", [&] {
        const scalar_t* input = z.data_ptr<scalar_t>();
        scalar_t* result = output.data_ptr<scalar_t>();
        const int64_t grain = (n * c >= 16384) ? std::max<int64_t>(1, 4096 / c) : (n + 1);
        at::parallel_for(0, n, grain, [&](int64_t begin, int64_t end) {
          std::vector<double> h(smooth ? c : 0);
          for (int64_t row = begin; row < end; ++row) {
            double weight = weights[reduction == 0 ? row : 0];
            if (reduction == 1) weight /= n;
            scalar_t* g = result + row * c;
            const double* st = s + 2 * row;
            if (!smooth) {
              std::fill(g, g + c, scalar_t(0));
              const double w = weight * st[1];
              g[static_cast<int64_t>(st[0])] = static_cast<scalar_t>(w);
              g[labels[row]] = static_cast<scalar_t>(-w);
              continue;
            }
            hinge_margins(input + row * c, c, labels[row], h.data());
            double total = 0;
            for (int64_t j = 0; j < c; ++j) {
              const double a = std::min(std::max((h[j] - st[0]) - st[1], 0.0), gamma);
              const double x = (a / gamma) * weight;
              g[j] = static_cast<scalar_t>(x);
              if (j != labels[row]) total += x;
            }
            // The target entry: minus the competitors' unrounded sum, rounded once.
            g[labels[row]] = static_cast<scalar_t>(-total);
          }
        });
      });
  return output;
}

inline double softplus_d(double x) { return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x))); }

inline double topk_projected(double h, double first, double second, double gamma, int64_t k) {
  const bool cone = second > 0.0;
  const double shifted = cone ? (h - first) : ((h - first) - second);
  const double cap = cone ? second : (gamma / static_cast<double>(k));
  return std::min(std::max(shifted, 0.0), cap);
}

inline void topk_knapsack(const double* h, int64_t c, const double* top, double cap, double rhs,
                          int64_t k, double& origin, double& correction) {
  int64_t lo = 0, hi = k;
  while (lo < hi) {
    const int64_t mid = (lo + hi + 1) / 2;
    const double level = top[mid - 1];
    double sum = 0.0;
    for (int64_t j = 0; j < c; ++j) {
      if (std::isfinite(h[j])) sum += std::min(std::max((h[j] - level) + cap, 0.0), cap);
    }
    if (sum <= rhs)
      lo = mid;
    else
      hi = mid - 1;
  }
  origin = top[std::min(lo, k - 1)];
  if (lo == k) {
    correction = -cap;
    return;
  }
  const double rem = rhs - cap * static_cast<double>(lo);
  int64_t count = 0;
  double sum = 0.0;
  for (int64_t j = 0; j < c; ++j) {
    if (std::isfinite(h[j]) && h[j] <= origin) {
      const double g = h[j] - origin;
      if (g > -cap) {
        ++count;
        sum += g;
      }
    }
  }
  double s = (sum - rem) / static_cast<double>(count);
  // The set {g > level} only shrinks, so rounding cannot cycle (see hinge_row).
  double level = std::max(-cap, s);
  int64_t previous = count + 1;
  while (count != previous) {
    previous = count;
    count = 0;
    sum = 0.0;
    for (int64_t j = 0; j < c; ++j) {
      if (std::isfinite(h[j]) && h[j] <= origin) {
        const double g = h[j] - origin;
        if (g > level) {
          ++count;
          sum += g;
        }
      }
    }
    s = (sum - rem) / static_cast<double>(count);
    level = std::max(level, s);
  }
  correction = s;
}

inline void topk_cone(const double* h, int64_t c, const double* top, int64_t k, double& t_out,
                      double& c_out) {
  double whole = 0.0, low = 0.0, scale = 0.0;
  for (int64_t i = 0; i < k; ++i) {
    const double x = top[i];
    const double total = whole + x;
    const double rounded = total - whole;
    low += (whole - (total - rounded)) + (x - rounded);
    whole = total;
    scale += std::abs(x);
  }
  whole += low;
  const double tol = 16.0 * std::numeric_limits<double>::epsilon() * std::min(scale, 1.0);
  const double kd = static_cast<double>(k);
  double cap = whole / kd;
  double t = top[k - 1] - cap;
  auto above = [&](double level) {
    double s = 0.0;
    for (int64_t j = 0; j < c; ++j) {
      if (std::isfinite(h[j]) && h[j] > level) s += h[j] - level;
    }
    return s;
  };
  if (t < top[k] - tol && whole > tol) {
    int64_t lo = -1, hi = k - 1;
    while (lo < hi) {
      const int64_t mid = lo + (hi - lo + 1) / 2;
      const double level = top[mid];
      const double shift = -above(level) / kd;
      if (above(shift) / kd + 2.0 * shift < level)
        lo = mid;
      else
        hi = mid - 1;
    }
    const int64_t u = lo + 1;
    if (u < k) {
      double head = 0.0;
      for (int64_t i = 0; i < u; ++i) head += top[i];
      const double bound = top[u];
      int64_t m = 0;
      double rest = 0.0;
      for (int64_t j = 0; j < c; ++j) {
        if (std::isfinite(h[j]) && h[j] <= bound) {
          ++m;
          rest += h[j];
        }
      }
      const double ku = static_cast<double>(k - u), ud = static_cast<double>(u);
      double d = ku * ku + static_cast<double>(m) * ud;
      t = (ud * rest - ku * head) / d;
      // The set {h > level} only shrinks, so rounding cannot cycle (see hinge_row).
      double level = t;
      int64_t previous = m + 1;
      while (m != previous) {
        previous = m;
        m = 0;
        rest = 0.0;
        for (int64_t j = 0; j < c; ++j) {
          if (std::isfinite(h[j]) && h[j] <= bound && h[j] > level) {
            ++m;
            rest += h[j];
          }
        }
        d = ku * ku + static_cast<double>(m) * ud;
        t = (ud * rest - ku * head) / d;
        level = std::max(level, t);
      }
      cap = (static_cast<double>(m) * head + ku * rest) / d;
    }
  }
  if (whole <= tol) {
    t_out = 0.0;
    c_out = 0.0;
  } else {
    t_out = t;
    c_out = cap;
  }
}

// Compensated sum of exp(v[j] - center) on double lanes; exponentials of entries that are -inf
// (or below the normal range) are zero.
inline double sum_exp_shifted_simd(const double* v, int64_t n, double center, double min_val) {
  if (min_val - center >= -708.0) {
    dlanes::Accumulator<true> acc;
    int64_t j = 0;
    for (; j + dlanes::L <= n; j += dlanes::L) acc.add(dlanes::exp(dlanes::load(v + j) - center));
    Sum<true> tail;
    tail.add(acc.value());
    for (; j < n; ++j) {
      if (std::isfinite(v[j])) tail.add(std::exp(v[j] - center));
    }
    return tail.value();
  }
  Sum<true> sum;
  for (int64_t j = 0; j < n; ++j) {
    if (std::isfinite(v[j])) sum.add(std::exp(v[j] - center));
  }
  return sum.value();
}

// out[j] = exp(v[j] - shift), with the inlined exponential where every result is normal.
inline void exp_shifted(const double* v, int64_t n, double shift, double minimum, double* out) {
#ifdef SDCA_SIMD
  if (minimum - shift >= -708.0) {
    const simd::vd vs = simd::set1(shift);
    int64_t j = 0;
    for (; j + simd::W <= n; j += simd::W)
      simd::store(out + j, exp_pd_normal(simd::sub(simd::load(v + j), vs)));
    for (; j < n; ++j) out[j] = std::exp(v[j] - shift);
    return;
  }
#else
  if (minimum - shift >= -708.0) {
    int64_t j = 0;
    for (; j + dlanes::L <= n; j += dlanes::L)
      dlanes::store(out + j, dlanes::exp(dlanes::load(v + j) - shift));
    for (; j < n; ++j) out[j] = std::exp(v[j] - shift);
    return;
  }
#endif
  at::vec::map([shift](Vec x) { return (x - Vec(shift)).exp(); }, out, v, n);
}

// Top-k entropy (k < competitors) of a row from its largest competitor, the label score and
// others = sum exp(z_j - maximum) over the other competitors. Where the largest one's cap
// binds, tail() puts the k largest competitors into top in descending order and returns the
// sum of exp(z_j - top[k - 1]) over the rest. Returns the value; stats (free score, the free
// competitors' sum of exp(z_j - free) per uncapped slot, mass) as topk_backward reads them.
template <typename F>
double topk_entropy_value(double maximum, double label, double others, int64_t k, double* stats,
                          F&& tail_of, const double* top) {
  const double kd = static_cast<double>(k), log_k = std::log(kd);
  const double eps = 32.0 * std::numeric_limits<double>::epsilon();
  const double sum = 1.0 + others;
  if (!(std::log(sum) - log_k + eps * std::max(std::abs(std::log(sum) - log_k), 1.0) < 0.0)) {
    const double gap = maximum - label, relative = gap + std::log(sum);
    double loss, mass;
    if (relative < -36.0) {
      const double half = std::exp(gap * 0.5), tail = half * (half * sum);
      loss = std::log1p(tail);
      mass = tail / (1.0 + tail);
    } else {
      loss = softplus_d(relative);
      mass = std::exp(-softplus_d(-relative));
    }
    stats[0] = maximum;
    stats[1] = sum / kd;
    stats[2] = mass;
    return loss;
  }
  double lse = std::log1p(others), total = sum;
  double threshold = lse - log_k;
  int64_t active = 0;
  Sum<true> head;
  double free_val = maximum;
  if (threshold + eps * std::max(std::abs(threshold), 1.0) < 0.0) {
    const double tail = tail_of();
    const double last = top[k - 1];
    // The spread of the free competitors when the first m are capped.
    auto spread_at = [&](int64_t m) {
      // Below the k largest every exponential underflows where scale does.
      const double scale = std::exp(last - top[m]);
      Sum<true> mid;
      for (int64_t j = m + 1; j < k; ++j) mid.add(std::exp(top[j] - top[m]));
      return mid.value() + scale * tail;
    };
    // The first m in [1, k - 1) not capped, else k - 1, by probing m = 1, 2, 4, ... and
    // bisecting the bracket (as loss.h): being capped is monotone in m. Every m < lo is
    // capped; hi is not (found its spread), or hi = k - 1.
    int64_t lo = 1, hi = k - 1;
    double found = 0.0, found_lse = 0.0;
    bool have = false;
    auto capped = [&](int64_t m) {
      const double s = spread_at(m), l = std::log1p(s);
      const double t = l - std::log(static_cast<double>(k - m));
      if (t + eps * std::max(std::abs(t), 1.0) < 0.0) return true;
      found = s;
      found_lse = l;
      have = true;
      return false;
    };
    for (int64_t probe = 1; probe < hi && !have; probe *= 2) {
      if (capped(probe)) {
        lo = probe + 1;
      } else {
        hi = probe;
      }
    }
    while (lo < hi) {
      const int64_t m = lo + (hi - lo) / 2;
      if (capped(m)) {
        lo = m + 1;
      } else {
        hi = m;
      }
    }
    active = lo;
    free_val = top[active];
    const double spread = have ? found : spread_at(active);
    lse = have ? found_lse : std::log1p(spread);
    total = 1.0 + spread;
    for (int64_t i = 1; i < active; ++i) head.add(top[i] - maximum);
  }
  const double remaining = static_cast<double>(k - active);
  const double rest = remaining * ((free_val - maximum) + lse - std::log(remaining));
  const double mean_term = (head.value() + rest) / kd;
  const double odds = (maximum - label) + mean_term + log_k;
  const double logs = -softplus_d(-odds);
  stats[0] = free_val;
  stats[1] = total / remaining;
  if (odds < -36.0) {
    // A positive gap (a capped competitor far above the free ones) joins the mean.
    const double gap = maximum - label, above = std::max(gap, 0.0);
    const double half = std::exp((gap - above) * 0.5);
    const double tail = half * (half * std::exp(mean_term + above) * kd);
    stats[2] = tail / (1.0 + tail);
    return std::log1p(tail);
  }
  stats[2] = std::exp(logs);
  return softplus_d(odds);
}

// The k largest of [first, last) in order at the front: partial_sort for a small k, else
// nth_element and a sort of the k, O(n + k log k) instead of a heap's O(n log k). The rest is
// left in an unspecified order, so sums over it can differ in the last bits from another
// selection's.
inline void sort_largest(double* first, int64_t k, double* last) {
  if (k <= 16) {
    std::partial_sort(first, first + k, last, std::greater<double>());
    return;
  }
  std::nth_element(first, first + (k - 1), last, std::greater<double>());
  std::sort(first, first + k, std::greater<double>());
}

template <typename scalar_t>
double topk_entropy_row(const scalar_t* z, int64_t c, int64_t target, int64_t k, double* v,
                        double* stats) {
  convert_to_double(z, v, c);
  const double label = v[target];
  // Extrema of the competitors: the target takes a neighbour's value meanwhile.
  v[target] = v[target == 0 ? 1 : 0];
  // Named locals, not a structured binding: the lambda below captures minimum (C++17 and
  // clang 14 cannot capture bindings).
  const std::pair<double, double> range = extrema(v, c);
  const double top = range.first, minimum = range.second;
  v[target] = label;
  if (!(std::isfinite(label) && std::isfinite(top) && std::isfinite(minimum)))
    return std::numeric_limits<double>::quiet_NaN();
  const double kd = static_cast<double>(k);
  const double log_k = std::log(kd);
  if (k == c - 1) {
    // As entropy_mean_offset: where long double is double, the sums of finite scores
    // can overflow; an exact power-of-two scale keeps finite means finite.
    auto scaled_mean = [&](long double scale) {
      long double sum = 0.0L, comp = 0.0L;
      const long double off = -static_cast<long double>(label) * scale;
      for (int64_t j = 0; j < c; ++j) {
        if (j == target) continue;
        const long double x = static_cast<long double>(v[j]) * scale;
        const long double term = x + off, dv = term - x;
        const long double err = (x - (term - dv)) + (off - dv);
        const long double next = sum + term;
        comp += (std::abs(sum) >= std::abs(term) ? (sum - next) + term : (term - next) + sum) + err;
        sum = next;
      }
      return (sum + comp) / static_cast<long double>(k) / scale;
    };
    long double wide_mean = scaled_mean(1.0L);
    if (!std::isfinite(wide_mean))
      wide_mean = scaled_mean(std::ldexp(1.0L, -(std::ilogb(static_cast<long double>(k)) + 2)));
    const double mean = static_cast<double>(wide_mean);
    const double odds = mean + log_k;
    const double logs = -softplus_d(-odds);
    stats[0] = std::numeric_limits<double>::infinity();
    stats[1] = 0.0;
    if (odds < -36.0) {
      const double half = std::exp(mean * 0.5);
      const double tail = half * (half * kd);
      stats[2] = tail / (1.0 + tail);
      return std::log1p(tail);
    }
    stats[2] = std::exp(logs);
    return softplus_d(odds);
  }
  v[target] = -std::numeric_limits<double>::infinity();
  const int64_t first_idx = std::find(v, v + c, top) - v;
  v[first_idx] = -std::numeric_limits<double>::infinity();
  const double others = sum_exp_shifted_simd(v, c, top, minimum);
  v[first_idx] = top;
  return topk_entropy_value(
      top, label, others, k, stats,
      [&] {
        sort_largest(v, k, v + c);
        return sum_exp_shifted_simd(v + k, c - k, v[k - 1], minimum);
      },
      v);
}

// Top-k entropy rows in float32 exponentials take k up to this; larger k the double rows.
constexpr int64_t kFloatTopK = 64;

// topk_entropy_row (k < c - 1, k <= kFloatTopK) in one sweep of float32 exponentials over a
// non-double row, with float64 row statistics: the k largest competitors so far in top, and
// the exponentials of the others from the k-th largest (rescaled in double as it rises). A
// vector's competitors above the k-th largest enter top; its others and the entries they
// leave out add their exponentials. buffer holds kFloatBlock floats; NaN for nonfinite
// logits.
template <typename scalar_t>
double topk_entropy_row_float(const scalar_t* src, int64_t c, int64_t target, int64_t k,
                              float* buffer, double* stats) {
  using namespace lanes;
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float top[kFloatTopK];
  std::fill(top, top + k, -kInf);
  float last = -kInf;  // top[k - 1] as of the sums
  // sum exp(z_j - last) over the competitors out of top: tail, and sums of the vectors
  // added since last rose.
  double tail = 0.0;
  Accumulator sums;
  // Puts x into top where it beats top[k - 1]; returns the entry left out (x or that one).
  auto insert = [&](float x) {
    if (!(x > top[k - 1])) return x;
    const float out = top[k - 1];
    int64_t p = k - 1;
    for (; p > 0 && x > top[p - 1]; --p) top[p] = top[p - 1];
    top[p] = x;
    return out;
  };
  // After insertions: the sums from the new k-th largest.
  auto rise = [&] {
    if (top[k - 1] == last) return;
    // While top fills, last is -inf and the sums are empty.
    tail = (tail + sums.value()) * std::exp(static_cast<double>(last) - top[k - 1]);
    sums = Accumulator();
    last = top[k - 1];
  };
  vf bad = splat(0.0f);
  bool finite = true;
  for (int64_t b = 0; b < c; b += kFloatBlock) {
    const int64_t n = std::min(kFloatBlock, c - b);
    const float* x = float_row(src + b, n, buffer);
    const int64_t skip = target - b;  // outside [0, n) where the label is elsewhere
    int64_t j = 0;
    for (; j + L <= n; j += L) {
      vf v = load(x + j);
      bad = bad + v * 0.0f;  // NaN for inf and NaN entries
      if (skip >= j && skip < j + L)
        v = select(iota() == static_cast<int32_t>(skip - j), splat(-kInf), v);
      const vi above = v > splat(last);
      if (any_lane(above)) {
        // Each lane above enters top; the entry left out takes its place.
        for (int i = 0; i < L; ++i)
          if (above[i]) v[i] = insert(v[i]);
        rise();
      }
      // Only the label, entries left out of a filling top (and nonfinite logits) can be
      // -inf while last is.
      if (last > -kInf) sums.add(exp(v - last));
    }
    for (; j < n; ++j) {
      finite &= std::isfinite(x[j]);
      if (j == skip) continue;
      const float out = insert(x[j]);
      rise();
      if (out > -kInf) tail += std::exp(static_cast<double>(out) - last);
    }
  }
  for (int i = 0; i < L; ++i) finite &= bad[i] == 0.0f;
  if (!finite) return std::numeric_limits<double>::quiet_NaN();
  tail += sums.value();
  double wide[kFloatTopK];
  std::copy(top, top + k, wide);
  // The competitors but the largest, from it.
  double others = tail * std::exp(wide[k - 1] - wide[0]);
  for (int64_t i = 1; i < k; ++i) others += std::exp(wide[i] - wide[0]);
  return topk_entropy_value(
      wide[0], static_cast<float>(src[target]), others, k, stats, [&] { return tail; }, wide);
}

inline void find_top_k(const double* h, int64_t c, int64_t k_need, double* top, double* scratch) {
  if (k_need <= 64) {
    std::fill(top, top + k_need, -std::numeric_limits<double>::infinity());
    double min_top = -std::numeric_limits<double>::infinity();
    for (int64_t j = 0; j < c; ++j) {
      // Skip whole vectors at or below the k-th largest so far.
      if ((j & (dlanes::L - 1)) == 0 && j + dlanes::L <= c &&
          !any_lane(dlanes::load(h + j) > min_top)) {
        j += dlanes::L - 1;
        continue;
      }
      const double v = h[j];
      if (v > min_top) {
        int64_t pos = k_need - 1;
        while (pos > 0 && v > top[pos - 1]) {
          top[pos] = top[pos - 1];
          --pos;
        }
        top[pos] = v;
        min_top = top[k_need - 1];
      }
    }
    return;
  }
  std::copy(h, h + c, scratch);
  sort_largest(scratch, k_need, scratch + c);
}

double topk_hinge_row(const double* h, int64_t c, int64_t loss_code, int64_t k, double gamma,
                      double* scratch, double* stats) {
  const double kd = static_cast<double>(k);
  double local_top[65];
  double* top = (k + 1 <= 64) ? local_top : scratch;
  if (loss_code <= 2) {
    find_top_k(h, c, k, top, scratch);
    double value = 0.0;
    for (int64_t i = 0; i < k; ++i) {
      const double x = loss_code == 2 ? std::max(top[i], 0.0) : top[i];
      value += x / kd;
    }
    value = std::max(value, 0.0);
    const double last = top[k - 1];
    int64_t need = 0;
    for (int64_t i = 0; i < k; ++i) {
      if (top[i] == last) ++need;
    }
    int64_t cut = -1;
    for (int64_t j = 0; j < c; ++j) {
      if (h[j] == last) {
        cut = j;
        if (--need == 0) break;
      }
    }
    stats[0] = last;
    stats[1] = static_cast<double>(cut);
    return value;
  }
  const double cap = gamma / kd;
  double origin = 0.0, correction = 0.0;
  if (loss_code == 3) {
    const int64_t k1 = std::min(c, k + 1);
    find_top_k(h, c, k1, top, scratch);
    double t = 0.0, cap_c = 0.0;
    topk_cone(h, c, top, k, t, cap_c);
    if (kd * cap_c > gamma) {
      topk_knapsack(h, c, top, cap, gamma, k, origin, correction);
    } else {
      origin = cap_c > 0.0 ? t : std::numeric_limits<double>::infinity();
      correction = cap_c;
    }
  } else {
    double box = 0.0;
    for (int64_t j = 0; j < c; ++j) {
      if (std::isfinite(h[j])) box += std::min(std::max(h[j], 0.0), cap);
    }
    if (box > gamma) {
      find_top_k(h, c, k, top, scratch);
      topk_knapsack(h, c, top, cap, gamma, k, origin, correction);
    }
  }
  stats[0] = origin;
  stats[1] = correction;
  double value = 0.0;
  constexpr double tiny =
      std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon();
  for (int64_t j = 0; j < c; ++j) {
    if (!std::isfinite(h[j])) continue;
    const double a = topk_projected(h[j], origin, correction, gamma, k);
    if (!(a > 0.0)) continue;
    const double ha = h[j] * a, aa = a * a / 2.0, v = std::abs(ha - aa);
    value += v <= std::numeric_limits<double>::max() && v >= tiny ? (ha - aa) / gamma
                                                                  : (h[j] - a / 2.0) * (a / gamma);
  }
  return value;
}

// The nonsmooth top-k hinges (loss_code 1, 2; k <= 64) of a float or 16-bit row read in
// place, with value and stats as topk_hinge_row of hinge_margins: margins rise with the
// score, so the k largest competitor scores (one vector scan) give the k largest margins;
// the index cut of the last tie then comes from the scores within rounding of the k-th
// margin (as topk_hinge_chosen). false for nonfinite logits.
template <typename T>
bool topk_hinge_floats(const T* x, int64_t c, int64_t target, int64_t loss_code, int64_t k,
                       double* stats, double& value) {
  using namespace lanes;
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float top[64];
  std::fill(top, top + k, -kInf);
  float least = -kInf;
  auto insert = [&](float v) {
    if (!(v > least)) return;
    int64_t pos = k - 1;
    for (; pos > 0 && v > top[pos - 1]; --pos) top[pos] = top[pos - 1];
    top[pos] = v;
    least = top[k - 1];
  };
  vf bad = splat(0.0f);
  int64_t j = 0;
  for (; j + L <= c; j += L) {
    vf v = load_float(x + j);
    bad = bad + v * 0.0f;
    if (target >= j && target < j + L)
      v = select(iota() == static_cast<int32_t>(target - j), splat(-kInf), v);
    if (any_lane(v > least))
      for (int i = 0; i < L; ++i) insert(v[i]);
  }
  bool finite = true;
  for (int i = 0; i < L; ++i) finite &= bad[i] == 0.0f;
  for (; j < c; ++j) {
    const float v = static_cast<float>(x[j]);
    finite &= std::isfinite(v);
    if (j != target) insert(v);
  }
  if (!finite) return false;
  const double label = static_cast<double>(x[target]), kd = static_cast<double>(k);
  double margins[64], total = 0.0;
  for (int64_t i = 0; i < k; ++i) {
    margins[i] = hinge_margin(top[i], label);
    total += (loss_code == 2 ? std::max(margins[i], 0.0) : margins[i]) / kd;
  }
  value = std::max(total, 0.0);
  const double last = margins[k - 1], eps = std::numeric_limits<double>::epsilon();
  int64_t need = 0;
  for (int64_t i = 0; i < k; ++i) need += margins[i] == last;
  const float floor =
      static_cast<float>(((last - 1.0) + label) - 8 * eps * (std::abs(last) + std::abs(label) + 2));
  int64_t cut = -1;
  for (j = find_at_least_rows(x, 0, c, floor); j < c; j = find_at_least_rows(x, j + 1, c, floor)) {
    if (j != target && hinge_margin(static_cast<double>(x[j]), label) == last) {
      cut = j;
      if (--need == 0) break;
    }
  }
  stats[0] = last;
  stats[1] = static_cast<double>(cut);
  return true;
}
template <typename T>
bool topk_hinge_top(const T* x, int64_t c, int64_t target, int64_t loss_code, int64_t k,
                    double* stats, double& value) {
  if constexpr (std::is_same_v<T, double>) {
    throw std::logic_error("float64 rows take hinge_margins");
  } else {
    return topk_hinge_floats(x, c, target, loss_code, k, stats, value);
  }
}

// fp32: top-k entropy rows of non-double logits in float32 exponentials (k <= kFloatTopK).
std::tuple<at::Tensor, at::Tensor> topk_forward(const at::Tensor& z, const at::Tensor& y,
                                                int64_t reduction, int64_t loss_code, int64_t k,
                                                double gamma, bool fp32) {
  const int64_t n = z.size(0), c = z.size(1);
  const bool work32 =
      fp32 && loss_code == 0 && z.scalar_type() != at::kDouble && k <= kFloatTopK && k < c - 1;
  auto stats = at::empty({n, 3}, z.options().dtype(at::kDouble));
  std::vector<double> values(n);
  RowErrors errors;
  double* s = stats.data_ptr<double>();
  const int64_t* labels = y.data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, z.scalar_type(), "sdca_topk_forward", [&] {
        const scalar_t* data = z.data_ptr<scalar_t>();
        const int64_t grain = (n * c >= 16384) ? std::max<int64_t>(1, 4096 / c) : (n + 1);
        at::parallel_for(0, n, grain, [&](int64_t begin, int64_t end) {
          std::vector<double> h(work32 ? 0 : c), scratch(loss_code == 0 ? 0 : c);
          std::vector<float> narrow(work32 ? kFloatBlock : 0);
          for (int64_t row = begin; row < end; ++row) {
            double* st = s + 3 * row;
            const scalar_t* z_row = data + row * c;
            const int64_t target = labels[row];
            if (loss_code == 0) {
              checked_row(z_row, c, target, errors, [&] {
                values[row] = work32
                                  ? topk_entropy_row_float(z_row, c, target, k, narrow.data(), st)
                                  : topk_entropy_row(z_row, c, target, k, h.data(), st);
                return !std::isnan(values[row]);
              });
            } else if (!std::is_same_v<scalar_t, double> && loss_code <= 2 && k <= 64) {
              if (checked_row(z_row, c, target, errors, [&] {
                    return topk_hinge_top(z_row, c, target, loss_code, k, st, values[row]);
                  }))
                st[2] = values[row];
            } else if (checked_row(z_row, c, target, errors,
                                   [&] { return hinge_margins(z_row, c, target, h.data()); })) {
              values[row] = topk_hinge_row(h.data(), c, loss_code, k, gamma, scratch.data(), st);
              st[2] = values[row];
            }
          }
        });
      });
  errors.raise();
  return {reduced_losses(values, reduction, z), stats};
}

// The nonsmooth top-k hinge gradient of a row (zeros, then g_val at its chosen competitors:
// margins above the threshold first, or equal to it up to index cut, and positive if
// positive) from its logits x alone: margins rise with x, so only entries within rounding of
// the threshold's score need theirs. Returns the competitors' unrounded sum.
template <typename T, typename scalar_t>
double topk_hinge_chosen(const T* x, int64_t c, int64_t target, double first, int64_t cut,
                         bool positive, scalar_t g_val, double step, scalar_t* g) {
  std::fill(g, g + c, scalar_t(0));
  const double label = x[target], eps = std::numeric_limits<double>::epsilon();
  const double floor = ((first - 1.0) + label) - 8 * eps * (std::abs(first) + std::abs(label) + 2);
  double total = 0.0;
  for (int64_t j = find_at_least(x, 0, c, floor); j < c; j = find_at_least(x, j + 1, c, floor)) {
    if (j == target) continue;
    const double h = hinge_margin(x[j], label);
    bool chosen = h > first || (h == first && j <= cut);
    if (positive) chosen = chosen && h > 0.0;
    if (chosen) g[j] = g_val, total += step;
  }
  return total;
}

// The top-k entropy competitors' entries cap * min(1, exp(min(h_j - first, log D)) / D) of a
// double row in place, arguments at least -708: one product with cap / D where that is normal
// (the exponent's bound keeps the ratio at most 1), else with 1 / D and then cap.
inline void topk_entropy_entries(double* h, int64_t c, double first, double spread, double cap) {
  const double bound = std::log(spread), inverse = 1.0 / spread, factor = cap * inverse;
  const dlanes::vd top = dlanes::splat(bound);
  int64_t j = 0;
  if (factor >= std::numeric_limits<double>::min()) {
    const dlanes::vd most = dlanes::splat(cap);
    for (; j + dlanes::L <= c; j += dlanes::L) {
      const dlanes::vd e = dlanes::exp(dlanes::min(dlanes::load(h + j) - first, top));
      dlanes::store(h + j, dlanes::min(e * factor, most));
    }
    for (; j < c; ++j) h[j] = std::min(std::exp(std::min(h[j] - first, bound)) * factor, cap);
    return;
  }
  const dlanes::vd one = dlanes::splat(1.0);
  for (; j + dlanes::L <= c; j += dlanes::L) {
    const dlanes::vd e = dlanes::exp(dlanes::min(dlanes::load(h + j) - first, top));
    dlanes::store(h + j, dlanes::min(e * inverse, one) * cap);
  }
  for (; j < c; ++j) h[j] = std::min(std::exp(std::min(h[j] - first, bound)) * inverse, 1.0) * cap;
}

// The top-k entropy gradient of a non-double row in float32 exponentials (first, mass and
// weight as topk_backward's, shift = log(cap / D)): the competitors' sign * (exp(min(x -
// first, log(cap) - shift) / 2) * root)^2, root = sqrt(|weight| exp(shift)), and the target
// entry from the row statistics, -weight mass (the competitors' mass: a sum of their float32
// entries would lose the terms below float32's normal range). False where root exceeds
// float32.
template <typename scalar_t>
bool topk_gradient_float(const scalar_t* src, int64_t c, int64_t target, int64_t k, double first,
                         double shift, double mass, double weight, float* buffer, scalar_t* out) {
  using namespace lanes;
  const double root = std::exp(shift / 2) * std::sqrt(std::abs(weight));
  if (!(root <= std::numeric_limits<float>::max())) return false;
  const double cap = mass / static_cast<double>(k);
  const float* x = float_row(src, c, buffer);
  float* g = float_output(out, buffer);
  const float center = static_cast<float>(first), r = static_cast<float>(root);
  const vf top = splat(static_cast<float>(std::log(cap) - shift));
  each(x, c, -std::numeric_limits<float>::infinity(), [&](vf v, int64_t j, int count) {
    const vf half = exp(min(v - center, top) * 0.5f) * r;
    vf e = half * half;
    if (weight < 0) e = -e;
    if (count == L) {
      store(g + j, e);
    } else {
      for (int i = 0; i < count; ++i) g[j + i] = e[i];
    }
  });
  if constexpr (!std::is_same_v<scalar_t, float>) from_float(g, out, c);
  out[target] = static_cast<scalar_t>(-weight * mass);
  return true;
}

// fp32: top-k entropy rows of non-double logits in float32 exponentials.
at::Tensor topk_backward(const at::Tensor& z, const at::Tensor& y, const at::Tensor& stats,
                         const at::Tensor& upstream, int64_t reduction, int64_t loss_code,
                         int64_t k, double gamma, bool fp32) {
  const int64_t n = z.size(0), c = z.size(1);
  const bool work32 = fp32 && z.scalar_type() != at::kDouble;
  auto output = gradient_like(z);
  const std::vector<double> u = double_values(upstream);
  const double* weights = u.data();
  const double* s = stats.data_ptr<double>();
  const int64_t* labels = y.data_ptr<int64_t>();
  const double kd = static_cast<double>(k);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, z.scalar_type(), "sdca_topk_backward", [&] {
        const scalar_t* input = z.data_ptr<scalar_t>();
        scalar_t* result = output.data_ptr<scalar_t>();
        const int64_t grain = (n * c >= 16384) ? std::max<int64_t>(1, 4096 / c) : (n + 1);
        at::parallel_for(0, n, grain, [&](int64_t begin, int64_t end) {
          std::vector<double> h(c);
          std::vector<float> narrow(loss_code <= 2 && !std::is_same_v<scalar_t, float> ? c : 0);
          for (int64_t row = begin; row < end; ++row) {
            double weight = weights[reduction == 0 ? row : 0];
            if (reduction == 1) weight /= n;
            scalar_t* g = result + row * c;
            const double* st = s + 3 * row;
            const int64_t target = labels[row];
            const double first = st[0], second = st[1], row_val = st[2];
            if (loss_code == 0) {
              const double mass = row_val, cap = mass / kd;
              if (!(mass > 0.0)) {
                std::fill(g, g + c, scalar_t(0));
                continue;
              }
              if (std::isinf(first) && first > 0.0) {
                const scalar_t g_val = static_cast<scalar_t>(cap * weight);
                std::fill(g, g + c, g_val);
                g[target] = static_cast<scalar_t>(-weight * mass);
                continue;
              }
              // second = D, the free competitors' sum of exp(z_j - first) per uncapped slot:
              // each one cap * exp(z_j - first) / D, at most cap; a log-domain shift would
              // round at the magnitude of log(mass).
              const double spread = second, bound = std::log(spread), inverse = 1.0 / spread;
              if (work32 &&
                  topk_gradient_float(input + row * c, c, target, k, first, std::log(cap) - bound,
                                      mass, weight, narrow.data(), g))
                continue;
              convert_to_double(input + row * c, h.data(), c);
              // The smallest argument, monotone in h: from the competitors' minimum.
              const double saved = h[target];
              h[target] = h[target == 0 ? 1 : 0];
              const double min_arg = extrema(h.data(), c).second - first;
              h[target] = saved;
              if (min_arg >= -708.0) {
                topk_entropy_entries(h.data(), c, first, spread, cap);
                h[target] = 0.0;
                scale_and_store(h.data(), g, c, weight);
                g[target] = static_cast<scalar_t>(-weight * mass);
                continue;
              }
              for (int64_t j = 0; j < c; ++j) {
                if (j == target) continue;
                const double e = std::exp(std::min(h[j] - first, bound));
                g[j] = static_cast<scalar_t>(std::min(e * inverse, 1.0) * cap * weight);
              }
              g[target] = static_cast<scalar_t>(-weight * mass);
              continue;
            }
            if (loss_code <= 2 && !(row_val > 0.0)) {
              std::fill(g, g + c, scalar_t(0));
              continue;
            }
            double total = 0.0;
            if (loss_code <= 2) {
              const int64_t cut = static_cast<int64_t>(second);
              const double step = weight / kd;
              const scalar_t g_val = static_cast<scalar_t>(step);
              const scalar_t* z_row = input + row * c;
              total = std::is_same_v<scalar_t, double>
                          ? topk_hinge_chosen(reinterpret_cast<const double*>(z_row), c, target,
                                              first, cut, loss_code == 2, g_val, step, g)
                          : topk_hinge_chosen(float_row(z_row, c, narrow.data()), c, target, first,
                                              cut, loss_code == 2, g_val, step, g);
            } else {
              hinge_margins(input + row * c, c, target, h.data());
              for (int64_t j = 0; j < c; ++j) {
                const double a =
                    (j == target) ? 0.0 : topk_projected(h[j], first, second, gamma, k);
                const double x = (a / gamma) * weight;
                g[j] = static_cast<scalar_t>(x);
                total += x;
              }
            }
            // The target entry: minus the competitors' unrounded sum, rounded once.
            g[target] = static_cast<scalar_t>(-total);
          }
        });
      });
  return output;
}

inline double centered_exp_diff(double e, double e0, double dz) {
  return dz >= 0.0 ? -e * std::expm1(-dz) : e0 * std::expm1(dz);
}

// The sum of exp(z[j] - top) over the j where keep(j), on the scale of their own maximum and
// rounded once into subnormals (as single_entropy_row's competitors), compensated: the
// terms exp(z[j] - top) would each round there first.
template <typename F>
double scaled_exp_sum(const double* z, int64_t c, double top, F&& keep) {
  double high = -std::numeric_limits<double>::infinity(), sum = 0.0, carry = 0.0;
  for (int64_t j = 0; j < c; ++j)
    if (keep(j) && z[j] > high) high = z[j];
  if (high == -std::numeric_limits<double>::infinity()) return 0.0;
  for (int64_t j = 0; j < c; ++j) {
    if (!keep(j)) continue;
    const double y = std::exp(z[j] - high) - carry, t = sum + y;
    carry = (t - sum) - y;
    sum = t;
  }
  const double half = std::exp((high - top) / 2);
  return half * (half * sum);
}

// f(j) for every j < c with m[j] != 0, skipping zero words of the mask.
template <typename F>
inline void each_positive(const uint8_t* m, int64_t c, F&& f) {
  int64_t j = 0;
  for (; j + 8 <= c; j += 8) {
    uint64_t word;
    std::memcpy(&word, m + j, sizeof word);
    if (word == 0) continue;
    for (int i = 0; i < 8; ++i)
      if (m[j + i] != 0) f(j + i);
  }
  for (; j < c; ++j)
    if (m[j] != 0) f(j);
}

// Mask lanes of m[j, j + count): all ones where m is nonzero. Full lanes widen the bytes
// with one instruction where the build has AVX2 (GCC unpacks a byte vector byte by byte).
inline dlanes::vl mask_lanes(const uint8_t* m, int count) {
#if defined(SDCA_SIMD)
  if (count == dlanes::L) {
    dlanes::vl v;
    if constexpr (dlanes::L == 4) {
      int32_t word;
      std::memcpy(&word, m, sizeof word);
      const __m256i w = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(word));
      std::memcpy(&v, &w, sizeof w);
      return v != 0;
    }
#if defined(SDCA_AVX512)
    if constexpr (dlanes::L == 8) {
      const __m512i w = _mm512_cvtepu8_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(m)));
      std::memcpy(&v, &w, sizeof w);
      return v != 0;
    }
#endif
  }
#endif
  typedef uint8_t vb __attribute__((vector_size(dlanes::L)));
  vb bytes{};
  std::memcpy(&bytes, m, static_cast<std::size_t>(count));
  return __builtin_convertvector(bytes, dlanes::vl) != 0;
}

// Boolean-mask entropy of a row of double logits z whose exponentials exp(z - max) are all
// normal, on double lanes: the value and (need) stats of multilabel_row_forward,
// h = exp(z - max). false (h untouched) where the row is outside that range.
template <typename T>
bool multilabel_entropy_lanes(const T* z, const uint8_t* m, int64_t c, double lo, double top,
                              int64_t count, int64_t first, double* h, double* stats, double& value,
                              bool need) {
  using namespace dlanes;
  if (!(lo - top >= -708.0)) return false;
  const int64_t index_top = find_at_least(z, 0, c, static_cast<T>(top));
  const double anchor = z[first];
  const double p = static_cast<double>(count);
  Accumulator<true> den, tail, negs;  // compensated: C - 1 equal terms beside a 1
  for (int64_t j = 0; j < c; j += L) {
    const int n = static_cast<int>(std::min<int64_t>(L, c - j));
    const vd e = head(exp(part(z, j, n, top) - top), n);
    tail.add(select(iota() + j == index_top, splat(0.0), e));
    if (!need) continue;
    put(h + j, e, n);
    den.add(e);
    negs.add(select(mask_lanes(m + j, n), splat(0.0), e));
  }
  double gap = 0.0;
  each_positive(m, c, [&](int64_t j) {
    const double raw_gap = top - z[j];
    gap += std::isfinite(raw_gap) ? (raw_gap / p) : (top / p - z[j] / p);
  });
  value = gap + std::log1p(tail.value());
  if (!need) return true;
  const double e0 = std::exp(anchor - top);
  double diffs = 0.0;
  each_positive(m, c, [&](int64_t j) { diffs += centered_exp_diff(h[j], e0, z[j] - anchor); });
  const double total = den.value();
  stats[0] = top;
  stats[1] = total;
  stats[2] = anchor;
  stats[3] = (diffs / total + negs.value() / total) / p;
  stats[4] = lo;  // the backward's exponentials are normal too
  return true;
}

// Mask lanes of m[j, j + count) for float lanes: all ones where m is nonzero.
inline lanes::vi mask_lanes32(const uint8_t* m, int count) {
#if defined(SDCA_SIMD)
  if (count == lanes::L) {  // as mask_lanes: eight bytes at a time
    lanes::vi v;
    for (int i = 0; i < lanes::L; i += 8) {
      const __m256i w =
          _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(m + i)));
      std::memcpy(reinterpret_cast<int32_t*>(&v) + i, &w, sizeof w);
    }
    return v != 0;
  }
#endif
  typedef uint8_t vb __attribute__((vector_size(lanes::L)));
  vb bytes{};
  std::memcpy(&bytes, m, static_cast<std::size_t>(count));
  return __builtin_convertvector(bytes, lanes::vi) != 0;
}

// Rows whose float exponentials exp(z - max) all stay normal take float32 work.
constexpr float kFloatSpread = -80.0f;

// lanes::exp of x in [-86.6, 0] without its clamps and underflow flag (the same bits there):
// the arguments z - max and (z - max) / 2 of the float row kernels.
inline lanes::vf exp_nonpositive(lanes::vf x) {
  using namespace lanes;
  vi k;
  const vf n = round(x * 1.44269504088896341f, k);
  const vf r = fma(n, 2.12194440e-4f, fma(n, -0.693359375f, x));
  vf p = fma(splat(1.9875691500e-4f), r, 1.3981999507e-3f);
  p = fma(p, r, 8.3334519073e-3f);
  p = fma(p, r, 4.1665795894e-2f);
  p = fma(p, r, 1.6666665459e-1f);
  p = fma(p, r, 5.0000001201e-1f);
  const vf y = fma(p * r, r, r) + 1.0f;
  return (vf)((vi)y + (k << 23));
}

// body(v, j, count) on the float lanes of a float or 16-bit row read in place (16-bit ones
// widened in registers), the last lanes padded with pad.
template <typename T, typename F>
inline void each_row(const T* x, int64_t n, float pad, F&& body) {
  using namespace lanes;
  int64_t j = 0;
  for (; j + L <= n; j += L) body(load_float(x + j), j, L);
  if (j < n) {
    float tail[L];
    for (int i = 0; i < L; ++i) tail[i] = i < n - j ? static_cast<float>(x[j + i]) : pad;
    body(load(tail), j, static_cast<int>(n - j));
  }
}

// The extrema of a float or 16-bit row read in place, two independent vectors of each per
// step (named, as in max_except_rows); finite turns false on any nonfinite entry.
template <typename T>
void extrema_rows(const T* x, int64_t n, float& lo, float& hi, bool& finite) {
  using namespace lanes;
  constexpr float kInf = std::numeric_limits<float>::infinity();
  vf t0 = splat(-kInf), t1 = t0, b0 = splat(kInf), b1 = b0, s0 = splat(0.0f), s1 = s0;
  auto lane = [&](vf& top, vf& bottom, vf& sum, int64_t at) {
    const vf v = load_float(x + at);
    top = max_lanes(top, v), bottom = min_lanes(bottom, v), sum = sum + v;
  };
  int64_t j = 0;
  for (; j + 2 * L <= n; j += 2 * L) lane(t0, b0, s0, j), lane(t1, b1, s1, j + L);
  const vf high = max_lanes(t0, t1), low = min_lanes(b0, b1), total = s0 + s1;
  lo = kInf, hi = -kInf;
  bool sums = true;
  for (int i = 0; i < L; ++i) {
    lo = std::min(lo, low[i]), hi = std::max(hi, high[i]);
    sums &= std::isfinite(total[i]);
  }
  // Lane sums that are not finite: a nonfinite entry, or large finite ones.
  for (int64_t i = 0; !sums && i < j; ++i) finite &= std::isfinite(static_cast<float>(x[i]));
  for (; j < n; ++j) {
    const float v = static_cast<float>(x[j]);
    lo = std::min(lo, v), hi = std::max(hi, v);
    finite &= std::isfinite(v);
  }
}

// Boolean-mask entropy of a float or 16-bit row in float32 exponentials with double row
// sums, in one pass over the row: the value and (need) stats of multilabel_row_forward
// (stats[5] = 1). The positives (count, the first at first) take double: their centered
// differences e0 expm1(z - anchor) and the average gap. false (nothing written) where some
// exp(z - max) would leave float's normal range. Flattened: the lane exponentials inline,
// and the sums stay in registers.
template <typename T>
__attribute__((flatten)) bool multilabel_entropy_float(const T* x, const uint8_t* m, int64_t c,
                                                       float lo, float top, int64_t count,
                                                       int64_t first, double* stats, double& value,
                                                       bool need) {
  using namespace lanes;
  if (!(lo - top >= kFloatSpread)) return false;
  // With one positive, the value itself can be a tiny tail.
  if (count == 1 && static_cast<float>(x[first]) == top && lo - top < -16.0f) return false;
  Accumulator rest, negs;
  vi ties{};  // minus the count of entries equal to top, whose exponentials are exactly 1
  vi negative_ties{};
  each_row(x, c, top, [&](vf v, int64_t j, int n) {
    vf e = exp_nonpositive(v - top);
    vi at_top = v == top;
    if (n < L) {  // the padding (at top) adds nothing
      const vi live = iota() < n;
      e = select(live, e, splat(0.0f)), at_top = at_top & live;
    }
    ties += at_top;
    rest.add(select(at_top, splat(0.0f), e));
    if (need) {
      const vi positive = mask_lanes32(m + j, n);
      negs.add(select(positive, splat(0.0f), e));
      negative_ties |= at_top & ~positive;
    }
  });
  int64_t tops = 0;
  bool negative_maximum = false;
  for (int i = 0; i < L; ++i) {
    tops -= ties[i];
    negative_maximum = negative_maximum || negative_ties[i] != 0;
  }
  // Without a negative maximum the gradient norm may be tiny. Form large score
  // gaps in double instead of rounding them before their float exponential.
  if (need && !negative_maximum && lo - top < -16.0f) return false;
  const double p = static_cast<double>(count);
  double gap = 0.0;
  each_positive(
      m, c, [&](int64_t j) { gap += (static_cast<double>(top) - static_cast<float>(x[j])) / p; });
  // The sum without one entry at top: exp(z - max) over the others.
  const double tail = rest.value() + static_cast<double>(tops - 1), total = tail + 1.0;
  value = gap + std::log1p(tail);
  if (!need) return true;
  const double anchor = static_cast<float>(x[first]);
  const double e0 = std::exp(anchor - static_cast<double>(top));
  double diffs = 0.0;
  each_positive(m, c,
                [&](int64_t j) { diffs += e0 * std::expm1(static_cast<float>(x[j]) - anchor); });
  stats[0] = top;
  stats[1] = total;
  stats[2] = anchor;
  stats[3] = (diffs / total + negs.value() / total) / p;
  stats[4] = lo;
  stats[5] = 1.0;  // float32 work, which the backward takes too
  return true;
}

inline void simplex_michelot(const double* h, const uint8_t* m, int64_t c, uint8_t target_flag,
                             double gamma, double& origin, double& correction) {
  double top = -std::numeric_limits<double>::infinity();
  for (int64_t j = 0; j < c; ++j) {
    if ((m[j] != 0) == (target_flag != 0) && h[j] > top) top = h[j];
  }
  int64_t count = 0;
  double sum = 0.0;
  for (int64_t j = 0; j < c; ++j) {
    if ((m[j] != 0) == (target_flag != 0)) {
      const double g = h[j] - top;
      if (g > -gamma) {
        ++count;
        sum += g;
      }
    }
  }
  double s = std::max((sum - gamma) / static_cast<double>(count), -gamma);
  for (int64_t step = 0; step <= c; ++step) {
    int64_t n_in = 0;
    double sum_in = 0.0;
    double least = std::numeric_limits<double>::infinity();
    for (int64_t j = 0; j < c; ++j) {
      if ((m[j] != 0) == (target_flag != 0)) {
        const double g = h[j] - top;
        if (g > s) {
          ++n_in;
          sum_in += g;
          if (g < least) least = g;
        }
      }
    }
    s = (sum_in - gamma) / static_cast<double>(n_in);
    if (least > s) break;
  }
  origin = top;
  correction = s;
}

template <typename scalar_t>
double multilabel_row_forward(const scalar_t* z_in, const uint8_t* m_in, int64_t c,
                              int64_t obj_code, double gamma, double* z, double* h, double* stats,
                              RowErrors& errors, float* narrow, bool fp32, bool need) {
  if (obj_code == 0) {
    // Float lanes (fp32) on the row in place, or double lanes straight from float rows
    // (16-bit ones widened to float first).
    double value = 0.0, lo, hi;
    stats[5] = 0.0;
    bool finite = true;
    if constexpr (std::is_same_v<scalar_t, double>) {
      dlanes::extrema(z_in, c, lo, hi, finite);
    } else {
      float low, high;
      extrema_rows(z_in, c, low, high, finite);
      lo = low, hi = high;
    }
    int64_t count = 0, first = c;
    each_positive(m_in, c, [&](int64_t j) { first = count++ == 0 ? j : first; });
    if (!finite) errors.finite = true;
    if (count == 0) errors.positive = true;
    if (!finite || count == 0) return 0.0;
    bool done;
    if constexpr (std::is_same_v<scalar_t, double>) {
      done = multilabel_entropy_lanes(z_in, m_in, c, lo, hi, count, first, h, stats, value, need);
    } else {
      done = (fp32 &&
              multilabel_entropy_float(z_in, m_in, c, static_cast<float>(lo),
                                       static_cast<float>(hi), count, first, stats, value, need)) ||
             multilabel_entropy_lanes(float_row(z_in, c, narrow), m_in, c, lo, hi, count, first, h,
                                      stats, value, need);
    }
    if (done) return value;
  }
  convert_to_double(z_in, z, c);
  bool finite = true;
  int64_t count = 0, first = c, index_top = 0, i_low = 0, i_high = 0;
  double top = -std::numeric_limits<double>::infinity();
  double low = std::numeric_limits<double>::infinity();
  double high = -std::numeric_limits<double>::infinity();
  for (int64_t j = 0; j < c; ++j) {
    const double v = z[j];
    finite &= std::isfinite(v);
    if (v > top) {
      top = v;
      index_top = j;
    }
    if (m_in[j] != 0) {
      if (count == 0) first = j;
      ++count;
      if (v < low) {
        low = v;
        i_low = j;
      }
    } else {
      if (v > high) {
        high = v;
        i_high = j;
      }
    }
  }
  if (!finite) errors.finite = true;
  if (count == 0) errors.positive = true;
  if (obj_code != 0 && count == c) errors.negative = true;
  if (!finite || count == 0 || (obj_code != 0 && count == c)) return 0.0;
  if (obj_code == 0) {
    const double anchor = z[first];
    const double e0 = std::exp(anchor - top);
    const double p = static_cast<double>(count);
    double den = 0.0, gap = 0.0, diffs = 0.0;
    exp_shifted(z, c, top, extrema(z, c).second, h);
    for (int64_t j = 0; j < c; ++j) {
      const double e = h[j];
      den += e;
      if (m_in[j] != 0) {
        const double raw_gap = top - z[j];
        gap += std::isfinite(raw_gap) ? (raw_gap / p) : (top / p - z[j] / p);
        diffs += centered_exp_diff(e, e0, z[j] - anchor);
      }
    }
    // Rows beyond the lanes' spread: their tails may leave double's normal range.
    const double tail = scaled_exp_sum(z, c, top, [&](int64_t j) { return j != index_top; });
    const double negs = scaled_exp_sum(z, c, top, [&](int64_t j) { return m_in[j] == 0; });
    const double val = gap + std::log1p(tail);
    stats[0] = top;
    stats[1] = den;
    stats[2] = anchor;
    stats[3] = (diffs / den + negs / den) / p;
    stats[4] = -std::numeric_limits<double>::infinity();  // exponentials beyond lanes
    return val;
  }
  if (obj_code == 1) {
    const double val = std::max(hinge_margin(high, low), 0.0);
    stats[0] = static_cast<double>(i_low);
    stats[1] = static_cast<double>(i_high);
    stats[2] = val > 0.0 ? 1.0 : 0.0;
    return val;
  }
  const double z0 = z[first];
  const bool recenter =
      std::abs((high - low) + 1.0) < 0.0625 * (1.0 + std::abs(low - z0) + std::abs(high - z0));
  for (int64_t j = 0; j < c; ++j) {
    if (recenter) {
      h[j] = (m_in[j] != 0) ? (low - z[j]) : hinge_margin(z[j], low);
    } else {
      const double d = z[j] - z0;
      h[j] = (m_in[j] != 0) ? (0.5 - d) : (0.5 + d);
    }
  }
  double top_a = 0.0, s_a = 0.0, top_b = 0.0, s_b = 0.0;
  simplex_michelot(h, m_in, c, 1, gamma, top_a, s_a);
  simplex_michelot(h, m_in, c, 0, gamma, top_b, s_b);
  const double eps = std::numeric_limits<double>::epsilon();
  const double terms =
      std::max(gamma, (std::abs(top_a) + std::abs(top_b)) + (std::abs(s_a) + std::abs(s_b)));
  const bool active = (top_a + top_b) + (s_a + s_b) >= -eps * terms;
  double oa = top_a, ca = s_a, ob = top_b, cb = s_b;
  if (!active) {
    const double origin = top_a * 0.5 - top_b * 0.5;
    double scale = 0.0, sum_a = 0.0, sum_b = 0.0;
    int64_t m_cnt = 0;
    for (int64_t j = 0; j < c; ++j) {
      if (m_in[j] != 0) {
        if (h[j] - top_a > s_a) {
          const double da = h[j] - origin;
          scale += std::abs(da);
          sum_a += da;
          ++m_cnt;
        }
      } else {
        if (h[j] - top_b > s_b) {
          const double db = h[j] + origin;
          scale += std::abs(db);
          sum_b += db;
          ++m_cnt;
        }
      }
    }
    const double tol = 8.0 * eps * scale;
    double t = m_cnt > 0 ? (sum_a - sum_b) / static_cast<double>(m_cnt) : 0.0;
    double cut_a = -std::numeric_limits<double>::infinity();
    double cut_b = -std::numeric_limits<double>::infinity();
    for (int64_t fixes = 0; fixes <= c && m_cnt > 0; ++fixes) {
      double sum_x = 0.0, sum_y = 0.0;
      int64_t n_x = 0, n_y = 0;
      for (int64_t j = 0; j < c; ++j) {
        if (m_in[j] != 0) {
          const double da = h[j] - origin;
          if ((h[j] - top_a > s_a) && (da > cut_a) && !(da > t)) {
            sum_x += da;
            ++n_x;
          }
        } else {
          const double db = h[j] + origin;
          if ((h[j] - top_b > s_b) && (db > cut_b) && !(db > -t)) {
            sum_y += db;
            ++n_y;
          }
        }
      }
      const double infeas = (sum_x - sum_y) - static_cast<double>(n_x + n_y) * t;
      const bool drop_y = (n_y > 0) && (infeas > tol);
      const bool drop_x = (!drop_y) && (n_x > 0) && (infeas < -tol);
      if (!(drop_x || drop_y)) break;
      const double mt = static_cast<double>(m_cnt) * t;
      const double following = drop_y ? (mt + sum_y) : (mt - sum_x);
      if (drop_x) cut_a = std::max(cut_a, t);
      if (drop_y) cut_b = std::max(cut_b, -t);
      m_cnt -= drop_y ? n_y : n_x;
      if (m_cnt > 0) t = following / static_cast<double>(m_cnt);
    }
    oa = origin;
    ca = t;
    ob = -origin;
    cb = -t;
  }
  double val = 0.0;
  constexpr double tiny =
      std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon();
  for (int64_t j = 0; j < c; ++j) {
    const double shift = (m_in[j] != 0) ? ((h[j] - oa) - ca) : ((h[j] - ob) - cb);
    const double x = std::min(std::max(shift, 0.0), gamma);
    if (!(x > 0.0)) continue;
    const double hx = h[j] * x, xx = x * x / 2.0, v = std::abs(hx - xx);
    val += v <= std::numeric_limits<double>::max() && v >= tiny ? (hx - xx) / gamma
                                                                : (h[j] - x / 2.0) * (x / gamma);
  }
  stats[0] = z0;
  stats[1] = low;
  stats[2] = recenter ? 1.0 : 0.0;
  stats[3] = oa;
  stats[4] = ca;
  stats[5] = ob;
  stats[6] = cb;
  return val;
}

std::tuple<at::Tensor, at::Tensor> multilabel_forward(const at::Tensor& z, const at::Tensor& mask,
                                                      int64_t reduction, int64_t obj_code,
                                                      double gamma, bool fp32, bool need) {
  const int64_t n = z.size(0), c = z.size(1);
  auto stats = at::empty({n, 7}, z.options().dtype(at::kDouble));
  std::vector<double> values(n);
  RowErrors errors;
  double* s = stats.data_ptr<double>();
  const uint8_t* m_ptr = mask_bytes(mask);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, z.scalar_type(), "sdca_multilabel_forward", [&] {
        const scalar_t* z_ptr = z.data_ptr<scalar_t>();
        const int64_t grain = (n * c >= 16384) ? std::max<int64_t>(1, 4096 / c) : (n + 1);
        at::parallel_for(0, n, grain, [&](int64_t begin, int64_t end) {
          std::vector<double> zr(c), hr(obj_code != 1 ? c : 0);
          std::vector<float> narrow(obj_code == 0 && !std::is_same_v<scalar_t, double> ? c : 0);
          for (int64_t row = begin; row < end; ++row) {
            values[row] = multilabel_row_forward(z_ptr + row * c, m_ptr + row * c, c, obj_code,
                                                 gamma, zr.data(), hr.data(), s + 7 * row, errors,
                                                 narrow.data(), fp32, need);
          }
        });
      });
  errors.raise();
  return {reduced_losses(values, reduction, z), stats};
}

// The boolean-mask entropy gradient of a row whose exponentials exp(z - max) are normal (the
// row minimum saved in st[4]) times weight, on double lanes from the (float) row x: the
// negatives' (e / den) weight, then the few positives. e holds C doubles; out C floats for
// 16-bit logits.
template <typename T, typename scalar_t>
void multilabel_entropy_gradient(const T* x, const uint8_t* m, int64_t c, const double* st,
                                 double weight, double* e_row, float* buffer, scalar_t* g) {
  using namespace dlanes;
  const double top = st[0], den = st[1], anchor = st[2], share = st[3];
  const double e0 = std::exp(anchor - top), factor = weight / den;
  float* out = float_output(g, buffer);
  for (int64_t j = 0; j < c; j += L) {
    const int count = static_cast<int>(std::min<int64_t>(L, c - j));
    const vd e = exp(part(x, j, count, top) - top);
    put(e_row + j, e, count);
    const vd gr = e * factor;
    if constexpr (std::is_same_v<scalar_t, double>) {
      put(g + j, gr, count);
    } else {
      for (int i = 0; i < count; ++i) out[j + i] = static_cast<float>(gr[i]);
    }
  }
  each_positive(m, c, [&](int64_t j) {
    const double dz = static_cast<double>(x[j]) - anchor;
    const double gr = (centered_exp_diff(e_row[j], e0, dz) / den - share) * weight;
    if constexpr (std::is_same_v<scalar_t, double>) {
      g[j] = gr;
    } else {
      out[j] = static_cast<float>(gr);
    }
  });
  if constexpr (!std::is_same_v<scalar_t, double>) from_float(out, g, c);
}

// The boolean-mask entropy gradient of a row of multilabel_entropy_float times weight, from
// float32 exponentials, in one pass over the row in place: every entry's squared halves
// sign (exp((z - max) / 2) root)^2 (as lanes::halves), rounded from float into the logits'
// dtype, then the positives in double. false where the factor weight / den exceeds float.
template <typename scalar_t>
__attribute__((flatten)) bool multilabel_entropy_float_gradient(const scalar_t* z, const uint8_t* m,
                                                                int64_t c, const double* st,
                                                                double weight, scalar_t* g) {
  using namespace lanes;
  const double top = st[0], den = st[1], anchor = st[2], share = st[3];
  const double factor = std::abs(weight) / den;
  if (!(factor <= std::numeric_limits<float>::max())) return false;
  const float center = static_cast<float>(top), root = static_cast<float>(std::sqrt(factor));
  const float sign = weight < 0 ? -1.0f : 1.0f;
  each_row(z, c, center, [&](vf v, int64_t j, int count) {
    const vf half = exp_nonpositive((v - center) * 0.5f) * root;
    float out[L];
    store(out, (half * half) * sign);
    if constexpr (std::is_same_v<scalar_t, float>) {
      std::memcpy(g + j, out, static_cast<std::size_t>(count) * sizeof(float));
    } else {
      from_float(out, g + j, count);
    }
  });
  const double e0 = std::exp(anchor - top);
  each_positive(m, c, [&](int64_t j) {
    const double dz = static_cast<double>(static_cast<float>(z[j])) - anchor;
    g[j] = static_cast<scalar_t>(static_cast<float>((e0 * std::expm1(dz) / den - share) * weight));
  });
  return true;
}

// The boolean-mask entropy gradient of a row times weight: from float32 exponentials where
// the forward took them (st[5]), on double lanes where its exponentials are normal (st[4],
// the row minimum, from the forward), else in double through zr and e_row.
template <typename scalar_t>
void multilabel_entropy_row_gradient(const scalar_t* z, const uint8_t* m, int64_t c,
                                     const double* st, double weight, double* zr, double* e_row,
                                     float* buffer, float* narrow, scalar_t* g) {
  if constexpr (!std::is_same_v<scalar_t, double>) {
    if (st[5] != 0.0 && multilabel_entropy_float_gradient(z, m, c, st, weight, g)) return;
  }
  if (st[4] - st[0] >= -708.0) {
    if constexpr (std::is_same_v<scalar_t, double>) {
      multilabel_entropy_gradient(z, m, c, st, weight, e_row, buffer, g);
    } else {
      multilabel_entropy_gradient(float_row(z, c, narrow), m, c, st, weight, e_row, buffer, g);
    }
    return;
  }
  convert_to_double(z, zr, c);
  const double top = st[0], den = st[1], anchor = st[2], share = st[3];
  const double e0 = std::exp(anchor - top);
  exp_shifted(zr, c, top, extrema(zr, c).second, e_row);
  for (int64_t j = 0; j < c; ++j) {
    const double e = e_row[j];
    const double gr =
        (m[j] != 0) ? (centered_exp_diff(e, e0, zr[j] - anchor) / den - share) : (e / den);
    g[j] = static_cast<scalar_t>(gr * weight);
  }
}

at::Tensor multilabel_backward(const at::Tensor& z, const at::Tensor& mask, const at::Tensor& stats,
                               const at::Tensor& upstream, int64_t reduction, int64_t obj_code,
                               double gamma) {
  const int64_t n = z.size(0), c = z.size(1);
  auto output = gradient_like(z);
  const std::vector<double> u = double_values(upstream);
  const double* weights = u.data();
  const double* s = stats.data_ptr<double>();
  const uint8_t* m_ptr = mask_bytes(mask);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, z.scalar_type(), "sdca_multilabel_backward", [&] {
        const scalar_t* z_ptr = z.data_ptr<scalar_t>();
        scalar_t* g_ptr = output.data_ptr<scalar_t>();
        const int64_t grain = (n * c >= 16384) ? std::max<int64_t>(1, 4096 / c) : (n + 1);
        at::parallel_for(0, n, grain, [&](int64_t begin, int64_t end) {
          std::vector<double> zr(obj_code == 1 ? 0 : c), er(obj_code == 0 ? c : 0);
          std::vector<float> fr(obj_code == 0 && !std::is_same_v<scalar_t, double> ? c : 0);
          std::vector<float> xr(fr.size());
          for (int64_t row = begin; row < end; ++row) {
            double weight = weights[reduction == 0 ? row : 0];
            if (reduction == 1) weight /= n;
            scalar_t* g = g_ptr + row * c;
            const double* st = s + 7 * row;
            const uint8_t* mr = m_ptr + row * c;
            if (obj_code == 1) {
              std::fill(g, g + c, scalar_t(0));
              const double w = weight * st[2];
              g[static_cast<int64_t>(st[1])] = static_cast<scalar_t>(w);
              g[static_cast<int64_t>(st[0])] = static_cast<scalar_t>(-w);
              continue;
            }
            if (obj_code == 0) {
              multilabel_entropy_row_gradient(z_ptr + row * c, mr, c, st, weight, zr.data(),
                                              er.data(), fr.data(), xr.data(), g);
              continue;
            }
            convert_to_double(z_ptr + row * c, zr.data(), c);
            const double z0 = st[0], low = st[1];
            const bool recenter = st[2] != 0.0;
            const double oa = st[3], ca = st[4], ob = st[5], cb = st[6];
            for (int64_t j = 0; j < c; ++j) {
              const bool pos = mr[j] != 0;
              const double hj = recenter ? (pos ? (low - zr[j]) : hinge_margin(zr[j], low))
                                         : (pos ? (0.5 - (zr[j] - z0)) : (0.5 + (zr[j] - z0)));
              const double shift = pos ? ((hj - oa) - ca) : ((hj - ob) - cb);
              const double x = std::min(std::max(shift, 0.0), gamma) / gamma;
              g[j] = static_cast<scalar_t>((pos ? -x : x) * weight);
            }
          }
        });
      });
  return output;
}

}  // namespace

#endif  // SDCA_TORCH_TOPK_HINGE_H
