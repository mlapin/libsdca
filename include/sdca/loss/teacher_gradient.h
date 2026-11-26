#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_LOSS_TEACHER_GRADIENT_H)
#if SDCA_ISA_ID == 1
#define SDCA_LOSS_TEACHER_GRADIENT_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_LOSS_TEACHER_GRADIENT_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_LOSS_TEACHER_GRADIENT_H3 1
#else
#define SDCA_LOSS_TEACHER_GRADIENT_H4 1
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "sdca/core/math.h"
#include "sdca/core/reduce.h"
#include "sdca/core/storage.h"

SDCA_ISA_BEGIN
namespace loss {
namespace distill {

// A teacher row's VJP is p_j * E_p[z_i - z_j] / T^2. Center the
// scores on the most probable teacher class so its small gradient survives
// saturation. The partition and centered first moment share exponentials.
using TeacherLanes = core::Lanes<double, core::isa::native, 2>;

// A row with a nearly cancelling mean needs the low parts of both moment
// and mass. For almost-equal teacher scores, retain exp(x)-1 separately.
inline void refine_teacher_row(const float* z, const float* t, int64_t c, double top, double inv,
                               double norm, double origin, double mean, double bound, double factor,
                               const double* mass, float* gradient) {
  core::Neumaier weighted, total;
  for (int64_t j = 0; j < c; ++j) {
    if (mass[j] == 0.0) continue;
    const double x = (static_cast<double>(t[j]) - top) * inv;
    const bool near = x > -0.5;
    if (near) {
      total.add(1.0);
      weighted.add(z[j]);
    }
    const double e = near ? std::expm1(x) : mass[j], product = e * z[j];
    total.add(e);
    weighted.add(product);
    weighted.add(std::fma(e, static_cast<double>(z[j]), -product));
  }
  for (int64_t j = 0; j < c; ++j) {
    if (mass[j] == 0.0 || std::abs((origin - z[j]) + mean) > bound) continue;
    const double a = -static_cast<double>(z[j]);
    const double hi = std::fma(a, total.hi, weighted.hi);
    const double lo = std::fma(a, total.lo, weighted.lo);
    gradient[j] = static_cast<float>(((hi + lo) * norm) * mass[j] * factor);
  }
}

template <bool AtMaximum>
inline bool teacher_row_at(const float* z, const float* t, int64_t c, double inv, double upstream,
                           double* mass, float* gradient) {
  using D = TeacherLanes;
  constexpr double inf = std::numeric_limits<double>::infinity();
  const auto winner = AtMaximum ? std::max_element(t, t + c) - t : 0;
  const double top = t[winner], origin = z[winner];
  if (!std::isfinite(top) || !std::isfinite(origin) || !std::isfinite(upstream)) return false;
  D sum = D::zero(), moment = D::zero(), magnitude = D::zero();
  bool valid = true;
  for (int64_t j = 0; j < c; j += D::L) {
    const int count = static_cast<int>(std::min<int64_t>(D::L, c - j));
    const D a = core::load<D>(z + j, count, 0.0), b = core::load<D>(t + j, count, -inf);
    valid &= core::all((a < inf) & (b < inf) & ((a > -inf) | (b == -inf)));
    const D x = (b - D::splat(top)) * inv;
    D e;
    if (AtMaximum) {
      e = core::exp_nonpositive(x);
    } else {
      const auto bounded = core::abs(x) <= 16.0;
      valid &= core::all(bounded | (b == -inf));
      core::Ints<double, core::isa::native, 2> k;
      const D tail = core::detail::expm1_parts(core::select(bounded, x, D::zero()), k);
      const D scale = core::detail::pow2<double>(k);
      e = core::select(bounded, core::fma(scale, tail, scale), D::zero());
    }
    e.store(mass + j);
    sum += e;
    const D term = core::select(a > -inf, (a - D::splat(origin)) * e, D::zero());
    moment += term;
    magnitude += core::abs(term);
  }
  if (!valid) return false;
  const double norm = 1.0 / core::sum(sum), mean = core::sum(moment) * norm;
  const double factor = ((upstream * norm) * inv) * inv;
  const double bound =
      (c + 8.) * std::numeric_limits<double>::epsilon() * (norm * core::sum(magnitude));
  bool refine = false;
  for (int64_t j = 0; j < c; j += D::L) {
    const int count = static_cast<int>(std::min<int64_t>(D::L, c - j));
    const D e = D::load(mass + j), a = core::load<D>(z + j, count, 0.0);
    const D centered = (D::splat(origin) - a) + D::splat(mean);
    refine |= !core::all((core::abs(centered) > bound) | (e == 0.0));
    const D g = core::select(e > 0.0, centered * e, D::zero()) * factor;
    core::store(gradient + j, g, count);
  }
  if (refine && bound > 0.0) {
    if (!AtMaximum) return false;
    refine_teacher_row(z, t, c, top, inv, norm, origin, mean, bound, factor, mass, gradient);
  }
  return true;
}

// Center ordinary teacher scores on the first element so their exponentials
// and moments need one input scan. Outside the direct exponential range,
// or near a sign crossing, recenter on the largest teacher for the full solve.
inline bool teacher_row(const float* z, const float* t, int64_t c, double inv, double upstream,
                        double* mass, float* gradient) {
  return teacher_row_at<false>(z, t, c, inv, upstream, mass, gradient) ||
         teacher_row_at<true>(z, t, c, inv, upstream, mass, gradient);
}

// Consecutive rows occupy lanes, so a short class axis does not require
// a partially filled vector or horizontal reductions for each individual row.
inline void teacher_short_group(const float* z, const float* t, int64_t c, double inv,
                                const float* factors, int stride, double scale_factor, float* out,
                                unsigned char* exceptional, double* scratch) {
  using D = TeacherLanes;
  constexpr int lanes = D::L;
  double upstream[lanes];
  for (int r = 0; r < lanes; ++r) upstream[r] = factors[r * stride] * scale_factor;
  const D u = D::load(upstream);
  double* zd = scratch;
  double* td = zd + c * lanes;
  for (int r = 0; r < lanes; ++r) {
    for (int j = 0; j < c; ++j) {
      zd[j * lanes + r] = z[r * c + j];
      td[j * lanes + r] = t[r * c + j];
    }
  }
  const D origin = D::load(zd), initial = D::load(td);
  D sum = D::zero(), moment = D::zero(), magnitude = D::zero();
  D bad =
      core::select(core::abs(u) < std::numeric_limits<double>::infinity(), D::zero(), D::splat(1.));
  for (int j = 0; j < c; ++j) {
    const D x = (D::load(td + j * lanes) - initial) * inv;
    const D dz = D::load(zd + j * lanes) - origin;
    const auto safe = core::abs(x) <= 16.0;
    const auto ok = safe & (core::abs(dz) <= 2. * std::numeric_limits<float>::max());
    bad = core::max(bad, core::select(ok, D::zero(), D::splat(1.)));
    core::Ints<double, core::isa::native, 2> k;
    const D m = core::detail::expm1_parts(core::select(safe, x, D::zero()), k);
    const D scale = core::detail::pow2<double>(k);
    const D e = core::fma(scale, m, scale);
    e.store(td + j * lanes);
    dz.store(zd + j * lanes);
    sum += e;
    moment += dz * e;
    magnitude += core::abs(dz * e);
  }
  const D norm = D::splat(1.) / sum, mean = moment * norm;
  const D factor = (norm * u * inv) * inv;
  const D bound = magnitude * norm * ((c + 8.) * std::numeric_limits<double>::epsilon());
  for (int j = 0; j < c; ++j) {
    const D centered = mean - D::load(zd + j * lanes);
    const auto close = (core::abs(centered) <= bound) & (bound > 0.0);
    bad = core::max(bad, core::select(close, D::splat(1.), D::zero()));
    (centered * D::load(td + j * lanes) * factor).store(zd + j * lanes);
  }
  for (int r = 0; r < lanes; ++r) {
    if (bad[r] != 0.) {
      exceptional[r] = !teacher_row_at<true>(z + r * c, t + r * c, c, inv, upstream[r],
                                             scratch + 2 * c * lanes, out + r * c);
      if (exceptional[r]) std::fill(out + r * c, out + (r + 1) * c, 0.f);
    } else {
      exceptional[r] = 0;
      for (int j = 0; j < c; ++j) out[r * c + j] = static_cast<float>(zd[j * lanes + r]);
    }
  }
}

inline void teacher_rows(const float* z, const float* t, int64_t n, int64_t c, double inv,
                         const float* factors, int stride, double scale, float* out,
                         unsigned char* exceptional) {
  constexpr int lanes = TeacherLanes::L;
  // A short class axis leaves at most four full vectors per reduction. Put
  // independent rows in lanes to avoid a horizontal sum for every row.
  const bool across = lanes > 1 && n >= lanes && c <= 4 * lanes;
  std::vector<double> scratch(c + lanes + (across ? 2 * c * lanes : 0));
  int64_t i = 0;
  if (across) {
    for (; i + lanes <= n; i += lanes) {
      teacher_short_group(z + i * c, t + i * c, c, inv, factors + i * stride, stride, scale,
                          out + i * c, exceptional + i, scratch.data());
    }
  }
  for (; i < n; ++i) {
    exceptional[i] = !teacher_row(z + i * c, t + i * c, c, inv, factors[i * stride] * scale,
                                  scratch.data(), out + i * c);
    if (exceptional[i]) std::fill(out + i * c, out + (i + 1) * c, 0.f);
  }
}

}  // namespace distill
}  // namespace loss
SDCA_ISA_END
#endif
