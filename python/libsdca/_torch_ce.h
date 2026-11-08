#ifndef SDCA_TORCH_CE_H
#define SDCA_TORCH_CE_H

namespace {

// Entropy row statistics: the largest competitor, the label score and sum exp(z_j - maximum)
// over the competitors.
struct EntropyRow {
  double maximum, label, sum;
};

// Logits per block of the float row sweeps: reduced in L1 (16-bit rows widened as loaded).
constexpr int64_t kFloatBlock = 2048;

// float32 exponentials (fp32 with non-double storage) in one sweep over the row: per block,
// its largest competitor, then its exponentials from the running maximum (the earlier sum
// rescaled in double where that rises). False for nonfinite logits.
template <typename scalar_t>
bool entropy_row_float(const scalar_t* src, int64_t c, int64_t target, EntropyRow& row) {
  bool finite = true;
  float maximum = -std::numeric_limits<float>::infinity();
  double sum = 0.0;
  for (int64_t b = 0; b < c; b += kFloatBlock) {
    const int64_t n = std::min(kFloatBlock, c - b);
    const scalar_t* z = src + b;
    // The label's index in the block (outside [0, n) where it is elsewhere).
    const int64_t skip = target - b;
    const float top = max_except(z, n, skip, finite);
    if (top > maximum) {
      if (sum != 0.0) sum *= std::exp(static_cast<double>(maximum) - top);
      maximum = top;
    }
    sum += lanes::exp_sum_except(z, n, skip, maximum);
  }
  if (!finite) return false;
  row.maximum = maximum;
  row.label = static_cast<float>(src[target]);
  row.sum = sum;
  return true;
}

// float64 exponentials: compensated sums for float64 storage, Sleef or inlined exponentials.
template <typename scalar_t>
bool entropy_row_double(const scalar_t* src, int64_t c, int64_t target, double* buffer,
                        EntropyRow& row) {
  convert_to_double(src, buffer, c);
  const double label = buffer[target];
  buffer[target] = buffer[target == 0 ? 1 : 0];
  const auto [maximum, minimum] = extrema(buffer, c);
  if (!(std::isfinite(label) && std::isfinite(maximum) && std::isfinite(minimum))) return false;
  buffer[target] = -std::numeric_limits<double>::infinity();
  constexpr bool kCompensated = std::is_same_v<scalar_t, double>;
#ifdef SDCA_SIMD
  row.sum = minimum - maximum >= -708.0 && c >= simd::W
                ? exp_sum_simd<kCompensated>(buffer, c, maximum)
                : exp_sum_generic<kCompensated>(buffer, c, maximum);
#else
  row.sum = exp_sum_generic<kCompensated>(buffer, c, maximum);
#endif
  row.maximum = maximum;
  row.label = label;
  return true;
}

inline double relative_gap(const EntropyRow& row) {
  return (row.maximum - row.label) + std::log(row.sum);
}

// The row loss log(1 + exp(relative)). Long double holds only O(N) row scalars: losses that
// become representable after a sum, and finite means of overflowing rows.
inline long double entropy_loss(const EntropyRow& row) {
  // exp(relative) = exp(gap) sum without a logarithm where it is far from both branches'
  // limits (sum lies in [1, C]).
  const double gap = row.maximum - row.label;
  if (std::abs(gap) <= 30 && row.sum <= 1e4) return std::log1p(std::exp(gap) * row.sum);
  const double relative = relative_gap(row);
  if (relative < -36)
    return std::log1p(std::exp(static_cast<long double>(row.maximum) - row.label) * row.sum);
  if (relative > 36) return (static_cast<long double>(row.maximum) - row.label) + std::log(row.sum);
  return std::log1p(std::exp(relative));
}

// Long double wider than double in range and precision (x87, binary128; not Apple arm64, MSVC).
constexpr bool kWide = std::numeric_limits<long double>::digits >= 64 &&
                       std::numeric_limits<long double>::max_exponent > 1024;

// Whether a float64 loss needs no long double beyond double (always where long double is wide).
inline bool represented_value(long double v) {
  return kWide || (std::isfinite(static_cast<double>(v)) &&
                   std::abs(static_cast<double>(v)) >= std::numeric_limits<double>::min());
}

// Class weights in double (empty without weights); finite and nonnegative.
inline std::vector<double> class_weights(const std::optional<at::Tensor>& weight, int64_t c) {
  if (!weight.has_value() || !weight->defined()) return {};
  TORCH_CHECK_VALUE(weight->dim() == 1 && weight->numel() == c && weight->is_cpu(),
                    "weight must have shape (classes,) and the logits' device and dtype");
  const std::vector<double> values = double_values(*weight);
  for (double v : values)
    TORCH_CHECK_VALUE(std::isfinite(v) && v >= 0, "weight must be finite and nonnegative");
  return values;
}

// Entropy rows of at most kMaxClasses classes in float32 work, a block of lanes::L rows at a
// time (a lane per row, a vector per class): the per-row kernels' arithmetic, bitwise, without
// their padded tails and per-row horizontal sums, which cost more than a small row's logits.
// Rows the block leaves (ignored, invalid labels, nonfinite logits, gradient factors beyond
// float32) take the per-row kernels.
namespace entropy_small {
// At most four float vectors of lanes::L per row: one float group per lane, as
// lanes::Accumulator sums them (32 classes where L >= 8; 16 in 4-lane builds).
constexpr int64_t kMaxClasses = std::min<int64_t>(32, 4 * lanes::L);

// Rows [0, count) of z (count <= lanes::L) by column into col (c lanes::L floats; missing
// rows repeat the last) and their labels, -1 where ignored or invalid.
template <typename scalar_t>
lanes::vi transpose(const scalar_t* z, const int64_t* labels, int64_t ignore_index, int64_t count,
                    int64_t c, float* col) {
  using namespace lanes;
  vi target;
  for (int r = 0; r < L; ++r) {
    const int64_t row = r < count ? r : count - 1, label = labels[row];
    target[r] = label != ignore_index && label >= 0 && label < c ? static_cast<int32_t>(label) : -1;
    for (int64_t j = 0; j < c; ++j) col[j * L + r] = static_cast<float>(z[row * c + j]);
  }
  return target;
}

// entropy_row_float of the block's rows into rows; ok[r] false where the per-row kernel takes
// row r. Its maximum and label, and its exponentials summed as lanes::exp_sum_except sums them
// (float groups of the columns of a lane, then double over the lanes in order).
template <typename scalar_t>
void forward(const scalar_t* z, const int64_t* labels, int64_t ignore_index, int64_t count,
             int64_t c, float* col, EntropyRow* rows, bool* ok) {
  using namespace lanes;
  static_assert(kMaxClasses <= 4 * L, "one float group per lane, as lanes::Accumulator");
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const vi target = transpose(z, labels, ignore_index, count, c, col);
  vf top = splat(-kInf), label{}, bad{};
  for (int64_t j = 0; j < c; ++j) {
    const vf v = load(col + j * L);
    const vi at = target == static_cast<int32_t>(j);
    bad = bad + v * 0.0f;  // NaN for inf and NaN entries
    label = select(at, v, label);
    const vf competitor = select(at, splat(-kInf), v);
    top = select(competitor > top, competitor, top);
  }
  vd sum{};
  for (int64_t i = 0; i < std::min<int64_t>(L, c); ++i) {
    vf group{};
    for (int64_t j = i; j < c; j += L)
      group = group +
              exp(select(target == static_cast<int32_t>(j), splat(-kInf), load(col + j * L)) - top);
    sum = sum + __builtin_convertvector(group, vd);
  }
  for (int r = 0; r < count; ++r) {
    ok[r] = target[r] >= 0 && bad[r] == 0.0f;
    rows[r] = {top[r], label[r], sum[r]};
  }
}

// The block's row losses: entropy_loss's first branch, log1p(exp(gap) sum), on double lanes
// (within a few ulps of libm's; the library calls cost more than a small row's logits) where
// fast[r], else entropy_loss.
inline void losses(const EntropyRow* rows, int64_t count, long double* value) {
  static_assert(lanes::L % dlanes::L == 0, "whole double vectors per block");
  double gap[lanes::L], sum[lanes::L], fast[lanes::L];
  for (int r = 0; r < lanes::L; ++r) {
    gap[r] = r < count ? rows[r].maximum - rows[r].label : 0.0;
    sum[r] = r < count ? rows[r].sum : 1.0;
  }
  for (int r = 0; r < lanes::L; r += dlanes::L) {
    const dlanes::vd g = dlanes::load(gap + r), s = dlanes::load(sum + r);
    dlanes::store(fast + r, dlanes::log1p(dlanes::exp(g) * s));
  }
  for (int r = 0; r < count; ++r)
    value[r] = std::abs(gap[r]) <= 30 && sum[r] <= 1e4 ? fast[r] : entropy_loss(rows[r]);
}

// The block's entropy_float_factors from its statistics st (3 per row) and scales: the odds
// exp(-gap) / sum on double lanes (within a few ulps of libm's), elsewhere as that function.
inline void factors(const double* st, int64_t count, const double* scale, double* mass,
                    double* factor) {
  double gap[lanes::L], sum[lanes::L], odds[lanes::L];
  for (int r = 0; r < lanes::L; ++r) {
    gap[r] = r < count ? st[3 * r] - st[3 * r + 1] : 0.0;
    sum[r] = r < count ? st[3 * r + 2] : 1.0;
  }
  for (int r = 0; r < lanes::L; r += dlanes::L)
    dlanes::store(odds + r, dlanes::exp(-dlanes::load(gap + r)) / dlanes::load(sum + r));
  for (int r = 0; r < count; ++r) {
    if (!(std::abs(gap[r]) <= 600))
      odds[r] = std::exp(-relative_gap({st[3 * r], st[3 * r + 1], st[3 * r + 2]}));
    mass[r] = 1 / (1 + odds[r]);
    factor[r] = mass[r] / sum[r] * std::abs(scale[r]);
  }
}

// lanes::halves of the block's rows where ok into g (row-major, count x c): sign *
// (exp((z - center) / 2) * root)^2, by column in place in col.
template <typename scalar_t>
void halves(float* col, int64_t count, int64_t c, lanes::vf center, lanes::vf root, lanes::vf sign,
            const bool* ok, scalar_t* g) {
  using namespace lanes;
  for (int64_t j = 0; j < c; ++j) {
    const vf half = exp((load(col + j * L) - center) * 0.5f) * root;
    store(col + j * L, (half * half) * sign);
  }
  for (int r = 0; r < count; ++r)
    if (ok[r])
      for (int64_t j = 0; j < c; ++j) g[r * c + j] = static_cast<scalar_t>(col[j * L + r]);
}
}  // namespace entropy_small

void check_entropy_inputs(const at::Tensor& z, const at::Tensor& y, int64_t reduction) {
  TORCH_CHECK_VALUE(z.is_cpu() && z.dim() == 2 && z.is_contiguous() && z.size(1) >= 2,
                    "expected contiguous CPU (N, C) logits with C >= 2");
  TORCH_CHECK_VALUE(y.scalar_type() == at::kLong && y.is_contiguous() && y.is_cpu() &&
                        y.dim() == 1 && y.numel() == z.size(0),
                    "expected contiguous CPU int64 targets of shape (N,)");
  TORCH_CHECK_VALUE(reduction >= 0 && reduction <= 2, "invalid entropy reduction");
}

// Cross-entropy of (N, C) logits and int64 targets, as torch.nn.functional.cross_entropy:
// rows with target ignore_index are neither read nor validated and add nothing; weight scales
// row i by weight[y_i] (mean divides by the sum of those weights, NaN when it is zero).
// fp32 takes float32 exponentials for non-float64 logits (float64 row statistics).
// Returns (loss, stats, represented): loss (N,) for reduction 0 (none), else a scalar, float32
// (float64 for float64 logits); stats float64 of shape (3 N + 1,) when need_stats, else
// undefined: (maximum, label, sum) per row and the mean's divisor (1 for none and sum);
// represented is false only where float64 losses need a long double beyond double.
// Row losses of entropy_forward (validated; zero for ignored rows) and, given s, their
// statistics.
std::vector<long double> entropy_losses(const at::Tensor& z, const int64_t* labels,
                                        int64_t ignore_index, bool work32, double* s) {
  const int64_t n = z.size(0), c = z.size(1);
  std::vector<long double> losses(n);
  RowErrors errors;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, z.scalar_type(), "sdca_entropy", [&] {
    const scalar_t* data = z.data_ptr<scalar_t>();
    at::parallel_for(0, n, grain(n, c), [&](int64_t begin, int64_t end) {
      std::vector<double> wide(work32 ? 0 : c);
      auto store = [&](int64_t i, const EntropyRow& r, long double loss) {
        losses[i] = loss;
        if (s) s[3 * i] = r.maximum, s[3 * i + 1] = r.label, s[3 * i + 2] = r.sum;
      };
      auto row_loss = [&](int64_t i) {
        const int64_t target = labels[i];
        if (target == ignore_index) return;
        const scalar_t* row = data + i * c;
        EntropyRow r;
        if (checked_row(row, c, target, errors, [&] {
              // float64 logits never take float32 work.
              if constexpr (!std::is_same_v<scalar_t, double>)
                if (work32) return entropy_row_float(row, c, target, r);
              return entropy_row_double(row, c, target, wide.data(), r);
            }))
          store(i, r, entropy_loss(r));
      };
      if (!work32 || c > entropy_small::kMaxClasses) {
        for (int64_t i = begin; i < end; ++i) row_loss(i);
        return;
      }
      std::vector<float> col(c * lanes::L);
      for (int64_t i = begin; i < end; i += lanes::L) {
        const int64_t count = std::min<int64_t>(lanes::L, end - i);
        EntropyRow rows[lanes::L];
        bool ok[lanes::L];
        long double value[lanes::L];
        entropy_small::forward(data + i * c, labels + i, ignore_index, count, c, col.data(), rows,
                               ok);
        entropy_small::losses(rows, count, value);
        for (int64_t r = 0; r < count; ++r)
          ok[r] ? store(i + r, rows[r], value[r]) : row_loss(i + r);
      }
    });
  });
  errors.raise();
  return losses;
}

// The losses reduced as torch.nn.functional.cross_entropy does: row i weighs weight[y_i],
// ignored rows weigh zero and mean divides by the total weight (its divisor, 1 otherwise).
// Also whether float64 values are represented without a long double beyond double.
std::tuple<at::Tensor, bool> reduce_entropy(const std::vector<long double>& losses,
                                            const int64_t* labels, int64_t ignore_index,
                                            const std::vector<double>& weights, int64_t reduction,
                                            const at::Tensor& z, double& divisor) {
  const int64_t n = static_cast<int64_t>(losses.size());
  const bool is_double = z.scalar_type() == at::kDouble;
  at::Tensor output = at::empty(reduction == 0 ? at::IntArrayRef{n} : at::IntArrayRef{},
                                z.options().dtype(is_double ? at::kDouble : at::kFloat));
  double* out64 = is_double ? output.data_ptr<double>() : nullptr;
  float* out32 = is_double ? nullptr : output.data_ptr<float>();
  auto store = [&](int64_t i, long double v) {
    if (is_double)
      out64[i] = static_cast<double>(v);
    else
      out32[i] = static_cast<float>(v);
  };
  long double total = 0, weight_sum = 0;
  bool represented = true;
  for (int64_t i = 0; i < n; ++i) {
    if (labels[i] == ignore_index) {
      if (reduction == 0) store(i, 0);
      continue;
    }
    const double w = weights.empty() ? 1.0 : weights[labels[i]];
    represented = represented && represented_value(losses[i]);
    if (reduction == 0) store(i, w * losses[i]);
    total += w * losses[i];
    weight_sum += w;
  }
  divisor = reduction == 1 ? static_cast<double>(weight_sum) : 1.0;
  if (reduction != 0) {
    store(0, reduction == 1 ? total / weight_sum : total);
    represented = represented && (kWide || !std::isinf(static_cast<double>(total)));
  }
  return {output, !is_double || represented};
}

std::tuple<at::Tensor, at::Tensor, bool> entropy_forward(const at::Tensor& z, const at::Tensor& y,
                                                         int64_t reduction, int64_t ignore_index,
                                                         const std::optional<at::Tensor>& weight,
                                                         bool fp32, bool need_stats) {
  check_entropy_inputs(z, y, reduction);
  const int64_t n = z.size(0);
  const std::vector<double> weights = class_weights(weight, z.size(1));
  at::Tensor stats =
      need_stats ? at::empty({3 * n + 1}, z.options().dtype(at::kDouble)) : at::Tensor();
  double* s = need_stats ? stats.data_ptr<double>() : nullptr;
  if (s) std::fill_n(s, 3 * n + 1, 0.0);
  const int64_t* labels = y.data_ptr<int64_t>();
  const auto losses =
      entropy_losses(z, labels, ignore_index, fp32 && z.scalar_type() != at::kDouble, s);
  double divisor;
  auto [output, represented] =
      reduce_entropy(losses, labels, ignore_index, weights, reduction, z, divisor);
  if (s) s[3 * n] = divisor;
  return {output, stats, represented};
}

// sign * (exp((x - center) / 2) * root)^2 over a double row, in place.
void halves_double(double* row, int64_t c, double center, double root, double sign) {
#ifdef SDCA_SIMD
  if (c >= simd::W && (extrema(row, c).second - center) * 0.5 >= -708.0) {
    const simd::vd vc = simd::set1(center), vr = simd::set1(root), vs = simd::set1(sign);
    int64_t j = 0;
    for (; j + simd::W <= c; j += simd::W) {
      const simd::vd half = simd::mul(
          exp_pd_normal(simd::mul(simd::sub(simd::load(row + j), vc), simd::set1(0.5))), vr);
      simd::store(row + j, simd::mul(simd::mul(half, half), vs));
    }
    for (; j < c; ++j) {
      const double half = std::exp((row[j] - center) * 0.5) * root;
      row[j] = (half * half) * sign;
    }
    return;
  }
#endif
  at::vec::map(
      [center, root, sign](Vec x) {
        const Vec half = ((x - Vec(center)) * Vec(0.5)).exp() * Vec(root);
        return (half * half) * Vec(sign);
      },
      row, row, c);
}

// Gradient of one row times scale in double: the competitors' squared halves and the target
// entry -scale times the competitors' mass; confident rows factor from the label.
void entropy_gradient_double(double* row, int64_t c, int64_t target, const double* s,
                             double scale) {
  const EntropyRow r{s[0], s[1], s[2]};
  const double relative = relative_gap(r);
  const double sign = scale < 0 ? -1 : (scale > 0 ? 1 : 0);
  const double root = std::sqrt(std::abs(scale));
  double target_gradient;
  if (relative < -37) {
    const long double mass = std::exp(static_cast<long double>(r.maximum) - r.label) * r.sum;
    target_gradient = -static_cast<double>(mass * scale);
    halves_double(row, c, r.label, root, sign);
  } else {
    const double mass = 1 / (1 + std::exp(-relative));
    target_gradient = -mass * scale;
    halves_double(row, c, r.maximum, root * std::sqrt(mass / r.sum), sign);
  }
  row[target] = target_gradient;
}

// The factors of a row's fp32 gradient from its statistics s: the competitors' probability
// mass and their factor mass / sum |scale|; false where that factor exceeds float32.
inline bool entropy_float_factors(const double* s, double scale, double& mass, double& factor) {
  const EntropyRow r{s[0], s[1], s[2]};
  const double gap = r.maximum - r.label;
  // exp(-relative) = exp(-gap) / sum, without a logarithm where both are moderate.
  const double odds = std::abs(gap) <= 600 ? std::exp(-gap) / r.sum : std::exp(-relative_gap(r));
  mass = 1 / (1 + odds);
  factor = mass / r.sum * std::abs(scale);
  return factor <= std::numeric_limits<float>::max();
}

// The fp32 gradient of a non-double row; false where the row factor exceeds float32.
template <typename scalar_t>
bool entropy_gradient_float(const scalar_t* src, int64_t c, int64_t target, const double* s,
                            double scale, float* buffer, scalar_t* out) {
  double mass, factor;
  if (!entropy_float_factors(s, scale, mass, factor)) return false;
  const float* z = float_row(src, c, buffer);
  float* g = buffer;
  if constexpr (std::is_same_v<scalar_t, float>) g = out;
  lanes::halves(z, c, static_cast<float>(s[0]), static_cast<float>(std::sqrt(factor)),
                scale < 0 ? -1.0f : 1.0f, g);
  if constexpr (!std::is_same_v<scalar_t, float>) from_float(g, out, c);
  out[target] = static_cast<scalar_t>(-mass * scale);
  return true;
}

// The gradients of entropy_backward's rows: one row at a time, or for float32 work of at most
// entropy_small::kMaxClasses classes a block of rows (entropy_gradient_float's factors per
// row, its halves by column).
template <typename scalar_t>
struct EntropyGradients {
  const scalar_t* input;
  scalar_t* result;
  const int64_t* labels;
  const double *up, *s;
  const std::vector<double>& weights;
  double divisor;
  int64_t reduction, ignore_index, c;
  bool work32;

  // Row i's factor: its upstream derivative and class weight over the mean's divisor.
  double scale(int64_t i) const {
    const double w = weights.empty() ? 1.0 : weights[labels[i]];
    return up[reduction == 0 ? i : 0] * w / divisor;
  }

  // Row i: zero where ignored, float32 work where its factor fits, else double.
  void row(int64_t i, std::vector<double>& wide, std::vector<float>& narrow) const {
    const int64_t target = labels[i];
    scalar_t* g = result + i * c;
    if (target == ignore_index) {
      std::fill(g, g + c, scalar_t(0));
      return;
    }
    const double factor = scale(i);
    if (work32 &&
        entropy_gradient_float(input + i * c, c, target, s + 3 * i, factor, narrow.data(), g))
      return;
    wide.resize(c);
    convert_to_double(input + i * c, wide.data(), c);
    entropy_gradient_double(wide.data(), c, target, s + 3 * i, factor);
    at::vec::convert(wide.data(), g, c);
  }

  // Rows [i, i + count) (count <= lanes::L) a block at a time through col (c lanes::L
  // floats); rows the block leaves (ignored, factors beyond float32) take row.
  void block(int64_t i, int64_t count, float* col, std::vector<double>& wide,
             std::vector<float>& narrow) const {
    const lanes::vi target =
        entropy_small::transpose(input + i * c, labels + i, ignore_index, count, c, col);
    lanes::vf center{}, root{}, sign{};
    double mass[lanes::L], factor[lanes::L], row_scale[lanes::L];
    bool ok[lanes::L];
    for (int64_t r = 0; r < count; ++r) row_scale[r] = target[r] >= 0 ? scale(i + r) : 0.0;
    entropy_small::factors(s + 3 * i, count, row_scale, mass, factor);
    for (int64_t r = 0; r < count; ++r) {
      ok[r] = target[r] >= 0 && factor[r] <= std::numeric_limits<float>::max();
      center[r] = static_cast<float>(s[3 * (i + r)]);
      root[r] = static_cast<float>(std::sqrt(factor[r]));
      sign[r] = row_scale[r] < 0 ? -1.0f : 1.0f;
    }
    entropy_small::halves(col, count, c, center, root, sign, ok, result + i * c);
    for (int64_t r = 0; r < count; ++r) {
      if (ok[r])
        result[(i + r) * c + target[r]] = static_cast<scalar_t>(-mass[r] * row_scale[r]);
      else
        row(i + r, wide, narrow);
    }
  }

  // The rows [begin, end) of a task.
  void rows(int64_t begin, int64_t end) const {
    std::vector<double> wide;
    std::vector<float> narrow(work32 && !std::is_same_v<scalar_t, float> ? c : 0);
    if (!work32 || c > entropy_small::kMaxClasses) {
      for (int64_t i = begin; i < end; ++i) row(i, wide, narrow);
      return;
    }
    std::vector<float> col(c * lanes::L);
    for (int64_t i = begin; i < end; i += lanes::L)
      block(i, std::min<int64_t>(lanes::L, end - i), col.data(), wide, narrow);
  }
};

// Gradient of entropy_forward's loss times upstream (a scalar for mean and sum, (N,) for
// none; any float dtype), rounded once into the logits dtype. stats, reduction, ignore_index,
// weight and fp32 as in that forward call; ignored rows get zero gradients. out, if given (a
// tensor like z, z itself included), receives the gradient: each row is read before it is
// written.
at::Tensor entropy_backward(const at::Tensor& z, const at::Tensor& y, const at::Tensor& stats,
                            const at::Tensor& upstream, int64_t reduction, int64_t ignore_index,
                            const std::optional<at::Tensor>& weight, bool fp32,
                            const std::optional<at::Tensor>& out) {
  check_entropy_inputs(z, y, reduction);
  const int64_t n = z.size(0), c = z.size(1);
  TORCH_CHECK_VALUE(
      stats.scalar_type() == at::kDouble && stats.is_contiguous() && stats.numel() == 3 * n + 1,
      "invalid entropy statistics");
  TORCH_CHECK_VALUE(upstream.numel() == (reduction == 0 ? n : 1), "invalid upstream gradient");
  const bool into = out.has_value() && out->defined();
  TORCH_CHECK_VALUE(!into || (out->sizes() == z.sizes() && out->scalar_type() == z.scalar_type() &&
                              out->is_cpu() && out->is_contiguous()),
                    "out must be a contiguous CPU tensor of the logits' shape and dtype");
  const std::vector<double> weights = class_weights(weight, c);
  const bool work32 = fp32 && z.scalar_type() != at::kDouble;
  auto output = into ? *out : gradient_like(z);
  const std::vector<double> u = double_values(upstream);
  const double* up = u.data();
  const double* s = stats.data_ptr<double>();
  const double divisor = s[3 * n];
  const int64_t* labels = y.data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, z.scalar_type(), "sdca_entropy_backward", [&] {
        const EntropyGradients<scalar_t> gradients{z.data_ptr<scalar_t>(),
                                                   output.data_ptr<scalar_t>(),
                                                   labels,
                                                   up,
                                                   s,
                                                   weights,
                                                   divisor,
                                                   reduction,
                                                   ignore_index,
                                                   c,
                                                   work32};
        at::parallel_for(0, n, grain(n, c),
                         [&](int64_t begin, int64_t end) { gradients.rows(begin, end); });
      });
  return output;
}

// float64 integer entropy of rows of at most entropy64::kMaxClasses classes, as the native
// library's sdca_cross_entropy_forward and _backward (src/tensor_losses.h) compute them: the
// same row statistics (maximum, lowest, compensated sum, mass, tiny flag), values times the
// class weight, compensated totals and gradients (times the upstream factor and the weight
// before the one rounding), in their layout, so the torch layer's checks and log-domain
// fallbacks for double's extremes (_cpu._native_entropy, _cpu._beyond_double) apply as they
// do to the native library's. A block of rows at a time on double lanes (exponentials and
// logarithms within a few ulps of libm's); rows of extreme spreads take the native per-row
// arithmetic.
namespace entropy64 {
constexpr int64_t kMaxClasses = 32;
constexpr int64_t kStats = 5;  // maximum, lowest, sum, mass, tiny
constexpr int D = dlanes::L;

struct Row {
  double maximum, lowest, sum, mass, loss;
};

// entropy_statistics of tensor_losses.h: the loss softplus(relative) and the mass
// sigmoid(relative) share an exponential.
inline Row statistics(double maximum, double lowest, double sum, double target) {
  Row row{maximum, lowest, sum, 0.0, 0.0};
  const double gap = maximum - target, relative = gap + std::log(sum);
  if (relative < -36) {
    const double half = std::exp(gap / 2), tail = half * (half * sum);
    row.loss = std::log1p(tail);
    row.mass = tail / (1 + tail);
  } else {
    const double e = std::exp(-std::fabs(relative)), log = std::log1p(e);
    row.loss = relative > 0 ? relative + log : log;
    row.mass = (relative > 0 ? 1.0 : e) / (1 + e);
  }
  return row;
}

// row_tiny of tensor_losses.h: whether an unscaled derivative is below double's normal range.
inline bool tiny(const Row& r) {
  if (r.lowest - r.maximum > -600 && r.mass / r.sum > 1e-30) return false;
  const double least = std::numeric_limits<double>::min();
  return r.mass < least || std::exp(r.lowest - r.maximum) * (r.mass / r.sum) < least;
}

// short_double_row of tensor_losses.h for a valid label; false for nonfinite logits.
inline bool scalar_row(const double* z, int64_t c, int64_t label, Row& row) {
  const double target = z[label];
  double high = -std::numeric_limits<double>::infinity(), low = -high, bad = target - target;
  for (int64_t j = 0; j < c; ++j) {
    if (j == label) continue;
    high = z[j] > high ? z[j] : high;
    low = z[j] < low ? z[j] : low;
    bad += z[j] - z[j];
  }
  if (!(bad == 0)) return false;
  const bool normal = low - high >= -708;
  double sum = 0, carry = 0;
  for (int64_t j = 0; j < c; ++j) {
    if (j == label) continue;
    const double x = z[j] - high;
    const double e = (normal ? dlanes::exp(dlanes::splat(x))[0] : std::exp(x)) - carry;
    const double t = sum + e;
    carry = (t - sum) - e;
    sum = t;
  }
  row = statistics(high, low, sum, target);
  return true;
}

// The rows [0, count) (count <= D) of z by column into col (c D doubles; missing rows
// repeat the last) and their labels, -1 where ignored or invalid.
inline dlanes::vl transpose(const double* z, const int64_t* labels, int64_t ignore, int64_t count,
                            int64_t c, double* col) {
  dlanes::vl target;
  for (int r = 0; r < D; ++r) {
    const int64_t row = r < count ? r : count - 1, label = labels[row];
    target[r] = label != ignore && label >= 0 && label < c ? label : -1;
    for (int64_t j = 0; j < c; ++j) col[j * D + r] = z[row * c + j];
  }
  return target;
}

// The block's rows into rows; ok[r] false where the per-row arithmetic takes row r (ignored
// or invalid labels, nonfinite logits, spreads beyond -708).
inline void block(const double* z, const int64_t* labels, int64_t ignore, int64_t count, int64_t c,
                  double* col, Row* rows, bool* ok) {
  using namespace dlanes;
  constexpr double kInf = std::numeric_limits<double>::infinity();
  const vl target = transpose(z, labels, ignore, count, c, col);
  vd high = splat(-kInf), low = splat(kInf), label{}, bad{};
  for (int64_t j = 0; j < c; ++j) {
    const vd v = load(col + j * D);
    const vl at = target == j;
    bad = bad + (v - v);  // NaN for inf and NaN entries
    label = select(at, v, label);
    high = select(~at & (v > high), v, high);
    low = select(~at & (v < low), v, low);
  }
  vd sum{}, carry{};
  for (int64_t j = 0; j < c; ++j) {
    const vd e = exp(load(col + j * D) - high) - carry, t = sum + e;
    const vl at = target == j;
    carry = select(at, carry, (t - sum) - e);
    sum = select(at, sum, t);
  }
  // statistics on lanes: log(sum) = log1p(sum - 1), exact for sum >= 1.
  const vd gap = high - label, relative = gap + log1p(sum - 1.0);
  const vd e = exp(-abs(relative)), log = log1p(e);
  const vl far = relative < -36.0, above = relative > 0.0;
  vd loss = select(above, relative + log, log), mass = select(above, splat(1.0), e) / (1.0 + e);
  if (any_lane(far)) {
    const vd half = exp(gap * 0.5), tail = half * (half * sum);
    loss = select(far, log1p(tail), loss);
    mass = select(far, tail / (1.0 + tail), mass);
  }
  const vl normal = (low - high) >= -708.0;
  for (int r = 0; r < count; ++r) {
    ok[r] = target[r] >= 0 && bad[r] == 0.0 && normal[r] != 0;
    rows[r] = {high[r], low[r], sum[r], mass[r], loss[r]};
  }
}

// The rows of sdca_cross_entropy_forward: (values (n), saved): values[i] = weight[y_i] loss_i
// (0 where ignored); saved the (n, kStats) statistics when need (zero where ignored), then the
// totals of the values and of the weights of the rows not ignored, compensated in row order.
std::tuple<at::Tensor, at::Tensor> forward(const at::Tensor& z, const at::Tensor& y, int64_t ignore,
                                           const std::optional<at::Tensor>& weight, bool need) {
  check_entropy_inputs(z, y, 0);
  TORCH_CHECK_VALUE(z.scalar_type() == at::kDouble && z.size(1) <= kMaxClasses,
                    "float64 logits of at most 32 classes");
  const int64_t n = z.size(0), c = z.size(1);
  const std::vector<double> weights = class_weights(weight, c);
  auto values = at::empty({n}, z.options());
  auto saved = at::empty({need ? n * kStats + 2 : 2}, z.options());
  std::fill_n(saved.data_ptr<double>(), saved.numel(), 0.0);
  double *v = values.data_ptr<double>(), *s = need ? saved.data_ptr<double>() : nullptr;
  const double* data = z.data_ptr<double>();
  const int64_t* labels = y.data_ptr<int64_t>();
  RowErrors errors;
  at::parallel_for(0, n, grain(n, c), [&](int64_t begin, int64_t end) {
    std::vector<double> col(c * D);
    auto store = [&](int64_t i, const Row& r) {
      v[i] = (weights.empty() ? 1.0 : weights[labels[i]]) * r.loss;
      if (!s) return;
      double* row = s + kStats * i;
      row[0] = r.maximum, row[1] = r.lowest, row[2] = r.sum, row[3] = r.mass;
      row[4] = tiny(r) ? 1 : 0;
    };
    auto row_loss = [&](int64_t i) {
      v[i] = 0;
      if (labels[i] == ignore) return;
      Row r;
      if (checked_row(data + i * c, c, labels[i], errors,
                      [&] { return scalar_row(data + i * c, c, labels[i], r); }))
        store(i, r);
    };
    for (int64_t i = begin; i < end; i += D) {
      const int64_t count = std::min<int64_t>(D, end - i);
      Row rows[D];
      bool ok[D];
      block(data + i * c, labels + i, ignore, count, c, col.data(), rows, ok);
      for (int64_t r = 0; r < count; ++r) ok[r] ? store(i + r, rows[r]) : row_loss(i + r);
    }
  });
  errors.raise();
  double sum[2] = {0, 0}, carry[2] = {0, 0};
  for (int64_t i = 0; i < n; ++i) {
    if (labels[i] == ignore) continue;
    const double term[2] = {v[i], weights.empty() ? 1.0 : weights[labels[i]]};
    for (int t = 0; t < 2; ++t) {
      const double x = term[t] - carry[t], total = sum[t] + x, back = total - sum[t];
      carry[t] = -((sum[t] - (total - back)) + (x - back));
      sum[t] = total;
    }
  }
  double* totals = saved.data_ptr<double>() + saved.numel() - 2;
  totals[0] = sum[0] - carry[0];
  totals[1] = sum[1] - carry[1];
  return {values, saved};
}

// The upstream factor of each row as the native library's backward takes it: float64
// upstream ((N,) for reduction none, else a scalar) over the weight total for mean.
struct Scale {
  std::vector<double> upstream;
  const double* up;
  int64_t stride;
  double divisor;
  Scale(const at::Tensor& upstream_in, int64_t reduction, const at::Tensor& saved)
      : upstream(double_values(upstream_in)),
        up(upstream.data()),
        stride(upstream.size() == 1 ? 0 : 1),
        divisor(reduction == 1 ? saved.data_ptr<double>()[saved.numel() - 1] : 1.0) {}
  double operator[](int64_t i) const { return up[i * stride] / divisor; }
};

// The gradients of backward's rows: a block of rows at a time on double lanes; rows the
// block leaves (ignored, mass / sum below double's normal range, spreads beyond -708) take
// single_entropy_gradient per row.
struct Gradients {
  const double *data, *st;
  const int64_t* labels;
  double* result;
  const Scale& scale;
  const std::vector<double>& weights;
  int64_t ignore, c;

  // The factor of row i of a valid label: its upstream factor times the class weight.
  double factor(int64_t i, int64_t label) const {
    return scale[i] * (weights.empty() ? 1.0 : weights[label]);
  }

  // single_entropy_gradient of row i: one factor where mass / sum is normal, else dividing
  // first; zero where ignored.
  void row(int64_t i) const {
    double* g = result + i * c;
    const int64_t label = labels[i];
    if (label == ignore) {
      std::fill(g, g + c, 0.0);
      return;
    }
    const double f = factor(i, label);
    const double *z = data + i * c, *s = st + kStats * i;
    const double maximum = s[0], sum = s[2], mass = s[3], ratio = mass / sum;
    const bool normal = s[1] - maximum >= -708;
    for (int64_t j = 0; j < c; ++j) {
      const double x = z[j] - maximum;
      const double e = normal || x >= -708 ? dlanes::exp(dlanes::splat(x))[0] : std::exp(x);
      g[j] = ratio >= std::numeric_limits<double>::min() ? (e * ratio) * f : (mass * (e / sum)) * f;
    }
    g[label] = -mass * f;
  }

  // Rows [i, i + count) (count <= D) through col (c D doubles).
  void block(int64_t i, int64_t count, double* col) const {
    using namespace dlanes;
    const vl target = transpose(data + i * c, labels + i, ignore, count, c, col);
    vd shift{}, ratio{}, scaled{};
    bool ok[D];
    for (int64_t r = 0; r < count; ++r) {
      const double* s = st + kStats * (i + r);
      const int64_t label = target[r];
      const double rr = s[3] / s[2];
      ok[r] = label >= 0 && rr >= std::numeric_limits<double>::min() && s[1] - s[0] >= -708;
      if (!ok[r]) continue;
      shift[r] = s[0], ratio[r] = rr;
      scaled[r] = factor(i + r, label);
    }
    for (int64_t j = 0; j < c; ++j)
      store(col + j * D, (exp(load(col + j * D) - shift) * ratio) * scaled);
    for (int64_t r = 0; r < count; ++r) {
      if (!ok[r]) {
        row(i + r);
        continue;
      }
      double* g = result + (i + r) * c;
      for (int64_t j = 0; j < c; ++j) g[j] = col[j * D + r];
      g[target[r]] = -st[kStats * (i + r) + 3] * scaled[r];
    }
  }
};

// sdca_cross_entropy_backward of forward's saved statistics times the upstream factor (Scale)
// and the class weight: single_entropy_gradient of sdca/math/entropy.h per row, zero rows
// where ignored.
at::Tensor backward(const at::Tensor& z, const at::Tensor& y, const at::Tensor& saved,
                    const at::Tensor& upstream, int64_t reduction, int64_t ignore,
                    const std::optional<at::Tensor>& weight) {
  check_entropy_inputs(z, y, reduction);
  const int64_t n = z.size(0), c = z.size(1);
  TORCH_CHECK_VALUE(z.scalar_type() == at::kDouble && saved.scalar_type() == at::kDouble &&
                        saved.is_contiguous() && saved.numel() == n * kStats + 2 &&
                        upstream.numel() == (reduction == 0 ? n : 1),
                    "invalid float64 entropy statistics or upstream gradient");
  const std::vector<double> weights = class_weights(weight, c);
  auto output = at::empty_like(z, at::MemoryFormat::Contiguous);
  const double *data = z.data_ptr<double>(), *st = saved.data_ptr<double>();
  const Scale scale(upstream, reduction, saved);
  const int64_t* labels = y.data_ptr<int64_t>();
  double* result = output.data_ptr<double>();
  const Gradients gradients{data, st, labels, result, scale, weights, ignore, c};
  at::parallel_for(0, n, grain(n, c), [&](int64_t begin, int64_t end) {
    std::vector<double> col(c * D);
    for (int64_t i = begin; i < end; i += D)
      gradients.block(i, std::min<int64_t>(D, end - i), col.data());
  });
  return output;
}
}  // namespace entropy64

}  // namespace

#endif  // SDCA_TORCH_CE_H
