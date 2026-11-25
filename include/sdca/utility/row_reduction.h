// One final reduction of double row values, shared by the CPU framework bindings.
#ifndef SDCA_UTILITY_ROW_REDUCTION_H
#define SDCA_UTILITY_ROW_REDUCTION_H

#include <cmath>
#include <cstddef>

#include "sdca/utility/fpenv.h"

namespace sdca {
namespace detail {

// Accumulate independent lanes in a fixed order. Add the rounding residuals for
// a double result; a float result already has 29 guard bits in its double sum.
template <bool Precise>
inline double row_sum(const double* values, std::size_t n, double scale) {
  double hi[8] = {}, lo[8] = {};
  // Lane l takes the values i = l mod 8, whole groups of eight first: the same sums as one
  // loop, in a form the compiler vectorizes (the serial loop was a sixth of a CPU call of
  // 262,144 short rows on 32 threads).
  auto add = [&](std::size_t lane, double value) {
    const double x = value * scale;
    const double sum = hi[lane] + x;
    if (Precise) {
      const double b = sum - hi[lane];
      lo[lane] += (hi[lane] - (sum - b)) + (x - b);
    }
    hi[lane] = sum;
  };
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8)
    for (std::size_t lane = 0; lane < 8; ++lane) add(lane, values[i + lane]);
  for (std::size_t lane = 0; i + lane < n; ++lane) add(lane, values[i + lane]);
  double sum = 0, error = 0;
  for (std::size_t j = 0; j < 8; ++j) {
    const double next = sum + hi[j];
    if (Precise) {
      const double b = next - sum;
      error += lo[j] + ((sum - (next - b)) + (hi[j] - b));
    }
    sum = next;
  }
  return Precise && std::isfinite(sum) ? sum + error : sum;
}

// terms, if supplied, are the non-ignored label weights. Otherwise the mean
// denominator is count. A common power of two recovers finite means whose
// numerator or denominator overflows before division.
template <typename Out>
inline void store_row_values(Out* out, const double* values, const double* terms, std::size_t n,
                             int reduction, double count) {
  const subnormal_environment subnormals;
  if (reduction == 0) {
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<Out>(values[i]);
    return;
  }
  constexpr bool precise = sizeof(Out) == sizeof(double);
  double sum = row_sum<precise>(values, n, 1);
  if (reduction == 2) {
    double divisor = terms ? row_sum<precise>(terms, n, 1) : count;
    if (n && (std::isinf(sum) || std::isinf(divisor))) {
      const double scale = std::ldexp(1.0, -(std::ilogb(static_cast<double>(n)) + 2));
      sum = row_sum<precise>(values, n, scale);
      divisor = terms ? row_sum<precise>(terms, n, scale) : count * scale;
    }
    sum /= divisor;
  }
  *out = static_cast<Out>(sum);
}

// Each mapped example gets the same finalize-and-cast as an individual call.
template <typename Out>
inline void store_row_groups(Out* out, const double* values, const double* terms,
                             std::size_t groups, std::size_t rows, int reduction) {
  for (std::size_t i = 0; i < groups; ++i) {
    store_row_values(out + i, rows ? values + i * rows : values, terms ? terms + i * rows : nullptr,
                     rows, reduction, rows);
  }
}

}  // namespace detail
}  // namespace sdca

#endif
