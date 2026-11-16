#ifndef SDCA_MATH_TENSOR_FLOAT_H
#define SDCA_MATH_TENSOR_FLOAT_H

#include <cstddef>
#include "sdca/math/exp_rows.h"
#include "sdca/math/storage.h"

namespace sdca {
namespace tensor {

// exp(x) in float within about an ulp for x <= 88 (Cody-Waite reduction and
// the polynomial of Cephes' expf); 0 below -87.6, where 2^n leaves the normals.
SDCA_ROW_INLINE float exp_float(float x) {
  const float shift = 12582912.0f;  // 1.5 * 2^23: adding it rounds to an integer
  const float t = x * 1.44269504088896341f + shift, n = t - shift;
  const float r = (x - n * 0.693359375f) - n * -2.12194440e-4f;
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = (p * r) * r + r + 1.0f;
  // The low bits of t hold n; 2^n is its biased exponent.
  const float power = bits_float((float_bits(t) - float_bits(shift) + 127u) << 23);
  return bits_float(select(x < -87.6f, 0u, float_bits(p * power)));
}

// Lanes of the float row kernels; each lane adds at most float_chunk /
// float_lanes float terms before a double lane takes their sum.
const std::size_t float_lanes = 32, float_chunk = 512;

// Updates the lanes of the extrema of z[0, n); bad turns NaN at a nonfinite score.
template <typename Format>
SDCA_ROW_INLINE void extrema_lanes(std::size_t n, const typename Format::type* z, float* high,
                                   float* low, float* bad) {
  std::size_t j = 0;
  for (; j + float_lanes <= n; j += float_lanes) {
    for (std::size_t l = 0; l < float_lanes; ++l) {
      const float v = widen(Format(), z[j + l]);
      high[l] = v > high[l] ? v : high[l];
      low[l] = v < low[l] ? v : low[l];
      bad[l] += v - v;
    }
  }
  for (std::size_t l = 0; j + l < n; ++l) {
    const float v = widen(Format(), z[j + l]);
    high[l] = v > high[l] ? v : high[l];
    low[l] = v < low[l] ? v : low[l];
    bad[l] += v - v;
  }
}

}  // namespace tensor
}  // namespace sdca
#endif
