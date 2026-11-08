#ifndef SDCA_MATH_EXP_ROWS_H
#define SDCA_MATH_EXP_ROWS_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

// Runtime ISA dispatch of the row kernels (GCC, Clang >= 14 on glibc x86-64).
// The kernels use separate IEEE operations only: every clone gives the same bits.
// Not under ThreadSanitizer: the loader runs the clones' instrumented resolvers
// before the TSan runtime starts, and they crash the program at load.
#if defined(__SANITIZE_THREAD__)
#define SDCA_ROW_NO_CLONES 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define SDCA_ROW_NO_CLONES 1
#endif
#endif
#if defined(__x86_64__) && defined(__linux__) && defined(__GLIBC__) && defined(__has_attribute) && \
    !defined(SDCA_ROW_NO_CLONES)
#if __has_attribute(target_clones)
#define SDCA_ROW_KERNEL static inline __attribute__((target_clones("avx512f", "avx2", "default")))
#endif
#endif
#ifndef SDCA_ROW_KERNEL
#define SDCA_ROW_KERNEL static inline
#endif
#undef SDCA_ROW_NO_CLONES
// Helpers inline into each clone, compiled for its ISA.
#if defined(__GNUC__) || defined(__clang__)
#define SDCA_ROW_INLINE inline __attribute__((always_inline))
#else
#define SDCA_ROW_INLINE inline
#endif

namespace sdca {

namespace detail {

// Lanes of the row kernels, fixed so that results do not depend on the ISA.
const std::size_t row_lanes = 32;

// The bits of a double and back.
#if defined(__has_builtin)
#if __has_builtin(__builtin_bit_cast)
#define SDCA_BIT_CAST(To, x) __builtin_bit_cast(To, x)
#endif
#endif
SDCA_ROW_INLINE std::uint64_t double_bits(double x) {
#ifdef SDCA_BIT_CAST
  return SDCA_BIT_CAST(std::uint64_t, x);
#else
  std::uint64_t bits;
  std::memcpy(&bits, &x, sizeof bits);
  return bits;
#endif
}
SDCA_ROW_INLINE double bits_double(std::uint64_t bits) {
#ifdef SDCA_BIT_CAST
  return SDCA_BIT_CAST(double, bits);
#else
  double x;
  std::memcpy(&x, &bits, sizeof x);
  return x;
#endif
}

// exp(x) for x >= -708 (a normal result), within an ulp: x = n ln2 + r
// (Cody-Waite), Taylor's polynomial of degree 13 in |r| <= ln2/2, times 2^n.
SDCA_ROW_INLINE double exp_normal(double x) {
  const double shift = 6755399441055744.0;  // 1.5 * 2^52: adding it rounds to an integer
  const double t = x * 1.4426950408889634074 + shift, n = t - shift;
  const double r = (x - n * 6.93147180369123816490e-01) - n * 1.90821492927058770002e-10;
  double p = 1.0 / 6227020800.0;
  p = p * r + 1.0 / 479001600.0;
  p = p * r + 1.0 / 39916800.0;
  p = p * r + 1.0 / 3628800.0;
  p = p * r + 1.0 / 362880.0;
  p = p * r + 1.0 / 40320.0;
  p = p * r + 1.0 / 5040.0;
  p = p * r + 1.0 / 720.0;
  p = p * r + 1.0 / 120.0;
  p = p * r + 1.0 / 24.0;
  p = p * r + 1.0 / 6.0;
  p = p * r + 0.5;
  p = p * r + 1.0;
  p = p * r + 1.0;
  // The low bits of t hold n; 2^n is its biased exponent.
  return p * bits_double((double_bits(t) - double_bits(shift) + 1023) << 52);
}

// p as a Work buffer if it is one, else null.
template <typename Work, typename Data>
struct same_type_buffer {
  static Work* of(Data*) { return nullptr; }
};
template <typename Work>
struct same_type_buffer<Work, Work> {
  static Work* of(Work* p) { return p; }
};

// Updates the lanes of the extrema of z[0, n).
template <typename Data>
SDCA_ROW_INLINE void extrema_lanes(std::size_t n, const Data* z, Data* high, Data* low) {
  std::size_t j = 0;
  for (; j + row_lanes <= n; j += row_lanes) {
    for (std::size_t l = 0; l < row_lanes; ++l) {
      high[l] = z[j + l] > high[l] ? z[j + l] : high[l];
      low[l] = z[j + l] < low[l] ? z[j + l] : low[l];
    }
  }
  for (std::size_t l = 0; j + l < n; ++l) {
    high[l] = z[j + l] > high[l] ? z[j + l] : high[l];
    low[l] = z[j + l] < low[l] ? z[j + l] : low[l];
  }
}

// The largest z[j], j != skip (n >= 2); low receives the smallest.
template <typename Data>
SDCA_ROW_INLINE Data extrema_except(std::size_t n, const Data* z, std::size_t skip, Data& low) {
  const Data infinity = std::numeric_limits<Data>::infinity();
  Data high_lanes[row_lanes], low_lanes[row_lanes];
  for (std::size_t l = 0; l < row_lanes; ++l) {
    high_lanes[l] = -infinity;
    low_lanes[l] = infinity;
  }
  extrema_lanes(skip < n ? skip : n, z, high_lanes, low_lanes);
  if (skip < n) extrema_lanes(n - skip - 1, z + skip + 1, high_lanes, low_lanes);
  Data high = high_lanes[0];
  low = low_lanes[0];
  for (std::size_t l = 1; l < row_lanes; ++l) {
    high = high_lanes[l] > high ? high_lanes[l] : high;
    low = low_lanes[l] < low ? low_lanes[l] : low;
  }
  return high;
}

SDCA_ROW_KERNEL double extrema_kernel(std::size_t n, const double* z, std::size_t skip,
                                      double& low) {
  return extrema_except(n, z, skip, low);
}

SDCA_ROW_KERNEL float extrema_kernel(std::size_t n, const float* z, std::size_t skip, float& low) {
  return extrema_except(n, z, skip, low);
}

// The largest z[j], j != skip (n >= 2); low receives the smallest.
template <typename Data>
inline Data max_except(std::size_t n, const Data* z, std::size_t skip, Data& low) {
  return extrema_except(n, z, skip, low);
}
inline double max_except(std::size_t n, const double* z, std::size_t skip, double& low) {
  return extrema_kernel(n, z, skip, low);
}
inline float max_except(std::size_t n, const float* z, std::size_t skip, float& low) {
  return extrema_kernel(n, z, skip, low);
}

// extrema_lanes, with bad turning NaN at a nonfinite score (v - v is NaN exactly there).
template <typename Data>
SDCA_ROW_INLINE void finite_extrema_lanes(std::size_t n, const Data* z, Data* high, Data* low,
                                          Data* bad) {
  std::size_t j = 0;
  for (; j + row_lanes <= n; j += row_lanes) {
    for (std::size_t l = 0; l < row_lanes; ++l) {
      high[l] = z[j + l] > high[l] ? z[j + l] : high[l];
      low[l] = z[j + l] < low[l] ? z[j + l] : low[l];
      bad[l] += z[j + l] - z[j + l];
    }
  }
  for (std::size_t l = 0; j + l < n; ++l) {
    high[l] = z[j + l] > high[l] ? z[j + l] : high[l];
    low[l] = z[j + l] < low[l] ? z[j + l] : low[l];
    bad[l] += z[j + l] - z[j + l];
  }
}

// extrema_except (skip < n), and whether every z[j], z[skip] included, is finite.
template <typename Data>
SDCA_ROW_INLINE Data finite_extrema_except(std::size_t n, const Data* z, std::size_t skip,
                                           Data& low, bool& finite) {
  const Data infinity = std::numeric_limits<Data>::infinity();
  Data high_lanes[row_lanes], low_lanes[row_lanes], bad[row_lanes];
  for (std::size_t l = 0; l < row_lanes; ++l) {
    high_lanes[l] = -infinity;
    low_lanes[l] = infinity;
    bad[l] = 0;
  }
  finite_extrema_lanes(skip, z, high_lanes, low_lanes, bad);
  finite_extrema_lanes(n - skip - 1, z + skip + 1, high_lanes, low_lanes, bad);
  Data high = high_lanes[0];
  low = low_lanes[0];
  finite = z[skip] - z[skip] == 0;
  for (std::size_t l = 0; l < row_lanes; ++l) {
    high = high_lanes[l] > high ? high_lanes[l] : high;
    low = low_lanes[l] < low ? low_lanes[l] : low;
    finite = finite && bad[l] == 0;
  }
  return high;
}

SDCA_ROW_KERNEL double finite_extrema_kernel(std::size_t n, const double* z, std::size_t skip,
                                             double& low, bool& finite) {
  return finite_extrema_except(n, z, skip, low, finite);
}

SDCA_ROW_KERNEL float finite_extrema_kernel(std::size_t n, const float* z, std::size_t skip,
                                            float& low, bool& finite) {
  return finite_extrema_except(n, z, skip, low, finite);
}

// The first j0 >= j, j0 = j + m row_lanes, whose block z[j0, j0 + row_lanes) holds an
// entry at least limit; else the first j0 with fewer than row_lanes entries left.
template <typename Data>
SDCA_ROW_INLINE std::size_t reaching_block(std::size_t j, std::size_t n, const Data* z,
                                           Data limit) {
  for (; j + row_lanes <= n; j += row_lanes) {
    int reached = 0;
    for (std::size_t l = 0; l < row_lanes; ++l) reached |= z[j + l] >= limit;
    if (reached) return j;
  }
  return j;
}

SDCA_ROW_KERNEL std::size_t reaching_kernel(std::size_t j, std::size_t n, const double* z,
                                            double limit) {
  return reaching_block(j, n, z, limit);
}

SDCA_ROW_KERNEL std::size_t reaching_kernel(std::size_t j, std::size_t n, const float* z,
                                            float limit) {
  return reaching_block(j, n, z, limit);
}

template <typename Data>
inline std::size_t next_reaching(std::size_t j, std::size_t n, const Data* z, Data limit) {
  return reaching_block(j, n, z, limit);
}
inline std::size_t next_reaching(std::size_t j, std::size_t n, const double* z, double limit) {
  return reaching_kernel(j, n, z, limit);
}
inline std::size_t next_reaching(std::size_t j, std::size_t n, const float* z, float limit) {
  return reaching_kernel(j, n, z, limit);
}

// Adds a block of terms to the compensated (Kahan) sums of the lanes.
SDCA_ROW_INLINE void add_block(const double* e, double* sum, double* carry) {
  for (std::size_t l = 0; l < row_lanes; ++l) {
    const double y = e[l] - carry[l], t = sum[l] + y;
    carry[l] = (t - sum[l]) - y;
    sum[l] = t;
  }
}

// One term: exp_normal where the result is normal, else the library's exp.
SDCA_ROW_INLINE double exp_term(double x, bool normal) {
  return normal || x >= -708.0 ? exp_normal(x) : std::exp(x);
}

// Elements per chunk of exp_sum_lanes: its terms first, then their sums.
const std::size_t row_chunk = 256;

// The lanes of exp_sum: element j in lane j % row_lanes, whatever the path.
template <typename Data>
SDCA_ROW_INLINE double exp_sum_lanes(std::size_t n, const Data* z, std::size_t skip, double shift,
                                     double* out, bool normal) {
  double sum[row_lanes] = {}, carry[row_lanes] = {}, local[row_chunk];
  for (std::size_t begin = 0; begin < n; begin += row_chunk) {
    const std::size_t count = n - begin < row_chunk ? n - begin : row_chunk;
    const Data* x = z + begin;
    double* e = out ? out + begin : local;
    if (normal) {
      for (std::size_t j = 0; j < count; ++j) e[j] = exp_normal(double(x[j]) - shift);
    } else {
      for (std::size_t j = 0; j < count; ++j) e[j] = exp_term(double(x[j]) - shift, false);
    }
    if (skip - begin < count) e[skip - begin] = 0.0;
    std::size_t j = 0;
    for (; j + row_lanes <= count; j += row_lanes) add_block(e + j, sum, carry);
    if (j < count) {
      double last[row_lanes] = {};
      for (std::size_t l = 0; j + l < count; ++l) last[l] = e[j + l];
      add_block(last, sum, carry);
    }
  }
  double total = 0, correction = 0;
  for (std::size_t l = 0; l < 2 * row_lanes; ++l) {
    const double y = (l < row_lanes ? sum[l] : -carry[l - row_lanes]) - correction;
    const double t = total + y;
    correction = (t - total) - y;
    total = t;
  }
  return total;
}

SDCA_ROW_KERNEL double exp_sum_kernel(std::size_t n, const double* z, std::size_t skip,
                                      double shift, double* out, bool normal) {
  return exp_sum_lanes(n, z, skip, shift, out, normal);
}

SDCA_ROW_KERNEL double exp_sum_kernel(std::size_t n, const float* z, std::size_t skip, double shift,
                                      double* out, bool normal) {
  return exp_sum_lanes(n, z, skip, shift, out, normal);
}

// out[j] = Data((exp(z[j] - shift) * ratio) * scale), as exp_sum_lanes.
template <typename Data>
SDCA_ROW_INLINE void exp_scale_lanes(std::size_t n, const Data* z, double shift, double ratio,
                                     double scale, Data* out, bool normal) {
  if (normal) {
    for (std::size_t j = 0; j < n; ++j)
      out[j] = Data((exp_normal(double(z[j]) - shift) * ratio) * scale);
  } else {
    for (std::size_t j = 0; j < n; ++j)
      out[j] = Data((exp_term(double(z[j]) - shift, false) * ratio) * scale);
  }
}

SDCA_ROW_KERNEL void exp_scale_kernel(std::size_t n, const double* z, double shift, double ratio,
                                      double scale, double* out, bool normal) {
  exp_scale_lanes(n, z, shift, ratio, scale, out, normal);
}

SDCA_ROW_KERNEL void exp_scale_kernel(std::size_t n, const float* z, double shift, double ratio,
                                      double scale, float* out, bool normal) {
  exp_scale_lanes(n, z, shift, ratio, scale, out, normal);
}

// out[j] = Data((exp(z[j] - shift) * ratio) * scale); normal as for exp_sum.
template <typename Work, typename Data>
inline void exp_scale(std::size_t n, const Data* z, Work shift, Work ratio, Work scale, Data* out,
                      bool) {
  for (std::size_t j = 0; j < n; ++j) out[j] = Data((std::exp(Work(z[j]) - shift) * ratio) * scale);
}

inline void exp_scale(std::size_t n, const double* z, double shift, double ratio, double scale,
                      double* out, bool normal) {
  exp_scale_kernel(n, z, shift, ratio, scale, out, normal);
}

inline void exp_scale(std::size_t n, const float* z, double shift, double ratio, double scale,
                      float* out, bool normal) {
  exp_scale_kernel(n, z, shift, ratio, scale, out, normal);
}

// The sum of exp(z[j] - shift) over j != skip, compensated; out[j] receives
// the terms (out[skip] = 0) when given. Double terms take the vector kernel;
// normal says that every term is normal (z[j] - shift >= -708).
template <typename Work, typename Data>
inline Work exp_sum(std::size_t n, const Data* z, std::size_t skip, Work shift, Work* out, bool) {
  Work sum(0), correction(0);
  for (std::size_t j = 0; j < n; ++j) {
    if (j == skip) {
      if (out) out[j] = Work(0);
      continue;
    }
    const Work e = std::exp(Work(z[j]) - shift);
    if (out) out[j] = e;
    Work adjusted = e - correction, next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
  }
  return sum;
}

inline double exp_sum(std::size_t n, const double* z, std::size_t skip, double shift, double* out,
                      bool normal) {
  return exp_sum_kernel(n, z, skip, shift, out, normal);
}

inline double exp_sum(std::size_t n, const float* z, std::size_t skip, double shift, double* out,
                      bool normal) {
  return exp_sum_kernel(n, z, skip, shift, out, normal);
}

}  // namespace detail

}  // namespace sdca

#endif
