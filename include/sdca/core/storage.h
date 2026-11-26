#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_STORAGE_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_STORAGE_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_STORAGE_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_STORAGE_H3 1
#else
#define SDCA_CORE_STORAGE_H4 1
#endif

// Storage types of scores and gradients (float, double, and the 16-bit half and bfloat16 as
// raw bits) and their lane loads and stores: exact widening, rounding to nearest even once.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "lanes.h"

SDCA_ISA_BEGIN
namespace core {

namespace detail {

SDCA_LANE uint32_t pick(bool c, uint32_t a, uint32_t b) {
  const uint32_t mask = 0u - static_cast<uint32_t>(c);
  return (a & mask) | (b & ~mask);
}
SDCA_LANE float widen_half(uint16_t h) {
#if defined(__aarch64__)
  __fp16 x;
  std::memcpy(&x, &h, sizeof x);
  return static_cast<float>(x);
#else
  const uint32_t sign = (h & 0x8000u) << 16, em = h & 0x7fffu;
  const float subnormal = float(static_cast<int32_t>(em)) * 5.9604644775390625e-8f;
  const uint32_t normal = (em << 13) + (112u << 23), special = (em << 13) | 0x7f800000u;
  const uint32_t b =
      pick(em < 0x400u, bit_cast<uint32_t>(subnormal), pick(em >= 0x7c00u, special, normal));
  return bit_cast<float>(b | sign);
#endif
}
SDCA_LANE uint16_t narrow_half(float x) {
#if defined(__aarch64__)
  const __fp16 h = static_cast<__fp16>(x);
  return bit_cast<uint16_t>(h);
#else
  const uint32_t b = bit_cast<uint32_t>(x), sign = (b >> 16) & 0x8000u, a = b & 0x7fffffffu;
  const uint32_t big = pick(a > 0x7f800000u, 0x7e00u | ((a >> 13) & 0x3ffu), 0x7c00u);
  // Below 2^-14 the float sum with 0.5 rounds to a multiple of 2^-24 (the subnormal unit).
  const uint32_t small = bit_cast<uint32_t>(bit_cast<float>(a) + 0.5f) - 0x3f000000u;
  const uint32_t normal = (a - (112u << 23) + 0xfffu + ((a >> 13) & 1u)) >> 13;
  const uint32_t h = pick(a >= 0x47800000u, big, pick(a < 0x38800000u, small, normal));
  return static_cast<uint16_t>(h | sign);
#endif
}
SDCA_LANE uint16_t narrow_bfloat16(float x) {
  const uint32_t b = bit_cast<uint32_t>(x);
  const uint32_t rounded = (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
  return static_cast<uint16_t>(pick((b & 0x7fffffffu) > 0x7f800000u, (b >> 16) | 0x40u, rounded));
}
// x rounded to odd into float: a 16-bit format rounds that to nearest even as it would round
// x itself (float has at least two more digits), so x rounds once.
SDCA_LANE float odd_float(double x) {
  const float nearest = static_cast<float>(x);
  const uint32_t b = bit_cast<uint32_t>(nearest) - (std::fabs(double(nearest)) > std::fabs(x));
  return bit_cast<float>(b | (double(bit_cast<float>(b)) != x));
}

}  // namespace detail

// Values of storage S in work type T, and back (rounded once).
template <typename T>
SDCA_LANE T to_work(float x) {
  return static_cast<T>(x);
}
template <typename T>
SDCA_LANE T to_work(double x) {
  return static_cast<T>(x);
}
template <typename T>
SDCA_LANE T to_work(half x) {
  return static_cast<T>(detail::widen_half(x.bits));
}
template <typename T>
SDCA_LANE T to_work(bfloat16 x) {
  return static_cast<T>(detail::bit_cast<float>(uint32_t(x.bits) << 16));
}

template <typename S>
struct Stored;
template <>
struct Stored<float> {
  static SDCA_LANE float from(float x) { return x; }
  static SDCA_LANE float from(double x) { return static_cast<float>(x); }
};
template <>
struct Stored<double> {
  static SDCA_LANE double from(double x) { return x; }
  static SDCA_LANE double from(float x) { return static_cast<double>(x); }
};
template <>
struct Stored<half> {
  static SDCA_LANE half from(float x) {
    half h;
    h.bits = detail::narrow_half(x);
    return h;
  }
  static SDCA_LANE half from(double x) { return from(detail::odd_float(x)); }
};
template <>
struct Stored<bfloat16> {
  static SDCA_LANE bfloat16 from(float x) {
    bfloat16 h;
    h.bits = detail::narrow_bfloat16(x);
    return h;
  }
  static SDCA_LANE bfloat16 from(double x) { return from(detail::odd_float(x)); }
};
template <typename S, typename T>
SDCA_LANE S to_storage(T x) {
  return Stored<S>::from(x);
}

// The quiet NaN of a storage type.
template <typename S>
SDCA_LANE S storage_nan() {
  return to_storage<S>(std::numeric_limits<float>::quiet_NaN());
}

// The entries of x[0, n) above -inf (the classes a row keeps), in order and in work type T,
// into kept: their count.
template <typename T, typename S>
inline std::size_t gather_kept(const S* x, std::size_t n, T* kept) {
  std::size_t m = 0;
  for (std::size_t j = 0; j < n; ++j) {
    const T v = to_work<T>(x[j]);
    if (v > -std::numeric_limits<T>::infinity()) kept[m++] = v;
  }
  return m;
}

// out[j] = scale times the next of solved where x[j] is kept (above -inf), else times
// other(j), each rounded once (+0.0 for zeros). out may be x: x[j] is read before out[j].
template <typename T, typename S, typename G, typename Other>
inline void scatter_kept(const S* x, std::size_t n, const T* solved, T scale, Other other, G* out) {
  for (std::size_t j = 0, k = 0; j < n; ++j) {
    const T v = to_work<T>(x[j]) > -std::numeric_limits<T>::infinity() ? solved[k++] : other(j);
    out[j] = to_storage<G>(v * scale + T(0));
  }
}

namespace detail {

// One vector of T from B entries of S, and back: element by element unless specialized.
template <typename S, typename T, typename Isa>
struct Convert {
  typedef typename Register<T, Isa>::V V;
  static const int B = Register<T, Isa>::L;
  static SDCA_LANE void load(V& v, const S* p) {
    T lane[B];
    for (int i = 0; i < B; ++i) lane[i] = to_work<T>(p[i]);
    std::memcpy(&v, lane, sizeof v);
  }
  static SDCA_LANE void store(S* p, const V& v) {
    T lane[B];
    std::memcpy(lane, &v, sizeof v);
    for (int i = 0; i < B; ++i) p[i] = to_storage<S>(lane[i]);
  }
};
template <typename T, typename Isa>
struct Convert<T, T, Isa> {
  typedef typename Register<T, Isa>::V V;
  static SDCA_LANE void load(V& v, const T* p) { std::memcpy(&v, p, sizeof v); }
  static SDCA_LANE void store(T* p, const V& v) { std::memcpy(p, &v, sizeof v); }
};
template <typename Isa>
struct Convert<float, double, Isa> {
  typedef typename Register<double, Isa>::V V;
  static const int B = Register<double, Isa>::L;
  typedef typename Vector<float, B>::type F;
  static SDCA_LANE void load(V& v, const float* p) {
    F f;
    std::memcpy(&f, p, sizeof f);
    v = __builtin_convertvector(f, V);
  }
  static SDCA_LANE void store(float* p, const V& v) {
    const F f = __builtin_convertvector(v, F);
    std::memcpy(p, &f, sizeof f);
  }
};
template <typename Isa>
struct Convert<bfloat16, float, Isa> {
  typedef typename Register<float, Isa>::V V;
  typedef typename Register<float, Isa>::U U;
  static const int B = Register<float, Isa>::L;
  typedef typename Vector<uint16_t, B>::type H;
  static SDCA_LANE void load(V& v, const bfloat16* p) {
    H h;
    std::memcpy(&h, p, sizeof h);
    v = (V)(__builtin_convertvector(h, U) << 16);
  }
  static SDCA_LANE void store(bfloat16* p, const V& v) {
    const U u = (U)v;
    const U rounded = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;
    const U nan = (U)((u & 0x7fffffffu) > 0x7f800000u);
    const H h = __builtin_convertvector((nan & ((u >> 16) | 0x40u)) | (~nan & rounded), H);
    std::memcpy(p, &h, sizeof h);
  }
};
#if defined(SDCA_ISA_AVX2) || defined(SDCA_ISA_AVX512)
template <typename Isa>
struct ConvertHalfF16C {
  typedef typename Register<float, Isa>::V V;
  static const int B = Register<float, Isa>::L;
  static SDCA_LANE void load(V& v, const half* p) {
    for (int k = 0; k < B; k += 8) {
      const __m256 x = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + k)));
      std::memcpy(reinterpret_cast<float*>(&v) + k, &x, sizeof x);
    }
  }
  static SDCA_LANE void store(half* p, const V& v) {
    for (int k = 0; k < B; k += 8) {
      __m256 x;
      std::memcpy(&x, reinterpret_cast<const float*>(&v) + k, sizeof x);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p + k),
                       _mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT));
    }
  }
};
#if defined(SDCA_ISA_AVX2)
template <>
struct Convert<half, float, isa::avx2> : ConvertHalfF16C<isa::avx2> {};
#endif
#if defined(SDCA_ISA_AVX512)
template <>
struct Convert<half, float, isa::avx512> : ConvertHalfF16C<isa::avx512> {};
#endif
#endif
#if defined(SDCA_ISA_NEON) && defined(__clang__)
template <>
struct Convert<half, float, isa::neon> {
  typedef Register<float, isa::neon>::V V;
  static const int B = Register<float, isa::neon>::L;
  typedef __fp16 H __attribute__((vector_size(2 * B)));
  static SDCA_LANE void load(V& v, const half* p) {
    H h;
    std::memcpy(&h, p, sizeof h);
    v = __builtin_convertvector(h, V);
  }
  static SDCA_LANE void store(half* p, const V& v) {
    const H h = __builtin_convertvector(v, H);
    std::memcpy(p, &h, sizeof h);
  }
};
#endif

}  // namespace detail

// The L lanes of p of storage S as lanes A, and the first count of them (pad elsewhere).
template <typename A, typename S>
SDCA_LANE A load(const S* p) {
  typedef detail::Convert<S, typename A::value_type, typename A::isa_type> C;
  A r;
  for (int k = 0; k < A::chains; ++k) C::load(r.v[k], p + k * A::B);
  return r;
}
namespace detail {
template <typename A, typename S>
SDCA_LANE A load_tail(const S* p, int count, typename A::value_type pad, std::false_type) {
  typedef typename A::value_type T;
  T lane[A::L];
  for (int i = 0; i < A::L; ++i) lane[i] = i < count ? to_work<T>(p[i]) : pad;
  return A::load(lane);
}
template <typename A>
SDCA_LANE A load_tail(const typename A::value_type* p, int count, typename A::value_type pad,
                      std::true_type) {
  return A::load(p, count, pad);
}
template <typename S, typename T, typename Isa, int K>
SDCA_LANE void store_tail(S* p, const Lanes<T, Isa, K>& a, int count) {
  for (int i = 0; i < count; ++i) p[i] = to_storage<S>(a[i]);
}
template <typename T, typename Isa, int K>
SDCA_LANE void store_tail(T* p, const Lanes<T, Isa, K>& a, int count) {
  a.store(p, count);
}
}  // namespace detail

template <typename A, typename S>
SDCA_LANE A load(const S* p, int count, typename A::value_type pad) {
  if (count >= A::L) return load<A>(p);
  return detail::load_tail<A>(p, count, pad, std::is_same<S, typename A::value_type>());
}
// The lanes rounded once into storage S: all L, or the first count.
template <typename S, typename T, typename Isa, int K>
SDCA_LANE void store(S* p, const Lanes<T, Isa, K>& a) {
  typedef detail::Convert<S, T, Isa> C;
  for (int k = 0; k < K; ++k) C::store(p + k * Lanes<T, Isa, K>::B, a.v[k]);
}
template <typename S, typename T, typename Isa, int K>
SDCA_LANE void store(S* p, const Lanes<T, Isa, K>& a, int count) {
  if (count >= Lanes<T, Isa, K>::L) return store(p, a);
  detail::store_tail(p, a, count);
}

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_STORAGE_H
