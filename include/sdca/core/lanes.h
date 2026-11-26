#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_LANES_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_LANES_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_LANES_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_LANES_H3 1
#else
#define SDCA_CORE_LANES_H4 1
#endif

// Fixed-width lanes of float or double in GCC/Clang vector extensions: NEON, AVX2 or
// AVX-512 (sdca/core/isa.h), or one lane per vector (isa::scalar).
// Lanes<T, Isa, K> holds K vectors taken as one: K independent dependency chains.
// Arithmetic is never contracted; fma() fuses where the instruction set has it.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

SDCA_ISA_BEGIN
namespace core {

namespace isa {
struct scalar {};
struct avx2 {};
struct avx512 {};
struct neon {};
#if SDCA_ISA_ID == 4
typedef avx512 native;
#elif SDCA_ISA_ID == 3
typedef avx2 native;
#elif SDCA_ISA_ID == 2
typedef neon native;
#else
typedef scalar native;
#endif
}  // namespace isa

template <typename T>
struct IntOf;
template <>
struct IntOf<float> {
  typedef int32_t type;
  typedef uint32_t unsigned_type;
};
template <>
struct IntOf<double> {
  typedef int64_t type;
  typedef uint64_t unsigned_type;
};

template <typename T, int N>
struct Vector {
  typedef T type __attribute__((vector_size(sizeof(T) * N)));
};

// Lanes of T per vector. Clang splits vectors wider than the registers well (four NEON
// registers of floats); GCC scalarizes comparisons wider than the registers, so it takes
// their width.
template <typename T, typename Isa>
struct Width;
template <typename T>
struct Width<T, isa::scalar> {
  static const int value = 1;
};
#if defined(SDCA_ISA_AVX2)
template <typename T>
struct Width<T, isa::avx2> {
#if defined(__clang__)
  static const int value = sizeof(T) == 4 ? 16 : 4;
#else
  static const int value = 32 / sizeof(T);
#endif
};
#endif
#if defined(SDCA_ISA_AVX512)
template <typename T>
struct Width<T, isa::avx512> {
  static const int value = 64 / sizeof(T);
};
#endif
#if defined(SDCA_ISA_NEON)
template <typename T>
struct Width<T, isa::neon> {
#if defined(__clang__)
  static const int value = sizeof(T) == 4 ? 16 : 4;
#else
  static const int value = 16 / sizeof(T);
#endif
};
#endif

// One vector: its lanes V, their integers I (and masks: all ones or zero), unsigned U.
template <typename T, typename Isa>
struct Register {
  static const int L = Width<T, Isa>::value;
  typedef typename Vector<T, L>::type V;
  typedef typename Vector<typename IntOf<T>::type, L>::type I;
  typedef typename Vector<typename IntOf<T>::unsigned_type, L>::type U;
};

namespace detail {

template <typename To, typename From>
SDCA_LANE To bit_cast(const From& x) {
  static_assert(sizeof(To) == sizeof(From), "bit_cast of equal sizes");
  To y;
  std::memcpy(&y, &x, sizeof y);
  return y;
}

// x as the vector type To: converted where the types differ (an intrinsic's vector type and the
// extension vector of the same lanes are distinct types under some compilers), x itself
// where they are one type.
template <typename To, typename From, bool = std::is_same<To, From>::value>
struct VectorAs {
  static SDCA_LANE To cast(const From& x) { return (To)x; }
};
template <typename To, typename From>
struct VectorAs<To, From, true> {
  static SDCA_LANE To cast(const From& x) { return x; }
};
template <typename To, typename From>
SDCA_LANE To as(const From& x) {
  return VectorAs<To, From>::cast(x);
}

// r = select(a > b, a, b) and select(a < b, a, b): b where either is NaN or both are zeros.
// One instruction on x86 where GCC's conditional takes a compare and a blend. Vectors pass by
// reference: a vector wider than the pass's registers (clang's 16 floats under AVX2) would
// change the ABI by value. r may alias a or b.
template <typename Isa, typename V>
SDCA_LANE void max(V& r, const V& a, const V& b) {
  r = a > b ? a : b;
}
template <typename Isa, typename V>
SDCA_LANE void min(V& r, const V& a, const V& b) {
  r = a < b ? a : b;
}
// r = a * b + c, fused where the instruction set has it.
template <typename Isa, typename V>
SDCA_LANE void fma(V& r, const V& a, const V& b, const V& c) {
  r = a * b + c;
}

#define SDCA_LANES_INTRINSIC2(NAME, ISA, T, X, CALL)                              \
  template <>                                                                     \
  SDCA_LANE void NAME<ISA>(Register<T, ISA>::V & r, const Register<T, ISA>::V& a, \
                           const Register<T, ISA>::V& b) {                        \
    r = as<Register<T, ISA>::V>(CALL(as<X>(a), as<X>(b)));                        \
  }
#define SDCA_LANES_INTRINSIC_FMA(ISA, T, X, CALL)                                       \
  template <>                                                                           \
  SDCA_LANE void fma<ISA>(Register<T, ISA>::V & r, const Register<T, ISA>::V& a,        \
                          const Register<T, ISA>::V& b, const Register<T, ISA>::V& c) { \
    r = as<Register<T, ISA>::V>(CALL(as<X>(a), as<X>(b), as<X>(c)));                    \
  }
#define SDCA_LANES_CONTRACTED_FMA(ISA, T)                                               \
  template <>                                                                           \
  SDCA_LANE void fma<ISA>(Register<T, ISA>::V & r, const Register<T, ISA>::V& a,        \
                          const Register<T, ISA>::V& b, const Register<T, ISA>::V& c) { \
    _Pragma("clang fp contract(fast)") r = a * b + c;                                   \
  }

#if !defined(__clang__)
#if defined(SDCA_ISA_AVX2)
SDCA_LANES_INTRINSIC2(max, isa::avx2, float, __m256, _mm256_max_ps)
SDCA_LANES_INTRINSIC2(min, isa::avx2, float, __m256, _mm256_min_ps)
SDCA_LANES_INTRINSIC2(max, isa::avx2, double, __m256d, _mm256_max_pd)
SDCA_LANES_INTRINSIC2(min, isa::avx2, double, __m256d, _mm256_min_pd)
SDCA_LANES_INTRINSIC_FMA(isa::avx2, float, __m256, _mm256_fmadd_ps)
SDCA_LANES_INTRINSIC_FMA(isa::avx2, double, __m256d, _mm256_fmadd_pd)
#endif
#if defined(SDCA_ISA_AVX512)
SDCA_LANES_INTRINSIC2(max, isa::avx512, float, __m512, _mm512_max_ps)
SDCA_LANES_INTRINSIC2(min, isa::avx512, float, __m512, _mm512_min_ps)
SDCA_LANES_INTRINSIC2(max, isa::avx512, double, __m512d, _mm512_max_pd)
SDCA_LANES_INTRINSIC2(min, isa::avx512, double, __m512d, _mm512_min_pd)
SDCA_LANES_INTRINSIC_FMA(isa::avx512, float, __m512, _mm512_fmadd_ps)
SDCA_LANES_INTRINSIC_FMA(isa::avx512, double, __m512d, _mm512_fmadd_pd)
#endif
#if defined(SDCA_ISA_NEON)
template <>
SDCA_LANE void fma<isa::neon>(Register<float, isa::neon>::V& r,
                              const Register<float, isa::neon>::V& a,
                              const Register<float, isa::neon>::V& b,
                              const Register<float, isa::neon>::V& c) {
  r = as<Register<float, isa::neon>::V>(
      vfmaq_f32(as<float32x4_t>(c), as<float32x4_t>(a), as<float32x4_t>(b)));
}
template <>
SDCA_LANE void fma<isa::neon>(Register<double, isa::neon>::V& r,
                              const Register<double, isa::neon>::V& a,
                              const Register<double, isa::neon>::V& b,
                              const Register<double, isa::neon>::V& c) {
  r = as<Register<double, isa::neon>::V>(
      vfmaq_f64(as<float64x2_t>(c), as<float64x2_t>(a), as<float64x2_t>(b)));
}
#endif
#else
#if defined(SDCA_ISA_AVX2)
SDCA_LANES_CONTRACTED_FMA(isa::avx2, float)
SDCA_LANES_CONTRACTED_FMA(isa::avx2, double)
#endif
#if defined(SDCA_ISA_AVX512)
SDCA_LANES_CONTRACTED_FMA(isa::avx512, float)
SDCA_LANES_CONTRACTED_FMA(isa::avx512, double)
#endif
#if defined(SDCA_ISA_NEON)
SDCA_LANES_CONTRACTED_FMA(isa::neon, float)
SDCA_LANES_CONTRACTED_FMA(isa::neon, double)
#endif
#endif
#undef SDCA_LANES_INTRINSIC2
#undef SDCA_LANES_INTRINSIC_FMA
#undef SDCA_LANES_CONTRACTED_FMA

// The first count entries of p (none for count <= 0, all for count >= B), pad in the other
// lanes of a vector, and back: masked loads and stores where the instruction set has them.
template <typename T, typename Isa>
struct Tail {
  typedef typename Register<T, Isa>::V V;
  static const int B = Register<T, Isa>::L;
  static SDCA_LANE void load(V& v, const T* p, int count, T pad) {
    T lane[B];
    for (int i = 0; i < B; ++i) lane[i] = i < count ? p[i] : pad;
    std::memcpy(&v, lane, sizeof v);
  }
  static SDCA_LANE void store(T* p, const V& v, int count) {
    T lane[B];
    std::memcpy(lane, &v, sizeof v);
    for (int i = 0; i < count && i < B; ++i) p[i] = lane[i];
  }
};
#if defined(SDCA_ISA_AVX2)
template <>
struct Tail<float, isa::avx2> {
  typedef Register<float, isa::avx2>::V V;
  static const int B = Register<float, isa::avx2>::L;
  static SDCA_LANE __m256i mask(int count) {
    return _mm256_cmpgt_epi32(_mm256_set1_epi32(count), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
  }
  static SDCA_LANE void load(V& v, const float* p, int count, float pad) {
    for (int q = 0; q < B; q += 8) {
      const __m256i m = mask(count - q);
      const __m256 x = _mm256_blendv_ps(_mm256_set1_ps(pad), _mm256_maskload_ps(p + q, m),
                                        _mm256_castsi256_ps(m));
      std::memcpy(reinterpret_cast<float*>(&v) + q, &x, sizeof x);
    }
  }
  static SDCA_LANE void store(float* p, const V& v, int count) {
    for (int q = 0; q < B; q += 8) {
      __m256 x;
      std::memcpy(&x, reinterpret_cast<const float*>(&v) + q, sizeof x);
      _mm256_maskstore_ps(p + q, mask(count - q), x);
    }
  }
};
template <>
struct Tail<double, isa::avx2> {
  typedef Register<double, isa::avx2>::V V;
  static const int B = Register<double, isa::avx2>::L;
  static SDCA_LANE __m256i mask(int count) {
    return _mm256_cmpgt_epi64(_mm256_set1_epi64x(count), _mm256_setr_epi64x(0, 1, 2, 3));
  }
  static SDCA_LANE void load(V& v, const double* p, int count, double pad) {
    for (int q = 0; q < B; q += 4) {
      const __m256i m = mask(count - q);
      const __m256d x = _mm256_blendv_pd(_mm256_set1_pd(pad), _mm256_maskload_pd(p + q, m),
                                         _mm256_castsi256_pd(m));
      std::memcpy(reinterpret_cast<double*>(&v) + q, &x, sizeof x);
    }
  }
  static SDCA_LANE void store(double* p, const V& v, int count) {
    for (int q = 0; q < B; q += 4) {
      __m256d x;
      std::memcpy(&x, reinterpret_cast<const double*>(&v) + q, sizeof x);
      _mm256_maskstore_pd(p + q, mask(count - q), x);
    }
  }
};
#endif
#if defined(SDCA_ISA_AVX512)
template <>
struct Tail<float, isa::avx512> {
  typedef Register<float, isa::avx512>::V V;
  static SDCA_LANE __mmask16 mask(int count) {
    return count >= 16  ? __mmask16(0xffff)
           : count <= 0 ? __mmask16(0)
                        : __mmask16((1u << count) - 1);
  }
  static SDCA_LANE void load(V& v, const float* p, int count, float pad) {
    v = as<V>(_mm512_mask_loadu_ps(_mm512_set1_ps(pad), mask(count), p));
  }
  static SDCA_LANE void store(float* p, const V& v, int count) {
    _mm512_mask_storeu_ps(p, mask(count), as<__m512>(v));
  }
};
template <>
struct Tail<double, isa::avx512> {
  typedef Register<double, isa::avx512>::V V;
  static SDCA_LANE __mmask8 mask(int count) {
    return count >= 8 ? __mmask8(0xff) : count <= 0 ? __mmask8(0) : __mmask8((1u << count) - 1);
  }
  static SDCA_LANE void load(V& v, const double* p, int count, double pad) {
    v = as<V>(_mm512_mask_loadu_pd(_mm512_set1_pd(pad), mask(count), p));
  }
  static SDCA_LANE void store(double* p, const V& v, int count) {
    _mm512_mask_storeu_pd(p, mask(count), as<__m512d>(v));
  }
};
#endif

}  // namespace detail

template <typename T, typename Isa, int K = 1>
struct Ints;

// K vectors of lanes of T. Lane i of the whole is lane i % B of vector i / B.
template <typename T, typename Isa, int K = 1>
struct Lanes {
  typedef T value_type;
  typedef Isa isa_type;
  typedef Register<T, Isa> R;
  typedef typename R::V V;
  typedef Ints<T, Isa, K> mask_type;
  static const int B = R::L;  // lanes per vector
  static const int L = B * K;
  static const int chains = K;
  V v[K];

  static SDCA_LANE Lanes splat(T x) {
    Lanes r;
    for (int k = 0; k < K; ++k) r.v[k] = V{} + x;
    return r;
  }
  static SDCA_LANE Lanes zero() { return splat(T(0)); }
  // One copy per vector: GCC moves a copy of the whole array through the stack, which stalls
  // the loads that follow its stores.
  static SDCA_LANE Lanes load(const T* p) {
    Lanes r;
    for (int k = 0; k < K; ++k) std::memcpy(&r.v[k], p + k * B, sizeof r.v[k]);
    return r;
  }
  // The first count lanes of p (0 <= count <= L), pad in the others.
  static SDCA_LANE Lanes load(const T* p, int count, T pad) {
    if (count >= L) return load(p);
    Lanes r;
    for (int k = 0; k < K; ++k) detail::Tail<T, Isa>::load(r.v[k], p + k * B, count - k * B, pad);
    return r;
  }
  SDCA_LANE void store(T* p) const {
    for (int k = 0; k < K; ++k) std::memcpy(p + k * B, &v[k], sizeof v[k]);
  }
  SDCA_LANE void store(T* p, int count) const {
    if (count >= L) return store(p);
    for (int k = 0; k < K; ++k) detail::Tail<T, Isa>::store(p + k * B, v[k], count - k * B);
  }
  SDCA_LANE T operator[](int i) const { return v[i / B][i % B]; }
  SDCA_LANE void set(int i, T x) { v[i / B][i % B] = x; }
};

// Integer lanes of T's layout: masks (all ones or zero) of comparisons and the integers of
// T's bits.
template <typename T, typename Isa, int K>
struct Ints {
  typedef typename IntOf<T>::type value_type;
  typedef Register<T, Isa> R;
  typedef typename R::I V;
  static const int B = R::L;
  static const int L = B * K;
  V v[K];

  static SDCA_LANE Ints splat(value_type x) {
    Ints r;
    for (int k = 0; k < K; ++k) r.v[k] = V{} + x;
    return r;
  }
  static SDCA_LANE Ints zero() { return splat(0); }
  SDCA_LANE value_type operator[](int i) const { return v[i / B][i % B]; }
  SDCA_LANE void set(int i, value_type x) { v[i / B][i % B] = x; }
};

#define SDCA_LANES_TEMPLATE template <typename T, typename Isa, int K>
#define SDCA_LANES Lanes<T, Isa, K>
#define SDCA_INTS Ints<T, Isa, K>

#define SDCA_LANES_BINARY(op)                                                                      \
  SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES operator op(SDCA_LANES a, const SDCA_LANES& b) {        \
    for (int k = 0; k < K; ++k) a.v[k] = a.v[k] op b.v[k];                                         \
    return a;                                                                                      \
  }                                                                                                \
  SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES operator op(SDCA_LANES a, T b) {                        \
    for (int k = 0; k < K; ++k) a.v[k] = a.v[k] op b;                                              \
    return a;                                                                                      \
  }                                                                                                \
  SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES operator op(T a, SDCA_LANES b) {                        \
    for (int k = 0; k < K; ++k) b.v[k] = a op b.v[k];                                              \
    return b;                                                                                      \
  }                                                                                                \
  SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES& operator op##=(SDCA_LANES & a, const SDCA_LANES & b) { \
    for (int k = 0; k < K; ++k) a.v[k] = a.v[k] op b.v[k];                                         \
    return a;                                                                                      \
  }
SDCA_LANES_BINARY(+)
SDCA_LANES_BINARY(-)
SDCA_LANES_BINARY(*)
SDCA_LANES_BINARY(/)
#undef SDCA_LANES_BINARY

SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES operator-(SDCA_LANES a) {
  for (int k = 0; k < K; ++k) a.v[k] = -a.v[k];
  return a;
}

#define SDCA_LANES_COMPARE(op)                                                                    \
  SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator op(const SDCA_LANES& a, const SDCA_LANES& b) { \
    SDCA_INTS r;                                                                                  \
    for (int k = 0; k < K; ++k) r.v[k] = (typename SDCA_INTS::V)(a.v[k] op b.v[k]);               \
    return r;                                                                                     \
  }                                                                                               \
  SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator op(const SDCA_LANES& a, T b) {                 \
    SDCA_INTS r;                                                                                  \
    for (int k = 0; k < K; ++k) r.v[k] = (typename SDCA_INTS::V)(a.v[k] op b);                    \
    return r;                                                                                     \
  }
SDCA_LANES_COMPARE(<)
SDCA_LANES_COMPARE(<=)
SDCA_LANES_COMPARE(>)
SDCA_LANES_COMPARE(>=)
SDCA_LANES_COMPARE(==)
SDCA_LANES_COMPARE(!=)
#undef SDCA_LANES_COMPARE

#define SDCA_INTS_BINARY(op)                                                                    \
  SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator op(SDCA_INTS a, const SDCA_INTS& b) {        \
    for (int k = 0; k < K; ++k) a.v[k] = a.v[k] op b.v[k];                                      \
    return a;                                                                                   \
  }                                                                                             \
  SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator op(SDCA_INTS a,                              \
                                                      typename SDCA_INTS::value_type b) {       \
    for (int k = 0; k < K; ++k) a.v[k] = a.v[k] op b;                                           \
    return a;                                                                                   \
  }                                                                                             \
  SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS& operator op##=(SDCA_INTS & a, const SDCA_INTS & b) { \
    for (int k = 0; k < K; ++k) a.v[k] = a.v[k] op b.v[k];                                      \
    return a;                                                                                   \
  }
SDCA_INTS_BINARY(&)
SDCA_INTS_BINARY(|)
SDCA_INTS_BINARY(^)
#undef SDCA_INTS_BINARY

// Integer sums and differences wrap (they are taken unsigned).
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator+(SDCA_INTS a, const SDCA_INTS& b) {
  typedef typename SDCA_INTS::R::U U;
  for (int k = 0; k < K; ++k) a.v[k] = (typename SDCA_INTS::V)((U)a.v[k] + (U)b.v[k]);
  return a;
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator-(SDCA_INTS a, const SDCA_INTS& b) {
  typedef typename SDCA_INTS::R::U U;
  for (int k = 0; k < K; ++k) a.v[k] = (typename SDCA_INTS::V)((U)a.v[k] - (U)b.v[k]);
  return a;
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator+(SDCA_INTS a, typename SDCA_INTS::value_type b) {
  return a + SDCA_INTS::splat(b);
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator-(SDCA_INTS a, typename SDCA_INTS::value_type b) {
  return a - SDCA_INTS::splat(b);
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator~(SDCA_INTS a) {
  for (int k = 0; k < K; ++k) a.v[k] = ~a.v[k];
  return a;
}
// Left shifts of the bits (unsigned), right shifts arithmetic.
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator<<(SDCA_INTS a, int s) {
  typedef typename SDCA_INTS::R::U U;
  for (int k = 0; k < K; ++k) a.v[k] = (typename SDCA_INTS::V)((U)a.v[k] << s);
  return a;
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator>>(SDCA_INTS a, int s) {
  for (int k = 0; k < K; ++k) a.v[k] = a.v[k] >> s;
  return a;
}
// Logical right shift.
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS shift_right_logical(SDCA_INTS a, int s) {
  typedef typename SDCA_INTS::R::U U;
  for (int k = 0; k < K; ++k) a.v[k] = (typename SDCA_INTS::V)((U)a.v[k] >> s);
  return a;
}

#define SDCA_INTS_COMPARE(op)                                                                   \
  SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator op(const SDCA_INTS& a, const SDCA_INTS& b) { \
    SDCA_INTS r;                                                                                \
    for (int k = 0; k < K; ++k) r.v[k] = a.v[k] op b.v[k];                                      \
    return r;                                                                                   \
  }                                                                                             \
  SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS operator op(const SDCA_INTS& a,                       \
                                                      typename SDCA_INTS::value_type b) {       \
    SDCA_INTS r;                                                                                \
    for (int k = 0; k < K; ++k) r.v[k] = a.v[k] op b;                                           \
    return r;                                                                                   \
  }
SDCA_INTS_COMPARE(<)
SDCA_INTS_COMPARE(<=)
SDCA_INTS_COMPARE(>)
SDCA_INTS_COMPARE(>=)
SDCA_INTS_COMPARE(==)
SDCA_INTS_COMPARE(!=)
#undef SDCA_INTS_COMPARE

// a where m is set, else b.
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES select(const SDCA_INTS& m, SDCA_LANES a,
                                                const SDCA_LANES& b) {
  typedef typename SDCA_INTS::V I;
  typedef typename SDCA_LANES::V V;
  for (int k = 0; k < K; ++k) a.v[k] = (V)((m.v[k] & (I)a.v[k]) | (~m.v[k] & (I)b.v[k]));
  return a;
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS select(const SDCA_INTS& m, SDCA_INTS a,
                                               const SDCA_INTS& b) {
  for (int k = 0; k < K; ++k) a.v[k] = (m.v[k] & a.v[k]) | (~m.v[k] & b.v[k]);
  return a;
}

// a > b ? a : b and a < b ? a : b (b where either is NaN or both are zeros).
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES max(SDCA_LANES a, const SDCA_LANES& b) {
  for (int k = 0; k < K; ++k) detail::max<Isa>(a.v[k], a.v[k], b.v[k]);
  return a;
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES min(SDCA_LANES a, const SDCA_LANES& b) {
  for (int k = 0; k < K; ++k) detail::min<Isa>(a.v[k], a.v[k], b.v[k]);
  return a;
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES max(SDCA_LANES a, T b) {
  return max(a, SDCA_LANES::splat(b));
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES min(SDCA_LANES a, T b) {
  return min(a, SDCA_LANES::splat(b));
}
// a * b + c, fused where the instruction set has it.
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES fma(SDCA_LANES a, const SDCA_LANES& b,
                                             const SDCA_LANES& c) {
  for (int k = 0; k < K; ++k) detail::fma<Isa>(a.v[k], a.v[k], b.v[k], c.v[k]);
  return a;
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES fma(const SDCA_LANES& a, T b, T c) {
  return fma(a, SDCA_LANES::splat(b), SDCA_LANES::splat(c));
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES fma(const SDCA_LANES& a, const SDCA_LANES& b, T c) {
  return fma(a, b, SDCA_LANES::splat(c));
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES fma(const SDCA_LANES& a, T b, const SDCA_LANES& c) {
  return fma(a, SDCA_LANES::splat(b), c);
}

// The bits of the lanes as integers, and back; integers converted to T.
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_INTS bits(const SDCA_LANES& a) {
  SDCA_INTS r;
  for (int k = 0; k < K; ++k) r.v[k] = (typename SDCA_INTS::V)a.v[k];
  return r;
}
template <typename T, typename Isa, int K>
SDCA_LANE SDCA_LANES from_bits(const SDCA_INTS& a) {
  SDCA_LANES r;
  for (int k = 0; k < K; ++k) r.v[k] = (typename SDCA_LANES::V)a.v[k];
  return r;
}
template <typename T, typename Isa, int K>
SDCA_LANE SDCA_LANES convert(const SDCA_INTS& a) {
  SDCA_LANES r;
  for (int k = 0; k < K; ++k) r.v[k] = __builtin_convertvector(a.v[k], typename SDCA_LANES::V);
  return r;
}
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES abs(const SDCA_LANES& a) {
  return from_bits<T, Isa, K>(bits(a) & std::numeric_limits<typename IntOf<T>::type>::max());
}

// x rounded to the nearest integer (|x| < 2^22 for float, 2^51 for double): its value in T,
// and the integer in k.
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES round(const SDCA_LANES& x, SDCA_INTS& k) {
  const T magic = sizeof(T) == 4 ? T(12582912.0) : T(6755399441055744.0);  // 1.5 * 2^(digits-1)
  const SDCA_LANES t = x + magic;
  k = bits(t) - bits(SDCA_LANES::splat(magic));
  return t - magic;
}

// Whether any lane of a mask is set.
SDCA_LANES_TEMPLATE SDCA_LANE bool any(const SDCA_INTS& m) {
  typename SDCA_INTS::V acc = m.v[0];
  for (int k = 1; k < K; ++k) acc = acc | m.v[k];
  uint64_t words[(sizeof(acc) + sizeof(uint64_t) - 1) / sizeof(uint64_t)] = {};
  std::memcpy(words, &acc, sizeof acc);
  uint64_t any_word = 0;
  for (std::size_t i = 0; i < sizeof words / sizeof words[0]; ++i) any_word |= words[i];
  return any_word != 0;
}
SDCA_LANES_TEMPLATE SDCA_LANE bool all(const SDCA_INTS& m) { return !any(~m); }

// The lane indices 0, 1, ..., L - 1.
template <typename T, typename Isa, int K>
SDCA_LANE SDCA_INTS iota() {
  SDCA_INTS r;
  for (int i = 0; i < SDCA_INTS::L; ++i) r.set(i, static_cast<typename SDCA_INTS::value_type>(i));
  return r;
}
// The lanes i < count.
template <typename T, typename Isa, int K>
SDCA_LANE SDCA_INTS first_lanes(int count) {
  return iota<T, Isa, K>() < static_cast<typename SDCA_INTS::value_type>(count);
}
// Lanes i < count of a, zeros elsewhere.
SDCA_LANES_TEMPLATE SDCA_LANE SDCA_LANES head(const SDCA_LANES& a, int count) {
  if (count >= SDCA_LANES::L) return a;
  return select(first_lanes<T, Isa, K>(count), a, SDCA_LANES::zero());
}

// Horizontal sums and extrema, pairwise: the vectors' halves, then the lanes' halves.
namespace detail {
// a[i] = f(a[i], a[i + W]) for i < W, then the same on a[0, W): one step per width, which GCC
// unrolls where it leaves a loop over the halving widths rolled.
template <int W>
struct Halves {
  template <typename T, typename F>
  static SDCA_LANE void fold(T* a, F f) {
    for (int i = 0; i < W; ++i) a[i] = f(a[i], a[i + W]);
    Halves<W / 2>::fold(a, f);
  }
};
template <>
struct Halves<0> {
  template <typename T, typename F>
  static SDCA_LANE void fold(T*, F) {}
};
template <typename T, int N, typename F>
SDCA_LANE T pairwise(T* a, F f) {
  Halves<N / 2>::fold(a, f);
  return a[0];
}
struct Add {
  template <typename T>
  SDCA_LANE T operator()(T a, T b) const {
    return a + b;
  }
};
struct Max {
  template <typename T>
  SDCA_LANE T operator()(T a, T b) const {
    return a > b ? a : b;
  }
};
struct Min {
  template <typename T>
  SDCA_LANE T operator()(T a, T b) const {
    return a < b ? a : b;
  }
};
}  // namespace detail
SDCA_LANES_TEMPLATE SDCA_LANE T sum(const SDCA_LANES& a) {
  T lane[SDCA_LANES::L];
  a.store(lane);
  return detail::pairwise<T, SDCA_LANES::L>(lane, detail::Add());
}
SDCA_LANES_TEMPLATE SDCA_LANE T max_lane(const SDCA_LANES& a) {
  T lane[SDCA_LANES::L];
  a.store(lane);
  return detail::pairwise<T, SDCA_LANES::L>(lane, detail::Max());
}
SDCA_LANES_TEMPLATE SDCA_LANE T min_lane(const SDCA_LANES& a) {
  T lane[SDCA_LANES::L];
  a.store(lane);
  return detail::pairwise<T, SDCA_LANES::L>(lane, detail::Min());
}

// f(x) of every lane of a set in m, one at a time, for the rare lanes of a slow path.
template <typename T, typename Isa, int K, typename F>
inline SDCA_LANES per_lane(SDCA_LANES r, const SDCA_LANES& x, const SDCA_INTS& m, F f) {
  for (int i = 0; i < SDCA_LANES::L; ++i)
    if (m[i]) r.set(i, f(x[i]));
  return r;
}

// Float lanes as double lanes of the same count (R double vectors per float vector), and back.
template <typename Isa>
struct Widening {
  static const int R = Register<float, Isa>::L / Register<double, Isa>::L;
};
template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K * Widening<Isa>::R> widen(const Lanes<float, Isa, K>& a) {
  typedef Lanes<double, Isa, K * Widening<Isa>::R> D;
  const int R = Widening<Isa>::R, N = D::B;
  typedef typename Vector<float, N>::type Part;
  D r;
  for (int k = 0; k < K; ++k)
    for (int i = 0; i < R; ++i) {
      Part p;
      std::memcpy(&p, reinterpret_cast<const char*>(&a.v[k]) + i * sizeof p, sizeof p);
      r.v[k * R + i] = __builtin_convertvector(p, typename D::V);
    }
  return r;
}
template <typename Isa, int K>
SDCA_LANE Lanes<float, Isa, K / Widening<Isa>::R> narrow(const Lanes<double, Isa, K>& a) {
  typedef Lanes<float, Isa, K / Widening<Isa>::R> F;
  const int R = Widening<Isa>::R, N = Lanes<double, Isa, K>::B;
  static_assert(K % Widening<Isa>::R == 0, "whole float vectors");
  typedef typename Vector<float, N>::type Part;
  F r;
  for (int k = 0; k < K / R; ++k)
    for (int i = 0; i < R; ++i) {
      const Part p = __builtin_convertvector(a.v[k * R + i], Part);
      std::memcpy(reinterpret_cast<char*>(&r.v[k]) + i * sizeof p, &p, sizeof p);
    }
  return r;
}

// Lanes as double lanes: float lanes widened, double lanes themselves.
template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K * Widening<Isa>::R> to_double(const Lanes<float, Isa, K>& a) {
  return widen(a);
}
template <typename Isa, int K>
SDCA_LANE Lanes<double, Isa, K> to_double(const Lanes<double, Isa, K>& a) {
  return a;
}

#undef SDCA_LANES_TEMPLATE
#undef SDCA_LANES
#undef SDCA_INTS

// The lanes of the build: Lanes<T> on isa::native, one vector.
template <typename T, int K = 1>
struct Native {
  typedef Lanes<T, isa::native, K> type;
};

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_LANES_H
