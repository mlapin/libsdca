// The instruction set of the code that follows (no include guard: each include decides anew).
//
// The ISA-dependent headers (sdca/core, sdca/loss and the per-ISA part of sdca/kernel.h)
// compile once per instruction set into a namespace of their own, sdca::isa_<name>: the one of
// the compiler's flags (the default), and, from sdca/kernel.h, every other instruction set of
// the architecture under target attributes, so one translation unit without -m flags holds
// them all and chooses at run time. SDCA_TARGET_ISA (1 scalar, 2 neon, 3 avx2, 4 avx512)
// overrides the flags for such a pass.
//
// An ISA-dependent header includes this one, guards itself once per instruction set and puts
// its code between SDCA_ISA_BEGIN and SDCA_ISA_END:
//
//   #include "sdca/core/isa.h"
//   #if SDCA_ISA_FIRST(SDCA_LOSS_TOPK_H)
//   #if SDCA_ISA_ID == 1
//   #define SDCA_LOSS_TOPK_H1 1
//   #elif SDCA_ISA_ID == 2
//   #define SDCA_LOSS_TOPK_H2 1
//   #elif SDCA_ISA_ID == 3
//   #define SDCA_LOSS_TOPK_H3 1
//   #else
//   #define SDCA_LOSS_TOPK_H4 1
//   #endif
//   ...
//   #endif
//
// Inside, core:: and loss:: name the instruction set's own namespaces, which also see the
// neutral sdca::loss (sdca/loss/row.h) and sdca::half, sdca::bfloat16.
#include "sdca/core/types.h"
#include "sdca/loss/row.h"

// The generic lanes lower to halves or scalars where a vector is wider than the instruction
// set's (64-byte lanes without AVX-512, isa::scalar), as intended; GCC reports some of these
// without a location, so the warning is off for the whole translation unit.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wvector-operation-performance"
#endif

#if !defined(__GNUC__) && !defined(__clang__)
#error "sdca/core needs the vector extensions of GCC or Clang"
#endif

#undef SDCA_ISA_ID
#undef SDCA_ISA_AVX512
#undef SDCA_ISA_AVX2
#undef SDCA_ISA_NEON
#undef SDCA_ISA_NAMESPACE

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) && \
    defined(__AVX512VL__) && defined(__FMA__) && defined(__F16C__)
#define SDCA_DEFAULT_ISA_ID 4
#elif defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define SDCA_DEFAULT_ISA_ID 3
#elif defined(__aarch64__) && defined(__ARM_NEON)
#define SDCA_DEFAULT_ISA_ID 2
#else
#define SDCA_DEFAULT_ISA_ID 1
#endif

#if defined(SDCA_TARGET_ISA)
#define SDCA_ISA_ID SDCA_TARGET_ISA
#else
#define SDCA_ISA_ID SDCA_DEFAULT_ISA_ID
#endif

#if SDCA_ISA_ID == 4
#define SDCA_ISA_AVX512 1
#define SDCA_ISA_NAMESPACE isa_avx512
#elif SDCA_ISA_ID == 3
#define SDCA_ISA_AVX2 1
#define SDCA_ISA_NAMESPACE isa_avx2
#elif SDCA_ISA_ID == 2
#define SDCA_ISA_NEON 1
#define SDCA_ISA_NAMESPACE isa_neon
#else
#define SDCA_ISA_NAMESPACE isa_scalar
#endif

#ifndef SDCA_ISA_BEGIN
// Whether the guard of this instruction set (the guard's name and its ID) is undefined.
#define SDCA_ISA_FIRST(g)                                                    \
  ((SDCA_ISA_ID == 1 && !(g##1 + 0)) || (SDCA_ISA_ID == 2 && !(g##2 + 0)) || \
   (SDCA_ISA_ID == 3 && !(g##3 + 0)) || (SDCA_ISA_ID == 4 && !(g##4 + 0)))
// Between SDCA_ISA_BEGIN and SDCA_ISA_END the compiler fuses no multiply and add the source
// does not write as fma(): every instruction set and every build of the headers rounds the
// same, whatever -ffp-contract says (GCC's default fuses them under FMA targets). Clang's
// -ffp-contract=fast disregards the pragma by design (=fast-honor-pragmas keeps it). GCC
// inlines these functions only into callers of the same optimization options.
#if defined(__clang__)
#define SDCA_FP_BEGIN _Pragma("float_control(push)") _Pragma("clang fp contract(off)")
#define SDCA_FP_END _Pragma("float_control(pop)")
#else
#define SDCA_FP_BEGIN _Pragma("GCC push_options") _Pragma("GCC optimize(\"fp-contract=off\")")
#define SDCA_FP_END _Pragma("GCC pop_options")
#endif
#define SDCA_ISA_BEGIN           \
  SDCA_FP_BEGIN                  \
  namespace sdca {               \
  namespace SDCA_ISA_NAMESPACE { \
  namespace core {               \
  using ::sdca::bfloat16;        \
  using ::sdca::half;            \
  }                              \
  namespace loss {               \
  using namespace ::sdca::loss;  \
  }
#define SDCA_ISA_END \
  }                  \
  }                  \
  SDCA_FP_END

// The lane functions inline into the row loops (GCC otherwise calls them out of line and
// spills every vector live across the call), and row kernels flatten their lambdas.
#define SDCA_LANE inline __attribute__((always_inline))
#define SDCA_FLAT __attribute__((flatten))
#endif
