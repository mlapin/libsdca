// Includes the header named by SDCA_EACH_ISA once per instruction set of the architecture, as
// sdca/kernel.h includes its kernels: the one of the compiler's flags, then on x86-64 the
// others (scalar, AVX2, AVX-512) under target attributes. No include guard: the header
// guards itself once per instruction set (sdca/core/isa.h) and includes only standard headers
// already included before this one. The instruction set of the flags holds afterwards.
//
//   #define SDCA_EACH_ISA "sdca/solver/eval/products.h"
//   #include "sdca/core/each_isa.h"
#ifndef SDCA_EACH_ISA
#error "SDCA_EACH_ISA names the header to include"
#endif
#include "sdca/core/isa.h"
#include SDCA_EACH_ISA
#if defined(__x86_64__)
#if SDCA_DEFAULT_ISA_ID != 1
#define SDCA_TARGET_ISA 1
#include SDCA_EACH_ISA
#undef SDCA_TARGET_ISA
#endif
#if SDCA_DEFAULT_ISA_ID != 3
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma,f16c"))), apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")
#endif
#define SDCA_TARGET_ISA 3
#include SDCA_EACH_ISA
#undef SDCA_TARGET_ISA
#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif
#endif
#if SDCA_DEFAULT_ISA_ID != 4
#if defined(__clang__)
#pragma clang attribute push(                                                    \
    __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,avx2,fma,f16c"))), \
    apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avx512f,avx512dq,avx512bw,avx512vl,avx2,fma,f16c")
#endif
#define SDCA_TARGET_ISA 4
#include SDCA_EACH_ISA
#undef SDCA_TARGET_ISA
#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif
#endif
#endif
#undef SDCA_EACH_ISA
#include "sdca/core/isa.h"
