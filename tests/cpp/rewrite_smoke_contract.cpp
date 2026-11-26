// Smoke test of the headers' floating-point contraction: in every instruction set's pass, a * b + c
// rounds the product (no fused multiply-add the source does not ask for), while the lanes'
// fma() fuses where the instruction set has FMA. Built without -ffp-contract, as a user of the
// headers would (GCC fuses by default under FMA targets):
//
//   c++ -std=c++11 -O2 -Wall -Wextra -Werror -Iinclude -Itests/cpp THIS -o contract && ./contract
#include <cstdio>

#define SDCA_EACH_ISA "rewrite_smoke_contract_probe.h"
#include "sdca/core/each_isa.h"

namespace {

// 1 + 2^-30 and 1 - 2^-30: their product 1 - 2^-60 rounds to 1, so a * b - 1 is 0 rounded
// and -2^-60 fused.
volatile double a = 1 + 1.0 / (1 << 30), b = 1 - 1.0 / (1 << 30), c = -1;

int failures = 0;

template <typename Probe>
void check(const char* isa, bool has_fma) {
  const double exact = -1.0 / (1 << 30) / (1 << 30);
  const double rounded[] = {Probe::scalar(a, b, c), Probe::lanes(a, b, c),
                            Probe::template instantiated<double>(a, b, c)};
  const char* names[] = {"scalar", "lanes", "template"};
  for (int i = 0; i < 3; ++i) {
    const bool ok = rounded[i] == 0;
    failures += !ok;
    std::printf("%-7s %-8s a * b + c = %-12g %s\n", isa, names[i], rounded[i],
                ok ? "rounded" : "FUSED");
  }
  const double fused = Probe::fused(a, b, c);
  const bool ok = has_fma ? fused == exact : (fused == exact || fused == 0);
  failures += !ok;
  std::printf("%-7s %-8s fma(a, b, c) = %-12g %s\n", isa, "fma", fused, ok ? "ok" : "WRONG");
}

#define SDCA_PROBE(NS)                                     \
  struct NS##_probe {                                      \
    static double scalar(double x, double y, double z) {   \
      return sdca::NS::core::probe::scalar(x, y, z);       \
    }                                                      \
    static double lanes(double x, double y, double z) {    \
      return sdca::NS::core::probe::lanes(x, y, z);        \
    }                                                      \
    template <typename T>                                  \
    static T instantiated(T x, T y, T z) {                 \
      return sdca::NS::core::probe::instantiated(x, y, z); \
    }                                                      \
    static double fused(double x, double y, double z) {    \
      return sdca::NS::core::probe::fused(x, y, z);        \
    }                                                      \
  };

#if defined(__x86_64__)
SDCA_PROBE(isa_scalar)
SDCA_PROBE(isa_avx2)
SDCA_PROBE(isa_avx512)
#else
SDCA_PROBE(SDCA_ISA_NAMESPACE)
#endif
#undef SDCA_PROBE

}  // namespace

int main() {
#if defined(__x86_64__)
  __builtin_cpu_init();
  check<isa_scalar_probe>("scalar", false);
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
    check<isa_avx2_probe>("avx2", true);
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
      __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl"))
    check<isa_avx512_probe>("avx512", true);
#else
  check<SDCA_ISA_NAMESPACE_probe>("native", SDCA_ISA_ID == 2);
#endif
  std::printf("%s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
