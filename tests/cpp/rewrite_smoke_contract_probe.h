// a * b + c in one instruction set's pass of the headers (rewrite_smoke_contract.cpp includes it
// through sdca/core/each_isa.h): as scalars, as lanes, in a template the caller instantiates,
// and through the lanes' fma().
#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_TEST_CONTRACT_PROBE_H)
#if SDCA_ISA_ID == 1
#define SDCA_TEST_CONTRACT_PROBE_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_TEST_CONTRACT_PROBE_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_TEST_CONTRACT_PROBE_H3 1
#else
#define SDCA_TEST_CONTRACT_PROBE_H4 1
#endif

#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN
namespace core {
namespace probe {

__attribute__((noinline)) inline double scalar(double a, double b, double c) { return a * b + c; }
__attribute__((noinline)) inline double lanes(double a, double b, double c) {
  typedef Lanes<double, isa::native, 2> D;
  return (D::splat(a) * D::splat(b) + D::splat(c))[0];
}
template <typename T>
T instantiated(T a, T b, T c) {
  return a * b + c;
}
__attribute__((noinline)) inline double fused(double a, double b, double c) {
  typedef Lanes<double, isa::native, 2> D;
  return fma(D::splat(a), D::splat(b), D::splat(c))[0];
}

}  // namespace probe
}  // namespace core
SDCA_ISA_END

#endif
