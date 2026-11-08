#ifndef SDCA_UTILITY_FPENV_H
#define SDCA_UTILITY_FPENV_H

#include <cfenv>
#include <cstdint>
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#define SDCA_FPENV_X86 1
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#define SDCA_FPENV_AARCH64 1
#endif

namespace sdca {

namespace detail {

// The modes (not the flags) of the SIMD control register: MXCSR on x86-64,
// FPCR on aarch64; 0 elsewhere.
inline std::uint64_t fp_control() {
#if defined(SDCA_FPENV_X86)
  return _mm_getcsr() & ~0x3fu;
#elif defined(SDCA_FPENV_AARCH64)
  std::uint64_t fpcr;
  __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
  return fpcr;
#else
  return 0;
#endif
}

// Sets the modes of the control register, keeping its exception flags.
inline void set_fp_control(std::uint64_t modes) {
#if defined(SDCA_FPENV_X86)
  const unsigned csr = _mm_getcsr();
  if ((csr & ~0x3fu) != modes) _mm_setcsr(static_cast<unsigned>(modes) | (csr & 0x3fu));
#elif defined(SDCA_FPENV_AARCH64)
  if (fp_control() != modes) __asm__ __volatile__("msr fpcr, %0" : : "r"(modes));
#else
  (void)modes;
#endif
}

// Flush-to-zero and denormals-are-zero: MXCSR bits 15 and 6, FPCR FZ (24).
inline std::uint64_t flush_to_zero_bits() {
#if defined(SDCA_FPENV_X86)
  return 0x8040;
#elif defined(SDCA_FPENV_AARCH64)
  return std::uint64_t(1) << 24;
#else
  return 0;
#endif
}

// The floating-point modes of a thread: rounding, traps, flush-to-zero and,
// on x86-64, the x87 precision. apply() keeps the raised exception flags.
class fp_modes {
 public:
  static fp_modes current() {
    fp_modes modes;
#if defined(SDCA_FPENV_X86) || defined(SDCA_FPENV_AARCH64)
    modes.control_ = fp_control();
    modes.x87_ = x87();
#else
    std::fegetenv(&modes.env_);
#endif
    return modes;
  }

  // Round to nearest, no traps, subnormals kept (and 64-bit x87 precision).
  static fp_modes ieee() {
    fp_modes modes;
#if defined(SDCA_FPENV_X86)
    modes.control_ = 0x1f80;
    modes.x87_ = 0x037f;
#elif defined(SDCA_FPENV_AARCH64)
    modes.control_ = 0;
#else
    modes.default_ = true;
#endif
    return modes;
  }

  void apply() const {
#if defined(SDCA_FPENV_X86) || defined(SDCA_FPENV_AARCH64)
    set_fp_control(control_);
    if (x87() != x87_) set_x87(x87_);
#else
    std::fexcept_t flags;
    std::fegetexceptflag(&flags, FE_ALL_EXCEPT);
    std::fesetenv(default_ ? FE_DFL_ENV : &env_);
    std::fesetexceptflag(&flags, FE_ALL_EXCEPT);
#endif
  }

 private:
  fp_modes() {}

  // The x87 control word, of the long double arithmetic (GCC, Clang).
  static unsigned short x87() {
    unsigned short word = 0;
#if defined(SDCA_FPENV_X86) && (defined(__GNUC__) || defined(__clang__))
    __asm__ __volatile__("fnstcw %0" : "=m"(word));
#endif
    return word;
  }
  static void set_x87(unsigned short word) {
#if defined(SDCA_FPENV_X86) && (defined(__GNUC__) || defined(__clang__))
    __asm__ __volatile__("fldcw %0" : : "m"(word));
#else
    (void)word;
#endif
  }

#if defined(SDCA_FPENV_X86) || defined(SDCA_FPENV_AARCH64)
  std::uint64_t control_ = 0;
  unsigned short x87_ = 0;
#else
  std::fenv_t env_;
  bool default_ = false;
#endif
};

// IEEE arithmetic in a scope (fp_modes::ieee); the caller's modes return on
// exit, and the exception flags raised inside stay raised, as feupdateenv.
class ieee_environment {
 public:
  ieee_environment() : saved_(fp_modes::current()) { fp_modes::ieee().apply(); }
  ~ieee_environment() { saved_.apply(); }
  ieee_environment(const ieee_environment&) = delete;
  ieee_environment& operator=(const ieee_environment&) = delete;

 private:
  fp_modes saved_;
};

}  // namespace detail

}  // namespace sdca

#endif
