/* The C ABI computes with subnormals under a caller's flush-to-zero and restores
 * the caller's floating-point modes, on success and on error; raised exception
 * flags stay raised. */
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sdca/c_api.h"

#if defined(__SSE__) || defined(_M_X64)
#include <xmmintrin.h>
/* The modes of MXCSR, without its exception flags. */
static uint64_t control(void) { return _mm_getcsr() & ~0x3fu; }
static void set_control(uint64_t value) { _mm_setcsr((unsigned)value | (_mm_getcsr() & 0x3fu)); }
static const uint64_t flush = 0x8040; /* MXCSR FTZ and DAZ */
#elif defined(__aarch64__)
static uint64_t control(void) {
  uint64_t value;
  __asm__ __volatile__("mrs %0, fpcr" : "=r"(value));
  return value;
}
static void set_control(uint64_t value) { __asm__ __volatile__("msr fpcr, %0" : : "r"(value)); }
static const uint64_t flush = (uint64_t)1 << 24; /* FPCR FZ */
#else
#define NO_FLUSH_CONTROL
#endif

#define N 50000 /* milliseconds of rows: workers run */
static double z[2 * N], gradient[2 * N], reference[2 * N], values[N];
static uint8_t positive[2 * N];
static int failures;

static void check(int ok, const char* what) {
  if (!ok) {
    ++failures;
    fprintf(stderr, "FAIL: %s\n", what);
  }
}

#ifndef NO_FLUSH_CONTROL
static uint64_t saved_control;

/* The modes the test set and its raised flag, kept by a call; checked before any arithmetic. */
static void check_restored(const char* call) {
  char what[128];
  snprintf(what, sizeof what, "%s restores the environment", call);
  check(control() == saved_control && fegetround() == FE_UPWARD &&
            fetestexcept(FE_INEXACT) == FE_INEXACT,
        what);
}

static int entropy(size_t threads, double* out) {
  struct sdca_options options = SDCA_OPTIONS_DEFAULT;
  options.gamma = 1.0;
  options.num_threads = threads;
  return sdca_loss_gradient(&options, N, 2, z, positive, values, out);
}
#endif

int main(void) {
#ifdef NO_FLUSH_CONTROL
  puts("c_api_fpenv: no flush-to-zero control on this target; skipped");
  return 0;
#else
  /* Rows [0, -740]: the entropy gradient exp(-740) is subnormal. */
  for (size_t i = 0; i < N; ++i) z[2 * i] = 0.0, z[2 * i + 1] = -740.0, positive[2 * i] = 1;
  check(entropy(1, reference) == SDCA_OK, "entropy in the default environment");
  check(reference[1] > 0 && reference[1] < DBL_MIN, "the reference gradient is subnormal");

  set_control(control() | flush);
  volatile double tiny = 1e-300;
  if (tiny * 1e-10 != 0) {
    puts("c_api_fpenv: flush-to-zero has no effect on this target; skipped");
    return 0;
  }
  fesetround(FE_UPWARD);
  feclearexcept(FE_ALL_EXCEPT);
  feraiseexcept(FE_INEXACT);
  saved_control = control();

  for (size_t threads = 1; threads <= 3; threads += 2) {
    memset(gradient, 0, sizeof gradient);
    int code = entropy(threads, gradient);
    check_restored("sdca_loss_gradient on threads");
    check(code == SDCA_OK && memcmp(gradient, reference, sizeof gradient) == 0,
          "subnormal gradients under flush-to-zero, on 1 and 3 threads");
  }

  /* Its normalization converges only with subnormals. */
  const double row[4] = {0.0, 20.0, 20.0, -100.0};
  const uint8_t labels[4] = {1, 0, 1, 0};
  double value, row_gradient[4];
  struct sdca_options options = SDCA_OPTIONS_DEFAULT;
  options.loss = SDCA_PROXIMAL_CROSS_ENTROPY;
  options.gamma = 1.0;
  options.alpha = 1e300;
  int code = sdca_loss_gradient(&options, 1, 4, row, labels, &value, row_gradient);
  check_restored("sdca_loss_gradient");
  check(code == SDCA_OK && fabs(value - log(2.0)) < 1e-12, "proximal_entropy at alpha = 1e300");

  const double bad[2] = {0.0, NAN};
  options.loss = SDCA_CROSS_ENTROPY;
  options.alpha = 1.0;
  code = sdca_loss_gradient(&options, 1, 2, bad, positive, &value, row_gradient);
  check_restored("a failing sdca_loss_gradient");
  check(code == SDCA_INVALID_ARGUMENT, "NaN scores are rejected");

  const int64_t y[2] = {0, 1};
  double previous[4] = {0}, out[4];
  code = sdca_update_objective(&options, 2, 2, z, y, NULL, previous, 1.0, out);
  check_restored("sdca_update_objective");
  check(code == SDCA_OK, "sdca_update_objective");

  const double x[4] = {1.0, 0.0, 0.0, 1.0};
  double w[4], a[4], history[3 * 10];
  struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;
  options.max_epoch = 5;
  options.eval_epoch = 5;
  code = sdca_fit(&options, 2, 2, 2, x, y, NULL, NULL, w, a, history, 3, &result);
  check_restored("sdca_fit");
  check(code == SDCA_OK, "sdca_fit");

  fesetround(FE_TONEAREST);
  set_control(control() & ~flush);
  printf("c_api_fpenv: %d failures\n", failures);
  return failures != 0;
#endif
}
