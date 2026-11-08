/* Malformed direct ABI requests must fail before accessing caller buffers. */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sdca/c_api.h"

static int failures;
static double scores[6] = {0}, previous[6] = {0}, output[6], values[2];
static double history[20];
static int64_t labels[2] = {0, 1};
static uint8_t positive[6] = {1, 0, 0, 0, 1, 0};
static struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;

static void rejected(int status, const char* what) {
  if (status != SDCA_INVALID_ARGUMENT || !*sdca_last_error()) {
    ++failures;
    fprintf(stderr, "FAIL: %s returned %d (%s)\n", what, status, sdca_last_error());
  }
}

/* The fit options of these tests: no epochs, one evaluation, gamma 1. */
static struct sdca_options fit_options(void) {
  struct sdca_options options = SDCA_OPTIONS_DEFAULT;
  options.gamma = 1.0;
  options.max_epoch = 0;
  options.eval_epoch = 1;
  options.tol = 0.0;
  return options;
}

static void null_buffers(void) {
  struct sdca_options options = SDCA_OPTIONS_DEFAULT;
  options.gamma = 1.0;
  for (int missing = 0; missing < 4; ++missing)
    rejected(sdca_loss_gradient(&options, 2, 3, missing == 0 ? NULL : scores,
                                missing == 1 ? NULL : positive, missing == 2 ? NULL : values,
                                missing == 3 ? NULL : output),
             "loss NULL buffer");
  for (int missing = 0; missing < 4; ++missing)
    rejected(sdca_update_objective(&options, 2, 3, missing == 0 ? NULL : scores,
                                   missing == 1 ? NULL : labels, NULL,
                                   missing == 2 ? NULL : previous, 1.0,
                                   missing == 3 ? NULL : output),
             "update NULL buffer");
  options = fit_options();
  for (int missing = 0; missing < 6; ++missing)
    rejected(sdca_fit(&options, 2, 1, 3, missing == 0 ? NULL : scores,
                      missing == 1 ? NULL : labels, NULL, NULL, missing == 2 ? NULL : output,
                      missing == 3 ? NULL : previous, missing == 4 ? NULL : history, 2,
                      missing == 5 ? NULL : &result),
             "fit NULL buffer");
}

/* NULL or mis-sized structs, e.g. from another header. */
static void malformed_structs(void) {
  struct sdca_options options = fit_options(), sized = fit_options();
  struct sdca_fit_result other = SDCA_FIT_RESULT_INIT;
  rejected(sdca_loss_gradient(NULL, 2, 3, scores, positive, values, output), "loss NULL options");
  rejected(sdca_update_objective(NULL, 2, 3, scores, labels, NULL, previous, 1.0, output),
           "update NULL options");
  rejected(sdca_fit(NULL, 2, 1, 3, scores, labels, NULL, NULL, output, previous, history, 2,
                    &result),
           "fit NULL options");
  const size_t sizes[2] = {sizeof options - sizeof(int), sizeof options + 8};
  for (size_t i = 0; i < 2; ++i) {
    sized.size = sizes[i];
    rejected(sdca_loss_gradient(&sized, 2, 3, scores, positive, values, output),
             "loss options size");
    rejected(sdca_update_objective(&sized, 2, 3, scores, labels, NULL, previous, 1.0, output),
             "update options size");
    rejected(sdca_fit(&sized, 2, 1, 3, scores, labels, NULL, NULL, output, previous, history, 2,
                      &result),
             "fit options size");
    other.size = sizes[i];
    rejected(sdca_fit(&options, 2, 1, 3, scores, labels, NULL, NULL, output, previous, history, 2,
                      &other),
             "fit result size");
  }
}

static void malformed_offsets(void) {
  const int64_t offsets[][3] = {{0, 1, INT64_MIN}, {0, -1, 0}, {0, 1, 1}, {0, 1, 0}};
  struct sdca_options options = SDCA_OPTIONS_DEFAULT;
  options.loss = SDCA_MULTILABEL_CROSS_ENTROPY;
  for (size_t i = 0; i < sizeof offsets / sizeof offsets[0]; ++i)
    rejected(sdca_update_objective(&options, 2, 3, scores, labels, offsets[i], previous, 1.0,
                                   output),
             "non-increasing or overflowing offsets");
}

static void overflowing_extents(void) {
  const size_t n = SIZE_MAX / sizeof(double) / 3 + 1;
  struct sdca_options options = SDCA_OPTIONS_DEFAULT;
  rejected(sdca_loss_gradient(&options, n, 3, scores, positive, values, output),
           "loss byte extent overflow");
  rejected(sdca_update_objective(&options, n, 3, scores, labels, NULL, previous, 1.0, output),
           "update byte extent overflow");
  options = fit_options();
  rejected(sdca_fit(&options, n, 1, 3, scores, labels, NULL, NULL, output, previous, history, 2,
                    &result),
           "fit dual byte extent overflow");
  rejected(sdca_fit(&options, 2, 1, 3, scores, labels, NULL, NULL, output, previous, history,
                    SIZE_MAX / 10, &result),
           "fit history byte extent overflow");
}

static void overflowing_blas_dimensions(void) {
#if !defined(BLAS_MKL) && !defined(BLAS_MATLAB)
  const size_t large = (size_t)INT_MAX + 1;
  const size_t shapes[][3] = {
      {large, 1, 2}, {1, large, 2}, {1, 1, large}, {1, large / 2, 2}, {large / 2, 1, 2}};
  const struct sdca_options options = fit_options();
  for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; ++i)
    rejected(sdca_fit(&options, shapes[i][0], shapes[i][1], shapes[i][2], scores, labels, NULL,
                      NULL, output, previous, history, 2, &result),
             "fit BLAS integer overflow");
#endif
}

static void invalid_projection(void) {
  const int modes[] = {-1, 2};
  for (size_t i = 0; i < 2; ++i) {
    struct sdca_options options = fit_options();
    options.projection_precision = modes[i];
    options.num_threads = 1;
    options.loss = SDCA_MULTILABEL_HINGE;
    rejected(sdca_loss_gradient(&options, 2, 3, scores, positive, values, output),
             "loss projection precision");
    options.loss = SDCA_CROSS_ENTROPY;
    rejected(sdca_update_objective(&options, 2, 3, scores, labels, NULL, previous, 1.0, output),
             "update projection precision");
    rejected(sdca_fit(&options, 2, 1, 3, scores, labels, NULL, NULL, output, previous, history, 2,
                      &result),
             "fit projection precision");
  }
}

/* Both projection precisions fit a smooth multilabel hinge alike to rounding. Shrinking is off:
   its keep-or-drop tests would turn rounding differences into different paths. */
static void projection_modes(void) {
  const double x[8] = {1.0, 0.2, -0.5, 1.1, 0.3, -1.2, -0.7, 0.4};
  const int64_t sets[6] = {0, 2, 1, 0, 1, 2}, offsets[5] = {0, 2, 3, 4, 6};
  const int modes[2] = {SDCA_PROJECTION_FLOAT64, SDCA_PROJECTION_EXTENDED};
  double w[2][6], a[2][12], hist[3 * 10];
  struct sdca_fit_result fit_result = SDCA_FIT_RESULT_INIT;
  struct sdca_options options = SDCA_OPTIONS_DEFAULT;
  options.loss = SDCA_MULTILABEL_HINGE;
  options.gamma = 0.5;
  options.max_epoch = 20;
  options.eval_epoch = 20;
  options.tol = 0.0;
  options.shrinking = SDCA_SHRINKING_OFF;
  for (int m = 0; m < 2; ++m) {
    options.projection_precision = modes[m];
    if (sdca_fit(&options, 4, 2, 3, x, sets, offsets, NULL, w[m], a[m], hist, 3, &fit_result) !=
        SDCA_OK) {
      ++failures;
      fprintf(stderr, "FAIL: projection mode %d fit (%s)\n", modes[m], sdca_last_error());
      return;
    }
  }
  for (size_t i = 0; i < 6; ++i) {
    const double diff = w[0][i] - w[1][i], scale = 1.0 + (w[0][i] < 0 ? -w[0][i] : w[0][i]);
    if (!(diff < 1e-12 * scale && -diff < 1e-12 * scale)) {
      ++failures;
      fputs("FAIL: the projection precisions disagree beyond rounding\n", stderr);
      return;
    }
  }
}

static void precision_pairs(void) {
  float z[3] = {0, 0, 0}, gradient[3];
  uint8_t target[3] = {1, 0, 0};
  double loss;
  struct sdca_options options = SDCA_OPTIONS_DEFAULT;
  options.precision = SDCA_FLOAT32_FLOAT64;
  if (SDCA_FLOAT32 != 0 || SDCA_FLOAT32_FLOAT64 != 1 || SDCA_FLOAT64 != 2 ||
      sdca_loss_gradient(&options, 1, 3, z, target, &loss, gradient) != SDCA_OK ||
      !(loss > 1.09 && loss < 1.10) || !(gradient[0] < -0.66f && gradient[1] > 0.33f)) {
    ++failures;
    fputs("FAIL: float32 storage / float64 compute ABI pair\n", stderr);
  }
}

/* Every loss rejects k, gamma and alpha out of range, the in-place entropies included. */
static void invalid_loss_options(void) {
  for (int loss = SDCA_CROSS_ENTROPY; loss <= SDCA_PROXIMAL_CROSS_ENTROPY; ++loss) {
    for (int bad = 0; bad < 3; ++bad) {
      struct sdca_options options = SDCA_OPTIONS_DEFAULT;
      options.loss = loss;
      options.num_threads = 1;
      if (bad == 0) options.k = 3;
      if (bad == 1) options.gamma = -1.0;
      if (bad == 2) options.alpha = 0.0;
      char what[48];
      snprintf(what, sizeof what, "loss %d invalid option %d", loss, bad);
      rejected(sdca_loss_gradient(&options, 2, 3, scores, positive, values, output), what);
    }
  }
}

/* The fit result reports the status, the epochs and the history rows used. */
static void fit_result_fields(void) {
  const double x[6] = {1.0, 0.1, -0.5, 0.9, -0.4, -0.8};
  const int64_t y[3] = {0, 1, 2};
  double w[6], a[9], hist[7 * 10];
  struct sdca_fit_result fit_result = SDCA_FIT_RESULT_INIT;
  struct sdca_options options = SDCA_OPTIONS_DEFAULT;
  options.max_epoch = 50;
  options.tol = 0.0;
  const int code = sdca_fit(&options, 3, 2, 3, x, y, NULL, NULL, w, a, hist, 7, &fit_result);
  const size_t used = fit_result.history_used;
  if (code != SDCA_OK || fit_result.size != sizeof fit_result ||
      fit_result.status < SDCA_STATUS_SOLVED || fit_result.status > SDCA_STATUS_MAX_EPOCH ||
      fit_result.epochs < 1 || fit_result.epochs > 50 || used < 1 || used > 7 ||
      hist[(used - 1) * 10] != (double)fit_result.epochs) {
    ++failures;
    fprintf(stderr, "FAIL: fit result %d %zu %zu (%s)\n", fit_result.status, fit_result.epochs,
            fit_result.history_used, sdca_last_error());
  }
}

/* sdca_cross_entropy_backward may write the gradient over its scores: the same bits as into
 * a separate buffer, in every storage type and working precision, with class weights, an
 * ignored row and one scale per row. */
static void in_place_gradient(void) {
  enum { N = 3, K = 37 };
  static const int64_t target[N] = {4, -100, 36};
  static const double scale[N] = {0.5, 2.0, -1.5};
  static double weights[K], stats[N * 5], totals[2], row_values[N];
  static unsigned char z[N * K * 8], separate[N * K * 8];
  for (int j = 0; j < K; ++j) weights[j] = 0.5 + (double)j / K;
  for (int dtype = SDCA_DTYPE_FLOAT32; dtype <= SDCA_DTYPE_BFLOAT16; ++dtype) {
    const size_t size = dtype == SDCA_DTYPE_FLOAT64 ? 8 : dtype == SDCA_DTYPE_FLOAT32 ? 4 : 2;
    for (int precision = SDCA_FLOAT32; precision <= SDCA_FLOAT64; precision += 2) {
      if (precision == SDCA_FLOAT32 && dtype == SDCA_DTYPE_FLOAT64) continue;
      struct sdca_options options = SDCA_OPTIONS_DEFAULT;
      options.precision = precision;
      for (size_t i = 0; i < N * K; ++i) {
        const double x = 3.0 * ((double)((i * 37) % 101) / 50.0 - 1.0);
        const float f = (float)x;
        uint32_t bits;
        memcpy(&bits, &f, sizeof bits);
        /* 16-bit scores: sign, exponents 13-17 (float16) or 125-129 (bfloat16), mantissa. */
        const uint16_t half =
            (uint16_t)(((i % 2) << 15) | ((13 + i % 5) << 10) | ((i * 97) % 1024));
        const uint16_t brain = (uint16_t)(bits >> 16);
        if (dtype == SDCA_DTYPE_FLOAT64) memcpy(z + i * 8, &x, 8);
        if (dtype == SDCA_DTYPE_FLOAT32) memcpy(z + i * 4, &f, 4);
        if (dtype == SDCA_DTYPE_FLOAT16) memcpy(z + i * 2, &half, 2);
        if (dtype == SDCA_DTYPE_BFLOAT16) memcpy(z + i * 2, &brain, 2);
      }
      int status = sdca_cross_entropy_forward(&options, dtype, N, K, z, target, -100, weights,
                                              row_values, stats, totals);
      if (status == SDCA_OK)
        status = sdca_cross_entropy_backward(&options, dtype, N, K, z, target, -100, weights,
                                             stats, scale, 1, separate);
      if (status == SDCA_OK)
        status = sdca_cross_entropy_backward(&options, dtype, N, K, z, target, -100, weights,
                                             stats, scale, 1, z);
      if (status != SDCA_OK || memcmp(z, separate, N * K * size) != 0) {
        ++failures;
        fprintf(stderr, "FAIL: in-place gradient dtype %d precision %d (%d, %s)\n", dtype,
                precision, status, sdca_last_error());
      }
    }
  }
}

/* SDCA_FLOAT32_FLOAT64: C = 1e40 is normal in double but not in the float duals, although
   norm2 = 1e-10 and C * norm2 = 1e30 are; an update would store duals up to C. */
static void step_scale_beyond_float(void) {
  const float z[6] = {0}, old[6] = {0};
  const double targets[6] = {0.5, 0.5, 0, 0, 0.5, 0.5};
  float out[6] = {0};
  struct sdca_options options = SDCA_OPTIONS_DEFAULT;
  options.precision = SDCA_FLOAT32_FLOAT64;
  options.c = 1e40;
  rejected(sdca_update_objective(&options, 2, 3, z, labels, NULL, old, 1e-10, out),
           "update C beyond the data type");
  rejected(sdca_update_distillation(&options, 2, 3, z, targets, old, 1e-10, out),
           "distillation update C beyond the data type");
}

int main(void) {
  in_place_gradient();
  step_scale_beyond_float();
  precision_pairs();
  projection_modes();
  fit_result_fields();
  for (size_t i = 0; i < 6; ++i) output[i] = 123.0;
  null_buffers();
  malformed_structs();
  malformed_offsets();
  overflowing_extents();
  overflowing_blas_dimensions();
  invalid_projection();
  invalid_loss_options();
  for (size_t i = 0; i < 6; ++i) {
    if (output[i] != 123.0) {
      ++failures;
      fputs("FAIL: rejected call changed its output\n", stderr);
      break;
    }
  }
  /* Optional kernel weights and single-label offsets remain nullable. */
  struct sdca_options kernel = fit_options();
  kernel.kernel = 1;
  if (sdca_fit(&kernel, 2, 1, 3, scores, labels, NULL, NULL, NULL, previous, history, 2,
               &result) != SDCA_OK ||
      strcmp(sdca_last_error(), "") != 0) {
    ++failures;
    fprintf(stderr, "FAIL: optional NULL buffers (%s)\n", sdca_last_error());
  }
  printf("c_api_validation: %d failures\n", failures);
  return failures != 0;
}
