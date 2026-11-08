/* Train a softmax model through the C ABI and evaluate a loss gradient. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sdca/c_api.h"

#define N 6
#define D 2
#define K 3

int main(void) {
  /* Row-major features (N, D) and zero-based labels. */
  const double x[N * D] = {1.0, 0.1, 0.9, -0.1, -0.5, 0.9, -0.4, 0.8, -0.5, -0.9, -0.6, -0.8};
  const int64_t y[N] = {0, 0, 1, 1, 2, 2};
  const size_t max_epoch = 100, eval_epoch = 10, rows = max_epoch / eval_epoch + 2;
  double w[D * K], a[N * K], values[1], gradient[K];
  struct sdca_options options = SDCA_OPTIONS_DEFAULT;
  struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;
  if (sdca_abi_version() != SDCA_ABI_VERSION) {
    fprintf(stderr, "libsdca_c %s has an incompatible ABI\n", sdca_version());
    return 1;
  }
  /* One history row per evaluation; column 6 is the relative duality gap. */
  const size_t fields = sdca_history_fields();
  double* history = malloc(rows * fields * sizeof(double));
  if (!history) return 1;
  options.c = 10.0;
  options.max_epoch = max_epoch;
  options.eval_epoch = eval_epoch;
  options.tol = 1e-6;
  if (sdca_fit(&options, N, D, K, x, y, NULL, NULL, w, a, history, rows, &result) != SDCA_OK) {
    fprintf(stderr, "sdca_fit: %s\n", sdca_last_error());
    free(history);
    return 1;
  }
  printf("status %d after %zu epochs, relative gap %g\n", result.status, result.epochs,
         history[(result.history_used - 1) * fields + 6]);
  free(history);

  /* Loss and gradient at one row of scores; positive marks the label. */
  const double z[K] = {0.3, -0.8, 1.2};
  const uint8_t positive[K] = {1, 0, 0};
  if (sdca_loss_gradient(&options, 1, K, z, positive, values, gradient) != SDCA_OK) {
    fprintf(stderr, "sdca_loss_gradient: %s\n", sdca_last_error());
    return 1;
  }
  printf("loss %.6f gradient %.6f %.6f %.6f\n", values[0], gradient[0], gradient[1], gradient[2]);
  return result.status == SDCA_STATUS_SOLVED ? 0 : 1;
}
