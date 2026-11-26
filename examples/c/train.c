/* Train a softmax model through the C ABI and evaluate a loss and its gradient. */
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
  double w[D * K], a[N * K], values[1], gradient[K];
  struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_CROSS_ENTROPY);
  struct sdca_fit_options options = SDCA_FIT_OPTIONS_INIT;
  struct sdca_execution execution = SDCA_EXECUTION_INIT;
  struct sdca_input input = SDCA_INPUT_INIT;
  struct sdca_targets targets = SDCA_TARGETS_INIT;
  struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;
  if (sdca_abi_version() != SDCA_ABI_VERSION) {
    fprintf(stderr, "libsdca_c %s has an incompatible ABI\n", sdca_version());
    return 1;
  }
  options.c = 10.0;
  options.max_epoch = 100;
  options.tol = 1e-6;
  input.n = N;
  input.d = D;
  input.x = x;
  targets.labels = y;
  /* One history row per evaluation, at most sdca_history_rows(&options). */
  const size_t rows = sdca_history_rows(&options);
  double* history = malloc(rows * SDCA_HISTORY_FIELDS * sizeof(double));
  if (!history) return 1;
  if (sdca_model_fit(&loss, &options, &execution, &input, K, &targets, NULL, NULL, w, a, history,
                     rows, &result) != SDCA_OK) {
    fprintf(stderr, "sdca_model_fit: %s\n", sdca_last_error());
    free(history);
    return 1;
  }
  printf("status %d after %zu epochs, relative gap %g\n", result.status, result.epochs,
         history[(result.history_rows - 1) * SDCA_HISTORY_FIELDS + SDCA_HISTORY_RELATIVE_GAP]);
  free(history);

  /* The top-2 hinge (topk_hinge_beta, k = 2) and its gradient at one row of scores. */
  const double z[K] = {0.3, -0.8, 1.2};
  const int64_t label[1] = {0};
  struct sdca_loss hinge = SDCA_LOSS_INIT(SDCA_LOSS_TOPK_HINGE_BETA);
  hinge.k = 2;
  targets.labels = label;
  if (sdca_loss_evaluate(&hinge, &execution, SDCA_DTYPE_FLOAT64, 1, K, z, &targets, values,
                         gradient) != SDCA_OK) {
    fprintf(stderr, "sdca_loss_evaluate: %s\n", sdca_last_error());
    return 1;
  }
  printf("loss %.6f gradient %.6f %.6f %.6f\n", values[0], gradient[0], gradient[1], gradient[2]);
  return result.status == SDCA_STATUS_SOLVED ? 0 : 1;
}
