/* Every entry point of the C ABI (include/sdca/c_api.h) from C: the losses with each target
 * kind, the row options of the framework bindings, the reductions, the coordinate step, fits
 * on features and Gram matrices with warm starts, centers and histories, the scores, and the
 * error paths (invalid arguments, invalid rows, check_rows). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sdca/c_api.h"

static int failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      ++failures;                                                            \
      fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #cond,         \
              sdca_last_error());                                            \
    }                                                                        \
  } while (0)

#define N 6
#define C 4
#define D 3

static int finite_all(const double* x, size_t n) {
  size_t i;
  for (i = 0; i < n; ++i)
    if (!isfinite(x[i])) return 0;
  return 1;
}

/* Each gradient row of a shift-invariant loss sums to zero up to rounding. */
static int rows_sum_to_zero(const double* g, size_t n, size_t c) {
  size_t i, j;
  for (i = 0; i < n; ++i) {
    double s = 0, a = 0;
    for (j = 0; j < c; ++j) s += g[i * c + j], a += fabs(g[i * c + j]);
    if (fabs(s) > 1e-12 * (a + 1)) return 0;
  }
  return 1;
}

static const double scores[N * C] = {
    1.0, 0.5, -0.5, 0.0,  0.2, 2.0, 0.1,  -1.0, -0.3, 0.4, 1.5, 0.0,
    0.0, 0.0, 0.0,  3.0,  2.5, -2.0, 1.0, 0.5,  0.1,  0.3, -0.7, 0.9};
static const int64_t labels[N] = {0, 1, 2, 3, 0, 1};

static void info(void) {
  struct sdca_fit_options options = SDCA_FIT_OPTIONS_INIT;
  CHECK(sdca_abi_version() == SDCA_ABI_VERSION);
  CHECK(strlen(sdca_version()) > 0);
  CHECK(strlen(sdca_blas_backend()) > 0);
  CHECK(sdca_max_extent() > 0);
  CHECK(sdca_history_rows(&options) == options.max_epoch / options.eval_epoch + 2);
  options.eval_epoch = 0;
  CHECK(sdca_history_rows(&options) == 2);
  options.c = -1;
  CHECK(sdca_history_rows(&options) == 0 && strlen(sdca_last_error()) > 0);
}

static int evaluate(struct sdca_loss loss, int dtype, const struct sdca_targets* targets,
                    double* values, double* gradients) {
  struct sdca_execution execution = SDCA_EXECUTION_INIT;
  if (dtype == SDCA_DTYPE_FLOAT32) {
    float s[N * C], g[N * C];
    size_t i;
    int status;
    for (i = 0; i < N * C; ++i) s[i] = (float)scores[i];
    status = sdca_loss_evaluate(&loss, &execution, dtype, N, C, s, targets, values, g);
    for (i = 0; i < N * C; ++i) gradients[i] = g[i];
    return status;
  }
  return sdca_loss_evaluate(&loss, &execution, dtype, N, C, scores, targets, values, gradients);
}

static void losses(void) {
  static const int kinds[] = {SDCA_LOSS_CROSS_ENTROPY, SDCA_LOSS_TOPK_CROSS_ENTROPY,
                              SDCA_LOSS_MULTICLASS_HINGE, SDCA_LOSS_TOPK_HINGE_ALPHA,
                              SDCA_LOSS_TOPK_HINGE_BETA, SDCA_LOSS_PROXIMAL_CROSS_ENTROPY};
  static const int64_t set_labels[] = {0, 1, 2, 3, 0, 1, 2, 1};
  static const int64_t offsets[N + 1] = {0, 1, 3, 4, 5, 7, 8};
  uint8_t mask[N * C] = {0};
  double probabilities[N * C], weight[C] = {1.0, 2.0, 0.5, 1.0};
  double values[N], gradients[N * C];
  struct sdca_targets targets = SDCA_TARGETS_INIT;
  size_t i, k, dtype;
  targets.labels = labels;
  for (k = 0; k < sizeof kinds / sizeof kinds[0]; ++k) {
    for (dtype = 0; dtype < 2; ++dtype) {
      struct sdca_loss loss = SDCA_LOSS_INIT(kinds[k]);
      if (kinds[k] == SDCA_LOSS_TOPK_CROSS_ENTROPY || kinds[k] == SDCA_LOSS_TOPK_HINGE_ALPHA ||
          kinds[k] == SDCA_LOSS_TOPK_HINGE_BETA)
        loss.k = 2;
      if (kinds[k] == SDCA_LOSS_MULTICLASS_HINGE) loss.gamma = dtype ? 0.5 : 0.0;
      CHECK(evaluate(loss, (int)dtype, &targets, values, gradients) == SDCA_OK);
      CHECK(finite_all(values, N) && finite_all(gradients, N * C));
      CHECK(rows_sum_to_zero(gradients, N, C) || dtype == SDCA_DTYPE_FLOAT32);
    }
  }
  /* Cross-entropy options: class weights, label smoothing and an ignored label. */
  {
    struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_CROSS_ENTROPY);
    int64_t ignored[N];
    memcpy(ignored, labels, sizeof ignored);
    ignored[2] = -100;
    targets.labels = ignored;
    loss.weight = weight, loss.label_smoothing = 0.1, loss.ignore = 1;
    CHECK(evaluate(loss, SDCA_DTYPE_FLOAT64, &targets, values, gradients) == SDCA_OK);
    CHECK(values[2] == 0.0 && gradients[2 * C] == 0.0 && values[1] > 0.0);
    targets.labels = labels;
  }
  /* Probability targets. */
  for (i = 0; i < N * C; ++i) probabilities[i] = (i % C == (size_t)labels[i / C]) ? 0.7 : 0.1;
  {
    struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_CROSS_ENTROPY);
    struct sdca_targets p = SDCA_TARGETS_INIT;
    p.kind = SDCA_TARGET_PROBABILITIES, p.probabilities = probabilities;
    CHECK(evaluate(loss, SDCA_DTYPE_FLOAT64, &p, values, gradients) == SDCA_OK);
    CHECK(finite_all(values, N) && rows_sum_to_zero(gradients, N, C));
  }
  /* Label sets and masks for the multilabel losses: the same rows give the same results. */
  for (i = 0; i < offsets[N]; ++i) {
    size_t row = 0;
    while ((int64_t)i >= offsets[row + 1]) ++row;
    mask[row * C + (size_t)set_labels[i]] = 1;
  }
  for (k = 0; k < 2; ++k) {
    struct sdca_loss loss =
        SDCA_LOSS_INIT(k ? SDCA_LOSS_MULTILABEL_CROSS_ENTROPY : SDCA_LOSS_MULTILABEL_HINGE);
    struct sdca_targets sets = SDCA_TARGETS_INIT, masks = SDCA_TARGETS_INIT;
    double v2[N], g2[N * C];
    sets.kind = SDCA_TARGET_LABEL_SETS, sets.labels = set_labels, sets.offsets = offsets;
    masks.kind = SDCA_TARGET_MASK, masks.mask = mask;
    CHECK(evaluate(loss, SDCA_DTYPE_FLOAT64, &sets, values, gradients) == SDCA_OK);
    CHECK(evaluate(loss, SDCA_DTYPE_FLOAT64, &masks, v2, g2) == SDCA_OK);
    CHECK(memcmp(values, v2, sizeof v2) == 0 && memcmp(gradients, g2, sizeof g2) == 0);
  }
  /* Distillation: a student equal to its teacher has a zero gradient. */
  for (k = 0; k < 2; ++k) {
    struct sdca_loss loss =
        SDCA_LOSS_INIT(k ? SDCA_LOSS_PROXIMAL_DISTILLATION : SDCA_LOSS_DISTILLATION);
    struct sdca_targets teacher = SDCA_TARGETS_INIT;
    teacher.kind = SDCA_TARGET_LOGITS, teacher.logits = scores;
    loss.temperature = 2.0;
    CHECK(evaluate(loss, SDCA_DTYPE_FLOAT64, &teacher, values, gradients) == SDCA_OK);
    if (!k)
      for (i = 0; i < N * C; ++i) CHECK(gradients[i] == 0.0);
  }
  /* Softmax collapse: the target gradient of a confident row stays representable. */
  {
    struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_CROSS_ENTROPY);
    struct sdca_execution execution = SDCA_EXECUTION_INIT;
    const float z[2] = {0.0f, -20.0f};
    const int64_t y = 0;
    float g[2];
    double v;
    targets.labels = &y;
    CHECK(sdca_loss_evaluate(&loss, &execution, SDCA_DTYPE_FLOAT32, 1, 2, z, &targets, &v, g) ==
          SDCA_OK);
    CHECK(fabs(v - 2.061153622438558e-9) < 1e-15 && g[0] < 0.0f && g[1] > 0.0f);
    targets.labels = labels;
  }
}

static void errors(void) {
  struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_TOPK_HINGE_BETA);
  struct sdca_execution execution = SDCA_EXECUTION_INIT;
  struct sdca_loss_report report = SDCA_LOSS_REPORT_INIT;
  struct sdca_targets targets = SDCA_TARGETS_INIT;
  double values[N], gradients[N * C], bad[N * C];
  int64_t out_of_range[N];
  targets.labels = labels;
  /* k is required for the top-k losses; a parameter of another loss is refused. */
  CHECK(sdca_loss_evaluate(&loss, &execution, SDCA_DTYPE_FLOAT64, N, C, scores, &targets,
                           values, gradients) == SDCA_INVALID_ARGUMENT);
  loss = (struct sdca_loss)SDCA_LOSS_INIT(SDCA_LOSS_MULTILABEL_HINGE);
  loss.temperature = 2.0;
  CHECK(sdca_loss_evaluate(&loss, &execution, SDCA_DTYPE_FLOAT64, N, C, scores, &targets,
                           values, gradients) == SDCA_INVALID_ARGUMENT);
  /* A label out of range is an invalid argument, as in PyTorch. */
  loss = (struct sdca_loss)SDCA_LOSS_INIT(SDCA_LOSS_CROSS_ENTROPY);
  memcpy(out_of_range, labels, sizeof out_of_range);
  out_of_range[3] = 7;
  targets.labels = out_of_range;
  CHECK(sdca_loss_evaluate(&loss, &execution, SDCA_DTYPE_FLOAT64, N, C, scores, &targets,
                           values, gradients) == SDCA_INVALID_ARGUMENT);
  CHECK(strstr(sdca_last_error(), "out of bounds") != NULL);
  targets.labels = labels;
  /* A NaN row: NaN outputs and SDCA_INVALID_ROWS, or a refusal with check_rows. */
  memcpy(bad, scores, sizeof bad);
  bad[4 * C + 1] = NAN;
  execution.report = &report;
  CHECK(sdca_loss_evaluate(&loss, &execution, SDCA_DTYPE_FLOAT64, N, C, bad, &targets, values,
                           gradients) == SDCA_INVALID_ROWS);
  CHECK(report.invalid_rows == 1 && report.first_invalid_row == 4 && isnan(values[4]) &&
        isfinite(values[3]));
  execution.check_rows = 1;
  CHECK(sdca_loss_evaluate(&loss, &execution, SDCA_DTYPE_FLOAT64, N, C, bad, &targets, values,
                           gradients) == SDCA_INVALID_ARGUMENT);
  CHECK(sdca_loss_evaluate(NULL, &execution, SDCA_DTYPE_FLOAT64, N, C, scores, &targets, values,
                           gradients) == SDCA_INVALID_ARGUMENT);
  CHECK(strcmp(sdca_last_error(), "") != 0);
}

/* float -> bfloat16 by truncation is enough for test scores. */
static uint16_t bf16(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof u);
  return (uint16_t)(u >> 16);
}

static void rows_and_reductions(void) {
  struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_CROSS_ENTROPY);
  struct sdca_row_options options = SDCA_ROW_OPTIONS_INIT;
  struct sdca_execution execution = SDCA_EXECUTION_INIT;
  struct sdca_targets targets = SDCA_TARGETS_INIT;
  const double row_upstream[N] = {1, 2, 0.5, 1, 1, 3};
  double values[N], ref[N], gradients[N * C], terms[N] = {1, 1, 2, 1, 1, 2}, d;
  uint16_t s16[N * C];
  float out[N];
  size_t i;
  targets.labels = labels;
  for (i = 0; i < N * C; ++i) s16[i] = bf16((float)scores[i]);
  /* 16-bit storage, a row factor, and the gradient written over the scores. */
  options.upstream = 0.5, options.row_upstream = row_upstream;
  CHECK(sdca_loss_rows(&loss, &options, &execution, SDCA_DTYPE_BFLOAT16, N, C, s16, &targets,
                       values, s16) == SDCA_OK);
  CHECK(finite_all(values, N));
  options = (struct sdca_row_options)SDCA_ROW_OPTIONS_INIT;
  CHECK(sdca_loss_rows(&loss, &options, &execution, SDCA_DTYPE_FLOAT64, N, C, scores, &targets,
                       values, gradients) == SDCA_OK);
  CHECK(sdca_loss_evaluate(&loss, &execution, SDCA_DTYPE_FLOAT64, N, C, scores, &targets, ref,
                           NULL) == SDCA_OK);
  CHECK(memcmp(values, ref, sizeof ref) == 0);
  /* Reductions: none, sum and mean (by the row count or by the terms). */
  CHECK(sdca_reduce_rows(SDCA_DTYPE_FLOAT32, 0, N, values, NULL, out) == SDCA_OK);
  CHECK(out[2] == (float)values[2]);
  CHECK(sdca_reduce_rows(SDCA_DTYPE_FLOAT64, 1, N, values, NULL, &d) == SDCA_OK);
  {
    double sum = 0, weight = 0;
    for (i = 0; i < N; ++i) sum += values[i], weight += terms[i];
    CHECK(fabs(d - sum) <= 1e-14 * sum);
    CHECK(sdca_reduce_rows(SDCA_DTYPE_FLOAT64, 2, N, values, NULL, &d) == SDCA_OK);
    CHECK(fabs(d - sum / N) <= 1e-14 * sum);
    CHECK(sdca_reduce_rows(SDCA_DTYPE_FLOAT64, 2, N, values, terms, &d) == SDCA_OK);
    CHECK(fabs(d - sum / weight) <= 1e-14 * sum);
  }
  {
    double groups[2];
    CHECK(sdca_reduce_row_groups(SDCA_DTYPE_FLOAT64, 1, 2, N / 2, values, NULL, groups) ==
          SDCA_OK);
    CHECK(fabs(groups[0] - (values[0] + values[1] + values[2])) <= 1e-14 * groups[0]);
    CHECK(sdca_reduce_row_groups(SDCA_DTYPE_FLOAT32, 2, 2, N / 2, values, terms, out) == SDCA_OK);
  }
  CHECK(sdca_reduce_rows(SDCA_DTYPE_FLOAT64, 5, N, values, NULL, &d) == SDCA_INVALID_ARGUMENT);
}

static void dual_updates(void) {
  static const int kinds[] = {SDCA_LOSS_CROSS_ENTROPY, SDCA_LOSS_MULTICLASS_HINGE,
                              SDCA_LOSS_TOPK_HINGE_ALPHA, SDCA_LOSS_TOPK_CROSS_ENTROPY};
  struct sdca_execution execution = SDCA_EXECUTION_INIT;
  struct sdca_targets targets = SDCA_TARGETS_INIT;
  const double norm2[N] = {1, 2, 0.5, 1, 1.5, 1};
  double first[N * C], second[N * C];
  size_t k;
  targets.labels = labels;
  for (k = 0; k < sizeof kinds / sizeof kinds[0]; ++k) {
    struct sdca_loss loss = SDCA_LOSS_INIT(kinds[k]);
    if (kinds[k] == SDCA_LOSS_TOPK_HINGE_ALPHA || kinds[k] == SDCA_LOSS_TOPK_CROSS_ENTROPY)
      loss.k = 2;
    CHECK(sdca_dual_update(&loss, 1.0, &execution, SDCA_DTYPE_FLOAT64, N, C, scores, &targets,
                           NULL, norm2, first) == SDCA_OK);
    CHECK(finite_all(first, N * C) && rows_sum_to_zero(first, N, C));
    CHECK(sdca_dual_update(&loss, 1.0, &execution, SDCA_DTYPE_FLOAT64, N, C, scores, &targets,
                           first, norm2, second) == SDCA_OK);
    CHECK(sdca_dual_update(&loss, 1.0, &execution, SDCA_DTYPE_FLOAT64, N, C, scores, &targets,
                           second, norm2, second) == SDCA_OK);
  }
  {
    struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_DISTILLATION);
    struct sdca_targets teacher = SDCA_TARGETS_INIT;
    teacher.kind = SDCA_TARGET_LOGITS, teacher.logits = scores;
    CHECK(sdca_dual_update(&loss, 1.0, &execution, SDCA_DTYPE_FLOAT64, N, C, scores, &teacher,
                           NULL, norm2, first) == SDCA_INVALID_ARGUMENT);
  }
}

static void fits(void) {
  /* Features (n, d) of N examples; the labels above. */
  static const double x[N * D] = {1.0, 0.2, -0.5, -0.3, 1.1, 0.4, 0.5, -0.9, 1.2,
                                  -1.0, -0.4, -0.8, 0.9, 0.1, -0.2, -0.2, 0.8, 0.6};
  struct sdca_fit_options options = SDCA_FIT_OPTIONS_INIT;
  struct sdca_execution execution = SDCA_EXECUTION_INIT;
  struct sdca_targets targets = SDCA_TARGETS_INIT;
  struct sdca_input input = SDCA_INPUT_INIT;
  double weights[C * D], dual[N * C], center[C * D], fitted[N * C], kscores[N * C];
  double gram[N * N], history[(1000 / 10 + 2) * SDCA_HISTORY_FIELDS];
  float x32[N * D], w32[C * D], a32[N * C];
  size_t rows = sdca_history_rows(&options), i, j, l, k;
  static const int kinds[] = {SDCA_LOSS_CROSS_ENTROPY, SDCA_LOSS_MULTICLASS_HINGE,
                              SDCA_LOSS_TOPK_HINGE_BETA, SDCA_LOSS_TOPK_CROSS_ENTROPY};
  targets.labels = labels;
  input.n = N, input.d = D, input.x = x;
  options.c = 10.0, options.tol = 1e-8, options.eval_on_start = 1;
  for (k = 0; k < sizeof kinds / sizeof kinds[0]; ++k) {
    struct sdca_loss loss = SDCA_LOSS_INIT(kinds[k]);
    struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;
    if (kinds[k] == SDCA_LOSS_TOPK_HINGE_BETA || kinds[k] == SDCA_LOSS_TOPK_CROSS_ENTROPY)
      loss.k = 2;
    CHECK(sdca_model_fit(&loss, &options, &execution, &input, C, &targets, NULL, NULL, weights,
                         dual, history, rows, &result) == SDCA_OK);
    CHECK(result.status == SDCA_STATUS_SOLVED && result.history_rows >= 2);
    CHECK(history[(result.history_rows - 1) * SDCA_HISTORY_FIELDS + SDCA_HISTORY_RELATIVE_GAP] <=
          options.tol);
    /* A warm start from the solution (in place) is solved at once. */
    result = (struct sdca_fit_result)SDCA_FIT_RESULT_INIT;
    CHECK(sdca_model_fit(&loss, &options, &execution, &input, C, &targets, NULL, dual, weights,
                         dual, NULL, 0, &result) == SDCA_OK);
    CHECK(result.status == SDCA_STATUS_SOLVED && result.epochs <= 1);
  }
  /* Scores of the fitted cross-entropy... of the last model, then a center. */
  CHECK(sdca_model_scores(&input, C, weights, fitted) == SDCA_OK);
  for (i = 0; i < C * D; ++i) center[i] = 0.1 * (double)i;
  {
    struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_MULTICLASS_HINGE);
    struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;
    options.shrinking = SDCA_SHRINKING_OFF, options.eval_epoch = 0;
    CHECK(sdca_model_fit(&loss, &options, &execution, &input, C, &targets, center, NULL, weights,
                         dual, NULL, 0, &result) == SDCA_OK);
    CHECK(result.status == SDCA_STATUS_SOLVED);
    options.shrinking = SDCA_SHRINKING_AUTO, options.eval_epoch = 10;
  }
  /* float32 features with a double iterate. */
  for (i = 0; i < N * D; ++i) x32[i] = (float)x[i];
  {
    struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_CROSS_ENTROPY);
    struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;
    struct sdca_input in32 = input;
    in32.dtype = SDCA_DTYPE_FLOAT32, in32.x = x32;
    execution.precision = SDCA_PRECISION_FLOAT64;
    options.tol = 1e-4;
    CHECK(sdca_model_fit(&loss, &options, &execution, &in32, C, &targets, NULL, NULL, w32, a32,
                         NULL, 0, &result) == SDCA_OK);
    CHECK(result.status == SDCA_STATUS_SOLVED);
    execution.precision = SDCA_PRECISION_DEFAULT;
  }
  /* A Gram matrix: the duals, and the scores of the kernel to the training examples. */
  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j) {
      double s = 0;
      for (l = 0; l < D; ++l) s += x[i * D + l] * x[j * D + l];
      gram[i * N + j] = s;
    }
  {
    struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_TOPK_HINGE_ALPHA);
    struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;
    struct sdca_input kernel = SDCA_INPUT_INIT;
    kernel.kind = SDCA_INPUT_KERNEL, kernel.n = N, kernel.d = N, kernel.x = gram;
    loss.k = 2, loss.gamma = 0.5;
    CHECK(sdca_model_fit(&loss, &options, &execution, &kernel, C, &targets, NULL, NULL, NULL,
                         dual, history, rows, &result) == SDCA_OK);
    CHECK(result.status == SDCA_STATUS_SOLVED);
    CHECK(sdca_model_scores(&kernel, C, dual, kscores) == SDCA_OK && finite_all(kscores, N * C));
  }
  /* Multilabel training on label sets. */
  {
    static const int64_t set_labels[] = {0, 1, 2, 3, 0, 1, 2, 1};
    static const int64_t offsets[N + 1] = {0, 1, 3, 4, 5, 7, 8};
    struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_MULTILABEL_HINGE);
    struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;
    struct sdca_targets sets = SDCA_TARGETS_INIT;
    sets.kind = SDCA_TARGET_LABEL_SETS, sets.labels = set_labels, sets.offsets = offsets;
    CHECK(sdca_model_fit(&loss, &options, &execution, &input, C, &sets, NULL, NULL, weights, dual,
                         NULL, 0, &result) == SDCA_OK);
    CHECK(result.status == SDCA_STATUS_SOLVED);
  }
  /* Invalid arguments: c <= 0, a history too short, distillation (no fit). */
  {
    struct sdca_loss loss = SDCA_LOSS_INIT(SDCA_LOSS_CROSS_ENTROPY);
    struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;
    CHECK(sdca_model_fit(&loss, &options, &execution, &input, C, &targets, NULL, NULL, weights,
                         dual, history, 1, &result) == SDCA_INVALID_ARGUMENT);
    options.c = 0;
    CHECK(sdca_model_fit(&loss, &options, &execution, &input, C, &targets, NULL, NULL, weights,
                         dual, NULL, 0, &result) == SDCA_INVALID_ARGUMENT);
  }
}

int main(void) {
  info();
  losses();
  errors();
  rows_and_reductions();
  dual_updates();
  fits();
  if (failures) fprintf(stderr, "%d failed checks\n", failures);
  return failures != 0;
}
