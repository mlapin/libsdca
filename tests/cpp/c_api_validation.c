/* The C ABI: malformed requests fail before writing any output, every loss takes only its
 * own parameters and targets, and the fit builds and checks its warm start. */
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "c_api_internal.h"
#include "sdca/c_api.h"

static int failures;
static double scores[6] = {0}, previous[6] = {0}, output[6], values[2];
static double norm2[2] = {1.0, 1.0}, history[2 * SDCA_HISTORY_FIELDS];
static int64_t labels[2] = {0, 1};
static uint8_t positive[6] = {1, 0, 0, 0, 1, 0};
static struct sdca_fit_result result = SDCA_FIT_RESULT_INIT;

static void fail(const char* what) {
  ++failures;
  fprintf(stderr, "FAIL: %s (%s)\n", what, sdca_last_error());
}

static void rejected(int status, const char* what) {
  if (status != SDCA_INVALID_ARGUMENT || !*sdca_last_error()) {
    ++failures;
    fprintf(stderr, "FAIL: %s returned %d (%s)\n", what, status, sdca_last_error());
  }
}

/* An accepted call; its outputs are reset for the checks of the rejected ones. */
static void accepted(int status, const char* what) {
  if (status != SDCA_OK || *sdca_last_error()) fail(what);
  for (size_t i = 0; i < 6; ++i) output[i] = 123.0;
}

static struct sdca_targets label_targets(const int64_t* y) {
  struct sdca_targets t = SDCA_TARGETS_INIT;
  t.labels = y;
  return t;
}

/* The fit options of the rejection tests: no epochs, one evaluation. */
static struct sdca_fit_options no_epochs(void) {
  struct sdca_fit_options options = SDCA_FIT_OPTIONS_INIT;
  options.max_epoch = 0;
  options.eval_epoch = 1;
  options.tol = 0.0;
  return options;
}

/* Features (2, 1) of 3 classes: scores holds x. */
static struct sdca_input two_examples(void) {
  struct sdca_input input = SDCA_INPUT_INIT;
  input.n = 2;
  input.d = 1;
  input.x = scores;
  return input;
}

static int evaluate(const struct sdca_loss* loss, const struct sdca_targets* t) {
  const struct sdca_execution e = SDCA_EXECUTION_INIT;
  return sdca_loss_evaluate(loss, &e, SDCA_DTYPE_FLOAT64, 2, 3, scores, t, values, output);
}

static int update(const struct sdca_loss* loss, const struct sdca_targets* t) {
  const struct sdca_execution e = SDCA_EXECUTION_INIT;
  return sdca_dual_update(loss, 1.0, &e, SDCA_DTYPE_FLOAT64, 2, 3, scores, t, previous, norm2,
                          output);
}

static int fit(const struct sdca_loss* loss, const struct sdca_targets* t) {
  const struct sdca_fit_options o = no_epochs();
  const struct sdca_execution e = SDCA_EXECUTION_INIT;
  const struct sdca_input in = two_examples();
  return sdca_model_fit(loss, &o, &e, &in, 3, t, NULL, NULL, output, previous, history, 2, &result);
}

static void null_buffers(void) {
  const struct sdca_loss loss = SDCA_LOSS_INIT;
  const struct sdca_execution e = SDCA_EXECUTION_INIT;
  const struct sdca_targets t = label_targets(labels), none = label_targets(NULL);
  rejected(sdca_loss_evaluate(&loss, &e, SDCA_DTYPE_FLOAT64, 2, 3, NULL, &t, values, output),
           "loss NULL scores");
  rejected(sdca_loss_evaluate(&loss, &e, SDCA_DTYPE_FLOAT64, 2, 3, scores, &t, NULL, output),
           "loss NULL values");
  rejected(evaluate(&loss, &none), "loss NULL labels");
  rejected(update(&loss, &none), "update NULL labels");
  rejected(sdca_dual_update(&loss, 1.0, &e, SDCA_DTYPE_FLOAT64, 2, 3, scores, &t, previous, NULL,
                            output),
           "update NULL norm2");
  rejected(
      sdca_dual_update(&loss, 1.0, &e, SDCA_DTYPE_FLOAT64, 2, 3, scores, &t, previous, norm2, NULL),
      "update NULL out");
  const struct sdca_fit_options o = no_epochs();
  struct sdca_input in = two_examples();
  rejected(
      sdca_model_fit(&loss, &o, &e, &in, 3, &t, NULL, NULL, NULL, previous, history, 2, &result),
      "fit NULL weights");
  rejected(sdca_model_fit(&loss, &o, &e, &in, 3, &t, NULL, NULL, output, NULL, history, 2, &result),
           "fit NULL dual");
  rejected(
      sdca_model_fit(&loss, &o, &e, &in, 3, &t, NULL, NULL, output, previous, NULL, 2, &result),
      "fit NULL history with rows");
  rejected(
      sdca_model_fit(&loss, &o, &e, &in, 3, &t, NULL, NULL, output, previous, history, 2, NULL),
      "fit NULL result");
  in.x = NULL;
  rejected(
      sdca_model_fit(&loss, &o, &e, &in, 3, &t, NULL, NULL, output, previous, history, 2, &result),
      "fit NULL x");
}

/* NULL or mis-sized structs, e.g. from another header. */
static void malformed_structs(void) {
  const struct sdca_targets t = label_targets(labels);
  const struct sdca_execution e = SDCA_EXECUTION_INIT;
  const struct sdca_fit_options o = no_epochs();
  const struct sdca_input in = two_examples();
  struct sdca_loss loss = SDCA_LOSS_INIT;
  rejected(evaluate(NULL, &t), "loss NULL loss");
  rejected(evaluate(&loss, NULL), "loss NULL targets");
  rejected(sdca_loss_evaluate(&loss, NULL, SDCA_DTYPE_FLOAT64, 2, 3, scores, &t, values, output),
           "loss NULL execution");
  rejected(sdca_model_fit(&loss, NULL, &e, &in, 3, &t, NULL, NULL, output, previous, history, 2,
                          &result),
           "fit NULL options");
  rejected(
      sdca_model_fit(&loss, &o, &e, NULL, 3, &t, NULL, NULL, output, previous, history, 2, &result),
      "fit NULL input");
  for (size_t i = 0; i < 2; ++i) {
#define SIZE(T) (i == 0 ? sizeof(T) - sizeof(int) : sizeof(T) + 8)
    struct sdca_loss sized = SDCA_LOSS_INIT;
    struct sdca_execution execution = SDCA_EXECUTION_INIT;
    struct sdca_targets targets = label_targets(labels);
    struct sdca_fit_options options = no_epochs();
    struct sdca_input input = two_examples();
    struct sdca_fit_result other = SDCA_FIT_RESULT_INIT;
    sized.size = SIZE(struct sdca_loss);
    rejected(evaluate(&sized, &t), "loss size");
    execution.size = SIZE(struct sdca_execution);
    rejected(
        sdca_loss_evaluate(&loss, &execution, SDCA_DTYPE_FLOAT64, 2, 3, scores, &t, values, output),
        "execution size");
    targets.size = SIZE(struct sdca_targets);
    rejected(evaluate(&loss, &targets), "targets size");
    options.size = SIZE(struct sdca_fit_options);
    rejected(sdca_model_fit(&loss, &options, &e, &in, 3, &t, NULL, NULL, output, previous, history,
                            2, &result),
             "fit options size");
    input.size = SIZE(struct sdca_input);
    rejected(sdca_model_fit(&loss, &o, &e, &input, 3, &t, NULL, NULL, output, previous, history, 2,
                            &result),
             "input size");
    other.size = SIZE(struct sdca_fit_result);
    rejected(
        sdca_model_fit(&loss, &o, &e, &in, 3, &t, NULL, NULL, output, previous, history, 2, &other),
        "fit result size");
    if (sdca_history_rows(&options) != 0 || !*sdca_last_error()) fail("history rows size");
#undef SIZE
  }
  loss.kind = 10;
  rejected(evaluate(&loss, &t), "unknown loss kind");
}

/* A loss takes only its own parameters; the defaults of the others are k 0, gamma 0, alpha 1,
 * temperature 1 and projection_precision FLOAT64. */
static void foreign_parameters(void) {
  const struct sdca_targets t = label_targets(labels);
  struct sdca_loss loss = SDCA_LOSS_INIT;
  loss.k = 1;
  rejected(evaluate(&loss, &t), "cross_entropy with k");
  loss = (struct sdca_loss)SDCA_LOSS_INIT;
  loss.gamma = 0.5;
  rejected(evaluate(&loss, &t), "cross_entropy with gamma");
  rejected(update(&loss, &t), "update cross_entropy with gamma");
  rejected(fit(&loss, &t), "fit cross_entropy with gamma");
  loss = (struct sdca_loss)SDCA_LOSS_INIT;
  loss.alpha = 2.0;
  rejected(evaluate(&loss, &t), "cross_entropy with alpha");
  loss = (struct sdca_loss)SDCA_LOSS_INIT;
  loss.temperature = 2.0;
  rejected(evaluate(&loss, &t), "cross_entropy with temperature");
  loss = (struct sdca_loss)SDCA_LOSS_INIT;
  loss.projection_precision = SDCA_PROJECTION_EXTENDED;
  rejected(evaluate(&loss, &t), "cross_entropy with projection_precision");
  loss = (struct sdca_loss)SDCA_LOSS_INIT;
  loss.kind = SDCA_LOSS_MULTICLASS_HINGE;
  loss.k = 2;
  rejected(evaluate(&loss, &t), "multiclass_hinge with k");
  loss.k = 0;
  loss.gamma = 0.5;
  accepted(evaluate(&loss, &t), "multiclass_hinge with gamma");
  loss = (struct sdca_loss)SDCA_LOSS_INIT;
  loss.kind = SDCA_LOSS_TOPK_HINGE_BETA;
  rejected(evaluate(&loss, &t), "topk_hinge_beta without k");
  loss.k = 3;
  rejected(evaluate(&loss, &t), "topk_hinge_beta with k = num_classes");
  loss.k = 2;
  accepted(evaluate(&loss, &t), "topk_hinge_beta with k = 2");
  loss.gamma = -1.0;
  rejected(evaluate(&loss, &t), "negative gamma");
  loss = (struct sdca_loss)SDCA_LOSS_INIT;
  loss.kind = SDCA_LOSS_PROXIMAL_CROSS_ENTROPY;
  loss.alpha = 0.0;
  rejected(evaluate(&loss, &t), "proximal_cross_entropy with alpha 0");
  loss.alpha = 0.5;
  accepted(evaluate(&loss, &t), "proximal_cross_entropy with alpha 0.5");
  rejected(update(&loss, &t), "proximal_cross_entropy is not trainable");
  rejected(fit(&loss, &t), "fit proximal_cross_entropy");
}

/* Each loss takes its own kind of targets, with only that kind's fields. */
static void target_kinds(void) {
  struct sdca_loss loss = SDCA_LOSS_INIT;
  struct sdca_targets t = SDCA_TARGETS_INIT;
  t.kind = SDCA_TARGET_MASK;
  t.mask = positive;
  rejected(evaluate(&loss, &t), "cross_entropy with a mask");
  loss.kind = SDCA_LOSS_MULTILABEL_CROSS_ENTROPY;
  accepted(evaluate(&loss, &t), "multilabel_cross_entropy with a mask");
  t.labels = labels;
  rejected(evaluate(&loss, &t), "a mask with labels");
  t = label_targets(labels);
  rejected(evaluate(&loss, &t), "multilabel_cross_entropy with labels");
  const int64_t sets[3] = {0, 1, 2}, offsets[3] = {0, 1, 3}, unsorted[3] = {0, 2, 1};
  t.kind = SDCA_TARGET_LABEL_SETS;
  t.labels = sets;
  t.offsets = offsets;
  accepted(evaluate(&loss, &t), "multilabel_cross_entropy with label sets");
  accepted(update(&loss, &t), "update multilabel_cross_entropy with label sets");
  t.labels = unsorted;
  rejected(evaluate(&loss, &t), "unsorted label sets");
  const int64_t bad_offsets[][3] = {{0, 1, INT64_MIN}, {0, -1, 0}, {0, 1, 1}, {1, 1, 3}};
  t.labels = sets;
  for (size_t i = 0; i < sizeof bad_offsets / sizeof bad_offsets[0]; ++i) {
    t.offsets = bad_offsets[i];
    rejected(evaluate(&loss, &t), "malformed offsets");
    rejected(update(&loss, &t), "update malformed offsets");
  }
  /* A set of every class: the steps need a class outside it, and the losses follow them. */
  const int64_t full[4] = {0, 0, 1, 2}, full_offsets[3] = {0, 1, 4};
  t.labels = full;
  t.offsets = full_offsets;
  rejected(evaluate(&loss, &t), "multilabel_cross_entropy with a full set");
  rejected(update(&loss, &t), "update multilabel_cross_entropy with a full set");
  loss.kind = SDCA_LOSS_MULTILABEL_HINGE;
  rejected(evaluate(&loss, &t), "multilabel_hinge with a full set");
  rejected(update(&loss, &t), "update multilabel_hinge with a full set");
  /* Probabilities train cross_entropy (the distillation objective) and nothing else. */
  const double p[6] = {0.5, 0.5, 0, 0, 0.25, 0.75}, off[6] = {0.5, 0.6, 0, 0, 0.25, 0.75};
  t = (struct sdca_targets)SDCA_TARGETS_INIT;
  t.kind = SDCA_TARGET_PROBABILITIES;
  t.probabilities = p;
  loss = (struct sdca_loss)SDCA_LOSS_INIT;
  rejected(evaluate(&loss, &t), "loss with probabilities");
  accepted(update(&loss, &t), "update with probabilities");
  t.probabilities = off;
  rejected(update(&loss, &t), "probabilities off the simplex");
  loss.kind = SDCA_LOSS_TOPK_CROSS_ENTROPY;
  loss.k = 2;
  t.probabilities = p;
  rejected(update(&loss, &t), "topk_cross_entropy with probabilities");
  /* Teacher logits: the distillation losses only. */
  t = (struct sdca_targets)SDCA_TARGETS_INIT;
  t.kind = SDCA_TARGET_LOGITS;
  t.logits = previous;
  loss = (struct sdca_loss)SDCA_LOSS_INIT;
  loss.kind = SDCA_LOSS_DISTILLATION;
  loss.temperature = 2.0;
  accepted(evaluate(&loss, &t), "distillation with teacher logits");
  rejected(update(&loss, &t), "distillation is not trainable");
  loss.alpha = 2.0;
  rejected(evaluate(&loss, &t), "distillation with alpha");
  loss.kind = SDCA_LOSS_PROXIMAL_DISTILLATION;
  accepted(evaluate(&loss, &t), "proximal_distillation with alpha");
  t.kind = 9;
  rejected(evaluate(&loss, &t), "unknown target kind");
}

static void overflowing_extents(void) {
  const size_t n = SIZE_MAX / sizeof(double) / 3 + 1;
  const struct sdca_loss loss = SDCA_LOSS_INIT;
  const struct sdca_execution e = SDCA_EXECUTION_INIT;
  const struct sdca_targets t = label_targets(labels);
  rejected(sdca_loss_evaluate(&loss, &e, SDCA_DTYPE_FLOAT64, n, 3, scores, &t, values, output),
           "loss byte extent overflow");
  rejected(sdca_dual_update(&loss, 1.0, &e, SDCA_DTYPE_FLOAT64, n, 3, scores, &t, previous, norm2,
                            output),
           "update byte extent overflow");
  const struct sdca_fit_options o = no_epochs();
  struct sdca_input in = two_examples();
  in.n = n;
  rejected(
      sdca_model_fit(&loss, &o, &e, &in, 3, &t, NULL, NULL, output, previous, history, 2, &result),
      "fit dual byte extent overflow");
  in = two_examples();
  rejected(sdca_model_fit(&loss, &o, &e, &in, 3, &t, NULL, NULL, output, previous, history,
                          SIZE_MAX / 10, &result),
           "fit history byte extent overflow");
  /* Scores: (n, d), (n, num_classes) and (num_classes, d) beyond the addressable bytes. */
  const size_t extents[][3] = {{n, 3, 1}, {n, 1, 3}, {1, n, 3}};
  for (size_t i = 0; i < sizeof extents / sizeof extents[0]; ++i) {
    in = two_examples();
    in.n = extents[i][0];
    in.d = extents[i][1];
    rejected(sdca_model_scores(&in, extents[i][2], scores, output), "scores extent overflow");
  }
}

static void overflowing_blas_dimensions(void) {
#if !defined(BLAS_MKL) && !defined(BLAS_MATLAB)
  const size_t large = (size_t)INT_MAX + 1;
  const size_t shapes[][3] = {
      {large, 1, 2}, {1, large, 2}, {1, 1, large}, {1, large / 2, 2}, {large / 2, 1, 2}};
  const struct sdca_loss loss = SDCA_LOSS_INIT;
  const struct sdca_fit_options o = no_epochs();
  const struct sdca_execution e = SDCA_EXECUTION_INIT;
  const struct sdca_targets t = label_targets(labels);
  for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; ++i) {
    struct sdca_input in = two_examples();
    in.n = shapes[i][0];
    in.d = shapes[i][1];
    rejected(sdca_model_fit(&loss, &o, &e, &in, shapes[i][2], &t, NULL, NULL, output, previous,
                            history, 2, &result),
             "fit BLAS integer overflow");
  }
  /* Scores are one product: each dimension must fit the BLAS integer, their products need
   * not (the last two shapes pass that check; their NULL buffers are then rejected). */
  for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; ++i) {
    struct sdca_input in = two_examples();
    in.n = shapes[i][0];
    in.d = shapes[i][1];
    const int status = sdca_model_scores(&in, shapes[i][2], NULL, output);
    rejected(status, "scores BLAS integer overflow");
    const int limit = strstr(sdca_last_error(), "BLAS integer limit") != NULL;
    if (limit != (i < 3)) fail("scores check each dimension against the BLAS integer limit");
  }
#endif
}

static void invalid_settings(void) {
  const struct sdca_targets t = label_targets(labels);
  struct sdca_loss loss = SDCA_LOSS_INIT;
  struct sdca_execution e = SDCA_EXECUTION_INIT;
  e.precision = 3;
  rejected(sdca_loss_evaluate(&loss, &e, SDCA_DTYPE_FLOAT64, 2, 3, scores, &t, values, output),
           "unknown precision");
  e.precision = SDCA_PRECISION_FLOAT32;
  rejected(sdca_loss_evaluate(&loss, &e, SDCA_DTYPE_FLOAT64, 2, 3, scores, &t, values, output),
           "float32 work of float64 storage");
  e.precision = SDCA_PRECISION_DEFAULT;
  rejected(sdca_loss_evaluate(&loss, &e, 2, 2, 3, scores, &t, values, output), "16-bit storage");
  loss.kind = SDCA_LOSS_MULTILABEL_HINGE;
  loss.projection_precision = 2;
  rejected(evaluate(&loss, &t), "unknown projection precision");
  const struct sdca_loss ce = SDCA_LOSS_INIT;
  const struct sdca_input in = two_examples();
  const int bad[][4] = {{2, 1, 0, 0}, {0, 2, 0, 0}, {0, 1, 5, 0}, {0, 1, -1, 0}};
  for (size_t i = 0; i < sizeof bad / sizeof bad[0]; ++i) {
    struct sdca_fit_options o = no_epochs();
    o.eval_on_start = bad[i][0];
    o.recompute_primal = bad[i][1];
    o.shrinking = bad[i][2];
    rejected(
        sdca_model_fit(&ce, &o, &e, &in, 3, &t, NULL, NULL, output, previous, history, 2, &result),
        "invalid fit options");
  }
  struct sdca_fit_options o = no_epochs();
  o.c = 0.0;
  rejected(
      sdca_model_fit(&ce, &o, &e, &in, 3, &t, NULL, NULL, output, previous, history, 2, &result),
      "C = 0");
  o = no_epochs();
  o.tol = NAN;
  rejected(
      sdca_model_fit(&ce, &o, &e, &in, 3, &t, NULL, NULL, output, previous, history, 2, &result),
      "tol NaN");
  o = no_epochs();
  o.max_epoch = 10;
  rejected(
      sdca_model_fit(&ce, &o, &e, &in, 3, &t, NULL, NULL, output, previous, history, 2, &result),
      "a history shorter than sdca_history_rows");
  /* A Gram matrix takes neither weights nor a center, and must be square. */
  struct sdca_input kernel = two_examples();
  kernel.kind = SDCA_INPUT_KERNEL;
  o = no_epochs();
  rejected(sdca_model_fit(&ce, &o, &e, &kernel, 3, &t, NULL, NULL, output, previous, history, 2,
                          &result),
           "kernel with weights");
  kernel.d = 2;
  if (sdca_model_fit(&ce, &o, &e, &kernel, 3, &t, NULL, NULL, NULL, previous, history, 2,
                     &result) != SDCA_OK)
    fail("kernel fit without weights");
  kernel.d = 3;
  rejected(
      sdca_model_fit(&ce, &o, &e, &kernel, 3, &t, NULL, NULL, NULL, previous, history, 2, &result),
      "a non-square Gram matrix");
}

/* Every loss rejects k, gamma and alpha out of range. */
static void invalid_loss_values(void) {
  const struct sdca_targets t = label_targets(labels);
  struct sdca_loss loss = SDCA_LOSS_INIT;
  loss.kind = SDCA_LOSS_TOPK_CROSS_ENTROPY;
  loss.k = 0;
  rejected(evaluate(&loss, &t), "k 0");
  loss.kind = SDCA_LOSS_TOPK_HINGE_ALPHA;
  loss.k = 1;
  loss.gamma = INFINITY;
  rejected(evaluate(&loss, &t), "gamma infinite");
  loss = (struct sdca_loss)SDCA_LOSS_INIT;
  loss.kind = SDCA_LOSS_DISTILLATION;
  loss.temperature = -1.0;
  struct sdca_targets logits = SDCA_TARGETS_INIT;
  logits.kind = SDCA_TARGET_LOGITS;
  logits.logits = previous;
  rejected(evaluate(&loss, &logits), "negative temperature");
  scores[3] = NAN;
  loss = (struct sdca_loss)SDCA_LOSS_INIT;
  rejected(evaluate(&loss, &t), "NaN scores");
  scores[3] = 0.0;
  const int64_t out_of_range[2] = {0, 3};
  const struct sdca_targets far = label_targets(out_of_range);
  rejected(evaluate(&loss, &far), "a label out of range");
}

/* Both projection precisions fit a smooth multilabel hinge alike to rounding. Shrinking is off:
   its keep-or-drop tests would turn rounding differences into different paths. */
static void projection_modes(void) {
  const double x[8] = {1.0, 0.2, -0.5, 1.1, 0.3, -1.2, -0.7, 0.4};
  const int64_t sets[6] = {0, 2, 1, 0, 1, 2}, offsets[5] = {0, 2, 3, 4, 6};
  const int modes[2] = {SDCA_PROJECTION_FLOAT64, SDCA_PROJECTION_EXTENDED};
  double w[2][6], a[2][12];
  struct sdca_fit_result fit_result = SDCA_FIT_RESULT_INIT;
  struct sdca_loss loss = SDCA_LOSS_INIT;
  struct sdca_fit_options options = SDCA_FIT_OPTIONS_INIT;
  struct sdca_execution e = SDCA_EXECUTION_INIT;
  struct sdca_input in = SDCA_INPUT_INIT;
  struct sdca_targets t = SDCA_TARGETS_INIT;
  loss.kind = SDCA_LOSS_MULTILABEL_HINGE;
  loss.gamma = 0.5;
  options.max_epoch = 20;
  options.eval_epoch = 20;
  options.tol = 0.0;
  options.shrinking = SDCA_SHRINKING_OFF;
  in.n = 4;
  in.d = 2;
  in.x = x;
  t.kind = SDCA_TARGET_LABEL_SETS;
  t.labels = sets;
  t.offsets = offsets;
  for (int m = 0; m < 2; ++m) {
    loss.projection_precision = modes[m];
    if (sdca_model_fit(&loss, &options, &e, &in, 3, &t, NULL, NULL, w[m], a[m], NULL, 0,
                       &fit_result) != SDCA_OK) {
      fail("projection mode fit");
      return;
    }
  }
  for (size_t i = 0; i < 6; ++i) {
    const double diff = w[0][i] - w[1][i], scale = 1.0 + fabs(w[0][i]);
    if (!(fabs(diff) < 1e-12 * scale)) {
      fail("the projection precisions disagree beyond rounding");
      return;
    }
  }
}

static void precision_pairs(void) {
  float z[3] = {0, 0, 0}, gradient[3];
  const int64_t y[1] = {0};
  const struct sdca_targets t = label_targets(y);
  const struct sdca_loss loss = SDCA_LOSS_INIT;
  double value;
  struct sdca_execution e = SDCA_EXECUTION_INIT;
  for (int precision = SDCA_PRECISION_DEFAULT; precision <= SDCA_PRECISION_FLOAT64; ++precision) {
    e.precision = precision;
    if (sdca_loss_evaluate(&loss, &e, SDCA_DTYPE_FLOAT32, 1, 3, z, &t, &value, gradient) !=
            SDCA_OK ||
        !(value > 1.09 && value < 1.10) || !(gradient[0] < -0.66f && gradient[1] > 0.33f))
      fail("float32 storage at every working precision");
  }
  /* The values alone. */
  e.precision = SDCA_PRECISION_DEFAULT;
  if (sdca_loss_evaluate(&loss, &e, SDCA_DTYPE_FLOAT32, 1, 3, z, &t, &value, NULL) != SDCA_OK ||
      !(value > 1.09 && value < 1.10))
    fail("the values alone");
}

/* The fit result reports the status, the epochs and the history rows written. */
static void fit_result_fields(void) {
  const double x[6] = {1.0, 0.1, -0.5, 0.9, -0.4, -0.8};
  const int64_t y[3] = {0, 1, 2};
  double w[6], a[9], hist[7 * SDCA_HISTORY_FIELDS];
  struct sdca_fit_result fit_result = SDCA_FIT_RESULT_INIT;
  const struct sdca_loss loss = SDCA_LOSS_INIT;
  struct sdca_fit_options options = SDCA_FIT_OPTIONS_INIT;
  const struct sdca_execution e = SDCA_EXECUTION_INIT;
  struct sdca_input in = SDCA_INPUT_INIT;
  const struct sdca_targets t = label_targets(y);
  options.max_epoch = 50;
  options.tol = 0.0;
  in.n = 3;
  in.d = 2;
  in.x = x;
  if (sdca_history_rows(&options) != 7) fail("sdca_history_rows");
  const int code =
      sdca_model_fit(&loss, &options, &e, &in, 3, &t, NULL, NULL, w, a, hist, 7, &fit_result);
  const size_t used = fit_result.history_rows;
  if (code != SDCA_OK || fit_result.size != sizeof fit_result ||
      fit_result.status != SDCA_STATUS_MAX_EPOCH || fit_result.epochs != 50 || used < 1 ||
      used > 7 || hist[(used - 1) * SDCA_HISTORY_FIELDS + SDCA_HISTORY_EPOCH] != 50.0)
    fail("fit result");
  /* eval_epoch 0: only the end. */
  options.eval_epoch = 0;
  if (sdca_history_rows(&options) != 2) fail("sdca_history_rows of eval_epoch 0");
  if (sdca_model_fit(&loss, &options, &e, &in, 3, &t, NULL, NULL, w, a, hist, 2, &fit_result) !=
          SDCA_OK ||
      fit_result.history_rows != 1 || hist[SDCA_HISTORY_EPOCH] != 50.0)
    fail("eval_epoch 0 evaluates at the end");
}

/* A seed orders the epochs: the same seed repeats a fit; another solves it too. */
static void seeds(void) {
  const double x[12] = {1.0, 0.1, 0.9, -0.1, -0.5, 0.9, -0.4, 0.8, -0.5, -0.9, -0.6, -0.8};
  const int64_t y[6] = {0, 0, 1, 1, 2, 2};
  double w[3][6], a[3][18];
  const struct sdca_loss loss = SDCA_LOSS_INIT;
  struct sdca_fit_options options = SDCA_FIT_OPTIONS_INIT;
  const struct sdca_execution e = SDCA_EXECUTION_INIT;
  struct sdca_input in = SDCA_INPUT_INIT;
  const struct sdca_targets t = label_targets(y);
  struct sdca_fit_result r[3] = {SDCA_FIT_RESULT_INIT, SDCA_FIT_RESULT_INIT, SDCA_FIT_RESULT_INIT};
  const uint64_t seed[3] = {7, 7, 8};
  options.max_epoch = 3;
  options.tol = 0.0;
  in.n = 6;
  in.d = 2;
  in.x = x;
  for (int i = 0; i < 3; ++i) {
    options.seed = seed[i];
    if (sdca_model_fit(&loss, &options, &e, &in, 3, &t, NULL, NULL, w[i], a[i], NULL, 0, &r[i]) !=
        SDCA_OK)
      fail("seeded fit");
  }
  if (memcmp(a[0], a[1], sizeof a[0]) != 0) fail("the same seed repeats a fit");
  if (memcmp(a[0], a[2], sizeof a[0]) == 0) fail("another seed takes another order");
}

/* A warm start: the library checks the duals, makes single-label rows sum to zero and builds
 * W = center + x' a; infeasible duals fail without output. */
static void warm_start(void) {
  const double x[12] = {1.0, 0.1, 0.9, -0.1, -0.5, 0.9, -0.4, 0.8, -0.5, -0.9, -0.6, -0.8};
  const int64_t y[6] = {0, 0, 1, 1, 2, 2};
  const double center[6] = {0.1, -0.2, 0.3, 0.0, -0.1, 0.2};
  double w[6], a[18], w0[6], a0[18];
  const struct sdca_loss loss = SDCA_LOSS_INIT;
  struct sdca_fit_options options = SDCA_FIT_OPTIONS_INIT;
  const struct sdca_execution e = SDCA_EXECUTION_INIT;
  struct sdca_input in = SDCA_INPUT_INIT;
  const struct sdca_targets t = label_targets(y);
  struct sdca_fit_result r = SDCA_FIT_RESULT_INIT;
  options.max_epoch = 5;
  options.tol = 0.0;
  in.n = 6;
  in.d = 2;
  in.x = x;
  if (sdca_model_fit(&loss, &options, &e, &in, 3, &t, center, NULL, w0, a0, NULL, 0, &r) != SDCA_OK)
    fail("a cold fit");
  /* No epochs from the duals of the cold fit: W = center + x' a, as the fit returned it. */
  options.max_epoch = 0;
  options.eval_epoch = 0;
  memcpy(a, a0, sizeof a);
  if (sdca_model_fit(&loss, &options, &e, &in, 3, &t, center, a, w, a, NULL, 0, &r) != SDCA_OK)
    fail("a warm fit in place");
  for (int j = 0; j < 6; ++j)
    if (fabs(w[j] - w0[j]) > 1e-12) fail("the warm W is center + x' a");
  /* Infeasible: a positive entry of another class. */
  double bad[18];
  memcpy(bad, a0, sizeof bad);
  bad[1] = fabs(bad[1]) + 0.5;
  bad[0] -= fabs(bad[1]);
  for (int j = 0; j < 6; ++j) w[j] = 123.0;
  rejected(sdca_model_fit(&loss, &options, &e, &in, 3, &t, center, bad, w, a, NULL, 0, &r),
           "an infeasible warm start");
  for (int j = 0; j < 6; ++j)
    if (w[j] != 123.0) fail("a rejected warm start writes no weights");
  /* Rows that do not sum to zero. */
  memcpy(bad, a0, sizeof bad);
  bad[2] -= 0.25;
  rejected(sdca_model_fit(&loss, &options, &e, &in, 3, &t, center, bad, w, a, NULL, 0, &r),
           "warm duals that do not sum to zero");
}

/* sdca_model_scores: x weights' of row-major features (n, d) and weights (num_classes, d),
 * and x dual of a cross-kernel (n, d) and duals (d, num_classes), in both storage types.
 * The entries are small dyadic numbers, so every sum is exact in any BLAS. */
static void model_scores(void) {
  const double x[6] = {1.0, 2.0, -0.5, 0.25, 3.0, -1.0};          /* (3, 2) */
  const double w[8] = {1.0, 0.0, 0.0, 1.0, -1.0, 2.0, 0.5, 0.5};  /* (4, 2): one row per class */
  const double a[8] = {1.0, -1.0, 0.0, 2.0, 0.5, 0.0, -2.0, 1.0}; /* (2, 4) */
  float xf[6], wf[8], af[8], zf[12];
  double z[12];
  for (int j = 0; j < 6; ++j) xf[j] = (float)x[j];
  for (int j = 0; j < 8; ++j) wf[j] = (float)w[j], af[j] = (float)a[j];
  for (int kernel = 0; kernel < 2; ++kernel) {
    struct sdca_input in = SDCA_INPUT_INIT;
    in.kind = kernel ? SDCA_INPUT_KERNEL : SDCA_INPUT_FEATURES;
    in.n = 3;
    in.d = 2;
    in.x = x;
    if (sdca_model_scores(&in, 4, kernel ? a : w, z) != SDCA_OK || *sdca_last_error())
      fail("float64 scores");
    in.dtype = SDCA_DTYPE_FLOAT32;
    in.x = xf;
    if (sdca_model_scores(&in, 4, kernel ? af : wf, zf) != SDCA_OK) fail("float32 scores");
    for (int i = 0; i < 3; ++i)
      for (int c = 0; c < 4; ++c) {
        const double expected = kernel ? x[2 * i] * a[c] + x[2 * i + 1] * a[4 + c]
                                       : x[2 * i] * w[2 * c] + x[2 * i + 1] * w[2 * c + 1];
        if (z[4 * i + c] != expected) fail("float64 scores are x weights' (x dual)");
        if ((double)zf[4 * i + c] != expected) fail("float32 scores are x weights' (x dual)");
      }
    /* No examples: no buffer but the coefficients is read or written. */
    in.n = 0;
    in.x = NULL;
    if (sdca_model_scores(&in, 4, kernel ? af : wf, NULL) != SDCA_OK) fail("scores of no examples");
  }
  /* Rejected calls write no scores. */
  for (int j = 0; j < 12; ++j) z[j] = 123.0;
  struct sdca_input in = SDCA_INPUT_INIT;
  in.n = 3;
  in.d = 2;
  in.x = x;
  rejected(sdca_model_scores(NULL, 4, w, z), "scores NULL input");
  rejected(sdca_model_scores(&in, 4, NULL, z), "scores NULL coefficients");
  rejected(sdca_model_scores(&in, 4, w, NULL), "scores NULL output");
  rejected(sdca_model_scores(&in, 0, w, z), "scores of no classes");
  struct sdca_input bad = in;
  bad.x = NULL;
  rejected(sdca_model_scores(&bad, 4, w, z), "scores NULL x");
  bad = in;
  bad.d = 0;
  rejected(sdca_model_scores(&bad, 4, w, z), "scores of no features");
  bad = in;
  bad.kind = 7;
  rejected(sdca_model_scores(&bad, 4, w, z), "scores of an unknown input kind");
  bad = in;
  bad.dtype = SDCA_INTERNAL_DTYPE_FLOAT16;
  rejected(sdca_model_scores(&bad, 4, w, z), "scores of 16-bit storage");
  bad = in;
  bad.size = sizeof bad - 1;
  rejected(sdca_model_scores(&bad, 4, w, z), "scores of a truncated input struct");
  for (int j = 0; j < 12; ++j)
    if (z[j] != 123.0) fail("a rejected scores call writes no output");
}

/* The float32 storage / float64 work pair: C = 1e40 is normal in double but not in the float
   duals, although norm2 = 1e-10 and C * norm2 = 1e30 are; an update would store duals up to C. */
static void step_scale_beyond_float(void) {
  const float z[6] = {0}, old[6] = {0};
  const double small[2] = {1e-10, 1e-10};
  const double p[6] = {0.5, 0.5, 0, 0, 0.5, 0.5};
  float out[6] = {0};
  const struct sdca_loss loss = SDCA_LOSS_INIT;
  struct sdca_execution e = SDCA_EXECUTION_INIT;
  struct sdca_targets t = label_targets(labels);
  e.precision = SDCA_PRECISION_FLOAT64;
  rejected(sdca_dual_update(&loss, 1e40, &e, SDCA_DTYPE_FLOAT32, 2, 3, z, &t, old, small, out),
           "update C beyond the storage type");
  t = (struct sdca_targets)SDCA_TARGETS_INIT;
  t.kind = SDCA_TARGET_PROBABILITIES;
  t.probabilities = p;
  rejected(sdca_dual_update(&loss, 1e40, &e, SDCA_DTYPE_FLOAT32, 2, 3, z, &t, old, small, out),
           "distillation update C beyond the storage type");
}

/* sdca_internal_cross_entropy_backward may write the gradient over its scores: the same bits as
 * into a separate buffer, in every storage type and working precision, with class weights, an
 * ignored row and one scale per row. */
static void in_place_gradient(void) {
  enum { N = 3, K = 37 };
  static const int64_t target[N] = {4, -100, 36};
  static const double scale[N] = {0.5, 2.0, -1.5};
  static double weights[K], stats[N * 5], totals[2], row_values[N];
  static unsigned char z[N * K * 8], separate[N * K * 8];
  for (int j = 0; j < K; ++j) weights[j] = 0.5 + (double)j / K;
  for (int dtype = SDCA_DTYPE_FLOAT32; dtype <= SDCA_INTERNAL_DTYPE_BFLOAT16; ++dtype) {
    const size_t size = dtype == SDCA_DTYPE_FLOAT64 ? 8 : dtype == SDCA_DTYPE_FLOAT32 ? 4 : 2;
    for (int float_work = 0; float_work <= 1; ++float_work) {
      if (float_work && dtype == SDCA_DTYPE_FLOAT64) continue;
      struct sdca_internal_options options = {
          sizeof(struct sdca_internal_options), 0, 0, 1, 0.0, 1.0, SDCA_PROJECTION_FLOAT64, 0};
      options.float_work = float_work;
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
        if (dtype == SDCA_INTERNAL_DTYPE_FLOAT16) memcpy(z + i * 2, &half, 2);
        if (dtype == SDCA_INTERNAL_DTYPE_BFLOAT16) memcpy(z + i * 2, &brain, 2);
      }
      int status = sdca_internal_cross_entropy_forward(&options, dtype, N, K, z, target, -100,
                                                       weights, row_values, stats, totals);
      if (status == SDCA_OK)
        status = sdca_internal_cross_entropy_backward(&options, dtype, N, K, z, target, -100,
                                                      weights, stats, scale, 1, separate);
      if (status == SDCA_OK)
        status = sdca_internal_cross_entropy_backward(&options, dtype, N, K, z, target, -100,
                                                      weights, stats, scale, 1, z);
      if (status != SDCA_OK || memcmp(z, separate, N * K * size) != 0) {
        ++failures;
        fprintf(stderr, "FAIL: in-place gradient dtype %d float_work %d (%d, %s)\n", dtype,
                float_work, status, sdca_last_error());
      }
    }
  }
}

static void abort_on_row(void* context, size_t begin, size_t end) {
  (void)context;
  (void)begin;
  (void)end;
  ++failures;
  fputs("FAIL: sdca_internal_run_rows called callback on empty range\n", stderr);
}

static void mark_rows(void* context, size_t begin, size_t end) {
  unsigned char* seen = (unsigned char*)context;
  size_t i;
  for (i = begin; i < end; ++i) seen[i] = 1;
}

static void run_rows_bounds_and_empty(void) {
  unsigned char seen[8] = {0};
  size_t i;
  rejected(sdca_internal_run_rows(1, 1, 1, NULL, NULL), "null row callback");
  if (sdca_internal_run_rows(0, 1000, 8, abort_on_row, NULL) != SDCA_OK)
    fail("sdca_internal_run_rows rejected empty batch");
  if (sdca_internal_run_rows(8, 1000, 100000, mark_rows, seen) != SDCA_OK)
    fail("sdca_internal_run_rows uncapped threads");
  for (i = 0; i < 8; ++i) {
    if (seen[i] != 1) {
      fail("sdca_internal_run_rows missed row under capped threads");
      break;
    }
  }
}

int main(void) {
  if (sdca_abi_version() != SDCA_ABI_VERSION) fail("ABI version");
  for (size_t i = 0; i < 6; ++i) output[i] = 123.0;
  in_place_gradient();
  step_scale_beyond_float();
  run_rows_bounds_and_empty();
  precision_pairs();
  projection_modes();
  fit_result_fields();
  seeds();
  warm_start();
  model_scores();
  target_kinds();
  null_buffers();
  malformed_structs();
  foreign_parameters();
  overflowing_extents();
  overflowing_blas_dimensions();
  invalid_settings();
  invalid_loss_values();
  for (size_t i = 0; i < 6; ++i) {
    if (output[i] != 123.0) {
      fail("rejected call changed its output");
      break;
    }
  }
  printf("c_api_validation: %d failures\n", failures);
  return failures != 0;
}
