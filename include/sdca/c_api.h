#ifndef SDCA_C_API_H
#define SDCA_C_API_H

/*
 * The C ABI of libsdca (src/c_api.cpp), used by the Python and Rust packages.
 *
 * Entry points: sdca_loss_evaluate (losses and gradients), sdca_dual_update
 * (coordinate steps), sdca_model_fit (training) and sdca_history_rows (the
 * history a fit needs); sdca_abi_version, sdca_version and sdca_last_error
 * describe the library and the last failure.
 *
 * Arguments. A loss (struct sdca_loss) takes only its own parameters: every
 * field of another loss keeps its default (SDCA_LOSS_INIT). The execution
 * settings (struct sdca_execution: working precision and threads) and the fit
 * options (struct sdca_fit_options) are separate structs. Targets (struct
 * sdca_targets) name their kind: class labels, label sets (as labels and
 * offsets, or as a 0/1 mask), target probabilities or teacher logits.
 *
 * Structs. Each struct begins with size, the caller's sizeof of it: start
 * from its SDCA_*_INIT initializer. Within an ABI version fields are only
 * appended: a library accepts every size from the first header of its ABI
 * version up to its own and gives the fields beyond a caller's size their
 * defaults.
 *
 * Results. Every computing entry point returns SDCA_OK, SDCA_INVALID_ARGUMENT
 * or SDCA_FAILURE, and sdca_last_error() describes a failure (thread-local;
 * "" after a success). Each validates every argument before it writes an
 * output: SDCA_INVALID_ARGUMENT leaves every output as given. SDCA_FAILURE (a
 * failed computation or allocation) leaves the outputs unspecified. A fit
 * whose coordinate steps fail numerically returns SDCA_OK with the status
 * SDCA_STATUS_FAILED and the last consistent iterate.
 *
 * Arrays are dense and row-major, in the storage type of a dtype argument:
 * scores (n, num_classes), features x (n, d), a Gram matrix (n, n) and dual
 * variables (n, num_classes); the weights W and a regularizer center are
 * (d, num_classes) column-major (num_classes rows of d). Class labels are
 * 0-based. Required pointers must be non-NULL and their buffers large enough
 * for their shapes; the library rejects NULL pointers and extents that
 * overflow, but cannot check a buffer's capacity. Inputs must not overlap
 * outputs unless stated.
 *
 * Threads. The loss and update entry points compute the rows independently,
 * on up to num_threads threads of the calling process (a loaded OpenMP
 * runtime's team, else a persistent pool); the results are identical for every
 * thread count. num_threads 0 takes SDCA_NUM_THREADS, else the CPUs the
 * process may run on (at most a loaded OpenMP runtime's thread count, else
 * OMP_NUM_THREADS), split among concurrent calls; 1 is serial. A call adds
 * threads only when its work would take at least 1 ms on one thread (0.2 ms on
 * an OpenMP team). A fit's epochs are sequential; its evaluations and checks
 * use the threads (docs/API.md#threads).
 *
 * The computing entry points run in the default IEEE floating-point modes
 * (subnormals kept, round to nearest, no traps) and restore the caller's.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The library builds with -fvisibility=hidden: only these declarations are exported. */
#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

/* Changes with every incompatible change of these declarations; a binding
 * compares it with sdca_abi_version() before any other call. */
#define SDCA_ABI_VERSION 6

/* Return codes of the computing entry points. */
enum sdca_result { SDCA_OK = 0, SDCA_FAILURE = -1, SDCA_INVALID_ARGUMENT = -2 };

/* Storage types. */
enum sdca_dtype { SDCA_DTYPE_FLOAT32 = 0, SDCA_DTYPE_FLOAT64 = 1 };

/* Working precision. DEFAULT works in the storage type; FLOAT32 needs float32
 * storage; FLOAT64 works in double (float32 storage keeps a fit's iterate in
 * double and returns it rounded once). float32 work bounds the accuracy of the
 * results, but takes double where float cannot hold a parameter or a row's
 * spread (docs/API.md#precision); the losses of the smoothed hinges and of
 * proximal distillation compute in double. */
enum sdca_precision {
  SDCA_PRECISION_DEFAULT = 0,
  SDCA_PRECISION_FLOAT32 = 1,
  SDCA_PRECISION_FLOAT64 = 2
};

/* The losses by their shared names (docs/API.md#loss-definitions). */
enum sdca_loss_kind {
  SDCA_LOSS_CROSS_ENTROPY = 0,            /* labels; fit: labels or probabilities */
  SDCA_LOSS_TOPK_CROSS_ENTROPY = 1,       /* labels; k */
  SDCA_LOSS_MULTICLASS_HINGE = 2,         /* labels; gamma */
  SDCA_LOSS_TOPK_HINGE_ALPHA = 3,         /* labels; k, gamma */
  SDCA_LOSS_TOPK_HINGE_BETA = 4,          /* labels; k, gamma */
  SDCA_LOSS_MULTILABEL_HINGE = 5,         /* label sets; gamma, projection_precision */
  SDCA_LOSS_MULTILABEL_CROSS_ENTROPY = 6, /* label sets */
  SDCA_LOSS_PROXIMAL_CROSS_ENTROPY = 7,   /* labels; alpha (no fit) */
  SDCA_LOSS_DISTILLATION = 8,             /* teacher logits; temperature (no fit) */
  SDCA_LOSS_PROXIMAL_DISTILLATION = 9     /* teacher logits; temperature, alpha (no fit) */
};

/* projection_precision of multilabel_hinge: FLOAT64 solves its two-simplex
 * projections in double, checks each result and re-solves with EXTENDED where
 * the check fails; EXTENDED always runs the extended-precision solver. */
enum sdca_projection_precision { SDCA_PROJECTION_FLOAT64 = 0, SDCA_PROJECTION_EXTENDED = 1 };

/* A loss and its own parameters; start from SDCA_LOSS_INIT. The fields of the
 * other losses must keep their defaults (in brackets). */
struct sdca_loss {
  size_t size;
  int kind;                 /* enum sdca_loss_kind [SDCA_LOSS_CROSS_ENTROPY] */
  size_t k;                 /* the top-k losses: 1 <= k < num_classes, required [0] */
  double gamma;             /* the hinges: finite, >= 0 (0: nonsmooth) [0] */
  double alpha;             /* the proximal losses: finite, > 0 [1] */
  double temperature;       /* the distillation losses: finite, > 0 [1] */
  int projection_precision; /* multilabel_hinge: SDCA_PROJECTION_* [FLOAT64] */
};

#define SDCA_LOSS_INIT \
  {sizeof(struct sdca_loss), SDCA_LOSS_CROSS_ENTROPY, 0, 0.0, 1.0, 1.0, SDCA_PROJECTION_FLOAT64}

/* Execution settings: the working precision and the threads. */
struct sdca_execution {
  size_t size;
  int precision;      /* enum sdca_precision [DEFAULT] */
  size_t num_threads; /* 0: SDCA_NUM_THREADS, else the available CPUs; n: up to n [0] */
};

#define SDCA_EXECUTION_INIT {sizeof(struct sdca_execution), SDCA_PRECISION_DEFAULT, 0}

/* Kinds of targets. */
enum sdca_target_kind {
  SDCA_TARGET_LABELS = 0,        /* labels: one class per example (n) */
  SDCA_TARGET_LABEL_SETS = 1,    /* labels and offsets: sorted label sets without duplicates */
  SDCA_TARGET_MASK = 2,          /* mask: label sets as an (n, num_classes) 0/1 matrix */
  SDCA_TARGET_PROBABILITIES = 3, /* probabilities: (n, num_classes) rows on the simplex */
  SDCA_TARGET_LOGITS = 4         /* logits: (n, num_classes) teacher logits, in the dtype */
};

/* The targets of n examples: the fields of kind, the others NULL. Label sets
 * (LABEL_SETS, MASK) are nonempty and leave a class out (the steps of the
 * multilabel objectives need one); example i of LABEL_SETS has the labels
 * labels[offsets[i]] .. labels[offsets[i + 1] - 1], offsets[0] = 0.
 * Probabilities are finite, nonnegative and sum to 1 within 64 n_classes
 * double eps per row. Teacher logits are finite. */
struct sdca_targets {
  size_t size;
  int kind;                    /* enum sdca_target_kind [LABELS] */
  const int64_t* labels;       /* LABELS (n); LABEL_SETS (offsets[n]) */
  const int64_t* offsets;      /* LABEL_SETS (n + 1) */
  const uint8_t* mask;         /* MASK (n, num_classes) */
  const double* probabilities; /* PROBABILITIES (n, num_classes) */
  const void* logits;          /* LOGITS (n, num_classes) */
};

#define SDCA_TARGETS_INIT \
  {sizeof(struct sdca_targets), SDCA_TARGET_LABELS, NULL, NULL, NULL, NULL, NULL}

/* Which targets each loss takes:
 *   cross_entropy, topk_cross_entropy, multiclass_hinge, topk_hinge_alpha,
 *   topk_hinge_beta, proximal_cross_entropy: LABELS (and cross_entropy in
 *   sdca_dual_update and sdca_model_fit: PROBABILITIES, the distillation
 *   objective H(p_i, softmax(x_i W)));
 *   multilabel_hinge, multilabel_cross_entropy: LABEL_SETS or MASK;
 *   distillation, proximal_distillation: LOGITS (sdca_loss_evaluate only).
 * sdca_dual_update and sdca_model_fit train every loss but the proximal
 * and distillation ones (the training objectives). */

/* Losses (n) and, unless gradients is NULL, their gradients (n, num_classes)
 * at the finite scores of dtype, n >= 0 and num_classes >= 2. Values are
 * double; gradients are in the dtype (subgradients at the kinks of the
 * nonsmooth hinges). Proximal distillation computes in double. */
int sdca_loss_evaluate(const struct sdca_loss* loss, const struct sdca_execution* execution,
                       int dtype, size_t n, size_t num_classes, const void* scores,
                       const struct sdca_targets* targets, double* values, void* gradients);

/* The coordinate step of a training objective: new dual blocks out (n,
 * num_classes) from the blocks previous (NULL: zeros) at the finite scores,
 * n >= 0 and num_classes >= 2, all in dtype. norm2 (n) holds each example's
 * squared norm; c and every norm2[i] are finite and positive, and c, norm2[i]
 * and c * norm2[i] normal in the storage type. previous must be feasible for
 * the objective (as a fit's duals are; checked to the tolerances of
 * docs/API.md); out may be previous itself. */
int sdca_dual_update(const struct sdca_loss* loss, double c, const struct sdca_execution* execution,
                     int dtype, size_t n, size_t num_classes, const void* scores,
                     const struct sdca_targets* targets, const void* previous, const double* norm2,
                     void* out);

/* Training data: features x (n, d) or a Gram matrix x (n, n) with d = n. A
 * Gram matrix must be symmetric positive semidefinite: a nonnegative diagonal,
 * symmetry and |K_ij| <= sqrt(K_ii K_jj) are checked to the tolerances of
 * docs/API.md. */
enum sdca_input_kind { SDCA_INPUT_FEATURES = 0, SDCA_INPUT_KERNEL = 1 };

struct sdca_input {
  size_t size;
  int kind;      /* enum sdca_input_kind [FEATURES] */
  int dtype;     /* enum sdca_dtype [FLOAT64] */
  size_t n, d;   /* n, d >= 1 [0, 0] */
  const void* x; /* finite [NULL] */
};

#define SDCA_INPUT_INIT \
  {sizeof(struct sdca_input), SDCA_INPUT_FEATURES, SDCA_DTYPE_FLOAT64, 0, 0, NULL}

/* shrinking: what the epochs of the hinge objectives leave out (the entropy
 * objectives update every class of every example and take AUTO only). HEURISTIC: between full
 * passes (after eval_epoch full passes of work, at most 50 epochs), skip the
 * examples whose last step left their duals unchanged and step a single-label
 * example on its classes of nonzero dual. SAFE: leave out for good the zero
 * duals that the evaluated gap proves zero at the optimum (gap-safe screening;
 * single-label hinges on features). BOTH: the two. AUTO: the default, BOTH.
 * Every evaluation covers every example. */
enum sdca_shrinking {
  SDCA_SHRINKING_AUTO = 0,
  SDCA_SHRINKING_OFF = 1,
  SDCA_SHRINKING_HEURISTIC = 2,
  SDCA_SHRINKING_SAFE = 3,
  SDCA_SHRINKING_BOTH = 4
};

/* The options of sdca_model_fit; start from SDCA_FIT_OPTIONS_INIT. */
struct sdca_fit_options {
  size_t size;
  double c;             /* the loss weight: finite, > 0 [1] */
  double tol;           /* the relative duality gap of SOLVED: finite, >= 0 (0: never) [1e-3] */
  size_t max_epoch;     /* [1000] */
  size_t eval_epoch;    /* evaluate every eval_epoch epochs (0: only at the end; with
                           shrinking, after eval_epoch full passes of work, at most 50
                           epochs apart), and up to twice between where the gap's trend
                           reaches tol [10] */
  int eval_on_start;    /* also evaluate before the first epoch [0] */
  int recompute_primal; /* rebuild W = center + x' a at each evaluation (features) [1] */
  int shrinking;        /* enum sdca_shrinking [AUTO] */
  uint64_t seed;        /* the order of the examples in the epochs; seeds equal modulo
                           2^31 - 2 give the same order [0] */
};

#define SDCA_FIT_OPTIONS_INIT \
  {sizeof(struct sdca_fit_options), 1.0, 1e-3, 1000, 10, 0, 1, SDCA_SHRINKING_AUTO, 0}

/* Status of a fit. Only SOLVED guarantees the duality gap: the relative gap
 * reached tol. NO_PROGRESS: the objectives stopped improving. MAX_EPOCH: the
 * epoch limit. FAILED: a numerical failure, with the last consistent iterate. */
enum sdca_status {
  SDCA_STATUS_SOLVED = 0,
  SDCA_STATUS_NO_PROGRESS = 1,
  SDCA_STATUS_MAX_EPOCH = 2,
  SDCA_STATUS_FAILED = 3
};

/* The columns of a history row, one row per evaluation. */
enum sdca_history_field {
  SDCA_HISTORY_EPOCH = 0,
  SDCA_HISTORY_PRIMAL = 1,
  SDCA_HISTORY_DUAL = 2,
  SDCA_HISTORY_PRIMAL_LOSS = 3,
  SDCA_HISTORY_DUAL_LOSS = 4,
  SDCA_HISTORY_REGULARIZER = 5,
  SDCA_HISTORY_RELATIVE_GAP = 6,
  SDCA_HISTORY_ACCURACY = 7, /* top-1 accuracy; the rank loss of the multilabel losses */
  SDCA_HISTORY_CPU_TIME = 8,
  SDCA_HISTORY_WALL_TIME = 9,
  SDCA_HISTORY_FIELDS = 10
};

/* What sdca_model_fit returns. */
struct sdca_fit_result {
  size_t size;
  int status;          /* enum sdca_status */
  size_t epochs;       /* the epochs run */
  size_t history_rows; /* the rows of history written */
};

#define SDCA_FIT_RESULT_INIT {sizeof(struct sdca_fit_result), SDCA_STATUS_MAX_EPOCH, 0, 0}

/* The history rows that a fit with these options may write: max_epoch /
 * eval_epoch + 2 (2 for eval_epoch 0); 0 for invalid options, with the
 * message in sdca_last_error(). */
size_t sdca_history_rows(const struct sdca_fit_options* options);

/* The largest n, d, num_classes and weight or dual element count that a fit
 * takes: the maximum of the linked BLAS integer type (normally INT_MAX).
 * Bindings check their sizes against it before allocating the outputs. */
size_t sdca_max_extent(void);

/* Trains a loss of the training objectives, minimizing
 *   0.5 ||W - center||^2 + c sum_i loss(x_i' W, target_i)
 * over W (features) or the dual alone (a Gram matrix), with n >= 1, d >= 1
 * and num_classes >= 2. Outputs: dual (n, num_classes) and, for features,
 * weights (d, num_classes) with W = center + x' dual (NULL for a Gram matrix),
 * in the dtype of the input. center (d, num_classes) may be NULL (zero; NULL
 * for a Gram matrix). initial_dual (n, num_classes) starts the fit from these
 * duals (NULL: zeros), which must be feasible for the objective (checked to
 * the tolerances of docs/API.md; single-label rows get the label entry that
 * makes them sum to zero exactly); the library builds the starting W = center +
 * x' initial_dual in double. initial_dual may be dual itself. history holds
 * history_rows rows of SDCA_HISTORY_FIELDS, at least sdca_history_rows(options),
 * or is NULL with history_rows 0 (no history). c and every nonzero squared row
 * norm (Gram diagonal entry) and c times it must be normal in the storage type,
 * and c in the working precision. With a center, the history and the stopping
 * test use the full objective, with no center-dependent constant omitted.
 * n, d, num_classes and the weight and dual counts must fit the linked BLAS
 * integer type (normally INT_MAX, including the built-in backend). */
int sdca_model_fit(const struct sdca_loss* loss, const struct sdca_fit_options* options,
                   const struct sdca_execution* execution, const struct sdca_input* input,
                   size_t num_classes, const struct sdca_targets* targets, const void* center,
                   const void* initial_dual, void* weights, void* dual, double* history,
                   size_t history_rows, struct sdca_fit_result* result);

/* SDCA_ABI_VERSION of the library; compare it with the header's first. */
int sdca_abi_version(void);
/* The library version (the VERSION file at build time); a static string. */
const char* sdca_version(void);
/* The message of this thread's last failed call, "" after a success; valid
 * until the thread's next computing call. */
const char* sdca_last_error(void);

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif
