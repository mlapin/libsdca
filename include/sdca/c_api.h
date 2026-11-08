#ifndef SDCA_C_API_H
#define SDCA_C_API_H

/*
 * C ABI of libsdca, used by the Python and Rust packages (src/c_api.cpp).
 *
 * Every computing function returns SDCA_OK, SDCA_INVALID_ARGUMENT for invalid
 * inputs or options, or SDCA_FAILURE for a failed computation; then
 * sdca_last_error() describes the failure (thread-local; cleared by the next
 * success). Arrays are row-major: features x (n, d), a Gram matrix (n, n),
 * scores z and dual variables (n, num_classes); the weights w and the center
 * are (d, num_classes) column-major. Labels are zero-based; a label set lists
 * sorted labels without duplicates, with offsets (n + 1) delimiting the rows.
 * The loss and update functions leave their outputs as given when they fail.
 * Required buffers must be non-NULL and large enough for their stated shapes;
 * the ABI rejects NULL pointers and overflowing extents, but cannot check an
 * allocation's actual capacity. Optional buffers are identified below.
 *
 * The loss and update functions compute the rows independently, on up to
 * num_threads threads of the calling process (a loaded OpenMP runtime's team,
 * else a persistent pool); the results are identical for every thread count.
 * num_threads SDCA_THREADS_DEFAULT takes SDCA_NUM_THREADS, else 0: automatic,
 * the CPUs the process may run on (at most a loaded OpenMP runtime's thread
 * count, else OMP_NUM_THREADS), split among concurrent calls; 1 is serial. A
 * call adds threads only when its work would take at least 1 ms on one thread
 * (0.2 ms on an OpenMP team), so small inputs run on the calling thread
 * (docs/API.md#threads).
 *
 * The computing functions run in the default IEEE floating-point modes
 * (subnormals kept, round to nearest, no traps) and restore the caller's.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Changes with every incompatible change of these declarations; a binding
 * compares it with sdca_abi_version() before any other call. Fields appended
 * to the structs below are compatible changes (see their size). */
#define SDCA_ABI_VERSION 5

/* num_threads of a call that leaves the count to SDCA_NUM_THREADS (else automatic). */
#define SDCA_THREADS_DEFAULT ((size_t) - 1)

/* projection_precision: how the hinge projections solve (multilabel_hinge's
 * two-simplex steps and smooth loss values; the top-k hinges always run the
 * extended solver). FLOAT64 solves in double, checks each result and re-solves
 * with EXTENDED where the check fails; EXTENDED always runs the
 * extended-precision solver. */
#define SDCA_PROJECTION_FLOAT64 0
#define SDCA_PROJECTION_EXTENDED 1

/* shrinking: what the epochs of fit leave out for the hinge objectives (the
 * entropy objectives update every class of every example). HEURISTIC: between
 * full passes (every tenth epoch and the first), skip the examples whose last
 * step left their duals unchanged and step a single-label example on its
 * classes of nonzero dual. SAFE: leave out for good the zero duals that the
 * certified gap proves zero at the optimum (gap-safe screening; single-label
 * hinges on features). BOTH: the two. AUTO: the default, BOTH. Every
 * evaluation, and so the certificate of SOLVED, covers every example. */
#define SDCA_SHRINKING_AUTO 0
#define SDCA_SHRINKING_OFF 1
#define SDCA_SHRINKING_HEURISTIC 2
#define SDCA_SHRINKING_SAFE 3
#define SDCA_SHRINKING_BOTH 4

/* Return codes of the computing functions. */
enum sdca_result { SDCA_OK = 0, SDCA_FAILURE = -1, SDCA_INVALID_ARGUMENT = -2 };

/* Storage/working arithmetic pairs: float32/float32, float32/float64, float64/float64.
 * sdca_fit keeps its iterate (w and a) in the working type and returns it rounded to the
 * storage type, certified as returned. */
enum sdca_precision { SDCA_FLOAT32 = 0, SDCA_FLOAT32_FLOAT64 = 1, SDCA_FLOAT64 = 2 };

/* Losses and training objectives (every id but the last). The first five take
 * one label per example, the multilabel ones label sets. The hinges are
 * smoothed by gamma > 0 (gamma = 0: nonsmooth); the names without topk ignore k. */
enum sdca_loss {
  SDCA_CROSS_ENTROPY = 0,
  SDCA_TOPK_CROSS_ENTROPY = 1,
  SDCA_MULTICLASS_HINGE = 2,
  SDCA_TOPK_HINGE_ALPHA = 3,
  SDCA_TOPK_HINGE_BETA = 4,
  SDCA_TOPK_HINGE = SDCA_TOPK_HINGE_BETA,
  SDCA_MULTILABEL_HINGE = 5,
  SDCA_MULTILABEL_CROSS_ENTROPY = 6,
  SDCA_PROXIMAL_CROSS_ENTROPY = 7
};

/* Solver status of sdca_fit (result->status). A finished fit reports SOLVED (the
 * relative gap reached tol: the only certified result), NO_PROGRESS (the
 * objectives stopped improving), MAX_EPOCH or FAILED (a numerical failure).
 * NONE and SOLVING are the states before and during a solve; the time limits
 * of MAX_CPU_TIME and MAX_WALL_TIME are not set through this ABI. */
enum sdca_status {
  SDCA_STATUS_NONE = 0,
  SDCA_STATUS_SOLVING = 1,
  SDCA_STATUS_SOLVED = 2,
  SDCA_STATUS_NO_PROGRESS = 3,
  SDCA_STATUS_MAX_EPOCH = 4,
  SDCA_STATUS_MAX_CPU_TIME = 5,
  SDCA_STATUS_MAX_WALL_TIME = 6,
  SDCA_STATUS_FAILED = 7
};

/* The options of sdca_loss_gradient (loss), sdca_update_objective (update) and
 * sdca_fit (fit); each reads the fields that name it. Start from
 * SDCA_OPTIONS_DEFAULT, whose values are given in brackets. size is the
 * caller's sizeof(struct sdca_options), which must be this header's; a later
 * library that appends fields will also accept the sizes of earlier headers
 * and give the appended fields their defaults. */
struct sdca_options {
  size_t size;
  int precision; /* all: enum sdca_precision [SDCA_FLOAT64] */
  int loss;      /* all: enum sdca_loss, the objective of update and fit [cross-entropy] */
  double c;      /* update, fit: the loss weight, c > 0 [1] */
  size_t k;      /* all: 1 <= k < num_classes [1] */
  double gamma;  /* all: the hinge smoothing, gamma >= 0 [0: nonsmooth] */
  double alpha;  /* loss: the step of proximal_cross_entropy, alpha > 0 [1] */
  int projection_precision; /* all: SDCA_PROJECTION_* [FLOAT64] */
  size_t num_threads;       /* all: for fit the evaluations [SDCA_THREADS_DEFAULT] */
  size_t max_epoch;         /* fit: [1000] */
  size_t eval_epoch;        /* fit: evaluate every eval_epoch >= 1 epochs, and up to
                               twice between where the gap's trend reaches tol [10] */
  double tol;               /* fit: the relative duality gap to reach, tol >= 0 [1e-3] */
  int eval_on_start;        /* fit: also evaluate before the first epoch [0] */
  int recompute_primal;     /* fit: rebuild w = center + x' a at each evaluation [1] */
  int kernel;               /* fit: x is a Gram matrix (n, n) [0] */
  int warm;                 /* fit: start from the given w and a [0] */
  int shrinking;            /* fit: SDCA_SHRINKING_* [AUTO]; the size of a header
                               without it gives AUTO */
};

#define SDCA_OPTIONS_DEFAULT    \
  {sizeof(struct sdca_options), \
   SDCA_FLOAT64,                \
   SDCA_CROSS_ENTROPY,          \
   1.0,                         \
   1,                           \
   0.0,                         \
   1.0,                         \
   SDCA_PROJECTION_FLOAT64,     \
   SDCA_THREADS_DEFAULT,        \
   1000,                        \
   10,                          \
   1e-3,                        \
   0,                           \
   1,                           \
   0,                           \
   0,                           \
   SDCA_SHRINKING_AUTO}

/* What sdca_fit returns; size is the caller's sizeof(struct sdca_fit_result),
 * as for sdca_options. */
struct sdca_fit_result {
  size_t size;
  int status;          /* enum sdca_status */
  size_t epochs;       /* the epochs run */
  size_t history_used; /* the rows of history written */
};

#define SDCA_FIT_RESULT_INIT {sizeof(struct sdca_fit_result), SDCA_STATUS_NONE, 0, 0}

/* SDCA_ABI_VERSION of the library; compare it with the header's first. */
int sdca_abi_version(void);
/* The library version, e.g. "0.3.0"; a static string. */
const char* sdca_version(void);
/* The message of this thread's last failed call, "" after a success; valid
 * until the thread's next computing call. */
const char* sdca_last_error(void);

/* Columns of a history row: epoch, primal, dual, primal_loss, dual_loss,
 * regularizer, relative_gap, accuracy (rank loss if multilabel), cpu_time,
 * wall_time. */
size_t sdca_history_fields(void);

/* Loss values (n) and gradients (n, num_classes) at the finite scores z, with
 * n >= 1 and num_classes >= 2. positive is an (n, num_classes) 0/1 mask of the
 * labels: at least one per row, exactly one for the top-k and multiclass
 * losses, and a negative class for multilabel_hinge and proximal_cross_entropy.
 * gamma and alpha are finite. SDCA_FLOAT32 computes in float, in double where
 * float cannot hold gamma, alpha or a row's spread. */
int sdca_loss_gradient(const struct sdca_options* options, size_t n, size_t num_classes,
                       const void* z, const uint8_t* positive, double* values, void* gradients);

/* Cross-entropy of one label per row, labels (n) in [0, num_classes), at the
 * finite scores z (n, num_classes), in double arithmetic: the losses values (n)
 * and, unless stats is NULL, the row statistics (n, 5) of sdca_entropy_gradient.
 * Reads precision (the storage type) and num_threads. */
int sdca_entropy_forward(const struct sdca_options* options, size_t n, size_t num_classes,
                         const void* z, const int64_t* labels, double* values, double* stats);

/* gradients (n, num_classes) of the losses of sdca_entropy_forward, row i
 * multiplied by scale[i] in double and rounded once into the storage type,
 * from the same z and labels and their stats (not checked again: z must be the
 * finite scores of the forward call). stats[i * 5 + 4] is 1 where an unscaled
 * derivative is below the smallest normal double, else 0. */
int sdca_entropy_gradient(const struct sdca_options* options, size_t n, size_t num_classes,
                          const void* z, const int64_t* labels, const double* stats,
                          const double* scale, void* gradients);

/* Coordinate step of any objective: new dual blocks out (n, num_classes) from
 * the finite blocks previous at the finite scores z, n >= 1, num_classes >= 2.
 * labels holds one label per row, or label sets delimited by offsets for the
 * multilabel objectives (offsets is NULL otherwise). norm2 > 0 is the squared
 * norm of every example; c, norm2 and c * norm2 must be normal in the data
 * type. previous should be feasible for the objective (as a fit's duals are);
 * it is not checked. */
int sdca_update_objective(const struct sdca_options* options, size_t n, size_t num_classes,
                          const void* z, const int64_t* labels, const int64_t* offsets,
                          const void* previous, double norm2, void* out);

/* Train on features x (n, d), or a Gram matrix (n, n) if kernel. Outputs,
 * and inputs if warm: weights w (d, num_classes) column-major, unused for a
 * kernel, and duals a (n, num_classes) row-major, with w = center + x' a.
 * center (d, num_classes) may be NULL. With a center, history and stopping use
 * the full regularizer ||w-center||^2/2 in the primal and the corresponding
 * full dual objective; no center-dependent constant is omitted. w may be NULL
 * and center is ignored for a kernel. n, d >= 1, num_classes >= 2 and c is
 * normal in the working precision; c, every nonzero squared row norm (kernel
 * diagonal entry) and c times it must be normal in the data type. A Gram
 * matrix must be symmetric positive semidefinite: a nonnegative
 * diagonal, symmetry and |K_ij| <= sqrt(K_ii K_jj) are checked to the
 * tolerances of docs/API.md. A coordinate step that throws ends the fit as
 * FAILED with its block unchanged. With warm, a must be feasible for the
 * objective and w = center + x' a (finiteness is checked, feasibility is not).
 * recompute_primal 0 keeps the w maintained by the updates of a feature model.
 * history holds history_rows rows of sdca_history_fields(), at least
 * max_epoch / eval_epoch + 2. result receives the status, the epochs run and
 * the history rows written. The buffers must not overlap; inputs must be
 * finite. n, d, num_classes and the weight/dual element counts must fit the
 * linked BLAS integer type (normally INT_MAX, including the built-in backend). */
int sdca_fit(const struct sdca_options* options, size_t n, size_t d, size_t num_classes,
             const void* x, const int64_t* labels, const int64_t* offsets, const void* center,
             void* w, void* a, double* history, size_t history_rows,
             struct sdca_fit_result* result);

/* Soft-target (distillation) counterparts of sdca_update_objective and sdca_fit:
 * targets is an (n, num_classes) row-major double probability matrix (each row
 * nonnegative and summing to 1). Minimizes 0.5 ||W - center||^2 +
 * C sum_i H(targets[i], softmax(x[i] W)). */
int sdca_update_distillation(const struct sdca_options* options, size_t n, size_t num_classes,
                             const void* z, const double* targets, const void* previous,
                             double norm2, void* out);

int sdca_fit_distillation(const struct sdca_options* options, size_t n, size_t d,
                          size_t num_classes, const void* x, const double* targets,
                          const void* center, void* w, void* a, double* history,
                          size_t history_rows, struct sdca_fit_result* result);

/* Storage types of the tensor entry points below: IEEE binary16 and bfloat16
 * as their raw 16 bits. */
enum sdca_dtype {
  SDCA_DTYPE_FLOAT32 = 0,
  SDCA_DTYPE_FLOAT64 = 1,
  SDCA_DTYPE_FLOAT16 = 2,
  SDCA_DTYPE_BFLOAT16 = 3
};

/* The tensor entry points take scores z (n, num_classes) in any sdca_dtype,
 * n >= 0 and num_classes >= 2, and read the options size, precision and
 * num_threads (sdca_loss_gradient_rows also its loss options). precision
 * selects the working arithmetic only: SDCA_FLOAT32 float (not for float64
 * storage), SDCA_FLOAT32_FLOAT64 and SDCA_FLOAT64 double. Conversions into
 * 16-bit storage round to nearest even, once. */

/* Cross-entropy of integer labels (n) as torch.nn.functional.cross_entropy:
 * rows whose label equals ignore_index are neither read nor checked; the
 * others need labels in [0, num_classes) and finite scores. class_weights
 * (num_classes, not checked) may be NULL (all 1). Outputs: values (n),
 * w[label] * loss per row, 0 where ignored; unless NULL, stats (n, 5) for
 * sdca_cross_entropy_backward and totals (2): the sums of values and of
 * w[label] over the rows not ignored, in row order. SDCA_FLOAT32 takes float
 * exponentials and keeps the row statistics (maximum, log-sum, loss) in
 * double; double arithmetic gives the values of sdca_entropy_forward. stats
 * rows are those of sdca_entropy_forward (maximum, lowest, sum, mass, tiny:
 * 1 where an unweighted, unscaled derivative is below the smallest normal
 * double), zero where ignored. A failed call leaves the outputs as given. */
int sdca_cross_entropy_forward(const struct sdca_options* options, int dtype, size_t n,
                               size_t num_classes, const void* z, const int64_t* labels,
                               int64_t ignore_index, const double* class_weights, double* values,
                               double* stats, double* totals);

/* gradients (n, num_classes) in dtype: row i is scale[i * scale_stride] *
 * w[label] times the derivative of its loss, rounded once into dtype, zero
 * where ignored; scale_stride is 0 (one scale) or 1 (one per row). z, labels,
 * ignore_index, class_weights and precision must be those of the forward call
 * that gave stats (z is not checked again). Double arithmetic gives the
 * gradients of sdca_entropy_gradient. SDCA_FLOAT32 takes the competitors'
 * entries as squared float halves, exp((z_j - maximum) / 2)^2 times the row
 * factor, in double where that factor exceeds the float range, and the target
 * entry in double. gradients may be z itself (each row is read before it is
 * written). A failed call writes no gradient. */
int sdca_cross_entropy_backward(const struct sdca_options* options, int dtype, size_t n,
                                size_t num_classes, const void* z, const int64_t* labels,
                                int64_t ignore_index, const double* class_weights,
                                const double* stats, const double* scale, size_t scale_stride,
                                void* gradients);

/* sdca_loss_gradient of scores in any dtype, with the targets as one label
 * per row (labels) or an (n, num_classes) 0/1 mask (positive): exactly one of
 * them is non-NULL. 16-bit rows convert exactly to float; float and 16-bit
 * storage write float gradients, float64 storage double ones. The results
 * equal those of sdca_loss_gradient on the float or double scores. Each row
 * is checked before its loss: finite scores, labels in [0, num_classes), a
 * positive label, a single one for the top-k and multiclass losses, and a
 * negative one for multilabel_hinge and proximal_cross_entropy. gradients may
 * be NULL: the values alone. A failed call may have written some rows. */
int sdca_loss_gradient_rows(const struct sdca_options* options, int dtype, size_t n,
                            size_t num_classes, const void* z, const int64_t* labels,
                            const uint8_t* positive, double* values, void* gradients);

/* distillation_loss of student and teacher logits (n, num_classes), both in
 * dtype, at temperature T (finite, positive): values (n) the cross-entropy
 * H(p, q) per row of p = softmax(teacher / T) and q = softmax(student / T),
 * and unless NULL stats (n, 7) for sdca_distillation_backward. Logits must be
 * finite. SDCA_FLOAT32 takes float exponentials with double row statistics
 * (rows whose spread, tails or T float cannot hold take double); double
 * arithmetic compensates the row sums. A failed call may have written some
 * rows. */
int sdca_distillation_forward(const struct sdca_options* options, int dtype, size_t n,
                              size_t num_classes, const void* student, const void* teacher,
                              double temperature, double* values, double* stats);

/* gradients (n, num_classes) in dtype: row i is scale[i * scale_stride] (q -
 * p) / T, rounded once into dtype; scale_stride is 0 or 1. student, teacher,
 * temperature and precision must be those of the forward call that gave
 * stats. The gradient sums to zero (q - p from the log-ratio log(q / p):
 * equal logits give zero). */
int sdca_distillation_backward(const struct sdca_options* options, int dtype, size_t n,
                               size_t num_classes, const void* student, const void* teacher,
                               double temperature, const double* stats, const double* scale,
                               size_t scale_stride, void* gradients);

/* proximal_distillation_loss: per row min_u H(p, softmax(u)) + |u -
 * student / T|^2 / (2 alpha), alpha finite and positive, in double
 * arithmetic: values (n) and gradients (n, num_classes) (pi - p) / T, float
 * for float and 16-bit storage, double for float64. Logits must be finite. A
 * failed call may have written some rows. */
int sdca_proximal_distillation_rows(const struct sdca_options* options, int dtype, size_t n,
                                    size_t num_classes, const void* student, const void* teacher,
                                    double temperature, double alpha, double* values,
                                    void* gradients);

#ifdef __cplusplus
}
#endif

#endif
