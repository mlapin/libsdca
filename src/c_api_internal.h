#ifndef SDCA_C_API_INTERNAL_H
#define SDCA_C_API_INTERNAL_H

/*
 * Internal entry points of the native library for the PyTorch and JAX CPU
 * paths of the Python package (python/libsdca/_core.py, _cpu.py and
 * _jax_ffi.cpp). They are not part of the public ABI (include/sdca/c_api.h):
 * they change with the package, without notice, and take storage types and
 * options the public entry points do not.
 *
 * Storage types: the public SDCA_DTYPE_FLOAT32 and _FLOAT64, and IEEE binary16
 * and bfloat16 as their raw 16 bits. Scores z are (n, num_classes), n >= 0 and
 * num_classes >= 2. Conversions into 16-bit storage round to nearest even, once.
 * Return codes and sdca_last_error() are those of the public entry points, but
 * the failure semantics are each function's own (below).
 */

#include <stddef.h>
#include <stdint.h>

#include "sdca/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

enum sdca_internal_dtype { SDCA_INTERNAL_DTYPE_FLOAT16 = 2, SDCA_INTERNAL_DTYPE_BFLOAT16 = 3 };

/* The options of the internal entry points. Each reads the fields it names;
 * the label losses ignore the parameters their loss does not take. */
struct sdca_internal_options {
  size_t size;              /* sizeof(struct sdca_internal_options) */
  int float_work;           /* 1: float arithmetic (not for float64 storage), 0: double */
  int loss;                 /* sdca_internal_loss_rows: SDCA_LOSS_* of a label loss */
  size_t k;                 /* sdca_internal_loss_rows: 1 <= k < num_classes */
  double gamma;             /* sdca_internal_loss_rows: finite, >= 0 */
  double alpha;             /* sdca_internal_loss_rows: finite, > 0 */
  int projection_precision; /* sdca_internal_loss_rows: SDCA_PROJECTION_* */
  size_t num_threads;       /* as struct sdca_execution */
};

/* The losses of the label losses (kinds 0 to 7), with the targets as one label
 * per row (labels) or an (n, num_classes) 0/1 mask (positive): exactly one of
 * them is non-NULL. cross_entropy of a mask has the uniform target on its
 * labels. 16-bit rows convert exactly to float; float and 16-bit storage write
 * float gradients, float64 storage double ones; gradients may be NULL (the
 * values alone). Each row is checked before its loss: finite scores, labels in
 * [0, num_classes), a positive label, a single one for the top-k and
 * multiclass losses, and a negative one for multilabel_hinge and
 * proximal_cross_entropy. A failed call may have written some rows. */
int sdca_internal_loss_rows(const struct sdca_internal_options* options, int dtype, size_t n,
                            size_t num_classes, const void* z, const int64_t* labels,
                            const uint8_t* positive, double* values, void* gradients);

/* Cross-entropy of integer labels (n) as torch.nn.functional.cross_entropy:
 * rows whose label equals ignore_index are neither read nor checked; the
 * others need labels in [0, num_classes) and finite scores. class_weights
 * (num_classes, not checked) may be NULL (all 1). Outputs: values (n),
 * w[label] * loss per row, 0 where ignored; unless NULL, stats (n, 5) for
 * sdca_internal_cross_entropy_backward and totals (2): the sums of values and
 * of w[label] over the rows not ignored, in row order. Float work takes float
 * exponentials and keeps the row statistics (maximum, log-sum, loss) in double.
 * stats rows are (maximum, lowest, sum, mass, tiny: 1 where an unweighted,
 * unscaled derivative is below the smallest normal double), zero where
 * ignored. A failed call leaves the outputs as given. */
int sdca_internal_cross_entropy_forward(const struct sdca_internal_options* options, int dtype,
                                        size_t n, size_t num_classes, const void* z,
                                        const int64_t* labels, int64_t ignore_index,
                                        const double* class_weights, double* values, double* stats,
                                        double* totals);

/* gradients (n, num_classes) in dtype: row i is scale[i * scale_stride] *
 * w[label] times the derivative of its loss, rounded once into dtype, zero
 * where ignored; scale_stride is 0 (one scale) or 1 (one per row). z, labels,
 * ignore_index, class_weights and float_work must be those of the forward call
 * that gave stats (z and scale are not checked: a nonfinite scale gives NaN
 * gradients). Float work takes the competitors' entries as squared float
 * halves, exp((z_j - maximum) / 2)^2 times the row factor, in double where
 * that factor exceeds the float range, and the target entry in double.
 * gradients may be z itself (each row is read before it is written). A failed
 * call writes no gradient. */
int sdca_internal_cross_entropy_backward(const struct sdca_internal_options* options, int dtype,
                                         size_t n, size_t num_classes, const void* z,
                                         const int64_t* labels, int64_t ignore_index,
                                         const double* class_weights, const double* stats,
                                         const double* scale, size_t scale_stride, void* gradients);

/* distillation of student and teacher logits (n, num_classes), both in dtype,
 * at temperature T (finite, positive): values (n) H(p, q) per row of p =
 * softmax(teacher / T) and q = softmax(student / T), and unless NULL stats
 * (n, 7) for sdca_internal_distillation_backward. Logits must be finite.
 * Float work takes float exponentials with double row statistics (rows whose
 * spread, tails or T float cannot hold take double); double arithmetic
 * compensates the row sums. A failed call may have written some rows. */
int sdca_internal_distillation_forward(const struct sdca_internal_options* options, int dtype,
                                       size_t n, size_t num_classes, const void* student,
                                       const void* teacher, double temperature, double* values,
                                       double* stats);

/* gradients (n, num_classes) in dtype: row i is scale[i * scale_stride] (q -
 * p) / T, rounded once into dtype; scale_stride is 0 or 1 (scale is not
 * checked). student, teacher, temperature and float_work must be those of the
 * forward call that gave stats. */
int sdca_internal_distillation_backward(const struct sdca_internal_options* options, int dtype,
                                        size_t n, size_t num_classes, const void* student,
                                        const void* teacher, double temperature,
                                        const double* stats, const double* scale,
                                        size_t scale_stride, void* gradients);

/* proximal distillation: per row min_u H(p, softmax(u)) + |u - student / T|^2
 * / (2 alpha), alpha finite and positive, in double arithmetic: values (n) and
 * gradients (n, num_classes) (pi - p) / T, float for float and 16-bit storage,
 * double for float64. Logits must be finite. A failed call may have written
 * some rows. */
int sdca_internal_proximal_distillation_rows(const struct sdca_internal_options* options, int dtype,
                                             size_t n, size_t num_classes, const void* student,
                                             const void* teacher, double temperature, double alpha,
                                             double* values, void* gradients);

/* Synchronous row scheduling for the extension kernels: callback processes
 * [begin, end) and may be called concurrently, but must not call back into
 * this scheduler; context stays valid until this call returns. num_threads is
 * as in struct sdca_execution. */
typedef void (*sdca_internal_row_callback)(void* context, size_t begin, size_t end);
int sdca_internal_run_rows(size_t rows, size_t work_per_row, size_t num_threads,
                           sdca_internal_row_callback callback, void* context);

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif
