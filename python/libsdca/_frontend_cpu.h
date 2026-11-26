// The CPU losses of libsdca.torch (_torch_cpu.cpp) and libsdca.jax (_jax_cpu.cpp): one call
// of sdca::kernel::evaluate per batch on the framework's own buffers (the instruction set
// chosen at run time inside kernel.h).
#ifndef LIBSDCA_FRONTEND_CPU_H
#define LIBSDCA_FRONTEND_CPU_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "sdca/kernel.h"
#include "sdca/utility/fpenv.h"

#define SDCA_EACH_ISA "sdca/loss/teacher_gradient.h"
#include "sdca/core/each_isa.h"

namespace libsdca_frontend {

using TeacherRows = decltype(&sdca::native::loss::distill::teacher_rows);

inline TeacherRows teacher_kernel() {
#if defined(__x86_64__)
  switch (sdca::kernel::isa()) {
    case 4:
      return sdca::isa_avx512::loss::distill::teacher_rows;
    case 3:
      return sdca::isa_avx2::loss::distill::teacher_rows;
    default:
      return sdca::isa_scalar::loss::distill::teacher_rows;
  }
#else
  return sdca::native::loss::distill::teacher_rows;
#endif
}

constexpr int kDistillation = sdca::kernel::distillation;
constexpr int kProximalDistillation = sdca::kernel::proximal_distillation;

// A call's loss and options, as the frameworks pass them.
struct Spec {
  int kind = sdca::kernel::cross_entropy;
  size_t k = 1;
  double gamma = 0;
  double temperature = 1;
  double label_smoothing = 0;
  int64_t ignore_index = 0;
  bool ignore = false;
  int precision = 0;  // sdca_precision
  size_t num_threads = 0;
  double upstream = 1;
};

// One batch: n rows of `classes` scores, per-row values (null: not wanted), the gradient
// (null: not wanted; may be scores) times upstream and row_upstream (null: 1).
template <typename S>
struct Batch {
  size_t n;
  size_t classes;
  const S* scores;
  double* values;
  S* grads;
  const double* row_upstream;
};

// Linux may map a newly written gradient with transparent huge pages. Advise only complete
// 2 MiB pages inside a large separate output, before the row workers first touch it.
template <typename S>
void advise_gradient(const Batch<S>& b) {
#if defined(MADV_HUGEPAGE)
  const size_t bytes = b.n * b.classes * sizeof(S);
  if (b.grads == nullptr || b.grads == b.scores || bytes < (size_t(32) << 20)) return;
  constexpr uintptr_t boundary = uintptr_t(2) << 20;
  const uintptr_t p = reinterpret_cast<uintptr_t>(b.grads);
  const uintptr_t first = (p + boundary - 1) & ~(boundary - 1);
  const uintptr_t end = (p + bytes) & ~(boundary - 1);
  if (first < end) madvise(reinterpret_cast<void*>(first), end - first, MADV_HUGEPAGE);
#else
  (void)b;
#endif
}

// The kernels with subnormals kept (no flush-to-zero or denormals-are-zero, which XLA and
// torch.set_flush_denormal set), as the C ABI; the caller's modes return after.
template <typename S, typename Targets>
sdca::kernel::Result run(const Spec& spec, const Batch<S>& b, const Targets& targets) {
  const sdca::kernel::Loss loss = {spec.kind, spec.k, spec.gamma, spec.temperature,
                                   spec.label_smoothing};
  const sdca::kernel::Execution e = {spec.precision, spec.num_threads, false};
  advise_gradient(b);
  sdca::kernel::Result r;
  {
    const sdca::detail::subnormal_environment subnormals;
    r = sdca::kernel::evaluate(loss, e, b.n, b.classes, b.scores, targets, b.values, b.grads,
                               spec.upstream, b.row_upstream);
  }
  if (r.status == sdca::kernel::status_unsupported) {
    throw std::invalid_argument("libsdca: no kernel of loss " + std::to_string(spec.kind) +
                                " takes these targets");
  }
  if (r.status != sdca::kernel::status_ok) throw std::invalid_argument("libsdca: invalid call");
  return r;
}

// Class labels, one per row; class weights (null: 1).
template <typename S>
sdca::kernel::Result labels(const Spec& spec, const Batch<S>& b, const int64_t* labels,
                            const double* class_weight) {
  const sdca::kernel::Labels t = {labels, class_weight, spec.ignore_index, spec.ignore};
  const auto result = run(spec, b, t);
  if (result.invalid_rows != 0) {
    const auto i =
        spec.ignore ? sdca::kernel::first_invalid_label(labels, b.n, b.classes, spec.ignore_index)
                    : sdca::kernel::first_invalid_label(labels, b.n, b.classes);
    if (i < b.n) throw std::out_of_range(sdca::kernel::invalid_label_text(labels[i]));
  }
  return result;
}

// Label masks: one byte per score, 0 or 1 (bool storage).
template <typename S>
sdca::kernel::Result mask(const Spec& spec, const Batch<S>& b, const uint8_t* mask) {
  const sdca::kernel::Mask t = {mask};
  return run(spec, b, t);
}

// Class probabilities of cross_entropy, in P (the scores' type, float or double).
template <typename S, typename P>
sdca::kernel::Result probabilities(const Spec& spec, const Batch<S>& b, const P* p,
                                   const double* class_weight) {
  const sdca::kernel::Probabilities<P> t = {p, class_weight};
  return run(spec, b, t);
}

// Teacher logits of the distillation losses, in the scores' type.
template <typename S>
sdca::kernel::Result teacher(const Spec& spec, const Batch<S>& b, const S* logits) {
  const sdca::kernel::Teacher<S> t = {logits};
  return run(spec, b, t);
}

// The losses with kernels, a bit per kind: a call of no rows on each kind's own targets.
inline uint64_t kinds() {
  uint64_t bits = 0;
  for (int kind = 0; kind < sdca::kernel::num_kinds; ++kind) {
    const sdca::kernel::Loss loss = {kind, 1, 0.0, 1.0, 0.0};
    const sdca::kernel::Execution e = {0, 1, false};
    const float* none = nullptr;
    sdca::kernel::Result r;
    if (sdca::kernel::traits(kind).multilabel) {
      r = sdca::kernel::evaluate(loss, e, 0, 1, none, sdca::kernel::Mask{nullptr}, nullptr,
                                 static_cast<float*>(nullptr));
    } else if (sdca::kernel::traits(kind).distillation) {
      r = sdca::kernel::evaluate(loss, e, 0, 1, none, sdca::kernel::Teacher<float>{nullptr},
                                 nullptr, static_cast<float*>(nullptr));
    } else {
      const sdca::kernel::Labels t = {nullptr, nullptr, 0, false};
      r = sdca::kernel::evaluate(loss, e, 0, 1, none, t, nullptr, static_cast<float*>(nullptr));
    }
    if (r.status != sdca::kernel::status_unsupported) bits |= uint64_t(1) << kind;
  }
  return bits;
}

}  // namespace libsdca_frontend

#endif
