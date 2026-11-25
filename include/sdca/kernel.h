#if !defined(SDCA_KERNEL_PASS)
#ifndef SDCA_KERNEL_H
#define SDCA_KERNEL_H

// The losses on every instruction set of the architecture from one translation unit, chosen
// at run time: kernel::evaluate(loss, ...) of a loss kind (the C ABI's SDCA_LOSS_*). The
// instruction set of the compiler's flags compiles as given (sdca::native names its
// namespace); on x86-64 the others (scalar, AVX2, AVX-512) compile under target attributes.
// SDCA_CPU_CAPABILITY (default, avx2, avx512) lowers the choice, never above the CPU's.
//
// The standard headers of every ISA-dependent header are included here first: a header first
// included inside a target region would compile its inline functions for that target.
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "sdca/c_api.h"
#include "sdca/core/execution.h"
#include "sdca/core/types.h"
#include "sdca/loss/row.h"
#include "sdca/utility/buffer.h"
#include "sdca/utility/parallel.h"

namespace sdca {
namespace kernel {

// The losses by their shared names (docs/API.md), as the C ABI's enum sdca_loss_kind.
enum Kind {
  cross_entropy = SDCA_LOSS_CROSS_ENTROPY,
  topk_cross_entropy = SDCA_LOSS_TOPK_CROSS_ENTROPY,
  multiclass_hinge = SDCA_LOSS_MULTICLASS_HINGE,
  topk_hinge_alpha = SDCA_LOSS_TOPK_HINGE_ALPHA,
  topk_hinge_beta = SDCA_LOSS_TOPK_HINGE_BETA,
  multilabel_hinge = SDCA_LOSS_MULTILABEL_HINGE,
  multilabel_cross_entropy = SDCA_LOSS_MULTILABEL_CROSS_ENTROPY,
  proximal_cross_entropy = SDCA_LOSS_PROXIMAL_CROSS_ENTROPY,
  distillation = SDCA_LOSS_DISTILLATION,
  proximal_distillation = SDCA_LOSS_PROXIMAL_DISTILLATION
};
const int num_kinds = proximal_distillation + 1;

// A loss kind's name and what it takes. A parameter a loss does not take is 0 (k, gamma) or
// 1 (temperature).
struct Traits {
  const char* name;
  bool takes_k;       // the top-k losses: 1 <= k < classes
  bool hinge;         // the hinges: smoothing gamma, finite and >= 0 (0: nonsmooth)
  bool proximal;      // a Moreau step gamma, finite and > 0
  bool multilabel;    // label sets
  bool distillation;  // teacher logits and a temperature, finite and > 0
  bool objective;     // a training objective of the solver
};

// The traits of a kind in [0, num_kinds).
inline const Traits& traits(const int kind) {
  static const Traits table[num_kinds] = {
      {"cross_entropy", false, false, false, false, false, true},
      {"topk_cross_entropy", true, false, false, false, false, true},
      {"multiclass_hinge", false, true, false, false, false, true},
      {"topk_hinge_alpha", true, true, false, false, false, true},
      {"topk_hinge_beta", true, true, false, false, false, true},
      {"multilabel_hinge", false, true, false, true, false, true},
      {"multilabel_cross_entropy", false, false, false, true, false, true},
      {"proximal_cross_entropy", false, false, true, false, false, false},
      {"distillation", false, false, false, false, true, false},
      {"proximal_distillation", false, false, true, false, true, false}};
  return table[kind];
}

namespace detail {
// A real in the shortest form that reads back as it, as Python's repr of a float.
inline std::string real_text(const double value) {
  char text[32];
  for (int digits = 1; digits <= 17; ++digits) {
    std::snprintf(text, sizeof(text), "%.*g", digits, value);
    if (!(std::strtod(text, nullptr) != value)) break;
  }
  std::string out(text);
  if (out.find_first_of(".eni") == std::string::npos) out += ".0";
  return out;
}
[[noreturn]] inline void invalid(const std::string& message) {
  throw std::invalid_argument(message);
}
[[noreturn]] inline void untaken(const int kind, const char* parameter) {
  invalid(std::string(traits(kind).name) + " takes no parameter " + parameter);
}
inline void check_unused_parameters(const Traits& t, int kind, std::size_t k, double gamma,
                                    double temperature) {
  if (!t.takes_k && k != 0) untaken(kind, "k");
  if (!t.hinge && !t.proximal && gamma != 0) untaken(kind, "gamma");
  if (!t.distillation && temperature != 1) untaken(kind, "temperature");
}
}  // namespace detail

// std::invalid_argument unless kind is a loss and the parameters suit it: those it does not
// take at their neutral values (see Traits), k in [1, classes - 1] (classes 0: not known yet),
// gamma and temperature in their ranges. The texts are those of the Python frontend.
inline void check_parameters(const int kind, const std::size_t classes, const std::size_t k,
                             const double gamma, const double temperature) {
  if (kind < 0 || kind >= num_kinds) detail::invalid("Unknown loss kind");
  const Traits& t = traits(kind);
  detail::check_unused_parameters(t, kind, k, gamma, temperature);
  if (t.takes_k && k < 1) detail::invalid("k must be at least 1, got " + std::to_string(k));
  if (t.proximal && !(gamma > 0 && std::isfinite(gamma)))
    detail::invalid("gamma must be finite and positive, got " + detail::real_text(gamma));
  if (t.hinge && !(gamma >= 0 && std::isfinite(gamma)))
    detail::invalid("gamma must be finite and nonnegative, got " + detail::real_text(gamma));
  if (t.distillation && !(temperature > 0 && std::isfinite(temperature)))
    detail::invalid("temperature must be finite and positive, got " +
                    detail::real_text(temperature));
  if (t.takes_k && classes != 0 && k >= classes)
    detail::invalid("k must be in [1, classes - 1] = [1, " + std::to_string(classes - 1) +
                    "], got " + std::to_string(k));
}

// std::invalid_argument unless loss kind is a training objective of the solver.
inline void check_objective(const int kind) {
  if (kind < 0 || kind >= num_kinds) detail::invalid("Unknown loss kind");
  const Traits& t = traits(kind);
  if (!t.objective)
    detail::invalid(std::string(t.name) + " is not a training objective" +
                    (t.distillation ? "; train on target probabilities with cross_entropy" : ""));
}

// The first of n labels outside [0, classes), skipping ignore_index (the overload without it
// skips none), else n: an O(n) scan that the frontends run before a loss evaluation, as
// PyTorch rejects such labels. Label sets pass their concatenated labels.
inline std::size_t first_invalid_label(const int64_t* labels, const std::size_t n,
                                       const std::size_t classes, const int64_t ignore_index) {
  bool invalid = false;
  for (std::size_t i = 0; i < n; ++i)
    invalid |= static_cast<uint64_t>(labels[i]) >= classes && labels[i] != ignore_index;
  if (!invalid) return n;
  std::size_t i = 0;
  while (static_cast<uint64_t>(labels[i]) < classes || labels[i] == ignore_index) ++i;
  return i;
}
inline std::size_t first_invalid_label(const int64_t* labels, const std::size_t n,
                                       const std::size_t classes) {
  bool invalid = false;
  for (std::size_t i = 0; i < n; ++i) invalid |= static_cast<uint64_t>(labels[i]) >= classes;
  if (!invalid) return n;
  std::size_t i = 0;
  while (static_cast<uint64_t>(labels[i]) < classes) ++i;
  return i;
}

// The error of a label out of range, in PyTorch's words.
inline std::string invalid_label_text(const int64_t label) {
  return "Target " + std::to_string(label) + " is out of bounds.";
}

// std::invalid_argument (invalid_label_text) for the first label out of range.
inline void check_labels(const int64_t* labels, const std::size_t n, const std::size_t classes) {
  const std::size_t i = first_invalid_label(labels, n, classes);
  if (i < n) detail::invalid(invalid_label_text(labels[i]));
}
inline void check_labels(const int64_t* labels, const std::size_t n, const std::size_t classes,
                         const int64_t ignore_index) {
  const std::size_t i = first_invalid_label(labels, n, classes, ignore_index);
  if (i < n) detail::invalid(invalid_label_text(labels[i]));
}

struct Loss {
  int kind;
  std::size_t k;
  double gamma, temperature, label_smoothing;
};

using loss::Labels;
using loss::LabelSets;
using loss::Mask;
using loss::Probabilities;
using loss::Teacher;

}  // namespace kernel
}  // namespace sdca

#define SDCA_KERNEL_PASS 1
#define SDCA_EACH_ISA "sdca/kernel.h"
#include "sdca/core/each_isa.h"
#undef SDCA_KERNEL_PASS

namespace sdca {
namespace native = SDCA_ISA_NAMESPACE;

namespace kernel {

// The instruction set of the kernels on this CPU: 1 scalar, 2 neon, 3 avx2, 4 avx512.
inline int isa() {
  static const int chosen = [] {
#if defined(__x86_64__)
    __builtin_cpu_init();
    // F16C by its CPUID bit: clang before 16 has no "f16c" for __builtin_cpu_supports.
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    const bool f16c = __get_cpuid(1, &eax, &ebx, &ecx, &edx) && (ecx & bit_F16C) != 0;
    const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && f16c;
    const bool avx512 = avx2 && __builtin_cpu_supports("avx512f") &&
                        __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512bw") &&
                        __builtin_cpu_supports("avx512vl");
    int level = avx512 ? 4 : (avx2 ? 3 : 1);
    if (level < SDCA_DEFAULT_ISA_ID) level = SDCA_DEFAULT_ISA_ID;
    const char* cap = std::getenv("SDCA_CPU_CAPABILITY");
    if (cap && std::strcmp(cap, "default") == 0) level = 1;
    if (cap && std::strcmp(cap, "avx2") == 0 && level > 3) level = 3;
    return level;
#else
    return SDCA_DEFAULT_ISA_ID;
#endif
  }();
  return chosen;
}
inline const char* isa_name() {
  static const char* const names[] = {"", "default", "neon", "avx2", "avx512"};
  return names[isa()];
}

// The loss whose kernels compute loss: at k = 1, topk_cross_entropy is cross_entropy and both
// top-k hinges are multiclass_hinge with the same gamma (their feasible sets coincide), so
// every interface takes the dedicated k = 1 kernels (python/libsdca/_frontend.py, kernel_kind).
inline Loss canonical(Loss loss) {
  if (loss.k == 1) {
    if (loss.kind == topk_cross_entropy) loss.kind = cross_entropy;
    if (loss.kind == topk_hinge_alpha || loss.kind == topk_hinge_beta) loss.kind = multiclass_hinge;
  }
  return loss;
}

// Values (n, unless null) and gradients (n x classes, unless null) of n rows of a loss at
// row-major scores of storage S (float, double, half, bfloat16) and its targets; the gradient
// of row i times upstream (and row_upstream[i] unless null) rounded once into G (S, or float
// for 16-bit S). grads may be scores (G = S).
template <typename S, typename G, typename Targets>
Result evaluate(const Loss& loss, const Execution& e, std::size_t n, std::size_t classes,
                const S* scores, const Targets& t, double* values, G* grads, double upstream = 1.0,
                const double* row_upstream = 0) {
  const Loss spec = canonical(loss);
  switch (isa()) {
#if defined(__x86_64__)
    case 4:
      return isa_avx512::core::evaluate_kind(spec, e, n, classes, scores, t, values, grads,
                                             upstream, row_upstream);
    case 3:
      return isa_avx2::core::evaluate_kind(spec, e, n, classes, scores, t, values, grads, upstream,
                                           row_upstream);
    default:
      return isa_scalar::core::evaluate_kind(spec, e, n, classes, scores, t, values, grads,
                                             upstream, row_upstream);
#else
    default:
      return native::core::evaluate_kind(spec, e, n, classes, scores, t, values, grads, upstream,
                                         row_upstream);
#endif
  }
}

}  // namespace kernel
}  // namespace sdca

#endif  // SDCA_KERNEL_H

#else  // SDCA_KERNEL_PASS: the kernels of one instruction set
#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_KERNEL_H)
#if SDCA_ISA_ID == 1
#define SDCA_KERNEL_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_KERNEL_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_KERNEL_H3 1
#else
#define SDCA_KERNEL_H4 1
#endif

#include "sdca/core/driver.h"
#include "sdca/loss/ce.h"
#include "sdca/loss/distill.h"
#include "sdca/loss/multilabel.h"
#include "sdca/loss/proximal.h"
#include "sdca/loss/topk.h"

SDCA_ISA_BEGIN
namespace core {
namespace driver {

// Whether a loss functor takes the row targets T of scores S and gradients G.
template <typename F, typename S, typename G, typename T>
struct Takes {
  template <typename U>
  static char test(decltype(std::declval<const U&>().template operator()<double, isa::native>(
      std::declval<const loss::RowArgs<S, G>&>(), std::declval<const T&>(),
      std::declval<loss::Scratch&>()))*);
  template <typename U>
  static long test(...);
  static const bool value = sizeof(test<F>(0)) == 1;
};

template <typename F, typename S, typename G, typename Targets>
kernel::Result call(const F& f, const kernel::Execution& e, std::size_t n, std::size_t classes,
                    const S* scores, const Targets& t, double* values, G* grads, double upstream,
                    const double* row_upstream, std::true_type) {
  return core::evaluate(f, e, n, classes, scores, t, values, grads, upstream, row_upstream);
}
template <typename F, typename S, typename G, typename Targets>
kernel::Result call(const F&, const kernel::Execution&, std::size_t n, std::size_t, const S*,
                    const Targets&, double*, G*, double, const double*, std::false_type) {
  const kernel::Result r = {kernel::status_unsupported, 0, n};
  return r;
}
template <typename F, typename S, typename G, typename Targets>
kernel::Result call(const F& f, const kernel::Execution& e, std::size_t n, std::size_t classes,
                    const S* scores, const Targets& t, double* values, G* grads, double upstream,
                    const double* row_upstream) {
  typedef decltype(t.row(0, 0)) Row;
  return call(f, e, n, classes, scores, t, values, grads, upstream, row_upstream,
              std::integral_constant<bool, Takes<F, S, G, Row>::value>());
}

}  // namespace driver

template <typename S, typename G, typename Targets>
kernel::Result evaluate_kind(const kernel::Loss& spec, const kernel::Execution& e, std::size_t n,
                             std::size_t classes, const S* scores, const Targets& t, double* values,
                             G* grads, double upstream, const double* row_upstream) {
  switch (spec.kind) {
    case kernel::cross_entropy: {
      loss::CrossEntropy f;
      f.label_smoothing = spec.label_smoothing;
      return driver::call(f, e, n, classes, scores, t, values, grads, upstream, row_upstream);
    }
    case kernel::multilabel_cross_entropy: {
      loss::MultilabelCrossEntropy f;
      return driver::call(f, e, n, classes, scores, t, values, grads, upstream, row_upstream);
    }
    case kernel::topk_cross_entropy: {
      loss::TopKCrossEntropy f;
      f.k = spec.k;
      return driver::call(f, e, n, classes, scores, t, values, grads, upstream, row_upstream);
    }
    case kernel::multiclass_hinge: {
      loss::MulticlassHinge f;
      f.gamma = spec.gamma;
      return driver::call(f, e, n, classes, scores, t, values, grads, upstream, row_upstream);
    }
    case kernel::topk_hinge_alpha: {
      loss::TopKHingeAlpha f;
      f.k = spec.k;
      f.gamma = spec.gamma;
      return driver::call(f, e, n, classes, scores, t, values, grads, upstream, row_upstream);
    }
    case kernel::topk_hinge_beta: {
      loss::TopKHingeBeta f;
      f.k = spec.k;
      f.gamma = spec.gamma;
      return driver::call(f, e, n, classes, scores, t, values, grads, upstream, row_upstream);
    }
    case kernel::multilabel_hinge: {
      loss::MultilabelHinge f;
      f.gamma = spec.gamma;
      return driver::call(f, e, n, classes, scores, t, values, grads, upstream, row_upstream);
    }
    case kernel::proximal_cross_entropy: {
      loss::ProximalCrossEntropy f;
      f.gamma = spec.gamma;
      return driver::call(f, e, n, classes, scores, t, values, grads, upstream, row_upstream);
    }
    case kernel::distillation: {
      loss::Distillation f;
      f.temperature = spec.temperature;
      return driver::call(f, e, n, classes, scores, t, values, grads, upstream, row_upstream);
    }
    case kernel::proximal_distillation: {
      loss::ProximalDistillation f;
      f.temperature = spec.temperature;
      f.gamma = spec.gamma;
      return driver::call(f, e, n, classes, scores, t, values, grads, upstream, row_upstream);
    }
    default: {
      const kernel::Result r = {kernel::status_unsupported, 0, n};
      return r;
    }
  }
}

}  // namespace core
SDCA_ISA_END

#endif
#endif  // SDCA_KERNEL_PASS
