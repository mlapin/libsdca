// XLA FFI handlers of the native JAX losses on the CPU (libsdca.jax, backend "native"), built
// on first use by _jax_ffi.py against the headers of the running jaxlib. They call
// sdca_internal_loss_rows of the loaded native library (bound once by sdca_jax_bind) on XLA's
// buffers, as the host callback they replace calls _core._host_loss_and_grad: the same scores, label mask,
// options and threads. Rows the native library rejects give NaN, as in the host callback.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"
#include "_cpu_distill.h"
namespace {
namespace sdca_cpu {
inline dlanes::vd proximal_log(dlanes::vd x) {
  for (int i = 0; i < dlanes::L; ++i) x[i] = std::log(x[i]);
  return x;
}
}  // namespace sdca_cpu
}  // namespace
#include "_cpu_prox_distill.h"

namespace ffi = xla::ffi;

namespace {

// struct sdca_internal_options of src/c_api_internal.h; sdca_jax_bind checks its size.
struct sdca_internal_options {
  std::size_t size;
  int float_work, loss;
  std::size_t k;
  double gamma, alpha;
  int projection_precision;
  std::size_t num_threads;
};

using Rows = int (*)(const sdca_internal_options*, int, std::size_t, std::size_t, const void*,
                     const std::int64_t*, const std::uint8_t*, double*, void*);
using LastError = const char* (*)();
Rows loss_rows = nullptr;
LastError last_error = nullptr;

// Of include/sdca/c_api.h.
const int invalid_argument = -2;
const int dtype_float32 = 0, dtype_float64 = 1;
const int proximal_cross_entropy = 7;
// num_threads 0: SDCA_NUM_THREADS, else the available CPUs, as XLA's own CPU operations.
const std::size_t threads_default = 0;

// Integer labels of any width as int64, as NumPy's astype (unsigned ones wrap).
template <typename T>
void widen(const void* data, std::size_t n, std::int64_t* out) {
  const T* labels = static_cast<const T*>(data);
  for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<std::int64_t>(labels[i]);
}

bool widen_labels(ffi::DataType type, const void* data, std::size_t n, std::int64_t* out) {
  switch (type) {
    case ffi::DataType::S8:
      widen<std::int8_t>(data, n, out);
      return true;
    case ffi::DataType::S16:
      widen<std::int16_t>(data, n, out);
      return true;
    case ffi::DataType::S32:
      widen<std::int32_t>(data, n, out);
      return true;
    case ffi::DataType::S64:
      widen<std::int64_t>(data, n, out);
      return true;
    case ffi::DataType::U8:
      widen<std::uint8_t>(data, n, out);
      return true;
    case ffi::DataType::U16:
      widen<std::uint16_t>(data, n, out);
      return true;
    case ffi::DataType::U32:
      widen<std::uint32_t>(data, n, out);
      return true;
    case ffi::DataType::U64:
      widen<std::uint64_t>(data, n, out);
      return true;
    default:
      return false;
  }
}

template <typename T>
bool finite(const void* data, std::size_t size) {
  const T* z = static_cast<const T*>(data);
  for (std::size_t i = 0; i < size; ++i)
    if (!std::isfinite(z[i])) return false;
  return true;
}

// The message of a failed call, as the host callback raised it: loss_and_grad checks the
// scores, then a positive label per row, before the native call, whose row messages it renames.
std::string failure(int loss, const ffi::AnyBuffer& z, std::size_t n, std::size_t k,
                    const std::int64_t* labels, const std::uint8_t* positive) {
  const bool fp32 = z.element_type() == ffi::DataType::F32;
  if (!(fp32 ? finite<float>(z.untyped_data(), n * k) : finite<double>(z.untyped_data(), n * k)))
    return "Inputs must be finite";
  for (std::size_t i = 0; i < n; ++i) {
    bool marked = false;
    if (labels) {
      marked = labels[i] >= 0 && static_cast<std::uint64_t>(labels[i]) < k;
    } else {
      for (std::size_t j = 0; j < k && !marked; ++j) marked = positive[i * k + j] != 0;
    }
    if (!marked) return "Every example must have at least one positive label";
  }
  const std::string message = last_error();
  if (message == "This loss takes one positive label per row")
    return "This loss needs exactly one target";
  if (message == "This loss needs a negative label per row")
    return loss == proximal_cross_entropy ? "Proximal entropy needs a non-target class"
                                          : "Multilabel hinge needs a negative label";
  return message;
}

// The rows n and classes of float32 or float64 scores z (..., classes).
ffi::Error rows_of(const ffi::AnyBuffer& z, std::size_t& n, std::size_t& classes,
                   bool allow_half = false) {
  const ffi::DataType type = z.element_type();
  if (type != ffi::DataType::F32 && type != ffi::DataType::F64 &&
      !(allow_half && (type == ffi::DataType::F16 || type == ffi::DataType::BF16)))
    return ffi::Error::InvalidArgument("libsdca: scores must be float32 or float64");
  const auto dims = z.dimensions();
  if (dims.size() == 0 || dims[dims.size() - 1] < 2)
    return ffi::Error::InvalidArgument("libsdca: scores need a class axis of two classes");
  classes = static_cast<std::size_t>(dims[dims.size() - 1]);
  n = z.element_count() / classes;
  if (n == 0) return ffi::Error::InvalidArgument("logits must have shape (n>=1,classes>=2)");
  return ffi::Error::Success();
}

// The targets as sdca_internal_loss_rows takes them: int64 labels or a 0/1 mask
// (exactly one of labels and positive is set).
struct targets {
  std::vector<std::int64_t> wide;
  const std::int64_t* labels = nullptr;
  const std::uint8_t* positive = nullptr;
};

ffi::Error targets_of(const ffi::AnyBuffer& target, std::size_t n, std::size_t classes,
                      targets& out) {
  const ffi::DataType type = target.element_type();
  if (target.element_count() != (type == ffi::DataType::PRED ? n * classes : n))
    return ffi::Error::InvalidArgument("libsdca: target does not match the scores");
  if (type == ffi::DataType::PRED) {
    out.positive = static_cast<const std::uint8_t*>(target.untyped_data());
    return ffi::Error::Success();
  }
  out.labels = static_cast<const std::int64_t*>(target.untyped_data());
  if (type != ffi::DataType::S64) {
    out.wide.resize(n);
    if (!widen_labels(type, target.untyped_data(), n, out.wide.data()))
      return ffi::Error::InvalidArgument("libsdca: labels must be integers");
    out.labels = out.wide.data();
  }
  return ffi::Error::Success();
}

// The options of a call; precision is the code of the FFI attribute: 0 float work on float32
// storage, 1 double work on float32, 2 double work on float64.
sdca_internal_options options_of(int loss, std::int64_t k, double gamma, double alpha,
                                 int precision) {
  sdca_internal_options options = {};
  options.size = sizeof(sdca_internal_options);
  options.float_work = precision == 0;
  options.loss = loss;
  options.k = static_cast<std::size_t>(k);
  options.gamma = gamma;
  options.alpha = alpha;
  options.num_threads = threads_default;
  return options;
}

// After a failed call, each row again on its own: a row the native library rejects (invalid
// traced inputs, or a proximal normalization that does not converge) gives NaN losses and
// gradients, as the jax.numpy losses. False if every row succeeds alone (the failure is not
// a row's: the call fails).
bool nan_rows(const sdca_internal_options& options, int dtype, std::size_t n, std::size_t classes,
              const ffi::AnyBuffer& z, const targets& t, double* values, void* gradients) {
  const std::size_t width = dtype == dtype_float32 ? sizeof(float) : sizeof(double);
  const char* scores = static_cast<const char*>(z.untyped_data());
  char* entries = static_cast<char*>(gradients);
  bool failed = false;
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t offset = i * classes * width;
    void* row_gradient = entries ? entries + offset : nullptr;
    const int code =
        loss_rows(&options, dtype, 1, classes, scores + offset, t.labels ? t.labels + i : nullptr,
                  t.positive ? t.positive + i * classes : nullptr, values + i, row_gradient);
    if (code == 0) continue;
    failed = true;
    values[i] = std::nan("");
    if (!entries) continue;
    if (dtype == dtype_float32) {
      float* g = reinterpret_cast<float*>(row_gradient);
      for (std::size_t j = 0; j < classes; ++j) g[j] = std::nanf("");
    } else {
      double* g = reinterpret_cast<double*>(row_gradient);
      for (std::size_t j = 0; j < classes; ++j) g[j] = std::nan("");
    }
  }
  return failed;
}

// Losses (float64, or float32 rounded once from double) and gradients (the dtype of z, or
// null for the losses alone) of scores z (..., classes) and integer labels (...) or a label
// mask (..., classes).
ffi::Error loss_gradient(ffi::AnyBuffer z, ffi::AnyBuffer target, ffi::AnyBuffer values,
                         void* gradients, std::int32_t loss, std::int64_t k, double gamma,
                         double alpha, std::int32_t precision) {
  if (!loss_rows || !last_error) return ffi::Error::Internal("libsdca: sdca_jax_bind not called");
  std::size_t n = 0, classes = 0;
  targets t;
  ffi::Error error = rows_of(z, n, classes);
  if (error.success()) error = targets_of(target, n, classes, t);
  if (error.failure()) return error;
  const sdca_internal_options options = options_of(loss, k, gamma, alpha, precision);
  const bool float64_values = values.element_type() == ffi::DataType::F64;
  std::vector<double> computed(float64_values ? 0 : n);
  double* out = float64_values ? static_cast<double*>(values.untyped_data()) : computed.data();
  const int dtype = z.element_type() == ffi::DataType::F32 ? dtype_float32 : dtype_float64;
  const int code = loss_rows(&options, dtype, n, classes, z.untyped_data(), t.labels, t.positive,
                             out, gradients);
  if (code != 0 && !nan_rows(options, dtype, n, classes, z, t, out, gradients)) {
    const std::string message = failure(loss, z, n, classes, t.labels, t.positive);
    return code == invalid_argument ? ffi::Error::InvalidArgument(message)
                                    : ffi::Error::Internal(message);
  }
  if (!float64_values) {
    float* rounded = static_cast<float*>(values.untyped_data());
    for (std::size_t i = 0; i < n; ++i) rounded[i] = static_cast<float>(computed[i]);
  }
  return ffi::Error::Success();
}

ffi::Error loss_and_gradient(ffi::AnyBuffer z, ffi::AnyBuffer target,
                             ffi::Result<ffi::AnyBuffer> values,
                             ffi::Result<ffi::AnyBuffer> gradients, std::int32_t loss,
                             std::int64_t k, double gamma, double alpha, std::int32_t precision) {
  return loss_gradient(z, target, *values, gradients->untyped_data(), loss, k, gamma, alpha,
                       precision);
}

// The losses alone; no gradient buffer is allocated.
ffi::Error loss_values(ffi::AnyBuffer z, ffi::AnyBuffer target, ffi::Result<ffi::AnyBuffer> values,
                       std::int32_t loss, std::int64_t k, double gamma, double alpha,
                       std::int32_t precision) {
  return loss_gradient(z, target, *values, nullptr, loss, k, gamma, alpha, precision);
}

// Shared SIMD distillation, with native general-precision rows for exceptional inputs.
using RowCallback = void (*)(void*, std::size_t, std::size_t);
using RunRows = int (*)(std::size_t, std::size_t, std::size_t, RowCallback, void*);
using DistillForward = int (*)(const sdca_internal_options*, int, std::size_t, std::size_t,
                               const void*, const void*, double, double*, double*);
using DistillBackward = int (*)(const sdca_internal_options*, int, std::size_t, std::size_t,
                                const void*, const void*, double, const double*, const double*,
                                std::size_t, void*);
RunRows run_rows = nullptr;
DistillForward distill_forward = nullptr;
DistillBackward distill_backward = nullptr;
using ProxDistill = int (*)(const sdca_internal_options*, int, std::size_t, std::size_t,
                            const void*, const void*, double, double, double*, void*);
ProxDistill prox_distill = nullptr;

template <typename Out, typename In>
const Out* row_view(const In* x, int64_t n, Out* buffer) {
  if constexpr (std::is_same_v<Out, In>)
    return x;
  else {
    for (int64_t i = 0; i < n; ++i) buffer[i] = static_cast<Out>(x[i]);
    return buffer;
  }
}

template <typename T>
bool general_distillation(const T* s, const T* t, int64_t c, double temp, double& value,
                          T* gradient, sdca_cpu::DistillationBuffers&) {
  constexpr bool wide = std::is_same_v<T, double>;
  sdca_internal_options options = options_of(0, 1, 1, 0, wide ? 2 : 1);
  options.num_threads = 1;
  double stats[7];
  const int dtype = wide ? dtype_float64 : dtype_float32;
  int code = distill_forward(&options, dtype, 1, c, s, t, temp, &value, stats);
  if (!code && gradient)
    code = distill_backward(&options, dtype, 1, c, s, t, temp, stats, &temp, 0, gradient);
  return !code;
}

template <typename T>
bool shared_distillation_value(const T* s, const T* t, int64_t c, double temp, double& value,
                               sdca_cpu::DistillationBuffers& buffers) {
  return sdca_cpu::distillation_forward_row(
      s, t, c, temp, !std::is_same_v<T, double>, nullptr, value, buffers, row_view<double, T>,
      [](const T* sr, const T* tr, int64_t count, double temperature, double*, double& v,
         sdca_cpu::DistillationBuffers& scratch) {
        return general_distillation(sr, tr, count, temperature, v, static_cast<T*>(nullptr),
                                    scratch);
      });
}

template <typename T, bool NeedGradient>
struct DistillationTask {
  const T *s, *t;
  T* gradient;
  double* values;
  int64_t classes;
  double temperature;

  static void call(void* context, std::size_t begin, std::size_t end) {
    auto& task = *static_cast<DistillationTask*>(context);
    const int64_t c = task.classes;
    constexpr bool wide = std::is_same_v<T, double>;
    const int64_t cp = (c + sdca_cpu::lanes::L - 1) / sdca_cpu::lanes::L * sdca_cpu::lanes::L;
    sdca_cpu::DistillationBuffers buffers{std::vector<float>(wide ? 0 : 10 * cp), {}, {}};
    std::vector<char> done;
    if (c <= sdca_cpu::distill_small::kMaxClasses) {
      using V = std::conditional_t<wide, sdca_cpu::dlanes::vd, sdca_cpu::lanes::vf>;
      done = sdca_cpu::distillation_blocks<V>(task.s, task.t, begin, end, c, task.temperature,
                                              task.values, task.gradient);
    }
    for (std::size_t row = begin; row < end; ++row) {
      if (!done.empty() && done[row - begin]) continue;
      const T *s = task.s + row * c, *t = task.t + row * c;
      T* g = NeedGradient ? task.gradient + row * c : nullptr;
      double& v = task.values[row];
      const bool ok = NeedGradient
                          ? sdca_cpu::distillation_grad_row(
                                s, t, c, cp, task.temperature, v, g, buffers, row_view<float, T>,
                                row_view<double, T>, general_distillation<T>)
                          : shared_distillation_value(s, t, c, task.temperature, v, buffers);
      if (!ok) {
        v = std::numeric_limits<double>::quiet_NaN();
        if (g) std::fill(g, g + c, std::numeric_limits<T>::quiet_NaN());
      }
    }
  }
};

template <typename T, bool NeedGradient>
ffi::Error shared_distillation(const ffi::AnyBuffer& s, const ffi::AnyBuffer& t,
                               ffi::AnyBuffer values, ffi::AnyBuffer gradient, std::size_t n,
                               std::size_t c, double temp, int reduction) {
  std::vector<double> losses(n);
  DistillationTask<T, NeedGradient> task{
      static_cast<const T*>(s.untyped_data()),
      static_cast<const T*>(t.untyped_data()),
      gradient.element_count() ? static_cast<T*>(gradient.untyped_data()) : nullptr,
      losses.data(),
      static_cast<int64_t>(c),
      temp};
  const auto threads = threads_default;
  const int code = run_rows(n, c, threads, DistillationTask<T, NeedGradient>::call, &task);
  if (code) return ffi::Error::Internal(last_error());
  T* out = static_cast<T*>(values.untyped_data());
  if (reduction == 0) {
    for (std::size_t row = 0; row < n; ++row) out[row] = static_cast<T>(losses[row]);
  } else {
    sdca_cpu::Sum<std::is_same_v<T, double>> total;
    for (double v : losses) total.add(v);
    const double sum = total.value();
    out[0] = static_cast<T>(reduction == 2 ? sum / n : sum);
  }
  return ffi::Error::Success();
}

ffi::Error distillation_shared(ffi::AnyBuffer s, ffi::AnyBuffer t,
                               ffi::Result<ffi::AnyBuffer> values,
                               ffi::Result<ffi::AnyBuffer> gradient, double temperature,
                               std::int32_t reduction) {
  if (!run_rows || !distill_forward || !distill_backward)
    return ffi::Error::Internal("libsdca: shared CPU kernels are not bound");
  std::size_t n = 0, c = 0;
  ffi::Error error = rows_of(s, n, c);
  if (error.failure()) return error;
  if (t.element_type() != s.element_type() || t.element_count() != s.element_count() ||
      values->element_type() != s.element_type() || gradient->element_type() != s.element_type() ||
      (gradient->element_count() != 0 && gradient->element_count() != s.element_count()) ||
      values->element_count() != (reduction == 0 ? n : 1) || reduction < 0 || reduction > 2 ||
      !std::isfinite(temperature) || temperature <= 0)
    return ffi::Error::InvalidArgument("libsdca: invalid shared distillation buffers or options");
  if (s.element_type() == ffi::DataType::F32) {
    if (gradient->element_count())
      return shared_distillation<float, true>(s, t, *values, *gradient, n, c, temperature,
                                              reduction);
    return shared_distillation<float, false>(s, t, *values, *gradient, n, c, temperature,
                                             reduction);
  }
  if (gradient->element_count())
    return shared_distillation<double, true>(s, t, *values, *gradient, n, c, temperature,
                                             reduction);
  return shared_distillation<double, false>(s, t, *values, *gradient, n, c, temperature, reduction);
}

// Statistics path for 16-bit storage and explicit fp64 work on fp32 storage. The
// seven doubles per row are opaque uint32 words in JAX, including with x64 disabled.
template <typename T>
constexpr int storage_dtype() {
  if constexpr (std::is_same_v<T, sdca_cpu::Half>) return 2;
  if constexpr (std::is_same_v<T, sdca_cpu::BFloat16>) return 3;
  return std::is_same_v<T, double> ? dtype_float64 : dtype_float32;
}

template <typename T, bool Backward>
struct DistillationStatsTask {
  const T *s, *t;
  T* gradient;
  double *values, *stats;
  const float* upstream;
  int64_t classes;
  std::size_t rows;
  double temperature;
  bool work32;
  int reduction;

  void backward_row(std::size_t row, int64_t cp, sdca_cpu::DistillationBuffers& buffers,
                    const sdca_internal_options& options) {
    const T *sr = s + row * classes, *tr = t + row * classes;
    double* st = stats ? stats + row * 7 : nullptr;
    const double u = double(upstream[reduction == 0 ? row : 0]) / (reduction == 2 ? rows : 1);
    sdca_cpu::distillation_backward_row(
        sr, tr, classes, cp, temperature, work32, st, u, gradient + row * classes, buffers.narrow,
        buffers.lanes64, row_view<double, T>,
        [&](const T* student, const T* teacher, int64_t count, double temp, double weight, T* out,
            std::vector<double>&) {
          double value, saved[7];
          int code = distill_forward(&options, storage_dtype<T>(), 1, count, student, teacher, temp,
                                     &value, saved);
          if (!code)
            code = distill_backward(&options, storage_dtype<T>(), 1, count, student, teacher, temp,
                                    saved, &weight, 0, out);
          if (code) std::fill(out, out + count, T(std::numeric_limits<double>::quiet_NaN()));
        });
  }

  void forward_row(std::size_t row, sdca_cpu::DistillationBuffers& buffers,
                   const sdca_internal_options& options) {
    double* st = stats ? stats + row * 7 : nullptr;
    double& value = values[row];
    const bool ok = sdca_cpu::distillation_forward_row(
        s + row * classes, t + row * classes, classes, temperature, work32, st, value, buffers,
        row_view<double, T>,
        [&](const T* sr, const T* tr, int64_t count, double temp, double* saved, double& v,
            sdca_cpu::DistillationBuffers&) {
          double native_stats[7];
          const int code = distill_forward(&options, storage_dtype<T>(), 1, count, sr, tr, temp, &v,
                                           native_stats);
          if (saved) saved[0] = -1;
          return !code && !std::isnan(v);
        });
    if (!ok) {
      value = std::numeric_limits<double>::quiet_NaN();
      if (st) st[0] = -1;
    }
    // Torch's statistics path rounds individual values to their public dtype.
    value = static_cast<float>(value);
  }

  static void call(void* context, std::size_t begin, std::size_t end) {
    auto& task = *static_cast<DistillationStatsTask*>(context);
    const int64_t c = task.classes;
    const int64_t cp = (c + sdca_cpu::lanes::L - 1) / sdca_cpu::lanes::L * sdca_cpu::lanes::L;
    sdca_cpu::DistillationBuffers buffers{
        std::vector<float>(Backward && task.work32 ? 6 * cp : 0), {}, {}};
    sdca_internal_options options = options_of(0, 1, 1, 0, task.work32 ? 1 : 2);
    options.num_threads = 1;
    std::vector<char> done;
    if constexpr (!Backward) {
      if (!task.stats && c <= sdca_cpu::distill_small::kMaxClasses) {
        done = task.work32 ? sdca_cpu::distillation_blocks<sdca_cpu::lanes::vf>(
                                 task.s, task.t, begin, end, c, task.temperature, task.values,
                                 static_cast<T*>(nullptr))
                           : sdca_cpu::distillation_blocks<sdca_cpu::dlanes::vd>(
                                 task.s, task.t, begin, end, c, task.temperature, task.values,
                                 static_cast<T*>(nullptr));
      }
    }
    for (std::size_t row = begin; row < end; ++row) {
      if (!done.empty() && done[row - begin]) {
        task.values[row] = static_cast<float>(task.values[row]);
        continue;
      }
      if constexpr (Backward) {
        task.backward_row(row, cp, buffers, options);
      } else {
        task.forward_row(row, buffers, options);
      }
    }
  }
};

template <typename T, bool Backward>
ffi::Error shared_stats(const ffi::AnyBuffer& s, const ffi::AnyBuffer& t, ffi::AnyBuffer values,
                        ffi::AnyBuffer stats, ffi::AnyBuffer upstream, ffi::AnyBuffer gradient,
                        std::size_t n, std::size_t c, double temperature, bool work32,
                        int reduction) {
  std::vector<double> losses(Backward ? 0 : n);
  DistillationStatsTask<T, Backward> task{
      static_cast<const T*>(s.untyped_data()),
      static_cast<const T*>(t.untyped_data()),
      Backward ? static_cast<T*>(gradient.untyped_data()) : nullptr,
      losses.data(),
      stats.element_count() ? static_cast<double*>(stats.untyped_data()) : nullptr,
      Backward ? static_cast<const float*>(upstream.untyped_data()) : nullptr,
      static_cast<int64_t>(c),
      n,
      temperature,
      work32,
      reduction};
  const auto threads = threads_default;
  const int code = run_rows(n, c, threads, DistillationStatsTask<T, Backward>::call, &task);
  if (code) return ffi::Error::Internal(last_error());
  if constexpr (!Backward) {
    auto* out = static_cast<float*>(values.untyped_data());
    if (reduction == 0) {
      for (std::size_t row = 0; row < n; ++row) out[row] = static_cast<float>(losses[row]);
    } else {
      double sum = 0;
      for (double v : losses) sum += v;
      out[0] = static_cast<float>(reduction == 2 ? sum / n : sum);
    }
  }
  return ffi::Error::Success();
}

template <bool Backward>
bool stats_outputs_valid(const ffi::AnyBuffer& s, const ffi::AnyBuffer& values,
                         const ffi::AnyBuffer& stats, const ffi::AnyBuffer& upstream,
                         const ffi::AnyBuffer& gradient, std::size_t n, std::size_t c,
                         int reduction) {
  return Backward
             ? gradient.element_type() == s.element_type() && gradient.element_count() == n * c &&
                   upstream.element_type() == ffi::DataType::F32 &&
                   upstream.element_count() == (reduction == 0 ? n : 1) &&
                   stats.element_count() == 14 * n
             : values.element_type() == ffi::DataType::F32 &&
                   values.element_count() == (reduction == 0 ? n : 1) &&
                   (stats.element_count() == 0 || stats.element_count() == 14 * n);
}

template <bool Backward>
ffi::Error dispatch_stats(ffi::AnyBuffer s, ffi::AnyBuffer t, ffi::AnyBuffer values,
                          ffi::AnyBuffer stats, ffi::AnyBuffer upstream, ffi::AnyBuffer gradient,
                          double temperature, std::int32_t work32, std::int32_t reduction) {
  if (!run_rows || !distill_forward || !distill_backward)
    return ffi::Error::Internal("libsdca: shared CPU kernels are not bound");
  std::size_t n = 0, c = 0;
  ffi::Error error = rows_of(s, n, c, true);
  if (error.failure()) return error;
  if (t.element_type() != s.element_type() || t.element_count() != n * c ||
      !stats_outputs_valid<Backward>(s, values, stats, upstream, gradient, n, c, reduction) ||
      stats.element_type() != ffi::DataType::U32 || reduction < 0 || reduction > 2 ||
      !std::isfinite(temperature) || temperature <= 0)
    return ffi::Error::InvalidArgument("libsdca: invalid shared CPU distillation statistics");
  if (s.element_type() == ffi::DataType::F16)
    return shared_stats<sdca_cpu::Half, Backward>(s, t, values, stats, upstream, gradient, n, c,
                                                  temperature, work32, reduction);
  if (s.element_type() == ffi::DataType::BF16)
    return shared_stats<sdca_cpu::BFloat16, Backward>(s, t, values, stats, upstream, gradient, n, c,
                                                      temperature, work32, reduction);
  if (s.element_type() == ffi::DataType::F32)
    return shared_stats<float, Backward>(s, t, values, stats, upstream, gradient, n, c, temperature,
                                         work32, reduction);
  return ffi::Error::InvalidArgument("libsdca: unsupported statistics storage dtype");
}

ffi::Error distillation_stats_forward(ffi::AnyBuffer s, ffi::AnyBuffer t,
                                      ffi::Result<ffi::AnyBuffer> values,
                                      ffi::Result<ffi::AnyBuffer> stats, double temperature,
                                      std::int32_t work32, std::int32_t reduction) {
  return dispatch_stats<false>(s, t, *values, *stats, *values, s, temperature, work32, reduction);
}
ffi::Error distillation_stats_backward(ffi::AnyBuffer s, ffi::AnyBuffer t, ffi::AnyBuffer stats,
                                       ffi::AnyBuffer upstream,
                                       ffi::Result<ffi::AnyBuffer> gradient, double temperature,
                                       std::int32_t work32, std::int32_t reduction) {
  return dispatch_stats<true>(s, t, upstream, stats, upstream, *gradient, temperature, work32,
                              reduction);
}

// The same SIMD solve as Torch. Rare inputs outside its fast statistics use
// the native scalar solver, which preserves the full double input range.
template <typename T, typename G>
struct ProximalDistillationTask {
  const T *student, *teacher;
  G *values, *gradient;
  int64_t classes;
  double temperature, alpha;

  static void call(void* context, std::size_t begin, std::size_t end) {
    const auto& task = *static_cast<ProximalDistillationTask*>(context);
    const int64_t c = task.classes;
    constexpr bool compensated = std::is_same_v<T, double>;
    std::vector<double> buffer(13 * c);
    std::vector<int32_t> index(c);
    double* b = buffer.data();
    sdca_internal_options options = options_of(0, 1, 1, task.alpha, compensated ? 2 : 1);
    options.num_threads = 1;
    for (std::size_t row = begin; row < end; ++row) {
      const T *s = task.student + row * c, *t = task.teacher + row * c;
      G* gradient = task.gradient + row * c;
      sdca_cpu::DistillationStats stats;
      const sdca_cpu::distill64::Row inputs{row_view<double, T>(s, c, b),
                                            row_view<double, T>(t, c, b + c),
                                            b + 4 * c,
                                            b + 2 * c,
                                            b + 3 * c,
                                            c};
      const auto status = sdca_cpu::distill64::prox_stats<compensated>(
          inputs, task.temperature, b + 5 * c, b + 6 * c, b, b + c, stats);
      double value;
      if (status == sdca_cpu::distill64::kFast) {
        value = sdca_cpu::prox_distillation_solve<G, compensated>(
            c, task.alpha, stats, b, b + c, b + 2 * c, b + 3 * c, b + 4 * c, b + 5 * c, b + 6 * c,
            index.data(), gradient);
      } else {
        const int code = prox_distill(&options, storage_dtype<T>(), 1, c, s, t, task.temperature,
                                      task.alpha, &value, gradient);
        if (code) {
          value = std::numeric_limits<double>::quiet_NaN();
          std::fill(gradient, gradient + c, std::numeric_limits<G>::quiet_NaN());
        }
      }
      task.values[row] = static_cast<G>(value);
    }
  }
};

template <typename T, typename G>
ffi::Error proximal_distillation_rows(const ffi::AnyBuffer& s, const ffi::AnyBuffer& t,
                                      ffi::AnyBuffer values, ffi::AnyBuffer gradient, std::size_t n,
                                      std::size_t c, double temperature, double alpha) {
  ProximalDistillationTask<T, G> task{static_cast<const T*>(s.untyped_data()),
                                      static_cast<const T*>(t.untyped_data()),
                                      static_cast<G*>(values.untyped_data()),
                                      static_cast<G*>(gradient.untyped_data()),
                                      static_cast<int64_t>(c),
                                      temperature,
                                      alpha};
  const auto threads = threads_default;
  const int code = run_rows(n, 8 * c, threads, ProximalDistillationTask<T, G>::call, &task);
  return code ? ffi::Error::Internal(last_error()) : ffi::Error::Success();
}

ffi::Error proximal_distillation(ffi::AnyBuffer s, ffi::AnyBuffer t,
                                 ffi::Result<ffi::AnyBuffer> values,
                                 ffi::Result<ffi::AnyBuffer> gradient, double temperature,
                                 double alpha) {
  if (!prox_distill || !run_rows)
    return ffi::Error::Internal("libsdca: proximal CPU kernel is not bound");
  std::size_t n = 0, c = 0;
  ffi::Error error = rows_of(s, n, c, true);
  if (error.failure()) return error;
  const auto output =
      s.element_type() == ffi::DataType::F64 ? ffi::DataType::F64 : ffi::DataType::F32;
  if (t.element_type() != s.element_type() || t.element_count() != n * c ||
      values->element_type() != output || values->element_count() != n ||
      gradient->element_type() != output || gradient->element_count() != n * c ||
      !std::isfinite(temperature) || temperature <= 0 || !std::isfinite(alpha) || alpha <= 0)
    return ffi::Error::InvalidArgument("libsdca: invalid proximal distillation buffers or options");
  if (s.element_type() == ffi::DataType::F16)
    return proximal_distillation_rows<sdca_cpu::Half, float>(s, t, *values, *gradient, n, c,
                                                             temperature, alpha);
  if (s.element_type() == ffi::DataType::BF16)
    return proximal_distillation_rows<sdca_cpu::BFloat16, float>(s, t, *values, *gradient, n, c,
                                                                 temperature, alpha);
  if (s.element_type() == ffi::DataType::F32)
    return proximal_distillation_rows<float, float>(s, t, *values, *gradient, n, c, temperature,
                                                    alpha);
  return proximal_distillation_rows<double, double>(s, t, *values, *gradient, n, c, temperature,
                                                    alpha);
}

}  // namespace

// Binds the native library's sdca_internal_loss_rows and sdca_last_error; 0, or -1 if its
// struct sdca_internal_options (options_size) is not the one above.
extern "C" int sdca_jax_bind(void* rows, void* error, std::size_t options_size) {
  if (options_size != sizeof(sdca_internal_options)) return -1;
  loss_rows = reinterpret_cast<Rows>(rows);
  last_error = reinterpret_cast<LastError>(error);
  return 0;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(sdca_jax_loss_gradient, loss_and_gradient,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<std::int32_t>("loss")
                                  .Attr<std::int64_t>("k")
                                  .Attr<double>("gamma")
                                  .Attr<double>("alpha")
                                  .Attr<std::int32_t>("precision"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(sdca_jax_loss, loss_values,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<std::int32_t>("loss")
                                  .Attr<std::int64_t>("k")
                                  .Attr<double>("gamma")
                                  .Attr<double>("alpha")
                                  .Attr<std::int32_t>("precision"));

extern "C" void sdca_jax_bind_shared(void* rows, void* forward, void* backward) {
  run_rows = reinterpret_cast<RunRows>(rows);
  distill_forward = reinterpret_cast<DistillForward>(forward);
  distill_backward = reinterpret_cast<DistillBackward>(backward);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(sdca_jax_distillation, distillation_shared,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<double>("temperature")
                                  .Attr<std::int32_t>("reduction"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(sdca_jax_distillation_stats_forward, distillation_stats_forward,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<double>("temperature")
                                  .Attr<std::int32_t>("work32")
                                  .Attr<std::int32_t>("reduction"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(sdca_jax_distillation_stats_backward, distillation_stats_backward,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<double>("temperature")
                                  .Attr<std::int32_t>("work32")
                                  .Attr<std::int32_t>("reduction"));

extern "C" void sdca_jax_bind_prox(void* rows) {
  prox_distill = reinterpret_cast<ProxDistill>(rows);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(sdca_jax_proximal_distillation, proximal_distillation,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<double>("temperature")
                                  .Attr<double>("alpha"));
