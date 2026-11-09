// XLA FFI handlers of the native JAX losses on the CPU (libsdca.jax, backend "native"), built
// on first use by _jax_ffi.py against the headers of the running jaxlib. They call
// sdca_loss_gradient_rows of the loaded native library (bound once by sdca_jax_bind) on XLA's
// buffers, as the host callback they replace calls loss_and_grad: the same scores, label mask,
// options and threads, and the same messages in the same order for invalid inputs.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

// struct sdca_options of include/sdca/c_api.h; sdca_jax_bind checks its size.
struct sdca_options {
  std::size_t size;
  int precision, loss;
  double c;
  std::size_t k;
  double gamma, alpha;
  int projection_precision;
  std::size_t num_threads, max_epoch, eval_epoch;
  double tol;
  int eval_on_start, recompute_primal, kernel, warm, shrinking;
};

using Rows = int (*)(const sdca_options*, int, std::size_t, std::size_t, const void*,
                     const std::int64_t*, const std::uint8_t*, double*, void*);
using LastError = const char* (*)();
Rows loss_rows = nullptr;
LastError last_error = nullptr;

// Of include/sdca/c_api.h.
const int invalid_argument = -2;
const int dtype_float32 = 0, dtype_float64 = 1;
const int proximal_cross_entropy = 7;
const std::size_t threads_default = static_cast<std::size_t>(-1);

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
ffi::Error rows_of(const ffi::AnyBuffer& z, std::size_t& n, std::size_t& classes) {
  const ffi::DataType type = z.element_type();
  if (type != ffi::DataType::F32 && type != ffi::DataType::F64)
    return ffi::Error::InvalidArgument("libsdca: scores must be float32 or float64");
  const auto dims = z.dimensions();
  if (dims.size() == 0 || dims[dims.size() - 1] < 2)
    return ffi::Error::InvalidArgument("libsdca: scores need a class axis of two classes");
  classes = static_cast<std::size_t>(dims[dims.size() - 1]);
  n = z.element_count() / classes;
  if (n == 0) return ffi::Error::InvalidArgument("logits must have shape (n>=1,classes>=2)");
  return ffi::Error::Success();
}

// The targets as sdca_loss_gradient_rows takes them: int64 labels or a 0/1 mask
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

// SDCA_OPTIONS_DEFAULT with the loss options of a call.
sdca_options options_of(int loss, std::int64_t k, double gamma, double alpha, int precision) {
  sdca_options options = {};
  options.size = sizeof(sdca_options);
  options.precision = precision;
  options.loss = loss;
  options.c = 1.0;
  options.k = static_cast<std::size_t>(k);
  options.gamma = gamma;
  options.alpha = alpha;
  // The available CPUs, as XLA's own CPU operations, unless SDCA_NUM_THREADS is set.
  options.num_threads = std::getenv("SDCA_NUM_THREADS") ? threads_default : 0;
  options.max_epoch = 1000;
  options.eval_epoch = 10;
  options.tol = 1e-3;
  options.recompute_primal = 1;
  return options;
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
  const sdca_options options = options_of(loss, k, gamma, alpha, precision);
  const bool float64_values = values.element_type() == ffi::DataType::F64;
  std::vector<double> computed(float64_values ? 0 : n);
  double* out = float64_values ? static_cast<double*>(values.untyped_data()) : computed.data();
  const int dtype = z.element_type() == ffi::DataType::F32 ? dtype_float32 : dtype_float64;
  const int code = loss_rows(&options, dtype, n, classes, z.untyped_data(), t.labels, t.positive,
                             out, gradients);
  if (code != 0) {
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

}  // namespace

// Binds the native library's sdca_loss_gradient_rows and sdca_last_error; 0, or -1 if its
// struct sdca_options (options_size) is not the one above.
extern "C" int sdca_jax_bind(void* rows, void* error, std::size_t options_size) {
  if (options_size != sizeof(sdca_options)) return -1;
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
