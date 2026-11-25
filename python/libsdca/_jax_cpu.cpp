// The XLA FFI CPU handler of the libsdca_loss target (libsdca.jax):
// sdca::kernel::evaluate on XLA's own buffers.
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

#include "_frontend_cpu.h"
#include "sdca/utility/row_reduction.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

using libsdca_frontend::Batch;
using libsdca_frontend::Spec;

size_t rows_of(const ffi::AnyBuffer& scores) {
  size_t rows = 1;
  const auto dims = scores.dimensions();
  for (size_t i = 0; i + 1 < dims.size(); ++i) rows *= static_cast<size_t>(dims[i]);
  return rows;
}

size_t classes_of(const ffi::AnyBuffer& scores) {
  const auto dims = scores.dimensions();
  return dims.size() == 0 ? 0 : static_cast<size_t>(dims[dims.size() - 1]);
}

std::vector<double> doubles(const ffi::AnyBuffer& b) {
  const size_t count = b.element_count();
  std::vector<double> out(count);
  if (b.element_type() == ffi::DataType::F64) {
    const double* p = static_cast<const double*>(b.untyped_data());
    out.assign(p, p + count);
  } else {
    const float* p = static_cast<const float*>(b.untyped_data());
    for (size_t i = 0; i < count; ++i) out[i] = p[i];
  }
  return out;
}

// The labels as int64 (JAX without x64 passes int32).
std::vector<int64_t> labels_of(const ffi::AnyBuffer& target) {
  const size_t count = target.element_count();
  std::vector<int64_t> out(count);
  if (target.element_type() == ffi::DataType::S64) {
    const int64_t* p = static_cast<const int64_t*>(target.untyped_data());
    out.assign(p, p + count);
  } else if (target.element_type() == ffi::DataType::S32) {
    const int32_t* p = static_cast<const int32_t*>(target.untyped_data());
    for (size_t i = 0; i < count; ++i) out[i] = p[i];
  } else if (target.element_type() == ffi::DataType::U32) {
    const uint32_t* p = static_cast<const uint32_t*>(target.untyped_data());
    for (size_t i = 0; i < count; ++i) out[i] = p[i];
  } else {
    throw std::invalid_argument("libsdca: labels must be int32, uint32 or int64");
  }
  return out;
}

bool is_teacher(const Spec& spec) {
  return spec.kind == libsdca_frontend::kDistillation ||
         spec.kind == libsdca_frontend::kProximalDistillation;
}

// One group of rows: under vmap, weight and label_smoothing may hold one entry per group.
template <typename S>
void group(const Spec& spec, const Batch<S>& b, const ffi::AnyBuffer& target, const int64_t* labels,
           const double* class_weight, size_t row0) {
  const auto dtype = target.element_type();
  const size_t offset = row0 * b.classes;
  if (labels != nullptr) {
    libsdca_frontend::labels(spec, b, labels + row0, class_weight);
  } else if (dtype == ffi::DataType::PRED) {
    libsdca_frontend::mask(spec, b, static_cast<const uint8_t*>(target.untyped_data()) + offset);
  } else if (is_teacher(spec)) {
    libsdca_frontend::teacher(spec, b, static_cast<const S*>(target.untyped_data()) + offset);
  } else if (dtype == ffi::DataType::F64) {
    libsdca_frontend::probabilities(
        spec, b, static_cast<const double*>(target.untyped_data()) + offset, class_weight);
  } else if (dtype == ffi::DataType::F32) {
    libsdca_frontend::probabilities(
        spec, b, static_cast<const float*>(target.untyped_data()) + offset, class_weight);
  } else {
    libsdca_frontend::probabilities(spec, b, static_cast<const S*>(target.untyped_data()) + offset,
                                    class_weight);
  }
}

size_t mapped_groups(size_t n, size_t weights, size_t smoothing, size_t upstream) {
  const size_t scales = upstream < n ? upstream : 0;
  const size_t groups = std::max<size_t>({1, weights, smoothing, scales});
  const auto incompatible = [groups](size_t count) { return count > 1 && count != groups; };
  if (incompatible(weights) || incompatible(smoothing) || incompatible(scales) || n % groups != 0 ||
      upstream > n) {
    throw std::invalid_argument(
        "libsdca: mapped weight, label_smoothing and scale must share the rows");
  }
  return groups;
}

// A vmap batch can have separate class weights or broadcast one vector over all rows.
std::vector<double> denominator_terms(const Spec& spec, const std::vector<int64_t>& labels,
                                      const std::vector<double>& weights, size_t classes,
                                      int64_t reduction) {
  if (reduction != 2 || labels.empty() || (!spec.ignore && weights.empty())) return {};
  const size_t per = weights.empty() ? labels.size() : labels.size() / (weights.size() / classes);
  std::vector<double> out(labels.size());
  for (size_t i = 0; i < labels.size(); ++i) {
    const int64_t label = labels[i];
    const size_t c =
        static_cast<size_t>(std::max<int64_t>(0, std::min<int64_t>(classes - 1, label)));
    const double w = weights.empty() ? 1.0 : weights[(i / per) * classes + c];
    out[i] = spec.ignore && label == spec.ignore_index ? 0.0 : w;
  }
  return out;
}

template <typename Out>
void store_values(Out* out, const std::vector<double>& rows, const std::vector<double>& terms,
                  size_t outputs, int64_t reduction) {
  if (reduction == 0) {
    sdca::detail::store_row_values(out, rows.data(), nullptr, rows.size(), 0, rows.size());
    return;
  }
  if (outputs == 0) return;
  const size_t per = rows.size() / outputs;
  for (size_t i = 0; i < outputs; ++i) {
    const double* values = rows.empty() ? nullptr : rows.data() + i * per;
    const double* weights = terms.empty() ? nullptr : terms.data() + i * per;
    sdca::detail::store_row_values(out + i, values, weights, per, static_cast<int>(reduction), per);
  }
}

void store_output(ffi::AnyBuffer* values, const std::vector<double>& rows,
                  const std::vector<double>& terms, int64_t reduction) {
  void* out = values->untyped_data();
  if (values->element_type() == ffi::DataType::F64)
    store_values(static_cast<double*>(out), rows, terms, values->element_count(), reduction);
  else
    store_values(static_cast<float*>(out), rows, terms, values->element_count(), reduction);
}

// A scalar upstream (possibly one per mapped example) is the row kernel's
// scalar factor. Only an actual row-wise cotangent needs its own array.
const double* upstream_for_group(Spec& spec, const std::vector<double>& up, size_t n, size_t group,
                                 size_t row0) {
  if (up.empty()) return nullptr;
  if (up.size() == n) return up.data() + row0;
  spec.upstream = up[up.size() == 1 ? 0 : group];
  return nullptr;
}

// Values where the values result has a row each, the gradient (times row_upstream, where it
// has a row each) where the gradient result has the scores' size. The gradient result may be
// the scores' buffer (input_output_aliases): each group reads only its own rows of either.
template <typename S>
void evaluate(Spec spec, const ffi::AnyBuffer& scores, const ffi::AnyBuffer& target,
              const ffi::AnyBuffer& weight, const ffi::AnyBuffer& smoothing,
              const ffi::AnyBuffer& upstream, ffi::AnyBuffer* values, ffi::AnyBuffer* grad,
              int64_t reduction) {
  // Keep subnormal weights and upstreams during their conversion to double as well.
  const sdca::detail::subnormal_environment subnormals;
  const size_t n = rows_of(scores), classes = classes_of(scores);
  const std::vector<double> w = doubles(weight), eps = doubles(smoothing);
  const std::vector<double> up = doubles(upstream);
  const size_t weights = classes == 0 ? 0 : w.size() / classes;
  const size_t groups = mapped_groups(n, weights, eps.size(), up.size());
  std::vector<int64_t> labels;
  const auto dtype = target.element_type();
  if (dtype == ffi::DataType::S32 || dtype == ffi::DataType::S64 || dtype == ffi::DataType::U32) {
    labels = labels_of(target);
  }
  const bool want_values = reduction != 0 || values->element_count() == n;
  std::vector<double> rows(want_values ? n : 0);
  const S* s = static_cast<const S*>(scores.untyped_data());
  S* g = grad->element_count() == n * classes && n > 0 ? static_cast<S*>(grad->untyped_data())
                                                       : nullptr;
  const size_t per = n / groups;
  const bool traced = std::isnan(spec.label_smoothing);
  for (size_t i = 0; i < groups; ++i) {
    if (traced && !eps.empty()) spec.label_smoothing = eps[eps.size() > 1 ? i : 0];
    const double* cw = weights == 0 ? nullptr : w.data() + (weights > 1 ? i * classes : 0);
    const size_t row0 = i * per;
    const Batch<S> b{per,
                     classes,
                     s + row0 * classes,
                     want_values ? rows.data() + row0 : nullptr,
                     g == nullptr ? nullptr : g + row0 * classes,
                     upstream_for_group(spec, up, n, i, row0)};
    group(spec, b, target, labels.empty() ? nullptr : labels.data(), cw, row0);
  }
  if (want_values) {
    const auto terms = denominator_terms(spec, labels, w, classes, reduction);
    store_output(values, rows, terms, reduction);
  }
}

ffi::Error loss(ffi::AnyBuffer scores, ffi::AnyBuffer target, ffi::AnyBuffer weight,
                ffi::AnyBuffer smoothing, ffi::AnyBuffer upstream,
                ffi::Result<ffi::AnyBuffer> values, ffi::Result<ffi::AnyBuffer> grad, int64_t kind,
                int64_t k, double gamma, double temperature, double label_smoothing,
                int64_t ignore_index, int64_t ignore, int64_t precision, int64_t reduction) {
  Spec spec;
  spec.kind = static_cast<int>(kind);
  spec.k = static_cast<size_t>(k);
  spec.gamma = gamma;
  spec.temperature = temperature;
  spec.label_smoothing = label_smoothing;  // NaN: traced, in the smoothing buffer
  spec.ignore_index = ignore_index;
  spec.ignore = ignore != 0;
  spec.precision = static_cast<int>(precision);
  ffi::AnyBuffer* v = &*values;
  ffi::AnyBuffer* g = &*grad;
  try {
    switch (scores.element_type()) {
      case ffi::DataType::F32:
        evaluate<float>(spec, scores, target, weight, smoothing, upstream, v, g, reduction);
        break;
      case ffi::DataType::F64:
        evaluate<double>(spec, scores, target, weight, smoothing, upstream, v, g, reduction);
        break;
      case ffi::DataType::F16:
        evaluate<sdca::half>(spec, scores, target, weight, smoothing, upstream, v, g, reduction);
        break;
      case ffi::DataType::BF16:
        evaluate<sdca::bfloat16>(spec, scores, target, weight, smoothing, upstream, v, g,
                                 reduction);
        break;
      default:
        return ffi::Error::InvalidArgument("libsdca: unsupported scores dtype");
    }
  } catch (const std::out_of_range& error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::invalid_argument& error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
  return ffi::Error::Success();
}

// The row kernel reports exceptional rows so their established framework
// derivative can still supply the result on just those rows.
ffi::Error teacher_vjp(ffi::Buffer<ffi::F32> scores, ffi::Buffer<ffi::F32> teacher,
                       ffi::Buffer<ffi::F32> factors, ffi::ResultBuffer<ffi::F32> gradient,
                       ffi::ResultBuffer<ffi::U8> exceptional, double temperature) {
  const size_t n = exceptional->element_count();
  const auto dims = scores.dimensions();
  if (dims.size() == 0 || dims[dims.size() - 1] <= 0 ||
      teacher.element_count() != scores.element_count() ||
      gradient->element_count() != scores.element_count() ||
      (factors.element_count() != 1 && factors.element_count() != n))
    return ffi::Error::InvalidArgument("libsdca: invalid teacher gradient dimensions");
  const size_t c = static_cast<size_t>(dims[dims.size() - 1]);
  if (scores.element_count() / c != n || !(temperature > 0.0) || !std::isfinite(temperature))
    return ffi::Error::InvalidArgument("libsdca: invalid teacher gradient row or temperature");
  const bool scalar = factors.element_count() == 1;
  const auto kernel = libsdca_frontend::teacher_kernel();
  try {
    sdca::detail::parallel_rows(
        n, c, sdca::detail::environment_threads(),
        [&](size_t lo, size_t hi) {
          const sdca::detail::subnormal_environment subnormals;
          kernel(scores.typed_data() + lo * c, teacher.typed_data() + lo * c, hi - lo, c,
                 1.0 / temperature, factors.typed_data() + (scalar ? 0 : lo), scalar ? 0 : 1, 1.0,
                 gradient->typed_data() + lo * c, exceptional->typed_data() + lo);
        },
        sdca::detail::kWarmLossPoolUs);
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
  return ffi::Error::Success();
}

}  // namespace

// The losses of this build, a bit per SDCA_LOSS_* id.
extern "C" __attribute__((visibility("default"))) uint64_t LibsdcaKinds() {
  return libsdca_frontend::kinds();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(LibsdcaLoss, loss,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int64_t>("kind")
                                  .Attr<int64_t>("k")
                                  .Attr<double>("gamma")
                                  .Attr<double>("temperature")
                                  .Attr<double>("label_smoothing")
                                  .Attr<int64_t>("ignore_index")
                                  .Attr<int64_t>("ignore")
                                  .Attr<int64_t>("precision")
                                  .Attr<int64_t>("reduction"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(LibsdcaTeacherVjp, teacher_vjp,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::U8>>()
                                  .Attr<double>("temperature"));
