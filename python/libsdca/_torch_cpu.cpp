// The CPU kernels of the libsdca::loss and libsdca::loss_ operators (schemas in
// _torch_ops.py): sdca::kernel::evaluate on the tensors' own buffers. And apply, the eager
// autograd of a loss on any device: forward computes the values (on the CPU, with the
// gradient of a sum or mean at an upstream of 1), backward the gradient of other upstreams,
// written over the logits where nothing else can read them.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef LIBSDCA_KEEP_GRAPH
#include <torch/csrc/autograd/graph_task.h>
#endif

#include "_frontend_cpu.h"
#include "_torch_cuda_plan.h"
#include "sdca/utility/row_reduction.h"

namespace {

using libsdca_frontend::Batch;
using libsdca_frontend::Spec;

Spec spec_of(int64_t kind, int64_t k, double gamma, double temperature, double label_smoothing,
             int64_t ignore_index, bool ignore, int64_t precision, double upstream) {
  Spec spec;
  spec.kind = static_cast<int>(kind);
  spec.k = static_cast<size_t>(k);
  spec.gamma = gamma;
  spec.temperature = temperature;
  spec.label_smoothing = label_smoothing;
  spec.ignore_index = ignore_index;
  spec.ignore = ignore;
  spec.precision = static_cast<int>(precision);
  spec.num_threads = static_cast<size_t>(at::get_num_threads());
  spec.upstream = upstream;
  return spec;
}

template <typename T>
const T* data(const at::Tensor& t) {
  return reinterpret_cast<const T*>(t.const_data_ptr());
}

const double* optional_data(const c10::optional<at::Tensor>& t) {
  return t.has_value() && t->numel() > 0 ? t->const_data_ptr<double>() : nullptr;
}

template <typename S>
void evaluate(const Spec& spec, const at::Tensor& scores, const at::Tensor& target,
              const c10::optional<at::Tensor>& weight, const double* row_upstream, double* values,
              S* grads) {
  const Batch<S> b{static_cast<size_t>(scores.size(0)),
                   static_cast<size_t>(scores.size(1)),
                   data<S>(scores),
                   values,
                   grads,
                   row_upstream};
  const double* class_weight = optional_data(weight);
  const auto dtype = target.scalar_type();
  if (dtype == at::kLong) {
    libsdca_frontend::labels(spec, b, target.const_data_ptr<int64_t>(), class_weight);
  } else if (dtype == at::kBool) {
    libsdca_frontend::mask(spec, b, data<uint8_t>(target));
  } else if (spec.kind == libsdca_frontend::kDistillation ||
             spec.kind == libsdca_frontend::kProximalDistillation) {
    libsdca_frontend::teacher(spec, b, data<S>(target));
  } else if (dtype == scores.scalar_type()) {
    libsdca_frontend::probabilities(spec, b, data<S>(target), class_weight);
  } else if (dtype == at::kFloat) {
    libsdca_frontend::probabilities(spec, b, data<float>(target), class_weight);
  } else {
    libsdca_frontend::probabilities(spec, b, data<double>(target), class_weight);
  }
}

void check(const at::Tensor& scores, const at::Tensor& target,
           const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& rows) {
  TORCH_CHECK(scores.dim() == 2 && scores.is_contiguous(), "libsdca: scores must be (N, C)");
  TORCH_CHECK(target.is_contiguous() && target.size(0) == scores.size(0),
              "libsdca: target must be contiguous with a row per score row");
  TORCH_CHECK(
      !weight.has_value() || (weight->scalar_type() == at::kDouble && weight->is_contiguous() &&
                              (weight->numel() == 0 || weight->numel() == scores.size(1))),
      "libsdca: weight must be float64 (C)");
  TORCH_CHECK(!rows.has_value() || (rows->scalar_type() == at::kDouble && rows->is_contiguous() &&
                                    rows->numel() == scores.size(0)),
              "libsdca: row_upstream must be float64 (N)");
}

// The kernels of scores' dtype: values (or null) and the gradient into grads (or null).
void dispatch(const Spec& spec, const at::Tensor& scores, const at::Tensor& target,
              const c10::optional<at::Tensor>& weight, const double* rows, double* values,
              void* grads) {
  switch (scores.scalar_type()) {
    case at::kFloat:
      return evaluate<float>(spec, scores, target, weight, rows, values,
                             static_cast<float*>(grads));
    case at::kDouble:
      return evaluate<double>(spec, scores, target, weight, rows, values,
                              static_cast<double*>(grads));
    case at::kHalf:
      return evaluate<sdca::half>(spec, scores, target, weight, rows, values,
                                  static_cast<sdca::half*>(grads));
    case at::kBFloat16:
      return evaluate<sdca::bfloat16>(spec, scores, target, weight, rows, values,
                                      static_cast<sdca::bfloat16*>(grads));
    default:
      TORCH_CHECK(false, "libsdca: unsupported scores dtype ", scores.scalar_type());
  }
}

// row_upstream as the CPU kernels take it, float64 (N): from one factor per row or a 0-d one
// (any float dtype), over a 0-d divisor.
c10::optional<at::Tensor> row_factors(const at::Tensor& scores,
                                      const c10::optional<at::Tensor>& row_upstream,
                                      const c10::optional<at::Tensor>& divisor) {
  if (!divisor.has_value() && (!row_upstream.has_value() || row_upstream->dim() == 1)) {
    return row_upstream;
  }
  at::Tensor f = row_upstream.has_value() ? row_upstream->to(at::kDouble)
                                          : at::ones({}, scores.options().dtype(at::kDouble));
  if (divisor.has_value()) f = f / divisor->to(at::kDouble);
  return f.expand({scores.size(0)}).contiguous();
}

std::tuple<at::Tensor, at::Tensor> loss(const at::Tensor& scores, const at::Tensor& target,
                                        const c10::optional<at::Tensor>& weight,
                                        const c10::optional<at::Tensor>& upstreams, int64_t kind,
                                        int64_t k, double gamma, double temperature,
                                        double label_smoothing, int64_t ignore_index, bool ignore,
                                        int64_t precision, double upstream, bool values, bool grad,
                                        const c10::optional<at::Tensor>& divisor,
                                        const c10::optional<at::Tensor>& /*stats*/) {
  const auto row_upstream = row_factors(scores, upstreams, divisor);
  check(scores, target, weight, row_upstream);
  const Spec spec = spec_of(kind, k, gamma, temperature, label_smoothing, ignore_index, ignore,
                            precision, upstream);
  at::Tensor out = at::empty({values ? scores.size(0) : 0}, scores.options().dtype(at::kDouble));
  at::Tensor g = grad ? at::empty_like(scores) : at::empty({0}, scores.options());
  dispatch(spec, scores, target, weight, optional_data(row_upstream),
           values ? out.data_ptr<double>() : nullptr, grad ? g.data_ptr() : nullptr);
  return std::make_tuple(out, g);
}

at::Tensor loss_(at::Tensor& scores, const at::Tensor& target,
                 const c10::optional<at::Tensor>& weight,
                 const c10::optional<at::Tensor>& upstreams, int64_t kind, int64_t k, double gamma,
                 double temperature, double label_smoothing, int64_t ignore_index, bool ignore,
                 int64_t precision, double upstream, bool values, bool grad,
                 const c10::optional<at::Tensor>& divisor,
                 const c10::optional<at::Tensor>& /*stats*/) {
  const auto row_upstream = row_factors(scores, upstreams, divisor);
  check(scores, target, weight, row_upstream);
  TORCH_CHECK(grad, "libsdca: loss_ writes the gradient");
  const Spec spec = spec_of(kind, k, gamma, temperature, label_smoothing, ignore_index, ignore,
                            precision, upstream);
  at::Tensor out = at::empty({values ? scores.size(0) : 0}, scores.options().dtype(at::kDouble));
  dispatch(spec, scores, target, weight, optional_data(row_upstream),
           values ? out.data_ptr<double>() : nullptr, scores.data_ptr());
  return out;
}

using LossOp = std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&,
                                                  const c10::optional<at::Tensor>&,
                                                  const c10::optional<at::Tensor>&, int64_t,
                                                  int64_t, double, double, double, int64_t, bool,
                                                  int64_t, double, bool, bool,
                                                  const c10::optional<at::Tensor>&,
                                                  const c10::optional<at::Tensor>&);
using InplaceOp = at::Tensor(at::Tensor&, const at::Tensor&, const c10::optional<at::Tensor>&,
                             const c10::optional<at::Tensor>&, int64_t, int64_t, double, double,
                             double, int64_t, bool, int64_t, double, bool, bool,
                             const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&);
using ForwardOp = std::tuple<at::Tensor, at::Tensor, at::Tensor>(
    const at::Tensor&, const at::Tensor&, const c10::optional<at::Tensor>&, int64_t, int64_t,
    double, double, double, int64_t, bool, int64_t, int64_t, bool);

// The operators through the dispatcher: the CPU kernels above or the CUDA ones of
// triton/launch_torch.py.
const c10::TypedOperatorHandle<LossOp>& loss_op() {
  static const auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("libsdca::loss", "").typed<LossOp>();
  return op;
}

const c10::TypedOperatorHandle<InplaceOp>& inplace_op() {
  static const auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("libsdca::loss_", "").typed<InplaceOp>();
  return op;
}

// The CUDA forward (triton/launch_torch.py): reduction, divisor and saved statistics in the
// kernels, so a call is its Triton launches.
const c10::TypedOperatorHandle<ForwardOp>& forward_op() {
  static const auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("libsdca::forward", "").typed<ForwardOp>();
  return op;
}

// _frontend.FIRST_ORDER, defined by the build (_torch_ops._cpu_build).
constexpr const char* kFirstOrder = LIBSDCA_FIRST_ORDER;
constexpr const char* kReferenced =
    "libsdca: the logits are still referenced, so backward allocates a new gradient instead of "
    "reusing their memory. Delete the references before backward() (e.g. `del logits`), pass "
    "reuse_logits=False, or filter this warning.";

// PyTorch >= 2.10 wrappers (PyObjectSlot::try_incref): the wrapper holds a reference to
// its TensorImpl or StorageImpl, and C++ references hold one to the wrapper.
template <class T, class = void>
struct counted_wrappers : std::false_type {};
template <class T>
struct counted_wrappers<T, std::void_t<decltype(std::declval<const T&>().try_incref())>>
    : std::true_type {};

template <class Impl>
PyObject* python_wrapper(const Impl* impl) {
  return impl->pyobj_slot()->load_pyobj();
}

// Whether an object (TensorImpl or StorageImpl) of `count` references has exactly `refs`
// from C++. A Python wrapper holds one more; its own count is 1 while only C++ keeps it.
bool cpp_only(PyObject* wrapper, size_t count, size_t refs) {
  if (wrapper == nullptr) return count == refs;
  if (count != refs + 1) return false;
  pybind11::gil_scoped_acquire gil;
  if (Py_REFCNT(wrapper) != 1) return false;
  // A constant-initialized pointer has no init guard, which a thread could wait on while the
  // lookup drops the GIL; two threads racing through the lookup leak one reference at worst.
  static PyObject* weakrefs = nullptr;
  if (weakrefs == nullptr) {
    pybind11::object found = pybind11::module_::import("weakref").attr("getweakrefcount");
    weakrefs = found.release().ptr();
  }
  return pybind11::handle(weakrefs)(pybind11::handle(wrapper)).cast<int>() == 0;
}

// Whether this PyTorch lets a backward tell that nothing else can read the saved logits.
template <class Slot = c10::impl::PyObjectSlot>
constexpr bool countable() {
#ifdef LIBSDCA_KEEP_GRAPH
  return counted_wrappers<Slot>::value;
#else
  return false;
#endif
}

// Whether a backward may write its gradient over the saved logits z, which it holds twice
// (saved variable and unpacked copy): the graph is freed afterwards and no other tensor,
// view, storage, Python or foreign owner (e.g. DLPack) can read z.
template <class Slot = c10::impl::PyObjectSlot>
bool dead_logits(const at::Tensor& z) {
#ifdef LIBSDCA_KEEP_GRAPH
  if constexpr (countable<Slot>()) {
    if (at::GradMode::is_enabled() || torch::autograd::get_current_graph_task_keep_graph()) {
      return false;
    }
    const auto& storage = z.storage();
    const auto deleter = c10::GetAllocator(z.device().type())->raw_deleter();
    if (storage.data_ptr().get_deleter() != deleter) return false;
    if (!cpp_only(python_wrapper(z.unsafeGetTensorImpl()), z.use_count(), 2)) return false;
    // A view, e.g. (B, T, V) logits reshaped to (N, V), shares storage only with its base.
    const bool view = z.is_view();
    const auto& base = view ? z._base() : z;
    if (view && !cpp_only(python_wrapper(base.unsafeGetTensorImpl()), base.use_count(), 1)) {
      return false;
    }
    return cpp_only(python_wrapper(storage.unsafeGetStorageImpl()), storage.use_count(),
                    view ? 2 : 1);
  }
#endif
  (void)z;
  return false;
}

// A new node of type T in the pointer type of this PyTorch's graph edges: a shared_ptr
// deleted by deleteNode (found by argument-dependent lookup: newer releases have none), or an
// intrusive_ptr (PyTorch 2.14).
template <typename T>
decltype(torch::autograd::Edge::function) new_node() {
  using Pointer = decltype(torch::autograd::Edge::function);
  if constexpr (std::is_same_v<Pointer, std::shared_ptr<torch::autograd::Node>>) {
    return std::shared_ptr<T>(new T(), [](T* node) { deleteNode(node); });
  } else {
    return c10::make_intrusive<T>();
  }
}

// The node of a gradient computed under create_graph: differentiating it raises.
struct FirstOrderBackward : public torch::autograd::Node {
  std::string name() const override { return "FirstOrderBackward"; }
  torch::autograd::variable_list apply(torch::autograd::variable_list&&) override {
    TORCH_CHECK(false, kFirstOrder);
  }
};

// Whether the upstream gradient of a sum or mean is 1.
bool unit_upstream(const at::Tensor& upstream) {
  return upstream.is_cpu() && upstream.item<double>() == 1.0;
}

struct LossBackward : public torch::autograd::Node {
  torch::autograd::SavedVariable scores, target, weight, total, stats;
  // The gradient of an upstream of 1, from the forward's pass (CPU sums and means), until a
  // backward takes it.
  at::Tensor unit_gradient;
  // CUDA (forward_op): the divisor of a mean of label weights and the row statistics, or
  // undefined; the factors then reach the kernels as they are.
  bool device = false;
  Spec spec;
  int64_t reduction = 0;
  double count = 1;  // the divisor of a mean without total
  bool reuse_logits = true;

  std::string name() const override { return "LossBackward"; }
  void release_variables() override {
    scores.reset_data();
    target.reset_data();
    weight.reset_data();
    total.reset_data();
    stats.reset_data();
    unit_gradient.reset();
  }

  void save_reduction(int64_t mode, double denominator, bool cuda,
                      const at::Tensor& forward_divisor, const at::Tensor& forward_stats) {
    reduction = mode;
    count = denominator;
    device = cuda;
    const bool divided = reduction == 2 && forward_divisor.defined();
    total =
        torch::autograd::SavedVariable(device || divided ? forward_divisor : at::Tensor(), false);
    if (device && forward_stats.defined() && forward_stats.numel() > 0)
      stats = torch::autograd::SavedVariable(forward_stats, false);
  }

  struct Upstream {
    at::Tensor factor;
    double scalar = 1.0;
    c10::optional<at::Tensor> over, saved;
  };

  // Keep scalar CUDA factors on device; CPU kernels use a contiguous vector of doubles.
  Upstream scaled_upstream(const at::Tensor& grad, const at::Tensor& d, const at::Tensor& z) const {
    Upstream u;
    u.factor = grad.detach();
    if (device) {
      u.factor = reduction == 0 ? u.factor.reshape({-1}).contiguous() : u.factor.reshape({});
      if (d.defined()) {
        u.over = d;
      } else if (reduction == 2) {
        u.scalar = 1.0 / count;
      }
      const at::Tensor saved = stats.unpack();
      if (saved.defined()) u.saved = saved;
    } else {
      u.factor = u.factor.to(at::kDouble);
      u.factor = reduction == 0 ? u.factor.reshape({-1}) : u.factor.reshape({}).expand({z.size(0)});
      if (d.defined()) {
        u.factor = u.factor / d;
      } else if (reduction == 2) {
        u.factor = u.factor / count;
      }
      u.factor = u.factor.contiguous();
    }
    return u;
  }

  torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!grads[0].defined()) return {at::Tensor()};
    // Unpacked first: an input changed in place since the forward raises.
    const at::Tensor z = scores.unpack(), t = target.unpack(), w = weight.unpack();
    const bool graph = at::GradMode::is_enabled();
    if (unit_gradient.defined() && !graph && unit_upstream(grads[0])) {
      return {std::exchange(unit_gradient, at::Tensor())};
    }
    unit_gradient.reset();
    const at::Tensor d = total.unpack();
    const Upstream u = scaled_upstream(grads[0], d, z);
    const bool reuse = reuse_logits && !graph && countable();
    const bool dead = reuse && dead_logits(z);
    if (reuse && !dead) {
      TORCH_WARN_ONCE(kReferenced);
    }
    at::Tensor out;
    {
      at::AutoDispatchBelowADInplaceOrView below;
      at::AutoGradMode no_grad(false);
      const auto weights = w.defined() ? c10::optional<at::Tensor>(w) : c10::nullopt;
      const auto k = static_cast<int64_t>(spec.k);
      if (dead) z.unsafeGetTensorImpl()->bump_version();
      auto dispatch = [&] {
        if (dead) {
          out = z.detach();
          inplace_op().call(out, t, weights, u.factor, spec.kind, k, spec.gamma, spec.temperature,
                            spec.label_smoothing, spec.ignore_index, spec.ignore, spec.precision,
                            u.scalar, false, true, u.over, u.saved);
        } else {
          out = std::get<1>(loss_op().call(z.detach(), t, weights, u.factor, spec.kind, k,
                                           spec.gamma, spec.temperature, spec.label_smoothing,
                                           spec.ignore_index, spec.ignore, spec.precision, u.scalar,
                                           false, true, u.over, u.saved));
        }
      };
#ifdef __linux__
      namespace gpu = libsdca_cuda;
      if (device) {
        gpu::Slots slots{
            z, t, w, u.factor, u.over.value_or(at::Tensor()), u.saved.value_or(at::Tensor())};
        if (gpu::execute(slots, spec, 0, false, true, dead, u.scalar, dispatch))
          out = dead ? z.detach() : slots[gpu::G];
      } else
#endif
        dispatch();
    }
    if (graph) {
      auto node = new_node<FirstOrderBackward>();
      node->set_next_edges(torch::autograd::collect_next_edges(z, grads[0]));
      torch::autograd::set_history(out, node);
    }
    return {out};
  }
};

// The divisor of a mean of labels with class weights or an ignored label, by the rule of
// _frontend.mean_divisor (float64 weights; kept here for speed, tested equal): the sum of the
// weights of the labels, clamped into the classes, of the rows not ignored, each times scale.
at::Tensor label_weights(const at::Tensor& target, const c10::optional<at::Tensor>& weight,
                         const Spec& spec, double scale = 1.0) {
  at::Tensor kept;
  if (spec.ignore) kept = target.ne(spec.ignore_index);
  if (!weight.has_value()) {
    at::Tensor count = kept.sum(at::kDouble);
    return scale == 1.0 ? count : count.mul(scale);
  }
  at::Tensor picked = weight->index_select(0, target.clamp(0, weight->size(0) - 1));
  if (kept.defined()) picked = picked.masked_fill(kept.logical_not(), 0);
  return (scale == 1.0 ? picked : picked.mul(scale)).sum();
}

at::Tensor reduce_rows(const at::Tensor& values, const c10::optional<at::Tensor>& terms,
                       at::ScalarType dtype, int64_t reduction) {
  TORCH_CHECK(values.scalar_type() == at::kDouble && values.is_contiguous() &&
                  (!terms || (terms->scalar_type() == at::kDouble && terms->is_contiguous() &&
                              terms->numel() == values.numel())) &&
                  (dtype == at::kDouble || dtype == at::kFloat) &&
                  (reduction == 1 || reduction == 2),
              "libsdca: invalid row reduction");
  TORCH_CHECK(values.dim() == 1 || values.dim() == 2, "libsdca: expected rows or groups of rows");
  const auto shape = values.sizes().slice(0, values.dim() - 1);
  auto out = at::empty(shape, values.options().dtype(dtype));
  const double* w = optional_data(terms);
  const double* v = values.const_data_ptr<double>();
  if (dtype == at::kDouble)
    sdca::detail::store_row_groups(out.data_ptr<double>(), v, w, out.numel(), values.size(-1),
                                   reduction);
  else
    sdca::detail::store_row_groups(out.data_ptr<float>(), v, w, out.numel(), values.size(-1),
                                   reduction);
  return out;
}

// Resolve an implicit label-weight divisor. An unweighted CPU count can stay a scalar.
bool resolve_label_divisor(const at::Tensor& target, const c10::optional<at::Tensor>& weight,
                           const Spec& spec, int64_t reduction, at::Tensor& divisor,
                           double& count) {
  const bool labeled = reduction == 2 && !divisor.defined() && target.scalar_type() == at::kLong &&
                       (weight.has_value() || spec.ignore);
  if (labeled && !weight.has_value() && target.is_cpu() && target.is_contiguous() &&
      target.numel() > 0) {
    const auto* labels = target.const_data_ptr<int64_t>();
    count = static_cast<double>(target.numel() -
                                std::count(labels, labels + target.numel(), spec.ignore_index));
  } else if (labeled) {
    divisor = label_weights(target, weight, spec);
  }
  return labeled;
}

at::Tensor reduce_values(const at::Tensor& values, const at::Tensor& scores,
                         const at::Tensor& target, const c10::optional<at::Tensor>& weight,
                         const Spec& spec, int64_t reduction, const at::Tensor& divisor,
                         bool labeled, double count) {
  // _frontend.value_dtype: float64 for float64 scores or class probabilities, else float32.
  auto dtype = at::promote_types(scores.scalar_type(), at::kFloat);
  if (target.is_floating_point()) dtype = at::promote_types(dtype, target.scalar_type());
  if (reduction == 0) return values.to(dtype);
  if (!values.is_cpu()) {
    at::Tensor result = values.sum();
    if (reduction == 2) result = divisor.defined() ? result / divisor : result / count;
    return result.to(dtype);
  }
  if (reduction == 1 || (!labeled && !divisor.defined()))
    return reduce_rows(values, c10::nullopt, dtype, reduction);
  // Preserve an explicitly supplied divisor. Only class weights require the
  // original per-row terms in order to recover a quotient of overflowing sums.
  at::Tensor terms;
  if (labeled && weight.has_value()) {
    terms = weight->index_select(0, target.clamp(0, weight->size(0) - 1));
    if (spec.ignore) terms.masked_fill_(target.eq(spec.ignore_index), 0);
  }
  if (terms.defined()) return reduce_rows(values, terms, dtype, reduction);
  auto out = at::empty({}, values.options().dtype(dtype));
  const double d = divisor.defined() ? divisor.item<double>() : count;
  const auto n = static_cast<size_t>(values.numel());
  if (dtype == at::kDouble)
    sdca::detail::store_row_values(out.data_ptr<double>(), values.const_data_ptr<double>(), nullptr,
                                   n, reduction, d);
  else
    sdca::detail::store_row_values(out.data_ptr<float>(), values.const_data_ptr<double>(), nullptr,
                                   n, reduction, d);
  return out;
}

// Reduced FP32 label values. Each worker counts ignored labels and sums the values
// while its tile is local. The calling thread combines the partial sums in tile order.
at::Tensor reduced_label_values(const at::Tensor& scores, const at::Tensor& target,
                                const Spec& spec, int64_t reduction) {
  constexpr size_t max_tile = 1024;
  constexpr size_t tile_bytes = 256 * 1024;
  const auto n = static_cast<size_t>(scores.size(0));
  const auto classes = static_cast<size_t>(scores.size(1));
  // Bound both score traffic per task and the worker's row-value scratch.
  // Tile boundaries also fix the summation order for every thread count.
  const size_t tile = std::min(max_tile, std::max<size_t>(1, tile_bytes / sizeof(float) / classes));
  const auto chunks = (n + tile - 1) / tile;
  const auto* z = scores.const_data_ptr<float>();
  const auto* y = target.const_data_ptr<int64_t>();
  std::vector<double> partial(chunks);
  const bool count_ignored = spec.ignore && reduction == 2;
  std::vector<size_t> counts(count_ignored ? chunks : 0);
  Spec serial = spec;
  serial.num_threads = 1;
  sdca::detail::parallel_rows(
      chunks, tile * classes, spec.num_threads,
      [&](size_t first, size_t last) {
        const sdca::detail::subnormal_environment subnormals;
        double values[max_tile];
        for (size_t i = first; i < last; ++i) {
          const size_t start = i * tile;
          const size_t rows = std::min(tile, n - start);
          const Batch<float> batch{rows, classes, z + start * classes, values, nullptr, nullptr};
          libsdca_frontend::labels(serial, batch, y + start, nullptr);
          partial[i] = sdca::detail::row_sum<false>(values, rows, 1);
          if (count_ignored) counts[i] = std::count(y + start, y + start + rows, spec.ignore_index);
        }
      },
      sdca::detail::kWarmLossPoolUs);
  const sdca::detail::subnormal_environment subnormals;
  double sum = sdca::detail::row_sum<false>(partial.data(), chunks, 1);
  if (reduction == 2) {
    size_t count = n;
    for (size_t ignored : counts) count -= ignored;
    sum /= static_cast<double>(count);
  }
  auto out = at::empty({}, scores.options());
  *out.data_ptr<float>() = static_cast<float>(sum);
  return out;
}

// The loss of (N, C) scores, reduced (0 none, 1 sum, 2 mean: the sum divided by total, or
// without total by the label weights of label_weights or else the number of rows),
// differentiable in the scores where grad mode records it (no forward mode). On the CPU the
// forward of a sum or mean computes the gradient of an upstream of 1 in the same pass, which
// backward returns for that upstream: the row kernels' cost is in their arithmetic there.
// Other backward calls compute the gradient of their upstream, written over scores computed
// by the graph where reuse_logits and nothing else can read them, and warn once where
// reuse_logits but something can.
at::Tensor apply(const at::Tensor& scores, const at::Tensor& target,
                 const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& total,
                 int64_t kind, int64_t k, double gamma, double temperature, double label_smoothing,
                 int64_t ignore_index, bool ignore, int64_t precision, int64_t reduction,
                 bool reuse_logits) {
  TORCH_CHECK(!scores._fw_grad(/*level=*/0).defined(),
              "libsdca: no forward-mode derivatives (jvp); use backend='composed'");
  const Spec spec =
      spec_of(kind, k, gamma, temperature, label_smoothing, ignore_index, ignore, precision, 1.0);
  at::Tensor loss, divisor = total.value_or(at::Tensor()), stats, unit_gradient;
  double count = static_cast<double>(scores.size(0));
  const bool record = at::GradMode::is_enabled() && scores.requires_grad();
  const bool device = scores.is_cuda() && !total.has_value();
  if (!record && reduction != 0 && scores.is_cpu() && scores.scalar_type() == at::kFloat &&
      spec.kind == sdca::kernel::cross_entropy && spec.label_smoothing == 0 && target.is_cpu() &&
      target.scalar_type() == at::kLong && target.dim() == 1 && scores.dim() == 2 &&
      scores.size(0) == target.size(0) && scores.size(0) > 0 && scores.size(1) > 0 &&
      scores.is_contiguous() && target.is_contiguous() && !weight.has_value() && !total.has_value())
    return reduced_label_values(scores, target, spec, reduction);
  if (device) {
    at::AutoDispatchBelowADInplaceOrView below;
    at::AutoGradMode no_grad(false);
    auto dispatch = [&] {
      auto out =
          forward_op().call(scores, target, weight, kind, k, gamma, temperature, label_smoothing,
                            ignore_index, ignore, precision, reduction, record);
      loss = std::get<0>(out);
      if (std::get<1>(out).numel() > 0) divisor = std::get<1>(out);
      stats = std::get<2>(out);
    };
#ifdef __linux__
    namespace gpu = libsdca_cuda;
    gpu::Slots slots{scores, target, weight.value_or(at::Tensor())};
    if (gpu::execute(slots, spec, reduction, record, false, false, 1, dispatch)) {
      loss = slots[reduction ? gpu::L : gpu::V];
      divisor = slots[gpu::D];
      stats = slots[gpu::S];
    }
#else
    dispatch();
#endif
  } else {
    at::AutoDispatchBelowADInplaceOrView below;
    at::AutoGradMode no_grad(false);
    const bool labeled = resolve_label_divisor(target, weight, spec, reduction, divisor, count);
    // The factor of each row's gradient at an upstream of 1, as backward forms it.
    const bool fused = record && reduction != 0 && scores.is_cpu();
    double unit = 1.0;
    if (fused && reduction == 2) unit /= divisor.defined() ? divisor.item<double>() : count;
    at::Tensor values;
    std::tie(values, unit_gradient) = loss_op().call(
        scores, target, weight, c10::nullopt, kind, k, gamma, temperature, label_smoothing,
        ignore_index, ignore, precision, unit, true, fused, c10::nullopt, c10::nullopt);
    if (!fused) unit_gradient = at::Tensor();
    loss = reduce_values(values, scores, target, weight, spec, reduction, divisor, labeled, count);
  }
  if (!record) return loss;
  auto node = new_node<LossBackward>();
  auto& backward = static_cast<LossBackward&>(*node);
  backward.set_next_edges(torch::autograd::collect_next_edges(scores));
  backward.scores = torch::autograd::SavedVariable(scores, false);
  backward.target = torch::autograd::SavedVariable(target, false);
  backward.weight = torch::autograd::SavedVariable(weight.value_or(at::Tensor()), false);
  backward.save_reduction(reduction, count, device, divisor, stats);
  backward.spec = spec;
  backward.unit_gradient = std::move(unit_gradient);
  // A leaf, or a view of one, holds the caller's own data: never overwritten, never warned of.
  const bool leaf = scores.is_view() ? scores._base().is_leaf() : scores.is_leaf();
  backward.reuse_logits = reuse_logits && !leaf;
  torch::autograd::set_history(loss, node);
  return loss;
}

std::pair<at::Tensor, std::vector<int64_t>> teacher_backward(const at::Tensor& z,
                                                             const at::Tensor& t,
                                                             const at::Tensor& upstream,
                                                             double scale, double temperature) {
  TORCH_CHECK(z.device().is_cpu() && t.device().is_cpu() && upstream.device().is_cpu() &&
                  z.scalar_type() == at::kFloat && t.scalar_type() == at::kFloat &&
                  upstream.scalar_type() == at::kFloat && z.dim() == 2 && t.sizes() == z.sizes() &&
                  z.is_contiguous() && t.is_contiguous() && z.size(1) > 0 &&
                  (upstream.numel() == 1 || upstream.numel() == z.size(0)),
              "invalid teacher VJP");
  const int64_t n = z.size(0), c = z.size(1);
  auto grad = at::empty_like(t);
  const auto u = upstream.contiguous();
  const auto *scores = data<float>(z), *teachers = data<float>(t), *factors = data<float>(u);
  auto* out = grad.mutable_data_ptr<float>();
  const int64_t stride = u.numel() == 1 ? 0 : 1;
  std::vector<unsigned char> exceptional(n, 0);
  const auto kernel = libsdca_frontend::teacher_kernel();
  const int threads = at::in_parallel_region() ? 1 : at::get_num_threads();
  sdca::detail::parallel_rows(n, c, threads, [&](int64_t first, int64_t last) {
    const sdca::detail::subnormal_environment subnormals;
    kernel(scores + first * c, teachers + first * c, last - first, c, 1.0 / temperature,
           factors + first * stride, stride, scale, out + first * c, exceptional.data() + first);
  });
  std::vector<int64_t> rows;
  for (int64_t i = 0; i < n; ++i)
    if (exceptional[i]) rows.push_back(i);
  return {grad, rows};
}

// Keep only validated CPU metadata for a particular loss. An undefined result returns a
// different geometry or differentiable target to the Python frontend for validation.
pybind11::cpp_function bind_matrix(int64_t kind, int64_t k, double gamma, double temperature,
                                   double smoothing, int64_t ignore_index, bool ignore,
                                   int64_t precision, int64_t reduction, bool reuse) {
  using Shape = std::tuple<int64_t, int64_t, c10::ScalarType, c10::ScalarType, bool>;
  auto known = std::make_shared<std::set<Shape>>();
  return pybind11::cpp_function(
      [=](const at::Tensor& scores, const at::Tensor& target, bool remember) {
        if (!scores.is_cpu() || !target.is_cpu() || target.requires_grad() || scores.dim() != 2 ||
            (target.dim() != 1 && target.dim() != 2) || target.size(0) != scores.size(0) ||
            (target.dim() == 2 && target.size(1) != scores.size(1)))
          return at::Tensor();
        const Shape shape{scores.size(0), scores.size(1), scores.scalar_type(),
                          target.scalar_type(), target.dim() == 2};
        if (remember) {
          known->insert(shape);
          return at::Tensor();
        }
        if (known->count(shape) == 0) return at::Tensor();
        return apply(scores.contiguous(), target.contiguous(), c10::nullopt, c10::nullopt, kind, k,
                     gamma, temperature, smoothing, ignore_index, ignore, precision, reduction,
                     reuse);
      },
      pybind11::arg("scores"), pybind11::arg("target"), pybind11::arg("remember") = false);
}

}  // namespace

TORCH_LIBRARY_IMPL(libsdca, CPU, m) {
  m.impl("reduce_rows", &reduce_rows);
  m.impl("loss", &loss);
  m.impl("loss_", &loss_);
}

PYBIND11_MODULE(_torch_cpu, m) {
#ifdef __linux__
  m.def("cuda_plan_requested", [] { return libsdca_cuda::requested != nullptr; });
  m.def("cuda_install_plan", &libsdca_cuda::install);
#endif
  m.def("apply", &apply);
  m.def("bind_matrix", &bind_matrix);
  m.def("teacher_backward", &teacher_backward);
  m.def("kinds", &libsdca_frontend::kinds);
}
