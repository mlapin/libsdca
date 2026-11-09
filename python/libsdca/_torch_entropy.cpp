// Optional PyTorch CPU/CUDA extension: ATen SIMD + OpenMP on CPU,
// Triton kernel launch + C++ autograd on CUDA.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <torch/library.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>
#ifdef __linux__
#include <sys/mman.h>
#endif
// Hand-vectorized double rows: 8 lanes in the AVX-512 build, 4 in the AVX2 build.
#if defined(CPU_CAPABILITY_AVX512) && defined(__AVX512F__) && defined(__AVX512DQ__) && \
    defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#define SDCA_SIMD 1
#define SDCA_AVX512 1
#elif defined(CPU_CAPABILITY_AVX2) && defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#define SDCA_SIMD 1
#endif
#include <torch/version.h>
#include <torch/csrc/autograd/functions/utils.h>
// get_current_graph_task_keep_graph: whether a backward keeps its graph (in every PyTorch the
// CUDA kernels build with; older ones keep the saved gradients of the CPU kernels).
#if defined(SDCA_CUDA) || TORCH_VERSION_MAJOR > 2 || TORCH_VERSION_MINOR >= 4
#include <torch/csrc/autograd/graph_task.h>
#define SDCA_KEEP_GRAPH 1
#endif
#ifdef SDCA_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include "_torch_lanes.h"
#include "_torch_ce.h"
#include "_torch_topk_hinge.h"
#include "_torch_prox.h"
#include "_torch_distill.h"
#include "_torch_cuda.h"

namespace {

// The CPU kernels of _cpu.KERNELS with their autograd in C++ (_cpu.apply): no Python between
// forward and backward. Codes of the kinds (_cpu.APPLIED), their ints and floats as _cpu's.
enum CpuKind : int64_t {
  kEntropy,
  kHinge,
  kTopk,
  kMultilabel,
  kDistillationStats,
  kDistillationGrad
};

// The torch layer's gradient of a kind (under create_graph, and float64 entropy's log-domain
// fallback): gradient(kind, logits, target, extra or None, saved, upstream, ints, floats); and
// float64 entropy's log-domain value: value(logits, target, extra or None, reduction,
// ignore_index) (set_gradient).
py::object& torch_gradient() {
  static auto* gradient = new py::object();  // never destroyed: it may outlive the interpreter
  return *gradient;
}
py::object& torch_value() {
  static auto* value = new py::object();
  return *value;
}

void set_gradient(py::object gradient, py::object value) {
  torch_gradient() = std::move(gradient);
  torch_value() = std::move(value);
}

inline py::object optional_object(const std::optional<at::Tensor>& t) {
  return t.has_value() && t->defined() ? py::cast(*t) : py::object(py::none());
}

// The torch layer's gradient of a kind (_cpu.gradient), with the GIL.
at::Tensor python_gradient(int64_t kind, const at::Tensor& z, const at::Tensor& target,
                           const std::optional<at::Tensor>& extra, const at::Tensor& saved,
                           const at::Tensor& upstream, const std::vector<int64_t>& ints,
                           const std::vector<double>& floats) {
  py::gil_scoped_acquire gil;
  TORCH_CHECK(torch_gradient(), "libsdca: no CPU gradient was set");
  return torch_gradient()(kind, z, target, optional_object(extra), saved, upstream, ints, floats)
      .cast<at::Tensor>();
}

// _cpu._native_entropy of entropy64::forward (ints: reduction, ignore_index): the rows, their
// sum or weighted mean, and saved; values below double's normal range or beyond it take the
// torch layer's log-domain value.
std::tuple<at::Tensor, at::Tensor> entropy64_loss(const at::Tensor& z, const at::Tensor& target,
                                                  const std::optional<at::Tensor>& extra,
                                                  const std::vector<int64_t>& ints, bool need) {
  const int64_t reduction = ints[0], ignore = ints[1];
  auto [values, saved] = entropy64::forward(z, target, ignore, extra, need);
  const double* v = values.data_ptr<double>();
  const double* totals = saved.data_ptr<double>() + saved.numel() - 2;
  const int64_t* labels = target.data_ptr<int64_t>();
  bool rare = reduction != 0 && !std::isfinite(totals[0]);
  const double least = std::numeric_limits<double>::min();
  for (int64_t i = 0; i < values.numel() && !rare; ++i)
    rare = !std::isfinite(v[i]) || (v[i] < least && labels[i] != ignore);
  if (rare) {
    py::gil_scoped_acquire gil;
    TORCH_CHECK(torch_value(), "libsdca: no CPU value was set");
    return {torch_value()(z, target, optional_object(extra), reduction, ignore).cast<at::Tensor>(),
            saved};
  }
  if (reduction == 0) return {values, saved};
  return {loss_value(reduction == 2 ? totals[0] : totals[0] / totals[1], z), saved};
}

// Whether a float64 entropy gradient needs the torch layer's log-domain path
// (_cpu._beyond_double): derivatives below double's normal range scaled up (by the upstream
// factor or a class weight), or weight sums beyond its range.
bool entropy64_beyond(const at::Tensor& saved, const at::Tensor& upstream, int64_t reduction,
                      const std::optional<at::Tensor>& extra) {
  const int64_t n = (saved.numel() - 2) / entropy64::kStats;
  const double* st = saved.data_ptr<double>();
  const entropy64::Scale scale(upstream, reduction, saved);
  double largest = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(scale.upstream.size()); ++i) {
    const double a = std::abs(scale[i]);
    largest = std::isnan(a) || std::isnan(largest) ? a + largest : std::max(largest, a);
  }
  if (extra.has_value() && extra->defined()) largest *= extra->abs().max().item<double>();
  bool tiny = false;
  for (int64_t i = 0; i < n && !tiny; ++i) tiny = st[entropy64::kStats * i + 4] != 0;
  const bool finite =
      std::isfinite(st[n * entropy64::kStats]) && std::isfinite(st[n * entropy64::kStats + 1]);
  return (largest > 1 && tiny) || !finite || !std::isfinite(largest);
}

// float64 row values reduced as reduction (0 none, 1 mean, 2 sum), rounded once into the value
// dtype (_cpu._reduced).
at::Tensor reduced_values(at::Tensor values, int64_t reduction, const at::Tensor& z) {
  if (reduction == 1) values = values.mean();
  if (reduction == 2) values = values.sum();
  return values.to(value_type(z));
}

// gradient * upstream[:, None] rounded once into dtype (_cpu.scaled).
at::Tensor scaled(at::Tensor gradient, const at::Tensor& upstream, at::ScalarType dtype,
                  bool inplace) {
  at::Tensor scale = upstream.to(gradient.scalar_type());
  if (scale.dim() == 1) scale = scale.unsqueeze(1);
  return (inplace ? gradient.mul_(scale) : gradient * scale).to(dtype);
}

// The saved distillation gradient (times T) times upstream / T (_cpu._distillation_grad_backward).
at::Tensor distillation_grad_backward(const at::Tensor& z, at::Tensor saved,
                                      const at::Tensor& upstream, int64_t reduction,
                                      double temperature, bool inplace) {
  const int64_t n = z.size(0);
  if (upstream.numel() == 1) {
    const double scalar =
        upstream.item<double>() / static_cast<double>(reduction == 1 ? n : 1) / temperature;
    const double largest = saved.scalar_type() == at::kDouble ? std::numeric_limits<double>::max()
                                                              : std::numeric_limits<float>::max();
    if (std::abs(scalar) <= largest)
      return (inplace ? saved.mul_(scalar) : saved * scalar).to(z.scalar_type());
  }
  at::Tensor rows = upstream.to(at::kDouble);
  if (reduction == 1) rows = rows / n;
  const at::Tensor factor = rows / temperature;
  if (at::isfinite(factor.to(saved.scalar_type())).all().item<bool>())
    return scaled(saved, factor, z.scalar_type(), inplace);
  const at::Tensor inverse = at::scalar_tensor(1 / temperature, saved.options());
  return scaled(inplace ? saved.mul_(inverse) : saved * inverse, rows, z.scalar_type(), true);
}

// A kind's (loss, saved) as _cpu.KERNELS[kind].forward (need: saved for a backward).
std::tuple<at::Tensor, at::Tensor> cpu_forward(int64_t kind, const at::Tensor& z,
                                               const at::Tensor& target,
                                               const std::optional<at::Tensor>& extra,
                                               const std::vector<int64_t>& ints,
                                               const std::vector<double>& floats, bool need) {
  switch (kind) {
    case kEntropy: {
      if (z.scalar_type() == at::kDouble) return entropy64_loss(z, target, extra, ints, need);
      auto [loss, stats, represented] =
          entropy_forward(z, target, ints[0], ints[1], extra, ints[2] != 0, need);
      return {loss, stats};
    }
    case kHinge:
      return hinge_forward(z, target, ints[0], ints[1] != 0, floats[0]);
    case kTopk:
      return topk_forward(z, target, ints[0], ints[1], ints[2], floats[0], ints[3] != 0);
    case kMultilabel:
      return multilabel_forward(z, target, ints[0], ints[1], floats[0], ints[2] != 0, need);
    case kDistillationStats: {
      auto [values, stats] = distillation_forward(z, target, floats[0], ints[1] != 0, need);
      return {reduced_values(values.to(at::kDouble), ints[3], z), stats};
    }
    case kDistillationGrad: {
      auto [values, saved] = distillation_forward_grad(z, target, floats[0]);
      return {reduced_values(values, ints[3], z), saved};
    }
  }
  TORCH_CHECK_VALUE(false, "unknown CPU kernel");
}

// A kind's first-order gradient as _cpu.KERNELS[kind].backward (inplace: saved is not read
// again).
at::Tensor cpu_backward(int64_t kind, const at::Tensor& z, const at::Tensor& target,
                        const std::optional<at::Tensor>& extra, const at::Tensor& saved,
                        const at::Tensor& upstream, const std::vector<int64_t>& ints,
                        const std::vector<double>& floats, bool inplace) {
  switch (kind) {
    case kEntropy: {
      if (z.scalar_type() != at::kDouble)
        return entropy_backward(z, target, saved, upstream, ints[0], ints[1], extra, ints[2] != 0,
                                std::nullopt);
      if (entropy64_beyond(saved, upstream, ints[0], extra))
        return python_gradient(kind, z, target, extra, saved, upstream, ints, floats);
      return entropy64::backward(z, target, saved, upstream, ints[0], ints[1], extra);
    }
    case kHinge:
      return hinge_backward(z, target, saved, upstream, ints[0], ints[1] != 0, floats[0]);
    case kTopk:
      return topk_backward(z, target, saved, upstream, ints[0], ints[1], ints[2], floats[0],
                           ints[3] != 0);
    case kMultilabel:
      return multilabel_backward(z, target, saved, upstream, ints[0], ints[1], floats[0]);
    case kDistillationStats: {
      at::Tensor scale = upstream.to(at::kDouble);
      if (ints[3] == 1) scale = scale / z.size(0);
      return distillation_backward(z, target, saved, scale.expand({z.size(0)}), floats[0],
                                   ints[1] != 0);
    }
    case kDistillationGrad:
      return distillation_grad_backward(z, saved, upstream, ints[3], floats[0], inplace);
  }
  TORCH_CHECK_VALUE(false, "unknown CPU kernel");
}

// Whether the running backward keeps its graph (retain_graph, create_graph or unknown).
inline bool keep_graph() {
#ifdef SDCA_KEEP_GRAPH
  return torch::autograd::get_current_graph_task_keep_graph();
#else
  return true;
#endif
}

// A new backward node of type T in the pointer type of this PyTorch's graph edges: a
// shared_ptr deleted by deleteNode (found by argument-dependent lookup, as newer releases have
// none), or an intrusive_ptr (PyTorch 2.14).
template <typename T>
decltype(torch::autograd::Edge::function) new_node() {
  using Pointer = decltype(torch::autograd::Edge::function);
  if constexpr (std::is_same_v<Pointer, std::shared_ptr<torch::autograd::Node>>) {
    return std::shared_ptr<T>(new T(), [](T* node) { deleteNode(node); });
  } else {
    return c10::make_intrusive<T>();
  }
}

// The backward of cpu_apply: a plain autograd node, as PyTorch's own (the node of a C++
// autograd Function demangles its type name at every backward and keeps the options in
// IValues, which costs more than a small call's gradient).
struct CpuLossBackward : public torch::autograd::Node {
  torch::autograd::SavedVariable z, target, extra, saved;
  int64_t kind = 0;
  std::vector<int64_t> ints;
  std::vector<double> floats;

  std::string name() const override { return "CpuLossBackward"; }
  void release_variables() override {
    z.reset_data();
    target.reset_data();
    extra.reset_data();
    saved.reset_data();
  }
  torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override {
    std::lock_guard<std::mutex> lock(mutex_);
    const at::Tensor upstream = grads[0].defined() ? grads[0] : input_metadata(0).zeros_like();
    const at::Tensor l = z.unpack(), t = target.unpack(), e = extra.unpack(), s = saved.unpack();
    const auto x = e.defined() ? std::optional<at::Tensor>(e) : std::nullopt;
    if (at::GradMode::is_enabled())
      return {python_gradient(kind, l, t, x, s, upstream, ints, floats)};
    return {cpu_backward(kind, l, t, x, s, upstream, ints, floats, !keep_graph())};
  }
};

// A kind's loss, differentiable in z where grad mode records it. Forward-mode gradients raise
// as for a C++ autograd Function.
at::Tensor cpu_apply(int64_t kind, const at::Tensor& z, const at::Tensor& target,
                     const std::optional<at::Tensor>& extra, const std::vector<int64_t>& ints,
                     const std::vector<double>& floats) {
  if (!at::GradMode::is_enabled() || !z.requires_grad())
    return std::get<0>(cpu_forward(kind, z, target, extra, ints, floats, false));
  TORCH_CHECK(!z._fw_grad(/*level=*/0).defined(),
              "jvp is not implemented for the c++ API of custom Function yet.",
              "Please open a feature request on GitHub if you need this.");
  at::Tensor loss, saved;
  {
    at::AutoGradMode no_grad(false);
    std::tie(loss, saved) = cpu_forward(kind, z, target, extra, ints, floats, true);
  }
  auto node = new_node<CpuLossBackward>();
  auto& backward = static_cast<CpuLossBackward&>(*node);
  backward.set_next_edges(torch::autograd::collect_next_edges(z));
  backward.z = torch::autograd::SavedVariable(z, false);
  backward.target = torch::autograd::SavedVariable(target, false);
  backward.extra = torch::autograd::SavedVariable(extra.value_or(at::Tensor()), false);
  backward.saved = torch::autograd::SavedVariable(saved, false);
  backward.kind = kind;
  backward.ints = ints;
  backward.floats = floats;
  torch::autograd::set_history(loss, node);
  return loss;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // CPU kernels: contiguous (N, C) CPU logits of float16/bfloat16/float32/float64; reduction
  // 0 none, 1 mean, 2 sum. Losses are float32 (float64 for float64 logits), (N,) for none and
  // scalars otherwise; gradients have the logits' dtype, scaled by upstream ((N,) for none,
  // a scalar otherwise; any float dtype) before their one rounding. Input errors raise
  // ValueError with the torch layer's messages. Statistics are float64.
  m.def("forward", &entropy_forward,
        "Cross-entropy -> (loss, stats (3 N + 1,) or None, represented); weight (C,) or None",
        py::arg("z"), py::arg("y"), py::arg("reduction"), py::arg("ignore_index"),
        py::arg("weight"), py::arg("fp32"), py::arg("need_stats"));
  // A kind's loss with its autograd in C++ (positional arguments: no keyword lookup).
  m.def("apply", &cpu_apply);
  m.def("set_gradient", &set_gradient);
  m.def("entropy64_forward", &entropy64::forward,
        "float64 entropy rows of at most 32 classes as the native library's -> (values, saved)");
  m.def("entropy64_backward", &entropy64::backward,
        "float64 entropy gradient from entropy64_forward's saved, the upstream gradient and the "
        "weight");
  m.def("backward", &entropy_backward, "Cross-entropy gradient from forward's stats", py::arg("z"),
        py::arg("y"), py::arg("stats"), py::arg("upstream"), py::arg("reduction"),
        py::arg("ignore_index"), py::arg("weight"), py::arg("fp32"), py::arg("out") = py::none());
  m.def("hinge_forward", &hinge_forward, "k = 1 (smooth) hinge -> (loss, stats (N, 2))",
        py::arg("z"), py::arg("y"), py::arg("reduction"), py::arg("smooth"), py::arg("gamma"));
  m.def("hinge_backward", &hinge_backward, "k = 1 hinge gradient", py::arg("z"), py::arg("y"),
        py::arg("stats"), py::arg("upstream"), py::arg("reduction"), py::arg("smooth"),
        py::arg("gamma"));
  m.def("topk_forward", &topk_forward,
        "Top-k loss (loss_code: 0 cross-entropy, 1 alpha, 2 beta, 3 alpha_smooth, "
        "4 beta_smooth hinge) -> (loss, stats (N, 3))",
        py::arg("z"), py::arg("y"), py::arg("reduction"), py::arg("loss_code"), py::arg("k"),
        py::arg("gamma"), py::arg("fp32"));
  m.def("topk_backward", &topk_backward, "Top-k loss gradient", py::arg("z"), py::arg("y"),
        py::arg("stats"), py::arg("upstream"), py::arg("reduction"), py::arg("loss_code"),
        py::arg("k"), py::arg("gamma"), py::arg("fp32"));
  m.def("multilabel_forward", &multilabel_forward,
        "Boolean-mask loss (mask uint8 (N, C); obj_code: 0 cross-entropy, 1 hinge, 2 smooth "
        "hinge) -> (loss, stats (N, 7)); fp32: float32 exponentials in cross-entropy of "
        "non-float64 logits; need: cross-entropy stats for a backward (else partly unset)",
        py::arg("z"), py::arg("mask"), py::arg("reduction"), py::arg("obj_code"), py::arg("gamma"),
        py::arg("fp32") = false, py::arg("need") = true);
  m.def("multilabel_backward", &multilabel_backward, "Boolean-mask loss gradient", py::arg("z"),
        py::arg("mask"), py::arg("stats"), py::arg("upstream"), py::arg("reduction"),
        py::arg("obj_code"), py::arg("gamma"));
  m.def("distillation_forward", &distillation_forward,
        "distillation_loss rows -> (values (N,), stats (N, 7) or None); is_double: float64 work",
        py::arg("student"), py::arg("teacher"), py::arg("temperature"), py::arg("is_double"),
        py::arg("need_stats"));
  m.def("distillation_backward", &distillation_backward,
        "distillation_loss gradient, upstream (N,)", py::arg("student"), py::arg("teacher"),
        py::arg("stats"), py::arg("upstream"), py::arg("temperature"), py::arg("is_double"));
  m.def("distillation_forward_grad", &distillation_forward_grad,
        "distillation_loss rows and their gradient times T of float32 or float64 logits -> "
        "(values (N,) float64, gradient (N, C) in their dtype)",
        py::arg("student"), py::arg("teacher"), py::arg("temperature"));
  m.def("prox_distillation_forward", &prox_distillation_forward,
        "proximal_distillation_loss rows -> (values (N,), gradient (N, C), float32 for 16-bit)",
        py::arg("student"), py::arg("teacher"), py::arg("temperature"), py::arg("alpha"));
  m.def("prox_entropy_forward", &prox_entropy_forward,
        "proximal_cross_entropy rows (target int64 (N,) or bool (N, C)) -> (values (N,), "
        "gradient (N, C), float32 for 16-bit)",
        py::arg("z"), py::arg("target"), py::arg("alpha"), py::arg("fp32_work"));
#ifdef SDCA_CUDA
  m.def("cuda_forward", &cuda_forward);
  m.def("weighted_cuda_forward", &weighted_cuda_forward);
  // Python callers keep their logits: never overwritten.
  m.def("weighted_cuda_backward",
        [](const at::Tensor& z, const at::Tensor& y, const at::Tensor& s, const at::Tensor& u,
           int64_t reduction, const at::Tensor& h,
           int64_t context) { return weighted_cuda_backward(z, y, s, u, reduction, h, context); });
  m.def("weighted_cuda_apply", &weighted_cuda_apply);
  m.def("cuda_backward", [](const at::Tensor& z, const at::Tensor& y, const at::Tensor& s,
                            const at::Tensor& u, int64_t reduction, const at::Tensor& h) {
    return cuda_backward(z, y, s, u, reduction, h);
  });
  m.def("cuda_apply", &cuda_apply);
  m.def("hinge_cuda_apply", &hinge_cuda_apply);
  m.def("distillation_cuda_apply", &distillation_cuda_apply);
  m.def("multilabel_cuda_apply", &multilabel_cuda_apply);
  // Forward (loss, statistics) and backward entries of the torch.library operators (_ops.py);
  // their backward never overwrites the logits (mutates_args=()).
  m.def("hinge_cuda_forward",
        [](const at::Tensor& logits, const at::Tensor& target, const at::Tensor& gamma,
           int64_t reduction, const at::Tensor& handles) {
          return hinge_cuda_forward(logits, target, gamma, reduction, handles);
        });
  m.def("hinge_cuda_backward",
        [](const at::Tensor& z, const at::Tensor& y, const at::Tensor& s, const at::Tensor& u,
           const at::Tensor& gamma,
           const at::Tensor& h) { return hinge_cuda_backward(z, y, s, u, gamma, h, false); });
  m.def("multilabel_cuda_forward",
        [](const at::Tensor& logits, const at::Tensor& mask, const at::Tensor& gamma,
           int64_t obj_id, int64_t reduction, const at::Tensor& handles) {
          return multilabel_cuda_forward(logits, mask, gamma, obj_id, reduction, handles);
        });
  m.def("multilabel_cuda_backward",
        [](const at::Tensor& z, const at::Tensor& mask, const at::Tensor& s, const at::Tensor& u,
           const at::Tensor& gamma, int64_t obj_id, const at::Tensor& h) {
          return multilabel_cuda_backward(z, mask, s, u, gamma, obj_id, h, false);
        });
#endif
}
