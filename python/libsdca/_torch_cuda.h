#ifndef SDCA_TORCH_CUDA_H
#define SDCA_TORCH_CUDA_H

namespace {

#ifdef SDCA_CUDA
inline void save_context(torch::autograd::AutogradContext* ctx,
                         torch::autograd::variable_list saved) {
  if (ctx != nullptr) {
    CUcontext ctx_handle = nullptr;
    TORCH_CHECK(cuCtxGetCurrent(&ctx_handle) == CUDA_SUCCESS, "cuCtxGetCurrent failed");
    // Save the inputs, not aligned copies: double backward needs their history,
    // and the version check catches an in-place target change.
    ctx->save_for_backward(std::move(saved));
    ctx->saved_data["cu_ctx"] = reinterpret_cast<int64_t>(ctx_handle);
  }
}

// A Triton kernel compiled for the launch below: its CUfunction, warps and shared bytes.
struct Kernel {
  uint64_t function;
  int warps;
  int shared;
};

// Handles from _cuda_launch.handles (a CPU int64 tensor): forward, block partials,
// multi-block finish, one-launch finish and backward kernels, then rows per partial
// block, maximum rows of the one-launch reduction and (when present) the backward
// column block.
struct Kernels {
  Kernel fwd, part, finish, direct, bwd;
  int64_t block, direct_rows, b_bwd, fwd_rows, bwd_rows;
};

Kernels unpack(const at::Tensor& handles) {
  TORCH_CHECK(handles.device().is_cpu() && handles.scalar_type() == at::kLong &&
                  handles.is_contiguous() &&
                  (handles.numel() == 17 || handles.numel() == 18 || handles.numel() == 20),
              "unexpected Triton kernel handles");
  const int64_t* h = handles.data_ptr<int64_t>();
  auto kernel = [&](int i) {
    return Kernel{static_cast<uint64_t>(h[3 * i]), static_cast<int>(h[3 * i + 1]),
                  static_cast<int>(h[3 * i + 2])};
  };
  return {kernel(0),
          kernel(1),
          kernel(2),
          kernel(3),
          kernel(4),
          h[15],
          h[16],
          handles.numel() >= 18 ? h[17] : 0,
          handles.numel() == 20 ? h[18] : 1,
          handles.numel() == 20 ? h[19] : 1};
}

// args holds the kernel's own arguments, then two null pointers for Triton's global
// and profile scratch (Triton >= 3.3); cuLaunchKernel reads only the declared ones.
inline void launch_triton(cudaStream_t stream, const Kernel& k, int64_t gx, int64_t gy,
                          void** args) {
  CUresult err = cuLaunchKernel(reinterpret_cast<CUfunction>(k.function), static_cast<unsigned>(gx),
                                static_cast<unsigned>(gy), 1, k.warps * 32, 1, 1, k.shared, stream,
                                args, nullptr);
  TORCH_CHECK(err == CUDA_SUCCESS, "cuLaunchKernel failed with code ", static_cast<int>(err));
}

// The kernels take the row count as int32; _cuda_launch.ROWS bounds it.
inline int32_t rows(const at::Tensor& logits) {
  TORCH_CHECK(logits.size(0) <= (int64_t{1} << 28), "the fused kernels take at most 2**28 rows");
  return static_cast<int32_t>(logits.size(0));
}

// Row statistics, the status word at status * n, then four partials per reduction block.
inline int64_t stats_size(int64_t n, int64_t status, const Kernels& k) {
  return status * n + 1 + (n > k.direct_rows ? 4 * ((n + k.block - 1) / k.block) : 0);
}

// As _triton_reduce.launch: the status word and the reduced or copied row values.
void reduce_rows(cudaStream_t stream, const Kernels& k, void* s_ptr, void* o_ptr, int32_t n) {
  void* zero_ptr = nullptr;
  void* args[] = {&s_ptr, &o_ptr, &n, &zero_ptr, &zero_ptr};
  if (n > k.direct_rows) {
    launch_triton(stream, k.part, (n + k.block - 1) / k.block, 1, args);
    launch_triton(stream, k.finish, 1, 1, args);
  } else {
    launch_triton(stream, k.direct, 1, 1, args);
  }
}

inline at::Tensor aligned16(const at::Tensor& t) {
  return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0 ? t : t.clone();
}

// PyTorch >= 2.10 wrappers (PyObjectSlot::try_incref): the wrapper holds a reference to
// its TensorImpl or StorageImpl, and C++ references hold one to the wrapper.
template <class T, class = void>
struct counted_wrappers : std::false_type {};
template <class T>
struct counted_wrappers<T, std::void_t<decltype(std::declval<const T&>().try_incref())>>
    : std::true_type {};

// Whether an object (TensorImpl or StorageImpl) of `count` references has exactly `refs`
// from C++. A Python wrapper holds one more; its own count is 1 while only C++ keeps it.
bool cpp_only(PyObject* wrapper, size_t count, size_t refs) {
  if (wrapper == nullptr) return count == refs;
  if (!counted_wrappers<c10::impl::PyObjectSlot>::value || count != refs + 1) return false;
  pybind11::gil_scoped_acquire gil;
  const auto weakrefs = pybind11::module_::import("weakref").attr("getweakrefcount");
  return Py_REFCNT(wrapper) == 1 && weakrefs(pybind11::handle(wrapper)).cast<int>() == 0;
}

// Whether a first-order backward may write its gradient over the saved logits z, which
// it holds twice (saved variable and unpacked copy): the graph is freed afterwards and no
// other tensor, view, storage, Python or foreign owner (e.g. DLPack) can read z.
bool dead_logits(const at::Tensor& z) {
  if (at::GradMode::is_enabled() || torch::autograd::get_current_graph_task_keep_graph()) {
    return false;
  }
  const auto& storage = z.storage();
  if (storage.data_ptr().get_deleter() != c10::GetAllocator(at::kCUDA)->raw_deleter()) {
    return false;
  }
  const auto wrapper = [](const auto* impl) { return impl->pyobj_slot()->load_pyobj(); };
  if (!cpp_only(wrapper(z.unsafeGetTensorImpl()), z.use_count(), 2)) return false;
  // A view, e.g. (B, T, V) logits reshaped to (N, V), shares storage only with its base.
  const bool view = z.is_view();
  if (view && !cpp_only(wrapper(z._base().unsafeGetTensorImpl()), z._base().use_count(), 1)) {
    return false;
  }
  return cpp_only(wrapper(storage.unsafeGetStorageImpl()), storage.use_count(), view ? 2 : 1);
}

// The gradient buffer: the dead logits themselves (detached), or a new tensor.
inline at::Tensor gradient_buffer(const at::Tensor& logits, bool overwrite) {
  if (!overwrite) return at::empty_like(logits);
  logits.unsafeGetTensorImpl()->bump_version();
  return logits.detach();
}

std::tuple<at::Tensor, at::Tensor> cuda_forward_impl(
    const at::Tensor& logits_in, const at::Tensor& target_in, int64_t reduction,
    const at::Tensor& handles, torch::autograd::AutogradContext* ctx = nullptr) {
  const Kernels k = unpack(handles);
  c10::cuda::CUDAGuard device_guard(logits_in.device());
  const auto logits = aligned16(logits_in);
  const auto target = aligned16(target_in);
  int32_t n = rows(logits);
  auto opts = logits.options();
  auto stats = at::empty({stats_size(n, 6, k)}, opts.dtype(at::kDouble));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.device().index()).stream();
  void* z_ptr = logits.data_ptr();
  void* y_ptr = target.data_ptr();
  void* s_ptr = stats.data_ptr();
  void* zero_ptr = nullptr;
  void* fwd_args[] = {&z_ptr, &y_ptr, &s_ptr, &n, &zero_ptr, &zero_ptr};
  launch_triton(stream, k.fwd, (n + k.fwd_rows - 1) / k.fwd_rows, 1, fwd_args);
  auto dtype = (logits.scalar_type() == at::kHalf || logits.scalar_type() == at::kBFloat16)
                   ? at::kFloat
                   : logits.scalar_type();
  auto out = at::empty(reduction == 0 ? at::IntArrayRef{logits.size(0)} : at::IntArrayRef{},
                       opts.dtype(dtype));
  reduce_rows(stream, k, s_ptr, out.data_ptr(), n);
  if (ctx != nullptr) ctx->saved_data["reduction"] = reduction;
  save_context(ctx, {logits_in, target_in, stats, handles});
  return {out, stats};
}

// Bounded weighted entropy uses the same reduction/gradient ABI as entropy, but
// includes class weights in forward and one extra saved denominator.
std::tuple<at::Tensor, at::Tensor, int64_t> weighted_cuda_forward(const at::Tensor& logits_in,
                                                                  const at::Tensor& target_in,
                                                                  const at::Tensor& weight_in,
                                                                  int64_t reduction,
                                                                  const at::Tensor& handles) {
  const Kernels k = unpack(handles);
  c10::cuda::CUDAGuard device_guard(logits_in.device());
  const auto logits = aligned16(logits_in);
  const auto target = aligned16(target_in);
  const auto weight = aligned16(weight_in);
  int32_t n = rows(logits);
  const bool full64 = logits.scalar_type() == at::kDouble;
  auto stats = at::empty({int64_t{full64 ? 6 : 5} * n + 2 + (full64 ? 1 : 0) +
                          (n > k.direct_rows ? 3 * ((n + k.block - 1) / k.block) : 0)},
                         logits.options().dtype(at::kDouble));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.device().index()).stream();
  void* z_ptr = logits.data_ptr();
  void* y_ptr = target.data_ptr();
  void* w_ptr = weight.data_ptr();
  void* s_ptr = stats.data_ptr();
  void* zero_ptr = nullptr;
  void* fwd_args[] = {&z_ptr, &y_ptr, &w_ptr, &s_ptr, &n, &zero_ptr, &zero_ptr};
  launch_triton(stream, k.fwd, (n + k.fwd_rows - 1) / k.fwd_rows, 1, fwd_args);
  auto out = at::empty(reduction == 0 ? at::IntArrayRef{logits.size(0)} : at::IntArrayRef{},
                       logits.options().dtype(full64 ? at::kDouble : at::kFloat));
  reduce_rows(stream, k, s_ptr, out.data_ptr(), n);
  CUcontext ctx_handle = nullptr;
  TORCH_CHECK(cuCtxGetCurrent(&ctx_handle) == CUDA_SUCCESS, "cuCtxGetCurrent failed");
  return {out, stats, reinterpret_cast<int64_t>(ctx_handle)};
}

std::tuple<at::Tensor, at::Tensor> cuda_forward(const at::Tensor& logits_in,
                                                const at::Tensor& target_in, int64_t reduction,
                                                const at::Tensor& handles) {
  return cuda_forward_impl(logits_in, target_in, reduction, handles);
}

// overwrite: write the gradient over logits_in (dead_logits); the kernels take the target
// entry from stats, not Z[row, y], and each program reads its columns before writing them.
at::Tensor cuda_backward(const at::Tensor& logits_in, const at::Tensor& target_in,
                         const at::Tensor& stats, const at::Tensor& upstream, int64_t reduction,
                         const at::Tensor& handles, bool overwrite = false) {
  (void)reduction;
  const Kernels k = unpack(handles);
  c10::cuda::CUDAGuard device_guard(logits_in.device());
  const auto logits = aligned16(logits_in);
  const auto target = aligned16(target_in);
  auto u_dtype = logits.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  auto u = (upstream.is_contiguous() && upstream.scalar_type() == u_dtype &&
            (reinterpret_cast<uintptr_t>(upstream.data_ptr()) & 15) == 0)
               ? upstream
               : upstream.to(u_dtype, false, true);
  auto grad = gradient_buffer(logits, overwrite);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.device().index()).stream();
  void* z_ptr = logits.data_ptr();
  void* y_ptr = target.data_ptr();
  void* s_ptr = stats.data_ptr();
  void* u_ptr = u.data_ptr();
  void* g_ptr = grad.data_ptr();
  void* zero_ptr = nullptr;
  int32_t n = rows(logits);
  void* bwd_args[] = {&z_ptr, &y_ptr, &s_ptr, &u_ptr, &g_ptr, &n, &zero_ptr, &zero_ptr};
  launch_triton(stream, k.bwd, (n + k.bwd_rows - 1) / k.bwd_rows,
                (logits.size(1) + k.b_bwd - 1) / k.b_bwd, bwd_args);
  return grad;
}

// Python autograd can call backward on a worker without a current driver context: the
// forward's context for the scope of a backward, then the caller's again.
struct ContextGuard {
  CUcontext previous = nullptr;
  explicit ContextGuard(int64_t context) {
    TORCH_CHECK(cuCtxGetCurrent(&previous) == CUDA_SUCCESS, "cuCtxGetCurrent failed");
    TORCH_CHECK(cuCtxSetCurrent(reinterpret_cast<CUcontext>(context)) == CUDA_SUCCESS,
                "cuCtxSetCurrent failed");
  }
  ContextGuard(const ContextGuard&) = delete;
  ContextGuard& operator=(const ContextGuard&) = delete;
  ~ContextGuard() { (void)cuCtxSetCurrent(previous); }
};

at::Tensor weighted_cuda_backward(const at::Tensor& logits, const at::Tensor& target,
                                  const at::Tensor& stats, const at::Tensor& upstream,
                                  int64_t reduction, const at::Tensor& handles,
                                  int64_t cuda_context, bool overwrite = false) {
  const ContextGuard guard(cuda_context);
  return cuda_backward(logits, target, stats, upstream, reduction, handles, overwrite);
}

struct CudaCallConfig {
  const at::Tensor* target;
  int64_t reduction;
  const at::Tensor* handles;
};

thread_local CudaCallConfig tls_cuda_cfg;

struct CudaEntropyFunction : public torch::autograd::Function<CudaEntropyFunction> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& logits_in) {
    const auto& cfg = tls_cuda_cfg;
    return std::get<0>(cuda_forward_impl(logits_in, *cfg.target, cfg.reduction, *cfg.handles, ctx));
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const int64_t reduction = ctx->saved_data["reduction"].toInt();
    if (at::GradMode::is_enabled()) {
      const auto n = saved[0].size(0);
      const auto invalid = saved[2].slice(0, 5 * n, 6 * n) != 0;
      return {differentiable_backward(saved[0], saved[1], grad_outputs[0], reduction, invalid)};
    }
    const ContextGuard guard(ctx->saved_data["cu_ctx"].toInt());
    return {cuda_backward(saved[0], saved[1], saved[2], grad_outputs[0], reduction, saved[3],
                          dead_logits(saved[0]))};
  }
};

at::Tensor cuda_apply(const at::Tensor& logits, const at::Tensor& target, int64_t reduction,
                      const at::Tensor& handles) {
  if (!at::GradMode::is_enabled() || !logits.requires_grad()) {
    return std::get<0>(cuda_forward(logits, target, reduction, handles));
  }
  tls_cuda_cfg = {&target, reduction, &handles};
  return CudaEntropyFunction::apply(logits);
}

struct WeightedCudaCallConfig {
  const at::Tensor* target;
  const at::Tensor* weight;
  int64_t reduction;
  const at::Tensor* handles;
  bool has_weight;
  int64_t ignore_index;
};

thread_local WeightedCudaCallConfig tls_weighted_cuda_cfg;

at::Tensor weighted_differentiable_backward(const at::Tensor& logits, const at::Tensor& target,
                                            const at::Tensor& weight_in, bool has_weight,
                                            int64_t ignore_index, const at::Tensor& upstream_in,
                                            int64_t reduction, const at::Tensor& stats) {
  const auto ignored = target == ignore_index;
  const auto labels = target.masked_fill(ignored, 0);
  const auto c = logits.size(1);
  const auto safe = labels.clamp(0, c - 1);
  auto weight = has_weight ? weight_in.to(at::kDouble).index({safe})
                           : at::ones_like(labels, logits.options().dtype(at::kDouble));
  weight = weight.masked_fill(ignored, 0.0);
  // Forward encoded row errors in negative weights and the global status at
  // 5*N. Reuse them without another input-sized validation pass.
  const auto n = logits.size(0);
  const auto status = stats.select(0, 5 * n);
  const auto invalid =
      reduction == 0 ? (stats.slice(0, 3, 4 * n, 4) < 0) | (status == 3) : (status > 0).expand({n});
  if (logits.scalar_type() != at::kDouble && upstream_in.scalar_type() != at::kDouble) {
    const auto factor = reduction == 1 ? weight / weight.sum() : weight;
    const auto scale =
        (upstream_in.to(at::kDouble).expand_as(weight) * factor).masked_fill(weight == 0, 0.0);
    auto gradient = differentiable_backward(logits, labels, scale, 0, invalid, true);
    if (reduction == 1) {
      const auto undefined = (~ignored) & (stats.select(0, 5 * n + 1) == 0);
      gradient =
          gradient.masked_fill(undefined.unsqueeze(1), std::numeric_limits<double>::quiet_NaN());
    }
    return gradient;
  }
  const auto z = logits.to(at::kDouble);
  const auto idx = safe.unsqueeze(1);
  const auto label = z.gather(1, idx).squeeze(1);
  const auto neg_inf = -std::numeric_limits<double>::infinity();
  const auto masked = z.scatter(1, idx, neg_inf);
  const auto maximum = std::get<0>(masked.max(1));
  const auto log_sum = (masked - maximum.unsqueeze(1)).exp().sum(1).log();
  const auto relative = (maximum - label) + log_sum;
  const auto log_mass = -at::logaddexp(at::zeros_like(relative), -relative);
  const auto logs = (z - maximum.unsqueeze(1)) + (log_mass - log_sum).unsqueeze(1);
  const auto log_grad = logs.scatter(1, idx, log_mass.unsqueeze(1));
  auto log_weight = weight.log();
  if (reduction == 1) {
    const auto denom = at::logsumexp(log_weight, 0);
    log_weight = log_weight - at::where(at::isfinite(denom), denom, 0.0);
  }
  const auto upstream = upstream_in.to(at::kDouble).expand_as(weight);
  const auto root = ((log_grad + log_weight.unsqueeze(1)) * 0.5).exp();
  auto grad = root * (root * upstream.unsqueeze(1));
  grad = grad.scatter(1, idx, -grad.gather(1, idx));
  grad = grad.masked_fill((weight == 0).unsqueeze(1), 0.0);
  if (reduction == 1) {
    const auto undefined = (~ignored) & (stats.select(0, 5 * n + 1) == 0);
    grad = grad.masked_fill(undefined.unsqueeze(1), std::numeric_limits<double>::quiet_NaN());
  }
  return grad.masked_fill(invalid.unsqueeze(1), std::numeric_limits<double>::quiet_NaN())
      .to(logits.scalar_type());
}

struct CudaWeightedEntropyFunction : public torch::autograd::Function<CudaWeightedEntropyFunction> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& logits_in) {
    const auto& cfg = tls_weighted_cuda_cfg;
    const auto& w = cfg.has_weight ? *cfg.weight : *cfg.target;
    auto [out, stats, cu_ctx] =
        weighted_cuda_forward(logits_in, *cfg.target, w, cfg.reduction, *cfg.handles);
    ctx->save_for_backward({logits_in, *cfg.target, w, stats, *cfg.handles});
    ctx->saved_data["reduction"] = cfg.reduction;
    ctx->saved_data["cu_ctx"] = cu_ctx;
    ctx->saved_data["has_weight"] = cfg.has_weight;
    ctx->saved_data["ignore_index"] = cfg.ignore_index;
    return out;
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const int64_t reduction = ctx->saved_data["reduction"].toInt();
    if (at::GradMode::is_enabled()) {
      return {weighted_differentiable_backward(
          saved[0], saved[1], saved[2], ctx->saved_data["has_weight"].toBool(),
          ctx->saved_data["ignore_index"].toInt(), grad_outputs[0], reduction, saved[3])};
    }
    return {weighted_cuda_backward(saved[0], saved[1], saved[3], grad_outputs[0], reduction,
                                   saved[4], ctx->saved_data["cu_ctx"].toInt(),
                                   dead_logits(saved[0]))};
  }
};

at::Tensor weighted_cuda_apply(const at::Tensor& logits, const at::Tensor& target,
                               const at::Tensor& weight, bool has_weight, int64_t ignore_index,
                               int64_t reduction, const at::Tensor& handles) {
  if (!at::GradMode::is_enabled() || !logits.requires_grad()) {
    return std::get<0>(
        weighted_cuda_forward(logits, target, has_weight ? weight : target, reduction, handles));
  }
  tls_weighted_cuda_cfg = {&target, &weight, reduction, &handles, has_weight, ignore_index};
  return CudaWeightedEntropyFunction::apply(logits);
}

// k = 1 hinge losses (_triton_hinge.py): Triton forward, reduce and backward
// launched from C++.
std::tuple<at::Tensor, at::Tensor> hinge_cuda_forward(
    const at::Tensor& logits_in, const at::Tensor& target_in, const at::Tensor& gamma,
    int64_t reduction, const at::Tensor& handles, torch::autograd::AutogradContext* ctx = nullptr) {
  const Kernels k = unpack(handles);
  c10::cuda::CUDAGuard device_guard(logits_in.device());
  const auto logits = aligned16(logits_in);
  const auto target = aligned16(target_in);
  int32_t n = rows(logits);
  auto stats = at::empty({stats_size(n, 4, k)}, logits.options().dtype(at::kDouble));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.device().index()).stream();
  void* z_ptr = logits.data_ptr();
  void* y_ptr = target.data_ptr();
  void* s_ptr = stats.data_ptr();
  void* g_ptr = gamma.data_ptr();
  void* zero_ptr = nullptr;
  void* fwd_args[] = {&z_ptr, &y_ptr, &s_ptr, &g_ptr, &n, &zero_ptr, &zero_ptr};
  launch_triton(stream, k.fwd, (n + k.fwd_rows - 1) / k.fwd_rows, 1, fwd_args);
  auto out = at::empty(reduction == 0 ? at::IntArrayRef{logits.size(0)} : at::IntArrayRef{},
                       logits.options().dtype(value_type(logits)));
  reduce_rows(stream, k, s_ptr, out.data_ptr(), n);
  save_context(ctx, {logits_in, target_in, stats, gamma, handles});
  return {out, stats};
}

at::Tensor hinge_cuda_backward(const at::Tensor& logits_in, const at::Tensor& target_in,
                               const at::Tensor& stats, const at::Tensor& upstream,
                               const at::Tensor& gamma, const at::Tensor& handles, bool overwrite) {
  const Kernels k = unpack(handles);
  c10::cuda::CUDAGuard device_guard(logits_in.device());
  const auto logits = aligned16(logits_in);
  const auto target = aligned16(target_in);
  // The backward kernel is compiled for a float64 upstream derivative.
  const auto u = aligned16(upstream.to(at::kDouble).contiguous());
  // Each row's program writes the target entry after its last read of Z.
  auto grad = gradient_buffer(logits, overwrite);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.device().index()).stream();
  void* z_ptr = logits.data_ptr();
  void* y_ptr = target.data_ptr();
  void* s_ptr = stats.data_ptr();
  void* u_ptr = u.data_ptr();
  void* g_ptr = grad.data_ptr();
  void* gamma_ptr = gamma.data_ptr();
  void* zero_ptr = nullptr;
  int32_t n = rows(logits);
  void* bwd_args[] = {&z_ptr, &y_ptr, &s_ptr, &u_ptr, &g_ptr, &gamma_ptr, &n, &zero_ptr, &zero_ptr};
  launch_triton(stream, k.bwd, (n + k.bwd_rows - 1) / k.bwd_rows, 1, bwd_args);
  return grad;
}

struct HingeCallConfig {
  const at::Tensor* target;
  const at::Tensor* gamma;
  int64_t reduction;
  const at::Tensor* handles;
};

thread_local HingeCallConfig tls_hinge_cfg;

// A first-order gradient under create_graph: differentiating it raises (_first_order.py).
struct FirstOrderGradient : public torch::autograd::Function<FirstOrderGradient> {
  static at::Tensor forward(torch::autograd::AutogradContext*, const at::Tensor& gradient,
                            const at::Tensor&) {
    return gradient.detach();
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext*,
                                                 torch::autograd::variable_list) {
    TORCH_CHECK(false,
                "This loss has first derivatives only: double backward is not supported. Only "
                "cross_entropy, multilabel_cross_entropy and topk_cross_entropy at k = 1 have "
                "higher derivatives (with boolean-mask targets, not in the native library)");
    return {};
  }
};

// First order only: under create_graph the gradient raises if differentiated.
struct CudaHingeFunction : public torch::autograd::Function<CudaHingeFunction> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& logits) {
    const auto& cfg = tls_hinge_cfg;
    return std::get<0>(
        hinge_cuda_forward(logits, *cfg.target, *cfg.gamma, cfg.reduction, *cfg.handles, ctx));
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const ContextGuard guard(ctx->saved_data["cu_ctx"].toInt());
    const bool graph = at::GradMode::is_enabled();
    const bool overwrite = dead_logits(saved[0]);
    at::Tensor gradient;
    {
      at::AutoGradMode no_graph(false);
      gradient = hinge_cuda_backward(saved[0], saved[1], saved[2], grad_outputs[0], saved[3],
                                     saved[4], overwrite);
    }
    if (graph && saved[0].requires_grad()) return {FirstOrderGradient::apply(gradient, saved[0])};
    return {gradient};
  }
};

at::Tensor hinge_cuda_apply(const at::Tensor& logits, const at::Tensor& target,
                            const at::Tensor& gamma, int64_t reduction, const at::Tensor& handles) {
  if (!at::GradMode::is_enabled() || !logits.requires_grad()) {
    return std::get<0>(hinge_cuda_forward(logits, target, gamma, reduction, handles));
  }
  tls_hinge_cfg = {&target, &gamma, reduction, &handles};
  return CudaHingeFunction::apply(logits);
}

// distillation_loss (_triton_distillation.py): Triton forward, reduce and backward
// launched from C++. Forward saves 13 row statistics; backward recomputes the gradient.
std::tuple<at::Tensor, at::Tensor> distillation_cuda_forward(
    const at::Tensor& student_in, const at::Tensor& teacher_in, const at::Tensor& temperature,
    int64_t reduction, const at::Tensor& handles, torch::autograd::AutogradContext* ctx = nullptr) {
  const Kernels k = unpack(handles);
  c10::cuda::CUDAGuard device_guard(student_in.device());
  const auto student = aligned16(student_in);
  const auto teacher = aligned16(teacher_in);
  int32_t n = rows(student);
  auto stats = at::empty({stats_size(n, 13, k)}, student.options().dtype(at::kDouble));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(student.device().index()).stream();
  void* s_ptr = student.data_ptr();
  void* t_ptr = teacher.data_ptr();
  void* temp_ptr = temperature.data_ptr();
  void* st_ptr = stats.data_ptr();
  void* zero_ptr = nullptr;
  void* fwd_args[] = {&s_ptr, &t_ptr, &temp_ptr, &st_ptr, &n, &zero_ptr, &zero_ptr};
  launch_triton(stream, k.fwd, (n + k.fwd_rows - 1) / k.fwd_rows, 1, fwd_args);
  auto out = at::empty(reduction == 0 ? at::IntArrayRef{student.size(0)} : at::IntArrayRef{},
                       student.options().dtype(value_type(student)));
  reduce_rows(stream, k, st_ptr, out.data_ptr(), n);
  save_context(ctx, {student_in, teacher_in, stats, temperature, handles});
  return {out, stats};
}

at::Tensor distillation_cuda_backward(const at::Tensor& student_in, const at::Tensor& teacher_in,
                                      const at::Tensor& stats, const at::Tensor& upstream,
                                      const at::Tensor& temperature, const at::Tensor& handles) {
  const Kernels k = unpack(handles);
  c10::cuda::CUDAGuard device_guard(student_in.device());
  const auto student = aligned16(student_in);
  const auto teacher = aligned16(teacher_in);
  // The backward kernel is compiled for an upstream derivative of the loss dtype.
  const auto u = aligned16(upstream.to(value_type(student)).contiguous());
  auto grad = at::empty_like(student);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(student.device().index()).stream();
  void* s_ptr = student.data_ptr();
  void* t_ptr = teacher.data_ptr();
  void* temp_ptr = temperature.data_ptr();
  void* st_ptr = stats.data_ptr();
  void* u_ptr = u.data_ptr();
  void* g_ptr = grad.data_ptr();
  void* zero_ptr = nullptr;
  int32_t n = rows(student);
  void* bwd_args[] = {&s_ptr, &t_ptr, &temp_ptr, &st_ptr, &u_ptr, &g_ptr, &n, &zero_ptr, &zero_ptr};
  launch_triton(stream, k.bwd, (n + k.bwd_rows - 1) / k.bwd_rows, 1, bwd_args);
  return grad;
}

struct DistillationCallConfig {
  const at::Tensor* teacher;
  const at::Tensor* temperature;
  int64_t reduction;
  const at::Tensor* handles;
};

thread_local DistillationCallConfig tls_distillation_cfg;

// First order only: under create_graph the gradient raises if differentiated.
struct CudaDistillationFunction : public torch::autograd::Function<CudaDistillationFunction> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& student) {
    const auto& cfg = tls_distillation_cfg;
    return std::get<0>(distillation_cuda_forward(student, *cfg.teacher, *cfg.temperature,
                                                 cfg.reduction, *cfg.handles, ctx));
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const ContextGuard guard(ctx->saved_data["cu_ctx"].toInt());
    const bool graph = at::GradMode::is_enabled();
    at::Tensor gradient;
    {
      at::AutoGradMode no_graph(false);
      gradient = distillation_cuda_backward(saved[0], saved[1], saved[2], grad_outputs[0], saved[3],
                                            saved[4]);
    }
    if (graph && saved[0].requires_grad()) return {FirstOrderGradient::apply(gradient, saved[0])};
    return {gradient};
  }
};

at::Tensor distillation_cuda_apply(const at::Tensor& student, const at::Tensor& teacher,
                                   const at::Tensor& temperature, int64_t reduction,
                                   const at::Tensor& handles) {
  if (!at::GradMode::is_enabled() || !student.requires_grad()) {
    return std::get<0>(
        distillation_cuda_forward(student, teacher, temperature, reduction, handles));
  }
  tls_distillation_cfg = {&teacher, &temperature, reduction, &handles};
  return CudaDistillationFunction::apply(student);
}

// Boolean-mask multilabel losses (_triton_multilabel.py): obj_id 0=entropy, 1=hinge, 2=smooth.
std::tuple<at::Tensor, at::Tensor> multilabel_cuda_forward(
    const at::Tensor& logits_in, const at::Tensor& mask_in, const at::Tensor& gamma, int64_t obj_id,
    int64_t reduction, const at::Tensor& handles, torch::autograd::AutogradContext* ctx = nullptr) {
  const Kernels k = unpack(handles);
  c10::cuda::CUDAGuard device_guard(logits_in.device());
  const auto logits = aligned16(logits_in);
  const auto mask = aligned16(mask_in);
  int32_t n = rows(logits);
  const int64_t status_slot = obj_id == 0 ? 6 : (obj_id == 1 ? 5 : 9);
  auto stats = at::empty({stats_size(n, status_slot, k)}, logits.options().dtype(at::kDouble));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.device().index()).stream();
  void* z_ptr = logits.data_ptr();
  void* m_ptr = mask.data_ptr();
  void* s_ptr = stats.data_ptr();
  void* g_ptr = gamma.data_ptr();
  void* zero_ptr = nullptr;
  const int64_t fwd_gx = (n + k.fwd_rows - 1) / k.fwd_rows;
  if (obj_id <= 1) {
    void* fwd_args[] = {&z_ptr, &m_ptr, &s_ptr, &n, &zero_ptr, &zero_ptr};
    launch_triton(stream, k.fwd, fwd_gx, 1, fwd_args);
  } else {
    void* fwd_args[] = {&z_ptr, &m_ptr, &s_ptr, &g_ptr, &n, &zero_ptr, &zero_ptr};
    launch_triton(stream, k.fwd, fwd_gx, 1, fwd_args);
  }
  auto out_dtype = value_type(logits);
  auto out = at::empty(reduction == 0 ? at::IntArrayRef{logits.size(0)} : at::IntArrayRef{},
                       logits.options().dtype(out_dtype));
  reduce_rows(stream, k, s_ptr, out.data_ptr(), n);
  save_context(ctx, {logits_in, mask_in, stats, gamma, handles});
  if (ctx != nullptr) {
    ctx->saved_data["obj_id"] = obj_id;
    ctx->saved_data["reduction"] = reduction;
  }
  return {out, stats};
}

at::Tensor multilabel_cuda_backward(const at::Tensor& logits_in, const at::Tensor& mask_in,
                                    const at::Tensor& stats, const at::Tensor& upstream,
                                    const at::Tensor& gamma, int64_t obj_id,
                                    const at::Tensor& handles, bool overwrite) {
  const Kernels k = unpack(handles);
  c10::cuda::CUDAGuard device_guard(logits_in.device());
  const auto logits = aligned16(logits_in);
  const auto mask = aligned16(mask_in);
  const auto u = aligned16(upstream.to(at::kDouble).contiguous());
  // The kernels read only the columns each program writes.
  auto grad = gradient_buffer(logits, overwrite);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.device().index()).stream();
  void* z_ptr = logits.data_ptr();
  void* m_ptr = mask.data_ptr();
  void* s_ptr = stats.data_ptr();
  void* u_ptr = u.data_ptr();
  void* g_ptr = grad.data_ptr();
  void* gamma_ptr = gamma.data_ptr();
  void* zero_ptr = nullptr;
  int32_t n = rows(logits);
  int64_t gx = (n + k.bwd_rows - 1) / k.bwd_rows;
  int64_t gy = (logits.size(1) + k.b_bwd - 1) / k.b_bwd;
  if (obj_id == 0) {
    void* bwd_args[] = {&z_ptr, &m_ptr, &s_ptr, &u_ptr, &g_ptr, &n, &zero_ptr, &zero_ptr};
    launch_triton(stream, k.bwd, gx, gy, bwd_args);
  } else if (obj_id == 1) {
    void* bwd_args[] = {&s_ptr, &u_ptr, &g_ptr, &n, &zero_ptr, &zero_ptr};
    launch_triton(stream, k.bwd, gx, gy, bwd_args);
  } else {
    void* bwd_args[] = {&z_ptr,     &m_ptr, &s_ptr,    &u_ptr,   &g_ptr,
                        &gamma_ptr, &n,     &zero_ptr, &zero_ptr};
    launch_triton(stream, k.bwd, gx, gy, bwd_args);
  }
  return grad;
}

struct MultilabelCallConfig {
  const at::Tensor* mask;
  const at::Tensor* gamma;
  int64_t obj_id;
  int64_t reduction;
  const at::Tensor* handles;
};

thread_local MultilabelCallConfig tls_multilabel_cfg;

// The multilabel cross-entropy gradient as differentiable tensor operations (the scaled branch
// of _TensorEntropy.backward): exponentials from squared half exponentials, the upstream
// derivative applied before the negatives' products, so that tails below the subnormal range
// at upstream 1 survive a large upstream; expm1 of clamped unselected branches, whose overflow
// would make the next derivative 0 * inf.
at::Tensor multilabel_entropy_diff_bwd(const at::Tensor& logits, const at::Tensor& mask_u8,
                                       const at::Tensor& upstream, int64_t reduction) {
  const auto pos = mask_u8.to(at::kBool);
  const auto z = logits.to(at::kDouble);
  const auto invalid = (pos.sum(-1, true) == 0) | (~z.isfinite().all(-1, true));
  const auto root = ((z - std::get<0>(z.max(-1, true))) * 0.5).exp();
  const auto e = root.square();
  const auto den = e.sum(-1, true);
  const auto anchor = mask_u8.argmax(-1, true);
  const auto delta = z - z.gather(-1, anchor);
  auto scale = upstream.to(at::kDouble).expand({logits.size(0)});
  if (reduction == 1) scale = scale / static_cast<double>(logits.size(0));
  const auto up = scale.unsqueeze(1);
  const auto differences = at::where(delta >= 0, -e * at::expm1((-delta).clamp_max(1.0)),
                                     e.gather(-1, anchor) * at::expm1(delta.clamp_max(1.0))) /
                           den * up;
  const auto negative_root = at::where(pos, 0.0, root);
  const auto largest = negative_root.amax(-1, true);
  const auto unit = at::where(largest > 0, largest, 1.0);
  const auto total = (negative_root / unit).square().sum(-1, true);
  const auto per_den = up / den;
  const auto negative_mass = unit * ((unit * per_den) * total);
  const auto neg = negative_root * (negative_root * per_den);
  const auto count = pos.sum(-1, true).to(at::kDouble);
  const auto correction = (at::where(pos, differences, 0.0).sum(-1, true) + negative_mass) / count;
  return at::where(pos, differences - correction, neg)
      .masked_fill(invalid, std::numeric_limits<double>::quiet_NaN())
      .to(logits.scalar_type());
}

struct CudaMultilabelFunction : public torch::autograd::Function<CudaMultilabelFunction> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& logits) {
    const auto& cfg = tls_multilabel_cfg;
    return std::get<0>(multilabel_cuda_forward(logits, *cfg.mask, *cfg.gamma, cfg.obj_id,
                                               cfg.reduction, *cfg.handles, ctx));
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const int64_t obj_id = ctx->saved_data["obj_id"].toInt();
    const int64_t reduction = ctx->saved_data["reduction"].toInt();
    const bool graph = at::GradMode::is_enabled();
    if (graph && obj_id == 0) {
      return {multilabel_entropy_diff_bwd(saved[0], saved[1], grad_outputs[0], reduction)};
    }
    const ContextGuard guard(ctx->saved_data["cu_ctx"].toInt());
    const bool overwrite = dead_logits(saved[0]);
    at::Tensor gradient;
    {
      at::AutoGradMode no_graph(false);
      gradient = multilabel_cuda_backward(saved[0], saved[1], saved[2], grad_outputs[0], saved[3],
                                          obj_id, saved[4], overwrite);
    }
    if (graph && saved[0].requires_grad()) return {FirstOrderGradient::apply(gradient, saved[0])};
    return {gradient};
  }
};

at::Tensor multilabel_cuda_apply(const at::Tensor& logits, const at::Tensor& mask,
                                 const at::Tensor& gamma, int64_t obj_id, int64_t reduction,
                                 const at::Tensor& handles) {
  if (!at::GradMode::is_enabled() || !logits.requires_grad()) {
    return std::get<0>(multilabel_cuda_forward(logits, mask, gamma, obj_id, reduction, handles));
  }
  tls_multilabel_cfg = {&mask, &gamma, obj_id, reduction, &handles};
  return CudaMultilabelFunction::apply(logits);
}
#endif

}  // namespace

#endif  // SDCA_TORCH_CUDA_H
