// XLA FFI handler for the shared CUDA launch plan (triton/jax_ffi.py). The initialize
// stage loads Triton binaries into the stream's context, including before graph capture.
// The execute stage issues cuLaunchKernel on that stream. Only the NVIDIA driver and the
// headers of the running jaxlib are required.
#include <dlfcn.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUstream = struct CUstream_st*;

// Of cuda.h (stable driver ABI).
const CUresult success = 0;
const int func_shared_size_bytes = 1, func_max_dynamic_shared_size_bytes = 8;
const int func_shared_memory_mode = 17, shared_memory_mode_allow_oversized = 3;
const int device_max_shared_per_multiprocessor = 81, device_max_shared_per_block_optin = 97;
const int func_cache_prefer_shared = 1;

struct Driver {
  CUresult (*module_load_data)(CUmodule*, const void*) = nullptr;
  CUresult (*module_get_function)(CUfunction*, CUmodule, const char*) = nullptr;
  CUresult (*launch_kernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                            unsigned, CUstream, void**, void**) = nullptr;
  CUresult (*stream_get_ctx)(CUstream, CUcontext*) = nullptr;
  CUresult (*ctx_get_current)(CUcontext*) = nullptr;
  CUresult (*ctx_push_current)(CUcontext) = nullptr;
  CUresult (*ctx_pop_current)(CUcontext*) = nullptr;
  CUresult (*ctx_get_device)(CUdevice*) = nullptr;
  CUresult (*device_get_attribute)(int*, int, CUdevice) = nullptr;
  CUresult (*func_get_attribute)(int*, int, CUfunction) = nullptr;
  CUresult (*func_set_attribute)(CUfunction, int, int) = nullptr;
  CUresult (*func_set_cache_config)(CUfunction, int) = nullptr;
  CUresult (*stream_is_capturing)(CUstream, int*) = nullptr;
  CUresult (*get_error_string)(CUresult, const char**) = nullptr;
  std::string missing;
};

template <typename F>
void bind(void* library, const char* name, F& function, std::string& missing) {
  function = reinterpret_cast<F>(dlsym(library, name));
  if (function == nullptr && missing.empty()) missing = name;
}

const Driver& driver() {
  static const Driver loaded = [] {
    Driver d;
    void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      d.missing = "libcuda.so.1";
      return d;
    }
    bind(library, "cuModuleLoadData", d.module_load_data, d.missing);
    bind(library, "cuModuleGetFunction", d.module_get_function, d.missing);
    bind(library, "cuLaunchKernel", d.launch_kernel, d.missing);
    bind(library, "cuStreamGetCtx", d.stream_get_ctx, d.missing);
    bind(library, "cuCtxGetCurrent", d.ctx_get_current, d.missing);
    bind(library, "cuCtxPushCurrent_v2", d.ctx_push_current, d.missing);
    bind(library, "cuCtxPopCurrent_v2", d.ctx_pop_current, d.missing);
    bind(library, "cuCtxGetDevice", d.ctx_get_device, d.missing);
    bind(library, "cuDeviceGetAttribute", d.device_get_attribute, d.missing);
    bind(library, "cuFuncGetAttribute", d.func_get_attribute, d.missing);
    bind(library, "cuFuncSetAttribute", d.func_set_attribute, d.missing);
    bind(library, "cuFuncSetCacheConfig", d.func_set_cache_config, d.missing);
    bind(library, "cuStreamIsCapturing", d.stream_is_capturing, d.missing);
    bind(library, "cuGetErrorString", d.get_error_string, d.missing);
    return d;
  }();
  return loaded;
}

ffi::Error failed(const char* call, CUresult code) {
  const char* text = nullptr;
  if (driver().get_error_string == nullptr || driver().get_error_string(code, &text) != success)
    text = nullptr;
  return ffi::Error::Internal(std::string("libsdca: ") + call + " failed with code " +
                              std::to_string(code) + (text ? std::string(" (") + text + ")" : ""));
}

// The kernel table: per kernel its id (of its binary's contents), the offsets and lengths of
// its binary and name in `binaries`, its warps and its shared memory bytes.
const std::size_t kernel_fields = 7;

// Arguments of a launch in `program`: (kind, a, b) triples.
enum Kind : std::int64_t { buffer = 0, int32 = 1, int64 = 2, null = 3 };

// The kernel binaries, base64-encoded (RFC 4648) in the attribute string.
bool decode(std::string_view text, std::string& out) {
  static const auto table = [] {
    std::array<int, 256> values{};
    values.fill(-1);
    const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) values[static_cast<unsigned char>(alphabet[i])] = i;
    return values;
  }();
  out.clear();
  out.reserve(text.size() / 4 * 3);
  std::uint32_t buffer = 0;
  int bits = 0;
  for (char c : text) {
    if (c == '=') break;
    const int value = table[static_cast<unsigned char>(c)];
    if (value < 0) return false;
    buffer = (buffer << 6) | static_cast<std::uint32_t>(value);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<char>((buffer >> bits) & 0xFF));
    }
  }
  return true;
}

// Functions loaded per context and kernel id; modules stay loaded for the process.
std::mutex loaded_lock;
std::map<std::pair<CUcontext, std::int64_t>, CUfunction>& loaded() {
  static auto* functions = new std::map<std::pair<CUcontext, std::int64_t>, CUfunction>();
  return *functions;
}

// As Triton's loader (backends/nvidia/driver.c, loadBinary): kernels above 48 KiB of shared
// memory opt in to the device's maximum.
ffi::Error opt_in(CUfunction function, int shared) {
  const Driver& d = driver();
  CUdevice device = 0;
  int optin = 0;
  CUresult code = d.ctx_get_device(&device);
  if (code == success)
    code = d.device_get_attribute(&optin, device_max_shared_per_block_optin, device);
  if (code != success) return failed("cuDeviceGetAttribute", code);
  if (shared <= 49152 || optin <= 49152) return ffi::Error::Success();
  if (d.func_set_attribute(function, func_shared_memory_mode, shared_memory_mode_allow_oversized) ==
      success)
    return ffi::Error::Success();
  int total = 0, fixed = 0;
  code = d.func_set_cache_config(function, func_cache_prefer_shared);
  if (code == success)
    code = d.device_get_attribute(&total, device_max_shared_per_multiprocessor, device);
  if (code == success) code = d.func_get_attribute(&fixed, func_shared_size_bytes, function);
  if (code == success)
    code = d.func_set_attribute(function, func_max_dynamic_shared_size_bytes, optin - fixed);
  return code == success ? ffi::Error::Success() : failed("cuFuncSetAttribute", code);
}

// The functions of the kernel table in context, loading their binaries on first use (with
// context current). Loading is not allowed while a CUDA graph captures the stream: XLA runs
// the initialize stage, which loads them, before it records command buffers.
ffi::Error functions(CUcontext context, CUstream stream, ffi::Span<const std::int64_t> kernels,
                     std::string_view binaries, std::vector<CUfunction>& out) {
  const Driver& d = driver();
  if (kernels.size() % kernel_fields != 0)
    return ffi::Error::InvalidArgument("libsdca: malformed kernel table");
  out.clear();
  std::lock_guard<std::mutex> guard(loaded_lock);
  for (std::size_t i = 0; i < kernels.size(); i += kernel_fields) {
    const std::int64_t id = kernels[i];
    auto found = loaded().find({context, id});
    if (found != loaded().end()) {
      out.push_back(found->second);
      continue;
    }
    const std::int64_t binary = kernels[i + 1], binary_size = kernels[i + 2];
    const std::int64_t name = kernels[i + 3], name_size = kernels[i + 4];
    if (binary < 0 || binary_size <= 0 || name < 0 || name_size <= 0 ||
        static_cast<std::size_t>(binary + binary_size) > binaries.size() ||
        static_cast<std::size_t>(name + name_size) > binaries.size())
      return ffi::Error::InvalidArgument("libsdca: malformed kernel table");
    int capturing = 0;
    if (d.stream_is_capturing(stream, &capturing) == success && capturing != 0)
      return ffi::Error::Internal(
          "libsdca: a CUDA graph captures a kernel launch before its kernels are loaded");
    std::string image;
    if (!decode(binaries.substr(binary, binary_size), image))
      return ffi::Error::InvalidArgument("libsdca: malformed kernel binary");
    const std::string function_name(binaries.substr(name, name_size));
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUresult code = d.module_load_data(&module, image.data());
    if (code != success) return failed("cuModuleLoadData", code);
    code = d.module_get_function(&function, module, function_name.c_str());
    if (code != success) return failed("cuModuleGetFunction", code);
    ffi::Error error = opt_in(function, static_cast<int>(kernels[i + 6]));
    if (error.failure()) return error;
    loaded()[{context, id}] = function;
    out.push_back(function);
  }
  return ffi::Error::Success();
}

// The stream's context, current for the scope of a call (XLA's thread usually has it).
struct ContextScope {
  CUcontext context = nullptr;
  bool pushed = false;
  ffi::Error enter(CUstream stream) {
    const Driver& d = driver();
    if (!d.missing.empty())
      return ffi::Error::Internal("libsdca: the CUDA driver lacks " + d.missing);
    CUresult code = d.stream_get_ctx(stream, &context);
    if (code != success) return failed("cuStreamGetCtx", code);
    CUcontext current = nullptr;
    code = d.ctx_get_current(&current);
    if (code != success) return failed("cuCtxGetCurrent", code);
    if (current != context) {
      code = d.ctx_push_current(context);
      if (code != success) return failed("cuCtxPushCurrent", code);
      pushed = true;
    }
    return ffi::Error::Success();
  }
  ContextScope() = default;
  ContextScope(const ContextScope&) = delete;
  ContextScope& operator=(const ContextScope&) = delete;
  ~ContextScope() {
    CUcontext popped = nullptr;
    if (pushed) (void)driver().ctx_pop_current(&popped);
  }
};

// Device addresses of the call's buffers: the arguments, then the results.
ffi::Error addresses(ffi::RemainingArgs args, ffi::RemainingRets rets, std::vector<char*>& out) {
  out.clear();
  for (std::size_t i = 0; i < args.size(); ++i) {
    auto buffer = args.get<ffi::AnyBuffer>(i);
    if (buffer.has_error()) return buffer.error();
    out.push_back(static_cast<char*>(buffer->untyped_data()));
  }
  for (std::size_t i = 0; i < rets.size(); ++i) {
    auto buffer = rets.get<ffi::AnyBuffer>(i);
    if (buffer.has_error()) return buffer.error();
    out.push_back(static_cast<char*>((*buffer)->untyped_data()));
  }
  return ffi::Error::Success();
}

// A launch is (kernel, grid x, y, z, argument count), followed by (kind, value) per argument.
ffi::Error run(CUstream stream, const std::vector<CUfunction>& kernels,
               ffi::Span<const std::int64_t> table, ffi::Span<const std::int64_t> program,
               const std::vector<char*>& buffers) {
  std::vector<std::uint64_t> values;
  std::vector<void*> params;
  std::size_t at = 0;
  while (at < program.size()) {
    if (at + 5 > program.size()) return ffi::Error::InvalidArgument("libsdca: malformed program");
    const std::int64_t kernel = program[at], count = program[at + 4];
    const std::int64_t gx = program[at + 1], gy = program[at + 2], gz = program[at + 3];
    at += 5;
    if (kernel < 0 || static_cast<std::size_t>(kernel) >= kernels.size() || count < 0 ||
        at + 2 * static_cast<std::size_t>(count) > program.size())
      return ffi::Error::InvalidArgument("libsdca: malformed program");
    values.assign(static_cast<std::size_t>(count), 0);
    params.assign(static_cast<std::size_t>(count), nullptr);
    for (std::int64_t i = 0; i < count; ++i, at += 2) {
      const std::int64_t kind = program[at], a = program[at + 1];
      if (kind == buffer) {
        if (a < 0 || static_cast<std::size_t>(a) >= buffers.size())
          return ffi::Error::InvalidArgument("libsdca: malformed program");
        char* address = buffers[a];
        if ((reinterpret_cast<std::uintptr_t>(address) & 15) != 0)
          return ffi::Error::InvalidArgument("libsdca: a kernel buffer is not 16-byte aligned");
        std::memcpy(&values[i], &address, sizeof(address));
      } else if (kind == int32) {
        const std::int32_t value = static_cast<std::int32_t>(a);
        std::memcpy(&values[i], &value, sizeof(value));
      } else if (kind == int64) {
        std::memcpy(&values[i], &a, sizeof(a));
      } else if (kind != null) {
        return ffi::Error::InvalidArgument("libsdca: malformed program");
      }
      params[i] = &values[i];
    }
    const int warps = static_cast<int>(table[kernel_fields * kernel + 5]);
    const int shared = static_cast<int>(table[kernel_fields * kernel + 6]);
    CUresult code = driver().launch_kernel(
        kernels[kernel], static_cast<unsigned>(gx), static_cast<unsigned>(gy),
        static_cast<unsigned>(gz), static_cast<unsigned>(32 * warps), 1, 1,
        static_cast<unsigned>(shared), stream, params.data(), nullptr);
    if (code != success) return failed("cuLaunchKernel", code);
  }
  return ffi::Error::Success();
}

ffi::Error initialize(CUstream stream, ffi::RemainingArgs, ffi::RemainingRets,
                      ffi::Span<const std::int64_t> kernels, std::string_view binaries,
                      ffi::Span<const std::int64_t>) {
  ContextScope scope;
  ffi::Error error = scope.enter(stream);
  if (error.failure()) return error;
  std::vector<CUfunction> loaded_kernels;
  return functions(scope.context, stream, kernels, binaries, loaded_kernels);
}

ffi::Error execute(CUstream stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
                   ffi::Span<const std::int64_t> kernels, std::string_view binaries,
                   ffi::Span<const std::int64_t> program) {
  ContextScope scope;
  ffi::Error error = scope.enter(stream);
  std::vector<CUfunction> loaded_kernels;
  std::vector<char*> buffers;
  if (error.success()) error = functions(scope.context, stream, kernels, binaries, loaded_kernels);
  if (error.success()) error = addresses(args, rets, buffers);
  if (error.failure()) return error;
  return run(stream, loaded_kernels, kernels, program, buffers);
}

}  // namespace

#define SDCA_JAX_CUDA_BINDING(stage)                  \
  ffi::Ffi::Bind<stage>()                             \
      .Ctx<ffi::PlatformStream<CUstream>>()           \
      .RemainingArgs()                                \
      .RemainingRets()                                \
      .Attr<ffi::Span<const std::int64_t>>("kernels") \
      .Attr<std::string_view>("binaries")             \
      .Attr<ffi::Span<const std::int64_t>>("program")

XLA_FFI_DEFINE_HANDLER_SYMBOL(sdca_jax_cuda_initialize, initialize,
                              SDCA_JAX_CUDA_BINDING(ffi::ExecutionStage::kInitialize));
XLA_FFI_DEFINE_HANDLER_SYMBOL(sdca_jax_cuda_execute, execute,
                              SDCA_JAX_CUDA_BINDING(ffi::ExecutionStage::kExecute));
