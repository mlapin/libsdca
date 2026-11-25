// Eager CUDA calls replay the shared Triton plan directly after its first launch. The Python
// launcher supplies the compiled kernels and their buffer and scalar arguments. All other
// CUDA calls keep going through the dispatcher. No CUDA toolkit headers are needed.
#pragma once
#ifdef __linux__
#include <dlfcn.h>
#include <c10/core/DeviceGuard.h>

#include <array>
#include <cstring>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
#include <vector>

namespace libsdca_cuda {

using Tensor = at::Tensor;
using Slots = std::array<Tensor, 12>;
using Key = std::array<int64_t, 30>;
using Shape = std::vector<int64_t>;
enum Slot { Z, Y, W, R, D, S, V, L, G, E };

struct Driver {
  int (*current)(void**);
  int (*parameter)(void*, size_t, size_t*, size_t*);
  int (*stream)(int32_t, void**);
  int (*launch)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*,
                void**, void**);
};

inline const Driver& driver() {
  static const Driver d = [] {
    Driver out{};
    void* cuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    void* torch = dlopen("libtorch_cuda.so", RTLD_LAZY | RTLD_NOLOAD);
    if (cuda && torch) {
      out.current = reinterpret_cast<decltype(out.current)>(dlsym(cuda, "cuCtxGetCurrent"));
      out.parameter = reinterpret_cast<decltype(out.parameter)>(dlsym(cuda, "cuFuncGetParamInfo"));
      out.launch = reinterpret_cast<decltype(out.launch)>(dlsym(cuda, "cuLaunchKernel"));
      // PyTorch's stable AOTInductor C shim returns the active stream, including external
      // and capturing streams, without a CUDA runtime dependency in this extension.
      out.stream = reinterpret_cast<decltype(out.stream)>(
          dlsym(torch, "aoti_torch_get_current_cuda_stream"));
    }
    return out;
  }();
  return d;
}

struct Allocation {
  size_t slot;
  at::ScalarType dtype;
  Shape shape;
};
struct Launch {
  uintptr_t function;
  unsigned x, y, z, threads, shared;
  // One 64-bit cell per driver argument. Pointer slots replace their cells at launch;
  // the driver reads the declared width for scalar arguments (i32 or i64, little-endian).
  std::vector<int64_t> cells;
  std::vector<std::pair<size_t, size_t>> buffers;
};

struct Program {
  std::vector<Allocation> arrays;
  std::vector<Launch> launches;
  pybind11::object kernels;  // Own Triton's loaded modules until this plan is released.
  ~Program() {
    pybind11::gil_scoped_acquire gil;
    kernels = {};
  }

  void run(Slots& slots, void* stream) const {
    for (const auto& a : arrays)
      slots[a.slot] = at::empty(a.shape, slots[Z].options().dtype(a.dtype));
    for (const auto& l : launches) {
      std::array<int64_t, 64> cells;
      std::array<void*, 64> args;
      std::copy(l.cells.begin(), l.cells.end(), cells.begin());
      for (const auto& b : l.buffers)
        cells[b.first] = reinterpret_cast<intptr_t>(slots[b.second].data_ptr());
      for (size_t i = 0; i < l.cells.size(); ++i) args[i] = &cells[i];
      const int error = driver().launch(reinterpret_cast<void*>(l.function), l.x, l.y, l.z,
                                        l.threads, 1, 1, l.shared, stream, args.data(), nullptr);
      TORCH_CHECK(error == 0, "libsdca: cuLaunchKernel failed with code ", error);
    }
  }
};

struct Cache {
  std::mutex mutex;
  using Entries = std::map<Key, std::shared_ptr<Program>>;
  Entries entries;
  std::array<Entries::iterator, 512> insertion_order;
  size_t next = 0;
};
inline Cache& cache() {
  // PyTorch's CUDA libraries and Triton's modules live for the process. Skip static
  // destruction after Python has finalized.
  static auto* c = new Cache;
  return *c;
}
inline thread_local const Key* requested = nullptr;

inline bool install(pybind11::object kernels,
                    const std::vector<std::tuple<size_t, at::ScalarType, Shape>>& arrays,
                    const std::vector<std::tuple<Shape, Shape, Shape, Shape>>& launches) {
  if (!requested || !driver().parameter) return false;
  auto program = std::make_shared<Program>();
  program->kernels = std::move(kernels);
  for (const auto& [slot, type, shape] : arrays) {
    TORCH_CHECK(slot < Slots{}.size(), "libsdca: invalid CUDA output slot");
    program->arrays.push_back({slot, type, shape});
  }
  for (const auto& [meta, cells, buffers, sizes] : launches) {
    TORCH_CHECK(meta.size() == 6 && cells.size() <= 62 && cells.size() == buffers.size() &&
                    cells.size() == sizes.size(),
                "libsdca: invalid CUDA launch");
    Launch l{static_cast<uintptr_t>(meta[0]),
             static_cast<unsigned>(meta[1]),
             static_cast<unsigned>(meta[2]),
             static_cast<unsigned>(meta[3]),
             static_cast<unsigned>(meta[4]),
             static_cast<unsigned>(meta[5]),
             cells,
             {}};
    for (size_t i = 0; i < buffers.size(); ++i) {
      if (buffers[i] < 0) continue;
      TORCH_CHECK(buffers[i] < static_cast<int64_t>(Slots{}.size()),
                  "libsdca: invalid CUDA input slot");
      l.buffers.emplace_back(i, static_cast<size_t>(buffers[i]));
    }
    // Verify the loaded function's actual parameter widths, including the 0-2 unused
    // pointer arguments that Triton's CUDA driver appends in the supported releases.
    size_t offset = 0, size = 0;
    void* function = reinterpret_cast<void*>(l.function);
    for (size_t i = 0; i < sizes.size(); ++i)
      if (driver().parameter(function, i, &offset, &size) || size != static_cast<size_t>(sizes[i]))
        return false;
    while (driver().parameter(function, l.cells.size(), &offset, &size) == 0) {
      if (size != sizeof(void*) || l.cells.size() >= sizes.size() + 2) return false;
      l.cells.push_back(0);
    }
    program->launches.push_back(std::move(l));
  }
  auto& c = cache();
  std::shared_ptr<Program> released;
  {
    std::lock_guard<std::mutex> lock(c.mutex);
    auto inserted = c.entries.try_emplace(*requested, std::move(program));
    if (inserted.second) {
      // Evict in insertion order. Erasing the smallest key repeatedly discards newly
      // warmed plans for short batches as soon as a second pointer alignment is installed.
      if (c.entries.size() > c.insertion_order.size()) {
        auto oldest = c.insertion_order[c.next];
        released = std::move(oldest->second);
        c.entries.erase(oldest);
      }
      c.insertion_order[c.next] = inserted.first;
      c.next = (c.next + 1) % c.insertion_order.size();
    }
  }
  return true;
}

// Label smoothing as a float64 device scalar (the kernels read it from memory), cached per
// device, value and stream as launch_torch._eps.
inline at::Tensor smoothing(const Tensor& like, double value, void* stream) {
  static std::mutex mutex;
  static auto* cached = new std::map<std::tuple<int64_t, int64_t, intptr_t>, Tensor>;
  int64_t bits;
  std::memcpy(&bits, &value, sizeof(value));
  const auto key = std::make_tuple(static_cast<int64_t>(like.device().index()), bits,
                                   reinterpret_cast<intptr_t>(stream));
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cached->find(key);
  if (it != cached->end()) return it->second;
  if (cached->size() > 64) cached->clear();
  auto t = at::full({}, value, like.options().dtype(at::kDouble));
  cached->emplace(key, t);
  return t;
}

// Reduce, record/save, gradient, and overwrite distinguish the forward and backward plans.
// Tensor dtype, pointer alignment and scalar/row upstream distinguish Triton specializations;
// the scores' row stride distinguishes strided plans (Call.stride) from contiguous ones.
template <typename F>
bool execute(Slots& slots, const libsdca_frontend::Spec& spec, int64_t reduction, bool record,
             bool grad, bool overwrite, double up, F&& fallback) {
  const Driver& d = driver();
  if (!d.launch || !d.current || !d.stream) {
    fallback();
    return false;
  }
  c10::DeviceGuard guard(slots[Z].device());
  void* context = nullptr;
  if (d.current(&context) || !context) {
    fallback();
    return false;
  }
  Key key{reinterpret_cast<intptr_t>(context),
          slots[Z].size(0),
          slots[Z].size(1),
          spec.kind,
          static_cast<int64_t>(spec.k),
          spec.ignore_index,
          spec.ignore,
          spec.precision,
          reduction,
          record,
          grad,
          overwrite};
  if (spec.label_smoothing != 0) {
    void* stream = nullptr;
    if (d.stream(slots[Z].device().index(), &stream) != 0) {
      fallback();
      return false;
    }
    slots[E] = smoothing(slots[Z], spec.label_smoothing, stream);
  }
  size_t i = 12;
  for (double v : {spec.gamma, spec.temperature, up, spec.label_smoothing}) {
    int64_t bits;
    std::memcpy(&bits, &v, sizeof(v));
    key[i++] = bits;
  }
  for (const auto& t : slots) {
    key[i++] = t.defined() ? (static_cast<int64_t>(t.scalar_type()) * 2 +
                              ((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) != 0))
                           : -1;
  }
  key[i++] = slots[R].defined() ? slots[R].dim() : -1;
  key[i] = slots[Z].stride(0);
  auto& c = cache();
  std::shared_ptr<Program> program;
  {
    std::lock_guard<std::mutex> lock(c.mutex);
    auto it = c.entries.find(key);
    if (it != c.entries.end()) program = it->second;
  }
  if (program) {
    void* stream = nullptr;
    TORCH_CHECK(d.stream(slots[Z].device().index(), &stream) == 0,
                "libsdca: could not read the current CUDA stream");
    program->run(slots, stream);
  } else {
    struct Scope {
      const Key* old;
      ~Scope() { requested = old; }
    } scope{requested};
    requested = &key;
    fallback();
    return false;
  }
  return true;
}
}  // namespace libsdca_cuda
#endif
