#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_CORE_PROJECT_NUMERIC_H)
#if SDCA_ISA_ID == 1
#define SDCA_CORE_PROJECT_NUMERIC_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_CORE_PROJECT_NUMERIC_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_CORE_PROJECT_NUMERIC_H3 1
#else
#define SDCA_CORE_PROJECT_NUMERIC_H4 1
#endif

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "sdca/core/reduce.h"
#include "sdca/core/lanes.h"

SDCA_ISA_BEGIN
namespace core {

// The type of a retry after a failed solve: double for float, else long double.
template <typename T>
struct wider {
  typedef long double type;
};
template <>
struct wider<float> {
  typedef double type;
};

// core::softplus takes double.
template <typename T>
inline T softplus(const T x) {
  return x > 0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}

// Scratch of n entries: on the stack for short rows, else from a per-thread pool.
template <typename T>
struct row_buffer {
  T local[64];
  std::vector<T> heap;
  bool pooled;
  explicit row_buffer(std::size_t n = 0) : pooled(false) {
    if (n > 64) acquire(n);
  }
  ~row_buffer() {
    if (pooled) release();
  }
  // At least n entries (the contents not kept), grown outside the pool: a buffer that outlives
  // the pool (thread_local) never returns to it.
  void resize(const std::size_t n) {
    if (n > 64 && heap.size() < n) heap.resize(n);
  }
  row_buffer(const row_buffer&) = delete;
  row_buffer& operator=(const row_buffer&) = delete;
  T* data() { return heap.empty() ? local : heap.data(); }
  const T* data() const { return heap.empty() ? local : heap.data(); }

 private:
  static std::vector<std::vector<T>>& pool() {
    static thread_local std::vector<std::vector<T>> free_list;
    return free_list;
  }
  void acquire(std::size_t n) {
    std::vector<std::vector<T>>& free_list = pool();
    // Here, not in release(): an allocation failure in a destructor would terminate.
    free_list.reserve(16);
    if (!free_list.empty()) {
      heap.swap(free_list.back());
      free_list.pop_back();
    }
    pooled = true;
    if (heap.size() < n) heap.resize(n);
  }
  void release() {
    if (heap.capacity() > (1u << 21)) return;
    std::vector<std::vector<T>>& free_list = pool();
    if (free_list.size() < 16) {
      free_list.push_back(std::vector<T>());
      free_list.back().swap(heap);
    }
  }
};

}  // namespace core
SDCA_ISA_END

#endif  // SDCA_CORE_PROJECT_NUMERIC_H
