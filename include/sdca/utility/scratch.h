#ifndef SDCA_UTILITY_SCRATCH_H
#define SDCA_UTILITY_SCRATCH_H

#include <cstddef>
#include <vector>

namespace sdca {

namespace detail {

// Small class counts are common. Keep per-coordinate scratch on the stack,
// and reuse per-thread heap vectors for larger rows up to 2^21 elements.
template <typename T>
struct entropy_buffer {
  T local[64];
  std::vector<T> heap;
  explicit entropy_buffer(std::size_t n = 0) {
    if (n > 64) acquire(n);
  }
  ~entropy_buffer() {
    if (!heap.empty()) release();
  }
  entropy_buffer(const entropy_buffer&) = delete;
  entropy_buffer& operator=(const entropy_buffer&) = delete;
  T* get(std::size_t n) {
    if (n > 64 && heap.size() < n) {
      if (heap.empty())
        acquire(n);
      else
        heap.resize(n);
    }
    return data();
  }
  T* data() { return heap.empty() ? local : heap.data(); }
  const T* data() const { return heap.empty() ? local : heap.data(); }

 private:
  static std::vector<std::vector<T>>& pool() {
    static thread_local std::vector<std::vector<T>> free_list;
    return free_list;
  }
  void acquire(std::size_t n) {
    std::vector<std::vector<T>>& free_list = pool();
    // Here, not in release(): a destructor's allocation failure would terminate.
    free_list.reserve(16);
    if (!free_list.empty()) {
      heap.swap(free_list.back());
      free_list.pop_back();
    }
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

// The type of a retry after a failed solve: double for float, else long double.
template <typename T>
struct wider {
  using type = long double;
};
template <>
struct wider<float> {
  using type = double;
};

}  // namespace detail

}  // namespace sdca

#endif
