#ifndef SDCA_UTILITY_BUFFER_H
#define SDCA_UTILITY_BUFFER_H

// Growable storage of trivially copyable values for the kernels' rows, zeroed when grown.
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>

namespace sdca {
namespace detail {

template <typename T>
class Buffer {
 public:
  Buffer() : data_(0), size_(0) {}
  explicit Buffer(std::size_t n) : data_(0), size_(0) { resize(n); }
  ~Buffer() { ::operator delete(data_); }

  // At least n values, zeroed when new; earlier contents are not kept.
  T* get(std::size_t n) {
    if (n > size_) resize(n);
    return data_;
  }
  void resize(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
    T* room = static_cast<T*>(::operator new((n ? n : 1) * sizeof(T)));
    std::memset(static_cast<void*>(room), 0, n * sizeof(T));
    ::operator delete(data_);
    data_ = room;
    size_ = n;
  }
  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  Buffer(const Buffer&);
  Buffer& operator=(const Buffer&);
  T* data_;
  std::size_t size_;
};

}  // namespace detail
}  // namespace sdca

#endif  // SDCA_UTILITY_BUFFER_H
