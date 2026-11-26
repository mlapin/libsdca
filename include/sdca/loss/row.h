#ifndef SDCA_LOSS_ROW_H
#define SDCA_LOSS_ROW_H

// The row contract of the losses (the same types for every instruction set). A loss is a
// functor of its instruction set's namespace (sdca/core/isa.h) with its parameters as
// members and
//
//   template <typename W, typename Isa, typename S, typename G, typename Target>
//   double operator()(const RowArgs<S, G>& row, const Target& target, Scratch& s) const;
//
// for each row target it takes: W the work type (float or double), Isa the lanes' instruction
// set, S the storage type of the scores, G that of the gradient (S, or float for 16-bit S). It
// reads the n scores and the target, writes the gradient times row.scale into row.grad (unless
// null) and returns the value (not scaled). An invalid row returns Row::invalid() (NaN) and
// leaves the gradient unspecified: the driver overwrites it with NaN and counts the row. A
// float-work row that float cannot represent sets s.needs_double before writing any gradient
// entry and returns: the driver reruns it in double work. Masked classes (scores of -inf) go
// through the same loops (their exponentials are +0.0). Zero gradient entries are +0.0, and
// row.grad may be row.scores (G = S): every score is read before its gradient entry is written.
//
// Labels: the driver gives ignored rows (Labels::ignore, label ignore_index) the value 0 and a
// zero gradient without calling the loss, and, unless the loss declares
// static const bool class_weights = true (it applies Label::class_weight itself), multiplies
// the value and the gradient of a row by class_weight[label].
//
// A loss may also take blocks of rows of at most Loss::block_classes classes (the static
// constant; none without it):
//
//   template <typename W, typename Isa, typename S, typename G, typename Targets>
//   bool block(const BlockArgs<S, G>& b, const Targets& t, std::size_t first, double* values,
//              bool* done, Scratch& s) const;
//
// on the b.rows <= Lanes<W, Isa>::L rows first, first + 1, ... of the batch targets t: false
// leaves every row to operator(); else the rows with done[r] have their value in values[r] and
// their gradient written, and operator() takes the others (and every ignored row). A row's
// results do not depend on the other rows of its block.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "sdca/utility/buffer.h"

namespace sdca {
namespace loss {

template <typename S, typename G = S>
struct RowArgs {
  const S* scores;
  G* grad;
  std::size_t n;
  double scale;  // the upstream factor of the gradient
};

template <typename S, typename G = S>
struct BlockArgs {
  const S* scores;  // rows x n
  G* grad;          // rows x n, or null
  std::size_t rows, n;
  const double* scale;  // rows
};

// The targets of one row.
struct Label {
  int64_t index;
  const double* class_weight;  // n entries, or null
};
struct LabelSet {
  const int64_t* labels;  // sorted, without duplicates
  std::size_t count;
};
struct MaskRow {
  const uint8_t* mask;  // n entries, nonzero for the labels
};
template <typename P>
struct ProbabilityRow {
  const P* p;                  // n entries
  const double* class_weight;  // n entries, or null
};
template <typename S>
struct TeacherRow {
  const S* t;  // n entries
};

struct Row {
  static double invalid() { return std::numeric_limits<double>::quiet_NaN(); }
};

// Per-thread state of the rows of a call: buffers reused across rows, and the retry flag.
class Scratch {
 public:
  Scratch() : needs_double(false) {}
  bool needs_double;
  // At least n values of a buffer of its own for each slot (0 to 3); contents not kept.
  double* doubles(std::size_t n, int slot = 0) { return doubles_[slot].get(n); }
  float* floats(std::size_t n, int slot = 0) { return floats_[slot].get(n); }
  template <typename T>
  T* buffer(std::size_t n, int slot = 0);

 private:
  detail::Buffer<double> doubles_[4];
  detail::Buffer<float> floats_[4];
};
template <>
inline double* Scratch::buffer<double>(std::size_t n, int slot) {
  return doubles(n, slot);
}
template <>
inline float* Scratch::buffer<float>(std::size_t n, int slot) {
  return floats(n, slot);
}

// Batch targets: the target of row i of n classes.
struct Labels {
  const int64_t* labels;
  const double* class_weight;  // n entries, or null
  int64_t ignore_index;
  bool ignore;  // rows of the label ignore_index are ignored
  Label row(std::size_t i, std::size_t) const {
    const Label l = {labels[i], class_weight};
    return l;
  }
  bool ignored(std::size_t i) const { return ignore && labels[i] == ignore_index; }
};
struct LabelSets {
  const int64_t* labels;
  const int64_t* offsets;  // row i: labels[offsets[i], offsets[i + 1])
  LabelSet row(std::size_t i, std::size_t) const {
    const LabelSet s = {labels + offsets[i], static_cast<std::size_t>(offsets[i + 1] - offsets[i])};
    return s;
  }
};
struct Mask {
  const uint8_t* mask;  // bytes 0 or 1, as bool storage
  MaskRow row(std::size_t i, std::size_t n) const {
    const MaskRow m = {mask + i * n};
    return m;
  }
};
template <typename P>
struct Probabilities {
  const P* probabilities;
  const double* class_weight;  // n entries, or null
  ProbabilityRow<P> row(std::size_t i, std::size_t n) const {
    const ProbabilityRow<P> r = {probabilities + i * n, class_weight};
    return r;
  }
};
template <typename S>
struct Teacher {
  const S* logits;
  TeacherRow<S> row(std::size_t i, std::size_t n) const {
    const TeacherRow<S> r = {logits + i * n};
    return r;
  }
};

// Loss::block_classes, or 0.
template <typename Loss>
struct BlockClasses {
  template <typename U>
  static std::integral_constant<std::size_t, U::block_classes> test(int);
  template <typename U>
  static std::integral_constant<std::size_t, 0> test(...);
  static const std::size_t value = decltype(test<Loss>(0))::value;
};
// Loss::float_parameters() (whether float work holds the loss's parameters), or true; the
// driver runs a call whose parameters float does not hold in double work.
template <typename Loss>
struct FloatParameters {
  template <typename U>
  static auto test(const U& l, int) -> decltype(l.float_parameters()) {
    return l.float_parameters();
  }
  template <typename U>
  static bool test(const U&, ...) {
    return true;
  }
  static bool held(const Loss& l) { return test(l, 0); }
};
// Loss::class_weights, or false.
template <typename Loss>
struct ClassWeights {
  template <typename U>
  static std::integral_constant<bool, U::class_weights> test(int);
  template <typename U>
  static std::false_type test(...);
  static const bool value = decltype(test<Loss>(0))::value;
};

}  // namespace loss
}  // namespace sdca

#endif  // SDCA_LOSS_ROW_H
