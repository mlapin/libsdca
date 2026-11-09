#ifndef SDCA_SOLVER_DATA_OUTPUT_H
#define SDCA_SOLVER_DATA_OUTPUT_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "sdca/math/compensated.h"
#include "sdca/utility/types.h"

namespace sdca {

inline void validate_labels_and_offsets(size_type, const std::vector<size_type>&,
                                        const std::vector<size_type>&);

template <typename Iterator>
inline std::vector<size_type> copy_class_labels(Iterator first, Iterator last) {
  std::vector<size_type> labels;
  for (; first != last; ++first) {
    const long double x = static_cast<long double>(*first);
    if (!std::isfinite(x) || x < 0 || std::floor(x) != x ||
        x >= static_cast<long double>(std::numeric_limits<size_type>::max()))
      throw std::invalid_argument("Class labels must be nonnegative integers");
    labels.push_back(static_cast<size_type>(x));
  }
  return labels;
}

struct multiclass_output {
  size_type num_classes = 0;
  std::vector<size_type> labels;

  multiclass_output(const size_type num_classes_, std::vector<size_type>& labels_)
      : num_classes(num_classes_), labels(std::move(labels_)) {
    if (num_classes == 0 || labels.empty() ||
        *std::max_element(labels.begin(), labels.end()) >= num_classes)
      throw std::invalid_argument("Invalid multiclass labels");
  }

  inline std::string to_string() const {
    std::ostringstream str;
    str << "labels (num_classes: " << num_classes << ", num_examples: " << labels.size() << ")";
    return str.str();
  }

  inline size_type num_labels(const size_type) const { return 1; }

  inline size_type num_examples() const { return labels.size(); }

  template <typename Data>
  inline void move_front(const size_type i, Data* x) const {
    std::swap(x[0], x[labels[i]]);
  }

  template <typename Data>
  inline void move_front(const size_type i, Data* x, Data* y) const {
    std::swap(x[0], x[labels[i]]);
    std::swap(y[0], y[labels[i]]);
  }

  template <typename Data>
  inline void move_back(const size_type i, Data* x) const {
    std::swap(x[0], x[labels[i]]);
  }
};

struct multilabel_output {
  size_type num_classes = 0;
  std::vector<size_type> labels;
  std::vector<size_type> offsets;

  multilabel_output(const size_type num_classes_, std::vector<size_type>& labels_,
                    std::vector<size_type>& offsets_)
      : num_classes(num_classes_), labels(std::move(labels_)), offsets(std::move(offsets_)) {
    validate_labels_and_offsets(num_classes, labels, offsets);
  }

  inline std::string to_string() const {
    std::ostringstream str;
    str << "labels (num_classes: " << num_classes << ", num_labels: " << labels.size()
        << ", num_examples: " << (offsets.size() > 0 ? offsets.size() - 1 : 0) << ")";
    return str.str();
  }

  inline size_type num_labels(const size_type i) const { return offsets[i + 1] - offsets[i]; }

  inline size_type num_examples() const { return offsets.size() - 1; }

  inline std::vector<size_type>::const_iterator labels_cbegin(const size_type i) const {
    return labels.cbegin() + static_cast<diff_type>(offsets[i]);
  }

  inline std::vector<size_type>::const_iterator labels_cend(const size_type i) const {
    return labels.cbegin() + static_cast<diff_type>(offsets[i + 1]);
  }

  template <typename Data>
  inline void move_front(const size_type i, Data* x) const {
    size_type offset = offsets[i];
    size_type num_labels_local = offsets[i + 1] - offset;
    for (size_type j = 0; j < num_labels_local; ++j) {
      size_type label = labels[offset + j];
      std::swap(x[j], x[label]);
    }
  }

  template <typename Data>
  inline void move_front(const size_type i, Data* x, Data* y) const {
    size_type offset = offsets[i];
    size_type num_labels_local = offsets[i + 1] - offset;
    for (size_type j = 0; j < num_labels_local; ++j) {
      size_type label = labels[offset + j];
      std::swap(x[j], x[label]);
      std::swap(y[j], y[label]);
    }
  }

  template <typename Data>
  inline void move_back(const size_type i, Data* x) const {
    size_type offset = offsets[i];
    size_type num_labels_local = offsets[i + 1] - offset;
    for (size_type j = num_labels_local - 1;;) {
      size_type label = labels[offset + j];
      std::swap(x[j], x[label]);
      if (j-- == 0) break;
    }
  }
};

template <typename Iterator>
inline std::pair<Iterator, Iterator> validate_labels(Iterator first, Iterator last) {
  if (first == last) throw std::invalid_argument("Labels must not be empty");
  auto minmax = std::minmax_element(first, last);
  if (*minmax.second == std::numeric_limits<size_type>::max())
    throw std::invalid_argument("Invalid class label");
  if (*minmax.first == 1) {
    std::for_each(first, last, [](size_type& x) { x -= 1; });
  } else if (*minmax.first != 0) {
    throw std::invalid_argument("Invalid class labels range.");
  }
  return minmax;
}

inline void validate_labels_and_offsets(const size_type num_classes,
                                        const std::vector<size_type>& labels,
                                        const std::vector<size_type>& offsets) {
  if (num_classes == 0 || offsets.size() < 2 || offsets.front() != 0 ||
      offsets.back() != labels.size())
    throw std::invalid_argument("Offsets must span the entire nonempty label array");
  for (size_type i = 0; i < offsets.size() - 1; ++i) {
    if (offsets[i] > offsets[i + 1] || offsets[i + 1] > labels.size())
      throw std::invalid_argument("Offsets must be increasing and within the label array");
    size_type num_labels = offsets[i + 1] - offsets[i];
    if (num_labels < 1 || num_labels >= num_classes) {
      throw std::invalid_argument(
          "Each example must have between "
          "1 and num_classes - 1 labels.");
    }
    auto first = &labels[offsets[i]];
    if (*std::max_element(first, first + num_labels) >= num_classes ||
        std::adjacent_find(first, first + num_labels, std::greater_equal<size_type>()) !=
            first + num_labels) {
      throw std::invalid_argument(
          "All labels for every example must be "
          "distinct and sorted.");
    }
  }
}

// Soft-target (distillation) output: a row-major (num_examples, num_classes)
// probability distribution q per example (q_j >= 0, sum_j q_j = 1).
struct distribution_output {
  size_type num_classes = 0;
  size_type examples = 0;
  std::vector<double> targets;

  distribution_output(const size_type num_classes_, const size_type num_examples_,
                      std::vector<double>&& targets_)
      : num_classes(num_classes_), examples(num_examples_), targets(std::move(targets_)) {
    if (num_classes < 2 || examples == 0 || targets.size() != examples * num_classes)
      throw std::invalid_argument("Invalid target distribution dimensions");
    const double tol = 64 * std::numeric_limits<double>::epsilon();
    for (size_type i = 0; i < examples; ++i) {
      const double* q = &targets[i * num_classes];
      detail::compensated_sum<double> sum;
      for (size_type j = 0; j < num_classes; ++j) {
        if (!std::isfinite(q[j]) || q[j] < 0)
          throw std::invalid_argument("Target probabilities must be finite and nonnegative");
        sum.add(q[j]);
      }
      const double s = sum.value();
      if (!(std::abs(s - 1.0) <= tol * static_cast<double>(num_classes)))
        throw std::invalid_argument("Target probability rows must sum to one");
    }
  }

  inline std::string to_string() const {
    std::ostringstream str;
    str << "distribution (num_classes: " << num_classes << ", num_examples: " << examples << ")";
    return str.str();
  }

  inline size_type num_examples() const { return examples; }

  inline const double* row(const size_type i) const { return &targets[i * num_classes]; }
};

}  // namespace sdca

#endif
