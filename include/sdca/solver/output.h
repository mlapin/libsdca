#ifndef SDCA_SOLVER_OUTPUT_H
#define SDCA_SOLVER_OUTPUT_H

#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "sdca/types.h"

namespace sdca {

struct multiclass_output {
  size_type num_classes = 0;
  std::vector<size_type> labels;


  multiclass_output(
      const size_type __num_classes,
      std::vector<size_type>& __labels
    ) :
      num_classes(__num_classes),
      labels(std::move(__labels))
  {}


  inline std::string
  to_string() const {
    std::ostringstream str;
    str << "labels (num_classes: " << num_classes <<
           ", num_examples: " << labels.size() << ")";
    return str.str();
  }

};


struct multilabel_output {
  size_type num_classes = 0;
  std::vector<size_type> labels;
  std::vector<size_type> offsets;


  multilabel_output(
      const size_type __num_classes,
      std::vector<size_type>& __labels,
      std::vector<size_type>& __offsets
    ) :
      num_classes(__num_classes),
      labels(std::move(__labels)),
      offsets(std::move(__offsets))
  {}


  inline std::string
  to_string() const {
    std::ostringstream str;
    str << "labels (num_classes: " << num_classes <<
           ", num_labels: " << labels.size() <<
           ", num_examples: " <<
           (offsets.size() > 0 ? offsets.size() - 1 : 0) << ")";
    return str.str();
  }


  inline size_type
  num_labels(const size_type i) const { return offsets[i + 1] - offsets[i]; }


  template <typename Data>
  inline void
  move_front(
      const size_type i,
      Data* x
    ) const {
    size_type offset = offsets[i];
    size_type num_labels = offsets[i + 1] - offset;
    for (size_type j = 0; j < num_labels; ++j) {
      size_type label = labels[offset + j];
      std::swap(x[j], x[label]);
    }
  }


  template <typename Data>
  inline void
  move_front(
      const size_type i,
      Data* x,
      Data* y
    ) const {
    size_type offset = offsets[i];
    size_type num_labels = offsets[i + 1] - offset;
    for (size_type j = 0; j < num_labels; ++j) {
      size_type label = labels[offset + j];
      std::swap(x[j], x[label]);
      std::swap(y[j], y[label]);
    }
  }


  template <typename Data>
  inline void
  move_back(
      const size_type i,
      Data* x
    ) const {
    size_type offset = offsets[i];
    size_type num_labels = offsets[i + 1] - offset;
    for (size_type j = num_labels - 1;;) {
      size_type label = labels[offset + j];
      std::swap(x[j], x[label]);
      if (j-- == 0) break;
    }
  }
};


template <typename Iterator>
inline std::pair<Iterator, Iterator>
validate_labels(
    Iterator first,
    Iterator last
  ) {
  auto minmax = std::minmax_element(first, last);
  if (*minmax.first == 1) {
    std::for_each(first, last, [](size_type &x){ x -= 1; });
  } else if (*minmax.first != 0) {
    throw std::invalid_argument("Invalid class labels range.");
  }
  return minmax;
}


inline void
validate_labels_and_offsets(
    const size_type num_classes,
    const std::vector<size_type>& labels,
    const std::vector<size_type>& offsets
  ) {
  assert(num_classes > 0);
  if (offsets[0] != 0) {
    throw std::invalid_argument("The first offset must be 0.");
  }
  for (size_type i = 0; i < offsets.size() - 1; ++i) {
    size_type num_labels = offsets[i + 1] - offsets[i];
    if (num_labels < 1 || num_labels >= num_classes) {
      throw std::invalid_argument("Each example must have between "
                                  "1 and num_classes - 1 labels.");
    }
    auto first = &labels[offsets[i]];
    if (!std::is_sorted(first, first + num_labels,
                        std::less_equal<size_type>())) {
      throw std::invalid_argument("All labels for every example must be "
                                  "distinct and sorted.");
    }
  }
}


template <typename Iterator>
inline multiclass_output
make_output_multiclass(
    Iterator first,
    Iterator last
  ) {
  std::vector<size_type> v(first, last);
  auto minmax = validate_labels(v.begin(), v.end());
  return multiclass_output(*minmax.second + 1, v);
}


template <typename Iterator>
inline multilabel_output
make_output_multilabel(
    Iterator label_first,
    Iterator label_last,
    Iterator offset_first,
    Iterator offset_last
  ) {
  std::vector<size_type> v(label_first, label_last);
  std::vector<size_type> u(offset_first, offset_last);
  auto minmax = validate_labels(v.begin(), v.end());
  const size_type num_classes = *minmax.second + 1;
  validate_labels_and_offsets(num_classes, v, u);
  return multilabel_output(num_classes, v, u);
}


template <typename Type>
inline multilabel_output
make_output_multilabel(
    std::vector<std::vector<Type>>& labels
  ) {
  std::vector<size_type> v;
  std::vector<size_type> u;
  u.push_back(0); // zero offset
  std::for_each(labels.begin(), labels.end(), [&](std::vector<Type>& yi){
    v.insert(v.end(), yi.begin(), yi.end());
    u.push_back(u.back() + yi.size());
  });
  auto minmax = validate_labels(v.begin(), v.end());
  const size_type num_classes = *minmax.second + 1;
  validate_labels_and_offsets(num_classes, v, u);
  return multilabel_output(num_classes, v, u);
}

}

#endif
