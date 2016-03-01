#ifndef SDCA_SOLVER_OUTPUT_H
#define SDCA_SOLVER_OUTPUT_H

#include <stdexcept>

#include "sdca/solver/solverdef.h"

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
    str << "labels (" << num_classes << "-by-" << labels.size() << ")";
    return str.str();
  }

};


template <typename Iterator>
inline multiclass_output
make_output_multiclass(
    Iterator first,
    Iterator last
  ) {
  std::vector<size_type> v(first, last);
  auto minmax = std::minmax_element(v.begin(), v.end());
  if (*minmax.first == 1) {
    std::for_each(v.begin(), v.end(), [](size_type &x){ x -= 1; });
  } else if (*minmax.first != 0) {
    throw std::invalid_argument("Invalid class labels range.");
  }

  return multiclass_output(*minmax.second + 1, v);
}

}

#endif
