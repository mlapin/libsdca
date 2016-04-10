#ifndef SDCA_SOLVER_DATA_DATASET_H
#define SDCA_SOLVER_DATA_DATASET_H

#include <sstream>
#include <vector>
#include <utility>

#include "sdca/utility/types.h"

namespace sdca {

template <typename Input,
          typename Output,
          typename Evaluation>
struct dataset {
  typedef Input input_type;
  typedef Output output_type;
  typedef Evaluation eval_type;

  input_type in;
  output_type out;
  std::vector<eval_type> evals;


  dataset(
      Input&& __in,
      Output&& __out
    ) :
      in(std::move(__in)),
      out(std::move(__out))
  {}


  inline size_type num_dimensions() const { return in.num_dimensions; }

  inline size_type num_examples() const { return in.num_examples; }

  inline size_type num_classes() const { return out.num_classes; }


  inline std::string
  to_string() const {
    std::ostringstream str;
    str << in.to_string() << ", " << out.to_string();
    return str.str();
  }

};

}

#endif
