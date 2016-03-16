#ifndef SDCA_SOLVER_DATASET_H
#define SDCA_SOLVER_DATASET_H

#include "sdca/solver/input.h"
#include "sdca/solver/output.h"
#include "sdca/solver/eval/types.h"

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


template <typename Result = double,
          typename Data,
          template <typename> class Input,
          typename Output>
inline dataset<Input<Data>, Output, eval_train<Result, Output>>
make_dataset_train(
    Input<Data>&& in,
    Output&& out
  ) {
  return dataset<Input<Data>, Output, eval_train<Result, Output>>(
    std::move(in), std::move(out));
}


template <typename Result = double,
          typename Data,
          template <typename> class Input,
          typename Output>
inline dataset<Input<Data>, Output, eval_test<Result, Output>>
make_dataset_test(
    Input<Data>&& in,
    Output&& out
  ) {
  return dataset<Input<Data>, Output, eval_test<Result, Output>>(
    std::move(in), std::move(out));
}

}

#endif
