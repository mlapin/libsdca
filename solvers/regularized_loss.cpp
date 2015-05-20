#include <algorithm>
#include <vector>

#include "math_util.hpp"
#include "regularized_loss.hpp"
#include "topk_simplex_biased_projector.hpp"

namespace sdca {

template <typename RealType>
class VariableDualUpdater<RealType, TopKLoss, L2Regularizer<RealType>> {
public:
  const TopKLoss &Loss;
  const L2Regularizer<RealType> &Regularizer;
  TopKSimplexBiasedProjector<RealType> Projector;

  VariableDualUpdater(
      const TopKLoss &loss,
      const L2Regularizer<RealType> &regularizer,
      const SizeType num_examples
    ) :
      Loss(loss),
      Regularizer(regularizer),
      Projector(loss.K, 1, static_cast<RealType>(1) /
        (static_cast<RealType>(num_examples) * regularizer.Lambda))
  {}

  void operator()(
      const SizeType num_tasks,
      const SizeType label,
      const RealType norm_squared,
      RealType *__restrict__ variables,
      std::vector<RealType> &scores) const {

    RealType a = Regularizer.Lambda / norm_squared;
    sdca_blas_axpby(static_cast<IndexType>(num_tasks), a, &scores[0],
      -Regularizer.Lambda, variables);

    // Place ground truth at the back
    RealType *variables_back = variables + num_tasks - 1;
    std::swap(*variables_back, variables[label]);
    a -= *variables_back;
    std::swap(scores.back(), scores[label]);
    scores.pop_back();

    // Project all but last one
    std::for_each(variables, variables_back, [&](RealType &x){ x += a; });
    Projector.Project(variables, variables_back, scores);

    // The last one is the sum
    *variables_back = std::accumulate(
      variables, variables_back, static_cast<RealType>(0));

    // Change sign of all but last one
    std::for_each(variables, variables_back, [](RealType &x){ x = -x; });

    // Revert ground truth index
    std::swap(*variables_back, variables[label]);
    scores.push_back(RealType());
  }
};


template <typename RealType>
class LossDualUpdater<RealType, TopKLoss, L2Regularizer<RealType>> {
public:
  const TopKLoss &Loss;
  const L2Regularizer<RealType> &Regularizer;
  const RealType KInverse;

  LossDualUpdater(
      const TopKLoss &loss,
      const L2Regularizer<RealType> &regularizer
    ) :
      Loss(loss),
      Regularizer(regularizer),
      KInverse(static_cast<RealType>(1) / static_cast<RealType>(loss.K))
  {}

  void operator()(
      const SizeType num_tasks,
      const SizeType label,
      const RealType *__restrict__ variables,
      std::vector<RealType> &scores,
      RealType &regularizer,
      RealType &primal_loss,
      RealType &dual_loss) const {

    regularizer += sdca_blas_dot(static_cast<IndexType>(num_tasks),
      &scores[0], variables);

    dual_loss += variables[label];

    RealType a = static_cast<RealType>(1) - scores[label];
    std::for_each(scores.begin(), scores.end(), [&](RealType &x){ x += a; });
    scores[label] = static_cast<RealType>(0);

    // Sum k largest elements
    std::nth_element(scores.begin(), scores.begin() + Loss.K - 1, scores.end(),
      std::greater<RealType>());
    a = std::accumulate(scores.begin(), scores.begin() + Loss.K,
      static_cast<RealType>(0));

    if (a > static_cast<RealType>(0)) {
      primal_loss += a * KInverse;
    }
  }
};


template <typename RealType>
class ObjectiveDualComputer<RealType, TopKLoss, L2Regularizer<RealType>> {
public:
  const TopKLoss &Loss;
  const L2Regularizer<RealType> &Regularizer;
  const RealType NInverse;
  const RealType LambdaHalf;

  ObjectiveDualComputer(
      const TopKLoss &loss,
      const L2Regularizer<RealType> &regularizer,
      const SizeType num_examples
    ) :
      Loss(loss),
      Regularizer(regularizer),
      NInverse(static_cast<RealType>(1) / static_cast<RealType>(num_examples)),
      LambdaHalf(static_cast<RealType>(0.5) * regularizer.Lambda)
  {}

  void operator()(
      const RealType regularizer,
      const RealType primal_loss,
      const RealType dual_loss,
      RealType &primal_objective,
      RealType &dual_objective) const {
    primal_objective = LambdaHalf * regularizer;
    dual_objective = Regularizer.Lambda * dual_loss - primal_objective;
    primal_objective += NInverse * primal_loss;
  }
};



template class VariableDualUpdater<float, TopKLoss, L2Regularizer<float>>;
template class VariableDualUpdater<double, TopKLoss, L2Regularizer<double>>;

template class LossDualUpdater<float, TopKLoss, L2Regularizer<float>>;
template class LossDualUpdater<double, TopKLoss, L2Regularizer<double>>;

template class ObjectiveDualComputer<float, TopKLoss, L2Regularizer<float>>;
template class ObjectiveDualComputer<double, TopKLoss, L2Regularizer<double>>;

}
