#ifndef SDCA_REGULARIZERS_HPP
#define SDCA_REGULARIZERS_HPP

#include "common.hpp"

namespace sdca {

template <typename RealType = double>
class L2Regularizer {
public:
  const RealType Lambda;
  const RealType LambdaInverse;
  L2Regularizer(RealType lambda) :
    Lambda(lambda),
    LambdaInverse(static_cast<RealType>(1)/lambda)
  {}
};

}

#endif // SDCA_REGULARIZERS_HPP
