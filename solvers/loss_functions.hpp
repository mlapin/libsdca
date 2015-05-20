#ifndef SDCA_LOSS_FUNCTIONS_HPP
#define SDCA_LOSS_FUNCTIONS_HPP

#include "common.hpp"

namespace sdca {

class TopKLoss {
public:
  const SizeType K;
  TopKLoss(SizeType k) : K(k) {}
};

}

#endif // SDCA_LOSS_FUNCTIONS_HPP
