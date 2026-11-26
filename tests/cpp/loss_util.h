#ifndef SDCA_TEST_LOSS_UTIL_H
#define SDCA_TEST_LOSS_UTIL_H

#include <cstddef>
#include <vector>

#include "sdca/loss.h"

// The loss of the n scores z whose first p are the labels (label 0, or the label set [0, p))
// by sdca::loss_and_gradient, and its gradient (if not null): k is taken by the top-k losses,
// gamma by the hinges and step by proximal_cross_entropy. cross_entropy of p > 1 labels is
// multilabel_cross_entropy.
template <typename Data, typename Result = Data>
double labels_first_loss(sdca::kernel::Kind kind, const std::size_t n, const std::size_t p,
                         const Data* z, Data* gradient, const std::size_t k = 1,
                         const double gamma = 0, const double step = 1) {
  if (kind == sdca::kernel::cross_entropy && p > 1) kind = sdca::kernel::multilabel_cross_entropy;
  const sdca::kernel::Traits& takes = sdca::kernel::traits(kind);
  sdca::loss_parameters params;
  if (takes.takes_k) params.k = k;
  if (takes.proximal) params.gamma = step;
  if (takes.hinge) params.gamma = gamma;
  if (!takes.multilabel)
    return sdca::loss_and_gradient<Data, Result>(kind, n, z, 0, gradient, params);
  std::vector<std::size_t> labels(p);
  for (std::size_t j = 0; j < p; ++j) labels[j] = j;
  return sdca::loss_and_gradient<Data, Result>(kind, n, z, labels.data(), p, gradient, params);
}

#endif
