#ifndef SDCA_LOSS_KIND_H
#define SDCA_LOSS_KIND_H

namespace sdca {

// The losses by their shared names (docs/API.md), with the ids of the C ABI's
// SDCA_LOSS_*; the hinges are smoothed by gamma > 0.
enum class loss_kind {
  cross_entropy = 0,
  topk_cross_entropy = 1,
  multiclass_hinge = 2,
  topk_hinge_alpha = 3,
  topk_hinge_beta = 4,
  multilabel_hinge = 5,
  multilabel_cross_entropy = 6,
  proximal_cross_entropy = 7,
  distillation = 8,
  proximal_distillation = 9
};

// The shared name of a loss ("unknown" for an invalid value).
inline const char* loss_name(const loss_kind kind) {
  switch (kind) {
    case loss_kind::cross_entropy:
      return "cross_entropy";
    case loss_kind::topk_cross_entropy:
      return "topk_cross_entropy";
    case loss_kind::multiclass_hinge:
      return "multiclass_hinge";
    case loss_kind::topk_hinge_alpha:
      return "topk_hinge_alpha";
    case loss_kind::topk_hinge_beta:
      return "topk_hinge_beta";
    case loss_kind::multilabel_hinge:
      return "multilabel_hinge";
    case loss_kind::multilabel_cross_entropy:
      return "multilabel_cross_entropy";
    case loss_kind::proximal_cross_entropy:
      return "proximal_cross_entropy";
    case loss_kind::distillation:
      return "distillation";
    case loss_kind::proximal_distillation:
      return "proximal_distillation";
  }
  return "unknown";
}

}  // namespace sdca

#endif
