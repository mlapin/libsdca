#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_LOSS_PROXIMAL_H)
#if SDCA_ISA_ID == 1
#define SDCA_LOSS_PROXIMAL_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_LOSS_PROXIMAL_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_LOSS_PROXIMAL_H3 1
#else
#define SDCA_LOSS_PROXIMAL_H4 1
#endif

// proximal_cross_entropy rows (sdca/loss/row.h): the Moreau envelope with step gamma of the
// cross-entropy of a label y at the scores z, min_u CE(u, y) + |u - z|^2 / (2 gamma). Its
// optimum u = z + gamma (e_y - pi) has pi = softmax(u), the entropic proximal step
// core::softmax_prox toward the target e_y; the loss is -log pi_y + gamma / 2 |e_y - pi|^2 and
// the gradient pi - e_y.
//
// Masked classes (-inf) are removed (gradient +0.0): the step solves on the others, a row of
// the label alone has loss 0. A masked label beside a class left gives +inf with the limit of
// the gradient: -1 at the label and, at the others, pi - 0 of the step on them toward no
// target.
// A row of -inf only, NaN, +inf and labels out of range are invalid.
// Float work reruns in double where float does not hold gamma, the scores' spread or the
// gradient factor, and where its solve fails.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "sdca/core/lanes.h"
#include "sdca/core/project_softmax.h"
#include "sdca/core/reduce.h"
#include "sdca/core/storage.h"
#include "sdca/loss/row.h"

SDCA_ISA_BEGIN
namespace loss {
struct ProximalCrossEntropy {
  double gamma;
  ProximalCrossEntropy() : gamma(1) {}
  bool float_parameters() const { return core::normal_float(gamma); }

  template <typename W, typename Isa, typename S, typename G>
  double operator()(const RowArgs<S, G>& row, const Label& y, Scratch& s) const {
    typedef core::Lanes<W, Isa> A;
    const W inf = std::numeric_limits<W>::infinity();
    const bool float_work = sizeof(W) < sizeof(double);
    const S* z = row.scores;
    const std::size_t n = row.n;
    if (y.index < 0 || static_cast<uint64_t>(y.index) >= n) return Row::invalid();
    W lo, hi;
    bool valid = true;
    core::extrema<A>(z, n, lo, hi, valid);
    if (!valid || !(hi > -inf)) return Row::invalid();
    // A spread or a scale beyond float: double work.
    if (float_work &&
        !(core::fits_float(static_cast<double>(hi - lo)) && core::fits_float(row.scale))) {
      s.needs_double = true;
      return 0;
    }
    const std::size_t label = static_cast<std::size_t>(y.index);
    const bool masked = !(core::to_work<W>(z[label]) > -inf);
    W* v = s.buffer<W>(n, 0);
    const std::size_t m = core::gather_kept(z, n, v);
    std::size_t at = 0;
    for (std::size_t j = 0; j < label; ++j) at += core::to_work<W>(z[j]) > -inf;
    if (m == 1 && !masked) {
      if (row.grad)
        for (std::size_t j = 0; j < n; ++j) row.grad[j] = core::to_storage<G>(W(0));
      return 0;
    }
    try {
      // A masked label: the limit as z_y goes to -inf is the step on the classes left toward
      // no target, whose pi is that of the step toward the label's mass spread evenly over
      // them (with a uniform shift of their scores, which leaves the step unchanged): the
      // gradient pi = 1 / m - r there and -1 at the label.
      core::softmax_prox<W> prox(m, static_cast<W>(gamma));
      if (masked)
        prox.set_labels(v, m, [](std::size_t) { return true; });
      else
        prox.set_labels(v, 1, [at](std::size_t j) { return j == at; });
      prox.solve();
      prox.balance();
      const double value =
          masked ? std::numeric_limits<double>::infinity() : static_cast<double>(prox.value());
      if (!masked && !(value < std::numeric_limits<double>::infinity())) return Row::invalid();
      if (row.grad) scatter_gradient(row, prox, m, label, masked);
      return value;
    } catch (...) {
      // The float solve failing: double work.
      if (!float_work) return Row::invalid();
      s.needs_double = true;
      return 0;
    }
  }

 private:
  template <typename W, typename S, typename G>
  static void scatter_gradient(const RowArgs<S, G>& row, core::softmax_prox<W>& prox, std::size_t m,
                               std::size_t label, bool masked) {
    const W share = masked ? W(1) / static_cast<W>(m) : W(0);
    for (std::size_t k = 0; k < m; ++k) prox.r[k] -= share;
    core::scatter_kept(
        row.scores, row.n, prox.r, static_cast<W>(-row.scale),
        [label](std::size_t j) { return j == label ? W(1) : W(0); }, row.grad);
  }
};

}  // namespace loss
SDCA_ISA_END

#endif  // SDCA_LOSS_PROXIMAL_H
