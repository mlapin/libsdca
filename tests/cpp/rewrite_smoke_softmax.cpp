// Smoke test of core::softmax_prox: the float and double lanes against the one-entry path of
// long double (the scalar C library's functions) on random rows of label and dense targets,
// for set_scores and set_inputs, cold and warm starts; and the solver's entropy steps on rows
// whose tails lie below the rounding of the normalizer or below the normal range, against
// high-precision references (mpmath, 90 digits; 500 where the duals are e^-800 of the scores);
// and the certificate's dual loss of float-stored duals off the zero sum.
//
//   c++ -std=c++11 -O2 -Wall -Wextra -Wshadow -Werror -Wdouble-promotion -Iinclude THIS -o smoke
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "sdca/core/project_softmax.h"
#include "sdca/solver.h"

namespace {

namespace core = sdca::SDCA_ISA_NAMESPACE::core;

struct row {
  std::vector<long double> v, p;
};

template <typename T>
struct result {
  std::vector<long double> r;
  long double value;
};

template <typename T>
result<T> solve(const row& x, const double alpha, const bool inputs, const bool warm) {
  const std::size_t n = x.v.size();
  std::vector<T> v(n), prior(n);
  for (std::size_t j = 0; j < n; ++j) v[j] = T(x.v[j]);
  core::softmax_prox<T> s(n, T(alpha));
  if (inputs) {
    T top = v[0];
    for (std::size_t j = 1; j < n; ++j) top = std::max(top, v[j]);
    long double norm = 0;
    for (std::size_t j = 0; j < n; ++j) norm += std::exp(static_cast<long double>(v[j] - top));
    const T log_norm = T(std::log(norm));
    s.set_inputs(top, log_norm, [&](std::size_t j, T& pj, T& log_p, T& log_q, T& ratio, T& q) {
      pj = T(x.p[j]);
      log_q = (v[j] - top) - log_norm;
      q = std::exp(log_q);
      log_p = pj > 0 ? std::log(pj) : -std::numeric_limits<T>::infinity();
      ratio = pj > 0 ? log_q - log_p : std::numeric_limits<T>::infinity();
    });
  } else {
    s.set_scores(v.data(), [&](std::size_t j) { return T(x.p[j]); });
  }
  if (warm) {
    s.solve();
    for (std::size_t j = 0; j < n; ++j) prior[j] = s.r[j] * T(0.9);
    s.solve(prior.data());
  } else {
    s.solve();
  }
  s.balance();
  result<T> out;
  out.r.assign(s.r, s.r + n);
  out.value = static_cast<long double>(s.value());
  return out;
}

struct error {
  double r, value;
  error() : r(0), value(0) {}
};

template <typename T>
void compare(const result<T>& got, const result<long double>& want, error& e) {
  for (std::size_t j = 0; j < want.r.size(); ++j)
    e.r = std::max(e.r, double(std::fabs(got.r[j] - want.r[j])));
  e.value = std::max(e.value, double(std::fabs(got.value - want.value) /
                                     std::max(std::fabs(want.value), 1e-300L)));
}

// One step of a solver objective from zero duals at scores z (labels first), against want.
template <typename Data, typename Step>
int pinned(const char* name, std::vector<Data> z, const std::vector<double>& want, double rtol,
           Step step) {
  std::vector<Data> a(z.size(), Data(0));
  step(a.data(), z.data());
  double worst = 0;
  for (std::size_t j = 0; j < z.size(); ++j)
    worst = std::max(worst, std::fabs(double(a[j]) - want[j]) / std::fabs(want[j]));
  const bool ok = worst <= rtol;
  std::printf("%-44s relative error %.1e %s\n", name, worst, ok ? "ok" : "FAILED");
  return ok ? 0 : 1;
}

int pinned_tails() {
  int failures = 0;
  // Two tied labels and a tail e^-40 below them: the tail's mass below the normalizer's
  // rounding, split evenly between the labels.
  const double c_tied[] = {1e-6, 1, 1e6};
  const double tied[][3] = {
      {1.0620885638228972e-24, 1.0620885638228972e-24, -2.1241771276457945e-24},
      {1.0620885638228972e-18, 1.0620885638228972e-18, -2.1241771276457944e-18},
      {1.0620885638195132e-12, 1.0620885638195132e-12, -2.1241771276390263e-12}};
  for (int i = 0; i < 3; ++i) {
    char name[64];
    std::snprintf(name, sizeof name, "[20, 20, -20] two labels, C = %g, double", c_tied[i]);
    const double c = c_tied[i];
    failures +=
        pinned<double>(name, {20, 20, -20}, std::vector<double>(tied[i], tied[i] + 3), 1e-14,
                       [c](double* a, double* z) {
                         sdca::l2_multilabel_cross_entropy<double, double>(c).update_dual_variables(
                             3, 2, 1.0, a, z);
                       });
  }
  failures +=
      pinned<float>("[20, 20, -20] two labels, C = 1, float", {20, 20, -20},
                    std::vector<double>(tied[1], tied[1] + 3), 1e-5, [](float* a, float* z) {
                      sdca::l2_multilabel_cross_entropy<float, float>(1.0f).update_dual_variables(
                          3, 2, 1.0f, a, z);
                    });
  // Labels 1e-10 apart, near the top: their difference below the rounding of the normalizer.
  const std::vector<double> shifted = {1.6666667728755232e-11, -1.6666665604578102e-11,
                                       -2.1241771275395856e-18};
  failures +=
      pinned<double>("[0, 1e-10, -40] two labels, C = 1, double", {0, 1e-10, -40}, shifted, 1e-9,
                     [](double* a, double* z) {
                       sdca::l2_multilabel_cross_entropy<double, double>(1.0).update_dual_variables(
                           3, 2, 1.0, a, z);
                     });
  failures +=
      pinned<float>("[0, 1e-10, -40] two labels, C = 1, float", {0, 1e-10f, -40}, shifted, 1e-5,
                    [](float* a, float* z) {
                      sdca::l2_multilabel_cross_entropy<float, float>(1.0f).update_dual_variables(
                          3, 2, 1.0f, a, z);
                    });
  // Two labels whose residuals are e^-800 of their probabilities, below a tail C e^-800 lifts
  // to the normal range: the tail's mass shared by the labels, not left to the pivot.
  failures += pinned<double>(
      "[400, 400, -400] two labels, C = 1e200, norm2 = 1e-200", {400, 400, -400},
      {9.169686460444218e-149, 9.169686460444218e-149, -1.8339372920888436e-148}, 1e-12,
      [](double* a, double* z) {
        sdca::l2_multilabel_cross_entropy<double, double>(1e200).update_dual_variables(3, 2, 1e-200,
                                                                                       a, z);
      });
  // A float dual e^-100, below float's normal range: subnormal (one ulp of it), not zero.
  failures += pinned<float>(
      "[100, 0] label 0, C = 1, float", {100, 0}, {3.720075976020836e-44, -3.720075976020836e-44},
      0.04, [](float* a, float* z) {
        sdca::l2_cross_entropy<float, float>(1.0f).update_dual_variables(2, 1.0f, a, z);
      });
  // C e^-margin normal where e^-margin is not.
  const double margins[] = {710, 745, 750, 1000};
  const double huge[][2] = {{4.47628622567513e-209, 4.47628622567513e-109},
                            {2.822350730471937e-224, 2.822350730471937e-124},
                            {1.9016849634750066e-226, 1.9016849634750064e-126},
                            {0, 5.075958897549457e-235}};
  const double cs[] = {1e100, 1e200};
  for (int i = 0; i < 4; ++i)
    for (int k = 0; k < 2; ++k) {
      if (huge[i][k] == 0) continue;
      char name[64];
      std::snprintf(name, sizeof name, "[%g, 0] label 0, C = %g, double", margins[i], cs[k]);
      const double c = cs[k];
      failures += pinned<double>(
          name, {margins[i], 0}, {huge[i][k], -huge[i][k]}, 3e-13, [c](double* a, double* z) {
            sdca::l2_cross_entropy<double, double>(c).update_dual_variables(2, 1.0, a, z);
          });
    }
  return failures;
}

// The certificate's dual loss of float-stored duals off the zero sum by 1.5e-8 and 2.2e-8,
// the label at probability 1.7e-8 (one label) and 8.5e-9 (two labels): the public
// projections (the excess or deficit at the labels) against mpmath, 50 digits. Restoring the
// zero sum at the likeliest class instead gives 0.44556013884721137 and 0.74266108407752464.
int pinned_certificate() {
  const double c = 0.7;
  const float stored_ce[] = {0.7f, -7e-13f, -0.233333333f, -0.466666667f};
  const float stored_ml[] = {0.35f, 0.1f, -0.15f, -0.3f};
  double ce[4], ml[4];
  for (int j = 0; j < 4; ++j) ce[j] = double(stored_ce[j]), ml[j] = double(stored_ml[j]);
  sdca::l2_cross_entropy<double, double>(c).feasible_dual(4, ce);
  sdca::l2_multilabel_cross_entropy<double, double>(c).feasible_dual(4, 2, ml);
  const double got[] = {sdca::l2_cross_entropy<double, double>(c).dual_loss(4, ce),
                        sdca::l2_multilabel_cross_entropy<double, double>(c).dual_loss(4, 2, ml)};
  const double want[] = {0.4455599178259651, 0.7426609694248376};
  const char* names[] = {"certificate, label 0, float storage", "certificate, two labels"};
  int failures = 0;
  for (int i = 0; i < 2; ++i) {
    const double err = std::fabs(got[i] - want[i]) / want[i];
    const bool ok = err <= 1e-14;
    std::printf("%-44s relative error %.1e %s\n", names[i], err, ok ? "ok" : "FAILED");
    failures += ok ? 0 : 1;
  }
  return failures;
}

}  // namespace

int main() {
  std::mt19937 gen(7);
  const std::size_t sizes[] = {2, 3, 10, 17, 64, 100, 1000};
  const double alphas[] = {0.01, 0.1, 1, 10, 100}, scales[] = {0.5, 2, 8};
  error ed, ef;
  int rows = 0;
  for (std::size_t n : sizes)
    for (double alpha : alphas)
      for (double scale : scales)
        for (int kind = 0; kind < 3; ++kind) {
          std::normal_distribution<double> nd(0, scale);
          row x;
          x.v.resize(n), x.p.assign(n, 0);
          for (auto& a : x.v) a = static_cast<long double>(nd(gen));
          if (kind == 0) {
            x.p[gen() % n] = 1;
          } else if (kind == 1) {
            const std::size_t k = 1 + gen() % std::max<std::size_t>(1, n / 2);
            for (std::size_t i = 0; i < k; ++i) x.p[i] = 1.0L / k;
          } else {
            long double total = 0;
            for (auto& a : x.p) total += (a = std::exp(static_cast<long double>(nd(gen))));
            for (auto& a : x.p) a /= total;
          }
          for (int mode = 0; mode < 3; ++mode) {
            const bool inputs = mode == 1, warm = mode == 2;
            const result<long double> want = solve<long double>(x, alpha, inputs, warm);
            compare(solve<double>(x, alpha, inputs, warm), want, ed);
            compare(solve<float>(x, alpha, inputs, warm), want, ef);
            ++rows;
          }
        }
  std::printf("%d rows: double |dr| %.1e, |dvalue|/value %.1e; float |dr| %.1e, %.1e\n", rows, ed.r,
              ed.value, ef.r, ef.value);
  const int failures = pinned_tails() + pinned_certificate();
  const bool ok =
      ed.r <= 1e-14 && ed.value <= 1e-13 && ef.r <= 1e-5 && ef.value <= 1e-4 && failures == 0;
  std::printf("%s\n", ok ? "ok" : "FAILED");
  return ok ? 0 : 1;
}
