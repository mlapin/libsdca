// Runs the MATLAB loss gateway (matsdca_loss) on the in-memory mx mock of
// mx_mock.h (no MATLAB needed): loss names and parameters, targets, precision,
// threads and errors. Built and run by tests/matlab/run_gateway_mock.sh.
#include "mex_loss.cpp"
#include "mx_mock.h"

#include <tuple>

namespace {

int failures = 0;

void check(const bool ok, const std::string& what) {
  std::cout << (ok ? "ok   " : "FAIL ") << what << '\n';
  if (!ok) ++failures;
}

struct result {
  std::vector<double> values, gradients;
  bool single = false;
};

result loss(mxArray* scores, mxArray* targets, mxArray* opts) {
  const mxArray* prhs[] = {scores, targets, opts};
  mxArray* plhs[2] = {nullptr, nullptr};
  mexFunction(2, plhs, opts ? 3 : 2, prhs);
  result r;
  const std::size_t n = plhs[0]->n, size = plhs[1]->m * plhs[1]->n;
  r.values.assign(mock::data<double>(plhs[0]), mock::data<double>(plhs[0]) + n);
  r.single = mxIsSingle(plhs[1]);
  for (std::size_t j = 0; j < size; ++j)
    r.gradients.push_back(r.single ? double(mock::data<float>(plhs[1])[j])
                                   : mock::data<double>(plhs[1])[j]);
  return r;
}

std::string loss_error(mxArray* scores, mxArray* targets, mxArray* opts) {
  try {
    loss(scores, targets, opts);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

// 4 classes, 2 examples (column-major).
const std::vector<double> z = {.3, -.8, 1.2, .1, -1, .4, .2, .5};
mxArray* scores(const mxClassID type = mxDOUBLE_CLASS) { return mock::array(4, 2, z, type); }
mxArray* labels() { return mock::array(1, 2, {3, 2}); }
mxArray* label_sets() { return mock::array(4, 2, {1, 1, 0, 0, 0, 1, 1, 0}, mxLOGICAL_CLASS); }

void every_loss() {
  struct named {
    const char* loss;
    std::map<std::string, mxArray*> extra;
  };
  const std::vector<named> cases = {
      {"cross_entropy", {}},
      {"topk_cross_entropy", {{"k", mock::scalar(2)}}},
      {"multiclass_hinge", {{"gamma", mock::scalar(.7)}}},
      {"topk_hinge_alpha", {{"k", mock::scalar(3)}}},
      {"topk_hinge_beta", {{"k", mock::scalar(2)}, {"gamma", mock::scalar(.7)}}},
      {"multilabel_hinge", {{"gamma", mock::scalar(.5)}}},
      {"multilabel_cross_entropy", {}},
      {"proximal_cross_entropy", {{"alpha", mock::scalar(1.2)}}},
      {"distillation", {{"temperature", mock::scalar(2)}}},
      {"proximal_distillation", {{"temperature", mock::scalar(2)}, {"alpha", mock::scalar(.5)}}}};
  for (const auto& c : cases) {
    std::map<std::string, mxArray*> fields = c.extra;
    fields["loss"] = mock::string(c.loss);
    const std::string name = c.loss;
    mxArray* targets = has(name, "multilabel")     ? label_sets()
                       : has(name, "distillation") ? mock::array(4, 2, {0, 1, -1, 2, 3, 0, 0, 1})
                                                   : labels();
    try {
      const result r = loss(scores(), targets, mock::options(fields));
      // Central differences of the values match the gradients.
      double error = 0;
      for (std::size_t j = 0; j < z.size(); ++j) {
        std::vector<double> plus = z, minus = z;
        plus[j] += 1e-6;
        minus[j] -= 1e-6;
        const double up =
            loss(mock::array(4, 2, plus), targets, mock::options(fields)).values[j / 4];
        const double down =
            loss(mock::array(4, 2, minus), targets, mock::options(fields)).values[j / 4];
        error = std::max(error, std::abs((up - down) / 2e-6 - r.gradients[j]));
      }
      check(r.values.size() == 2 && std::isfinite(r.values[0]) && error < 1e-6,
            "loss " + name + " (gradient error " + std::to_string(error) + ")");
    } catch (const std::exception& e) {
      check(false, "loss " + name + ": " + e.what());
    }
  }
}

void names_parameters_and_targets() {
  const std::vector<std::tuple<mxArray*, std::map<std::string, mxArray*>, std::string>> bad = {
      {labels(), {{"loss", mock::string("softmax")}}, "Unknown loss"},
      {labels(), {{"loss", mock::string("topk_hinge")}}, "Unknown loss"},
      {labels(), {{"objective", mock::string("cross_entropy")}}, "Unknown option 'objective'"},
      {labels(), {{"k", mock::scalar(1)}}, "takes no option 'k'"},
      {labels(), {{"gamma", mock::scalar(0)}}, "takes no option 'gamma'"},
      {labels(), {{"alpha", mock::scalar(1)}}, "takes no option 'alpha'"},
      {labels(), {{"temperature", mock::scalar(1)}}, "takes no option 'temperature'"},
      {labels(),
       {{"projection_precision", mock::string("float64")}},
       "takes no option 'projection_precision'"},
      {labels(), {{"loss", mock::string("topk_hinge_beta")}}, "needs opts.k"},
      {labels(), {{"loss", mock::string("topk_hinge_beta")}, {"k", mock::scalar(4)}}, "k must be"},
      {labels(),
       {{"loss", mock::string("proximal_cross_entropy")}, {"alpha", mock::scalar(0)}},
       "alpha"},
      {labels(), {{"precision", mock::string("single")}}, "Unknown precision"},
      {labels(), {{"precision", mock::string("float32")}}, "double data computes in float64"},
      {labels(), {{"num_threads", mock::scalar(0)}}, "num_threads"},
      {mock::array(1, 2, {0, 2}), {}, "Labels must be integers in 1:num_classes"},
      {mock::array(1, 2, {5, 2}), {}, "Labels must be integers in 1:num_classes"},
      {label_sets(), {}, "n labels"},
      {labels(), {{"loss", mock::string("multilabel_cross_entropy")}}, "label matrix"},
      {mock::array(4, 2, {1, 1, 1, 1, 0, 1, 1, 0}, mxLOGICAL_CLASS),
       {{"loss", mock::string("multilabel_hinge")}},
       "needs a negative class"},
      {mock::array(4, 2, {0, 0, 0, 0, 0, 1, 1, 0}, mxLOGICAL_CLASS),
       {{"loss", mock::string("multilabel_cross_entropy")}},
       "needs a label"},
      {labels(), {{"loss", mock::string("distillation")}}, "teacher logits"},
      {mock::array(4, 2, {0, 0, 0, 0, 0, 0, 0, NAN}),
       {{"loss", mock::string("distillation")}},
       "finite"}};
  for (const auto& b : bad) {
    const std::string error = loss_error(scores(), std::get<0>(b), mock::options(std::get<1>(b)));
    check(has(error, std::get<2>(b)), "rejects " + std::get<2>(b) + ": " + error);
  }
}

void precision_and_threads() {
  const result single = loss(scores(mxSINGLE_CLASS), labels(), nullptr);
  const result wide = loss(scores(mxSINGLE_CLASS), labels(),
                           mock::options({{"precision", mock::string("float64")}}));
  const result exact = loss(scores(), labels(), nullptr);
  check(single.single && wide.single && std::abs(single.values[0] - exact.values[0]) < 1e-6 &&
            std::abs(wide.values[0] - exact.values[0]) < 1e-7,
        "single scores: float32 by default, float64 on request, single gradients");
  // Equal student and teacher logits: an exact zero distillation gradient.
  const result same =
      loss(scores(), scores(), mock::options({{"loss", mock::string("distillation")}}));
  bool zero = true;
  for (double g : same.gradients) zero = zero && g == 0;
  check(zero, "distillation of equal logits has a zero gradient");
  // A huge proximal step: the envelope of cross-entropy nearly vanishes, without cancellation.
  const result flat = loss(mock::array(3, 1, {1e16, 0, 0}), mock::array(1, 1, {1}),
                           mock::options({{"loss", mock::string("proximal_cross_entropy")},
                                          {"alpha", mock::scalar(1e50)}}));
  check(flat.values[0] >= 0 && flat.values[0] < 1e-30,
        "proximal_cross_entropy at alpha 1e50: " + std::to_string(flat.values[0]));
  // Any thread count gives the results of one thread.
  std::vector<double> big(64 * 300), y(300);
  for (std::size_t j = 0; j < big.size(); ++j) big[j] = 5 * std::sin(double(j));
  for (std::size_t i = 0; i < y.size(); ++i) y[i] = double(i % 64 + 1);
  const result one = loss(mock::array(64, 300, big), mock::array(1, 300, y),
                          mock::options({{"loss", mock::string("topk_hinge_beta")},
                                         {"k", mock::scalar(3)},
                                         {"num_threads", mock::scalar(1)}}));
  const result many = loss(mock::array(64, 300, big), mock::array(1, 300, y),
                           mock::options({{"loss", mock::string("topk_hinge_beta")},
                                          {"k", mock::scalar(3)},
                                          {"num_threads", mock::scalar(7)}}));
  check(one.values == many.values && one.gradients == many.gradients,
        "num_threads does not change the results");
}

}  // namespace

int main() {
  mock::original_cout() = std::cout.rdbuf();
  every_loss();
  names_parameters_and_targets();
  precision_and_threads();
  std::cout << (failures == 0 ? "loss mock: ok" : "loss mock: FAILED") << '\n';
  return failures == 0 ? 0 : 1;
}
