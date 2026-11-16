// Runs the MATLAB training gateway (matsdca_fit) end to end on the in-memory mx
// mock of mx_mock.h (no MATLAB needed): options and loss names, 1-based labels,
// label sets, warm starts, precision, the model struct, and errors.
// Built and run by tests/matlab/run_gateway_mock.sh.
#include "mex_solve.cpp"
#include "mx_mock.h"

namespace {

int failures = 0;

void check(const bool ok, const std::string& what) {
  std::cout << (ok ? "ok   " : "FAIL ") << what << '\n';
  if (!ok) ++failures;
}

mxArray* fit(mxArray* x, mxArray* y, mxArray* opts) {
  const mxArray* prhs[] = {x, y, opts};
  mxArray* plhs[1] = {nullptr};
  mexFunction(1, plhs, opts ? 3 : 2, prhs);
  return plhs[0];
}

// The error message of a fit, or "" if it succeeds.
std::string fit_error(mxArray* x, mxArray* y, mxArray* opts) {
  try {
    fit(x, y, opts);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

double number(const mxArray* model, const char* name) {
  const mxArray* field = mxGetField(model, 0, name);
  return field ? mxGetScalar(field) : std::nan("");
}

std::string text(const mxArray* model, const char* name) {
  const mxArray* field = mxGetField(model, 0, name);
  return field && mxIsChar(field) ? field->text : "";
}

// X: d = 2 features, n = 3 examples (column-major); Y: 1-based labels.
mxArray* features(const mxClassID type = mxDOUBLE_CLASS) {
  return mock::array(2, 3, {1, 0, 0, 1, -1, -1}, type);
}
mxArray* labels() { return mock::array(1, 3, {1, 2, 3}); }

void losses_and_model() {
  const std::string statuses = " solved no_progress max_epoch failed ";
  struct named {
    const char* loss;
    std::map<std::string, mxArray*> extra;
  };
  const std::vector<named> cases = {
      {"cross_entropy", {}},
      {"topk_cross_entropy", {{"k", mock::scalar(2)}}},
      {"multiclass_hinge", {}},
      {"multiclass_hinge", {{"gamma", mock::scalar(.5)}}},
      {"topk_hinge_alpha", {{"k", mock::scalar(2)}}},
      {"topk_hinge_beta", {{"k", mock::scalar(2)}, {"gamma", mock::scalar(.7)}}},
      {"multilabel_hinge", {{"projection_precision", mock::string("extended")}}},
      {"multilabel_cross_entropy", {}}};
  for (const auto& c : cases) {
    std::map<std::string, mxArray*> fields = c.extra;
    fields["loss"] = mock::string(c.loss);
    fields["tol"] = mock::scalar(1e-8);
    const bool multilabel = has(c.loss, "multilabel");
    mxArray* y = multilabel ? mock::sparse(3, 3, {{0, 0}, {1, 0}, {1, 1}, {2, 2}}) : labels();
    try {
      const mxArray* model = fit(features(), y, mock::options(fields));
      const mxArray* W = mxGetField(model, 0, "W");
      const mxArray* A = mxGetField(model, 0, "A");
      const mxArray* history = mxGetField(model, 0, "history");
      // W = X A' and every dual column sums to zero.
      double error = 0;
      const double* w = mock::data<double>(W);
      const double* a = mock::data<double>(A);
      const double x[] = {1, 0, 0, 1, -1, -1};
      for (int j = 0; j < 3; ++j) {
        double sum = 0;
        for (int l = 0; l < 3; ++l) sum += a[l + 3 * j];
        error = std::max(error, std::abs(sum));
        for (int l = 0; l < 2; ++l) {
          double expected = 0;
          for (int i = 0; i < 3; ++i) expected += x[l + 2 * i] * a[j + 3 * i];
          error = std::max(error, std::abs(w[l + 2 * j] - expected));
        }
      }
      std::string what = std::string(c.loss) + (c.extra.count("gamma") ? " smooth" : "");
      check(text(model, "loss") == c.loss && text(model, "status") == "solved" &&
                number(model, "C") == 1 && number(model, "num_classes") == 3 &&
                number(model, "num_features") == 2 && number(model, "epochs") >= 1 &&
                number(model, "seed") == 0 && text(model, "precision") == "float64" &&
                (mxGetField(model, 0, "k") != nullptr) == (c.extra.count("k") > 0) &&
                (mxGetField(model, 0, "gamma") != nullptr) == has(c.loss, "hinge") &&
                (mxGetField(model, 0, "shrinking") != nullptr) == has(c.loss, "hinge") &&
                (mxGetField(model, 0, "projection_precision") != nullptr) ==
                    (std::string(c.loss) == "multilabel_hinge") &&
                mxGetField(model, 0, "num_threads") == nullptr && error < 1e-14 &&
                history->m >= 1 && mxGetField(history, history->m - 1, "accuracy") &&
                mxGetField(history, 0, "regularizer") && mxGetField(history, 0, "cpu_time"),
            "fit " + what + " model");
      check(has(statuses, " " + text(model, "status") + " "), "status of " + what);
    } catch (const std::exception& e) {
      check(false, std::string("fit ") + c.loss + ": " + e.what());
    }
  }
}

// Only the shared loss and option names, each loss with only its own parameters.
void names_and_options() {
  const std::vector<std::pair<std::map<std::string, mxArray*>, std::string>> bad = {
      {{{"loss", mock::string("softmax")}}, "Unknown loss"},
      {{{"loss", mock::string("msvm")}}, "Unknown loss"},
      {{{"loss", mock::string("l2_topk_hinge")}}, "Unknown loss"},
      {{{"loss", mock::string("topk_hinge")}}, "Unknown loss"},
      {{{"loss", mock::string("mlsoftmax")}}, "Unknown loss"},
      {{{"loss", mock::string("proximal_cross_entropy")}}, "Unknown loss"},
      {{{"objective", mock::string("cross_entropy")}}, "Unknown option 'objective'"},
      {{{"c", mock::scalar(1)}}, "Unknown option 'c'"},
      {{{"epsilon", mock::scalar(1e-3)}}, "Unknown option 'epsilon'"},
      {{{"is_dual", mock::scalar(0)}}, "Unknown option 'is_dual'"},
      {{{"W0", mock::array(2, 3, {})}}, "Unknown option 'W0'"},
      {{{"A", mock::array(3, 3, {})}}, "Unknown option 'A'"},
      {{{"W", mock::array(2, 3, {})}}, "Unknown option 'W'"},
      {{{"max_cpu_time", mock::scalar(1)}}, "Unknown option 'max_cpu_time'"},
      {{{"k", mock::scalar(1)}}, "takes no option 'k'"},
      {{{"gamma", mock::scalar(0)}}, "takes no option 'gamma'"},
      {{{"shrinking", mock::string("off")}}, "takes no option 'shrinking'"},
      {{{"loss", mock::string("multiclass_hinge")}, {"k", mock::scalar(1)}}, "takes no option 'k'"},
      {{{"loss", mock::string("multiclass_hinge")},
        {"projection_precision", mock::string("extended")}},
       "takes no option 'projection_precision'"},
      {{{"loss", mock::string("topk_hinge_beta")}}, "needs opts.k"},
      {{{"loss", mock::string("topk_hinge_beta")}, {"k", mock::scalar(3)}}, "k must be"},
      {{{"loss", mock::string("topk_hinge_beta")}, {"k", mock::scalar(0)}}, "k must be"},
      {{{"loss", mock::string("topk_hinge_beta")}, {"k", mock::scalar(1.5)}}, "k is outside"},
      {{{"loss", mock::string("multiclass_hinge")}, {"gamma", mock::scalar(-1)}}, "gamma"},
      {{{"C", mock::scalar(0)}}, "C must be"},
      {{{"tol", mock::scalar(-1)}}, "tol must be"},
      {{{"precision", mock::string("single")}}, "Unknown precision"},
      {{{"precision", mock::string("double")}}, "Unknown precision"},
      {{{"precision", mock::string("float32")}}, "double data computes in float64"},
      {{{"num_threads", mock::scalar(0)}}, "num_threads"},
      {{{"seed", mock::scalar(-1)}}, "seed is outside"},
      {{{"num_classes", mock::scalar(1)}}, "num_classes"},
      {{{"num_classes", mock::scalar(2)}}, "Labels must be integers"},
      {{{"log_level", mock::string("loud")}}, "log_level"},
      {{{"shrinking", mock::string("on")}, {"loss", mock::string("multiclass_hinge")}},
       "Unknown shrinking"}};
  for (const auto& b : bad) {
    const std::string error = fit_error(features(), labels(), mock::options(b.first));
    check(has(error, b.second), "rejects " + b.second + ": " + error);
  }
}

// Labels are 1-based integers; num_classes keeps classes without examples.
void labels_and_classes() {
  for (const std::vector<double>& y :
       {std::vector<double>{0, 1, 2}, {1, 2, 2.5}, {1, 2, -1}, {1, 2, NAN}}) {
    const std::string error = fit_error(features(), mock::array(1, 3, y), mock::options({}));
    check(has(error, "Labels must be integers in 1:num_classes"), "label rejected: " + error);
  }
  const std::string single = fit_error(features(), mock::array(1, 3, {1, 1, 1}), nullptr);
  check(has(single, "num_classes >= 2"), "one class rejected: " + single);
  const std::string sets =
      fit_error(features(), mock::sparse(3, 3, {{0, 0}, {1, 1}, {2, 2}}), nullptr);
  check(has(sets, "double vector"), "label sets for cross_entropy rejected: " + sets);
  const std::string vector =
      fit_error(features(), labels(), mock::options({{"loss", mock::string("multilabel_hinge")}}));
  check(has(vector, "sparse or logical"), "labels for multilabel_hinge rejected: " + vector);
  const mxArray* model =
      fit(features(), labels(), mock::options({{"num_classes", mock::scalar(5)}}));
  check(number(model, "num_classes") == 5 && mxGetField(model, 0, "W")->n == 5 &&
            mxGetField(model, 0, "A")->m == 5,
        "num_classes 5 keeps 5 classes");
}

// Label sets: sparse double or logical with spare capacity, or full logical; the rows
// are the classes, so classes without positives stay.
void label_sets() {
  const std::vector<std::pair<mwIndex, mwIndex>> entries = {{1, 0}, {2, 1}, {1, 2}};
  mxArray* full = mock::array(4, 3, {0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0}, mxLOGICAL_CLASS);
  std::vector<std::vector<double>> weights;
  for (mxArray* y :
       {mock::sparse(4, 3, entries), mock::sparse(4, 3, entries, mxLOGICAL_CLASS), full}) {
    const mxArray* model =
        fit(features(), y, mock::options({{"loss", mock::string("multilabel_cross_entropy")}}));
    const double* w = mock::data<double>(mxGetField(model, 0, "W"));
    weights.emplace_back(w, w + 8);
    check(number(model, "num_classes") == 4, "label sets keep 4 classes");
  }
  check(weights[0] == weights[1] && weights[0] == weights[2], "label set formats train alike");
  const std::string rows = fit_error(features(), mock::sparse(4, 3, entries),
                                     mock::options({{"loss", mock::string("multilabel_hinge")},
                                                    {"num_classes", mock::scalar(3)}}));
  check(has(rows, "num_classes rows"), "num_classes must match the rows: " + rows);
}

// Warm start: W = center + X A' from feasible duals; infeasible duals are rejected.
void warm_starts() {
  const double x[] = {1, 0, 0, 1, -1, -1};
  const std::vector<double> center = {.5, .3, -1, .1, .2, -.4};
  for (bool centered : {false, true}) {
    std::map<std::string, mxArray*> cold = {{"tol", mock::scalar(1e-10)},
                                            {"eval_epoch", mock::scalar(1)}};
    if (centered) cold["center"] = mock::array(2, 3, center);
    const mxArray* first = fit(features(), labels(), mock::options(cold));
    std::map<std::string, mxArray*> warm = cold;
    warm["initial_dual"] = mxDuplicateArray(mxGetField(first, 0, "A"));
    warm["max_epoch"] = mock::scalar(0);
    warm["eval_on_start"] = mock::scalar(1);
    const mxArray* model = fit(features(), labels(), mock::options(warm));
    const double* w = mock::data<double>(mxGetField(model, 0, "W"));
    const double* a = mock::data<double>(mxGetField(first, 0, "A"));
    double error = 0;
    for (int j = 0; j < 3; ++j)
      for (int l = 0; l < 2; ++l) {
        double expected = centered ? center[l + 2 * j] : 0;
        for (int i = 0; i < 3; ++i) expected += x[l + 2 * i] * a[j + 3 * i];
        error = std::max(error, std::abs(w[l + 2 * j] - expected));
      }
    const std::string what = centered ? " (centered)" : "";
    // The duals of a solved fit start at a gap within its tolerance.
    const double gap = number(mxGetField(model, 0, "history"), "relative_gap");
    check(error <= 1e-15 && gap <= 1e-10 && number(model, "epochs") == 0,
          "warm start W = center + X A' at the solved gap" + what);
  }
  const std::vector<std::pair<std::vector<double>, std::string>> bad = {
      {{-1, .5, .5, 0, 0, 0, 0, 0, 0}, "non-label entries must be nonpositive"},
      {{-.5, .5, .1, 0, 0, 0, 0, 0, 0}, "sum to zero"},
      {{NAN, 0, 0, 0, 0, 0, 0, 0, 0}, "finite"}};
  for (const auto& b : bad) {
    const std::string error = fit_error(
        features(), labels(), mock::options({{"initial_dual", mock::array(3, 3, b.first)}}));
    check(has(error, b.second), "initial_dual rejected: " + error);
  }
  const std::string shape =
      fit_error(features(), labels(), mock::options({{"initial_dual", mock::array(2, 3, {})}}));
  check(!shape.empty(), "initial_dual of the wrong shape rejected: " + shape);
}

// precision follows the class of X; single X may take float64 work; eval_epoch 0, seed.
void precision_and_schedule() {
  const mxArray* single = fit(features(mxSINGLE_CLASS), labels(), nullptr);
  check(text(single, "precision") == "float32" && mxIsSingle(mxGetField(single, 0, "W")),
        "single X computes in float32");
  const mxArray* wide = fit(features(mxSINGLE_CLASS), labels(),
                            mock::options({{"precision", mock::string("float64")}}));
  check(text(wide, "precision") == "float64" && mxIsSingle(mxGetField(wide, 0, "A")),
        "single X with precision float64 keeps single outputs");
  const mxArray* end = fit(features(), labels(),
                           mock::options({{"eval_epoch", mock::scalar(0)},
                                          {"max_epoch", mock::scalar(5)},
                                          {"tol", mock::scalar(0)}}));
  check(number(end, "epochs") == 5 && mxGetField(end, 0, "history")->m == 1 &&
            text(end, "status") == "max_epoch",
        "eval_epoch 0 evaluates once, at the end");
  const mxArray* seeded =
      fit(features(), labels(),
          mock::options({{"seed", mock::scalar(7)}, {"num_threads", mock::scalar(2)}}));
  check(number(seeded, "seed") == 7 && number(seeded, "num_threads") == 2 &&
            text(seeded, "status") == "solved",
        "seed and num_threads are echoed");
  const mxArray* kernel = fit(mock::array(3, 3, {1, 0, -1, 0, 1, -1, -1, -1, 2}), labels(),
                              mock::options({{"kernel", mock::scalar(1)}}));
  check(mxGetField(kernel, 0, "W") == nullptr && mxGetField(kernel, 0, "num_features") == nullptr &&
            mxGetScalar(mxGetField(kernel, 0, "kernel")) == 1 && text(kernel, "status") == "solved",
        "Gram matrix input: A only");
  const std::string centered =
      fit_error(mock::array(3, 3, {1, 0, -1, 0, 1, -1, -1, -1, 2}), labels(),
                mock::options({{"kernel", mock::scalar(1)}, {"center", mock::array(3, 3, {})}}));
  check(has(centered, "center"), "center rejected for a Gram matrix: " + centered);
}

// matsdca_fit rejects nonfinite data before it trains.
void nonfinite_data() {
  for (double bad : {NAN, INFINITY}) {
    const std::string error =
        fit_error(mock::array(2, 3, {1, 0, 0, bad, -1, -1}), labels(), nullptr);
    check(has(error, "X must be finite"), "data with nonfinite entries: " + error);
  }
}

// matsdca_fit restores std::cout (buffer and format) before it raises an error, for an
// unsupported data type and for an exception from training.
void cout_restored_on_error() {
  for (mxArray* data : {mock::array(2, 3, {1, 0, 0, 1, -1, -1}, mxINT32_CLASS),
                        mock::array(2, 3, {1, 0, 0, NAN, -1, -1})}) {
    const std::string what = mxIsDouble(data) ? "nonfinite data" : "int32 data";
    const std::ios::fmtflags flags = std::cout.flags();
    const std::streamsize precision = std::cout.precision();
    const int before = mock::errors_with_cout_redirected();
    const bool raised = !fit_error(data, labels(), mock::options({})).empty();
    check(raised && mock::errors_with_cout_redirected() == before &&
              std::cout.rdbuf() == mock::original_cout() && std::cout.flags() == flags &&
              std::cout.precision() == precision,
          "std::cout restored before the " + what + " error");
  }
}

}  // namespace

int main() {
  mock::original_cout() = std::cout.rdbuf();
  losses_and_model();
  names_and_options();
  labels_and_classes();
  label_sets();
  warm_starts();
  precision_and_schedule();
  nonfinite_data();
  cout_restored_on_error();
  std::cout << (failures == 0 ? "gateway mock: ok" : "gateway mock: FAILED") << '\n';
  return failures == 0 ? 0 : 1;
}
