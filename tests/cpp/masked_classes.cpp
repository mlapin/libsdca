// Masked classes (scores of -inf; docs/API.md, Masked classes): the rows of
// masked_cases.h (tests/python/masked_cases.py) through the C++ loss functions and
// the C ABI (the rows of probability targets: the C ABI only), in float and double storage
// and work. A valid row gives its expected
// loss and gradient: its exact entries (zeros, +0.0, and infinities) to the byte, the others
// to a tolerance. An invalid one is an invalid row (docs/API.md, Errors): the C ABI returns
// SDCA_INVALID_ROWS with the count and the first invalid row in its message, the C++
// functions throw std::invalid_argument; invalid target probabilities are invalid rows of the
// C ABI too. With check_rows 1 the C ABI gives the same message as an argument error and
// leaves the outputs as given.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "masked_cases.h"
#include "tracked.h"
#include "sdca/c_api.h"
#include "sdca/loss.h"

namespace {

tracked::counts counts;

void fail(const sdca_masked_case& c, const std::string& path, const std::string& what) {
  ++counts.failures;
  std::fprintf(stderr, "FAIL %s (%s): %s\n", c.name, path.c_str(), what.c_str());
}

// The rows whose expectation another owner decides (tracked.h).
const char* const huge_temperature_reason =
    "[core] the limit of a masked distillation row where 2000 T overflows (now: loss log 2)";

const char* const probability_targets_reason =
    "[ce] the C ABI takes loss probabilities as given (as PyTorch); the table's argument "
    "error waits for the kernels' target check (loss/ce.h ProbabilityWeights::valid)";

// The rows of probability_targets_reason: the table expects the argument error that
// sdca_loss_evaluate does not give.
const char* probability_targets_tracked(const sdca_masked_case& c) {
  return std::string(c.error) == "Target probabilities must be finite and nonnegative"
             ? probability_targets_reason
             : nullptr;
}

// A row's outcome: what failed ("" if it passes), and the reason of a tracked row (null if
// it is not tracked).
void settle(const std::string& path, const std::string& what, const char* reason) {
  tracked::settle(counts, path, what, reason);
}

bool distillation(const sdca_masked_case& c) {
  return c.loss == SDCA_LOSS_DISTILLATION || c.loss == SDCA_LOSS_PROXIMAL_DISTILLATION;
}

// Whether a computed entry is the expected one: by bytes where that is exact (a zero, with
// its sign, or an infinity), else within tol relative to 1 + its size.
bool same(double actual, double expected, double tol) {
  if (expected == 0 || std::isinf(expected))
    return actual == expected && std::signbit(actual) == std::signbit(expected);
  return std::fabs(actual - expected) <= tol * (1 + std::fabs(expected));
}

// A computed loss and gradient against the row's expected ones.
template <typename Data>
void compare(const sdca_masked_case& c, const std::string& path, double value,
             const std::vector<Data>& gradient, double tol) {
  if (!same(value, c.value, tol)) fail(c, path, "loss " + std::to_string(value));
  for (int j = 0; j < c.classes; ++j) {
    const double g = double(gradient[j]);
    if (!same(g, c.gradient[j], tol))
      fail(c, path,
           "gradient entry " + std::to_string(j) + ": " + std::to_string(g) +
               (g == 0 && std::signbit(g) ? " (-0.0)" : ""));
  }
}

// sdca::loss_and_gradient and the distillation functions of Data scores in Result work.
template <typename Data, typename Result>
void cpp_row(const sdca_masked_case& c, const std::string& path, double tol) {
  if (c.probabilities) return;
  const std::size_t n = std::size_t(c.classes);
  std::vector<Data> z(c.logits, c.logits + n), teacher(c.teacher, c.teacher + n), g(n, Data(7));
  double value = 0;
  bool rejected = false;
  try {
    if (c.loss == SDCA_LOSS_DISTILLATION) {
      value = sdca::distillation_loss<Data, Result>(n, z.data(), teacher.data(), c.temperature,
                                                    g.data());
    } else if (c.loss == SDCA_LOSS_PROXIMAL_DISTILLATION) {
      value = sdca::proximal_distillation_loss<Data, Result>(n, z.data(), teacher.data(),
                                                             c.temperature, c.alpha, g.data());
    } else {
      sdca::loss_parameters p;
      p.k = std::size_t(c.k);
      p.gamma = c.gamma;
      if (c.loss == SDCA_LOSS_PROXIMAL_CROSS_ENTROPY) p.gamma = c.alpha;
      const sdca::kernel::Kind kind = static_cast<sdca::kernel::Kind>(c.loss);
      if (c.label >= 0) {
        value = sdca::loss_and_gradient<Data, Result>(kind, n, z.data(), std::size_t(c.label),
                                                      g.data(), p);
      } else {
        std::vector<std::size_t> labels;
        for (std::size_t j = 0; j < n; ++j)
          if (c.positive[j]) labels.push_back(j);
        value = sdca::loss_and_gradient<Data, Result>(kind, n, z.data(), labels.data(),
                                                      labels.size(), g.data(), p);
      }
    }
  } catch (const std::invalid_argument& error) {
    rejected = true;
    if (*c.error && error.what() != std::string(c.error))
      fail(c, path, std::string("error: ") + error.what());
  }
  if (rejected != !c.valid) return fail(c, path, rejected ? "rejected" : "not rejected");
  if (c.valid) compare(c, path, value, g, tol);
}

// The C ABI's report of a call with count invalid rows of rows, the first at first: the row's
// argument error, or SDCA_INVALID_ROWS and a message that starts with the count and the
// first row.
bool reports_invalid(const sdca_masked_case& c, int status, std::size_t count, std::size_t rows,
                     std::size_t first) {
  const std::string message = sdca_last_error();
  if (*c.error) return status == SDCA_INVALID_ARGUMENT && message == c.error;
  const std::string prefix = std::to_string(count) + " of " + std::to_string(rows) +
                             " rows are invalid (first: row " + std::to_string(first) + "): ";
  return status == SDCA_INVALID_ROWS && message.compare(0, prefix.size(), prefix) == 0;
}

// sdca_loss_evaluate of two copies of the row (rows of a call share nothing).
template <typename Data>
void c_row(const sdca_masked_case& c, int dtype, int precision, const std::string& path,
           double tol) {
  const std::size_t n = std::size_t(c.classes);
  std::vector<Data> z, teacher, g(2 * n, Data(7));
  std::vector<double> probabilities;
  std::vector<std::uint8_t> mask;
  for (int copy = 0; copy < 2; ++copy) {
    z.insert(z.end(), c.logits, c.logits + n);
    teacher.insert(teacher.end(), c.teacher, c.teacher + n);
    probabilities.insert(probabilities.end(), c.teacher, c.teacher + n);
    mask.insert(mask.end(), c.positive, c.positive + n);
  }
  const std::int64_t labels[2] = {c.label, c.label};
  double values[2] = {7, 7};
  sdca_loss loss = SDCA_LOSS_INIT(c.loss);
  loss.k = std::size_t(c.k);
  loss.gamma = c.gamma;
  if (c.loss == SDCA_LOSS_PROXIMAL_CROSS_ENTROPY || c.loss == SDCA_LOSS_PROXIMAL_DISTILLATION)
    loss.gamma = c.alpha;
  loss.temperature = c.temperature;
  sdca_execution execution = SDCA_EXECUTION_INIT;
  execution.precision = precision;
  sdca_targets targets = SDCA_TARGETS_INIT;
  if (distillation(c)) {
    targets.kind = SDCA_TARGET_LOGITS;
    targets.logits = teacher.data();
  } else if (c.probabilities) {
    targets.kind = SDCA_TARGET_PROBABILITIES;
    targets.probabilities = probabilities.data();
  } else if (c.label >= 0) {
    targets.labels = labels;
  } else {
    targets.kind = SDCA_TARGET_MASK;
    targets.mask = mask.data();
  }
  const int status =
      sdca_loss_evaluate(&loss, &execution, dtype, 2, n, z.data(), &targets, values, g.data());
  if (!c.valid) {
    const bool reported = reports_invalid(c, status, 2, 2, 0);
    if (const char* reason = probability_targets_tracked(c))
      return settle(std::string(c.name) + " (" + path + ")",
                    reported ? "" : "returned " + std::to_string(status), reason);
    if (!reported) fail(c, path, "returned " + std::to_string(status) + ": " + sdca_last_error());
    const std::string error = sdca_last_error();
    std::fill(g.begin(), g.end(), Data(7));
    values[0] = values[1] = 7;
    execution.check_rows = 1;
    if (sdca_loss_evaluate(&loss, &execution, dtype, 2, n, z.data(), &targets, values, g.data()) !=
            SDCA_INVALID_ARGUMENT ||
        error != sdca_last_error())
      return fail(c, path, "check_rows 1 gave another error: " + std::string(sdca_last_error()));
    for (std::size_t j = 0; j < 2 * n; ++j)
      if (g[j] != Data(7) || values[j / n] != 7) return fail(c, path, "wrote an output");
    return;
  }
  if (status != SDCA_OK) return fail(c, path, std::string("failed: ") + sdca_last_error());
  for (int copy = 0; copy < 2; ++copy) {
    const std::vector<Data> row(g.begin() + copy * std::ptrdiff_t(n),
                                g.begin() + (copy + 1) * std::ptrdiff_t(n));
    compare(c, path, values[copy], row, tol);
  }
  // The losses alone give the same values; distillation in float32 work takes the kernel that
  // saves its rows' gradient when it takes the gradient (as the PyTorch layer), whose losses
  // agree with the losses alone to within the work's rounding.
  double alone[2] = {7, 7};
  const bool grad_kernel = c.loss == SDCA_LOSS_DISTILLATION && dtype == SDCA_DTYPE_FLOAT32 &&
                           precision != SDCA_PRECISION_FLOAT64;
  auto agree = [&](double a, double b) {
    return a == b || (grad_kernel && std::fabs(a - b) <= tol * (1 + std::fabs(b)));
  };
  const bool differ = sdca_loss_evaluate(&loss, &execution, dtype, 2, n, z.data(), &targets, alone,
                                         nullptr) != SDCA_OK ||
                      !(agree(alone[0], values[0]) && agree(alone[1], values[1]));
  settle(std::string(c.name) + " (" + path + ")", differ ? "the losses alone differ" : "", nullptr);
}

// How c_row_among surrounds the row: with finite rows, with copies of itself, or with valid
// rows of a masked class.
enum class among { finite, itself, masked };

// The row at row 5 of 16, among rows its scores and teacher logits with every nonfinite
// value at -3 (finite), copies of the row (itself), or those finite rows with a class
// masked that no row takes as a label (masked): the call checks every row from its pattern
// and then writes its outputs in place. An invalid row leaves every output as given; a
// valid one has its loss and gradient, and the other rows those of a call of their own.
template <typename Data>
void c_row_among(const sdca_masked_case& c, int dtype, int precision, const std::string& path,
                 double tol, among others) {
  const std::size_t n = std::size_t(c.classes), rows = 16, at = 5;
  const std::vector<Data> given(c.logits, c.logits + n), given_teacher(c.teacher, c.teacher + n);
  std::vector<Data> other = given, other_teacher = given_teacher;
  for (Data& v : other) v = std::isfinite(double(v)) ? v : Data(-3);
  for (Data& v : other_teacher) v = std::isfinite(double(v)) ? v : Data(-3);
  // Probability targets: finite and nonnegative in the other rows.
  if (c.probabilities)
    for (Data& v : other_teacher) v = v >= 0 ? v : Data(0.25);
  if (others == among::itself) {
    other = given;
    other_teacher = given_teacher;
  }
  if (others == among::masked) {
    std::size_t j = 0;
    while (j < n && (int64_t(j) == c.label || c.positive[j])) ++j;
    if (j == n) return;
    other[j] = -std::numeric_limits<Data>::infinity();
    // A class of target zero, which the masked class then removes.
    other_teacher[j] = c.probabilities ? Data(0) : -std::numeric_limits<Data>::infinity();
  }
  std::vector<Data> z, teacher, g(rows * n, Data(7));
  std::vector<std::uint8_t> mask;
  std::vector<std::int64_t> labels(rows, c.label);
  for (std::size_t r = 0; r < rows; ++r) {
    const std::vector<Data>& s = r == at ? given : other;
    const std::vector<Data>& t = r == at ? given_teacher : other_teacher;
    z.insert(z.end(), s.begin(), s.end());
    teacher.insert(teacher.end(), t.begin(), t.end());
    mask.insert(mask.end(), c.positive, c.positive + n);
  }
  const std::vector<double> probabilities(teacher.begin(), teacher.end());
  std::vector<double> values(rows, 7);
  sdca_loss loss = SDCA_LOSS_INIT(c.loss);
  loss.k = std::size_t(c.k);
  loss.gamma = c.gamma;
  if (c.loss == SDCA_LOSS_PROXIMAL_CROSS_ENTROPY || c.loss == SDCA_LOSS_PROXIMAL_DISTILLATION)
    loss.gamma = c.alpha;
  loss.temperature = c.temperature;
  sdca_execution execution = SDCA_EXECUTION_INIT;
  execution.precision = precision;
  sdca_targets targets = SDCA_TARGETS_INIT;
  if (distillation(c)) {
    targets.kind = SDCA_TARGET_LOGITS;
    targets.logits = teacher.data();
  } else if (c.probabilities) {
    targets.kind = SDCA_TARGET_PROBABILITIES;
    targets.probabilities = probabilities.data();
  } else if (c.label >= 0) {
    targets.labels = labels.data();
  } else {
    targets.kind = SDCA_TARGET_MASK;
    targets.mask = mask.data();
  }
  const int status = sdca_loss_evaluate(&loss, &execution, dtype, rows, n, z.data(), &targets,
                                        values.data(), g.data());
  if (!c.valid) {
    const bool all = others == among::itself;
    const bool reported = reports_invalid(c, status, all ? rows : 1, rows, all ? 0 : at);
    if (const char* reason = probability_targets_tracked(c))
      return settle(std::string(c.name) + " (" + path + ")",
                    reported ? "" : "returned " + std::to_string(status), reason);
    if (!reported) fail(c, path, "returned " + std::to_string(status) + ": " + sdca_last_error());
    const std::string error = sdca_last_error();
    std::fill(g.begin(), g.end(), Data(7));
    std::fill(values.begin(), values.end(), 7.0);
    execution.check_rows = 1;
    if (sdca_loss_evaluate(&loss, &execution, dtype, rows, n, z.data(), &targets, values.data(),
                           g.data()) != SDCA_INVALID_ARGUMENT ||
        error != sdca_last_error())
      return fail(c, path, "check_rows 1 gave another error: " + std::string(sdca_last_error()));
    for (std::size_t j = 0; j < rows * n; ++j)
      if (g[j] != Data(7) || values[j / n] != 7) return fail(c, path, "wrote an output");
    return;
  }
  if (status != SDCA_OK) return fail(c, path, std::string("failed: ") + sdca_last_error());
  compare(c, path, values[at],
          std::vector<Data>(g.begin() + std::ptrdiff_t(at * n),
                            g.begin() + std::ptrdiff_t((at + 1) * n)),
          tol);
  // Another row as a call of its own.
  std::vector<Data> alone_g(n, Data(7));
  double alone = 7;
  sdca_targets one = targets;
  const std::vector<double> other_probabilities(other_teacher.begin(), other_teacher.end());
  if (distillation(c)) one.logits = other_teacher.data();
  if (c.probabilities) one.probabilities = other_probabilities.data();
  if (sdca_loss_evaluate(&loss, &execution, dtype, 1, n, other.data(), &one, &alone,
                         alone_g.data()) != SDCA_OK)
    return fail(c, path, "another row alone failed");
  for (std::size_t r = 0; r < rows; ++r) {
    if (r == at) continue;
    bool same = values[r] == alone;
    for (std::size_t j = 0; j < n; ++j) same = same && g[r * n + j] == alone_g[j];
    if (!same) return fail(c, path, "row " + std::to_string(r) + " differs");
  }
}

// A masked row at temperatures near the top of the double range. Below 8.98e304 its masked
// class takes a finite logit far below the others: loss log 2 (the two classes left, equal
// at that temperature). Above, 2000 T overflows and no finite logit stands for the class:
// the row is invalid, as a row the loss cannot represent (it once recursed without end).
// Those rows are tracked (huge_temperature_reason).
void huge_temperature() {
  const double z[3] = {0.0, 1.0, -HUGE_VAL};
  const double log2 = std::log(2.0);
  for (const double temperature : {8e304, 8.98e304, 9e304, 1e305, 1e306, 1.7e308}) {
    const bool valid = 2000 * temperature < HUGE_VAL;
    for (const bool proximal : {false, true}) {
      char t[32];
      std::snprintf(t, sizeof t, "%g", temperature);
      const std::string path = std::string(proximal ? "proximal" : "distillation") + " T=" + t;
      double g[3] = {7, 7, 7}, value = 0;
      bool rejected = false;
      try {
        value = proximal ? sdca::proximal_distillation_loss<double>(3, z, z, temperature, 1.0, g)
                         : sdca::distillation_loss<double>(3, z, z, temperature, g);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      std::string what;
      if (rejected == valid)
        what = rejected ? "rejected" : "accepted";
      else if (valid && (std::fabs(value - log2) > 1e-12 || g[2] != 0 || std::signbit(g[2])))
        what = "loss " + std::to_string(value) + ", masked entry " + std::to_string(g[2]);
      settle(path, what, valid ? nullptr : huge_temperature_reason);
      sdca_loss loss =
          SDCA_LOSS_INIT(proximal ? SDCA_LOSS_PROXIMAL_DISTILLATION : SDCA_LOSS_DISTILLATION);
      loss.temperature = temperature;
      sdca_execution execution = SDCA_EXECUTION_INIT;
      sdca_targets targets = SDCA_TARGETS_INIT;
      targets.kind = SDCA_TARGET_LOGITS;
      targets.logits = z;
      double values[1] = {7}, grad[3] = {7, 7, 7};
      const int status = sdca_loss_evaluate(&loss, &execution, SDCA_DTYPE_FLOAT64, 1, 3, z,
                                            &targets, values, grad);
      what.clear();
      if ((status == SDCA_OK) != valid || (valid && std::fabs(values[0] - log2) > 1e-12) ||
          (!valid && values[0] != 7))
        what = "status " + std::to_string(status) + ", loss " + std::to_string(values[0]);
      settle("C " + path, what, valid ? nullptr : huge_temperature_reason);
    }
  }
}

}  // namespace

int main() {
  // float work resolves the losses to float eps; the smoothed hinges and proximal
  // distillation compute in double.
  const double eps32 = 4e-6, eps64 = 1e-12;
  for (std::size_t i = 0; i < SDCA_MASKED_CASES; ++i) {
    const sdca_masked_case& c = sdca_masked_cases[i];
    cpp_row<double, double>(c, "C++ double", eps64);
    cpp_row<float, double>(c, "C++ float in double", eps32);
    cpp_row<float, float>(c, "C++ float", eps32);
    c_row<double>(c, SDCA_DTYPE_FLOAT64, SDCA_PRECISION_DEFAULT, "C float64", eps64);
    c_row<float>(c, SDCA_DTYPE_FLOAT32, SDCA_PRECISION_DEFAULT, "C float32", eps32);
    c_row<float>(c, SDCA_DTYPE_FLOAT32, SDCA_PRECISION_FLOAT64, "C float32 in double", eps32);
    for (const among others : {among::finite, among::itself, among::masked}) {
      const std::string batch = others == among::finite   ? " among finite rows"
                                : others == among::itself ? " among copies"
                                                          : " among masked rows";
      c_row_among<double>(c, SDCA_DTYPE_FLOAT64, SDCA_PRECISION_DEFAULT, "C float64" + batch, eps64,
                          others);
      c_row_among<float>(c, SDCA_DTYPE_FLOAT32, SDCA_PRECISION_DEFAULT, "C float32" + batch, eps32,
                         others);
    }
  }
  huge_temperature();
  std::printf("%zu rows, %d failures, %d expected failures\n", SDCA_MASKED_CASES, counts.failures,
              counts.expected_failures);
  return counts.failures != 0;
}
