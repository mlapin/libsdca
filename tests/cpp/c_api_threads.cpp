// Every computing entry point on threads: each thread count, and calls from
// several threads at once, give the bits of the serial call. Every call of two
// rows or more takes threads; built with ThreadSanitizer, this checks the rows'
// sharing of memory (tools/run_cpp_tests.sh --sanitize).
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#include "c_api_internal.h"
#include "sdca/c_api.h"
#include "sdca/utility/parallel.h"

namespace {

typedef std::vector<unsigned char> bytes;
typedef std::function<int(std::size_t threads, bytes& out)> run_case;

struct named_case {
  std::string name;
  run_case run;
  bytes serial;
};

std::atomic<int> failures(0);

void visit_rows(void* context, std::size_t begin, std::size_t end) {
  auto* visits = static_cast<std::atomic<unsigned>*>(context);
  for (std::size_t row = begin; row < end; ++row) ++visits[row];
}

void add_scheduler(std::vector<named_case>& cases) {
  for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(137)})
    cases.push_back({"row scheduler " + std::to_string(n),
                     [n](std::size_t threads, bytes& out) {
                       std::atomic<unsigned> visits[137];
                       for (auto& count : visits) count = 0;
                       const int code =
                           sdca_internal_run_rows(n, 1000, threads, visit_rows, visits);
                       for (std::size_t row = 0; row < 137; ++row) {
                         const unsigned count = visits[row].load();
                         if (count != (row < n ? 1u : 0u)) ++failures;
                         out.push_back(static_cast<unsigned char>(count));
                       }
                       return code;
                     },
                     bytes()});
}

std::uint32_t next(std::uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  return state >> 8;
}

template <typename T>
void append(bytes& out, const std::vector<T>& v) {
  const unsigned char* p = reinterpret_cast<const unsigned char*>(v.data());
  out.insert(out.end(), p, p + v.size() * sizeof(T));
}

// Scores m / 16 with |m| <= 64: exact in every storage type.
struct inputs {
  std::size_t n, k;
  std::vector<double> z64, t64;
  std::vector<float> z32, t32;
  std::vector<std::uint16_t> z16, t16, zb16, tb16;
  std::vector<std::int64_t> labels, ignored, sets, offsets;
  std::vector<std::uint8_t> single, multi;
  std::vector<double> weights;

  inputs(std::size_t rows, std::size_t classes) : n(rows), k(classes) {
    std::uint32_t state = static_cast<std::uint32_t>(rows * 131 + classes);
    for (std::size_t i = 0; i < n * k; ++i) {
      z64.push_back((static_cast<int>(next(state) % 129) - 64) / 16.0);
      t64.push_back((static_cast<int>(next(state) % 129) - 64) / 16.0);
    }
    for (std::size_t i = 0; i < n * k; ++i) {
      z32.push_back(static_cast<float>(z64[i]));
      t32.push_back(static_cast<float>(t64[i]));
      z16.push_back(half(z32[i]));
      t16.push_back(half(t32[i]));
      zb16.push_back(bfloat(z32[i]));
      tb16.push_back(bfloat(t32[i]));
    }
    single.assign(n * k, 0);
    multi.assign(n * k, 0);
    offsets.push_back(0);
    for (std::size_t i = 0; i < n; ++i) {
      labels.push_back(static_cast<std::int64_t>(next(state) % k));
      ignored.push_back(i % 7 == 3 ? -100 : labels.back());
      single[i * k + static_cast<std::size_t>(labels.back())] = 1;
      for (std::size_t j = 0; j < k; ++j)
        if (j == i % k || (j != (i + 1) % k && next(state) % 5 == 0)) {
          multi[i * k + j] = 1;
          sets.push_back(static_cast<std::int64_t>(j));
        }
      offsets.push_back(static_cast<std::int64_t>(sets.size()));
    }
    for (std::size_t j = 0; j < k; ++j) weights.push_back(0.5 + (j % 4) * 0.25);
  }

  static std::uint32_t bits(float x) {
    std::uint32_t b;
    std::memcpy(&b, &x, sizeof b);
    return b;
  }
  static std::uint16_t bfloat(float x) { return static_cast<std::uint16_t>(bits(x) >> 16); }
  // Exact for these scores: zero, or normal halves with at most 10 mantissa bits.
  static std::uint16_t half(float x) {
    const std::uint32_t b = bits(x);
    if ((b & 0x7fffffffu) == 0) return static_cast<std::uint16_t>(b >> 16);
    const std::uint32_t sign = (b >> 16) & 0x8000u, exponent = ((b >> 23) & 0xffu) - 127 + 15;
    return static_cast<std::uint16_t>(sign | (exponent << 10) | ((b >> 13) & 0x3ffu));
  }

  const void* scores(int dtype) const {
    if (dtype == SDCA_DTYPE_FLOAT64) return z64.data();
    if (dtype == SDCA_DTYPE_FLOAT32) return z32.data();
    return dtype == SDCA_INTERNAL_DTYPE_FLOAT16 ? z16.data() : zb16.data();
  }
  const void* teacher(int dtype) const {
    if (dtype == SDCA_DTYPE_FLOAT64) return t64.data();
    if (dtype == SDCA_DTYPE_FLOAT32) return t32.data();
    return dtype == SDCA_INTERNAL_DTYPE_FLOAT16 ? t16.data() : tb16.data();
  }
};

std::size_t storage(int dtype) {
  return dtype == SDCA_DTYPE_FLOAT64 ? 8 : dtype == SDCA_DTYPE_FLOAT32 ? 4 : 2;
}

// Gradients of the internal loss rows and proximal distillation: float or double.
std::size_t wide(int dtype) { return dtype == SDCA_DTYPE_FLOAT64 ? 8 : 4; }

const char* dtype_name(int dtype) {
  static const char* names[] = {"float32", "float64", "float16", "bfloat16"};
  return names[dtype];
}

bool multilabel(int loss) {
  return loss == SDCA_LOSS_MULTILABEL_HINGE || loss == SDCA_LOSS_MULTILABEL_CROSS_ENTROPY;
}
bool topk(int loss) {
  return loss == SDCA_LOSS_TOPK_CROSS_ENTROPY || loss == SDCA_LOSS_TOPK_HINGE_ALPHA ||
         loss == SDCA_LOSS_TOPK_HINGE_BETA;
}
bool hinge(int loss) {
  return loss == SDCA_LOSS_MULTICLASS_HINGE || loss == SDCA_LOSS_TOPK_HINGE_ALPHA ||
         loss == SDCA_LOSS_TOPK_HINGE_BETA || loss == SDCA_LOSS_MULTILABEL_HINGE;
}

// The public loss of kind with its own parameters (k 3, the given gamma, alpha 0.7).
sdca_loss loss_of(int kind, double gamma) {
  sdca_loss loss = SDCA_LOSS_INIT;
  loss.kind = kind;
  if (topk(kind)) loss.k = 3;
  if (hinge(kind)) loss.gamma = gamma;
  if (kind == SDCA_LOSS_PROXIMAL_CROSS_ENTROPY || kind == SDCA_LOSS_PROXIMAL_DISTILLATION)
    loss.alpha = 0.7;
  if (kind == SDCA_LOSS_DISTILLATION || kind == SDCA_LOSS_PROXIMAL_DISTILLATION)
    loss.temperature = 2.0;
  return loss;
}

sdca_execution execution_of(int precision, std::size_t threads) {
  sdca_execution e = SDCA_EXECUTION_INIT;
  e.precision = precision;
  e.num_threads = threads;
  return e;
}

sdca_internal_options internal_of(bool float_work, std::size_t threads) {
  const sdca_internal_options o = {sizeof(sdca_internal_options),
                                   float_work ? 1 : 0,
                                   0,
                                   1,
                                   0.0,
                                   1.0,
                                   SDCA_PROJECTION_FLOAT64,
                                   threads};
  return o;
}

// The label targets of a loss: labels, a mask or label sets.
sdca_targets targets_of(const inputs& in, int loss, bool mask) {
  sdca_targets t = SDCA_TARGETS_INIT;
  if (!multilabel(loss)) {
    t.labels = in.labels.data();
  } else if (mask) {
    t.kind = SDCA_TARGET_MASK;
    t.mask = in.multi.data();
  } else {
    t.kind = SDCA_TARGET_LABEL_SETS;
    t.labels = in.sets.data();
    t.offsets = in.offsets.data();
  }
  return t;
}

void add_cases(const inputs& in, std::vector<named_case>& cases) {
  const std::size_t n = in.n, k = in.k;
  const std::string shape = " " + std::to_string(n) + "x" + std::to_string(k);
  const double gammas[] = {0.0, 0.5};
  for (int loss = SDCA_LOSS_CROSS_ENTROPY; loss <= SDCA_LOSS_PROXIMAL_CROSS_ENTROPY; ++loss) {
    for (double gamma : gammas) {
      if (gamma > 0 && !hinge(loss)) continue;
      const sdca_loss l = loss_of(loss, gamma);
      const std::string name = "loss " + std::to_string(loss) + " gamma " + std::to_string(gamma);
      for (int dtype : {SDCA_DTYPE_FLOAT32, SDCA_DTYPE_FLOAT64})
        for (int precision :
             {SDCA_PRECISION_DEFAULT, SDCA_PRECISION_FLOAT32, SDCA_PRECISION_FLOAT64}) {
          if (dtype == SDCA_DTYPE_FLOAT64 && precision == SDCA_PRECISION_FLOAT32) continue;
          cases.push_back(
              {name + " " + dtype_name(dtype) + " precision " + std::to_string(precision) + shape,
               [=, &in](std::size_t threads, bytes& out) {
                 const sdca_execution e = execution_of(precision, threads);
                 const sdca_targets t = targets_of(in, loss, false);
                 std::vector<double> values(n);
                 std::vector<unsigned char> g(n * k * storage(dtype));
                 const int code = sdca_loss_evaluate(&l, &e, dtype, n, k, in.scores(dtype), &t,
                                                     values.data(), g.data());
                 append(out, values);
                 append(out, g);
                 return code;
               },
               bytes()});
        }
      for (int dtype = 0; dtype < 4; ++dtype)
        for (bool float_work : {true, false}) {
          if (dtype == SDCA_DTYPE_FLOAT64 && float_work) continue;
          const bool labeled = !multilabel(loss);
          cases.push_back(
              {name + " rows " + dtype_name(dtype) + (float_work ? " float" : " double") + shape,
               [=, &in](std::size_t threads, bytes& out) {
                 sdca_internal_options t = internal_of(float_work, threads);
                 t.loss = loss;
                 t.k = l.k ? l.k : 1;
                 t.gamma = gamma;
                 t.alpha = 0.7;
                 std::vector<double> values(n);
                 std::vector<unsigned char> g(n * k * wide(dtype));
                 const int code = sdca_internal_loss_rows(
                     &t, dtype, n, k, in.scores(dtype), labeled ? in.labels.data() : nullptr,
                     labeled ? nullptr : in.multi.data(), values.data(), g.data());
                 append(out, values);
                 append(out, g);
                 return code;
               },
               bytes()});
        }
      if (loss == SDCA_LOSS_PROXIMAL_CROSS_ENTROPY) continue;
      for (bool mask : {false, true}) {
        if (mask && !multilabel(loss)) continue;
        cases.push_back({"update " + name + (mask ? " mask" : "") + shape,
                         [=, &in](std::size_t threads, bytes& out) {
                           const sdca_execution e = execution_of(SDCA_PRECISION_DEFAULT, threads);
                           const sdca_targets t = targets_of(in, loss, mask);
                           std::vector<double> norm2(n, 1.5), next_dual(n * k);
                           const int code = sdca_dual_update(&l, 2.0, &e, SDCA_DTYPE_FLOAT64, n, k,
                                                             in.z64.data(), &t, nullptr,
                                                             norm2.data(), next_dual.data());
                           append(out, next_dual);
                           return code;
                         },
                         bytes()});
      }
    }
  }
  for (int dtype : {SDCA_DTYPE_FLOAT32, SDCA_DTYPE_FLOAT64})
    for (int loss : {SDCA_LOSS_DISTILLATION, SDCA_LOSS_PROXIMAL_DISTILLATION})
      cases.push_back(
          {std::string("loss ") + std::to_string(loss) + " " + dtype_name(dtype) + shape,
           [=, &in](std::size_t threads, bytes& out) {
             const sdca_loss l = loss_of(loss, 0.0);
             const sdca_execution e = execution_of(SDCA_PRECISION_DEFAULT, threads);
             sdca_targets t = SDCA_TARGETS_INIT;
             t.kind = SDCA_TARGET_LOGITS;
             t.logits = in.teacher(dtype);
             std::vector<double> values(n);
             std::vector<unsigned char> g(n * k * storage(dtype));
             const int code = sdca_loss_evaluate(&l, &e, dtype, n, k, in.scores(dtype), &t,
                                                 values.data(), g.data());
             append(out, values);
             append(out, g);
             return code;
           },
           bytes()});
  for (int dtype = 0; dtype < 4; ++dtype)
    for (bool float_work : {true, false}) {
      if (dtype == SDCA_DTYPE_FLOAT64 && float_work) continue;
      const std::string name = std::string(dtype_name(dtype)) + (float_work ? " float" : " double");
      for (bool weighted : {false, true}) {
        cases.push_back({"cross_entropy " + name + (weighted ? " weighted" : "") + shape,
                         [=, &in](std::size_t threads, bytes& out) {
                           const sdca_internal_options t = internal_of(float_work, threads);
                           const std::vector<std::int64_t>& y = weighted ? in.ignored : in.labels;
                           const double* w = weighted ? in.weights.data() : nullptr;
                           std::vector<double> values(n), stats(n * 5), totals(2), scale(n);
                           for (std::size_t i = 0; i < n; ++i) scale[i] = 1.0 + (i % 3);
                           std::vector<unsigned char> g(n * k * storage(dtype));
                           int code = sdca_internal_cross_entropy_forward(
                               &t, dtype, n, k, in.scores(dtype), y.data(), -100, w, values.data(),
                               stats.data(), totals.data());
                           if (code == 0)
                             code = sdca_internal_cross_entropy_backward(
                                 &t, dtype, n, k, in.scores(dtype), y.data(), -100, w, stats.data(),
                                 scale.data(), 1, g.data());
                           append(out, values);
                           append(out, stats);
                           append(out, totals);
                           append(out, g);
                           return code;
                         },
                         bytes()});
      }
      cases.push_back({"distillation " + name + shape,
                       [=, &in](std::size_t threads, bytes& out) {
                         const sdca_internal_options t = internal_of(float_work, threads);
                         std::vector<double> values(n), stats(n * 7), scale(1, 0.5);
                         std::vector<unsigned char> g(n * k * storage(dtype));
                         int code = sdca_internal_distillation_forward(
                             &t, dtype, n, k, in.scores(dtype), in.teacher(dtype), 2.0,
                             values.data(), stats.data());
                         if (code == 0)
                           code = sdca_internal_distillation_backward(
                               &t, dtype, n, k, in.scores(dtype), in.teacher(dtype), 2.0,
                               stats.data(), scale.data(), 0, g.data());
                         append(out, values);
                         append(out, stats);
                         append(out, g);
                         return code;
                       },
                       bytes()});
    }
  for (int dtype = 0; dtype < 4; ++dtype)
    cases.push_back({std::string("proximal distillation ") + dtype_name(dtype) + shape,
                     [=, &in](std::size_t threads, bytes& out) {
                       const sdca_internal_options t = internal_of(false, threads);
                       std::vector<double> values(n);
                       std::vector<unsigned char> g(n * k * wide(dtype));
                       const int code = sdca_internal_proximal_distillation_rows(
                           &t, dtype, n, k, in.scores(dtype), in.teacher(dtype), 2.0, 0.7,
                           values.data(), g.data());
                       append(out, values);
                       append(out, g);
                       return code;
                     },
                     bytes()});
}

// Fits whose evaluations take threads: weights, duals and the history but its times.
void add_fits(const inputs& in, std::size_t d, std::vector<named_case>& cases) {
  const std::size_t n = in.n, k = in.k;
  for (int loss : {SDCA_LOSS_CROSS_ENTROPY, SDCA_LOSS_MULTICLASS_HINGE, SDCA_LOSS_TOPK_HINGE_BETA,
                   SDCA_LOSS_MULTILABEL_HINGE, SDCA_LOSS_MULTILABEL_CROSS_ENTROPY}) {
    cases.push_back({"fit " + std::to_string(loss),
                     [=, &in](std::size_t threads, bytes& out) {
                       sdca_loss l = loss_of(loss, 0.5);
                       if (loss == SDCA_LOSS_TOPK_HINGE_BETA) l.k = 2;
                       sdca_fit_options o = SDCA_FIT_OPTIONS_INIT;
                       o.max_epoch = 4;
                       o.eval_epoch = 1;
                       o.tol = 0;
                       const sdca_execution e = execution_of(SDCA_PRECISION_DEFAULT, threads);
                       sdca_input input = SDCA_INPUT_INIT;
                       input.n = n;
                       input.d = d;
                       input.x = in.z64.data();
                       const sdca_targets t = targets_of(in, loss, false);
                       std::vector<double> w(d * k), a(n * k), history(6 * SDCA_HISTORY_FIELDS);
                       sdca_fit_result result = SDCA_FIT_RESULT_INIT;
                       const int code =
                           sdca_model_fit(&l, &o, &e, &input, k, &t, nullptr, nullptr, w.data(),
                                          a.data(), history.data(), 6, &result);
                       for (std::size_t row = 0; row < 6; ++row)
                         history[row * SDCA_HISTORY_FIELDS + SDCA_HISTORY_CPU_TIME] =
                             history[row * SDCA_HISTORY_FIELDS + SDCA_HISTORY_WALL_TIME] = 0;
                       append(out, w);
                       append(out, a);
                       append(out, history);
                       return code;
                     },
                     bytes()});
  }
}

void check(const named_case& c, std::size_t threads) {
  bytes out;
  const int code = c.run(threads, out);
  if (code != 0 || out != c.serial) {
    ++failures;
    std::fprintf(stderr, "FAIL %s, %zu threads: code %d (%s), %s\n", c.name.c_str(), threads, code,
                 sdca_last_error(), out == c.serial ? "same bits" : "other bits");
  }
}

}  // namespace

int main() {
  // Every call of two rows or more takes threads, at once.
  sdca::detail::thread_work_setting() = 0;
#ifdef _OPENMP
  // Built with OpenMP: a running team takes the rows instead of the pool.
  std::atomic<int> team(0);
#pragma omp parallel
  ++team;
#endif
  const inputs narrow(37, 10), wide_rows(23, 70), fit_data(64, 6);
  std::vector<named_case> cases;
  if (sdca_internal_run_rows(1, 1, 1, nullptr, nullptr) != SDCA_INVALID_ARGUMENT) ++failures;
  add_scheduler(cases);
  add_cases(narrow, cases);
  add_cases(wide_rows, cases);
  add_fits(fit_data, 6, cases);
  for (named_case& c : cases) {
    const int code = c.run(1, c.serial);
    if (code != 0) {
      ++failures;
      std::fprintf(stderr, "FAIL %s, serial: code %d (%s)\n", c.name.c_str(), code,
                   sdca_last_error());
    }
  }
  for (const named_case& c : cases)
    for (std::size_t threads : {std::size_t(2), std::size_t(3), std::size_t(8), std::size_t(0)})
      check(c, threads);
  // Callers on four threads at once: one takes the pool, the others new threads.
  std::vector<std::thread> callers;
  for (std::size_t caller = 0; caller < 4; ++caller)
    callers.emplace_back([&cases, caller] {
      for (std::size_t i = caller; i < cases.size() + caller; ++i)
        check(cases[i % cases.size()], caller == 0 ? 0 : caller + 1);
    });
  for (std::thread& caller : callers) caller.join();
  std::printf("%zu cases, %d failures%s\n", cases.size(), failures.load(),
              sdca::detail::openmp_team() ? " (OpenMP team)" : "");
  return failures ? 1 : 0;
}
