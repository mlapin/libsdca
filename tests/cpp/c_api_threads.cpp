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
    return dtype == SDCA_DTYPE_FLOAT16 ? z16.data() : zb16.data();
  }
  const void* teacher(int dtype) const {
    if (dtype == SDCA_DTYPE_FLOAT64) return t64.data();
    if (dtype == SDCA_DTYPE_FLOAT32) return t32.data();
    return dtype == SDCA_DTYPE_FLOAT16 ? t16.data() : tb16.data();
  }
};

std::size_t storage(int dtype) {
  return dtype == SDCA_DTYPE_FLOAT64 ? 8 : dtype == SDCA_DTYPE_FLOAT32 ? 4 : 2;
}

// Gradients of sdca_loss_gradient_rows and sdca_proximal_distillation_rows: float or double.
std::size_t wide(int dtype) { return dtype == SDCA_DTYPE_FLOAT64 ? 8 : 4; }

sdca_options with_threads(sdca_options o, std::size_t threads) {
  o.num_threads = threads;
  return o;
}

const char* dtype_name(int dtype) {
  static const char* names[] = {"float32", "float64", "float16", "bfloat16"};
  return names[dtype];
}

bool multilabel(int loss) {
  return loss == SDCA_MULTILABEL_HINGE || loss == SDCA_MULTILABEL_CROSS_ENTROPY;
}

void add_cases(const inputs& in, std::vector<named_case>& cases) {
  const std::size_t n = in.n, k = in.k;
  const std::string shape = " " + std::to_string(n) + "x" + std::to_string(k);
  const double gammas[] = {0.0, 0.5};
  for (int loss = SDCA_CROSS_ENTROPY; loss <= SDCA_PROXIMAL_CROSS_ENTROPY; ++loss) {
    for (double gamma : gammas) {
      sdca_options o = SDCA_OPTIONS_DEFAULT;
      o.loss = loss;
      o.k = loss == SDCA_TOPK_CROSS_ENTROPY || loss == SDCA_TOPK_HINGE_ALPHA ||
                    loss == SDCA_TOPK_HINGE_BETA
                ? 3
                : 1;
      o.gamma = gamma;
      o.alpha = 0.7;
      const std::vector<std::uint8_t>& mask = multilabel(loss) ? in.multi : in.single;
      const std::string name = "loss " + std::to_string(loss) + " gamma " + std::to_string(gamma);
      for (int precision : {SDCA_FLOAT32, SDCA_FLOAT32_FLOAT64, SDCA_FLOAT64}) {
        cases.push_back({name + " precision " + std::to_string(precision) + shape,
                         [=, &in, &mask](std::size_t threads, bytes& out) {
                           sdca_options t = with_threads(o, threads);
                           t.precision = precision;
                           const bool d = precision == SDCA_FLOAT64;
                           std::vector<double> values(n);
                           std::vector<unsigned char> g(n * k * (d ? 8 : 4));
                           const int code =
                               sdca_loss_gradient(&t, n, k, d ? in.scores(1) : in.scores(0),
                                                  mask.data(), values.data(), g.data());
                           append(out, values);
                           append(out, g);
                           return code;
                         },
                         bytes()});
      }
      for (int dtype = 0; dtype < 4; ++dtype)
        for (int precision : {SDCA_FLOAT32, SDCA_FLOAT64}) {
          if (dtype == SDCA_DTYPE_FLOAT64 && precision == SDCA_FLOAT32) continue;
          const bool labeled = !multilabel(loss);
          cases.push_back(
              {name + " rows " + dtype_name(dtype) + " " + std::to_string(precision) + shape,
               [=, &in, &mask](std::size_t threads, bytes& out) {
                 sdca_options t = with_threads(o, threads);
                 t.precision = precision;
                 std::vector<double> values(n);
                 std::vector<unsigned char> g(n * k * wide(dtype));
                 const int code = sdca_loss_gradient_rows(
                     &t, dtype, n, k, in.scores(dtype), labeled ? in.labels.data() : nullptr,
                     labeled ? nullptr : mask.data(), values.data(), g.data());
                 append(out, values);
                 append(out, g);
                 return code;
               },
               bytes()});
        }
      if (loss == SDCA_PROXIMAL_CROSS_ENTROPY) continue;
      cases.push_back({"update " + name + shape,
                       [=, &in](std::size_t threads, bytes& out) {
                         sdca_options t = with_threads(o, threads);
                         t.c = 2.0;
                         std::vector<double> previous(n * k, 0.0), next_dual(n * k);
                         const bool sets = multilabel(loss);
                         const int code = sdca_update_objective(
                             &t, n, k, in.z64.data(), sets ? in.sets.data() : in.labels.data(),
                             sets ? in.offsets.data() : nullptr, previous.data(), 1.5,
                             next_dual.data());
                         append(out, next_dual);
                         return code;
                       },
                       bytes()});
    }
  }
  for (int precision : {SDCA_FLOAT32, SDCA_FLOAT64}) {
    cases.push_back({"entropy precision " + std::to_string(precision) + shape,
                     [=, &in](std::size_t threads, bytes& out) {
                       sdca_options t = SDCA_OPTIONS_DEFAULT;
                       t.precision = precision;
                       t.num_threads = threads;
                       const bool d = precision == SDCA_FLOAT64;
                       std::vector<double> values(n), stats(n * 5), scale(n, 0.75);
                       std::vector<unsigned char> g(n * k * (d ? 8 : 4));
                       const void* z = d ? in.scores(1) : in.scores(0);
                       int code = sdca_entropy_forward(&t, n, k, z, in.labels.data(), values.data(),
                                                       stats.data());
                       if (code == 0)
                         code = sdca_entropy_gradient(&t, n, k, z, in.labels.data(), stats.data(),
                                                      scale.data(), g.data());
                       append(out, values);
                       append(out, stats);
                       append(out, g);
                       return code;
                     },
                     bytes()});
  }
  for (int dtype = 0; dtype < 4; ++dtype)
    for (int precision : {SDCA_FLOAT32, SDCA_FLOAT64}) {
      if (dtype == SDCA_DTYPE_FLOAT64 && precision == SDCA_FLOAT32) continue;
      const std::string name = std::string(dtype_name(dtype)) + " " + std::to_string(precision);
      for (bool weighted : {false, true}) {
        cases.push_back({"cross_entropy " + name + (weighted ? " weighted" : "") + shape,
                         [=, &in](std::size_t threads, bytes& out) {
                           sdca_options t = SDCA_OPTIONS_DEFAULT;
                           t.precision = precision;
                           t.num_threads = threads;
                           const std::vector<std::int64_t>& y = weighted ? in.ignored : in.labels;
                           const double* w = weighted ? in.weights.data() : nullptr;
                           std::vector<double> values(n), stats(n * 5), totals(2), scale(n);
                           for (std::size_t i = 0; i < n; ++i) scale[i] = 1.0 + (i % 3);
                           std::vector<unsigned char> g(n * k * storage(dtype));
                           int code = sdca_cross_entropy_forward(&t, dtype, n, k, in.scores(dtype),
                                                                 y.data(), -100, w, values.data(),
                                                                 stats.data(), totals.data());
                           if (code == 0)
                             code = sdca_cross_entropy_backward(&t, dtype, n, k, in.scores(dtype),
                                                                y.data(), -100, w, stats.data(),
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
                         sdca_options t = SDCA_OPTIONS_DEFAULT;
                         t.precision = precision;
                         t.num_threads = threads;
                         std::vector<double> values(n), stats(n * 7), scale(1, 0.5);
                         std::vector<unsigned char> g(n * k * storage(dtype));
                         int code = sdca_distillation_forward(&t, dtype, n, k, in.scores(dtype),
                                                              in.teacher(dtype), 2.0, values.data(),
                                                              stats.data());
                         if (code == 0)
                           code = sdca_distillation_backward(&t, dtype, n, k, in.scores(dtype),
                                                             in.teacher(dtype), 2.0, stats.data(),
                                                             scale.data(), 0, g.data());
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
                       sdca_options t = SDCA_OPTIONS_DEFAULT;
                       t.num_threads = threads;
                       std::vector<double> values(n);
                       std::vector<unsigned char> g(n * k * wide(dtype));
                       const int code = sdca_proximal_distillation_rows(
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
  for (int loss : {SDCA_CROSS_ENTROPY, SDCA_MULTICLASS_HINGE, SDCA_TOPK_HINGE_BETA,
                   SDCA_MULTILABEL_HINGE, SDCA_MULTILABEL_CROSS_ENTROPY}) {
    cases.push_back({"fit " + std::to_string(loss),
                     [=, &in](std::size_t threads, bytes& out) {
                       sdca_options t = SDCA_OPTIONS_DEFAULT;
                       t.loss = loss;
                       t.k = loss == SDCA_TOPK_HINGE_BETA ? 2 : 1;
                       t.gamma = 0.5;
                       t.max_epoch = 4;
                       t.eval_epoch = 1;
                       t.tol = 0;
                       t.num_threads = threads;
                       std::vector<double> w(d * k), a(n * k), history(6 * 10);
                       sdca_fit_result result = SDCA_FIT_RESULT_INIT;
                       const bool sets = multilabel(loss);
                       const int code = sdca_fit(&t, n, d, k, in.z64.data(),
                                                 sets ? in.sets.data() : in.labels.data(),
                                                 sets ? in.offsets.data() : nullptr, nullptr,
                                                 w.data(), a.data(), history.data(), 6, &result);
                       for (std::size_t row = 0; row < 6; ++row)
                         history[row * 10 + 8] = history[row * 10 + 9] = 0;
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
    for (std::size_t threads :
         {std::size_t(2), std::size_t(3), std::size_t(8), std::size_t(0), SDCA_THREADS_DEFAULT})
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
