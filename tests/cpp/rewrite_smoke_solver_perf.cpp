// Smoke test and timing of the solver's hot loops (sdca/solver.h) against the solver of
// d297c1d2: fits and evaluations of small problems compared with the reference, results
// compared across thread counts, and the time of an epoch and of an evaluation on dense
// features. Compiled twice:
//
//   F="-std=c++11 -O3 -ffp-contract=off"
//   c++ $F -DSDCA_SMOKE_REFERENCE -Dsdca=sdca_ref -I<ref>/include -c THIS -o ref.o
//   c++ $F -Wall -Wextra -Wshadow -Werror -Iinclude THIS ref.o -o perf
//   ./perf; ./perf time [n dim m...]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace perf {

enum objective { multiclass_hinge, topk_hinge, multilabel_hinge, cross_entropy, num_objectives };

struct problem {
  std::size_t n, d, m;
  std::vector<double> X;
  std::vector<std::size_t> y;
  std::vector<std::vector<std::size_t>> labels;
};

struct spec {
  objective o;
  double c;
  bool single;  // float data and results
  std::size_t epochs, threads;
  bool shrink;
  double tol;
};

struct fit {
  std::vector<double> A, W;
  double primal, dual, gap_error, metric;
  std::string status;
  std::size_t epoch;
  double epoch_seconds, eval_seconds;
};

fit reference_fit(const problem& p, const spec& s);

}  // namespace perf

#include "sdca/solver.h"

namespace perf {
namespace {

template <typename Context>
void configure(Context& ctx, const spec& s) {
  auto& criteria = ctx.criteria;
  criteria.tol = s.tol;
  criteria.max_epoch = s.epochs;
  criteria.eval_epoch = s.tol > 0 ? 5 : 0;
  criteria.num_threads = s.threads;
  criteria.seed = 7;
  criteria.shrinking = s.shrink ? sdca::shrinking_mode::automatic : sdca::shrinking_mode::off;
}

template <typename Eval>
auto metric_of(const Eval& e, int) -> decltype(double(e.accuracy[0])) {
  return double(e.accuracy.empty() ? 0 : e.accuracy[0]);
}
template <typename Eval>
double metric_of(const Eval& e, long) {
  return double(e.rank_loss);
}

template <typename Context, typename Data>
fit finish(Context& ctx, const spec& s, const std::vector<Data>& W, const std::vector<Data>& A) {
  configure(ctx, s);
  sdca::make_solver(ctx).solve();
  const auto& e = ctx.train.evals.back();
  fit r;
  r.A.assign(A.begin(), A.end());
  r.W.assign(W.begin(), W.end());
  r.status = ctx.status_name();
  r.primal = double(e.primal);
  r.dual = double(e.dual);
  r.gap_error = double(e.gap_error);
  r.metric = metric_of(e, 0);
  r.epoch = ctx.epoch;
  r.epoch_seconds = ctx.solve_time.wall.elapsed / double(std::max<std::size_t>(1, ctx.epoch));
  r.eval_seconds = ctx.eval_time.wall.elapsed / double(ctx.train.evals.size());
  return r;
}

template <typename Data>
fit solve_typed(const problem& p, const spec& s) {
  const std::size_t n = p.n, d = p.d, m = p.m;
  const Data c = Data(s.c);
  std::vector<Data> X(p.X.begin(), p.X.end()), W(m * d), A(n * m);
  auto in = [&] { return sdca::make_input_feature(n, d, X.data()); };
  auto classes = [&] { return sdca::make_output_multiclass(p.y.begin(), p.y.end(), m); };
  switch (s.o) {
    case multiclass_hinge: {
      auto ctx = sdca::make_context(in(), classes(),
                                    sdca::make_objective_l2_topk_hinge_alpha<Data, Data>(c, 1),
                                    W.data(), A.data());
      return finish(ctx, s, W, A);
    }
    case topk_hinge: {
      auto ctx = sdca::make_context(in(), classes(),
                                    sdca::make_objective_l2_topk_hinge_beta<Data, Data>(c, 5),
                                    W.data(), A.data());
      return finish(ctx, s, W, A);
    }
    case multilabel_hinge: {
      auto ctx = sdca::make_context(in(), sdca::make_output_multilabel(p.labels, m),
                                    sdca::make_objective_l2_multilabel_hinge<Data, Data>(c),
                                    W.data(), A.data());
      return finish(ctx, s, W, A);
    }
    case cross_entropy:
    default: {
      auto ctx =
          sdca::make_context(in(), classes(), sdca::make_objective_l2_cross_entropy<Data, Data>(c),
                             W.data(), A.data());
      return finish(ctx, s, W, A);
    }
  }
}

fit solve_spec(const problem& p, const spec& s) {
  return s.single ? solve_typed<float>(p, s) : solve_typed<double>(p, s);
}

}  // namespace
}  // namespace perf

#ifdef SDCA_SMOKE_REFERENCE

namespace perf {

fit reference_fit(const problem& p, const spec& s) { return solve_spec(p, s); }

}  // namespace perf

#else

#include <new>

namespace perf {
std::size_t allocations = 0;
}  // namespace perf

void* operator new(std::size_t size) {
  ++perf::allocations;
  if (void* p = std::malloc(size ? size : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace perf {

const char* const names[] = {"multiclass_hinge", "topk_hinge(5)", "multilabel_hinge",
                             "cross_entropy"};

problem make_problem(std::size_t n, std::size_t d, std::size_t m, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal(0, 1);
  problem p;
  p.n = n, p.d = d, p.m = m;
  p.X.resize(n * d);
  for (double& x : p.X) x = normal(rng) / std::sqrt(double(d));
  std::vector<double> truth(m * d), s(m);
  for (double& w : truth) w = normal(rng);
  p.y.resize(n);
  p.labels.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t c = 0; c < m; ++c) {
      double sum = 0;
      for (std::size_t l = 0; l < d; ++l) sum += truth[c * d + l] * p.X[i * d + l];
      s[c] = sum + 0.5 * normal(rng);
    }
    p.y[i] = std::size_t(std::max_element(s.begin(), s.end()) - s.begin());
    std::vector<std::size_t> order(m);
    for (std::size_t c = 0; c < m; ++c) order[c] = c;
    std::partial_sort(order.begin(), order.begin() + 3, order.end(),
                      [&](std::size_t a, std::size_t b) { return s[a] > s[b]; });
    p.labels[i].assign(order.begin(), order.begin() + 3);
    std::sort(p.labels[i].begin(), p.labels[i].end());
  }
  return p;
}

double relative(double a, double b) { return std::abs(a - b) / std::max(1e-300, std::abs(b)); }

double max_difference(const std::vector<double>& a, const std::vector<double>& b) {
  double scale = 1e-300, diff = 0;
  for (std::size_t j = 0; j < a.size(); ++j) {
    scale = std::max(scale, std::abs(b[j]));
    diff = std::max(diff, std::abs(a[j] - b[j]));
  }
  return diff / scale;
}

// The bits of a fit (FNV-1a), to compare instruction sets.
unsigned long long fingerprint(const fit& f) {
  unsigned long long h = 1469598103934665603ull;
  auto add = [&](double x) {
    unsigned long long bits;
    std::memcpy(&bits, &x, sizeof bits);
    for (int b = 0; b < 8; ++b) h = (h ^ ((bits >> (8 * b)) & 255u)) * 1099511628211ull;
  };
  for (double x : f.A) add(x);
  for (double x : f.W) add(x);
  add(f.primal), add(f.dual), add(f.gap_error), add(f.metric);
  return h;
}

bool same_bits(const fit& a, const fit& b) {
  return a.A == b.A && a.W == b.W && a.primal == b.primal && a.dual == b.dual &&
         a.gap_error == b.gap_error && a.metric == b.metric && a.status == b.status;
}

// Fits of small problems (dimensions with lane tails) against the reference, and the same
// bits for every thread count.
int smoke() {
  int failures = 0, count = 0;
  std::printf("%-18s %5s %4s %6s %3s %13s %9s %9s %9s %16s %s\n", "objective", "c", "dim", "type",
              "shr", "primal", "d primal", "d duals", "d metric", "bits", "status (new/ref)");
  const std::size_t dims[] = {1, 7, 37, 70};
  for (int o = 0; o < num_objectives; ++o)
    for (std::size_t d : dims)
      for (double c : {0.1, 10.0})
        for (bool single : {false, true})
          for (bool shrink : {false, true}) {
            if (o == cross_entropy && shrink) continue;
            const problem p = make_problem(150, d, 9, 20261119 + d);
            const spec s = {objective(o), c, single, 60, 1, shrink, 1e-9};
            const fit want = reference_fit(p, s), got = solve_spec(p, s);
            spec t = s;
            t.threads = 3;
            const fit threaded = solve_spec(p, t);
            const double objective_error = relative(got.primal, want.primal);
            // Both bracket the optimum: the intervals [dual, primal] meet (to rounding).
            const double slack = (single ? 1e-5 : 1e-12) * std::abs(want.primal);
            const bool bad =
                std::max(got.dual, want.dual) > std::min(got.primal, want.primal) + slack ||
                !same_bits(got, threaded) || !(got.gap_error > 0) ||
                std::abs(got.metric - want.metric) > 0.05;
            failures += bad;
            ++count;
            std::printf("%-18s %5g %4zu %6s %3d %13.6g %9.2g %9.2g %9.2g %016llx %s/%s%s%s\n",
                        names[o], c, d, single ? "float" : "double", int(shrink), got.primal,
                        objective_error, max_difference(got.A, want.A),
                        std::abs(got.metric - want.metric), fingerprint(got), got.status.c_str(),
                        want.status.c_str(), same_bits(got, threaded) ? "" : "  THREADS",
                        bad ? "  FAIL" : "");
          }
  std::printf("%d of %d fits differ\n", failures, count);
  return failures;
}

// The allocations of the coordinate steps: none once the per-thread pools are warm (epochs
// beyond the first two allocate nothing).
int step_allocations() {
  int failures = 0;
  for (int o = 0; o < num_objectives; ++o)
    for (bool single : {false, true}) {
      const problem p = make_problem(200, 37, 80, 5);
      std::size_t count[3];
      for (int run = 0; run < 3; ++run) {
        const spec s = {objective(o), 1.0, single, std::size_t(run == 2 ? 6 : 2), 1, false, 0};
        const std::size_t before = allocations;
        solve_spec(p, s);
        count[run] = allocations - before;
      }
      const bool bad = count[2] != count[1];
      failures += bad;
      std::printf("%-18s %6s: %+lld allocations in 4 more epochs%s\n", names[o],
                  single ? "float" : "double", (long long)count[2] - (long long)count[1],
                  bad ? "  FAIL" : "");
    }
  return failures;
}

// Seconds per epoch and per evaluation, new and reference, on dense features.
void timing(std::size_t n, std::size_t d, const std::vector<std::size_t>& ms, int objectives) {
  std::printf("n=%zu dim=%zu, one thread, shrinking off; seconds per epoch / evaluation\n", n, d);
  std::printf("%-18s %5s %6s %10s %10s %10s %10s %9s %9s %9s\n", "objective", "m", "type", "epoch",
              "ref", "eval", "ref", "d primal", "d gap", "d W");
  for (std::size_t m : ms) {
    const problem p = make_problem(n, d, m, 1 + m);
    for (int o = 0; o < num_objectives; ++o) {
      if (!(objectives & (1 << o))) continue;
      for (bool single : {true, false}) {
        const spec s = {objective(o), 1.0, single, 2, 1, false, 0};
        const fit got = solve_spec(p, s), want = reference_fit(p, s);
        std::printf("%-18s %5zu %6s %10.4f %10.4f %10.4f %10.4f %9.2g %9.2g %9.2g\n", names[o], m,
                    single ? "float" : "double", got.epoch_seconds, want.epoch_seconds,
                    got.eval_seconds, want.eval_seconds, relative(got.primal, want.primal),
                    relative(got.primal - got.dual, want.primal - want.dual),
                    max_difference(got.W, want.W));
        std::fflush(stdout);
      }
    }
  }
}

}  // namespace perf

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "time") == 0) {
    const std::size_t n = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 20000;
    const std::size_t d = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 512;
    std::vector<std::size_t> ms;
    for (int a = 4; a < argc; ++a) ms.push_back(std::strtoul(argv[a], nullptr, 10));
    if (ms.empty()) ms = {100, 1000};
    const char* only = std::getenv("SDCA_PERF_OBJECTIVES");
    perf::timing(n, d, ms, only ? std::atoi(only) : 15);
    return 0;
  }
  const int failures = perf::smoke() + perf::step_allocations();
  return failures == 0 ? 0 : 1;
}

#endif
