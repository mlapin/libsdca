// Smoke test of the projections of sdca/prox.h: a hash of the output bits of every operator on
// random, tied, scaled, near-overflow and subnormal rows in float, double and long double
// storage, to compare two builds bit for bit; the time of each operator; and, with
// SDCA_SMOKE_PRECISION, the largest error of the float and double solutions against the
// long double one (where long double is wider than double).
//
//   c++ -std=c++11 -O3 -Wall -Wextra -Wshadow -Werror -Iinclude THIS -o smoke
//   ./smoke [repetitions] [dump|precision]   prints "op type hash seconds" per operator; with
//   precision, also the largest error of float and double against long double per operator
//   (in eps of the type, relative to the row's largest output)
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "sdca/prox.h"

namespace {

typedef std::vector<long double> Row;

// The rows: sizes from 1 to 300, each of several shapes and scales.
std::vector<Row> make_rows(const int count) {
  std::mt19937_64 rng(20261119);
  std::uniform_real_distribution<double> unit(-1, 1);
  std::vector<Row> rows;
  const int sizes[] = {1, 2, 3, 5, 8, 17, 33, 64, 100, 300};
  const double scales[] = {1, 1e-3, 1e3, 1e-150, 1e150, 1e-300, 1e300, 1e-310};
  for (int i = 0; i < count; ++i) {
    const int n = sizes[i % 10];
    const int shape = (i / 10) % 6;
    const double scale = scales[(i / 60) % 8];
    Row r(std::size_t(n), 0.0L);
    for (int j = 0; j < n; ++j) {
      double x = unit(rng);
      if (shape == 1) x = std::floor(4 * x) / 4;  // ties
      if (shape == 2) x = x > 0.9 ? 30 * x : x;   // outliers
      if (shape == 3) x = 1 + x * 1e-12;          // near-equal
      if (shape == 4) x = std::exp(20 * x);       // spread over magnitudes
      if (shape == 5) x = j == 0 ? 1e6 : x;       // one far above
      r[std::size_t(j)] = static_cast<long double>(x * scale);
    }
    rows.push_back(r);
  }
  return rows;
}

struct Hash {
  std::uint64_t h = 1469598103934665603ULL;
  template <typename T>
  void add(const T x) {
    unsigned char b[sizeof(T)];
    std::memset(b, 0, sizeof b);
    std::memcpy(b, &x, std::numeric_limits<T>::digits == 64 ? 10 : sizeof(T));
    for (std::size_t i = 0; i < sizeof b; ++i) h = (h ^ b[i]) * 1099511628211ULL;
  }
};

bool dump = false, precision = false;
// The outputs of each operator and type (precision), a row of -1 where it threw.
std::map<std::string, std::vector<std::vector<long double>>> outputs;

// The operator over every row (k and the parameters from the row's index), hashing the
// outputs; a row the operator refuses hashes its exception.
template <typename T, typename Op>
void run(const char* name, const char* type, const std::vector<Row>& rows, const int reps, Op op) {
  Hash hash;
  std::vector<T> x, aux;
  long errors = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int rep = 0; rep < reps; ++rep) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const Row& r = rows[i];
      x.assign(r.begin(), r.end());
      for (std::size_t j = 0; j < x.size(); ++j) x[j] = T(r[j]);
      aux.resize(x.size());
      try {
        op(x, aux, i);
      } catch (const std::exception& e) {
        ++errors;
        x.assign(1, T(-1));
      }
      if (rep == 0) {
        if (precision)
          outputs[std::string(name) + " " + type].push_back(
              std::vector<long double>(x.begin(), x.end()));
        for (const T v : x) hash.add(v);
        if (dump) {
          std::printf("D %s %s %zu", name, type, i);
          for (const T v : x) std::printf(" %La", static_cast<long double>(v));
          std::printf("\n");
        }
      }
    }
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("%-22s %-11s %016llx errors %4ld seconds %.4f\n", name, type,
              static_cast<unsigned long long>(hash.h), errors, seconds);
}

template <typename T>
void all(const char* type, const std::vector<Row>& rows, const int reps) {
  typedef std::vector<T> V;
  auto k_of = [](const V& x, std::size_t i) { return std::ptrdiff_t(1 + i % x.size()); };
  run<T>("knapsack_eq", type, rows, reps, [](V& x, V& a, std::size_t i) {
    const T hi = T(1 + i % 3) / T(3), rhs = T(x.size()) * hi * T(0.5);
    sdca::prox_knapsack_eq<T>(x.begin(), x.end(), a.begin(), T(0), hi, rhs);
  });
  run<T>("knapsack_eq_signed", type, rows, reps, [](V& x, V& a, std::size_t i) {
    const T lo = -T(1 + i % 2), hi = T(2), rhs = T(i % 5) - T(2);
    sdca::prox_knapsack_eq<T>(x.begin(), x.end(), a.begin(), lo, hi, rhs);
  });
  run<T>("knapsack_le", type, rows, reps, [](V& x, V& a, std::size_t i) {
    sdca::prox_knapsack_le<T>(x.begin(), x.end(), a.begin(), T(0), T(1), T(1 + i % 4) / T(4));
  });
  run<T>("knapsack_le_biased", type, rows, reps, [](V& x, V& a, std::size_t i) {
    sdca::prox_knapsack_le_biased<T>(x.begin(), x.end(), a.begin(), T(0), T(1), T(1),
                                     T(i % 3) / T(2));
  });
  run<T>("topk_cone", type, rows, reps, [&](V& x, V& a, std::size_t i) {
    sdca::prox_topk_cone<T>(x.begin(), x.end(), a.begin(), k_of(x, i));
  });
  run<T>("topk_cone_biased", type, rows, reps, [&](V& x, V& a, std::size_t i) {
    sdca::prox_topk_cone_biased<T>(x.begin(), x.end(), a.begin(), k_of(x, i), T(0.5));
  });
  run<T>("topk_simplex", type, rows, reps, [&](V& x, V& a, std::size_t i) {
    sdca::prox_topk_simplex<T>(x.begin(), x.end(), a.begin(), k_of(x, i), T(1));
  });
  run<T>("topk_simplex_biased", type, rows, reps, [&](V& x, V& a, std::size_t i) {
    sdca::prox_topk_simplex_biased<T>(x.begin(), x.end(), a.begin(), k_of(x, i), T(1), T(2));
  });
  run<T>("two_simplex", type, rows, reps, [](V& x, V& a, std::size_t i) {
    if (x.size() < 2) return;
    const std::ptrdiff_t p = std::ptrdiff_t(1 + i % (x.size() - 1));
    sdca::prox_two_simplex<T>(p, x.begin(), x.end(), a.begin(), T(1 + i % 2));
  });
  run<T>("entropy", type, rows, reps, [](V& x, V& a, std::size_t i) {
    const T hi = T(1) / T(1 + i % 3);
    if (T(x.size()) * hi < T(1)) return;
    sdca::prox_entropy<T>(x.begin(), x.end(), a.begin(), hi, T(1));
  });
  run<T>("entropy_norm", type, rows, reps, [](V& x, V& a, std::size_t i) {
    const T hi = T(1) / T(1 + i % 3);
    if (T(x.size()) * hi < T(1)) return;
    sdca::prox_entropy_norm<T>(x.begin(), x.end(), a.begin(), hi, T(1));
  });
  run<T>("topk_entropy", type, rows, reps, [&](V& x, V& a, std::size_t i) {
    sdca::prox_topk_entropy<T>(x.begin(), x.end(), a.begin(), k_of(x, i));
  });
  run<T>("topk_entropy_biased", type, rows, reps, [&](V& x, V& a, std::size_t i) {
    const T alpha = T(std::pow(10.0, double(int(i % 7) - 3)));
    sdca::prox_topk_entropy_biased<T>(x.begin(), x.end(), a.begin(), k_of(x, i), alpha);
  });
  run<T>("two_entropy", type, rows, reps, [](V& x, V& a, std::size_t i) {
    if (x.size() < 2) return;
    const std::ptrdiff_t p = std::ptrdiff_t(1 + i % (x.size() - 1));
    const T alpha = T(std::pow(10.0, double(int(i % 7) - 3)));
    sdca::prox_two_entropy<T>(p, x.begin(), x.end(), a.begin(), alpha);
  });
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 1;
  dump = argc > 2 && std::string(argv[2]) == "dump";
  precision = argc > 2 && std::string(argv[2]) == "precision";
  const std::vector<Row> rows = make_rows(960);
  all<float>("float", rows, reps);
  all<double>("double", rows, reps);
  all<long double>("long_double", rows, reps);
  if (!precision) return 0;
  for (const auto& entry : outputs) {
    const std::string op = entry.first.substr(0, entry.first.find(' '));
    const std::string type = entry.first.substr(entry.first.find(' ') + 1);
    if (type == "long_double") continue;
    const auto& wide = outputs[op + " long_double"];
    const long double eps = type == "float"
                                ? static_cast<long double>(std::numeric_limits<float>::epsilon())
                                : static_cast<long double>(std::numeric_limits<double>::epsilon());
    long double worst = 0;
    std::size_t at = 0;
    for (std::size_t i = 0; i < wide.size(); ++i) {
      const auto &a = entry.second[i], &b = wide[i];
      if (a.size() != b.size()) continue;
      long double scale = 0;
      for (const long double v : b) scale = std::max(scale, std::abs(v));
      if (!(scale > 0) || !std::isfinite(scale)) continue;
      bool finite = true;
      for (const long double v : a) finite = finite && std::isfinite(v);
      if (!finite) continue;
      for (std::size_t j = 0; j < a.size(); ++j) {
        const long double e = std::abs(a[j] - b[j]) / scale / eps;
        if (e > worst) worst = e, at = i;
      }
    }
    std::printf("precision %-22s %-6s %10.3Lg eps (row %zu)\n", op.c_str(), type.c_str(), worst,
                at);
  }
  return 0;
}
