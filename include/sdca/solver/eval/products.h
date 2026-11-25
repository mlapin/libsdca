#include "sdca/core/isa.h"
#if SDCA_ISA_FIRST(SDCA_SOLVER_EVAL_PRODUCTS_H)
#if SDCA_ISA_ID == 1
#define SDCA_SOLVER_EVAL_PRODUCTS_H1 1
#elif SDCA_ISA_ID == 2
#define SDCA_SOLVER_EVAL_PRODUCTS_H2 1
#elif SDCA_ISA_ID == 3
#define SDCA_SOLVER_EVAL_PRODUCTS_H3 1
#else
#define SDCA_SOLVER_EVAL_PRODUCTS_H4 1
#endif

// The solver's products of weights, examples and duals on lanes, accumulated in double
// (float entries widened exactly), in summation orders that the sizes alone fix: every
// instruction set gives the same bits. Dot products (scores, norms) add entry j into partial
// sum j % 16, a partial sum taking dot_chunk(n) blocks of 16 entries before its total takes
// it, the tail last, and the 16 totals pairwise; sums over examples (X' A, A' k) add in
// example order, in chunks whose partial sums the total takes in order.
#include <algorithm>
#include <cmath>
#include <cstddef>

#include "sdca/core/lanes.h"
#include "sdca/core/storage.h"

// The loops over a tile's lanes unroll before GCC splits its arrays into registers.
#ifndef SDCA_TILE
#define SDCA_TILE _Pragma("GCC unroll 16")
#endif

SDCA_ISA_BEGIN
namespace products {

typedef core::isa::native Isa;
const std::size_t kSums = 8;
typedef core::Lanes<double, Isa, int(kSums) / core::Register<double, Isa>::L> Sums;

// Dot products of E examples with R rows of W at once (they share the loads); R rows of
// sums over examples at once.
#if SDCA_ISA_ID == 4
const int kTileExamples = 2, kTileRows = 8;
#elif SDCA_ISA_ID == 1
const int kTileExamples = 1, kTileRows = 1;
#else
const int kTileExamples = 1, kTileRows = 4;
#endif

// The blocks of 8 entries a partial sum of a dot product of n entries takes: at least 32,
// else about the root of the blocks, so that the rounding grows with dot_steps(n), not n.
inline std::size_t dot_chunk(const std::size_t n) {
  const std::size_t blocks = n / kSums;
  std::size_t chunk = 32;
  while (chunk * chunk < blocks) chunk *= 2;
  return chunk;
}

// Roundings per term of a dot product of n entries: the product, the partial sum, its
// total, the tail and the three pairwise sums.
inline double dot_steps(const std::size_t n) {
  const std::size_t blocks = n / kSums, chunk = dot_chunk(n);
  return static_cast<double>(std::min(blocks, chunk) + blocks / chunk + 8);
}

// sums[q] += the lane sums of the products of w[r] and x[e] over [j0, j1), q = e * R + r:
// count < kSums entries of the tail padded with zeros, else whole blocks. The sums stay out
// of the registers, which the partial sums of a chunk fill.
template <int E, int R, typename Wt, typename Xt>
SDCA_FLAT inline void dot_part(const std::size_t j0, const std::size_t j1, const int count,
                               const Wt* const* w, const Xt* const* x, Sums* sums) {
  Sums part[E * R];
  SDCA_TILE
  for (int q = 0; q < E * R; ++q) part[q] = Sums::zero();
  for (std::size_t j = j0; j < j1; j += kSums) {
    Sums xs[E];
    SDCA_TILE
    for (int e = 0; e < E; ++e)
      xs[e] =
          count < int(kSums) ? core::load<Sums>(x[e] + j, count, 0.0) : core::load<Sums>(x[e] + j);
    SDCA_TILE
    for (int r = 0; r < R; ++r) {
      const Sums ws =
          count < int(kSums) ? core::load<Sums>(w[r] + j, count, 0.0) : core::load<Sums>(w[r] + j);
      SDCA_TILE
      for (int e = 0; e < E; ++e) part[e * R + r] += ws * xs[e];
    }
  }
  SDCA_TILE
  for (int q = 0; q < E * R; ++q) sums[q] += part[q];
}

// out[e * ldo + r] = <w[r], x[e]> for E examples and R rows.
template <int E, int R, typename Wt, typename Xt, typename O>
inline void dot_tile(const std::size_t n, const Wt* const* w, const Xt* const* x, O* out,
                     const std::size_t ldo) {
  Sums sums[E * R];
  SDCA_TILE
  for (int q = 0; q < E * R; ++q) sums[q] = Sums::zero();
  const std::size_t full = n - n % kSums, step = dot_chunk(n) * kSums;
  for (std::size_t j0 = 0; j0 < full; j0 += step)
    dot_part<E, R>(j0, std::min(full, j0 + step), int(kSums), w, x, sums);
  if (full < n) dot_part<E, R>(full, full + 1, static_cast<int>(n - full), w, x, sums);
  SDCA_TILE
  for (int e = 0; e < E; ++e) SDCA_TILE
  for (int r = 0; r < R; ++r) out[e * ldo + r] = static_cast<O>(core::sum(sums[e * R + r]));
}

// <w, x> of n < kSums entries, in the order of dot_tile without its lanes: lane j holds
// 0 + (0 + w_j x_j) (the padding +0), and the lanes total as core::sum adds them.
template <typename Wt, typename Xt>
inline double short_dot(const std::size_t n, const Wt* w, const Xt* x) {
  double lane[kSums];
  for (std::size_t j = 0; j < kSums; ++j) {
    double part = 0, sum = 0;
    if (j < n) part += static_cast<double>(w[j]) * static_cast<double>(x[j]);
    sum += part;
    lane[j] = sum;
  }
  return core::detail::pairwise<double, int(kSums)>(lane, core::detail::Add());
}

template <int E, int R, typename Wt, typename Xt, typename O>
inline void dot_rows(const std::size_t n, const std::size_t r0, const std::size_t e0, const Wt* W,
                     const std::size_t ldw, const std::size_t* index, const Xt* X,
                     const std::size_t ldx, O* out, const std::size_t ldo) {
  const Wt* w[R];
  const Xt* x[E];
  for (int r = 0; r < R; ++r) w[r] = W + ldw * (index ? index[r0 + r] : r0 + r);
  for (int e = 0; e < E; ++e) x[e] = X + ldx * (e0 + e);
  dot_tile<E, R>(n, w, x, out + ldo * e0 + r0, ldo);
}

// out[e * ldo + r] = <W_r, x_e> for the rows r < rows of W (row r at W + ldw * index[r], or
// ldw * r without index) and the examples e < count (at X + ldx * e), n entries each.
template <typename Wt, typename Xt, typename O>
inline void dots(const std::size_t n, const std::size_t rows, const Wt* W, const std::size_t ldw,
                 const std::size_t* index, const std::size_t count, const Xt* X,
                 const std::size_t ldx, O* out, const std::size_t ldo) {
  if (n < kSums) {
    for (std::size_t e = 0; e < count; ++e)
      for (std::size_t r = 0; r < rows; ++r)
        out[ldo * e + r] =
            static_cast<O>(short_dot(n, W + ldw * (index ? index[r] : r), X + ldx * e));
    return;
  }
  const int E = kTileExamples, R = kTileRows;
  std::size_t r = 0;
  for (; r + R <= rows; r += R) {
    std::size_t e = 0;
    for (; e + E <= count; e += E) dot_rows<E, R>(n, r, e, W, ldw, index, X, ldx, out, ldo);
    for (; e < count; ++e) dot_rows<1, R>(n, r, e, W, ldw, index, X, ldx, out, ldo);
  }
  for (; r < rows; ++r) {
    std::size_t e = 0;
    for (; e + E <= count; e += E) dot_rows<E, 1>(n, r, e, W, ldw, index, X, ldx, out, ldo);
    for (; e < count; ++e) dot_rows<1, 1>(n, r, e, W, ldw, index, X, ldx, out, ldo);
  }
}

// out[e] = <a_e, b_e> for the pairs e < count (at A + lda * e and B + ldb * e), in the order
// of dots.
template <typename T, typename U, typename O>
inline void pair_dots(const std::size_t n, const std::size_t count, const T* A,
                      const std::size_t lda, const U* B, const std::size_t ldb, O* out) {
  for (std::size_t e = 0; e < count; ++e) {
    const T* a = A + lda * e;
    const U* b = B + ldb * e;
    if (n < kSums)
      out[e] = static_cast<O>(short_dot(n, a, b));
    else
      dot_tile<1, 1>(n, &a, &b, out + e, 1);
  }
}

// W_r += x * (-delta[r]) in T, entry by entry, for the rows r < rows (as in dots) of nonzero
// delta: a plain loop, which the compilers vectorize for the pass's instruction set.
template <typename T, typename Xt>
inline void rank_one_update(const std::size_t n, const std::size_t rows, T* W,
                            const std::size_t ldw, const std::size_t* index, const Xt* x,
                            const T* delta) {
  for (std::size_t r = 0; r < rows; ++r) {
    if (delta[r] == 0) continue;
    const T scale = -delta[r];
    T* w = W + ldw * (index ? index[r] : r);
    for (std::size_t j = 0; j < n; ++j) w[j] += static_cast<T>(x[j]) * scale;
  }
}

// out[c] = sum_l A[m l + c] k[l] for the classes c < m over the rows l < n with k[l] != 0: in
// chunks of rows (row order within each) whose partial sums the total takes in order;
// magnitude[c], unless null, adds |A[m l + c] k[l]|.
template <typename T, typename K, typename O>
inline void column_sums(const std::size_t n, const std::size_t m, const T* A, const K* k,
                        const std::size_t chunk, O* out, double* magnitude) {
  const std::size_t full = m - m % kSums;
  for (std::size_t c = 0; c < full; c += kSums) {
    Sums total = Sums::zero(), size = Sums::zero();
    for (std::size_t l0 = 0; l0 < n; l0 += chunk) {
      Sums part = Sums::zero();
      for (std::size_t l = l0; l < std::min(n, l0 + chunk); ++l) {
        const double kl = static_cast<double>(k[l]);
        if (kl == 0) continue;
        const Sums product = core::load<Sums>(A + m * l + c) * kl;
        part += product;
        if (magnitude) size += core::abs(product);
      }
      total += part;
    }
    core::store(out + c, total);
    if (magnitude) (core::load<Sums>(magnitude + c) + size).store(magnitude + c);
  }
  for (std::size_t c = full; c < m; ++c) {
    double total = 0, size = 0;
    for (std::size_t l0 = 0; l0 < n; l0 += chunk) {
      double part = 0;
      for (std::size_t l = l0; l < std::min(n, l0 + chunk); ++l) {
        const double kl = static_cast<double>(k[l]);
        if (kl == 0) continue;
        const double product = static_cast<double>(A[m * l + c]) * kl;
        part += product;
        size += std::fabs(product);
      }
      total += part;
    }
    out[c] = static_cast<O>(total);
    if (magnitude) magnitude[c] += size;
  }
}

template <int R, typename Xt>
SDCA_FLAT inline void row_sums_tile(const std::size_t n, const std::size_t m,
                                    const std::size_t count, const Xt* X, const std::size_t ldx,
                                    const double* a, const std::size_t c, std::size_t* rows,
                                    double* H) {
  std::size_t used = 0;
  for (std::size_t r = 0; r < count; ++r) {
    bool zero = true;
    SDCA_TILE
    for (int q = 0; q < R; ++q) zero = zero && a[m * r + c + q] == 0;
    if (!zero) rows[used++] = r;
  }
  if (used == 0) return;
  for (std::size_t j = 0; j < n; j += kSums) {
    const int width = static_cast<int>(std::min(kSums, n - j));
    Sums sums[R];
    SDCA_TILE
    for (int q = 0; q < R; ++q) sums[q] = Sums::zero();
    for (std::size_t u = 0; u < used; ++u) {
      const std::size_t r = rows[u];
      const Xt* x = X + ldx * r + j;
      const Sums xs = width == int(kSums) ? core::load<Sums>(x) : core::load<Sums>(x, width, 0.0);
      SDCA_TILE
      for (int q = 0; q < R; ++q) sums[q] += xs * a[m * r + c + q];
    }
    SDCA_TILE
    for (int q = 0; q < R; ++q) {
      double* h = H + n * (c + q) + j;
      (core::load<Sums>(h, width, 0.0) + sums[q]).store(h, width);
    }
  }
}

// H[n c + j] += sum_r a[m r + c] x_r[j] for the classes c in [first, last), j < n, over the
// rows r < count of x (at X + ldx * r) in row order, rows of zero dual skipped; rows is
// scratch of count entries.
template <typename Xt>
inline void row_sums(const std::size_t n, const std::size_t m, const std::size_t count, const Xt* X,
                     const std::size_t ldx, const double* a, const std::size_t first,
                     const std::size_t last, std::size_t* rows, double* H) {
  const int R = kTileRows;
  std::size_t c = first;
  for (; c + R <= last; c += R) row_sums_tile<R>(n, m, count, X, ldx, a, c, rows, H);
  for (; c < last; ++c) row_sums_tile<1>(n, m, count, X, ldx, a, c, rows, H);
}

// The entries of x[0, n) at least value.
inline std::size_t count_at_least(const std::size_t n, const double* x, const double value) {
  typedef core::Lanes<double, Isa, 2> A;
  typename A::mask_type total = A::mask_type::zero();
  std::size_t j = 0;
  for (; j + A::L <= n; j += A::L) total = total - (A::load(x + j) >= value);
  std::size_t count = 0;
  for (int l = 0; l < A::L; ++l) count += static_cast<std::size_t>(total[l]);
  for (; j < n; ++j) count += x[j] >= value;
  return count;
}

// The largest |x[j]| of x[0, n) (0 if n is 0; NaN entries left out).
inline double max_abs(const std::size_t n, const double* x) {
  typedef core::Lanes<double, Isa, 2> A;
  A top = A::zero();
  std::size_t j = 0;
  for (; j + A::L <= n; j += A::L) top = core::max(core::abs(A::load(x + j)), top);
  double value = core::max_lane(top);
  for (; j < n; ++j) value = std::max(value, std::fabs(x[j]));
  return value;
}

}  // namespace products
SDCA_ISA_END

#endif  // SDCA_SOLVER_EVAL_PRODUCTS_H
