#include <cmath>
#include <vector>

#include "projector.hh"

namespace sdca {

template <typename RealType>
Projector<RealType>::Projector() { }

template <typename RealType>
void Projector<RealType>::VectorToKSimplex(const std::size_t k,
  const std::size_t n, RealType *x) {

  RealType kinv = 1.0 / static_cast<RealType>(k);
  RealType norm_a_i = (kinv * n - 2.0) * kinv + 1;
  RealType lambda = static_cast<RealType>(n+1); // anything > (2*n+1)/2

  // Copy x to z (x is modified in place, z remains unchanged)
  std::vector<RealType> z(x, x + n);

  // Initialize P = [a_1, ..., a_n, b_1, ..., b_n, c]
  std::vector<RealType> p;
  p.assign(n*n, -kinv); // p_i = -1/k, i = 1, ..., n
  p.resize((2*n + 1)*n);  // p_i = 0, i = n+1, ..., 2*n+1
  for (std::size_t i = 0; i < n; ++i) {
    // a_i = e_i - 1/k
    p[(n + 1)*i] += 1.0;
    // b_i = -e_i
    p[n*n + (n + 1)*i] = -1.0;
    // c = ones
    p[2*n*n + i] = 1.0;
  }

  // Buffer
  std::vector<RealType> y;
  y.resize(n);

  // Compute the projection
  bool negative_found;
  obj_val_ = -1.0;
  for (iter_ = 0; iter_ < kMaxNumIterations; ++iter_) {
    RealType t;
    obj_old_ = obj_val_;
    obj_val_ = 0.0;

    // x = z - sum_i p_i
    negative_found = false;
    for (std::size_t i = 0; i < n; ++i) {
      t = 0.0;
      for (std::size_t j = 0; j < 2*n + 1; ++j) {
        t += p[n*j + i];
      }
      x[i] = z[i] - t;
      negative_found |= x[i] < static_cast<RealType>(0.0);
      obj_val_ += t*t;
    }
    obj_val_ /= static_cast<RealType>(n);

    bool converged = std::abs(obj_val_ - obj_old_) < kObjectiveChangeEpsilon;
    if (negative_found && (converged || iter_ + 2 >= kMaxNumIterations)) {
      // close to finish - threshold negatives to zero and re-run
      for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::max(static_cast<RealType>(0.0), x[i]);
      }
    } else if (converged) {
      break;
    }

    // p_i = 1/lambda * (I - P_K_i)(x + lambda*p_i)
    for (std::size_t j = 0; j < 2*n + 1; ++j) {
      RealType sum_y = 0.0;

      // y = x + lambda*p_i
      for (std::size_t i = 0; i < n; ++i) {
        y[i] = x[i] + lambda*p[n*j + i];
        sum_y += y[i];
      }

      // p_i = t/lambda * v, where v = a_i, or b_i, or c

      if (j < n) {
        // <a_i,y> = y_i - 1/k sum_y
        // <a_i,a_i> = (n/k - 2)/k + 1
        t = std::max(static_cast<RealType>(0.0),
          (y[j] - kinv * sum_y) / norm_a_i);
        std::fill_n(&p[n*j], n, -kinv*t/lambda);
        p[(n + 1)*j] += t/lambda;
      } else if (j < 2*n) {
        // <b_i,y> = -y_i
        // <b_i,b_i> = 1
        t = std::max(static_cast<RealType>(0.0), -y[j-n]);
        p[n*n + (n + 1)*(j - n)] = -t/lambda;
      } else {
        // <c,y> = sum_y
        // <c,c> = n
        t = std::max(static_cast<RealType>(0.0),
          static_cast<RealType>((sum_y - 1.0) / n));
        std::fill_n(&p[2*n*n], n, t/lambda);
      }
    }
  }
}

template <typename RealType>
void Projector<RealType>::MatrixToKSimplex(const std::size_t k,
  const std::size_t n, const std::size_t m, RealType *x) {
  for (std::size_t i = 0; i < m; ++i) {
    this->VectorToKSimplex(k, n, x + n*i);
  }
}

template class Projector<float>;
template class Projector<double>;

}
