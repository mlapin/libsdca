#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

//#include "common.hpp"
//#include "math_util.hpp"

#include "prox/topk_simplex_biased.h"

using namespace sdca;



//#include <fstream>
//#include <iterator>

//  std::ofstream myfile;
//  myfile.open("Xorig.txt", std::ios::out | std::ios::app);
//  std::copy(variables, variables_back, std::ostream_iterator<RealType>(myfile, ","));
//  myfile << std::endl;
//  myfile.close();

//  myfile.open("Params.txt", std::ios::out | std::ios::app);
//  myfile << projector_.get_hi() << "," << projector_.get_rhs() << "," << projector_.get_rho() << std::endl;
//  myfile.close();

//  projector_.Project(variables, variables_back, scores);

//  myfile.open("Xproj.txt", std::ios::out | std::ios::app);
//  std::copy(variables, variables_back, std::ostream_iterator<RealType>(myfile, ","));
//  myfile << std::endl;
//  myfile.close();


template <typename T = double>
void display_vector(const std::vector<T> &v) {
  std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

int main() {
  std::mt19937 gen(0);
  std::normal_distribution<> d(0,1);

  using RealType = double;
  using IndexType = std::vector<RealType>::difference_type;
  using SizeType = std::vector<RealType>::size_type;

  SizeType m = 10;
  SizeType n = 1;
  std::vector<double> A, Ki, scores;
  A.reserve(m*n);
  for (IndexType i = 0; i < static_cast<IndexType>(m*n); ++i) {
    A.push_back(d(gen));
  }
  Ki.reserve(n);
  for (IndexType i = 0; i < static_cast<IndexType>(n); ++i) {
    Ki.push_back(d(gen));
  }
  scores.resize(m);

  display_vector(A);
//  auto t = thresholds_knapsack_eq(A.begin(), A.end());
//  project_topk_simplex_biased(A.begin(), A.end());
//  display_vector(A);
//  std::cout << "t = " << t.t << "; lo = " << t.lo <<
//               "; hi = " << t.hi << std::endl;

  double elapsedTime1(0);
//  double elapsedTime2(0);
//  const char *noTrans = "N";
//  const char *trans = "T";
//  double alpha = 1;
//  double beta = 0;
//  const IndexType kIndexOne = static_cast<IndexType>(1);

//  for (IndexType j = 0; j < 20; ++j) {

//    std::clock_t start = std::clock();

//    for (IndexType i = 0; i < n; ++i) {
//      dgemv(noTrans, &m, &n, &alpha, &A[0], &m, &Ki[0], &kIndexOne,
//           &beta, &scores[0], &kIndexOne);
//    }

//    elapsedTime1 += static_cast<double>(std::clock() - start) /
//      CLOCKS_PER_SEC;

//    start = std::clock();

//    for (IndexType i = 0; i < n; ++i) {
//      dgemv(trans, &n, &m, &alpha, &A[0], &n, &Ki[0], &kIndexOne,
//           &beta, &scores[0], &kIndexOne);
//    }

//    elapsedTime2 += static_cast<double>(std::clock() - start) /
//      CLOCKS_PER_SEC;
//  }

  std::cout << "time (no trans) = " << elapsedTime1 << std::endl;
  std::cout << "time (trans) = " << elapsedTime1 << std::endl;
  std::cout << "sizeof(int) = " << sizeof(int) << std::endl;
  std::cout << "sizeof(long int) = " << sizeof(long int) << std::endl;
  std::cout << "sizeof(long unsigned int) = " << sizeof(long unsigned int) << std::endl;
  std::cout << "sizeof(std::size_t) = " << sizeof(std::size_t) << std::endl;
  std::cout << "sizeof(std::vector<double>::size_type) = " << sizeof(std::vector<double>::size_type) << std::endl;
  std::cout << "sizeof(std::vector<float>::size_type) = " << sizeof(std::vector<float>::size_type) << std::endl;
  std::cout << "sizeof(IndexType) = " << sizeof(IndexType) << std::endl;
#ifdef MKL_INT
  std::cout << "sizeof(MKL_INT) = " << sizeof(MKL_INT) << std::endl;
#endif

  long double s, t = 3.0;
  s = 1.0 - (4.0/t - 1.0)*t;
  std::cout << "s = " << s << std::endl;

  return 0;
}
