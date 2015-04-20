#include <cmath>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "projector.hh"

template <typename T = double>
void display_vector(const std::vector<T> &v) {
  std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

int main() {
  std::mt19937 gen(0);
  std::normal_distribution<> d(0,1);

  std::size_t n = 5;
  std::vector<double> x;
  x.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    x.push_back(d(gen));
  }

  // Before projection
  display_vector(x);

  sdca::Projector<> proj;
  proj.VectorToKSimplex(2, x.size(), x.data());

  // After projection
  display_vector(x);

  std::cout << "Obj (old) : " << proj.ObjectiveValueOld() << std::endl;
  std::cout << "Obj (new) : " << proj.ObjectiveValue() << std::endl;
  std::cout << "Obj (dif) : "
    << std::abs(proj.ObjectiveValueOld() - proj.ObjectiveValue()) << std::endl;
  std::cout << "Iteration : " << proj.Iteration() << std::endl;

  return 0;
}
