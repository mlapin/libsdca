#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>


template <typename T = double>
void display_vector(const std::vector<T> &v) {
  std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

int main() {
//  std::mt19937 gen(0);
//  std::normal_distribution<> d(0,1);

//  std::size_t k_ = 5;
//  double sum_pos = 0;
//  double max_elem = -std::numeric_limits<double>::infinity();
//  double kp1_elem = -std::numeric_limits<double>::infinity();

//  std::clock_t start = std::clock();
//  for (std::size_t i = 0; i < 100000; ++i) {

//    std::size_t n = 400;
//    std::vector<double> x;
//    x.reserve(n);
//    for (std::size_t j = 0; i < n; ++i) {
//      x.push_back(d(gen));
//    }


//    std::sort(x.begin(), x.end(), std::greater<double>());


//    max_elem = x[0];
//    kp1_elem = x[k_];
//    typename std::vector<double>::iterator it = x.begin();
//    for (; *it <= 0; ++it) {
//      sum_pos += *it;
//    }
/*
    typename std::vector<double>::iterator it = x.begin();
    for (; it != x.begin() + k_; ++it) {
      if (*it > 0) sum_pos += *it;
      if (*it > max_elem) max_elem = *it;
    }
    for (; it != x.end(); ++it) {
      if (*it > 0) sum_pos += *it;
      if (*it > kp1_elem) kp1_elem = *it;
    }*/
//  }

//  double elapsedTime = static_cast<double>(std::clock() - start) /
//    CLOCKS_PER_SEC;

//  std::cout << "time     = " << elapsedTime << std::endl;
//  std::cout << "sum_pos  = " << sum_pos << std::endl;
//  std::cout << "max_elem = " << max_elem << std::endl;
//  std::cout << "kp1_elem = " << kp1_elem << std::endl;

  return 0;
}
