#include <cstring>

#include "sdca/prox/proxdef.h"
#include "test_util.h"

// Finite values of Type nearest to x.
template <typename Type>
inline Type narrow(const double x) {
  const double top = double(std::numeric_limits<Type>::max());
  return Type(std::max(-top, std::min(x, top)));
}

// The partition through the cut of a monotone test equals std::partition
// with the test itself: the same elements in the same order.
template <typename Type>
inline void test_partition_monotone_matches(std::mt19937& gen, const double origin, const double t,
                                            const double bound, const std::size_t size) {
  std::normal_distribution<double> normal(0, 1);
  std::uniform_int_distribution<int> pick(0, 5);
  const double spread = std::max({1.0, std::abs(origin), std::abs(t), std::abs(bound)});
  std::vector<Type> v(size);
  for (auto& x : v) {
    switch (pick(gen)) {
      case 0:
        x = Type(0);
        break;
      case 1:
        x = -Type(0);
        break;
      case 2:
        x = narrow<Type>((bound + t) + origin);
        break;
      default:
        x = narrow<Type>(spread * normal(gen));
    }
  }
  const auto test = [=](const Type x) { return (double(x) - origin) - t > bound; };
  std::vector<Type> expected = v, actual = v;
  const auto e = std::partition(expected.begin(), expected.end(), test);
  const auto a =
      sdca::detail::partition_monotone(actual.begin(), actual.end(), test, (bound + t) + origin);
  ASSERT_EQ(e - expected.begin(), a - actual.begin());
  ASSERT_EQ(0, std::memcmp(expected.data(), actual.data(), size * sizeof(Type)));
}

template <typename Type>
inline void test_partition_monotone() {
  std::mt19937 gen(4);
  const std::vector<double> values = {0.0, -0.0, 1e-300, -3.5, 1.0, 1e8, -1e16, 1e300};
  for (double origin : values)
    for (double t : values)
      for (double bound : {0.0, 1e-3, 2.0, -1e10})
        for (std::size_t size : {std::size_t(31), std::size_t(64), std::size_t(500)})
          test_partition_monotone_matches<Type>(gen, origin, t, bound, size);
}

TEST(ProxPartition, monotone_matches_std_partition_float) { test_partition_monotone<float>(); }

TEST(ProxPartition, monotone_matches_std_partition_double) { test_partition_monotone<double>(); }

TEST(ProxPartition, monotone_cut_brackets_the_test) {
  for (double guess : {0.0, 1e300, -1e300, 1e-310, std::nan("")}) {
    const auto always = [](double) { return true; };
    const auto never = [](double) { return false; };
    ASSERT_EQ(-std::numeric_limits<double>::infinity(),
              sdca::detail::monotone_cut<double>(guess, always));
    ASSERT_EQ(std::numeric_limits<double>::max(), sdca::detail::monotone_cut<double>(guess, never));
    for (double cut : {-1e200, -1.0, -0.0, 0.0, 1e-320, 2.5, 1e300}) {
      const auto above = [cut](double x) { return x > cut; };
      const double found = sdca::detail::monotone_cut<double>(guess, above);
      ASSERT_FALSE(above(found));
      ASSERT_TRUE(above(std::nextafter(found, std::numeric_limits<double>::infinity())));
    }
  }
}
