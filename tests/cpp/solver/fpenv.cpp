#include <atomic>
#include <cfenv>
#include "sdca/utility/fpenv.h"
#include "sdca/utility/parallel.h"
#include "test_util.h"

namespace {
using sdca::detail::fp_control;
using sdca::detail::set_fp_control;

double underflow(double x) {
  volatile double v = x;
  return v * 1e-10;
}
}  // namespace

TEST(FpenvTest, ieee_scope_keeps_subnormals_and_restores_the_caller) {
  const std::uint64_t flush = sdca::detail::flush_to_zero_bits();
  if (flush == 0) return;  // No flush-to-zero control on this target.
  const std::uint64_t original = fp_control();
  set_fp_control(original | flush);
  std::fesetround(FE_UPWARD);
  const bool flushes = underflow(1e-300) == 0;
  const std::uint64_t caller = fp_control();
  double inside = 0;
  int rounding = 0;
  {
    sdca::detail::ieee_environment ieee;
    inside = underflow(1e-300);
    rounding = std::fegetround();
  }
  const std::uint64_t after = fp_control();
  const int restored = std::fegetround();
  std::fesetround(FE_TONEAREST);
  set_fp_control(original);
  EXPECT_TRUE(flushes);
  EXPECT_GT(inside, 0);
  EXPECT_EQ(FE_TONEAREST, rounding);
  EXPECT_EQ(caller, after);
  EXPECT_EQ(FE_UPWARD, restored);
}

TEST(FpenvTest, workers_take_the_callers_environment) {
  const std::uint64_t flush = sdca::detail::flush_to_zero_bits();
  const std::uint64_t original = fp_control();
  for (std::uint64_t bits : {std::uint64_t(0), flush}) {
    set_fp_control(original | bits);
    const std::uint64_t caller = fp_control() & flush;
    std::atomic<int> mismatches(0);
    // Every call of two rows or more takes threads.
    const double saved = sdca::detail::thread_work_setting();
    sdca::detail::thread_work_setting() = 0;
    sdca::detail::parallel_rows(64, 1, 4, [&](std::size_t, std::size_t) {
      if ((fp_control() & flush) != caller) ++mismatches;
    });
    sdca::detail::thread_work_setting() = saved;
    set_fp_control(original);
    EXPECT_EQ(0, mismatches.load()) << bits;
  }
}
