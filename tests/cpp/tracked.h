#ifndef SDCA_TEST_TRACKED_H
#define SDCA_TEST_TRACKED_H

// Tracked expectations: checks whose expected outcome another owner decides. Each carries a
// reason of the form "[owner] why". Its failure is expected and counted (XFAIL); its pass is
// a failure (XPASS), as a strict xfail is, until its entry is removed.

#include <cstdio>
#include <string>

namespace tracked {

struct counts {
  int failures = 0, expected_failures = 0;
};

// The outcome of a check at where: what failed ("" if it passes) and the reason if it is
// tracked (null if not).
inline void settle(counts& n, const std::string& where, const std::string& what,
                   const char* reason) {
  if (reason && !what.empty()) {
    ++n.expected_failures;
    std::fprintf(stderr, "XFAIL %s: %s (%s)\n", where.c_str(), what.c_str(), reason);
  } else if (reason) {
    ++n.failures;
    std::fprintf(stderr, "XPASS %s: passes; remove its entry (%s)\n", where.c_str(), reason);
  } else if (!what.empty()) {
    ++n.failures;
    std::fprintf(stderr, "FAIL %s: %s\n", where.c_str(), what.c_str());
  }
}

}  // namespace tracked

#if defined(GTEST_API_)
#include <exception>

#include "gtest/gtest-spi.h"

namespace tracked {

// body() as a test tracked for reason (null: an ordinary test): its failures, fatal ones and
// exceptions too, are expected, and a pass fails the test.
inline void run(const char* reason, void (*body)()) {
  if (!reason) {
    body();
    return;
  }
  ::testing::TestPartResultArray results;
  {
    ::testing::ScopedFakeTestPartResultReporter intercept(
        ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
    try {
      body();
    } catch (const std::exception& e) {
      ADD_FAILURE() << "C++ exception: " << e.what();
    } catch (...) {
      ADD_FAILURE() << "unknown C++ exception";
    }
  }
  const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string what;
  for (int i = 0; i < results.size() && what.empty(); ++i)
    if (results.GetTestPartResult(i).failed()) what = results.GetTestPartResult(i).message();
  for (char& c : what)
    if (c == '\n') c = ' ';
  counts n;
  settle(n, std::string(info->test_case_name()) + "." + info->name(), what, reason);
  if (n.failures) ADD_FAILURE() << "passes; remove its tracked entry (" << reason << ")";
}

}  // namespace tracked

// TEST(suite, name) tracked for reason (a const char*; null runs it as an ordinary test).
#define TRACKED_TEST(suite, name, reason)                                     \
  static void suite##_##name##_tracked_body();                                \
  TEST(suite, name) { tracked::run(reason, &suite##_##name##_tracked_body); } \
  static void suite##_##name##_tracked_body()
#endif

#endif
