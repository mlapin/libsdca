#include "test_util.h"
#include "sdca/utility/logging.cpp"

GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from test_main.cc\n");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
