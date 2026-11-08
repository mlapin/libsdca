#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include "sdca/utility/parallel.h"
#include "test_util.h"
#ifdef SDCA_PARALLEL_PID
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#endif

namespace {
using sdca::detail::parallel_rows;

// Sets the work per added thread for a scope (0: every call of two rows or
// more adds threads after its first row).
struct thread_work {
  double saved;
  explicit thread_work(double us) : saved(sdca::detail::thread_work_setting()) {
    sdca::detail::thread_work_setting() = us;
  }
  ~thread_work() { sdca::detail::thread_work_setting() = saved; }
};

// Busy work of about us microseconds.
void spin(double us) {
  const auto end = std::chrono::steady_clock::now() +
                   std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                       std::chrono::duration<double, std::micro>(us));
  while (std::chrono::steady_clock::now() < end) {
  }
}

// How often parallel_rows visits each of n rows of row_size entries.
std::vector<int> visits(std::size_t n, std::size_t row_size, std::size_t num_threads) {
  std::vector<std::atomic<int>> count(n);
  for (auto& c : count) c = 0;
  parallel_rows(n, row_size, num_threads, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) ++count[i];
  });
  std::vector<int> result;
  for (auto& c : count) result.push_back(c.load());
  return result;
}

// The distinct threads that run a call of n rows of us microseconds each
// (rows of us entries: a microsecond per entry).
std::size_t threads_used(std::size_t n, std::size_t us, std::size_t num_threads) {
  std::mutex mutex;
  std::vector<std::thread::id> ids;
  parallel_rows(n, us, num_threads, [&](std::size_t begin, std::size_t end) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (std::find(ids.begin(), ids.end(), std::this_thread::get_id()) == ids.end())
        ids.push_back(std::this_thread::get_id());
    }
    spin(double(us * (end - begin)));
  });
  return ids.size();
}

// A call of five rows with threads forced: the caller runs the first row,
// then each expected thread holds a later row until all have arrived.
// Counting thread identities inside the row body avoids OS task-exit races
// and cannot miss a worker the scheduler ran early. With workers, calls
// inside the rows run serially; a serial call's rows (expected 1, e.g.
// under OMP_NUM_THREADS=1) may themselves take threads.
void check_workers(std::size_t requested, std::size_t expected) {
  const thread_work forced(0);
  std::mutex mutex;
  std::condition_variable arrived;
  std::vector<std::thread::id> ids;
  bool nested_elsewhere = false, timed_out = false;
  parallel_rows(5, 1, requested, [&](std::size_t begin, std::size_t) {
    const std::thread::id self = std::this_thread::get_id();
    {
      std::unique_lock<std::mutex> lock(mutex);
      if (std::find(ids.begin(), ids.end(), self) == ids.end()) ids.push_back(self);
      arrived.notify_all();
      if (begin > 0 &&
          !arrived.wait_for(lock, std::chrono::seconds(10), [&] { return ids.size() >= expected; }))
        timed_out = true;
    }
    if (expected > 1 && begin > 0) {
      parallel_rows(8, 1, 4, [&](std::size_t, std::size_t) {
        std::lock_guard<std::mutex> lock(mutex);
        nested_elsewhere = nested_elsewhere || std::this_thread::get_id() != self;
      });
    }
  });
  EXPECT_FALSE(timed_out);
  EXPECT_FALSE(nested_elsewhere);
  EXPECT_EQ(expected, ids.size());
  EXPECT_EQ(0u, sdca::detail::parallel_calls().load());
}

// The threads of an automatic call of check_workers.
std::size_t automatic_workers() {
  return std::min<std::size_t>(4, sdca::detail::automatic_threads());
}
}  // namespace

TEST(ParallelTest, every_row_once) {
  for (double us : {-1., 0.}) {
    const thread_work work(us);
    for (std::size_t threads : {0, 1, 2, 3, 8, 64}) {
      for (std::size_t n : {0, 1, 2, 7, 1000, 4099}) {
        for (std::size_t row_size : {1, 1000, 100000}) {
          EXPECT_EQ(std::vector<int>(n, 1), visits(n, row_size, threads))
              << threads << " threads, " << n << " x " << row_size;
        }
      }
    }
  }
  EXPECT_EQ(0u, sdca::detail::parallel_calls().load());
}

TEST(ParallelTest, threads_follow_the_work) {
  // 20 us of work stays on the caller; 40 ms takes threads where there are CPUs.
  for (std::size_t threads : {0, 8}) {
    EXPECT_EQ(1u, threads_used(20, 1, threads)) << threads;
    if (threads > 0 || sdca::detail::automatic_threads() > 1) {
      EXPECT_LT(1u, threads_used(400, 100, threads)) << threads;
    }
  }
  EXPECT_EQ(1u, threads_used(400, 100, 1));
}

TEST(ParallelTest, a_serial_calls_rows_may_take_threads) {
  // Rows that run with no workers beside them (here a call of one thread)
  // may take threads for calls inside them, as any caller.
  const thread_work forced(0);
  std::size_t inner = 0;
  parallel_rows(2, 1, 1, [&](std::size_t, std::size_t) { inner = threads_used(8, 2000, 4); });
  EXPECT_LT(1u, inner);
}

TEST(ParallelTest, uses_the_requested_threads_and_runs_nested_calls_serially) {
  for (std::size_t threads : {1, 2, 3, 4}) check_workers(threads, threads);
  check_workers(0, automatic_workers());
}

TEST(ParallelTest, rethrows_the_first_failure_of_a_serial_loop) {
  for (double us : {-1., 0.}) {
    const thread_work work(us);
    for (std::size_t threads : {0, 1, 2, 3, 8}) {
      for (std::size_t first : {0, 5, 2000, 4095}) {
        try {
          parallel_rows(4096, 1000, threads, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i)
              if (i == first || i == first + 1 || i == 4095)
                throw std::runtime_error(std::to_string(i));
          });
          ADD_FAILURE() << "no exception";
        } catch (const std::runtime_error& e) {
          EXPECT_EQ(std::to_string(first), e.what()) << threads << " threads";
        }
      }
    }
  }
  EXPECT_EQ(0u, sdca::detail::parallel_calls().load());
}

#ifdef SDCA_PARALLEL_PID
static void count_run(void* runs) { ++*static_cast<std::atomic<int>*>(runs); }

TEST(ParallelTest, a_forked_child_exits_normally_after_the_pool_ran) {
  // The child inherits the pool's condition variables, with the parent's sleeping workers
  // as waiters; destroying them at its exit blocked on Linux. alarm turns a hang into a
  // signal.
  std::atomic<int> runs(0);
  const sdca::detail::parallel_job job = {&count_run, &runs, sdca::detail::fp_modes::current()};
  ASSERT_TRUE(sdca::detail::worker_pool::instance().run(job, 2));
  std::fflush(nullptr);
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    alarm(20);
    // The workers do not exist here: the pool declines and the caller runs serially.
    std::exit(sdca::detail::worker_pool::instance().run(job, 2) ? 3 : 0);
  }
  int status = 0;
  ASSERT_EQ(child, waitpid(child, &status, 0));
  ASSERT_TRUE(WIFEXITED(status)) << "signal " << (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
  EXPECT_EQ(0, WEXITSTATUS(status));
}
#endif

TEST(ParallelTest, automatic_unless_asked) {
  using sdca::detail::default_threads;
  using sdca::detail::requested_threads;
  EXPECT_EQ(0u, requested_threads(0));
  EXPECT_EQ(5u, requested_threads(5));
  if (std::getenv("SDCA_NUM_THREADS")) return;  // The variable decides.
  EXPECT_EQ(0u, requested_threads(default_threads));
  check_workers(default_threads, automatic_workers());
}

TEST(ParallelEnvironmentTest, defaults_and_explicit_overrides) {
  using sdca::detail::default_threads;
  const char* text = std::getenv("SDCA_NUM_THREADS");
  const std::string setting = text ? text : "";
  // CTest launches each setting in a fresh process: the production environment
  // reader intentionally caches the first value it observes.
  ASSERT_TRUE(setting.empty() || setting == "1" || setting == "3" || setting == "0" ||
              setting == "many");
  const std::size_t expected = setting == "3" ? 3 : setting == "1" ? 1 : 0;
  EXPECT_EQ(expected, sdca::detail::requested_threads(default_threads));
  // OMP_NUM_THREADS=1 (as CI or a joblib worker sets it) leaves
  // automatic calls on their caller; explicit counts still take threads.
  const char* omp = std::getenv("OMP_NUM_THREADS");
  if (omp && std::string(omp) == "1") {
    EXPECT_EQ(1u, sdca::detail::automatic_threads());
  }
  check_workers(default_threads, expected == 0 ? automatic_workers() : expected);
  check_workers(1, 1);
  check_workers(4, 4);
  check_workers(0, automatic_workers());
}
