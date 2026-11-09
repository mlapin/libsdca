#ifndef SDCA_UTILITY_PARALLEL_H
#define SDCA_UTILITY_PARALLEL_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#include "sdca/utility/fpenv.h"
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#define SDCA_PARALLEL_PID 1
#endif

#if defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
// The OpenMP runtime of the process, if one is loaded (e.g. by PyTorch):
// its waiting team runs the rows, instead of new threads beside it.
extern "C" void sdca_gomp_parallel(void (*)(void*), void*, unsigned,
                                   unsigned) __asm__("GOMP_parallel") __attribute__((weak));
extern "C" int sdca_omp_in_parallel() __asm__("omp_in_parallel") __attribute__((weak));
extern "C" int sdca_omp_get_max_threads() __asm__("omp_get_max_threads") __attribute__((weak));
extern "C" int sdca_omp_get_thread_num() __asm__("omp_get_thread_num") __attribute__((weak));
extern "C" int sdca_omp_get_num_threads() __asm__("omp_get_num_threads") __attribute__((weak));
#define SDCA_PARALLEL_OPENMP 1
#endif

namespace sdca {

namespace detail {

// The CPUs this process may run on: its affinity mask on Linux.
inline std::size_t available_cpus() {
#ifdef __linux__
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0)
    return static_cast<std::size_t>(CPU_COUNT(&set));
#endif
  const unsigned count = std::thread::hardware_concurrency();
  return count > 0 ? count : 1;
}

// A thread count variable if an integer below 4096 (the first of a list), else 0.
inline std::size_t count_variable(const char* name) {
  const char* text = std::getenv(name);
  char* end = nullptr;
  const std::size_t value = text && *text ? std::strtoul(text, &end, 10) : 0;
  return text && *text && (*end == '\0' || *end == ',') && value < 4096 ? value : 0;
}

// num_threads of a call that leaves it to SDCA_NUM_THREADS (SDCA_THREADS_DEFAULT).
const std::size_t default_threads = static_cast<std::size_t>(-1);

// SDCA_NUM_THREADS if an integer below 4096 (read once), else default_threads.
inline std::size_t environment_threads() {
  static const std::size_t threads = []() -> std::size_t {
    const char* text = std::getenv("SDCA_NUM_THREADS");
    char* end = nullptr;
    const std::size_t value = text ? std::strtoul(text, &end, 10) : 0;
    return text && *text && *end == '\0' && value < 4096 ? value : default_threads;
  }();
  return threads;
}

// The threads a call asks for: num_threads if given, else SDCA_NUM_THREADS,
// else 0 (automatic).
inline std::size_t requested_threads(std::size_t num_threads) {
  if (num_threads != default_threads) return num_threads;
  return environment_threads() != default_threads ? environment_threads() : 0;
}

// Whether the caller runs inside a parallel region of the process's OpenMP runtime.
inline bool in_openmp_parallel() {
#ifdef SDCA_PARALLEL_OPENMP
  return sdca_omp_in_parallel && sdca_omp_in_parallel();
#else
  return false;
#endif
}

// The threads of an automatic call: the available CPUs, at most the OpenMP
// runtime's thread count (as torch.set_num_threads sets it) if one is loaded,
// else at most OMP_NUM_THREADS (as joblib sets it in its workers).
inline std::size_t automatic_threads() {
  std::size_t cap = 0;
#ifdef SDCA_PARALLEL_OPENMP
  if (sdca_omp_get_max_threads)
    cap = static_cast<std::size_t>(std::max(sdca_omp_get_max_threads(), 1));
#endif
  if (cap == 0) {
    static const std::size_t omp = count_variable("OMP_NUM_THREADS");
    cap = omp;
  }
  const std::size_t cpus = available_cpus();
  return cap > 0 ? std::min(cap, cpus) : cpus;
}

// Microseconds of serial work left for a call to add threads, then one
// thread per eighth of it: five times the cost of waking sleeping workers
// (measured at about 200 us for the pool in a VM or on macOS, 10 to 40 us for
// an OpenMP team), so that no call runs slower than on one thread. A test may
// set it (negative: the default).
inline double& thread_work_setting() {
  static double us = -1;
  return us;
}

inline double parallel_work_us(bool openmp_team) {
  const double us = thread_work_setting();
  return us >= 0 ? us : openmp_team ? 200 : 1000;
}

// Parallel calls in flight; automatic calls split the CPUs among them.
inline std::atomic<std::size_t>& parallel_calls() {
  static std::atomic<std::size_t> calls(0);
  return calls;
}

// Whether this thread is a worker: nested calls run serially.
inline bool& in_parallel_rows() {
  static thread_local bool inside = false;
  return inside;
}

// Counts a parallel call while it runs.
struct parallel_call {
  parallel_call() { ++parallel_calls(); }
  ~parallel_call() { --parallel_calls(); }
  parallel_call(const parallel_call&) = delete;
  parallel_call& operator=(const parallel_call&) = delete;
};

// Work of a parallel call: run(data) takes tasks until none remain.
struct parallel_job {
  void (*run)(void*);
  void* data;
  fp_modes modes;  // the caller's
};

// A worker runs a job in the caller's floating-point modes, then restores its own.
inline void run_worker(const parallel_job& job) {
  const fp_modes own = fp_modes::current();
  job.modes.apply();
  job.run(job.data);
  own.apply();
}

inline long process_id() {
#ifdef SDCA_PARALLEL_PID
  return static_cast<long>(getpid());
#else
  return 0;
#endif
}

// Persistent workers, sleeping between calls; one call at a time.
class worker_pool {
 public:
  static worker_pool& instance() {
    static holder pool;
    return *pool.pool;
  }

  // Runs job on the caller and up to extra workers; false while another call
  // holds the pool, and in a forked child (whose workers do not exist).
  bool run(const parallel_job& job, std::size_t extra) {
    if (process_id() != owner_) return false;
    std::unique_lock<std::mutex> busy(busy_, std::try_to_lock);
    if (!busy) return false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (std::size_t i = threads_.size(); i < extra; ++i) {
        try {
          threads_.emplace_back(&worker_pool::loop, this, i);
        } catch (...) {
          break;  // The started workers and the caller take the tasks.
        }
      }
      job_ = &job;
      active_ = pending_ = std::min(extra, threads_.size());
      ++generation_;
    }
    wake_.notify_all();
    job.run(job.data);
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
    job_ = nullptr;
    return true;
  }

  worker_pool(const worker_pool&) = delete;
  worker_pool& operator=(const worker_pool&) = delete;

 private:
  // Stops and frees the pool at exit in the process that made it. A forked child copied
  // its mutexes, condition variables and thread handles, with waiters and threads that do
  // not exist there: destroying any of them can block or terminate, so it keeps them all.
  struct holder {
    worker_pool* pool = new worker_pool;
    holder() = default;
    holder(const holder&) = delete;
    holder& operator=(const holder&) = delete;
    ~holder() {
      if (process_id() != pool->owner_) return;
      pool->stop();
      delete pool;
    }
  };

  worker_pool() : owner_(process_id()) {}
  ~worker_pool() = default;

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& worker : threads_) worker.join();
  }

  void loop(std::size_t index) {
    std::size_t seen = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
      if (index >= active_) continue;
      const parallel_job* job = job_;
      lock.unlock();
      run_worker(*job);
      lock.lock();
      if (--pending_ == 0) done_.notify_one();
    }
  }

  const long owner_;
  std::mutex busy_, mutex_;
  std::condition_variable wake_, done_;
  std::vector<std::thread> threads_;
  const parallel_job* job_ = nullptr;
  std::size_t active_ = 0, pending_ = 0, generation_ = 0;
  bool stop_ = false;
};

#ifdef SDCA_PARALLEL_OPENMP
inline void openmp_entry(void* job) { run_worker(*static_cast<const parallel_job*>(job)); }
#endif

#ifdef SDCA_PARALLEL_OPENMP
// Whether this process was forked after its first parallel call.
inline bool& forked_child() {
  static bool forked = false;
  return forked;
}

inline void mark_forked_child() { forked_child() = true; }

// Watches for forks from now on (once); true if it can.
inline bool watch_forks() {
  static const bool watching = pthread_atfork(nullptr, nullptr, &mark_forked_child) == 0;
  return watching;
}

// Whether the process ran other threads at its first parallel call: a forked
// child runs only the forking thread.
inline bool other_threads() {
  std::FILE* status = std::fopen("/proc/self/status", "r");
  if (!status) return false;
  char line[256];
  long threads = 0;
  while (std::fgets(line, sizeof(line), status))
    if (std::sscanf(line, "Threads: %ld", &threads) == 1) break;
  std::fclose(status);
  return threads > 1;
}
#endif

// Whether run_parallel takes the team of a loaded OpenMP runtime. Not in a
// forked child, whose libgomp would wait for the team threads of the parent:
// neither after a fork that follows the first parallel call nor where that
// call finds the process alone (a child of a parent that ran a team, or a
// process whose team has not started; the pool serves both).
inline bool openmp_team() {
#ifdef SDCA_PARALLEL_OPENMP
  if (!sdca_gomp_parallel || !sdca_omp_in_parallel) return false;
  static const bool usable = watch_forks() && other_threads();
  return usable && !forked_child() && !sdca_omp_in_parallel();
#else
  return false;
#endif
}

// Runs job on the caller and extra other threads: a team of team threads of
// a loaded OpenMP runtime outside its parallel regions (a team of constant
// size: libgomp resizing its team between calls was measured up to 1.8 times
// slower than one thread), else the pool, else new threads.
inline void run_parallel(const parallel_job& job, std::size_t extra, std::size_t team) {
#ifdef SDCA_PARALLEL_OPENMP
  if (openmp_team()) {
    sdca_gomp_parallel(&openmp_entry, const_cast<parallel_job*>(&job),
                       static_cast<unsigned>(std::max(team, extra + 1)), 0);
    return;
  }
#else
  (void)team;
#endif
  if (worker_pool::instance().run(job, extra)) return;
  std::vector<std::thread> threads;
  threads.reserve(extra);
  for (std::size_t i = 0; i < extra; ++i) {
    try {
      threads.emplace_back([&job] { run_worker(job); });
    } catch (...) {
      break;  // The started threads and the caller take the remaining tasks.
    }
  }
  job.run(job.data);
  for (std::thread& worker : threads) worker.join();
}

template <typename F>
void call_job(void* f) {
  (*static_cast<F*>(f))();
}

// The team thread number of the caller in a parallel region of the process's
// OpenMP runtime of at least threads threads, else -1.
inline int openmp_thread(std::size_t threads) {
#ifdef SDCA_PARALLEL_OPENMP
  if (!sdca_omp_get_thread_num || !sdca_omp_get_num_threads || !in_openmp_parallel()) return -1;
  return static_cast<std::size_t>(sdca_omp_get_num_threads()) >= threads ? sdca_omp_get_thread_num()
                                                                         : -1;
#else
  (void)threads;
  return -1;
#endif
}

// Rows [first, n) on threads threads (of a team of team, the others idle), in
// four tasks per thread. On an OpenMP team of threads or more, thread t first
// takes the tasks of rows [t, t + 1) * ceil(n / threads) (the caller, thread
// 0, ran [0, first)): the split of an OpenMP static loop, as PyTorch's, so the
// rows stay in the caches of the threads that take them in the next PyTorch
// call (tasks taken in order were measured to make a PyTorch pass over the
// rows up to 1.9 times slower); then it helps the next threads with theirs.
// Elsewhere every thread takes the tasks in order. Rethrows the exception of
// the first failing task.
template <typename Rows>
void parallel_tasks(std::size_t first, std::size_t n, std::size_t threads, std::size_t team,
                    const Rows& rows) {
  constexpr std::size_t kTasks = 4;  // per thread
  const std::size_t chunk = (n + threads - 1) / threads;
  const bool openmp = openmp_team();  // as run_parallel decides below
  // The next task of each thread's rows, of kTasks; one shared range without a team.
  std::vector<std::atomic<std::size_t>> next(openmp ? threads : 1);
  for (std::atomic<std::size_t>& t : next) t = 0;
  std::atomic<std::size_t> failed(n), joined(0);
  std::exception_ptr error;
  std::mutex mutex;
  // Runs the next of the tasks of rows [begin, end) (counter t); false if none remain.
  auto take = [&](std::size_t begin, std::size_t end, std::size_t tasks,
                  std::atomic<std::size_t>& t) {
    if (begin >= end) return false;
    const std::size_t step = (end - begin + tasks - 1) / tasks;
    const std::size_t start = begin + t++ * step;
    if (start >= end || start >= failed.load()) return false;
    try {
      rows(start, std::min(end, start + step));
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex);
      if (start < failed.load()) {
        failed = start;
        error = std::current_exception();
      }
    }
    return true;
  };
  // Tasks start in order within a range, and every task before a failed one runs.
  auto work = [&] {
    const int slot = openmp ? openmp_thread(threads) : -1;
    if (slot >= 0 ? static_cast<std::size_t>(slot) >= threads : joined++ >= threads) return;
    const bool outer = in_parallel_rows();
    in_parallel_rows() = true;
    if (slot >= 0) {
      for (std::size_t i = 0; i < threads; ++i) {
        const std::size_t s = (static_cast<std::size_t>(slot) + i) % threads;
        const std::size_t begin = std::max(first, s * chunk), end = std::min(n, (s + 1) * chunk);
        while (take(begin, end, kTasks, next[s])) {
        }
      }
    } else {
      while (take(first, n, kTasks * threads, next[0])) {
      }
    }
    in_parallel_rows() = outer;
  };
  const parallel_job job = {&call_job<decltype(work)>, &work, fp_modes::current()};
  {
    const parallel_call call;
    run_parallel(job, threads - 1, team);
  }
  if (error) std::rethrow_exception(error);
}

// Microseconds per entry of a call site's rows (its Rows type), as last
// measured; 0 before.
template <typename Rows>
std::atomic<double>& entry_us() {
  static std::atomic<double> us(0);
  return us;
}

// Calls rows(begin, end) on consecutive ranges covering [0, n) of row_size
// entries each, on up to requested_threads(num_threads) threads; 0 is
// automatic: automatic_threads(), split among the parallel calls in flight.
// The caller runs the first rows in doubling batches until their time
// predicts at least parallel_work_us() for the rest, which then runs on one
// thread per eighth of it; smaller calls stay on the caller, at once when the
// call site's last measured cost per entry predicts less than half of it.
// Calls inside the rows of a call with workers, and automatic calls inside
// an OpenMP parallel region, run serially; rows run with no workers beside
// them (a serial call's, or the ones the caller times first) are like any
// caller, and calls inside them may take threads. rows must be safe to call
// concurrently on disjoint ranges; workers take the caller's floating-point
// modes. Rethrows the exception of the first failing range, the one a serial
// loop would throw.
template <typename Rows>
void parallel_rows(std::size_t n, std::size_t row_size, std::size_t num_threads, const Rows& rows) {
  const std::size_t requested = requested_threads(num_threads);
  const double least = parallel_work_us(openmp_team());
  const double width = static_cast<double>(std::max<std::size_t>(row_size, 1));
  std::atomic<double>& cost = entry_us<Rows>();
  const double known = cost.load(std::memory_order_relaxed);
  if (n < 2 || requested == 1 || in_parallel_rows() || (requested == 0 && in_openmp_parallel()) ||
      (known > 0 && static_cast<double>(n) * width * known < least / 2)) {
    rows(std::size_t(0), n);
    return;
  }
  const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::size_t done = 0;
  for (std::size_t batch = 1;; batch *= 2) {
    const std::size_t end = done + std::min(batch, n - done);
    rows(done, end);
    done = end;
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    // The whole call, or a probe of a microsecond or more, updates the site's cost.
    if (us > 0 && (done == n || us >= 1))
      cost.store(us / (static_cast<double>(done) * width), std::memory_order_relaxed);
    if (done == n) return;
    // The rest at the probe's rate, or at the site's last rate if lower: a
    // first row slowed by cold caches does not add threads to a small call.
    double rest = us / static_cast<double>(done) * static_cast<double>(n - done);
    if (known > 0) rest = std::min(rest, known * width * static_cast<double>(n - done));
    if (rest < least) continue;
    // An OpenMP runtime starts a team of this size: at most 4096, as SDCA_NUM_THREADS.
    const std::size_t team =
        std::min<std::size_t>(requested > 0 ? requested : automatic_threads(), 4096);
    const std::size_t cpus = requested > 0 ? team : team / (parallel_calls().load() + 1);
    const double shares = least > 0 ? std::min(8 * rest / least, 4096.) : 4096.;
    const std::size_t threads =
        std::min(std::min(cpus, n - done), static_cast<std::size_t>(shares));
    if (threads < 2) continue;
    parallel_tasks(done, n, threads, team, rows);
    return;
  }
}

}  // namespace detail

}  // namespace sdca

#endif
