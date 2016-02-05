#ifndef SDCA_UTIL_STOPWATCH_H
#define SDCA_UTIL_STOPWATCH_H

#ifdef SDCA_USE_CHRONO
  #include <chrono>
#else
  #include <sys/time.h>
#endif

#include <ctime>

namespace sdca {

#ifdef SDCA_USE_CHRONO
  using wall_clock = std::chrono::high_resolution_clock;
  typedef typename std::chrono::time_point<wall_clock> wall_time_point;
#else
  typedef double wall_time_point;
#endif

typedef typename std::clock_t cpu_time_point;

struct stopwatch_wall {
  double elapsed = 0;
  wall_time_point mark;

  void start() {
    reset();
    resume();
  }

  void stop() {
#ifdef SDCA_USE_CHRONO
    elapsed += std::chrono::duration<double>(wall_clock::now() - mark).count();
#else
    elapsed += now() - mark;
#endif
  }

  void reset() {
    elapsed = 0;
  }

  void resume() {
#ifdef SDCA_USE_CHRONO
    mark = wall_clock::now();
#else
    mark = now();
#endif
  }

  double elapsed_now() {
#ifdef SDCA_USE_CHRONO
    return elapsed
      + std::chrono::duration<double>(wall_clock::now() - mark).count();
#else
    return elapsed + now() - mark;
#endif
  }

#ifndef SDCA_USE_CHRONO
  double now() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
      return 0;
    }
    return static_cast<double>(time.tv_sec) +
      static_cast<double>(time.tv_usec) * .000001;
  }
#endif
};

struct stopwatch_cpu {
  double elapsed = 0;
  cpu_time_point mark;

  void start() {
    reset();
    resume();
  }

  void stop() {
    elapsed += static_cast<double>(std::clock() - mark) / CLOCKS_PER_SEC;
  }

  void reset() {
    elapsed = 0;
  }

  void resume() {
    mark = std::clock();
  }

  double elapsed_now() {
    return elapsed
      + static_cast<double>(std::clock() - mark) / CLOCKS_PER_SEC;
  }
};

}

#endif
