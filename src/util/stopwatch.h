#ifndef SDCA_UTIL_STOPWATCH_H
#define SDCA_UTIL_STOPWATCH_H

#include <chrono>
#include <ctime>

namespace sdca {

using wall_clock = std::chrono::high_resolution_clock;
typedef typename std::chrono::time_point<wall_clock> wall_time_point;
typedef typename std::clock_t cpu_time_point;

struct stopwatch_wall {
  double elapsed = 0;
  wall_time_point mark;

  void start() {
    reset();
    resume();
  }

  void stop() {
    elapsed += std::chrono::duration<double>(wall_clock::now() - mark).count();
  }

  void reset() {
    elapsed = 0;
  }

  void resume() {
    mark = wall_clock::now();
  }

  double elapsed_now() {
    return elapsed
      + std::chrono::duration<double>(wall_clock::now() - mark).count();
  }
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
