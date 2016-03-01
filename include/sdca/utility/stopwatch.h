#ifndef SDCA_UTILITY_STOPWATCH_H
#define SDCA_UTILITY_STOPWATCH_H

#ifdef SDCA_USE_CHRONO
  #include <chrono>
#else
  #include <sys/time.h>
#endif

#include <ctime>

namespace sdca {

#ifdef SDCA_USE_CHRONO
  typedef typename std::chrono::high_resolution_clock wall_clock;
  typedef typename std::chrono::time_point<wall_clock> wall_time_point;
#else
  typedef double wall_time_point;
#endif

typedef typename std::clock_t cpu_time_point;


struct stopwatch_cpu {
  bool is_running = false;
  double elapsed = 0;
  cpu_time_point mark;

  void start() {
    reset();
    resume();
  }

  void stop() {
    elapsed += is_running ? increment() : 0;
    is_running = false;
  }

  void reset() {
    elapsed = 0;
  }

  void resume() {
    mark = std::clock();
    is_running = true;
  }

  double elapsed_now() const {
    return elapsed + (is_running ? increment() : 0);
  }

  double increment() const {
    return static_cast<double>(std::clock() - mark) / CLOCKS_PER_SEC;
  }
};


struct stopwatch_wall {
  bool is_running = false;
  double elapsed = 0;
  wall_time_point mark;

  void start() {
    reset();
    resume();
  }

  void stop() {
    elapsed += is_running ? increment() : 0;
    is_running = false;
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

  double elapsed_now() const {
    return elapsed + (is_running ? increment() : 0);
  }

  double increment() const {
#ifdef SDCA_USE_CHRONO
    return std::chrono::duration<double>(wall_clock::now() - mark).count();
#else
    return now() - mark;
#endif
  }

#ifndef SDCA_USE_CHRONO
  static double now() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
      return 0;
    }
    return static_cast<double>(time.tv_sec) +
           static_cast<double>(time.tv_usec) * .000001;
  }
#endif
};


struct stopwatch {
  stopwatch_cpu cpu;
  stopwatch_wall wall;

  void start() {
    cpu.start();
    wall.start();
  }

  void stop() {
    cpu.stop();
    wall.stop();
  }

  void reset() {
    cpu.reset();
    wall.reset();
  }

  void resume() {
    cpu.resume();
    wall.resume();
  }
};

}

#endif
