#ifndef SDCA_UTILITY_LOGGING_H
#define SDCA_UTILITY_LOGGING_H

#include <iomanip>
#include <iostream>

namespace sdca {

namespace logging {

enum level {
  none = 0,
  info,
  verbose,
  debug
};
static const char* level_name[] = {
  "none",
  "info",
  "verbose",
  "debug"
};

enum format {
  short_f = 0,
  short_e,
  long_f,
  long_e
};
static const char* format_name[] = {
  "short_f",
  "short_e",
  "long_f",
  "long_e"
};

extern level __level__;
extern format __format__;
extern std::ios __ios_state__; // can be initialized with nullptr

inline void set_level(level __level) { __level__ = __level; }

inline level get_level() { return __level__; }

inline const char* get_level_name() { return level_name[__level__]; }

inline void
set_format(format __format) {
  __format__ = __format;
  switch (__format) {
    case short_f:
      std::cout << std::setprecision(4) << std::fixed;
      break;
    case short_e:
      std::cout << std::setprecision(4) << std::scientific;
      break;
    case long_f:
      std::cout << std::setprecision(15) << std::fixed;
      break;
    case long_e:
      std::cout << std::setprecision(15) << std::scientific;
      break;
  }
}

inline format
get_format() { return __format__; }

inline const char* get_format_name() { return format_name[__format__]; }


inline void
format_push() {
  __ios_state__.copyfmt(std::cout);
}

inline void
format_pop() {
  std::cout.copyfmt(__ios_state__);
}

}

#define LOG_INFO if (logging::__level__ >= logging::info) std::cout
#define LOG_VERBOSE if (logging::__level__ >= logging::verbose) std::cout
#define LOG_DEBUG if (logging::__level__ >= logging::debug) std::cout

}

#endif
