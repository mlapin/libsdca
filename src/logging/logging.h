#ifndef SDCA_LOGGING_LOGGING_H
#define SDCA_LOGGING_LOGGING_H

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

enum format {
  short_f,
  short_e,
  long_f,
  long_e
};

level __level__;
std::ios __ios_state__(nullptr);

inline
void
set_level(level __level) {
  __level__ = __level;
}

inline
void
set_format(format __format) {
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

inline
void
format_push() {
  __ios_state__.copyfmt(std::cout);
}

inline
void
format_pop() {
  std::cout.copyfmt(__ios_state__);
}

}

#define LOG_INFO if (logging::__level__ >= logging::info) std::cout
#define LOG_VERBOSE if (logging::__level__ >= logging::verbose) std::cout
#define LOG_DEBUG if (logging::__level__ >= logging::debug) std::cout

}

#endif
