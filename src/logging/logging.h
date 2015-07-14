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

level __level__;

inline
void
set_level(level __level) {
  __level__ = __level;
}

inline
void
set_format_scientific() {
  std::cout << std::scientific << std::setprecision(16);
}

inline
void
set_format_default() {
  std::cout.copyfmt(std::ios(nullptr));
}

}

#define LOG_INFO if (logging::__level__ >= logging::info) std::cout
#define LOG_VERBOSE if (logging::__level__ >= logging::verbose) std::cout
#define LOG_DEBUG if (logging::__level__ >= logging::debug) std::cout

#define LOG_SCIENTIFIC if (logging::__level__ >= logging::info) std::cout

}

#endif
