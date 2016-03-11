#ifndef SDCA_UTILITY_LOGGING_H
#define SDCA_UTILITY_LOGGING_H

#include <iomanip>
#include <iostream>
#include <string>

namespace sdca {

namespace logging {

enum class level {
  none = 0,
  info,
  verbose,
  debug
};

enum class format {
  short_f = 0,
  short_e,
  long_f,
  long_e
};


extern level __level__;
extern format __format__;


inline level
get_level() { return __level__; }

inline void
set_level(level __level) { __level__ = __level; }


inline format
get_format() { return __format__; }

inline void
set_format(format __format) {
  __format__ = __format;
  switch (__format) {
    case format::short_f:
      std::cout << std::setprecision(4) << std::fixed;
      break;
    case format::short_e:
      std::cout << std::setprecision(4) << std::scientific;
      break;
    case format::long_f:
      std::cout << std::setprecision(16) << std::fixed;
      break;
    case format::long_e:
      std::cout << std::setprecision(16) << std::scientific;
      break;
  }
}


inline std::string
to_string(level __level) {
  switch (__level) {
    case level::none:
      return "none";
    case level::info:
      return "info";
    case level::verbose:
      return "verbose";
    case level::debug:
      return "debug";
  }
}


inline std::string
to_string(format __format) {
  switch (__format) {
    case format::short_f:
      return "short_f";
    case format::short_e:
      return "short_e";
    case format::long_f:
      return "long_f";
    case format::long_e:
      return "long_e";
  }
}

}

#define LOG_INFO if (logging::__level__ >= logging::level::info) std::cout
#define LOG_VERBOSE if (logging::__level__ >= logging::level::verbose) std::cout
#define LOG_DEBUG if (logging::__level__ >= logging::level::debug) std::cout

}

#endif
