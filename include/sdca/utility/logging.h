#ifndef SDCA_UTILITY_LOGGING_H
#define SDCA_UTILITY_LOGGING_H

#include <iomanip>
#include <iostream>
#include <string>

namespace sdca {

namespace logging {

enum class level { none = 0, warning, info, verbose, debug };

enum class format { short_e = 0, short_f, long_e, long_f };

extern level current_level;
extern format current_format;

inline level get_level() { return current_level; }

inline void set_level(level level_) { current_level = level_; }

inline format get_format() { return current_format; }

inline void set_format(format format_) {
  current_format = format_;
  switch (format_) {
    case format::short_e:
      std::cout << std::setprecision(4) << std::scientific;
      break;
    case format::short_f:
      std::cout << std::setprecision(4) << std::fixed;
      break;
    case format::long_e:
      std::cout << std::setprecision(16) << std::scientific;
      break;
    case format::long_f:
      std::cout << std::setprecision(16) << std::fixed;
      break;
  }
}

inline std::string to_string(level level_) {
  switch (level_) {
    case level::none:
      return "none";
    case level::warning:
      return "warning";
    case level::info:
      return "info";
    case level::verbose:
      return "verbose";
    case level::debug:
      return "debug";
  }
  return "unknown";
}

inline std::string to_string(format format_) {
  switch (format_) {
    case format::short_e:
      return "short_e";
    case format::short_f:
      return "short_f";
    case format::long_e:
      return "long_e";
    case format::long_f:
      return "long_f";
  }
  return "unknown";
}

}  // namespace logging

#define LOG_WARNING \
  if (logging::current_level >= logging::level::warning) std::cout
#define LOG_INFO \
  if (logging::current_level >= logging::level::info) std::cout
#define LOG_VERBOSE \
  if (logging::current_level >= logging::level::verbose) std::cout
#define LOG_DEBUG \
  if (logging::current_level >= logging::level::debug) std::cout

}  // namespace sdca

#endif
