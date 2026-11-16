#ifndef SDCA_UTILITY_LOGGING_H
#define SDCA_UTILITY_LOGGING_H

#include <atomic>
#include <iomanip>
#include <iostream>
#include <string>

namespace sdca {

// The solver's progress and warning messages. Logging is off by default
// (level::none); set_level enables it and set_stream redirects it (std::cout
// by default). The settings are process-wide, held in function-local statics,
// so the headers need no separate definition file.
namespace logging {

enum class level { none = 0, warning, info, verbose, debug };

enum class format { short_e = 0, short_f, long_e, long_f };

namespace detail {
inline std::atomic<int>& level_value() {
  static std::atomic<int> value(static_cast<int>(level::none));
  return value;
}
inline std::atomic<int>& format_value() {
  static std::atomic<int> value(static_cast<int>(format::short_e));
  return value;
}
inline std::atomic<std::ostream*>& stream_value() {
  static std::atomic<std::ostream*> value(&std::cout);
  return value;
}
}  // namespace detail

inline level get_level() { return static_cast<level>(detail::level_value().load()); }

inline void set_level(const level level_) { detail::level_value() = static_cast<int>(level_); }

// The stream of the messages; it must outlive its use.
inline std::ostream& stream() { return *detail::stream_value().load(); }

inline void set_stream(std::ostream& stream_) { detail::stream_value() = &stream_; }

inline format get_format() { return static_cast<format>(detail::format_value().load()); }

// The number format of the messages (applied to the stream).
inline void set_format(const format format_) {
  detail::format_value() = static_cast<int>(format_);
  std::ostream& out = stream();
  switch (format_) {
    case format::short_e:
      out << std::setprecision(4) << std::scientific;
      break;
    case format::short_f:
      out << std::setprecision(4) << std::fixed;
      break;
    case format::long_e:
      out << std::setprecision(16) << std::scientific;
      break;
    case format::long_f:
      out << std::setprecision(16) << std::fixed;
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

}  // namespace sdca

// The log stream if the level is enabled: SDCA_LOG_INFO << "message" << std::endl;
#define SDCA_LOG_AT(LEVEL)                                     \
  if (::sdca::logging::get_level() < ::sdca::logging::LEVEL) { \
  } else                                                       \
    ::sdca::logging::stream()
#define SDCA_LOG_WARNING SDCA_LOG_AT(level::warning)
#define SDCA_LOG_INFO SDCA_LOG_AT(level::info)
#define SDCA_LOG_VERBOSE SDCA_LOG_AT(level::verbose)
#define SDCA_LOG_DEBUG SDCA_LOG_AT(level::debug)

#endif
