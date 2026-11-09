#include "sdca/utility/logging.h"

/**
 * Defines the global variables declared in logging.h.
 * To be included (once) in the main file.
 */

namespace sdca {
namespace logging {

level current_level = level::warning;
format current_format = format::short_e;

}  // namespace logging
}  // namespace sdca
