#include "sdca/utility/logging.h"

/**
 * Defines the global variables declared in logging.h.
 * To be included (once) in the main file.
 */

namespace sdca {
namespace logging {

level __level__ = level::warning;
format __format__ = format::short_e;

}
}
