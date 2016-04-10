#ifndef SDCA_UTILITY_TYPES_H
#define SDCA_UTILITY_TYPES_H

#include <cstddef>

namespace sdca {

typedef typename std::size_t size_type;

typedef typename std::ptrdiff_t diff_type;


template <typename Type>
inline constexpr const char*
type_name() { return "unknown"; }

template <>
inline constexpr const char*
type_name<float>() { return "float"; }

template <>
inline constexpr const char*
type_name<double>() { return "double"; }

template <>
inline constexpr const char*
type_name<long double>() { return "long double"; }

}

#endif
