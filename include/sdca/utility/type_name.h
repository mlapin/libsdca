#ifndef SDCA_UTILITY_TYPE_NAME_H
#define SDCA_UTILITY_TYPE_NAME_H

namespace sdca {

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
