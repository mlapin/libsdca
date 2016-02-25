#ifndef SDCA_UTILITY_TYPE_NAME_H
#define SDCA_UTILITY_TYPE_NAME_H

namespace sdca {

template <typename Type>
struct type_name {
  static constexpr const char*
  name() { return "unknown type"; }
};

template <>
struct type_name<float> {
  static constexpr const char*
  name() { return "float"; }
};

template <>
struct type_name<double> {
  static constexpr const char*
  name() { return "double"; }
};

template <>
struct type_name<long double> {
  static constexpr const char*
  name() { return "long double"; }
};

}

#endif
