add_library(${LIB_UTIL} OBJECT
  common.hpp
  math_util.hpp
  math_util.cpp
  )
add_definitions(-fPIC)
set_target_properties(${LIB_UTIL}
  PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    C_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN "hidden"
)
