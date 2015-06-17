add_library(${LIB_SOLVERS} OBJECT
  solvers.hpp
  solver.hpp
  solver.cpp
  dual_solver.hpp
  dual_solver.cpp
  primal_solver.hpp
  primal_solver.cpp
  dual_variables_helper.hpp
  dual_variables_helper.cpp
  )
add_definitions(-fPIC)
set_target_properties(${LIB_SOLVERS}
  PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    C_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN "hidden"
)
