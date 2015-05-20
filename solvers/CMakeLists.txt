add_library(${LIB_SOLVERS} OBJECT
  solvers.hpp
  solver.hpp
  solver.cpp
  dual_solver.hpp
  dual_solver.cpp
  dual_solver_helper.hpp
  dual_solver_helper.cpp
  )
add_definitions(-fPIC)
