# Projections lib
add_library(${LIB_PROJECTIONS} OBJECT
  projector.hpp
  projector.cpp
  knapsackprojector.hpp
  knapsackprojector.cpp
  topkconeprojector.hpp
  topkconeprojector.cpp
  topksimplexprojector.hpp
  topksimplexprojector.cpp
  )
add_definitions(-fPIC)
