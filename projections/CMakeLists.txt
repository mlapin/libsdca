# Projections lib
add_library(${LIB_PROJECTIONS} OBJECT
  projector.hpp
  projector.cpp
  knapsackprojector.hpp
  knapsackprojector.cpp
  topkconeprojector.hpp
  topkconeprojector.cpp
  )
add_definitions(-fPIC)
