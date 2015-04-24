# Projections lib
add_library(${LIB_PROJECTIONS} OBJECT
  projector.cpp
  projector.hpp
  topkprojector.cpp
  topkprojector.hpp
  topkconeprojector.cpp
  topkconeprojector.hpp)
add_definitions(-fPIC)
