add_library(${LIB_PROJECTORS} OBJECT
  projector.hpp
  projector.cpp
  knapsack_projector.hpp
  knapsack_projector.cpp
  topk_cone_projector.hpp
  topk_cone_projector.cpp
  topk_cone_biased_projector.hpp
  topk_cone_biased_projector.cpp
  topk_simplex_projector.hpp
  topk_simplex_projector.cpp
  topk_simplex_biased_projector.hpp
  topk_simplex_biased_projector.cpp
  )
add_definitions(-fPIC)
