# Mex functions require Matlab headers
include_directories(${Matlab_INCLUDE_DIRS})

# A macro for adding Matlab mex functions
macro(ADD_MEX MEX_NAME)
  add_library(${MEX_NAME} SHARED ${ARGN} mex_function.def)
  target_link_libraries(${MEX_NAME} ${Matlab_LIBRARIES} ${BLAS_LIBRARIES})
  set_target_properties(${MEX_NAME} PROPERTIES SUFFIX .mexa64 PREFIX "")
  install(TARGETS ${MEX_NAME} DESTINATION bin)
endmacro(ADD_MEX)

# Projection onto the k-simplex
ADD_MEX(project_knapsack project_knapsack.cpp mex_util.hpp
  $<TARGET_OBJECTS:${LIB_PROJECTORS}> $<TARGET_OBJECTS:${LIB_UTIL}>)
ADD_MEX(project_topk_cone project_topk_cone.cpp mex_util.hpp
  $<TARGET_OBJECTS:${LIB_PROJECTORS}> $<TARGET_OBJECTS:${LIB_UTIL}>)
ADD_MEX(project_topk_cone_biased project_topk_cone_biased.cpp mex_util.hpp
  $<TARGET_OBJECTS:${LIB_PROJECTORS}> $<TARGET_OBJECTS:${LIB_UTIL}>)
ADD_MEX(project_topk_simplex project_topk_simplex.cpp mex_util.hpp
  $<TARGET_OBJECTS:${LIB_PROJECTORS}> $<TARGET_OBJECTS:${LIB_UTIL}>)
ADD_MEX(project_topk_simplex_biased project_topk_simplex_biased.cpp mex_util.hpp
  $<TARGET_OBJECTS:${LIB_PROJECTORS}> $<TARGET_OBJECTS:${LIB_UTIL}>)

ADD_MEX(solve_dual_topk_l2 solve_dual_topk_l2.cpp mex_util.hpp
  $<TARGET_OBJECTS:${LIB_SOLVERS}>
  $<TARGET_OBJECTS:${LIB_PROJECTORS}>
  $<TARGET_OBJECTS:${LIB_UTIL}>)

ADD_MEX(solve_primal_topk_l2 solve_primal_topk_l2.cpp mex_util.hpp
  $<TARGET_OBJECTS:${LIB_SOLVERS}>
  $<TARGET_OBJECTS:${LIB_PROJECTORS}>
  $<TARGET_OBJECTS:${LIB_UTIL}>)
