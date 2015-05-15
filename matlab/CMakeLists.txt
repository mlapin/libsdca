# Mex functions require Matlab headers
include_directories(${Matlab_INCLUDE_DIRS})

# A macro for adding Matlab mex functions
macro(ADD_MEX MEX_NAME)
  add_library(${MEX_NAME} SHARED ${ARGN} mex_function.def)
  target_link_libraries(${MEX_NAME} ${Matlab_LIBRARIES})
  set_target_properties(${MEX_NAME} PROPERTIES SUFFIX .mexa64 PREFIX "")
  install(TARGETS ${MEX_NAME} DESTINATION bin)
endmacro(ADD_MEX)

# Projection onto the k-simplex
ADD_MEX(projknapsack project_knapsack.cpp
  $<TARGET_OBJECTS:${LIB_PROJECTORS}>)
ADD_MEX(projtopkcone project_topk_cone.cpp
  $<TARGET_OBJECTS:${LIB_PROJECTORS}>)
ADD_MEX(projtopkconebiased project_topk_cone_biased.cpp
  $<TARGET_OBJECTS:${LIB_PROJECTORS}>)
ADD_MEX(projtopksimplex project_topk_simplex.cpp
  $<TARGET_OBJECTS:${LIB_PROJECTORS}>)
ADD_MEX(projtopksimplexbiased project_topk_simplex_biased.cpp
  $<TARGET_OBJECTS:${LIB_PROJECTORS}>)
