# Mex functions require Matlab headers
include_directories(${MATLAB_INCLUDE_DIR})

# A macro for adding Matlab mex functions
macro(ADD_MEX MEX_NAME)
  add_library(${MEX_NAME} SHARED ${ARGN} mex_function.def)
  target_link_libraries(${MEX_NAME} ${MATLAB_LIBRARIES})
  set_target_properties(${MEX_NAME} PROPERTIES SUFFIX .mexa64 PREFIX "")
  install(TARGETS ${MEX_NAME} DESTINATION bin)
endmacro(ADD_MEX)

# Projection onto the k-simplex
ADD_MEX(projtopksimplex project_onto_k_simplex.cpp
  $<TARGET_OBJECTS:${LIB_PROJECTIONS}>)

ADD_MEX(projtopkcone project_topkcone.cpp
  $<TARGET_OBJECTS:${LIB_PROJECTIONS}>)
