set(MEX_PROX
  "${SDCA_MATLAB_PREFIX}prox${SDCA_BINARY_SUFFIX}"
  )
set(MEX_SOLVE
  "${SDCA_MATLAB_PREFIX}fit${SDCA_BINARY_SUFFIX}"
  )
set(MEX_GD
  "${SDCA_MATLAB_PREFIX}gd${SDCA_BINARY_SUFFIX}"
  )

# Matlab interface
if(${Matlab_FOUND})
  include_directories(matlab)

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-switch-enum")  # to_string(mxClassID)
    set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-weak-vtables") # mat_streambuf
  endif()

  add_definitions(
    -DMEX_PROX="${MEX_PROX}"
    -DMEX_SOLVE="${MEX_SOLVE}"
    -DMEX_GD="${MEX_GD}"
    )

  matlab_add_mex(
    NAME ${MEX_PROX}
    SRC
      matlab/mex_prox.cpp
      matlab/mex_util.h
      ${libsdca_MATH_SOURCES}
      ${libsdca_PROX_SOURCES}
      ${libsdca_UTILITY_SOURCES}
    )

  matlab_add_mex(
    NAME ${MEX_SOLVE}
    SRC
      matlab/mex_solve.cpp
      matlab/mex_util.h
      ${libsdca_MATH_SOURCES}
      ${libsdca_PROX_SOURCES}
      ${libsdca_SOLVER_SOURCES}
      ${libsdca_UTILITY_SOURCES}
    LINK_TO
      ${BLAS_LIBRARIES}
    )

#  matlab_add_mex(
#    NAME ${MEX_GD}
#    SRC
#      matlab/mex_gd.cpp
#      matlab/mex_util.h
#      ${libsdca_MATH_SOURCES}
#      ${libsdca_SOLVER_SOURCES}
#      ${libsdca_UTILITY_SOURCES}
#    LINK_TO
#      ${BLAS_LIBRARIES}
#    )

  install(
    TARGETS
      ${MEX_PROX}
      ${MEX_SOLVE}
#      ${MEX_GD}
    DESTINATION
      ${SDCA_MATLAB_INSTALL_DIR}
    )

else()
  message(STATUS "Matlab not found, mex files will not be compiled.")
endif()
