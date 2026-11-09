set(MEX_PROX
  "${SDCA_MATLAB_PREFIX}prox${SDCA_BINARY_SUFFIX}"
  )
set(MEX_SOLVE
  "${SDCA_MATLAB_PREFIX}fit${SDCA_BINARY_SUFFIX}"
  )
set(MEX_LOSS "${SDCA_MATLAB_PREFIX}loss${SDCA_BINARY_SUFFIX}")

# Matlab interface
if(SDCA_BUILD_MATLAB AND Matlab_FOUND)
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
      Threads::Threads
    )

  matlab_add_mex(NAME ${MEX_LOSS} SRC matlab/mex_loss.cpp matlab/mex_util.h
    LINK_TO ${BLAS_LIBRARIES} Threads::Threads)

  # The floating-point flags of the libsdca target (defined after src/).
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(mex ${MEX_PROX} ${MEX_SOLVE} ${MEX_LOSS})
      target_compile_options(${mex} PRIVATE -fno-fast-math -ffp-contract=off)
    endforeach()
  endif()

  install(
    TARGETS
      ${MEX_PROX}
      ${MEX_SOLVE}
      ${MEX_LOSS}
    DESTINATION
      ${SDCA_MATLAB_INSTALL_DIR}
    )

else()
  message(STATUS "MATLAB interfaces disabled.")
endif()
