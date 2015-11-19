include_directories(.)

set(PROX_SRCS
  prox/prox.h
  prox/proxdef.h
  prox/knapsack_eq.h
  prox/knapsack_le.h
  prox/knapsack_le_biased.h
  prox/topk_cone.h
  prox/topk_cone_biased.h
  prox/topk_simplex.h
  prox/topk_simplex_biased.h
  )

set(SOLVE_SRCS
  solve/solve.h
  solve/solvedef.h
  solve/solver.h
  solve/dual_solver.h
  solve/primal_solver.h
  solve/objective_base.h
  solve/l2_hinge_topk.h
  solve/l2_topk_hinge.h
  )

set(UTIL_SRCS
  util/util.h
  util/blas.h
  util/fmath.h
  util/logging.h
  util/numeric.h
  util/stopwatch.h
  )

if(${USE_THREADING_SUFFIX})
  if(${USE_SEQUENTIAL})
    set(MEX_PROX libsdca_prox_st)
    set(MEX_SOLVE libsdca_solve_st)
  else()
    set(MEX_PROX libsdca_prox_mt)
    set(MEX_SOLVE libsdca_solve_mt)
  endif()
else()
  set(MEX_PROX libsdca_prox)
  set(MEX_SOLVE libsdca_solve)
endif()

# Matlab interface
if(${Matlab_FOUND})
  include_directories(matlab)

  add_definitions(
    -DMEX_PROX="${MEX_PROX}"
    -DMEX_SOLVE="${MEX_SOLVE}"
    )

  if(USE_MEX_COMPILER)
    matlab_add_mex_use_compiler(
      NAME ${MEX_PROX}
      SRC matlab/mex_prox.cpp
      DEPENDS matlab/mex_util.h ${PROX_SRCS} ${UTIL_SRCS}
      )

    matlab_add_mex_use_compiler(
      NAME ${MEX_SOLVE}
      SRC matlab/mex_solve.cpp
      DEPENDS matlab/mex_util.h ${SOLVE_SRCS} ${PROX_SRCS} ${UTIL_SRCS}
      LINK_TO ${BLAS_LIBRARIES}
      )

    get_property(_mex_prox TARGET ${MEX_PROX} PROPERTY FULL_NAME)
    get_property(_mex_solve TARGET ${MEX_SOLVE} PROPERTY FULL_NAME)
    install(
      FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${_mex_prox}
        ${CMAKE_CURRENT_BINARY_DIR}/${_mex_solve}
      DESTINATION ${MATLAB_API_INSTALL_DIR}
      )
  else()
    matlab_add_mex(
      NAME ${MEX_PROX}
      SRC matlab/mex_prox.cpp matlab/mex_util.h ${PROX_SRCS} ${UTIL_SRCS}
      )

    matlab_add_mex(
      NAME ${MEX_SOLVE}
      SRC matlab/mex_solve.cpp matlab/mex_util.h
          ${SOLVE_SRCS} ${PROX_SRCS} ${UTIL_SRCS}
      LINK_TO ${BLAS_LIBRARIES}
      )

    install(
      TARGETS ${MEX_PROX} ${MEX_SOLVE}
      DESTINATION ${MATLAB_API_INSTALL_DIR}
      )
  endif()
else()
  message(STATUS "Matlab interface is not compiled.")
endif()
