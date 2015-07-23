include_directories(.)

set(PLAY_SRCS
  play/main.cpp
  )

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
  solve/l2_hinge_topk.h
  solve/l2_topk_hinge.h
  )

set(UTIL_SRCS
  util/util.h
  util/blas.h
  util/numeric.h
  util/logging.h
  )

# Matlab interface
if(${Matlab_FOUND})
  include_directories(matlab)
  include_directories(${Matlab_INCLUDE_DIRS})

  set(MEX_PROX libsdca_prox)
  set(MEX_SOLVE libsdca_solve)
  add_definitions(-DMEX_PROX="${MEX_PROX}" -DMEX_SOLVE="${MEX_SOLVE}")

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

#  set_target_properties(${MEX_SOLVE} PROPERTIES
#    COMPILE_FLAGS "-Wno-unused-variable -Wno-unused-parameter"
#    )

  install(
    TARGETS ${MEX_PROX} ${MEX_SOLVE}
    DESTINATION matlab
    )
else()
  message(STATUS "Matlab interface is not compiled.")
endif()

# A playground for quick checks and small tests
add_executable(main ${PLAY_SRCS} ${PROX_SRCS} ${SOLVE_SRCS} ${UTIL_SRCS})
target_link_libraries(main ${BLAS_LIBRARIES})
