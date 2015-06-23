# Mex functions require Matlab headers
include_directories(${Matlab_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-conversion")

set(SRC_TARGETS)
set(LINK_TARGETS)

# Projectors
list(APPEND SRC_TARGETS mex_util.hpp)
list(APPEND SRC_TARGETS $<TARGET_OBJECTS:${LIB_UTIL}>)
list(APPEND SRC_TARGETS $<TARGET_OBJECTS:${LIB_PROJECTORS}>)
list(APPEND LINK_TARGETS ${BLAS_LIBRARIES})

matlab_add_mex(
  NAME project_knapsack
  SRC project_knapsack.cpp ${SRC_TARGETS}
  LINK_TO ${LINK_TARGETS}
  )
matlab_add_mex(
  NAME project_knapsack_le
  SRC project_knapsack_le.cpp ${SRC_TARGETS}
  LINK_TO ${LINK_TARGETS}
  )
matlab_add_mex(
  NAME project_knapsack_le_biased
  SRC project_knapsack_le_biased.cpp ${SRC_TARGETS}
  LINK_TO ${LINK_TARGETS}
  )
matlab_add_mex(
  NAME project_topk_cone
  SRC project_topk_cone.cpp ${SRC_TARGETS}
  LINK_TO ${LINK_TARGETS}
  )
matlab_add_mex(
  NAME project_topk_cone_biased
  SRC project_topk_cone_biased.cpp ${SRC_TARGETS}
  LINK_TO ${LINK_TARGETS}
  )
matlab_add_mex(
  NAME project_topk_simplex
  SRC project_topk_simplex.cpp ${SRC_TARGETS}
  LINK_TO ${LINK_TARGETS}
  )
matlab_add_mex(
  NAME project_topk_simplex_biased
  SRC project_topk_simplex_biased.cpp ${SRC_TARGETS}
  LINK_TO ${LINK_TARGETS}
  )


# Solvers
list(APPEND SRC_TARGETS "$<TARGET_OBJECTS:${LIB_SOLVERS}>")

matlab_add_mex(
  NAME solve_primal_topk_l2
  SRC solve_primal_topk_l2.cpp ${SRC_TARGETS}
  LINK_TO ${LINK_TARGETS}
  )
matlab_add_mex(
  NAME solve_dual_topk_l2
  SRC solve_dual_topk_l2.cpp ${SRC_TARGETS}
  LINK_TO ${LINK_TARGETS}
  )
matlab_add_mex(
  NAME solve_primal_hinge_topk_l2
  SRC solve_primal_hinge_topk_l2.cpp ${SRC_TARGETS}
  LINK_TO ${LINK_TARGETS}
  )
matlab_add_mex(
  NAME solve_dual_hinge_topk_l2
  SRC solve_dual_hinge_topk_l2.cpp ${SRC_TARGETS}
  LINK_TO ${LINK_TARGETS}
  )


# Installation
install(
  TARGETS
    project_knapsack
    project_knapsack_le
    project_knapsack_le_biased
    project_topk_cone
    project_topk_cone_biased
    project_topk_simplex
    project_topk_simplex_biased
    solve_primal_topk_l2
    solve_dual_topk_l2
    solve_primal_hinge_topk_l2
    solve_dual_hinge_topk_l2
  DESTINATION matlab
  )
