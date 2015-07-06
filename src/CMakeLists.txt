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

set(PLAY_SRCS
  play/main.cpp
  )

# Matlab interface
if(${Matlab_FOUND})
  include_directories(matlab)
  include_directories(${Matlab_INCLUDE_DIRS})

  #LINK_TO ${BLAS_LIBRARIES}
  matlab_add_mex(
    NAME sdca_project
    SRC matlab/mex_project.cpp matlab/mex_util.h ${PROX_SRCS}
    )

  set_target_properties(sdca_project
    PROPERTIES COMPILE_FLAGS "-Wno-unused-parameter"
    )

  install(
    TARGETS sdca_project
    DESTINATION matlab
    )
else()
  message(STATUS "Matlab interface will not be compiled.")
endif()

# A playground for quick checks and small tests
add_executable(main ${PLAY_SRCS} ${PROX_SRCS})
target_link_libraries(main ${BLAS_LIBRARIES})
