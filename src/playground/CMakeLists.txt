add_executable(main main.cpp $<TARGET_OBJECTS:${LIB_UTIL}>)
target_link_libraries(main ${BLAS_LIBRARIES})
