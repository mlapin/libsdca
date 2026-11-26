include(CMakeFindDependencyMacro)
# Threads for the headers' worker pool; the BLAS is a linker flag of the target.
find_dependency(Threads)
include("${CMAKE_CURRENT_LIST_DIR}/libsdcaTargets.cmake")
