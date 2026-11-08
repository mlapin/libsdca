include(CMakeFindDependencyMacro)
# Threads for the headers' worker pool; OpenMP only where threaded MKL exports
# it. A built-in installation looks for nothing else when consumed.
find_dependency(Threads)
include("${CMAKE_CURRENT_LIST_DIR}/libsdcaTargets.cmake")
get_target_property(_sdca_links libsdca::libsdca INTERFACE_LINK_LIBRARIES)
if("OpenMP::OpenMP_CXX" IN_LIST _sdca_links)
  find_dependency(OpenMP COMPONENTS CXX)
endif()
