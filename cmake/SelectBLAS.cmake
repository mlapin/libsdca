# The BLAS and the compile flags of src/native_build.txt, as in the Python and Rust builds.
#
# SDCA_BLAS_BACKEND is AUTO (the default), EXTERNAL or BUILTIN. AUTO and EXTERNAL link the
# first candidate of the file (or SDCA_BLAS, linker flags such as "-lopenblas") with which
# src/blas_probe.cpp links; AUTO without one takes the built-in products and warns.
#
# Sets SDCA_COMPILE_FLAGS, SDCA_BLAS_DEFINITIONS, BLAS_LIBRARIES and SDCA_SELECTED_BLAS
# (EXTERNAL, BUILTIN or FALLBACK).

set(SDCA_BLAS "$ENV{SDCA_BLAS}" CACHE STRING "Linker flags of an external CBLAS (AUTO, EXTERNAL)")
set(_sdca_settings "${CMAKE_CURRENT_SOURCE_DIR}/src/native_build.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_sdca_settings}")

# The entries of key in src/native_build.txt.
function(sdca_settings key out)
  file(STRINGS "${_sdca_settings}" _lines REGEX "^${key} ")
  set(_entries)
  foreach(_line IN LISTS _lines)
    string(REGEX REPLACE "^${key} +" "" _entry "${_line}")
    list(APPEND _entries "${_entry}")
  endforeach()
  set(${out} "${_entries}" PARENT_SCOPE)
endfunction()

# Linker flags as link items: "-framework <name>" stays one item.
function(sdca_link_items flags out)
  separate_arguments(_words UNIX_COMMAND "${flags}")
  set(_items)
  set(_framework FALSE)
  foreach(_word IN LISTS _words)
    if(_framework)
      list(APPEND _items "-framework ${_word}")
      set(_framework FALSE)
    elseif(_word STREQUAL "-framework")
      set(_framework TRUE)
    else()
      list(APPEND _items "${_word}")
    endif()
  endforeach()
  set(${out} "${_items}" PARENT_SCOPE)
endfunction()

sdca_settings(flags SDCA_COMPILE_FLAGS)

string(TOUPPER "${SDCA_BLAS_BACKEND}" SDCA_BLAS_BACKEND)
if(NOT SDCA_BLAS_BACKEND MATCHES "^(AUTO|EXTERNAL|BUILTIN)$")
  message(FATAL_ERROR "SDCA_BLAS_BACKEND must be AUTO, EXTERNAL, or BUILTIN")
endif()
set(BLAS_LIBRARIES)
if(SDCA_BLAS_BACKEND STREQUAL "BUILTIN")
  set(SDCA_SELECTED_BLAS BUILTIN)
  set(SDCA_BLAS_DEFINITIONS BLAS_BUILTIN)
else()
  if(SDCA_BLAS)
    set(_sdca_candidates "${SDCA_BLAS}")
  else()
    sdca_settings(blas _sdca_candidates)
  endif()
  set(SDCA_SELECTED_BLAS)
  set(_sdca_index 0)
  foreach(_candidate IN LISTS _sdca_candidates)
    sdca_link_items("${_candidate}" _items)
    math(EXPR _sdca_index "${_sdca_index} + 1")
    unset(_sdca_links CACHE)
    try_compile(_sdca_links "${CMAKE_CURRENT_BINARY_DIR}/blas_probe/${_sdca_index}"
      SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/blas_probe.cpp"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CURRENT_SOURCE_DIR}/include"
      LINK_LIBRARIES ${_items})
    if(_sdca_links)
      set(SDCA_SELECTED_BLAS EXTERNAL)
      set(SDCA_BLAS_DEFINITIONS BLAS_DEFAULT_LOCAL_HEADER)
      set(BLAS_LIBRARIES ${_items})
      break()
    endif()
  endforeach()
  if(NOT SDCA_SELECTED_BLAS)
    string(REPLACE ";" ", " _sdca_tried "${_sdca_candidates}")
    if(SDCA_BLAS_BACKEND STREQUAL "EXTERNAL" OR SDCA_BLAS)
      message(FATAL_ERROR "No CBLAS links with ${_sdca_tried}; set SDCA_BLAS to the linker "
        "flags of one, or choose SDCA_BLAS_BACKEND=BUILTIN")
    endif()
    message(WARNING "libsdca found no BLAS (tried ${_sdca_tried}) and uses its built-in "
      "products: install OpenBLAS (or another CBLAS) or set SDCA_BLAS to its linker flags")
    set(SDCA_SELECTED_BLAS FALLBACK)
    set(SDCA_BLAS_DEFINITIONS BLAS_BUILTIN SDCA_BLAS_FALLBACK)
  endif()
endif()
message(STATUS "libsdca BLAS backend: ${SDCA_SELECTED_BLAS} (${BLAS_LIBRARIES})")
