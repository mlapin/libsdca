# This module requires MATLAB_ROOT
# It then defines:
#  MATLAB_INCLUDE_DIR: include path for mex.h
#  MATLAB_LIBRARIES:   required libraries: libmex, libmx
#  MATLAB_MEX_LIBRARY: path to libmex
#  MATLAB_MX_LIBRARY:  path to libmx

SET(MATLAB_ROOT "${MATLAB_ROOT}" CACHE PATH "Matlab directory.")

SET(MATLAB_FOUND 0)
IF( "${MATLAB_ROOT}" STREQUAL "" )
    MESSAGE(WARNING "MATLAB_ROOT variable not set." )
ELSE("${MATLAB_ROOT}" STREQUAL "" )

        FIND_PATH(MATLAB_INCLUDE_DIR mex.h
                  ${MATLAB_ROOT}/extern/include)

        INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

        FIND_LIBRARY( MATLAB_MEX_LIBRARY
                      NAMES libmex mex
                      PATHS ${MATLAB_ROOT}/bin ${MATLAB_ROOT}/extern/lib
                      PATH_SUFFIXES glnxa64 glnx86 win64/microsoft win32/microsoft)

        FIND_LIBRARY( MATLAB_MX_LIBRARY
                      NAMES libmx mx
                      PATHS ${MATLAB_ROOT}/bin ${MATLAB_ROOT}/extern/lib
                      PATH_SUFFIXES glnxa64 glnx86 win64/microsoft win32/microsoft)

    MESSAGE (STATUS "MATLAB_ROOT=${MATLAB_ROOT}")

ENDIF("${MATLAB_ROOT}" STREQUAL "" )

# This is common to UNIX and Win32:
SET(MATLAB_LIBRARIES
  ${MATLAB_MEX_LIBRARY}
  ${MATLAB_MX_LIBRARY}
)

IF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
  SET(MATLAB_FOUND 1)
ENDIF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)

MARK_AS_ADVANCED(
  MATLAB_LIBRARIES
  MATLAB_MEX_LIBRARY
  MATLAB_MX_LIBRARY
  MATLAB_INCLUDE_DIR
  MATLAB_FOUND
)
