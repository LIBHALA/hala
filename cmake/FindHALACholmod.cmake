set(HALA_CHOLMOD_ROOT "" CACHE PATH "The root folder for the Cholmod libraries")

macro(_hala_cholmod_find_libs)
    cmake_parse_arguments(_hala_cholmod "" "PREFIX;LIST" "NAMES" ${ARGN} )
    foreach(_hala_lib ${_hala_cholmod_NAMES})
        #message("Looking for: ${_hala_lib}")
        find_library(_hala_${_hala_lib} ${_hala_lib}
                     HINTS "${_hala_cholmod_PREFIX}"
                     HINTS "${_hala_cholmod_PREFIX}/lib/"
                     HINTS "${_hala_cholmod_PREFIX}/${CMAKE_LIBRARY_ARCHITECTURE}/lib/"
                     HINTS "${_hala_cholmod_PREFIX}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
                     HINTS "${_hala_cholmod_PREFIX}/lib64/"
                     HINTS "${_hala_cholmod_PREFIX}/${CMAKE_LIBRARY_ARCHITECTURE}/lib64/"
                     HINTS "${_hala_cholmod_PREFIX}/lib64/${CMAKE_LIBRARY_ARCHITECTURE}")

        #message("Found : ${_hala_${_hala_lib}}")
        list(APPEND hala_${_hala_cholmod_LIST} ${_hala_${_hala_lib}})
    endforeach()
    unset(_hala_lib)
    unset(_hala_cholmod_NAMES)
    unset(_hala_cholmod_LIST)
    unset(_hala_cholmod_PREFIX)
endmacro()

# find cholmod amd colamd camd ccolamd metis suitesparseconfig blas lapack pthread

_hala_cholmod_find_libs(LIST cholmod_libs PREFIX ${HALA_CHOLMOD_ROOT_DIR}
                        NAMES cholmod amd colamd camd ccolamd libmetis.so.5
                              suitesparseconfig pthread)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HALACholmod REQUIRED_VARS _hala_cholmod _hala_suitesparseconfig)
