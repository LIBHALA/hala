macro(_hala_add_test)
    cmake_parse_arguments(_test "" "NAME;OPTION" "FILES" ${ARGN} )
    if (${_test_OPTION})
        add_executable(hala_${_test_NAME}_tests ${_test_FILES})
        target_link_libraries(hala_${_test_NAME}_tests HALA::HALA)
        set_target_properties(hala_${_test_NAME}_tests PROPERTIES OUTPUT_NAME ${_test_NAME}_hala_tests
                                                                  CXX_EXTENSIONS OFF)
        add_test(hala_${_test_NAME} ${_test_NAME}_hala_tests)
    endif()
    unset(_test_OPTION)
    unset(_test_NAME)
    unset(_test_FILES)
endmacro(_hala_add_test)

_hala_add_test(OPTION HALA_ENABLE_TESTING NAME core
               FILES core_tests.hpp
                     core_tests.cpp)


_hala_add_test(OPTION HALA_ENABLE_TESTING NAME vex
               FILES vex_tests.hpp
                     vex_tests.cpp)


_hala_add_test(OPTION HALA_ENABLE_TESTING NAME sparse
               FILES sparse_tests.hpp
                     sparse_with_engine.cpp
                     sparse_with_noengine.cpp
                     sparse_tests.cpp)


_hala_add_test(OPTION HALA_ENABLE_TESTING NAME blas
               FILES blas1_tests.hpp
                     blas2_tests.hpp
                     blas3_tests.hpp
                     blas_with_engine.cpp
                     blas_with_noengine.cpp
                     blas_tests.hpp
                     blas_tests.cpp)


_hala_add_test(OPTION HALA_ENABLE_TESTING NAME lapack
               FILES lapack1_tests.hpp
                     lapack_plu_tests.hpp
                     lapack_tests.hpp
                     lapack_with_engine.cpp
                     lapack_with_noengine.cpp
                     lapack_tests.cpp)


_hala_add_test(OPTION HALA_ENABLE_CUDA NAME cuda
               FILES cuda_core_tests.hpp
                     cuda_blas0_tests.hpp
                     cuda_blas1_tests.hpp
                     cuda_blas2_tests.hpp
                     cuda_blas3_tests.hpp
                     cuda_sparse_tests.hpp
                     cuda_tests.cpp)


_hala_add_test(OPTION HALA_ENABLE_CHOLMOD NAME cholmod
               FILES cholmod_tests.cpp)


_hala_add_test(OPTION HALA_ENABLE_TESTING NAME solvers
               FILES solvers_tests.hpp
                     solvers_tests.cpp)
