########################################################################
# compile tests that must not compile if HALA actually preserves
# const-correctness, array sizes and correct overloads, etc.
########################################################################

get_target_property(_hala_public_flags HALA INTERFACE_COMPILE_OPTIONS)

if (_hala_public_flags)
    list(APPEND _hala_all_flags ${_hala_public_flags})
endif()
list(APPEND _hala_all_flags "-I${CMAKE_CURRENT_SOURCE_DIR}/common/")
list(APPEND _hala_all_flags "-I${CMAKE_CURRENT_BINARY_DIR}/configured/")
list(APPEND _hala_all_flags "-I${CMAKE_CURRENT_SOURCE_DIR}/blas/")
list(APPEND _hala_all_flags "-I${CMAKE_CURRENT_SOURCE_DIR}/solvers/")
list(APPEND _hala_all_flags "-I${CMAKE_CURRENT_SOURCE_DIR}/sparse/")
list(APPEND _hala_all_flags "-I${CMAKE_CURRENT_SOURCE_DIR}/cuda/")
list(APPEND _hala_all_flags "-I${CMAKE_CURRENT_SOURCE_DIR}/vex/")
list(APPEND _hala_all_flags "-I${CMAKE_CURRENT_SOURCE_DIR}/lapack/")
list(APPEND _hala_all_flags "-I${CMAKE_CURRENT_SOURCE_DIR}/cholmod/")
list(APPEND _hala_all_flags "-I${CMAKE_CURRENT_SOURCE_DIR}/wax/")

get_target_property(_hala_public_libs HALA INTERFACE_LINK_LIBRARIES)

macro(_hala_test_checks)

cmake_parse_arguments(_hala_check "" "WORK_DIR;FILE;TEST;FLAG;STATUS" "" ${ARGN})

try_compile(_hala_check_build ${CMAKE_CURRENT_BINARY_DIR}/${_hala_check_WORK_DIR}/
                              ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${_hala_check_FILE}
                              COMPILE_DEFINITIONS "${_hala_all_flags};${_hala_check_FLAG}"
                              LINK_LIBRARIES "${_hala_public_libs}"
                              CXX_STANDARD 14
                              CXX_EXTENSIONS OFF
                              COPY_FILE "${CMAKE_CURRENT_BINARY_DIR}/${_hala_check_WORK_DIR}/${_hala_check_TEST}_program"
                              OUTPUT_VARIABLE _hala_check_verbose)

message("verbose output: ${_hala_check_verbose}")

if (_hala_check_STATUS)
    if (NOT _hala_check_build)
        message(FATAL_ERROR "\nGood test gone bad! Check ${_hala_check_TEST}\n")
    endif()
else()
    if (_hala_check_build)
        message(FATAL_ERROR "\nSomething slipped through ${_hala_check_TEST}\n")
    endif()
endif()

unset(_hala_check_WORK_DIR)
unset(_hala_check_FILE)
unset(_hala_check_TEST)
unset(_hala_check_FLAG)
unset(_hala_check_STATUS)
unset(_hala_check_verbose)
unset(_hala_check_verbose)

endmacro(_hala_test_checks)

# Tests are written in pairs, one test must fail one must pass.
# The tests are very similar and the passing test helps ensure that the test
# that failed did in fact failed for the right reason,
# i.e., if the failed test breaks due to an unrelated bug in HALA, then the
# passing test will fails too and thus reveal the problem.

_hala_test_checks(WORK_DIR check_the_checkers FILE "check_the_checks.cpp" TEST ArrayBad STATUS OFF FLAG "-DTEST_ARRAY_SIZE_BAD")
_hala_test_checks(WORK_DIR check_the_checkers FILE "check_the_checks.cpp" TEST ArrayOK  STATUS ON  FLAG "-DTEST_ARRAY_SIZE_OK")

_hala_test_checks(WORK_DIR check_the_checkers FILE "check_the_checks.cpp" TEST TypeBad  STATUS OFF FLAG "-DTEST_TYPES_BAD")
_hala_test_checks(WORK_DIR check_the_checkers FILE "check_the_checks.cpp" TEST TypeOK   STATUS ON  FLAG "-DTEST_TYPES_OK")

_hala_test_checks(WORK_DIR check_the_checkers FILE "check_the_checks.cpp" TEST ConstBad STATUS OFF FLAG "-DTEST_CONST_BAD")
_hala_test_checks(WORK_DIR check_the_checkers FILE "check_the_checks.cpp" TEST ConstOK  STATUS ON  FLAG "-DTEST_CONST_OD")

unset(_hala_public_libs)
unset(_hala_all_flags)
