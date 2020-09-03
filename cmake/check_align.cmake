macro(_hala_test_alignment)

cmake_parse_arguments(_hala_align "" "REG;WORK_DIR;FILE" "" ${ARGN})

get_target_property(_hala_public_flags HALA INTERFACE_COMPILE_OPTIONS)

try_compile(_hala_align_build ${CMAKE_CURRENT_BINARY_DIR}/${_hala_align_WORK_DIR}/
                              ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${_hala_align_FILE}
                              COMPILE_DEFINITIONS "${_hala_public_flags}"
                              CXX_STANDARD 14
                              COPY_FILE "${CMAKE_CURRENT_BINARY_DIR}/${_hala_align_WORK_DIR}/test_program"
                              OUTPUT_VARIABLE _hala_align_verbose)

#message("verbose output: ${_hala_align_verbose}")

if (NOT _hala_align_build)
    message(FATAL_ERROR "Failed to build a test program using the ${_hala_align_REG} instructions.")
endif()
message(STATUS "success in building a program with the ${_hala_align_REG} instruction set")

execute_process(COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${_hala_align_WORK_DIR}/test_program"
                        RESULT_VARIABLE _hala_align_run)

if ("${_hala_align_run}" STREQUAL "0")
    message(STATUS "success in running a program with the ${_hala_align_REG}, default alignment seems good")
    set(_hala_align_result ON)
else()
    message(STATUS "crash in running a program with the ${_hala_align_REG}, defaulting to unaligned mode")
    set(_hala_align_result OFF)
endif()

unset(_hala_align_run)
unset(_hala_align_build)
unset(_hala_public_flags)
unset(_hala_align_verbose)

unset(_hala_align_REG)
unset(_hala_align_FILE)
unset(_hala_align_WORK_DIR)

endmacro(_hala_test_alignment)


if (NOT HALA_EXTENDED_REGISTERS STREQUAL "NONE")
    _hala_test_alignment(REG sse WORK_DIR check_align16  FILE check_align16.cpp)

    set(HALA_ALIGN128 ${_hala_align_result} CACHE BOOL "sse vectorizaton default alignment")
    unset(_hala_align_result)
endif()

if (HALA_EXTENDED_REGISTERS STREQUAL "AVX" OR HALA_EXTENDED_REGISTERS STREQUAL "AVX512")
    _hala_test_alignment(REG avx WORK_DIR check_align32  FILE check_align32.cpp)

    set(HALA_ALIGN256 ${_hala_align_result} CACHE BOOL "avx vectorizaton default alignment")
    unset(_hala_align_result)
endif()

if (HALA_EXTENDED_REGISTERS STREQUAL "AVX512")
    _hala_test_alignment(REG avx512 WORK_DIR check_align64  FILE check_align64.cpp)

    set(HALA_ALIGN512 ${_hala_align_result} CACHE BOOL "avx512 vectorizaton default alignment")
    unset(_hala_align_result)
endif()
