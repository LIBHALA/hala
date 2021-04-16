include(CheckLibraryExists)

macro(_hala_configure)
    # configures the options in the header hala_config.hpp
    # must be called after the SSE and AVX default alignment has been resolved
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/hala_config.hpp"
                   "${CMAKE_CURRENT_BINARY_DIR}/configured/hala_config.hpp")
    target_sources(HALA INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/configured/hala_config.hpp>
                                  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/hala_config.hpp>)
    target_include_directories(HALA INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/configured/>)
endmacro(_hala_configure)

macro(_hala_add_sources)
    # adds a bunch of source files from a given folder
    cmake_parse_arguments(_hala_sources "" "PATH" "FILES" ${ARGN})
    target_include_directories(HALA INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${_hala_sources_PATH}>)
    foreach(_hala_file ${_hala_sources_FILES})
        #message(" adding: ${CMAKE_CURRENT_SOURCE_DIR}/${_hala_sources_PATH}/${_hala_file}") # good for debugging
        target_sources(HALA INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${_hala_sources_PATH}/${_hala_file}>
                                      $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/${_hala_file}>)
    endforeach()
    unset(_hala_file)
    unset(_hala_sources_FILES)
    unset(_hala_sources_PATH)
endmacro(_hala_add_sources)

macro(_hala_check_symbol)
    # usage: _hala_check_symbol(SYMBOL zspr2_ LIBRARIES ${BLAS_LIBRARIES} RESULT HALA_HAS_ZSPR2)
    #   loops over the libraries and checks if they contain the given symbol
    #   the result is a boolean ON/OFF variable
    cmake_parse_arguments(_hala_symbol "" "SYMBOL;RESULT" "LIBRARIES" ${ARGN})
    if (DEFINED CMAKE_REQUIRED_QUIET)
        set(_hala_save_CMAKE_REQUIRED_QUIET "${CMAKE_REQUIRED_QUIET}")
    endif()
    set(CMAKE_REQUIRED_QUIET ON)
    set(${_hala_symbol_RESULT} OFF)
    set(_hala_check_symbol_status "not found")
    foreach(_hala_lib ${_hala_symbol_LIBRARIES})
        if (NOT ${_hala_symbol_RESULT})
            check_library_exists(${_hala_lib} ${_hala_symbol_SYMBOL} "" _hala_symbol_found)
            if (_hala_symbol_found)
                set(${_hala_symbol_RESULT} ON)
                set(_hala_check_symbol_status "found")
            endif()
            unset(_hala_symbol_found CACHE)
        endif()
    endforeach()
    # message(STATUS "HALA check symbol ${_hala_symbol_SYMBOL} - ${_hala_check_symbol_status}")
    if (_hala_save_CMAKE_REQUIRED_QUIET)
        set(CMAKE_REQUIRED_QUIET "${_hala_save_CMAKE_REQUIRED_QUIET}")
    endif()
    unset(_hala_lib)
    unset(_hala_check_symbol_status)
    unset(_hala_save_CMAKE_REQUIRED_QUIET)
    unset(_hala_symbol_SYMBOL)
    unset(_hala_symbol_LIBRARIES)
    unset(_hala_symbol_SYMBOL)
endmacro()
