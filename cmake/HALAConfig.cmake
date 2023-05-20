cmake_minimum_required(VERSION 3.19)

@PACKAGE_INIT@

include("@CMAKE_INSTALL_PREFIX@/lib/HALA/HALA.cmake")

add_library(HALA::HALA INTERFACE IMPORTED GLOBAL)
set_target_properties(HALA::HALA PROPERTIES INTERFACE_LINK_LIBRARIES HALA)

foreach(_hala_components @_hala_components@)
    set(HALA_${_hala_components}_FOUND "ON")
endforeach()

if (HALA_CUDA_FOUND)
    if (NOT CUDAToolkit_ROOT AND NOT "@CUDAToolkit_ROOT@" STREQUAL "")
        set(CUDAToolkit_ROOT "@CUDAToolkit_ROOT@")
    endif()
    find_package(CUDAToolkit REQUIRED)
endif()

if (HALA_ROCM_FOUND)
    if (NOT "@HALA_ROCM_ROOT@" STREQUAL "")
        list(APPEND CMAKE_PREFIX_PATH "@HALA_ROCM_ROOT@")
    endif()
    find_package(rocblas REQUIRED)
    find_package(rocsparse REQUIRED)
endif()

check_required_components(HALA)

message(STATUS "Found HALA version @HALA_VERSION_MAJOR@.@HALA_VERSION_MINOR@.@HALA_VERSION_PATCH@ - @HALA_COMPONENTS@")
