cmake_minimum_required(VERSION 3.10)

project(HALA VERSION 1.0.0 LANGUAGES CXX)
set(hala_license "BSD 3-Clause")

# load all the _hala_* macros
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hala_helpers.cmake)

######## Default Debug ########
if ("${CMAKE_BUILD_TYPE}" STREQUAL "" OR NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

######## Options ########
set(HALA_GPU_BACKEND "NONE" CACHE STRING "Set the GPU backend used by HALA")
set_property(CACHE HALA_GPU_BACKEND PROPERTY STRINGS NONE CUDA ROCM)
set(HALA_EXTENDED_REGISTERS "NONE" CACHE STRING "Set the default extended register used by HALA")
set_property(CACHE HALA_EXTENDED_REGISTERS PROPERTY STRINGS NONE SSE AVX AVX512)
option(HALA_ENABLE_CHOLMOD  "Enable HALA wrappers for Cholmod"      OFF)
option(HALA_ENABLE_EXAMPLES "Install the HALA examples"             OFF)
option(HALA_ENABLE_DOXYGEN  "Generate the HALA documentation"       OFF)

set(HALA_CXX_FLAGS "" CACHE STRING "A ;-separated list of flags for the HALA interface compile options")

###### Interface Target ######
add_library(HALA INTERFACE)
add_library(HALA::HALA INTERFACE IMPORTED GLOBAL)
set_target_properties(HALA::HALA PROPERTIES INTERFACE_LINK_LIBRARIES HALA)

_hala_add_sources(PATH common
                  FILES hala_core.hpp
                        hala_types_helpers.hpp
                        hala_types_caster.hpp
                        hala_vector_defines.hpp
                        hala_wrap_array.hpp
                        hala_cpu_engine.hpp
                        hala_types_checker.hpp
                        hala_input_checker.hpp
                        hala_common.hpp
                        hala.h
                        hala.hpp)
_hala_add_sources(PATH hex/vex
                  FILES hala_intrinsics.hpp
                        hala_intrinsics_macros.hpp
                        hala_mmpack.hpp
                        hala_bindpack.hpp
                        hala_aligned_allocator.hpp
                        hala_vex.hpp)
_hala_add_sources(PATH blas
                  FILES hala_blas_external.hpp
                        hala_blas_1.hpp
                        hala_blas_2.hpp
                        hala_blas_3.hpp
                        hala_blas_overloads.hpp
                        hala_blas.hpp)
_hala_add_sources(PATH sparse
                  FILES hala_sparse_utils.hpp
                        hala_sparse_structs.hpp
                        hala_sparse_core.hpp
                        hala_sparse_blas.hpp
                        hala_sparse_ilu.hpp
                        hala_sparse.hpp)
_hala_add_sources(PATH lapack
                  FILES hala_lapack.hpp
                  FILES hala_lapack_plu.hpp
                        hala_lapack_external.hpp)
_hala_add_sources(PATH gpu
                  FILES hala_cuda_common.hpp
                        hala_rocm_common.hpp
                        hala_gpu_wrap_array.hpp
                        hala_gpu_vector.hpp
                        hala_gpu_engine.hpp
                        hala_gpu_blas0.hpp
                        hala_gpu_blas1.hpp
                        hala_gpu_blas2.hpp
                        hala_gpu_blas3.hpp
                        hala_cuda_sparse_general.hpp
                        hala_cuda_sparse_triangular.hpp
                        hala_rocm_sparse_general.hpp
                        hala_rocm_sparse_triangular.hpp
                        hala_gpu_ilu.hpp
                        hala_gpu_plu.hpp
                        hala_gpu_overloads.hpp
                        hala_cuda.hpp
                        hala_rocm.hpp
                        hala_gpu.hpp)
_hala_add_sources(PATH wax
                  FILES hala_extensions_checker.hpp
                        hala_lib_extensions.hpp
                        hala_mixed_overloads.hpp
                        hala_blas_extensions.hpp
                        hala_sparse_extensions.hpp
                        hala_noengine_structs.hpp
                        hala_noengine_overloads.hpp
                        hala_wrapped_extensions.hpp)
_hala_add_sources(PATH hex/cholmod
                  FILES hala_cholmod_defines.hpp
                        hala_cholmod.hpp)
_hala_add_sources(PATH hex/solvers
                  FILES hala_solvers_core.hpp
                        hala_solvers_cg.hpp
                        hala_solvers_cg_batch.hpp
                        hala_solvers_gmres.hpp
                        hala_solvers.hpp)

target_compile_features(HALA INTERFACE cxx_std_14)


###### Sanity Check ######
if (NOT HALA_GPU_BACKEND STREQUAL "NONE" AND
    NOT HALA_GPU_BACKEND STREQUAL "CUDA" AND
    NOT HALA_GPU_BACKEND STREQUAL "ROCM")
    message(FATAL_ERROR "Invalid GPU backend: ${HALA_GPU_BACKEND}\nvalid backends are: NONE, CUDA, ROCM")
endif()
if (NOT HALA_EXTENDED_REGISTERS STREQUAL "NONE" AND
    NOT HALA_EXTENDED_REGISTERS STREQUAL "SSE"  AND
    NOT HALA_EXTENDED_REGISTERS STREQUAL "AVX"  AND
    NOT HALA_EXTENDED_REGISTERS STREQUAL "AVX512")
    message(FATAL_ERROR "Invalid register extension: ${HALA_EXTENDED_REGISTERS}\nvalid extensions are: NONE, SSE, AVX, AVX512")
endif()


############ BLAS ############
if (NOT BLAS_LIBRARIES)
    find_package(BLAS REQUIRED)
endif()

target_link_libraries(HALA INTERFACE ${BLAS_LIBRARIES})
_hala_check_symbol(SYMBOL zsyr2_ LIBRARIES ${BLAS_LIBRARIES} RESULT HALA_HAS_ZSYR2)
_hala_check_symbol(SYMBOL zspr2_ LIBRARIES ${BLAS_LIBRARIES} RESULT HALA_HAS_ZSPR2)
_hala_check_symbol(SYMBOL zsbmv_ LIBRARIES ${BLAS_LIBRARIES} RESULT HALA_HAS_ZSBMV)

############ LAPACK ##########
if (NOT LAPACK_LIBRARIES)
    find_package(LAPACK REQUIRED)
endif()

target_link_libraries(HALA INTERFACE ${LAPACK_LIBRARIES})

############ Any GPU #########
if (NOT HALA_GPU_BACKEND STREQUAL "NONE")
    set(HALA_ENABLE_GPU ON)
    list(APPEND _hala_components "GPU")
endif()

############ CUDA ############
if (HALA_GPU_BACKEND STREQUAL "CUDA")
    find_package(CUDA REQUIRED)

    list(FILTER CUDA_CUBLAS_LIBRARIES EXCLUDE REGEX "-NOTFOUND$") # CMake 3.10 and CUDA 10 hack

    target_link_libraries(HALA INTERFACE ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
    target_include_directories(HALA INTERFACE ${CUDA_INCLUDE_DIRS})

    set(HALA_ENABLE_CUDA ON)
    list(APPEND _hala_components "CUDA")
endif()

############ ROCM ############
if (HALA_GPU_BACKEND STREQUAL "ROCM")
    set(HALA_ROCM_ROOT "$ENV{ROCM_ROOT}" CACHE PATH "The root folder for the Rocm framework installation")
    if (HALA_ROCM_ROOT)
        list(APPEND CMAKE_PREFIX_PATH "${HALA_ROCM_ROOT}")
    endif()

    find_package(rocblas REQUIRED)
    find_package(rocsparse REQUIRED)

    target_link_libraries(HALA INTERFACE roc::rocblas roc::rocsparse)
    target_include_directories(HALA INTERFACE ${hip_INCLUDE_DIR})

    set(HALA_ENABLE_ROCM ON)
    list(APPEND _hala_components "ROCM")
endif()

############ CHOLMOD #########
if (HALA_ENABLE_CHOLMOD)
    if (NOT HALA_CHOLMOD_LIBRARIES)
        list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)
        find_package(HALACholmod REQUIRED)
    endif()

    target_link_libraries(HALA INTERFACE ${hala_cholmod_libs})

    list(APPEND _hala_components "CHOLMOD")
endif()

######### VECTORIZATION #######
if (NOT HALA_EXTENDED_REGISTERS STREQUAL "NONE")
    target_compile_options(HALA INTERFACE "-msse2")
    target_compile_options(HALA INTERFACE "-msse3")
    list(APPEND _hala_components "SSE")
endif()
if (HALA_EXTENDED_REGISTERS STREQUAL "AVX" OR HALA_EXTENDED_REGISTERS STREQUAL "AVX512")
    target_compile_options(HALA INTERFACE "-mavx")
    list(APPEND _hala_components "AVX")
endif()
if (HALA_EXTENDED_REGISTERS STREQUAL "AVX512")
    target_compile_options(HALA INTERFACE "-mavx512f")
    list(APPEND _hala_components "AVX512")
endif()
if (HALA_CXX_FLAGS)
    foreach(_flag ${HALA_CXX_FLAGS})
        target_compile_options(HALA INTERFACE "${_flag}")
    endforeach()
    unset(_flag)
endif()
if (NOT HALA_EXTENDED_REGISTERS STREQUAL "NONE")
    # test the default alignment of vectors
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_align.cmake)
endif()

#### Config Header ####
# put later so we can resovle alignment for SSE2 and AVX
_hala_configure()

############ DOXYGEN #########
if (HALA_ENABLE_DOXYGEN)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/doxygen)
endif()

############ TESTING #########
if (HALA_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

######## CHECK CHECKS ########
if (HALA_CHECK_CHECKERS)
    # the script that watches the watchers, contains a series of tests
    # that are expected to fail to compile if hala checks work right
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_the_checks.cmake)
endif()

####### ENABLE EXAMPLES ######
if (HALA_ENABLE_EXAMPLES)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/)
endif()

# in case HALA should be installed in a different sub-path
set(HALA_INSTALL_SUB_PATH "include" CACHE PATH "relative path to install the HALA headers")

########################################################################
# Print final message with options and flags
########################################################################
message(STATUS "")
message(STATUS "Handy Accelerated Linear Algebra v${HALA_VERSION}, license ${hala_license}")
message(STATUS "  HALA Configuration Options:")

string(TOUPPER ${CMAKE_BUILD_TYPE} _hala_build_type)
set(_hala_cmake_opts "CMAKE_INSTALL_PREFIX;CMAKE_BUILD_TYPE;CMAKE_CXX_FLAGS_${_hala_build_type};CMAKE_CXX_FLAGS;HALA_CXX_FLAGS")
list(APPEND _hala_cmake_opts "BLAS_LIBRARIES;LAPACK_LIBRARIES")

if (HALA_ENABLE_CUDA)
    list(APPEND _hala_cmake_opts "CUDA_TOOLKIT_ROOT_DIR;CUDA_INCLUDE_DIRS;CUDA_CUBLAS_LIBRARIES;CUDA_cusparse_LIBRARY;CUDA_LIBRARIES")
endif()

if (HALA_ENABLE_CHOLMOD)
    list(APPEND _hala_cmake_opts "hala_cholmod_libs")
endif()

list(APPEND _hala_cmake_opts "HALA_GPU_BACKEND;HALA_EXTENDED_REGISTERS")
foreach(_hala_opt CHOLMOD TESTING EXAMPLES DOXYGEN)
    list(APPEND _hala_cmake_opts "HALA_ENABLE_${_hala_opt}")
endforeach()

foreach(_hala_opt ${_hala_cmake_opts})
    message(STATUS " -D ${_hala_opt}=${${_hala_opt}}")
endforeach()
message(STATUS "")
unset(_hala_opt)
unset(_hala_cmake_opts)
unset(_hala_build_type)

########################################################################
# Install
########################################################################
target_include_directories(HALA INTERFACE $<INSTALL_INTERFACE:${HALA_INSTALL_SUB_PATH}>)
get_target_property(hala_sources HALA INTERFACE_SOURCES)
install(FILES ${hala_sources} DESTINATION ${HALA_INSTALL_SUB_PATH})

if (${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME}) # if not using add_subdirectory()
    install(TARGETS HALA EXPORT hala_export)
    install(EXPORT hala_export DESTINATION "lib/HALA" FILE "HALA.cmake")

    string(REPLACE ";" " " HALA_COMPONENTS "${_hala_components}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt.cmake"
                   "${CMAKE_CURRENT_BINARY_DIR}/configured/CMakeLists.txt")

    # Package config
    include(CMakePackageConfigHelpers)
    configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/HALAConfig.cmake"
                                  "${CMAKE_CURRENT_BINARY_DIR}/HALAConfig.cmake"
                                  INSTALL_DESTINATION "lib/HALA/")
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/HALAConfigVersion.cmake"
                                     COMPATIBILITY AnyNewerVersion)

    # install the files
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/HALAConfig.cmake"
            DESTINATION "lib/HALA/")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/HALAConfigVersion.cmake"
            DESTINATION "lib/HALA/")

    # post-install test
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/post_install_test.sh"
                   "${CMAKE_CURRENT_BINARY_DIR}/post_install_test.sh")
    add_custom_target(test_install COMMAND "${CMAKE_CURRENT_BINARY_DIR}/post_install_test.sh"
                                           "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
endif()
