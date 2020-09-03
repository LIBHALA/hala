cmake_minimum_required(VERSION 3.10)

project(HALA_EXAMPLES VERSION
        @HALA_VERSION_MAJOR@.@HALA_VERSION_MINOR@.@HALA_VERSION_PATCH@
        LANGUAGES CXX)

find_package(HALA @HALA_VERSION_MAJOR@.@HALA_VERSION_MINOR@.@HALA_VERSION_PATCH@
             PATHS "@CMAKE_INSTALL_PREFIX@"
             REQUIRED @HALA_COMPONENTS@)

add_executable(example_gauss_seidel example_gauss_seidel.cpp)
target_link_libraries(example_gauss_seidel HALA::HALA)

# add_executable(example_cuda example_gpu.cpp)
# target_link_libraries(example_cuda HALA::HALA)
