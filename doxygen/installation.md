# Installation

[TOC]

| Feature | Tested versions |
|----|----|
| gcc     | 5 - 10          |
| clang   | 4 - 10          |
| cmake   | 3.10 - 3.16     |
| OpenBlas| 0.2.20 - 0.3.8  |
| CUDA    | 10.0 - 11.0     |
| ROCM    | 3.8             |

HALA supports three modes of installation and usage:
* CMake install that exports the dependencies and configures the headers
* CMake add sub-directory command to include HALA in an existing project
* using individual modules and including the headers
CMake is available from [https://cmake.org/](https://cmake.org/) and also included in the native package repositories of most Linux distributions (e.g., Ubuntu 18.04) as well as Mac OSX Homebrew. The advantage of CMake is that it can automatically detect the needed libraries, handle the header configurations and compiler flags. Using the individual modules does not require CMake, but the user has to manually link to the necessary libraries.

### Requirements

HALA requires a C++-14 capable compiler and an implementation of BLAS and LAPACK (tested with OpenBLAS). HALA also supports GPU backends with Nvidia CUDA and AMD ROCM frameworks. See the Table at the top of the page and note that even if a feature has not been tested, it may still work (e.g., icc compiler with the MKL library).


### CMake HALA Install

Using the CMake build engine:
```
  mkdir hala_build
  cd hala_build
  cmake -D CMAKE_INSTALL_PREFIX=<install-path> <hala-options> <path-to-hala>
  make
  make test
  make install
  make test_install
```
* note that `make test` works only with option `-D HALA_ENABLE_TESTING=ON`

Supported HALA options:
```
  -D HALA_INSTALL_SUB_PATH    (installs the headers in CMAKE_INSTALL_PREFIX/HALA_INSTALL_SUB_PATH)
  -D BLAS_LIBRARIES           (list with libraries, will skip the automated search)
  -D LAPACK_LIBRARIES         (list with libraries, will skip the automated search)
  -D HALA_GPU_BACKEND         (NONE/CUDA/ROCM - select the GPU backend)
  -D CUDA_TOOLKIT_ROOT_DIR    (when using CUDA, point to the CUDA installation)
  -D ROCM_ROOT                (when using ROCM, point to the ROCM installation)
  -D HALA_EXTENDED_REGISTERS  (NONE/SSE/AVX/AVX512 - select extended registers for the VEX module)
  -D HALA_ENABLE_CHOLMOD      (ON/OFF - search for Cholmod and enable the Cholmod solvers)
  -D HALA_CHOLMOD_ROOT        (when using CHOLMOD, point to the Cholmod installation)
  -D HALA_ENABLE_DOXYGEN      (ON/OFF - search for Doxygen and build the HALA documentation)
  -D DOXYGEN_EXECUTABLE       (when using Doxygen, select the Doxygen executable)
  -D HALA_ENABLE_TESTING      (ON/OFF - compile the HALA tests, if OFF there is nothing to compile)
```
* the BLAS and LAPACK libraries can be manually specified with `-D BLAS_LIBRARIES` and `-D LAPACK_LIBRARIES` circumventing the CMake search
* the automatic search for Cholmod may fail to pick out all the dependencies (e.g., because libmetis.so.x has the wrong .x version), a CMake list of libraries can be provided by the project with `-D HALA_CHOLMOD_LIBRARIES` which will overwrite the automatic search
* the `HALA_EXTENDED_REGISTERS` options will add flags to the HALA exported interface so that these flags can be propagated to a dependent target
* additional flags can be specified with `-D HALA_CXX_FLAGS`, for example `-mtune=native` and `-mfma`
* if a vectorization option is skipped on install time, it can still be enabled by adding the flags later, e.g., building without avx but adding `-mavx` later
    * adding more flags later may alter the defaults
    * the extra flags will not have been tested during install
    * the default alignment will fallback to `unaligned` since it has not been tested

Example CMake command for full install with the CUDA backend:
```
  cmake -D CMAKE_INSTALL_PREFIX=/home/<user>/.local/ \
        -D HALA_INSTALL_SUB_PATH=include \
        -D CMAKE_BUILD_TYPE=Release \
        -D BLAS_LIBRARIES=/usr/x86_64-linux-gnu/libopenblas.so \
        -D LAPACK_LIBRARIES=/usr/x86_64-linux-gnu/libopenblas.so \
        -D HALA_GPU_BACKEND=CUDA \
        -D CUDA_TOOLKIT_ROOT_DIR=/usr/loca/cuda-9.2 \
        -D HALA_ENABLE_CHOLMOD=ON \
        -D HALA_CHOLMOD_ROOT=/home/<user>/.local/ \
        -D HALA_EXTENDED_REGISTERS=AXV \
        -D HALA_CXX_FLAGS="-mfma;-mtune=native" \
        -D HALA_ENABLE_TESTING=ON \
        -D HALA_ENABLE_DOXYGEN=ON \
        <path-to-hala>
```

After the install, CMake projects can use the `find_package()` command to locate HALA:
```
find_package(HALA 1.0 PATHS <hala-install-path>)
target_link_libraries(FooMain HALA::HALA)
```
The PATHS can be ignored if the install is in a system default place, e.g., `/home/<user>/.local`, or if the environment variable `HALA_ROOT` is set.

Packages that do not use CMake have to manually add the include folder and the link libraries. The include folder is:
```
  CMAKE_INSTALL_PREFIX/HALA_INSTALL_SUB_PATH
```
The list of libraries required by HALA can be obtained with the command:
```
  cat <hala-install-path>/lib/HALA/HALA.cmake | grep INTERFACE_LINK_LIBRARIES
```


### Directly Include HALA in a CMake Project

HALA can be incorporated in another CMake project directly with the `add_subdirectory()` command and without the need to go through an install:
```
project(Foo VERSION x.y.z LANGUAGES CXX)
add_executable(FooMain foo_main.cpp)

set(HALA_GPU_BACKEND        ROCM CACHE INTERNAL)
set(HALA_EXTENDED_REGISTERS AVX  CACHE INTERNAL)

add_subdirectory(<path-to-hala> hala)
target_link_libraries(FooMain HALA)
```
* HALA will automatically install the headers in the `HALA_INSTALL_SUB_PATH` sub-folder of the install prefix for Foo (default is `include`).
* HALA will **not** install or export the `HALA` target, Foo must manually install/export the target to keep access to the dependencies.
