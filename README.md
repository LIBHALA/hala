# HALA

Handy Accelerated Linear Algebra **HALA** is a collection of C++ templates that wrap around  the accelerated linear algebra standards BLAS, LAPACK, CUDA, and ROCM with focus on the ease of use and the ability to express general algorithms independent from the backend CPU or GPU device. Additional templates are provided to interface with SuiteSparse-Cholmod, intrinsic CPU vectorization methods, and also offers basic implementation of some Krylov solvers.

The main objective of **HALA** is to provide a clean integration between the libraries written in other languages and C++ generic template programming, thus achieving generality, portability, and performance. A more detailed description regarding the issues addressed by **HALA** is provided in the [motivation page](doxygen/motivation.md), here is a brief list of the supported features.

* Name-space support, everything is properly encapsulated into **namespace hala{}**.

* The API can be used with either raw-arrays/pointers or containers which allows for dimensions to be verified and/or inferred automatically, e.g.,
```
  std::vector<double> x = {1.0, 2.0, 3.0};
  std::valarray<double> y = {2.0, 4.0, 6.0};
  hala::axpy(-2.0, x, y); // after the call, y will be {0.0, 0.0, 0.0}
```

* Supported containers include **std::vectors** and **std::valarray**; raw pointers and arrays are supported as-is and with container-like wrappers; custom containers can be used with specialization of some **HALA** templates, see the **HALA Array Wrapper** and **HALA Support for Custom Vector Class** modules.

* Type deduction is used to automatically call the appropriate function for matrices and vectors with scalars of type **float**, **double**, **std::complex<float>**, and **std::complex<double>**.

* The templates work with custom complex classes, so long as they are ABI compatible with C++ and Fortran complex numbers and the appropriate templates have been specialized, see **Custom Type Definitions** for the custom vector module.

* Default values are assigned to inputs such as matrix leading dimension (e.g., **lda**) and vector stride (e.g., **incx**).

* Handles, matrix descriptions, and matrix factors used by CUDA and ROCM libraries are wrapped in containers that ensure proper destruction and provide RAII style of resource management.

* Extended vectorization registers that allow multiple floating point operations in a single instruction can be used for complex as well as real numbers, and memory alignment issues can be cleanly resolved with compile-time tests to establish reasonable defaults and zero-overhead manual overrides.

* Sizes and dimensions are checked with **cassert**, checks are disabled in Release mode (with *DNDEBUG* directive); run-time error codes are always checked and converted into **std::runtime_error** exceptions.


### Quick Install

HALA can be used either as a collection of standalone individual modules (in which case the user is responsible to link to the appropriate external libraries), or though CMake which handles dependencies automatically. Detailed instructions are included in the [installation instructions](doxygen/installation.md), quick list of the CMake options and default options is given here:
```
  -D HALA_GPU_BACKEND=<NONE,CUDA,ROCM>
  -D HALA_EXTENDED_REGISTERS=<NONE,SSE,AVX,AVX512>
  -D HALA_ENABLE_CHOLMOD=OFF
  -D HALA_ENABLE_TESTING=OFF
  -D HALA_ENABLE_DOXYGEN=OFF
```
The **SSE** and **AVX** options test for 128/256/512-bit alignment of **std::vector** containers and enable the appropriate compiler flags.
