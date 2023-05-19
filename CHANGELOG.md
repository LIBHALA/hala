### Version: development

* dropped support for CUDA 9 and maybe 10
    * tested on CUDA 11.4 - 12.1
* changed the name of the main development branch to `develop`
* CMake now requires version 3.19
    * updated the way the CUDA libraries are found
* added a full example for the VEX module
* fixed some conflicts when using regtype::none and float or double arithmetic
* adjusted scaling on some tests to account for order of operations when using avx512

### Version: 1.0

* included full support for the BLAS standard
* included some sparse support for row-compresses matrices
    * using gemv, gemm, trsv, and trsm operations
    * included ilu preconditioner
* included custom iterative solvers
    * conjugate-gradient with single rhs and batched
    * general minimum residual GMRES
* the above is supported for CPU, CUDA and ROCM
* added support for cholmod direct solver (double precision only)
* support for some LAPACK methods (CPU only)
    * work will continue on the LAPACK front
* added support for extended registers
    * sse3 using 128-bit registers
    * avx using 256-bit registers
    * avx512 using 512-bit registers
    * support includes operation for real and complex numbers
    * support includes operation for single and double precision
