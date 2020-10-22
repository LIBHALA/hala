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
