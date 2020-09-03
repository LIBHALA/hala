#ifndef __HALA_CUDA_WRAPPERS_HPP
#define __HALA_CUDA_WRAPPERS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#ifdef HALA_ENABLE_ROCM
    #error "Cannot include both cuda and rocm!"
#endif

#define HALA_ENABLE_CUDA
#include "hala_gpu.hpp" // this is actually supposed to work

/*!
 * \file hala_cuda.hpp
 * \brief HALA wrapper for CUDA functions
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACUDA
 *
 * Includes all HALA-CUDA wrapper templates.
 */

/*!
 * \defgroup HALACUDA HALA CUDA Wrapper Templates
 *
 * \par CUDA Wrapper Templates
 * A set of templates that wrap around CUDA functions allowing for (near) seamless
 * generalization of linear algebra code written for the CPU version of BLAS
 * to be instantiated and used with cuBlas and cuSparse libraries.
 *
 * \par CUDA Vector
 * HALA provides a CUDA vector class that mimics the container structure of \b std::vector
 * but wraps around a raw-pointer holding data on a CUDA device.
 * The use of the hala::cuda_vector is optional, since all HALA templates work with general
 * vector-like classes; however, it is handy and fully compatible with the C++ standards
 * (unlike CUDA thrust) and doesn't require third party support.
 * The hala::cuda_vector is also the default internal vector for the hala::cuda_engine.
 *
 * \par CUDA Engine
 * Unlike the hala::cpu_engine class that is simply a collection of handy methods,
 * the hala::cuda_engine has internal state, namely the cuBlas and cuSparse handles
 * and the ID of the associated CUDA device.
 * The CPU version of BLAS accepts all scalars by value; however, CUDA libraries
 * can work with pointers on either the host or the device.
 * The hala::cuda_engine defaults to using points on the host which allows for
 * the HALA-CUDA template-wrappers to have identical signature to HALA-BLAS;
 * however, the template can also accept scalars passed in as pointers
 * and the hala::cuda_engine can be used to specify the host/device mode.
 * See also hala::cuda_pntr.
 *
 * \par cuBlas Wrappers
 * The cuBlas library provides an almost complete implementation of the BLAS
 * standard but using a CUDA device. HALA provides a complete set of BLAS
 * overloads that have identical signature and the CPU and GPU (with the
 * additional engine variable). The areas where cuBlas deviates from the
 * standard will result in performance penalty, but the results will always
 * be correct. Here are the cases where that matters:
 * - The symmetric banded method sbmv does not have a CUDA implementation
 *   for the non-conjugate mode, the HALA work-around uses the CPU implementation
 *   with data being moved between the CPU and the GPU and hence suffering
 *   a performance penalty (but giving the correct results).
 * - The packed methods spmv, spr and spr2 do not have native implementation
 *   for the symmetric non-conjugate case, while the methods will work
 *   they will involve packing and unpacking which will affect performance.
 * - The cuBlas implementation of triangular matrix-matrix multiply trmm
 *   does not overwrite the input matrix B but outputs the result in another
 *   matrix C; HALA will copy the output over to maintain a consistent API
 *   but there is the additional cost of the copy at the background.
 * - The HALA implementation of the Givens rotation generation uses BLAS,
 *   after all it is setting a 2x2 matrix with a few scalars, the cost is
 *   hardly worth the attention of the GPU.
 *
 * \par cuSparse Wrappers
 * The cuSparse library overs a large set of methods for handling sparse
 * matrices, but the API is very cumbersome and hard to use. Many methods,
 * especially in the newer API, use opaque C-style structures to wrap around
 * the data and those structures have to be manually created and destroyed.
 * Almost every call to cuSparse requires the additional workspace vector
 * to be manually allocated (and later deleted). Triangular solve methods
 * require additional analysis steps that look for ways to optimize the
 * solve phase, each analysis is stored in yet another opaque C-style structure.
 *
 * \par
 * HALA provides wrappers that automatically handle the many cuSparse API calls
 * and data-structures, and provide an API that is no harder to use than
 * the standard BLAS methods. In addition, HALA provides wrapper triangular
 * matrix classes that automatically handle the matrix analysis without
 * sacrificing generality. Similar wrappers are provided around the
 * cuSparse incomplete lower-upper preconditioner methods.
 *
 */

namespace hala{}

#endif
