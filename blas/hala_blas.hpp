#ifndef __HALA_BLAS_WRAPPERS_HPP
#define __HALA_BLAS_WRAPPERS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

/*!
 * \file hala_blas.hpp
 * \brief HALA wrapper for all BLAS functions.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALABLAS
 *
 * Includes all HALA BLAS wrappers.
 */

/*!
 * \internal
 * \ingroup HALABLAS
 * \brief Indicates the BLAS has been enabled, if using modules out-of-order.
 *
 * \endinternal
 */
#define HALA_ENABLE_BLAS

#include "hala_blas_overloads.hpp"

/*!
 * \defgroup HALABLAS HALA BLAS Wrapper Templates
 *
 * \par HALA BLAS Wrapper Templates
 * A set of templates that wrap around BLAS functions allowing for (near) seamless integration with
 * C++ containers such as \b std::vector and \b std::valarray.
 * The wrappers assume meaningful defaults for increments and leading dimensions
 * and in some cases the sizes are inferred automatically from the sizes of the vectors.
 * The methods can be called with either BLAS standard order of parameters,
 * or with overloads that push all defaults to the end of the calls.
 *
 * \par Naming Convention
 * The template names match the BLAS naming without the precision characters,
 * e.g., dgemm(), sgemm(), cgemm() and zgemm() are all called with gemm().
 * The Hermitian and symmetric variants are lumped into single wrappers that
 * accepts an extra template parameter indicating whether to use the Hermitian or just symmetric call.
 * The default is always set to Hermitian mode, which is more natural for complex
 * numbers and doesn't affect the real case. See also hala::non_conj.
 *
 * \par Vectors and Scalars
 * The type for the vector entries must be compatible with either float, double, std::complex<float>
 * and std::complex<double>, see \ref HALACTYPES.
 * - if multiple vectors are passed to a single call, all types must be compatible
 * - if the call accepts scalars, those will be automatically cast-ed to the vector
 *   scalar type and no warning would be issue regarding precision
 *   (the vectors determine the precision, the scalars have to be converted anyway and no conversion
 *    would result in loss of precision)
 * - out-of-the-box HALA works and is tested with std::vector and std::valarray,
 *   vectors that closely match the \b STL conventions are also compatible,
 *   see \ref HALACUSTOM
 *
 * \par Leading Dimensions, Increments and Sizes
 * Most BLAS functions can be called with custom leading matrix dimensions and vector increments,
 * e.g., the \b lda and \b incx in the calls to hala::gemv. HALA assumes default values for all
 * such parameters; \b incx is set to 1 which implies that the vectors are contiguous and the \b lda
 * dimensions default to -1 which tells HALA to assume the minimum value permissible in the context,
 * i.e., there is no padding and the matrix is not embedded in a larger one.
 * - if the number of entries is inferred from the vector size with \b incx > 1,
 *   there is no check if the size divides evenly
 * - if multiple vectors are passed in the call, the size is determined by the first vector,
 *   e.g., in the call to hala::axpy() the size is inferred from \b x and if the vector \b y
 *   contains extra entries, those would be ignored
 * - sizes cannot be inferred from raw-arrays, any array is assumed to have the sufficient size
 *   for whatever operation, but since the arrays don't \b know the size, this assumption cannot
 *   be verified and incorrect size will result in non-informative segfault error
 *
 * \par Resizing
 * Some BLAS calls (depending on inputs) have pure output variables,
 * e.g., calling hala::gemv() with \b beta equal to 0 will ignore all values stored in \b y;
 * therefore, there is no need to have a hard requirement on the size of \b y.
 * If HALA detects a case of a variable acting as a \a pure \a output and if the current
 * size is insufficient to store the result, then HALA will make an attempt to resize the vector.
 * The attempt may raise a std::runtime_error() if the vector cannot be resized, e.g.,
 * hala::wrap_array(), or if the resize method is not \a registered with the proper
 * specialization defined in \ref HALACUSTOM.
 *
 * \par Errors Handling:
 * Negative increments, small leading dimensions, and incorrect vector size will result in failed
 * \b assert() calls, i.e., using the \b casset macro.
 * All assertions are disabled if \b NDEBUG variable is defined,
 * except the sizes of the \b pure \b output variables discussed in the \b Resizing section
 * (the resizing requires a runtime check and cannot be disabled).
 *
 * \par Engine overloads
 * In addition to the overloads with default parameters, the HALA BLAS module adds overloads that
 * accept a hala::cpu_engine class. While the engine does not change anything about the calls,
 * it allows for a seamless switch between BLAS and cuBlas/rocBlas implementations by simply changing
 * the engine (also, when using hala::gpu_engine, all vector data must reside in GPU memory).
 *
 * \par BLAS Assumptions:
 *  - There is no explicit initialization of BLAS, i.e., there is no blas_init() call.
 *  - The functions follow the standard API, i.e., there is no need for a specialized header.
 *
 * \par Matrix Format Types
 * The BLAS standard defined several matrix formats:
 * - General, always in column major format.
 * - Banded, for example \f$ A = \left( \begin{array}{ccc} 1 & 0 & 0 \\ 2 & 3 & 0 \\ 0 & 4 & 5 \end{array} \right) \f$
 *   has dimension 3 but band kl = 1 and ku = 0 (one diagonal below the main is used),
 *   the vector storing \b A will have the format \f$ A = ( 1, 3, 5, 0, 2, 4 ) \f$.
 * - Symmetric \f$ A = \left( \begin{array}{cc} 1 & 2 + i \\ 2 + i & 4 \end{array} \right) \f$
 *   and Hermitian \f$ A = \left( \begin{array}{cc} 1 & 2 - i \\ 2 + i & 4 \end{array} \right) \f$,
 *   note that HALA always defaults to conjugate-symmetric Hermitian mode, see hala::non_conj.
 * - Triangular \f$ A = \left( \begin{array}{cc} 1 & 2 + i \\ 0 & 4 \end{array} \right) \f$,
 *   i.e., everything either below or above the diagonal is zero.
 * - Packed, triangular matrices that have only the lower/upper part stored, e.g.,
 *   \f$ A = \left( \begin{array}{cc} 1 & 2 \\ 0 & 4 \end{array} \right) \f$
 *   is packed into a vector \f$ A = ( 1, 2, 4 ) \f$ for \b uplo L.
 *
 * Note that when working with symmetric matrices, only the portion above or below diagonal is accessed,
 * thus symmetric matrices can also be stored in packed format.
 *
 */

#endif
