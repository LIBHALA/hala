#ifndef __HALA_BLAS_3_WRAPPERS_HPP
#define __HALA_BLAS_3_WRAPPERS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_blas_2.hpp"

namespace hala{
/*!
 * \internal
 * \file hala_blas_3.hpp
 * \brief HALA wrapper for BLAS level 3 functions.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALABLAS3
 *
 * Contains the HALA-BLAS level 3 wrapper templates.
 * \endinternal
*/

/*!
 * \ingroup HALABLAS
 * \addtogroup HALABLAS3 BLAS level 3 templates
 *
 * \par BLAS Level 3
 * Level 3 operations are defines as those that include a two or more matrices.
 * Examples of such operations are matrix-matrix product and triangular solve with multiple right hand sides.
 */

/*!
 * \ingroup HALABLAS3
 * \brief Wrapper to BLAS triangular matrix-matrix multiply xtrmm().
 *
 * Computes \f$ B = \alpha op(A) B \f$ or \f$ B = \alpha B op(A) \f$, where \b op is either identity or, conjugate or real transpose operators.
 * Here \b A is assumed to be upper or lower triangular matrix with either unit or non-unit diagonal.
 * - \b side must be lower or upper case L or R, indicates whether \b A is on the left (L) or right (R) of \b B
 * - \b uplo must lower or upper case U or L, indicates whether to work with the upper (U) or lower (L) part of \b A
 * - \b trans must be lower or upper case N (identity), T (transpose), or C (conjugate transpose), indicates \b op
 * - \b diag must be lower or upper case U or N, indicates whether \b A has unit (U) or non-unit (N) diagonal
 * - \b M indicates the number of rows of \b B
 * - \b N indicates the number of columns of \b B
 * - only the array elements of \b A indicated by \b uplo and \b diag are accessed, the rest is not read
 * - \b lda and \b ldb are the leading dimensions of \b A and \b B
 * - \b lda is minimum M for side L or N for size R, \b ldb is minimum M
 *
 * Overloads:
 * \code
 *  hala::trmm(side, uplo, trans, diag, M, N, alpha, A, B, lda = -1, ldb = -1);
 *  hala::trmm(engine, side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb);
 *  hala::trmm(engine, side, uplo, trans, diag, M, N, alpha, A, B, lda = -1, ldb = -1);
 * \endcode
 * The leading dimensions assume the minimum acceptable value, i.e., as if the matrices have no padding.
 */
template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trmm(char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, int lda, VectorLikeB &&B, int ldb){
    check_types(A, B);
    assert( valid::trmsm(side, uplo, trans, diag, M, N, A, lda, B, ldb) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto fsalpha = get_cast<scalar_type>(alpha);
    call_backend<scalar_type>(strmm_, dtrmm_, ctrmm_, ztrmm_,
                 &side, &uplo, &trans, &diag, &M, &N, pconvert(&fsalpha), convert(A), &lda, cconvert(B), &ldb);
}

/*!
 * \ingroup HALABLAS3
 * \brief Wrapper to BLAS triangular matrix-matrix solve xtrsm().
 *
 * Solves \f$ op(A) X = \alpha B \f$ or \f$ X op(A) = \alpha B \f$, where \b op is either identity or, conjugate or real transpose operators.
 * Here \b A is assumed to be upper or lower triangular matrix with either unit or non-unit diagonal.
 * - \b side must be lower or upper case L or R, indicates whether \b A is on the left (L) or right (R) of \b B
 * - \b uplo must lower or upper case U or L, indicates whether to work with the upper (U) or lower (L) part of \b A
 * - \b trans must be lower or upper case N (identity), T (transpose), or C (conjugate transpose), indicates \b op
 * - \b diag must be lower or upper case U or N, indicates whether \b A has unit (U) or non-unit (N) diagonal
 * - \b M indicates the number of rows of \b B
 * - \b N indicates the number of columns of \b B
 * - only the array elements of \b A indicated by \b uplo and \b diag are accessed, the rest is not read
 * - \b lda and \b ldb are the leading dimensions of \b A and \b B
 * - \b lda is minimum M for side L or N for size R, \b ldb is minimum M
 *
 * Overloads:
 * \code
 *  hala::trsm(side, uplo, trans, diag, M, N, alpha, A, B, lda = -1, ldb = -1);
 *  hala::trsm(engine, side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb);
 *  hala::trsm(engine, side, uplo, trans, diag, M, N, alpha, A, B, lda = -1, ldb = -1);
 * \endcode
 * The leading dimensions assume the minimum acceptable value, i.e., as if the matrices have no padding.
 */
template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trsm(char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, int lda, VectorLikeB &&B, int ldb){
    check_types(A, B);
    assert( valid::trmsm(side, uplo, trans, diag, M, N, A, lda, B, ldb) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto fsalpha = get_cast<scalar_type>(alpha);
    call_backend<scalar_type>(strsm_, dtrsm_, ctrsm_, ztrsm_,
                 &side, &uplo, &trans, &diag, &M, &N, pconvert(&fsalpha), convert(A), &lda, cconvert(B), &ldb);
}


/*!
 * \ingroup HALABLAS3
 * \brief Wrapper to BLAS symmetric or Hermitian rank-k matrix update xsyrk()/xherk().
 *
 * Computes \f$ C = \alpha op(A) A + \beta C \f$ or \f$ C = \alpha A op(A) + \beta C \f$.
 * Here \b C is assumed to be a symmetric or Hermitian matrix and only the upper or lower part of \b C is referenced.
 * - \b conjugate indicates whether \b C is conjugate-symmetric (Hermitian) or non-conjugate symmetric
 * - \b uplo is lower/upper case U/L, indicates whether to use the upper (U) or lower (L) part of \b C
 * - \b trans is lower/upper case N/T/C, indicates whether the op is on the first (T/C) or second (N) occurrence of \b A
 * - \b trans cannot be T if \b C is a Hermitian matrix
 * - \b N is the size of the square matrix \b C
 * - \b K indicates the number of rows (\b trans is T) or columns (\b trans is N) of \b A
 * - \b lda, \b ldc are the leading dimensions
 * - \b lda is minimum N if trans is N and minimum K if trans is T, \b ldc is minimum N
 *
 * Overloads:
 * \code
 *  hala::syrk(uplo, trans, N, K, alpha, A, beta, C, lda = -1, ldc = -1);
 *  hala::herk(uplo, trans, N, K, alpha, A, beta, C, lda = -1, ldc = -1);
 *  hala::syrk(engine, uplo, trans, N, K, alpha, A, lda, beta, C, ldc);
 *  hala::herk(engine, uplo, trans, N, K, alpha, A, lda, beta, C, ldc);
 *  hala::syrk(engine, uplo, trans, N, K, alpha, A, beta, C, lda = -1, ldc = -1);
 *  hala::herk(engine, uplo, trans, N, K, alpha, A, beta, C, lda = -1, ldc = -1);
 * \endcode
 * The leading dimensions assume the minimum acceptable value, i.e., as if the matrices have no padding.
 * The syrk() and herk() variants differ in whether the output is symmetric or conjugate symmetric (Hermitian).
 */
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void syrk(char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, int lda, FSb beta, VectorLikeC &&C, int ldc){
    check_types(A, C);
    check_set_size((hala_abs(beta) == 0.0), C, ldc, N);
    assert( valid::syrk(uplo, trans, N, K, A, lda, C, ldc) );

    using scalar_type = get_scalar_type<VectorLikeA>;

    auto castalpha = get_cast<scalar_type>(alpha);
    auto castbeta  = get_cast<scalar_type>(beta);

    call_backend6<conjugate, scalar_type>(ssyrk_, dsyrk_, cherk_, zherk_, csyrk_, zsyrk_,
        &uplo, &trans, &N, &K, pconvert(&castalpha), convert(A), &lda, pconvert(&castbeta), cconvert(C), &ldc);
}

/*!
 * \ingroup HALABLAS3
 * \brief Wrapper to BLAS symmetric or Hermitian 2x rank-k matrix update xsyr2k()/xher2k().
 *
 * Computes \f$ C = \alpha op(A) B + \bar \alpha op(B) A + \beta C \f$ or \f$ C = \alpha A op(B) + \bar \alpha B op(A) + \beta C \f$.
 * Here \b C is assumed to be a symmetric or Hermitian matrix and only the upper or lower part of \b C is referenced.
 * - \b conjugate indicates whether \b C is conjugate-symmetric (Hermitian) or non-conjugate symmetric
 * - \b uplo is lower/upper case U/L, indicates whether to use the upper (U) or lower (L) part of \b C
 * - \b trans is lower/upper case N/T/C, indicates whether the op is on the first (T/C) or second (N) occurrence of \b A
 * - \b N is the size of the square matrix \b C, \b ldc is minimum N
 * - \b K indicates the number of rows (\b trans is T) or columns (\b trans is N) of \b A and \b B
 * - \b lda and \b ldb are the leading dimensions, minimum the number of rows indicated above
 *
 * Overloads:
 * \code
 *  hala::syr2k(uplo, trans, N, K, alpha, A, B, beta, C, lda=-1, ldb=-1, ldc=-1);
 *  hala::her2k(uplo, trans, N, K, alpha, A, B, beta, C, lda=-1, ldb=-1, ldc=-1);
 *  hala::syr2k(engine, uplo, trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
 *  hala::her2k(engine, uplo, trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
 *  hala::syr2k(engine, uplo, trans, N, K, alpha, A, B, beta, C, lda=-1, ldb=-1, ldc=-1);
 *  hala::her2k(engine, uplo, trans, N, K, alpha, A, B, beta, C, lda=-1, ldb=-1, ldc=-1);
 * \endcode
 * The leading dimensions assume the minimum acceptable value, i.e., as if the matrices have no padding.
 * The difference between syr2k() and her2k() variants lies in the complex case where the matrix \b C is
 * either symmetric or conjugate-symmetric (Hermitian).
 */
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void syr2k(char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, int lda,
                  VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    check_types(A, B, C);
    check_set_size((hala_abs(beta) == 0.0), C, ldc, N);
    assert( valid::syr2k(uplo, trans, N, K, A, lda, B, ldb, C, ldc) );

    using scalar_type = get_scalar_type<VectorLikeA>;

    auto castalpha = get_cast<scalar_type>(alpha);
    auto castbeta  = get_cast<scalar_type>(beta);

    call_backend6<conjugate, scalar_type>(ssyr2k_, dsyr2k_, cher2k_, zher2k_, csyr2k_, zsyr2k_,
        &uplo, &trans, &N, &K, pconvert(&castalpha), convert(A), &lda, convert(B), &ldb, pconvert(&castbeta), cconvert(C), &ldc);
}

/*!
 * \ingroup HALABLAS3
 * \brief Wrapper to BLAS symmetric or Hermitian matrix multiply xsymm()/xhemm().
 *
 * Computes \f$ C = \alpha A B + \beta C \f$ or \f$ C = \alpha B A + \beta C \f$.
 * Here \b A is assumed to be a symmetric matrix and only the upper or lower part of \b A is referenced.
 * - \b conjugate indicates whether \b A is conjugate-symmetric (Hermitian) or non-conjugate symmetric
 * - \b side is lower/upper case L/R, indicates whether \b A is on the left (L) or right (R) of \b B
 * - \b uplo is lower/upper case U/L, indicates whether to use the upper (U) or lower (L) part of \b C
 * - \b M is the number of rows of \b B and \b C
 * - \b N is the number of columns of \b B and \b C
 * - \b lda, \b ldb and \b ldc are the leading dimensions
 * - \b lda is minimum M when side is L and N when side is R, \b ldb and \b ldc are minimum M
 *
 * Overloads:
 * \code
 *  hala::symm(side, uplo, M, N, alpha, A, B, beta, C, lda=-1, ldb=-1, ldc=-1);
 *  hala::hemm(side, uplo, M, N, alpha, A, B, beta, C, lda=-1, ldb=-1, ldc=-1);
 *  hala::symm(engine, side, uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
 *  hala::hemm(engine, side, uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
 *  hala::symm(engine, side, uplo, M, N, alpha, A, B, beta, C, lda=-1, ldb=-1, ldc=-1);
 *  hala::hemm(engine, side, uplo, M, N, alpha, A, B, beta, C, lda=-1, ldb=-1, ldc=-1);
 * \endcode
 * The leading dimensions assume the minimum acceptable value, i.e., as if the matrices have no padding.
 * The difference between symm() and hemm() variants lies in the complex case where the matrix is
 * either symmetric or conjugate-symmetric (Hermitian).
 */
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void symm(char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    check_types(A, B, C);
    check_set_size((hala_abs(beta) == 0.0), C, ldc, N);
    assert( valid::symm(side, uplo, M, N, A, lda, B, ldb, C, ldc) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto castalpha = static_cast<scalar_type>(alpha);
    auto castbeta  = static_cast<scalar_type>(beta);

    call_backend6<conjugate, scalar_type>(ssymm_, dsymm_, chemm_, zhemm_, csymm_, zsymm_,
        &side, &uplo, &M, &N, pconvert(&castalpha), convert(A), &lda, convert(B), &ldb, pconvert(&castbeta), cconvert(C), &ldc);
}

/*!
 * \ingroup HALABLAS3
 * \brief Wrapper to BLAS general matrix multiply xgemm().
 *
 * Computes \f$ C = \alpha op(A) op(B) + \beta C \f$, for general matrices, where \b op can be identity,
 * conjugate or transpose operations (there are two operations applied independently).
 * - \b transa and \b transb must be lower/upper case N (identity), T (transpose), or C (conjugate transpose)
 * - \b transa and \b transb indicate \b op on \b A and \b B respectively
 * - \b M is the rows of \b C and \b op(\b A)
 * - \b N is the columns of \b C and \b op(\b B)
 * - \b K indicates the inner dimension of the product
 * - \b lda, \b ldb or \b ldc are the leading dimensions
 * - \b lda is minimum M when transa is N and minimum K when transa is T or C
 * - \b ldc is minimum K when transb is N and minimum N when transb is T or C
 * - \b ldc is minimum M
 *
 * Overloads:
 * \code
 *  hala::gemm(transa, trnasb, M, N, K, alpha, A, B, beta, C, lda=-1, ldb =-1, ldc=-1);
 *  hala::gemm(engine, transa, trnasb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
 *  hala::gemm(engine, transa, trnasb, M, N, K, alpha, A, B, beta, C, lda=-1, ldb =-1, ldc=-1);
 * \endcode
 * The leading dimensions assume the minimum acceptable value, i.e., as if the matrices have no padding.
 */
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void gemm(char transa, char transb, int M, int N, int K, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    check_types(A, B, C);
    check_set_size((std::abs(beta) == 0.0), C, ldc, N);
    assert( valid::gemm(transa, transb, M, N, K, A, lda, B, ldb, C, ldc) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto castalpha = static_cast<scalar_type>(alpha);
    auto castbeta  = static_cast<scalar_type>(beta);
    call_backend<scalar_type>(sgemm_, dgemm_, cgemm_, zgemm_,
                 &transa, &transb, &M, &N, &K, pconvert(&castalpha), convert(A), &lda, convert(B), &ldb, pconvert(&castbeta), cconvert(C), &ldc);
}

}

#endif
