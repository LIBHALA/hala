#ifndef __HALA_BLAS_2_WRAPPERS_HPP
#define __HALA_BLAS_2_WRAPPERS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_blas_1.hpp"

namespace hala{
/*!
 * \internal
 * \file hala_blas_2.hpp
 * \brief HALA wrapper for BLAS level 2 functions.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALABLAS2
 *
 * Contains the HALA-BLAS level 2 wrapper templates.
 * \endinternal
*/

/*!
 * \ingroup HALABLAS
 * \addtogroup HALABLAS2 BLAS level 2 templates
 *
 * \par BLAS Level 2
 * Level 2 operations are defines as those that include a single matrix and one or more vectors.
 * Examples of such operations are matrix vector product and triangular solve.
 */

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS general matrix-vector multiply xgemv().
 *
 * Computes \f$ y = \alpha op(A) x + \beta y \f$, where \b op is either identity or, conjugate or real transpose operators.
 * - \b trans must be lower or upper case N (identity), T (transpose), or C (conjugate transpose), indicates \b op
 * - \b A must be at least N by lda
 * - \b lda is the leading dimension of \b A, minimum M
 * - \b incx and \b incy are strides of the entries of \b x and \b y
 * - if \b beta is zero, \b y is a pure output and will be resized if the size is not sufficient
 *
 * Overloads:
 * \code
 *  hala::gemv(trans, M, N, alpha, A, x, beta, y, lda = -1, incx = 1, incy = 1);
 *  hala::gemv(engine, trans, M, N, alpha, A, lda, x, incx, beta, y, incy);
 *  hala::gemv(engine, trans, M, N, alpha, A, x, beta, y, lda = -1, incx = 1, incy = 1);
 * \endcode
 * The increments default to 1 and the leading dimension is replaced by the minimum acceptable (i.e., the matrix has no padding).
 */
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gemv(char trans, int M, int N, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    check_types(A, x, y);
    check_set_size((hala_abs(beta) == 0.0), y, 1 + incy * ((is_n(trans) ? M : N) - 1));
    assert( valid::gemv(trans, M, N, A, lda, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto fsalpha = get_cast<scalar_type>(alpha);
    auto fsbeta  = get_cast<scalar_type>(beta);
    call_backend<scalar_type>(sgemv_, dgemv_, cgemv_, zgemv_,
                              &trans, &M, &N, pconvert(&fsalpha), convert(A), &lda, convert(x), &incx, pconvert(&fsbeta), cconvert(y), &incy);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS general banded matrix-vector multiply xgbmv().
 *
 * Essentially identical to hala::gemv() but using banded format for \b A.
 * - \b kl and \b ku indicates the number of sub-diagonals below and above the main diagonal
 * - \b lda is the leading dimension of the banded representation of \b A (minimum kl + ku + 1)
 *
 * Overloads:
 * \code
 *  hala::gbmv(trans, M, N, kl, ku, alpha, A, x, beta, y, lda=-1, incx=1, incy=1);
 *  hala::gbmv(engine, trans, M, N, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
 *  hala::gbmv(engine, trans, M, N, kl, ku, alpha, A, x, beta, y, lda=-1, incx=1, incy=1);
 * \endcode
 * The increments default to 1 and the leading dimension is replaced by the minimum acceptable.
 */
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gbmv(char trans, int M, int N, int kl, int ku, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    check_types(A, x, y);
    check_set_size((hala_abs(beta) == 0.0), y, 1 + incy * ((is_n(trans) ? M : N) - 1));
    assert( valid::gbmv(trans, M, N, kl, ku, A, lda, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto fsalpha = get_cast<scalar_type>(alpha);
    auto fsbeta  = get_cast<scalar_type>(beta);
    call_backend<scalar_type>(sgbmv_, dgbmv_, cgbmv_, zgbmv_,
                              &trans, &M, &N, &kl, &ku, pconvert(&fsalpha), convert(A), &lda, convert(x), &incx, pconvert(&fsbeta), cconvert(y), &incy);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS symmetric or Hermitian matrix-vector multiply xsymv().
 *
 * Computes \f$ y = \alpha A x + \beta y \f$, where \b A is either symmetric or conjugate-symmetric (Hermitian).
 * - \b uplo is lower/upper case U/L, indicates whether to use the upper (U) or lower (L) part of \b A
 * - \b A must be at least N by lda
 * - \b lda is the leading dimension of \b A, minimum N
 * - \b incx and \b incy are strides of the entries of \b x and \b y
 * - if \b beta is zero, \b y is a pure output and will be resized if the size is not sufficient
 *
 * Overloads:
 * \code
 *  hala::symv(uplo, N, alpha, A, x, beta, y, lda = -1, incx = 1, incy = 1);
 *  hala::hemv(uplo, N, alpha, A, x, beta, y, lda = -1, incx = 1, incy = 1);
 *  hala::symv(engine, uplo, N, alpha, A, lda, x, incx, beta, y, incy);
 *  hala::hemv(engine, uplo, N, alpha, A, lda, x, incx, beta, y, incy);
 *  hala::symv(engine, uplo, N, alpha, A, x, beta, y, lda = -1, incx = 1, incy = 1);
 *  hala::hemv(engine, uplo, N, alpha, A, x, beta, y, lda = -1, incx = 1, incy = 1);
 * \endcode
 * The increments default to 1 and the leading dimension is replaced by the minimum acceptable (i.e., the matrix has no padding).
 * The symv() and hemv() variants differ only in the complex case,
 * where \b A is assumed to be symmetric or conjugate symmetric (Hermitian).
 */
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void symv(char uplo, int N, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    check_types(A, x, y);
    check_set_size((hala_abs(beta) == 0.0), y, 1 + incy * (N - 1));
    assert( valid::symv(uplo, N, A, lda, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto fsalpha = get_cast<scalar_type>(alpha);
    auto fsbeta  = get_cast<scalar_type>(beta);

    call_backend6<conjugate, scalar_type>(ssymv_, dsymv_, chemv_, zhemv_, csymv_, zsymv_,
                                          &uplo, &N, pconvert(&fsalpha), convert(A), &lda, convert(x), &incx, pconvert(&fsbeta), cconvert(y), &incy);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS symmetric or Hermitian banded matrix-vector multiply xsbmv().
 *
 * Essentially identical to hala::symv() but using banded format for \b A.
 * - \b k indicates the number of sub-diagonals (symmetric on both sides)
 * - \b lda is the leading dimension of the banded representation of \b A (minimum k+1)
 *
 * Overloads:
 * \code
 *  hala::sbmv(uplo, N, k, alpha, A, x, beta, y, lda = -1, incx = 1, incy = 1);
 *  hala::hbmv(uplo, N, k, alpha, A, x, beta, y, lda = -1, incx = 1, incy = 1);
 *  hala::sbmv(engine, uplo, N, k, alpha, A, lda, x, incx, beta, y, incy);
 *  hala::hbmv(engine, uplo, N, k, alpha, A, lda, x, incx, beta, y, incy);
 *  hala::sbmv(engine, uplo, N, k, alpha, A, x, beta, y, lda = -1, incx = 1, incy = 1);
 *  hala::hbmv(engine, uplo, N, k, alpha, A, x, beta, y, lda = -1, incx = 1, incy = 1);
 * \endcode
 * The increments default to 1 and the leading dimension is replaced by the minimum acceptable (i.e., the matrix has no padding).
 * The sbmv() and hbmv() variants differ only in the complex case,
 * where \b A is assumed to be symmetric or conjugate symmetric (Hermitian).
 */
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void sbmv(char uplo, int N, int k, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    check_types(A, x, y);
    check_set_size((hala_abs(beta) == 0.0), y, 1 + incy * (N - 1));
    assert( valid::sbmv(uplo, N, k, A, lda, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto fsalpha = get_cast<scalar_type>(alpha);
    auto fsbeta  = get_cast<scalar_type>(beta);

    call_backend6<conjugate, scalar_type>(ssbmv_, dsbmv_, chbmv_, zhbmv_, csbmv_, zsbmv_,
                                          &uplo, &N, &k, pconvert(&fsalpha), convert(A), &lda, convert(x), &incx, pconvert(&fsbeta), cconvert(y), &incy);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS symmetric or Hermitian matrix-vector multiply xspmv() in packed format.
 *
 * Essentially identical to hala::symv() but using packed format for \b A.
 * - \b A must have size N * (N + 1) / 2, there is no \b lda
 *
 * Overloads:
 * \code
 *  hala::spmv(uplo, N, alpha, A, x, beta, y, incx = 1, incy = 1);
 *  hala::hpmv(uplo, N, alpha, A, x, beta, y, incx = 1, incy = 1);
 *  hala::spmv(engine, uplo, N, alpha, A, x, incx, beta, y, incy);
 *  hala::hpmv(engine, uplo, N, alpha, A, x, incx, beta, y, incy);
 *  hala::spmv(engine, uplo, N, alpha, A, x, beta, y, incx = 1, incy = 1);
 *  hala::hpmv(engine, uplo, N, alpha, A, x, beta, y, incx = 1, incy = 1);
 * \endcode
 * The increments default to 1 and the leading dimension is replaced by the minimum acceptable.
 */
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void spmv(char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    check_types(A, x, y);
    check_set_size((hala_abs(beta) == 0.0), y, 1 + incy * (N - 1));
    assert( valid::spmv(uplo, N, A, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto fsalpha = get_cast<scalar_type>(alpha);
    auto fsbeta  = get_cast<scalar_type>(beta);

    call_backend6<conjugate, scalar_type>(sspmv_, dspmv_, chpmv_, zhpmv_, cspmv_, zspmv_,
                                          &uplo, &N, pconvert(&fsalpha), convert(A), convert(x), &incx, pconvert(&fsbeta), cconvert(y), &incy);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS triangular matrix-vector multiply xtrmv().
 *
 * Computes \f$ x = op(A) x \f$, where \b op is either identity or, conjugate or real transpose operators.
 * Here \b A is assumed to be upper or lower triangular matrix with either unit or non-unit diagonal.
 * - \b uplo must lower or upper case U or L, indicates whether to work with the upper (U) or lower (L) part of \b A
 * - \b trans must be lower or upper case N (identity), T (transpose), or C (conjugate transpose), indicates \b op
 * - \b diag must be lower or upper case U or N, indicates whether \b A has unit (U) or non-unit (N) diagonal
 * - only the array elements of \b A indicated by \b uplo and \b diag are accessed, the rest is not read
 * - \b lda is the leading dimension of \b A (minimum N) and \b incx is the stride of vector entries
 *
 * Overloads:
 * \code
 *  hala::trmv(uplo, trans, diag, N, A, x, lda = -1, incx = 1);
 *  hala::trmv(engine, uplo, trans, diag, N, A, lda, x, incx);
 *  hala::trmv(engine, uplo, trans, diag, N, A, x, lda = -1, incx = 1);
 * \endcode
 * The increment defaults to 1 and the leading dimension is replaced by the minimum acceptable (i.e., the matrix has no padding).
 */
template<class VectorLikeA, class VectorLikeX>
inline void trmv(char uplo, char trans, char diag, int N, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    check_types(A, x);
    assert( valid::trmsv(uplo, trans, diag, N, A, lda, x, incx) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    call_backend<scalar_type>(strmv_, dtrmv_, ctrmv_, ztrmv_,
                              &uplo, &trans, &diag, &N, convert(A), &lda, cconvert(x), &incx);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS triangular matrix-vector solve xtrsv().
 *
 * Solves \f$ op(A) x = b \f$, where \b op is either identity or, conjugate or real transpose operators.
 * Here \b A is assumed to be upper or lower triangular matrix with either unit or non-unit diagonal.
 * - \b uplo must lower or upper case U or L, indicates whether to work with the upper (U) or lower (L) part of \b A
 * - \b trans must be lower or upper case N (identity), T (transpose), or C (conjugate transpose), indicates \b op
 * - \b diag must be lower or upper case U or N, indicates whether \b A has unit (U) or non-unit (N) diagonal
 * - only the array elements of \b A indicated by \b uplo and \b diag are accessed, the rest is not read
 * - \b x must have \b N active entries, where the active entries lay in a stride of \b incx
 * - on entry \b x is the right hand side of the equation, on exit \b x is the solution
 * - \b lda is the leading dimension of \b A (minimum N)
 *
 * Overloads:
 * \code
 *  hala::trsv(uplo, trans, diag, N, A, x, lda = -1, incx = 1);
 *  hala::trsv(engine, uplo, trans, diag, N, A, lda, x, incx);
 *  hala::trsv(engine, uplo, trans, diag, N, A, x, lda = -1, incx = 1);
 * \endcode
 * The increment defaults to 1 and the leading dimension is replaced by the minimum acceptable (i.e., the matrix has no padding).
 */
template<class VectorLikeA, class VectorLikeX>
inline void trsv(char uplo, char trans, char diag, int N, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    check_types(A, x);
    assert( valid::trmsv(uplo, trans, diag, N, A, lda, x, incx) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    call_backend<scalar_type>(strsv_, dtrsv_, ctrsv_, ztrsv_,
                              &uplo, &trans, &diag, &N, convert(A), &lda, cconvert(x), &incx);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS triangular banded matrix-vector multiply xtbmv().
 *
 * Essentially identical to hala::trsv() but using banded format for \b A.
 * - \b k indicates the number of sub-diagonals
 * - \b lda is the leading dimension of the banded representation of \b A (minimum k+1)
 *
 * Overloads:
 * \code
 *  hala::tbmv(uplo, trans, diag, N, k, A, x, lda = -1, incx = 1);
 *  hala::tbmv(engine, uplo, trans, diag, N, k, A, lda, x, incx);
 *  hala::tbmv(engine, uplo, trans, diag, N, k, A, x, lda = -1, incx = 1);
 * \endcode
 * The increment defaults to 1 and the leading dimension is replaced by the minimum acceptable.
 */
template<class VectorLikeA, class VectorLikeX>
inline void tbmv(char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    check_types(A, x);
    assert( valid::tbmsv(uplo, trans, diag, N, k, A, lda, x, incx) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    call_backend<scalar_type>(stbmv_, dtbmv_, ctbmv_, ztbmv_,
                              &uplo, &trans, &diag, &N, &k, convert(A), &lda, cconvert(x), &incx);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS triangular matrix-vector solve xtbsv().
 *
 * Essentially identical to hala::trsv() but using banded format for \b A.
 * - \b k indicates the number of sub-diagonals
 * - \b lda is the leading dimension of the banded representation of \b A (minimum k+1)
 *
 * Overloads:
 * \code
 *  hala::tbsv(uplo, trans, diag, N, k, A, x, lda = -1, incx = 1);
 *  hala::tbsv(engine, uplo, trans, diag, N, k, A, lda, x, incx);
 *  hala::tbsv(engine, uplo, trans, diag, N, k, A, x, lda = -1, incx = 1);
 * \endcode
 * The increment defaults to 1 and the leading dimension is replaced by the minimum acceptable.
 */
template<class VectorLikeA, class VectorLikeX>
inline void tbsv(char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    check_types(A, x);
    assert( valid::tbmsv(uplo, trans, diag, N, k, A, lda, x, incx) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    call_backend<scalar_type>(stbsv_, dtbsv_, ctbsv_, ztbsv_,
                              &uplo, &trans, &diag, &N, &k, convert(A), &lda, cconvert(x), &incx);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS triangular matrix-vector multiply in packed format xtpmv().
 *
 * Essentially identical to hala::trmv() but using packed format for \b A.
 * - \b A must have size N * (N + 1) / 2, there is no \b lda
 *
 * Overloads:
 * \code
 *  hala::tpmv(uplo, trans, diag, N, A, x, incx = 1);
 *  hala::tpmv(engine, uplo, trans, diag, N, A, x, incx = 1);
 * \endcode
 * The increment defaults to 1 and the leading dimension is replaced by the minimum acceptable (i.e., the matrix has no padding).
 */
template<class VectorLikeA, class VectorLikeX>
inline void tpmv(char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int incx = 1){
    check_types(A, x);
    assert( valid::tpmsv(uplo, trans, diag, N, A, x, incx) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    call_backend<scalar_type>(stpmv_, dtpmv_, ctpmv_, ztpmv_,
                              &uplo, &trans, &diag, &N, convert(A), cconvert(x), &incx);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS triangular matrix-vector solve in packed format xtpsv().
 *
 * Essentially identical to hala::trsv() but using packed format for \b A.
 * - \b A must have size N * (N + 1) / 2, there is no \b lda
 *
 * Overloads:
 * \code
 *  hala::tpsv(uplo, trans, diag, N, A, x, incx = 1);
 *  hala::tpsv(engine, uplo, trans, diag, N, A, x, incx = 1);
 * \endcode
 * The increment defaults to 1 and the leading dimension is replaced by the minimum acceptable (i.e., the matrix has no padding).
 */
template<class VectorLikeA, class VectorLikeX>
inline void tpsv(char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int incx = 1){
    check_types(A, x);
    assert( valid::tpmsv(uplo, trans, diag, N, A, x, incx) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    call_backend<scalar_type>(stpsv_, dtpsv_, ctpsv_, ztpsv_,
                              &uplo, &trans, &diag, &N, convert(A), cconvert(x), &incx);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS general rank-1 update xgerx().
 *
 * Computes \f$ A = A + \alpha x y^T \f$ with either conjugate or non-conjugate transpose.
 * - \b conjugate indicates whether to consider real or complex transpose (defaults to complex)
 * - \b A is an \b M by \b N matrix with leading dimension \b lda (minimum M)
 * - \b x and \b y are vectors with entries in strides \b incx and \b incy
 * - the number of active entries in \b x and \b y must be \b M and \b N respectively
 *
 * Overloads:
 * \code
 *  hala::ger(M, N, alpha, x, y, A, incx = 1, incy = 1, lda = -1);
 *  hala::geru(M, N, alpha, x, y, A, incx = 1, incy = 1, lda = -1);
 *  hala::ger(engine, M, N, alpha, x, incx, y, incy, A, lda);
 *  hala::geru(engine, M, N, alpha, x, incx, y, incy, A, lda);
 *  hala::ger(engine, M, N, alpha, x, y, A, incx = 1, incy = 1, lda = -1);
 *  hala::geru(engine, M, N, alpha, x, y, A, incx = 1, incy = 1, lda = -1);
 * \endcode
 * The increment defaults to 1 and the leading dimension is replaced by the minimum acceptable.
 * The geru() variant applies the transpose without conjugation.
 */
template<bool conjugate = true, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void ger(int M, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    check_types(x, y, A);
    assert( valid::ger(M, N, x, incx, y, incy, A, lda) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    auto fsalpha = get_cast<scalar_type>(alpha);

    call_backend6<conjugate, scalar_type>(sger_, dger_, cgerc_, zgerc_, cgeru_, zgeru_,
                                          &M, &N, pconvert(&fsalpha), convert(x), &incx, convert(y), &incy, cconvert(A), &lda);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS symmetric or Hermitian rank-1 update xsyr() and xher().
 *
 * Computes \f$ A = A + \alpha x x^T \f$ with either conjugate or non-conjugate transpose.
 * - \b conjugate indicates whether to consider real or complex transpose (defaults to complex)
 * - \b uplo is lower or upper case U or L indicating whether to use the lower or upper part of \b A
 * - \b A is an \b N by \b N matrix with leading dimension \b lda (minimum N)
 * - \b x is a vector with entries in a stride \b incx and must have \b N active entries
 *
 * Overloads:
 * \code
 *  hala::syr(uplo, N, alpha, x, A, incx = 1, lda = -1);
 *  hala::her(uplo, N, alpha, x, A, incx = 1, lda = -1);
 *  hala::syr(engine, uplo, N, alpha, x, incx, A, lda);
 *  hala::her(engine, uplo, N, alpha, x, incx, A, lda);
 *  hala::syr(engine, uplo, N, alpha, x, A, incx = 1, lda = -1);
 *  hala::her(engine, uplo, N, alpha, x, A, incx = 1, lda = -1);
 * \endcode
 * The increment defaults to 1 and the leading dimension is replaced by the minimum acceptable.
 * The difference between her() and syr() is that the her() variant uses the conjugate transpose.
 */
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void syr(char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A, int lda){
    check_types(x, A);
    assert( valid::syr(uplo, N, x, incx, A, lda) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    auto fsalpha = get_cast<scalar_type>(alpha);

    call_backend6<conjugate, scalar_type>(ssyr_, dsyr_, cher_, zher_, csyr_, zsyr_,
                                          &uplo, &N, pconvert(&fsalpha), convert(x), &incx, cconvert(A), &lda);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS symmetric or Hermitian rank-1 update using packed format xspr() and xhpr().
 *
 * Essentially identical to hala::syr() but the matrix \b A is stored in packed format.
 * - \b A must have size N * (N + 1) / 2, there is no \b lda
 *
 * Overloads:
 * \code
 *  hala::spr(uplo, N, alpha, x, A, incx = 1);
 *  hala::hpr(uplo, N, alpha, x, A, incx = 1);
 *  hala::spr(engine, uplo, N, alpha, x, incx, A);
 *  hala::hpr(engine, uplo, N, alpha, x, incx, A);
 *  hala::spr(engine, uplo, N, alpha, x, A, incx = 1);
 *  hala::hpr(engine, uplo, N, alpha, x, A, incx = 1);
 * \endcode
 * The increment defaults to 1 and the leading dimension is replaced by the minimum acceptable.
 */
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void spr(char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A){
    check_types(x, A);
    assert( valid::spr(uplo, N, x, incx, A) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    auto fsalpha = get_cast<scalar_type>(alpha);

    call_backend6<conjugate, scalar_type>(sspr_, dspr_, chpr_, zhpr_, cspr_, zspr_,
                                          &uplo, &N, pconvert(&fsalpha), convert(x), &incx, cconvert(A));
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS symmetric or Hermitian rank-2 update xsyr2() and xher2().
 *
 * Computes \f$ A = A + \alpha x y^T + y x^T \f$ with either conjugate or non-conjugate transpose.
 * - \b conjugate indicates whether to consider real or complex transpose (defaults to complex)
 * - \b uplo is lower or upper case U or L indicating whether to use the lower or upper part of \b A
 * - \b A is an \b N by \b N matrix with leading dimension \b lda (minimum N)
 * - \b x and \b y are vectors with entries in strides \b incx and \b incy
 * - the number of active entries in \b x and \b y must be \b N
 *
 * Overloads:
 * \code
 *  hala::syr2(uplo, N, alpha, x, y, A, incx = 1, incy = 1, lda = -1);
 *  hala::her2(uplo, N, alpha, x, y, A, incx = 1, incy = 1, lda = -1);
 *  hala::syr2(engine, uplo, N, alpha, x, incx, y, incy, A, lda);
 *  hala::her2(engine, uplo, N, alpha, x, incx, y, incy, A, lda);
 *  hala::syr2(engine, uplo, N, alpha, x, y, A, incx = 1, incy = 1, lda = -1);
 *  hala::her2(engine, uplo, N, alpha, x, y, A, incx = 1, incy = 1, lda = -1);
 * \endcode
 * The increment defaults to 1 and the leading dimension is replaced by the minimum acceptable.
 * The her2() and syr2() variants differ in the use of conjugate and non-conjugate transpose.
 */
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void syr2(char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    check_types(x, A);
    assert( valid::syr2(uplo, N, x, incx, y, incy, A, lda) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    auto fsalpha = get_cast<scalar_type>(alpha);

    call_backend6<conjugate, scalar_type>(ssyr2_, dsyr2_, cher2_, zher2_, csyr2_, zsyr2_,
                                          &uplo, &N, pconvert(&fsalpha), convert(x), &incx, convert(y), &incy, cconvert(A), &lda);
}

/*!
 * \ingroup HALABLAS2
 * \brief Wrapper to BLAS symmetric or Hermitian rank-2 update in packed format xspr2() and xhpr2().
 *
 * Essentially identical to hala::syr2() but the matrix \b A is stored in packed format.
 * - \b A must have size N * (N + 1) / 2, there is no \b lda
 *
 * Overloads:
 * \code
 *  hala::spr2(uplo, N, alpha, x, y, A, incx = 1, incy = 1);
 *  hala::hpr2(uplo, N, alpha, x, y, A, incx = 1, incy = 1);
 *  hala::spr2(engine, uplo, N, alpha, x, incx, y, incy, A);
 *  hala::hpr2(engine, uplo, N, alpha, x, incx, y, incy, A);
 *  hala::spr2(engine, uplo, N, alpha, x, y, A, incx = 1, incy = 1);
 *  hala::hpr2(engine, uplo, N, alpha, x, y, A, incx = 1, incy = 1);
 * \endcode
 * The increment defaults to 1 and the leading dimension is replaced by the minimum acceptable.
 * The hpr2() and spr2() variants differ in the use of conjugate and non-conjugate transpose.
 */
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void spr2(char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A){
    check_types(x, A);
    assert( valid::spr2(uplo, N, x, incx, y, incy, A) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    auto fsalpha = get_cast<scalar_type>(alpha);

    call_backend6<conjugate, scalar_type>(sspr2_, dspr2_, chpr2_, zhpr2_, cspr2_, zspr2_,
                                          &uplo, &N, pconvert(&fsalpha), convert(x), &incx, convert(y), &incy, cconvert(A));
}

#ifndef __HALA_DOXYGEN_SKIP
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hemv(char uplo, int N, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    symv<true>(uplo, N, alpha, A, lda, x, incx, beta, std::forward<VectorLikeY>(y), incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hbmv(char uplo, int N, int k, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    sbmv<true>(uplo, N, k, alpha, A, lda, x, incx, beta, std::forward<VectorLikeY>(y), incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hpmv(char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    spmv<true>(uplo, N, alpha, A, x, incx, beta, std::forward<VectorLikeY>(y), incy);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void geru(int M, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    ger<false>(M, N, alpha, x, incx, y, incy, std::forward<VectorLikeA>(A), lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeA>
inline void her(char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A, int lda){
    syr<true>(uplo, N, alpha, x, incx, std::forward<VectorLikeA>(A), lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeA>
inline void hpr(char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A){
    spr<true>(uplo, N, alpha, x, incx, A);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void her2(char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    syr2<true>(uplo, N, alpha, x, incx, y, incy, std::forward<VectorLikeA>(A), lda);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void hpr2(char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A){
    spr2<true>(uplo, N, alpha, x, incx, y, incy, std::forward<VectorLikeA>(A));
}
#endif

}

#endif
