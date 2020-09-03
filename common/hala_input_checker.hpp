#ifndef __HALA_INPUT_CHECKER_HPP
#define __HALA_INPUT_CHECKER_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_types_checker.hpp"

namespace hala{

/*!
 * \internal
 * \file hala_input_checker.hpp
 * \ingroup HALABLASCHECKER
 * \brief Check methods for common api calls.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 *
 * Checks for common API calls, e.g., BLAS and CUDA gemm() calls.
 * \endinternal
 */

/*!
 * \internal
 * \ingroup HALACORE
 * \defgroup HALABLASCHECKER Input Consistency Checks
 *
 * Several standards mimic BLAS in calling conventions, e.g., BLAS, cuBlas,
 * and the included mockup BLAS. The methdos here provide consistent checks for
 * types and dimensions of the inputs that can be uses by all BLAS variants.
 * \endinternal
 */

//! \brief Wrapper namespace for input consistency.
namespace valid{

/****************************************************************************************************/
/*   BLAS helpers to set default lda and size parameters                                            */
/****************************************************************************************************/
/*!
 * \ingroup HALABLASCHECKER
 * \brief Sets defaults for vector size N.
 */
template<class VectorLikeX>
inline void default_size(VectorLikeX const &x, int incx, int &N){
    static_assert(!std::is_pointer<VectorLikeX>::value, "raw-pointers cannot be used in overloads that infer a default size");
    if (N == -1) N = 1 + (get_size_int(x) - 1) / incx;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Returns the size of the vector \b x, but only if \b x is not a pointer.
 */
template<class VectorLikeX>
inline int infer_size(VectorLikeX const &x){
    static_assert(!std::is_pointer<VectorLikeX>::value, "raw-pointers cannot be used in overloads that infer a default size");
    return get_size_int(x);
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Sets default for lda, if \b lda is -1 then set it to \b N.
 */
inline void default_ld(int N, int &lda){ if (lda == -1) lda = N; }

/*!
 * \ingroup HALABLASCHECKER
 * \brief Sets default for lda, if \b lda is -1 then take the first or second int based on \b first.
 */
inline void default_ld(bool first, int M, int N, int &lda){ if (lda == -1) lda = (first) ? M : N; }

#ifndef NDEBUG
/****************************************************************************************************/
/*   BLAS level 1                                                                                   */
/****************************************************************************************************/

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to vector copy.
 */
template<class VectorLikeX, class VectorLikeY>
inline bool vcopy(int N, VectorLikeX const &x, int incx, VectorLikeY &y, int incy){
    assert( incx > 0 );
    assert( incy > 0 );
    assert( N > 0 );
    assert( check_size(x, 1 + (N - 1) * incx) );
    assert( check_size(y, 1 + (N - 1) * incy) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to vector swap.
 */
template<class VectorLikeX, class VectorLikeY>
inline bool vswap(int N, VectorLikeX const &x, int incx, VectorLikeY &y, int incy){
    assert( incx > 0 );
    assert( incy > 0 );
    assert( N > 0 );
    assert( check_size(x, 1 + (N - 1) * incx) );
    assert( check_size(y, 1 + (N - 1) * incy) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to vector 2-norm.
 */
template<class VectorLikeX>
inline bool norm2(int N, VectorLikeX const &x, int incx){
    assert( incx > 0 );
    assert( N > 0 );
    assert( check_size(x, 1 + (N - 1) * incx) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to vector scale add, also sets N.
 */
template<class VectorLikeX, class VectorLikeY>
inline bool axpy(int N, VectorLikeX const &x, int incx, VectorLikeY &y, int incy){
    assert( incx > 0 );
    assert( incy > 0 );
    assert( N > 0 );
    assert( check_size(x, (N - 1) * incx + 1) );
    assert( check_size(y, (N - 1) * incy + 1) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to vector dot, also sets N.
 */
template<class VectorLikeX, class VectorLikeY>
inline bool dot(int N, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy){
    assert( incx > 0 );
    assert( incy > 0 );
    assert( check_size(x, 1 + (N - 1) * incx) );
    assert( check_size(y, 1 + (N - 1) * incy) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to vector scal, also sets N.
 */
template<class VectorLikeX>
inline bool scal(int N, VectorLikeX &x, int incx){
    assert( incx > 0 );
    assert( check_size(x, 1 + (N - 1) * incx) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to general matrix rank-1 update.
 */
template<class VectorLikeX, class VectorLikeY>
inline bool rot(int N, VectorLikeX &x, int incx, VectorLikeY &y, int incy){
    assert( incx > 0 );
    assert( incy > 0 );
    assert( N >= 0 );
    assert( check_size(x, 1 + (N - 1) * incx) );
    assert( check_size(y, 1 + (N - 1) * incy) );
    return true;
}

/****************************************************************************************************/
/*   BLAS level 2                                                                                   */
/****************************************************************************************************/

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix gemv.
 */
template<class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline bool gemv(char trans, int M, int N, VectorLikeA const &A, int lda,
                 VectorLikeX const &x, int incx, VectorLikeY &y, int incy){
    assert( check_trans(trans) );
    assert( incx > 0 );
    assert( incy > 0 );
    assert( lda >= M );
    assert( check_size(A, lda, N) );
    assert( check_size(x, 1 + incx * ((is_n(trans) ? N : M) - 1)) );
    assert( check_size(y, 1 + incy * ((is_n(trans) ? M : N) - 1)) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix gemb.
 */
template<class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline bool gbmv(char trans, int M, int N, int kl, int ku, VectorLikeA const &A, int lda,
                 VectorLikeX const &x, int incx, VectorLikeY &y, int incy){
    assert( check_trans(trans) );
    assert( incx > 0 );
    assert( incy > 0 );
    assert( ku >= 0 );
    assert( kl >= 0 );
    assert( ku < N ); // sub-diagonal do not exceed dimensions
    assert( kl < M );
    assert( lda >= (1 + ku + kl) );
    assert( check_size(A, lda, N) );
    assert( check_size(x, 1 + incx * ((is_n(trans) ? N : M) - 1)) );
    assert( check_size(y, 1 + incy * ((is_n(trans) ? M : N) - 1)) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix gemv.
 */
template<class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline bool symv(char uplo, int N, const VectorLikeA &A, int lda, const VectorLikeX &x, int incx, VectorLikeY &y, int incy){
    assert( check_uplo(uplo) );
    assert( incx > 0 );
    assert( incy > 0 );
    assert( lda >= N );
    assert( check_size(A, lda, N) );
    assert( check_size(x, 1 + incx * (N - 1)) );
    assert( check_size(y, 1 + incy * (N - 1)) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix gbmv.
 */
template<class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline bool sbmv(char uplo, int N, int k, const VectorLikeA &A, int lda, const VectorLikeX &x, int incx, VectorLikeY &y, int incy){
    assert( check_uplo(uplo) );
    assert( incx > 0 );
    assert( incy > 0 );
    assert( k >= 0 );
    assert( k < N );
    assert( lda >= (k+1) );
    assert( check_size(A, lda, N) );
    assert( check_size(x, 1 + incx * (N - 1)) );
    assert( check_size(y, 1 + incy * (N - 1)) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix gemv.
 */
template<class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline bool spmv(char uplo, int N, const VectorLikeA &A, const VectorLikeX &x, int incx, VectorLikeY &y, int incy){
    assert( check_uplo(uplo) );
    assert( incx > 0 );
    assert( incy > 0 );
    assert( check_size(A, ((N + 1) * N) / 2 ) );
    assert( check_size(x, 1 + incx * (N - 1)) );
    assert( check_size(y, 1 + incy * (N - 1)) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix triangular solve and multiply.
 */
template<class VectorLikeA, class VectorLikeX>
inline bool trmsv(char uplo, char trans, char diag, int N, const VectorLikeA &A, int lda, VectorLikeX x, int incx){
    assert( check_uplo(uplo) );
    assert( check_trans(trans) );
    assert( check_diag(diag) );
    assert( lda >= N );
    assert( incx > 0 );
    assert( check_size(A, lda, N) );
    assert( check_size(x, 1 + (N - 1) * incx) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix triangular solve and multiply, packed format.
 */
template<class VectorLikeA, class VectorLikeX>
inline bool tpmsv(char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &x, int incx){
    assert( check_uplo(uplo) );
    assert( check_trans(trans) );
    assert( check_diag(diag) );
    assert( incx > 0 );
    assert( check_size(A, (N * (N + 1)) / 2 ) );
    assert( check_size(x, 1 + (N - 1) * incx) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix triangular solve and multiply.
 */
template<class VectorLikeA, class VectorLikeX>
inline bool tbmsv(char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, int &lda, VectorLikeX &x, int incx){
    assert( check_uplo(uplo) );
    assert( check_trans(trans) );
    assert( check_diag(diag) );
    assert( k >= 0 );
    assert( k < N );
    assert( lda >= (k+1) );
    assert( incx > 0 );
    assert( check_size(A, lda, N) );
    assert( check_size(x, 1 + (N - 1) * incx) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to general matrix rank-1 update.
 */
template<class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline bool ger(int M, int N, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &A, int lda){
    assert( incx > 0 );
    assert( incy > 0 );
    assert( lda >= M );
    assert( check_size(A, lda, N) );
    assert( check_size(x, 1 + (M - 1) * incx) );
    assert( check_size(y, 1 + (N - 1) * incy) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to symmetric matrix rank-1 update.
 */
template<class VectorLikeX, class VectorLikeA>
inline bool syr(char uplo, int N, VectorLikeX const &x, int incx, VectorLikeA &A, int lda){
    assert( check_uplo(uplo) );
    assert( incx > 0 );
    assert( lda >= N );
    assert( check_size(A, lda, N) );
    assert( check_size(x, 1 + (N - 1) * incx) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to symmetric packed rank-1 update.
 */
template<class VectorLikeX, class VectorLikeA>
inline bool spr(char uplo, int N, VectorLikeX const &x, int incx, VectorLikeA &A){
    assert( check_uplo(uplo) );
    assert( incx > 0 );
    assert( check_size(A, ((N+1) * N) / 2 ) );
    assert( check_size(x, 1 + (N - 1) * incx) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to symmetric matrix rank-2 update.
 */
template<class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline bool syr2(char uplo, int N, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &A, int lda){
    assert( check_uplo(uplo) );
    assert( incx > 0 );
    assert( incy > 0 );
    assert( lda >= N );
    assert( check_size(A, lda, N) );
    assert( check_size(x, 1 + (N - 1) * incx) );
    assert( check_size(y, 1 + (N - 1) * incy) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to symmetric packed matrix rank-2 update.
 */
template<class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline bool spr2(char uplo, int N, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &A){
    assert( check_uplo(uplo) );
    assert( incx > 0 );
    assert( incy > 0 );
    assert( check_size(A, ((N+1) * N) / 2 ) );
    assert( check_size(x, 1 + (N - 1) * incx) );
    assert( check_size(y, 1 + (N - 1) * incy) );
    return true;
}

/****************************************************************************************************/
/*   BLAS level 3                                                                                   */
/****************************************************************************************************/

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix-matrix triangular solve and multiply, also sets lda and ldb.
 */
template<class VectorLikeA, class VectorLikeB>
inline bool trmsm(char side, char uplo, char trans, char diag, int M, int N, const VectorLikeA &A, int lda, VectorLikeB &B, int ldb){
    assert( check_side(side) );
    assert( check_uplo(uplo) );
    assert( check_trans(trans) );
    assert( check_diag(diag) );
    assert( lda >= (is_l(side) ? M : N) );
    assert( check_size(A, lda, is_l(side) ? M : N) );
    assert( ldb >= M );
    assert( check_size(B, ldb, N) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix-matrix symmetric rank-k update.
 */
template<class VectorLikeA, class VectorLikeC>
inline bool syrk(char uplo, char trans, int N, int K, VectorLikeA const &A, int lda, VectorLikeC &C, int ldc){
    assert( check_uplo(uplo) );
    assert( check_trans(trans) );
    assert( lda >= (is_n(trans) ? N : K) );
    assert( ldc >= N );
    assert( check_size(A, lda, is_n(trans) ? K : N) );
    assert( check_size(C, ldc, N) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix-matrix symmetric 2-rank-k update.
 */
template<class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline bool syr2k(char uplo, char trans, int N, int K, VectorLikeA const &A, int lda, VectorLikeB const &B, int ldb, VectorLikeC &C, int ldc){
    assert( check_uplo(uplo) );
    assert( check_trans(trans) );
    assert( lda >= (is_n(trans) ? N : K) );
    assert( ldb >= (is_n(trans) ? N : K) );
    assert( ldc >= N );
    assert( check_size(A, lda, is_n(trans) ? K : N) );
    assert( check_size(B, ldb, is_n(trans) ? K : N) );
    assert( check_size(C, ldc, N) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix-matrix symmetric multiply.
 */
template<class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline bool symm(char side, char uplo, int M, int N, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, VectorLikeC &C, int ldc){
    assert( check_side(side) );
    assert( check_uplo(uplo) );
    assert( lda >= (is_l(side) ? M : N) );
    assert( ldb >= M );
    assert( ldc >= M );
    assert( check_size(A, lda, is_l(side) ? M : N) );
    assert( check_size(B, ldb, N) );
    assert( check_size(C, ldc, N) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix-matrix general multiply.
 */
template<class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline bool gemm(char transa, char transb, int M, int N, int K, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, VectorLikeC &C, int ldc){
    assert( check_trans(transa) );
    assert( check_trans(transb) );
    assert( lda >= (is_n(transa) ? M : K) );
    assert( ldb >= (is_n(transb) ? K : N) );
    assert( ldc >= M );
    assert( check_size(A, lda, is_n(transa) ? K : M) );
    assert( check_size(B, ldb, is_n(transb) ? N : K) );
    assert( check_size(C, ldc, N) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix-vector sparse multiply.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeY>
bool sparse_gemv(char trans, int M, int N, int nnz, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, VectorLikeX const &x, VectorLikeY &y){
    assert( check_trans(trans) );
    assert( check_size(pntr, (is_n(trans)) ? M+1 : N+1 ) );
    assert( check_size(y, (is_n(trans)) ? M : N ) );
    assert( check_size(x, (is_n(trans)) ? N : M ) );
    assert( check_size(indx, nnz) );
    assert( check_size(vals, nnz) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix-matrix sparse multiply.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeB, class VectorLikeC>
bool sparse_gemm(char transa, char transb, int M, int N, int K, int nnz,
                 VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                 VectorLikeB const &B, int ldb, VectorLikeC &C, int ldc){
    assert( check_trans(transa) );
    assert( check_trans(transb) );
    assert( check_size(pntr, (is_n(transa)) ? M+1 : N+1 ) );
    assert( ldb >= (is_n(transb) ? K : N) );
    assert( ldc >= M );
    assert( check_size(indx, nnz) );
    assert( check_size(vals, nnz) );
    assert( check_size(B, ldb, is_n(transb) ? N : K) );
    assert( check_size(C, ldc, N) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix-vector sparse solve.
 */
template<class TriangularMat, class VectorLikeB>
bool sparse_trsv(char trans, TriangularMat &tri, VectorLikeB const &b){
    assert( check_trans(trans) );
    assert( check_size(b, tri.rows()) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to matrix-matrix sparse solve.
 */
template<class TriangularMat, class VectorLikeB>
bool sparse_trsm(char transa, char transb, int nrhs, TriangularMat &tri, VectorLikeB const &B, int ldb){
    assert( check_trans(transa) );
    assert( check_trans(transb) );
    assert( nrhs > 0 );
    assert( ldb >= (is_n(transb) ? tri.rows() : nrhs) );
    assert( check_size(B, ldb, is_n(transb) ? nrhs : tri.rows()) );
    return true;
}

/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to potrf().
 */
template<class VectorLikeA>
bool potrf(char uplo, int N, VectorLikeA const &A, int lda){
    check_uplo(uplo);
    assert( lda >= N );
    assert( check_size(A, lda, N) );
    return true;
}
/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to potrs().
 */
template<class VectorLikeA, class VectorLikeB>
bool potrs(char uplo, int N, int nrhs, VectorLikeA const &A, int lda, VectorLikeB const &B, int ldb){
    check_uplo(uplo);
    assert( lda >= N );
    assert( ldb >= N );
    assert( check_size(A, lda, N) );
    assert( check_size(B, ldb, nrhs) );
    return true;
}
/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to getrf().
 */
template<class VectorLikeA>
bool getrf(int M, int N, VectorLikeA const &A, int lda){
    assert( lda >= M );
    assert( check_size(A, lda, N) );
    return true;
}
/*!
 * \ingroup HALABLASCHECKER
 * \brief Check inputs to getrs().
 */
template<class VectorLikeA, class VectorLikeI, class VectorLikeB>
bool getrs(char trans, int N, int nrhs, VectorLikeA const &A, int lda, VectorLikeI const &ipiv, VectorLikeB const &B, int ldb){
    check_trans(trans);
    assert( lda >= N );
    assert( ldb >= N );
    assert( check_size(A, lda, N) );
    assert( check_size(ipiv, N) );
    assert( check_size(B, ldb, nrhs) );
    return true;
}

#endif

}

}

#endif
