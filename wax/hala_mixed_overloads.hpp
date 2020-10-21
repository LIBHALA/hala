#ifndef __HALA_MIXED_OVERLOADS_HPP
#define __HALA_MIXED_OVERLOADS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_lib_extensions.hpp"

/*!
 * \internal
 * \file hala_mixed_overloads.hpp
 * \ingroup HALACORE
 * \brief Contains the overloads for the mixed engine.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 *
 * All overloads involving the mixed engine.
 * \endinternal
 */

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)

#ifdef HALA_ENABLE_GPU

namespace hala{

/***********************************************************************************************************************/
/* BLAS level 0                                                                                                        */
/***********************************************************************************************************************/
template<typename FPa, class VectorLikeA, typename FPb, class VectorLikeB, class VectorLikeC>
inline void geam(mixed_engine const &e, char transa, char transb, int M, int N, FPa alpha, VectorLikeA const &A,
                 FPb beta, VectorLikeB const &B, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    auto gA = e.gpu().load(A);
    auto gB = e.gpu().load(B);
    auto gC = gpu_bind_vector(e.gpu(), C);
    geam(e.gpu(), transa, transb, M, N, alpha, gA, beta, gB, gC, lda, ldb, ldc);
}
template<typename FPa, class VectorLikeA, typename FPb, class VectorLikeB, class VectorLikeC>
inline void geam(mixed_engine const &e, char transa, char transb, int M, int N, FPa alpha, VectorLikeA const &A, int lda,
                 FPb beta, VectorLikeB const &B, int ldb, VectorLikeC &&C, int ldc){
    geam(e, transa, transb, M, N, alpha, A, beta, B, C, lda, ldb, ldc);
}

template<class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void dgmm(mixed_engine const &e, char side, int M, int N, VectorLikeA const &A, int lda,
                 VectorLikeB const &x, int incx, VectorLikeC &&C, int ldc){
    auto gA = e.gpu().load(A);
    auto gx = e.gpu().load(x);
    auto gC = gpu_bind_vector(e.gpu(), C);
    dgmm(e.gpu(), side, M, N, gA, gx, gC, lda, incx, ldc);
}
template<class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void dgmm(mixed_engine const &e, char side, int M, int N, VectorLikeA const &A,
                 VectorLikeB const &x, VectorLikeC &&C, int lda = -1, int incx = 1, int ldc = -1){
    dgmm(e, side, M, N, A, lda, x, incx, C, ldc);
}

template<class VectorLikeA, class VectorLikeAP>
void tp2tr(mixed_engine const &e, char uplo, int N, VectorLikeAP const &AP, VectorLikeA &&A, int lda = -1){
    auto gAP = e.gpu().load(AP);
    auto gA = gpu_bind_vector(e.gpu(), A);
    tp2tr(e.gpu(), uplo, N, gAP, gA, lda);
}
template<class VectorLikeA, class VectorLikeAP>
void tr2tp(mixed_engine const &e, char uplo, int N, VectorLikeAP const &A, int lda, VectorLikeA &&AP){
    auto gA = e.gpu().load(A);
    auto gAP = gpu_bind_vector(e.gpu(), AP);
    tr2tp(e.gpu(), uplo, N, gA, gAP, lda);
}
template<class VectorLikeA, class VectorLikeAP>
void tr2tp(mixed_engine const &e, char uplo, int N, VectorLikeAP const &A, VectorLikeA &&AP, int lda = -1){
    tr2tp(e, uplo, N, A, lda, AP);
}

/***********************************************************************************************************************/
/* BLAS level 1                                                                                                        */
/***********************************************************************************************************************/
template<class VectorLikeX, class VectorLikeY>
inline void vcopy(mixed_engine const&, int N, VectorLikeX const &x, int incx, VectorLikeY &&y, int incy){ vcopy(N, x, incx, y, incy); }
template<class VectorLikeX, class VectorLikeY>
inline void vcopy(mixed_engine const&, VectorLikeX const &x, VectorLikeY &&y, int incx = 1, int incy = 1, int N = -1){
    vcopy(x, y, incx, incy, N);
}
template<class VectorLike>
auto vcopy(mixed_engine const&, VectorLike const &x){
    cpu_engine cpu;
    auto y = new_vector(cpu, x);
    vcopy(x, y);
    return y;
}

template<class VectorLikeX, class VectorLikeY>
inline void vswap(mixed_engine const &e, int N, VectorLikeX &&x, int incx, VectorLikeY &&y, int incy){
    auto gx = gpu_bind_vector(e.gpu(), x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    vswap(e.gpu(), gx, gy, incx, incy, N); // handles the defaults
}
template<class VectorLikeX, class VectorLikeY>
inline void vswap(mixed_engine const &e, VectorLikeX &&x, VectorLikeY &&y, int incx = 1, int incy = 1, int N = -1){ vswap(e, N, x, incx, y, incy); }

template<class VectorLikeX> inline auto norm2(mixed_engine const &e, int N, VectorLikeX const &x, int incx){
    auto gx = e.gpu().load(x);
    return norm2(e.gpu(), gx, incx, N);
}
template<class VectorLikeX> inline auto norm2(mixed_engine const &e, VectorLikeX const &x, int incx = 1, int N = -1){ return norm2(e, N, x, incx); }

template<class VectorLikeX> inline auto asum(mixed_engine const &e, int N, VectorLikeX const &x, int incx){
    auto gx = e.gpu().load(x);
    return asum(e.gpu(), gx, incx, N);
}
template<class VectorLikeX> inline auto asum(mixed_engine const &e, VectorLikeX const &x, int incx = 1, int N = -1){ return asum(e, N, x, incx); }

template<class VectorLikeX> inline int iamax(mixed_engine const &e, VectorLikeX const &x, int incx = 1, int N = -1){
    auto gx = e.gpu().load(x);
    return iamax(e.gpu(), gx, incx, N);
}
template<class VectorLikeX> inline int iamax(mixed_engine const &e, int N, VectorLikeX const &x, int incx){ return iamax(e, x, incx, N); }

template<typename FS, class VectorLikeX, class VectorLikeY>
inline void axpy(mixed_engine const &e, FS alpha, VectorLikeX const &x, VectorLikeY &&y, int incx = 1, int incy = 1, int N = -1){
    auto gx = e.gpu().load(x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    axpy(e.gpu(), alpha, gx, gy, incx, incy, N);
}
template<typename FS, class VectorLikeX, class VectorLikeY>
inline void axpy(mixed_engine const &e, int N, FS alpha, VectorLikeX const &x, int incx, VectorLikeY &&y, int incy){
    axpy(e, alpha, x, y, incx, incy, N);
}

template<bool conjugate = true, class VectorLikeX, class VectorLikeY>
inline auto dot(mixed_engine const &e, int N, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy){
    auto gx = e.gpu().load(x);
    auto gy = e.gpu().load(y);
    return dot<conjugate>(e.gpu(), gx, gy, incx, incy, N);
}
template<bool conjugate = true, class VectorLikeX, class VectorLikeY>
inline auto dot(mixed_engine const &e, VectorLikeX const &x, VectorLikeY const &y, int incx = 1, int incy = 1, int N = -1){
    return dot<conjugate>(e, N, x, incx, y, incy);
}

template<class VectorLikeX, class VectorLikeY>
inline auto dotu(mixed_engine const &e, int N, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy){
    auto gx = e.gpu().load(x);
    auto gy = e.gpu().load(y);
    return dot<false>(e.gpu(), gx, gy, incx, incy, N);
}
template<class VectorLikeX, class VectorLikeY>
inline auto dotu(mixed_engine const &e, VectorLikeX const &x, VectorLikeY const &y, int incx = 1, int incy = 1, int N = -1){
    return dot<false>(e, N, x, incx, y, incy);
}

template<typename FS, class VectorLikeX>
inline void scal(mixed_engine const &e, FS alpha, VectorLikeX &&x, int incx = 1, int N = -1){
    auto gx = gpu_bind_vector(e.gpu(), x);
    scal(e.gpu(), alpha, gx, incx, N);
}
template<typename FS, class VectorLikeX>
inline void scal(mixed_engine const &e, int N, FS alpha, VectorLikeX &&x, int incx){ scal(e, alpha, x, incx, N); }

template<typename T> void rotg(mixed_engine const&, T &SA, T &SB, typename define_standard_precision<T>::value_type &C, T &S){ rotg(SA, SB, C, S); }

template<typename FC, typename FS, class VectorLikeX, class VectorLikeY>
void rot(mixed_engine const &e, VectorLikeX &x, VectorLikeY &&y, FC C, FS S, int incx = 1, int incy = 1, int N = -1){
    auto gx = gpu_bind_vector(e.gpu(), x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    rot(e.gpu(), gx, gy, C, S, incx, incy, N);
}
template<typename FC, typename FS, class VectorLikeX, class VectorLikeY>
void rot(mixed_engine const &e, int N, VectorLikeX &x, int incx, VectorLikeY &&y, int incy, FC C, FS S){
    rot(e, x, y, C, S, incx, incy, N);
}

template<typename T, class VectorLike>
void rotmg(mixed_engine const &e, T &D1, T &D2, T &X, T const &Y, VectorLike &&param){ rotmg(e.gpu(), D1, D2, X, Y, param); } // using host vector param

template<class VectorLikeX, class VectorLikeY, class VectorLikeP>
void rotm(mixed_engine const &e, VectorLikeX &&x, VectorLikeY &&y, VectorLikeP const &param, int incx = 1, int incy = 1, int N = -1){
    auto gx = gpu_bind_vector(e.gpu(), x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    rotm(e.gpu(), gx, gy, param, incx, incy, N);
}
template<class VectorLikeX, class VectorLikeY, class VectorLikeP>
void rotm(mixed_engine const &e, int N, VectorLikeX &&x, int incx, VectorLikeY &&y, int incy, VectorLikeP const &param){
    rotm(e, x, y, param, incx, incy, N);
}

/***********************************************************************************************************************/
/* BLAS level 2                                                                                                        */
/***********************************************************************************************************************/
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gemv(mixed_engine const &e, char trans, int M, int N, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    auto gA = e.gpu().load(A);
    auto gx = e.gpu().load(x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    gemv(e.gpu(), trans, M, N, alpha, gA, gx, beta, gy, lda, incx, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gemv(mixed_engine const &e, char trans, int M, int N, FSa alpha, const VectorLikeA &A, const VectorLikeX &x,
                 FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    gemv(e, trans, M, N, alpha, A, lda, x, incx, beta, y, incy);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gbmv(mixed_engine const &e, char trans, int M, int N, int kl, int ku, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    auto gA = e.gpu().load(A);
    auto gx = e.gpu().load(x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    gbmv(e.gpu(), trans, M, N, kl, ku, alpha, gA, gx, beta, gy, lda, incx, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gbmv(mixed_engine const &e, char trans, int M, int N, int kl, int ku, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    gbmv(e, trans, M, N, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void symv(mixed_engine const &e, char uplo, int N, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    auto gA = e.gpu().load(A);
    auto gx = e.gpu().load(x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    symv<conjugate>(e.gpu(), uplo, N, alpha, gA, gx, beta, gy, lda, incx, incy);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void symv(mixed_engine const &e, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    symv<conjugate>(e, uplo, N, alpha, A, lda, x, incx, beta, y, incy);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hemv(mixed_engine const &e, char uplo, int N, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    auto gA = e.gpu().load(A);
    auto gx = e.gpu().load(x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    hemv(e.gpu(), uplo, N, alpha, gA, gx, beta, gy, lda, incx, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hemv(mixed_engine const &e, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    hemv(e, uplo, N, alpha, A, lda, x, incx, beta, y, incy);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void sbmv(mixed_engine const &e, char uplo, int N, int k, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    auto gA = e.gpu().load(A);
    auto gx = e.gpu().load(x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    sbmv<conjugate>(e.gpu(), uplo, N, k, alpha, gA, gx, beta, gy, lda, incx, incy);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void sbmv(mixed_engine const &e, char uplo, int N, int k, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    sbmv<conjugate>(e, uplo, N, k, alpha, A, lda, x, incx, beta, y, incy);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hbmv(mixed_engine const &e, char uplo, int N, int k, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    auto gA = e.gpu().load(A);
    auto gx = e.gpu().load(x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    hbmv(e.gpu(), uplo, N, k, alpha, gA, gx, beta, gy, lda, incx, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hbmv(mixed_engine const &e, char uplo, int N, int k, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    hbmv(e, uplo, N, k, alpha, A, lda, x, incx, beta, y, incy);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void spmv(mixed_engine const &e, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    auto gA = e.gpu().load(A);
    auto gx = e.gpu().load(x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    spmv<conjugate>(e.gpu(), uplo, N, alpha, gA, gx, beta, gy, incx, incy);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void spmv(mixed_engine const &e, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int incx = 1, int incy = 1){
    spmv<conjugate>(e, uplo, N, alpha, A, x, incx, beta, y, incy);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hpmv(mixed_engine const &e, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    spmv<true>(e, uplo, N, alpha, A, x, beta, y, incx, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hpmv(mixed_engine const &e, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int incx = 1, int incy = 1){
    hpmv(e, uplo, N, alpha, A, x, incx, beta, y, incy);
}

template<class VectorLikeA, class VectorLikeX>
inline void trmv(mixed_engine const &e, char uplo, char trans, char diag, int N, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    auto gA = e.gpu().load(A);
    auto gx = gpu_bind_vector(e.gpu(), x);
    trmv(e.gpu(), uplo, trans, diag, N, gA, gx, lda, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void trmv(mixed_engine const &e, char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    trmv(e, uplo, trans, diag, N, A, lda, x, incx);
}

template<class VectorLikeA, class VectorLikeX>
inline void trsv(mixed_engine const &e, char uplo, char trans, char diag, int N, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    auto gA = e.gpu().load(A);
    auto gx = gpu_bind_vector(e.gpu(), x);
    trsv(e.gpu(), uplo, trans, diag, N, gA, gx, lda, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void trsv(mixed_engine const &e, char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    trsv(e, uplo, trans, diag, N, A, lda, x, incx);
}

template<class VectorLikeA, class VectorLikeX>
inline void tbmv(mixed_engine const &e, char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    auto gA = e.gpu().load(A);
    auto gx = gpu_bind_vector(e.gpu(), x);
    tbmv(e.gpu(), uplo, trans, diag, N, k, gA, gx, lda, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void tbmv(mixed_engine const &e, char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    tbmv(e, uplo, trans, diag, N, k, A, lda, x, incx);
}

template<class VectorLikeA, class VectorLikeX>
inline void tbsv(mixed_engine const &e, char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    auto gA = e.gpu().load(A);
    auto gx = gpu_bind_vector(e.gpu(), x);
    tbsv(e.gpu(), uplo, trans, diag, N, k, gA, gx, lda, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void tbsv(mixed_engine const &e, char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    tbsv(e, uplo, trans, diag, N, k, A, lda, x, incx);
}

template<class VectorLikeA, class VectorLikeX>
inline void tpmv(mixed_engine const &e, char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int incx = 1){
    auto gA = e.gpu().load(A);
    auto gx = gpu_bind_vector(e.gpu(), x);
    tpmv(e.gpu(), uplo, trans, diag, N, gA, gx, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void tpsv(mixed_engine const &e, char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int incx = 1){
    auto gA = e.gpu().load(A);
    auto gx = gpu_bind_vector(e.gpu(), x);
    tpsv(e.gpu(), uplo, trans, diag, N, gA, gx, incx);
}

template<bool conjugate = true, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void ger(mixed_engine const &e, int M, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    auto gx = e.gpu().load(x);
    auto gy = e.gpu().load(y);
    auto gA = gpu_bind_vector(e.gpu(), A);
    ger<conjugate>(e.gpu(), M, N, alpha, gx, gy, gA, incx, incy, lda);
}
template<bool conjugate = true, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void ger(mixed_engine const &e, int M, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    ger<conjugate>(e, M, N, alpha, x, incx, y, incy, A, lda);
}

template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void geru(mixed_engine const &e, int M, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    ger<false>(e, M, N, alpha, x, y, A, incx, incy, lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void geru(mixed_engine const &e, int M, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    ger<false>(e, M, N, alpha, x, incx, y, incy, A, lda);
}

template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void syr(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A, int lda){
    auto gx = e.gpu().load(x);
    auto gA = gpu_bind_vector(e.gpu(), A);
    syr<conjugate>(e.gpu(), uplo, N, alpha, gx, gA, incx, lda);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void syr(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1, int lda = -1){
    syr<conjugate>(e, uplo, N, alpha, x, incx, A, lda);
}

template<typename FSa, class VectorLikeX, class VectorLikeA>
inline void her(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A, int lda){
    syr<true>(e, uplo, N, alpha, x, A, incx, lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeA>
inline void her(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1, int lda = -1){
    syr<true>(e, uplo, N, alpha, x, incx, A, lda);
}

template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void spr(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A){
    auto gx = e.gpu().load(x);
    auto gA = gpu_bind_vector(e.gpu(), A);
    spr<conjugate>(e.gpu(), uplo, N, alpha, gx, gA, incx);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void spr(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1){
    spr<conjugate>(e, uplo, N, alpha, x, incx, A);
}

template<typename FSa, class VectorLikeX, class VectorLikeA>
inline void hpr(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A){
    spr<true>(e, uplo, N, alpha, x, A, incx);
}
template<typename FSa, class VectorLikeX, class VectorLikeA>
inline void hpr(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1){
    spr<true>(e, uplo, N, alpha, x, incx, A);
}

template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void syr2(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    auto gx = e.gpu().load(x);
    auto gy = e.gpu().load(y);
    auto gA = gpu_bind_vector(e.gpu(), A);
    syr2<conjugate>(e.gpu(), uplo, N, alpha, gx, gy, gA, incx, incy, lda);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void syr2(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    syr2<conjugate>(e, uplo, N, alpha, x, incx, y, incy, A, lda);
}

template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void her2(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    syr2<true>(e, uplo, N, alpha, x, y, A, incx, incy, lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void her2(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    syr2<true>(e, uplo, N, alpha, x, incx, y, incy, A, lda);
}

template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void spr2(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A){
    auto gx = e.gpu().load(x);
    auto gy = e.gpu().load(y);
    auto gA = gpu_bind_vector(e.gpu(), A);
    spr2<conjugate>(e.gpu(), uplo, N, alpha, gx, gy, gA, incx, incy);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void spr2(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1){
    spr2<conjugate>(e, uplo, N, alpha, x, incx, y, incy, A);
}

template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void hpr2(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A){
    spr2<true>(e, uplo, N, alpha, x, y, std::forward<VectorLikeA>(A), incx, incy);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void hpr2(mixed_engine const &e, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1){
    spr2<true>(e, uplo, N, alpha, x, incx, y, incy, std::forward<VectorLikeA>(A));
}

/***********************************************************************************************************************/
/* BLAS level 3                                                                                                        */
/***********************************************************************************************************************/
template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trmm(mixed_engine const &e, char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, int lda, VectorLikeB &&B, int ldb){
    auto gA = e.gpu().load(A);
    auto gB = gpu_bind_vector(e.gpu(), B);
    trmm(e.gpu(), side, uplo, trans, diag, M, N, alpha, gA, gB, lda, ldb);
}
template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trmm(mixed_engine const &e, char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, VectorLikeB &&B, int lda = -1, int ldb = -1){
    trmm(e, side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb);
}

template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trsm(mixed_engine const &e, char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, int lda, VectorLikeB &&B, int ldb){
    auto gA = e.gpu().load(A);
    auto gB = gpu_bind_vector(e.gpu(), B);
    trsm(e.gpu(), side, uplo, trans, diag, M, N, alpha, gA, gB, lda, ldb);
}
template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trsm(mixed_engine const &e, char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, VectorLikeB &&B, int lda = -1, int ldb = -1){
    trsm(e, side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void syrk(mixed_engine const &e, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, int lda, FSb beta, VectorLikeC &&C, int ldc){
    auto gA = e.gpu().load(A);
    auto gC = gpu_bind_vector(e.gpu(), C);
    syrk<conjugate>(e.gpu(), uplo, trans, N, K, alpha, gA, beta, gC, lda, ldc);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void syrk(mixed_engine const &e, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, FSb beta, VectorLikeC &C, int lda = -1, int ldc = -1){
    syrk<conjugate>(e, uplo, trans, N, K, alpha, A, lda, beta, C, ldc);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void herk(mixed_engine const &e, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, int lda, FSb beta, VectorLikeC &&C, int ldc){
    syrk<true>(e, uplo, trans, N, K, alpha, A, beta, C, lda, ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void herk(mixed_engine const &e, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, FSb beta, VectorLikeC &C, int lda = -1, int ldc = -1){
    syrk<true>(e, uplo, trans, N, K, alpha, A, lda, beta, C, ldc);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void syr2k(mixed_engine const &e, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, int lda,
                  VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    auto gA = e.gpu().load(A);
    auto gB = e.gpu().load(B);
    auto gC = gpu_bind_vector(e.gpu(), C);
    syr2k<conjugate>(e.gpu(), uplo, trans, N, K, alpha, gA, gB, beta, gC, lda, ldb, ldc);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void syr2k(mixed_engine const &e, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A,
                  VectorLikeB const &B, FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    syr2k<conjugate>(e, uplo, trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void her2k(mixed_engine const &e, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, int lda,
                  VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    syr2k<true>(e, uplo, trans, N, K, alpha, A, B, beta, C, lda, ldb, ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void her2k(mixed_engine const &e, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A,
                  VectorLikeB const &B, FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    syr2k<true>(e, uplo, trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void symm(mixed_engine const &e, char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    auto gA = e.gpu().load(A);
    auto gB = e.gpu().load(B);
    auto gC = gpu_bind_vector(e.gpu(), C);
    symm<conjugate>(e.gpu(), side, uplo, M, N, alpha, gA, gB, beta, gC, lda, ldb, ldc);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void symm(mixed_engine const &e, char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, VectorLikeB const &B,
                 FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    symm<conjugate>(e, side, uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void hemm(mixed_engine const &e, char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    symm<true>(e, side, uplo, M, N, alpha, A, B, beta, C, lda, ldb, ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void hemm(mixed_engine const &e, char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, VectorLikeB const &B,
                 FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    symm<true>(e, side, uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void gemm(mixed_engine const &e, char transa, char transb, int M, int N, int K, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    auto gA = e.gpu().load(A);
    auto gB = e.gpu().load(B);
    auto gC = gpu_bind_vector(e.gpu(), C);
    gemm(e.gpu(), transa, transb, M, N, K, alpha, gA, gB, beta, gC, lda, ldb, ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void gemm(mixed_engine const &e, char transa, char transb, int M, int N, int K, FSa alpha, VectorLikeA const &A,
                 VectorLikeB const &B, FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    gemm(e, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

/***********************************************************************************************************************/
/* SPARSE                                                                                                              */
/***********************************************************************************************************************/
template<typename FPa, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, typename FPb, class VectorLikeY>
void sparse_gemv(mixed_engine const &e, char trans, int M, int N, FPa alpha, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                 VectorLikeX const &x, FPb beta, VectorLikeY &&y){
    auto gpntr = e.gpu().load(pntr);
    auto gindx = e.gpu().load(indx);
    auto gvals = e.gpu().load(vals);
    auto gx = e.gpu().load(x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    sparse_gemv(e.gpu(), trans, M, N, alpha, gpntr, gindx, gvals, gx, beta, gy);
}

template<typename FSA, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeB, typename FSB, class VectorLikeC>
inline void sparse_gemm(mixed_engine const &e, char transa, char transb, int M, int N, int K,
                        FSA alpha, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                        VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &&C, int ldc){
    auto gpntr = e.gpu().load(pntr);
    auto gindx = e.gpu().load(indx);
    auto gvals = e.gpu().load(vals);
    auto gB = e.gpu().load(B);
    auto gC = gpu_bind_vector(e.gpu(), C);
    sparse_gemm(e.gpu(), transa, transb, M, N, K, alpha, gpntr, gindx, gvals, gB, beta, gC, ldb, ldc);
}

template<typename FSA, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeB, typename FSB, class VectorLikeC>
inline void sparse_gemm(mixed_engine const &e, char transa, char transb, int M, int N, int K,
                        FSA alpha, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                        VectorLikeB const &B, FSB beta, VectorLikeC &&C, int ldb = -1, int ldc = -1){
    sparse_gemm(e, transa, transb, M, N, K, alpha, pntr, indx, vals, B, ldb, beta, C, ldc);
}

template<typename T, typename FPA, class VectorLikeB, class VectorLikeX>
void sparse_trsv(char trans, mixed_tiangular_matrix<T> const &tri, FPA alpha, VectorLikeB const &b, VectorLikeX &&x){
    auto gb = tri.gpu_tri().engine().load(b);
    auto gx = gpu_bind_vector(tri.gpu_tri().engine(), x);
    hala::sparse_trsv(trans, tri.gpu_tri(), alpha, gb, gx);
}

template<typename T, typename FPA, class VectorLikeB>
void sparse_trsm(char transa, char transb, int nrhs, mixed_tiangular_matrix<T> const &tri, FPA alpha, VectorLikeB &&B, int ldb = -1){
    auto gB = gpu_bind_vector(tri.gpu_tri().engine(), B);
    hala::sparse_trsm(transa, transb, nrhs, tri.gpu_tri(), alpha, gB, ldb);
}

/***********************************************************************************************************************/
/* LAPACK                                                                                                              */
/***********************************************************************************************************************/
template<class VectorLikeA> size_t potrf_size(mixed_engine const &e, char uplo, int N, VectorLikeA const &A, int lda = -1){
    auto gA = e.gpu().load(A);
    return potrf_size(e.gpu(), uplo, N, gA, lda);
}
template<class VectorLikeA>
inline void potrf(mixed_engine const &e, char uplo, int N, VectorLikeA &A, int lda = -1){
    valid::default_ld(N, lda);
    auto gA = gpu_bind_vector(e.gpu(), A);
    potrf(e.gpu(), uplo, N, gA, lda, new_vector(e.gpu(), A));
}
template<class VectorLikeA, class VectorLikeW>
inline void potrf(mixed_engine const &e, char uplo, int N, VectorLikeA &A, int lda, VectorLikeW &&W){
    auto gA = gpu_bind_vector(e.gpu(), A);
    potrf(e.gpu(), uplo, N, gA, lda, std::forward<VectorLikeW>(W));
}
template<class VectorLikeA, class VectorLikeB>
inline void potrs(mixed_engine const &e, char uplo, int N, int nrhs, VectorLikeA const &A, int lda, VectorLikeB &&B, int ldb){
    auto gA = e.gpu().load(A);
    auto gB = gpu_bind_vector(e.gpu(), B);
    potrs(e.gpu(), uplo, N, nrhs, gA, gB, lda, ldb);
}
template<class VectorLikeA, class VectorLikeB>
inline void potrs(mixed_engine const &e, char uplo, int N, int nrhs, VectorLikeA const &A, VectorLikeB &&B, int lda = -1, int ldb = -1){
    potrs(e, uplo, N, nrhs, A, lda, std::forward<VectorLikeB>(B), ldb);
}

template<class VectorLikeA> size_t getrf_size(mixed_engine const &e, int M, int N, VectorLikeA const &A, int lda = -1){
    auto gA = e.gpu().load(A);
    return getrf_size(e.gpu(), M, N, A, lda);
}

template<class VectorLikeA, class VectorLikeI, class VectorLikeW>
inline void getrf(mixed_engine const &e, int M, int N, VectorLikeA &&A, int lda, VectorLikeI &&ipiv, VectorLikeW &&W){
    auto gA = gpu_bind_vector(e.gpu(), A);
    auto gipiv = gpu_bind_vector(e.gpu(), ipiv);
    valid::default_ld(lda, M);
    getrf(e.gpu(), M, N, gA, gipiv, lda, std::forward<VectorLikeW>(W));
}
template<class VectorLikeA, class VectorLikeI>
inline void getrf(mixed_engine const &e, int M, int N, VectorLikeA &&A, VectorLikeI &&ipiv, int lda = -1){
    getrf(e, M, N, std::forward<VectorLikeA>(A), lda, std::forward<VectorLikeI>(ipiv), new_vector(e.gpu(), A));
}
template<class VectorLikeA, class VectorLikeI, class VectorLikeB>
inline void getrs(mixed_engine const &e, char uplo, int N, int nrhs, VectorLikeA const &A, int lda, VectorLikeI const &ipiv,
                  VectorLikeB &&B, int ldb){
    getrs(e, uplo, N, nrhs, A, ipiv, std::forward<VectorLikeB>(B), lda, ldb);
}
template<class VectorLikeA, class VectorLikeI, class VectorLikeB>
inline void getrs(mixed_engine const &e, char uplo, int N, int nrhs, VectorLikeA const &A, VectorLikeI const &ipiv,
                  VectorLikeB &&B, int lda = -1, int ldb = -1){
    auto gA = e.gpu().load(A);
    auto gipiv = e.gpu().load(ipiv);
    auto gB = gpu_bind_vector(e.gpu(), B);
    getrs(e.gpu(), uplo, N, nrhs, gA, gipiv, gB, lda, ldb);
}


}

#endif

#endif

#endif
