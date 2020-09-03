#ifndef __HALA_BLAS_ENGINE_WRAPPERS_HPP
#define __HALA_BLAS_ENGINE_WRAPPERS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_blas_3.hpp"

namespace hala{
/*!
 * \internal
 * \file hala_blas_overloads.hpp
 * \brief HALA BLAS overloads for the CPU engine.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALABLAS
 *
 * Contains the overloads for the BLAS methods that accept default parameters
 * and/or have the \b hala::cpu_engine as the first parameter in the signature.
 * \endinternal
*/

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)

/***********************************************************************************************************************/
/* BLAS level 1                                                                                                        */
/***********************************************************************************************************************/
template<class VectorLikeX, class VectorLikeY>
inline void vcopy(cpu_engine const&, int N, VectorLikeX const &x, int incx, VectorLikeY &&y, int incy){ vcopy(N, x, incx, y, incy); }

template<class VectorLikeX, class VectorLikeY>
inline void vcopy(VectorLikeX const &x, VectorLikeY &&y, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    vcopy(N, x, incx, y, incy);
}
template<class VectorLikeX, class VectorLikeY>
inline void vcopy(cpu_engine const&, VectorLikeX const &x, VectorLikeY &&y, int incx = 1, int incy = 1, int N = -1){
    vcopy(x, y, incx, incy, N);
}
template<class VectorLike>
auto vcopy(cpu_engine const &engine, VectorLike const &x){
    auto y = get_vector(engine, x);
    vcopy(x, y);
    return y;
}

template<class VectorLikeX, class VectorLikeY>
inline void vswap(cpu_engine const&, int N, VectorLikeX &&x, int incx, VectorLikeY &&y, int incy){ vswap(N, x, incx, y, incy); }

template<class VectorLikeX, class VectorLikeY>
inline void vswap(VectorLikeX &&x, VectorLikeY &&y, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    vswap(N, x, incx, y, incy);
}
template<class VectorLikeX, class VectorLikeY>
inline void vswap(cpu_engine const&, VectorLikeX &&x, VectorLikeY &&y, int incx = 1, int incy = 1, int N = -1){
    vswap(x, y, incx, incy, N);
}

template<class VectorLikeX> inline auto norm2(VectorLikeX const &x, int incx = 1, int N = -1){
    valid::default_size(x, incx, N);
    return norm2(N, x, incx);
}
template<class VectorLikeX> inline auto norm2(cpu_engine const&, VectorLikeX const &x, int incx = 1, int N = -1){ return norm2(x, incx, N); }
template<class VectorLikeX> inline auto norm2(cpu_engine const&, int N, VectorLikeX const &x, int incx){ return norm2(N, x, incx); }

template<class VectorLikeX> inline auto asum(VectorLikeX const &x, int incx = 1, int N = -1){
    valid::default_size(x, incx, N);
    return asum(N, x, incx);
}
template<class VectorLikeX> inline auto asum(cpu_engine const&, VectorLikeX const &x, int incx = 1, int N = -1){ return asum(x, incx, N); }
template<class VectorLikeX> inline auto asum(cpu_engine const&, int N, VectorLikeX const &x, int incx){ return asum(N, x, incx); }

template<class VectorLikeX> inline int iamax(VectorLikeX const &x, int incx = 1, int N = -1){
    valid::default_size(x, incx, N);
    return iamax(N, x, incx);
}
template<class VectorLikeX> inline int iamax(cpu_engine const&, VectorLikeX const &x, int incx = 1, int N = -1){ return iamax(x, incx, N); }
template<class VectorLikeX> inline int iamax(cpu_engine const&, int N, VectorLikeX const &x, int incx){ return iamax(N, x, incx); }

template<typename FS, class VectorLikeX, class VectorLikeY>
inline void axpy(FS alpha, VectorLikeX const &x, VectorLikeY &&y, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    axpy(N, alpha, x, incx, y, incy);
}
template<typename FS, class VectorLikeX, class VectorLikeY>
inline void axpy(cpu_engine const&, FS alpha, VectorLikeX const &x, VectorLikeY &&y, int incx = 1, int incy = 1, int N = -1){
    axpy(alpha, x, y, incx, incy, N);
}
template<typename FS, class VectorLikeX, class VectorLikeY>
inline void axpy(cpu_engine const&, int N, FS alpha, VectorLikeX const &x, int incx, VectorLikeY &&y, int incy){
    axpy(alpha, x, y, incx, incy, N);
}

template<bool conjugate = true, class VectorLikeX, class VectorLikeY>
inline auto dot(VectorLikeX const &x, VectorLikeY const &y, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    return dot<conjugate, VectorLikeX, VectorLikeY>(N, x, incx, y, incy);
}
template<bool conjugate = true, class VectorLikeX, class VectorLikeY>
inline auto dot(cpu_engine const&, int N, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy){
    return dot<conjugate, VectorLikeX, VectorLikeY>(x, y, incx, incy, N);
}
template<bool conjugate = true, class VectorLikeX, class VectorLikeY>
inline auto dot(cpu_engine const&, VectorLikeX const &x, VectorLikeY const &y, int incx = 1, int incy = 1, int N = -1){
    return dot<conjugate, VectorLikeX, VectorLikeY>(x, y, incx, incy, N);
}

template<class VectorLikeX, class VectorLikeY>
inline auto dotu(VectorLikeX const &x, VectorLikeY const &y, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    return dotu<VectorLikeX, VectorLikeY>(N, x, incx, y, incy);
}
template<class VectorLikeX, class VectorLikeY>
inline auto dotu(cpu_engine const&, int N, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy){
    return dotu<VectorLikeX, VectorLikeY>(x, y, incx, incy, N);
}
template<class VectorLikeX, class VectorLikeY>
inline auto dotu(cpu_engine const&, VectorLikeX const &x, VectorLikeY const &y, int incx = 1, int incy = 1, int N = -1){
    return dotu<VectorLikeX, VectorLikeY>(x, y, incx, incy, N);
}

template<typename FS, class VectorLikeX>
inline void scal(FS alpha, VectorLikeX &&x, int incx = 1, int N = -1){
    valid::default_size(x, incx, N);
    scal(N, alpha, x, incx);
}
template<typename FS, class VectorLikeX>
inline void scal(cpu_engine const&, FS alpha, VectorLikeX &&x, int incx = 1, int N = -1){ scal(alpha, x, incx, N); }
template<typename FS, class VectorLikeX>
inline void scal(cpu_engine const&, int N, FS alpha, VectorLikeX &&x, int incx){ scal(N, alpha, x, incx); }

template<typename T> inline void rotg(cpu_engine const&, T &SA, T &SB, typename define_standard_precision<T>::value_type &C, T &S){ rotg(SA, SB, C, S); }
template<typename T, class VectorLike> inline void rotmg(cpu_engine const&, T &D1, T &D2, T &X, T const &Y, VectorLike &param){ rotmg(D1, D2, X, Y, param); }

template<typename FC, typename FS, class VectorLikeX, class VectorLikeY>
inline void rot(VectorLikeX &&x, VectorLikeY &&y, FC C, FS S, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    rot(N, x, incx, y, incy, C, S);
}
template<typename FC, typename FS, class VectorLikeX, class VectorLikeY>
inline void rot(cpu_engine const&, int N, VectorLikeX &&x, int incx, VectorLikeY &&y, int incy, FC C, FS S){ rot(N, x, incx, y, incy, C, S); }
template<typename FC, typename FS, class VectorLikeX, class VectorLikeY>
inline void rot(cpu_engine const&, VectorLikeX &&x, VectorLikeY &&y, FC C, FS S, int incx = 1, int incy = 1, int N = -1){ rot(x, y, C, S, incx, incy, N); }

template<class VectorLikeX, class VectorLikeY, class VectorLikeP>
inline void rotm(VectorLikeX &&x, VectorLikeY &&y, VectorLikeP const &param, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    rotm(N, x, incx, y, incy, param);
}
template<class VectorLikeX, class VectorLikeY, class VectorLikeP>
inline void rotm(cpu_engine const&, int N, VectorLikeX &&x, int incx, VectorLikeY &&y, int incy, VectorLikeP const &param){ rotm(N, x, incx, y, incy, param); }
template<class VectorLikeX, class VectorLikeY, class VectorLikeP>
inline void rotm(cpu_engine const&, VectorLikeX &&x, VectorLikeY &&y, VectorLikeP const &param, int incx = 1, int incy = 1, int N = -1){ rotm(x, y, param, incx, incy, N); }

/***********************************************************************************************************************/
/* BLAS level 2                                                                                                        */
/***********************************************************************************************************************/
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gemv(char trans, int M, int N, FSa alpha, VectorLikeA const &A, VectorLikeX const &x,
                 FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    valid::default_ld(M, lda);
    gemv(trans, M, N, alpha, A, lda, x, incx, beta, y, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gemv(cpu_engine const&, char trans, int M, int N, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeX const &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    gemv(trans, M, N, alpha, A, lda, x, incx, beta, y, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gemv(cpu_engine const&, char trans, int M, int N, FSa alpha, VectorLikeA const &A, VectorLikeX const &x,
                 FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    gemv(trans, M, N, alpha, A, x, beta, y, lda, incx, incy);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gbmv(char trans, int M, int N, int kl, int ku, FSa alpha, VectorLikeA const &A,
                 VectorLikeX const &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    valid::default_ld(1 + kl + ku, lda);
    gbmv(trans, M, N, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gbmv(cpu_engine const&, char trans, int M, int N, int kl, int ku, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeX const &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    gbmv(trans, M, N, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gbmv(cpu_engine const&, char trans, int M, int N, int kl, int ku, FSa alpha, VectorLikeA const &A,
                 VectorLikeX const &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    gbmv(trans, M, N, kl, ku, alpha, A, x, beta, y, lda, incx, incy);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void symv(char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    valid::default_ld(N, lda);
    symv<conjugate>(uplo, N, alpha, A, lda, x, incx, beta, y, incy);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void symv(cpu_engine const&, char uplo, int N, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    symv<conjugate>(uplo, N, alpha, A, lda, x, incx, beta, y, incy);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void symv(cpu_engine const&, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &y, int lda = -1, int incx = 1, int incy = 1){
    symv<conjugate>(uplo, N, alpha, A, x, beta, y, lda, incx, incy);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hemv(char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    valid::default_ld(N, lda);
    hemv(uplo, N, alpha, A, lda, x, incx, beta, y, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hemv(cpu_engine const&, char uplo, int N, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    hemv(uplo, N, alpha, A, lda, x, incx, beta, y, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hemv(cpu_engine const&, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &y, int lda = -1, int incx = 1, int incy = 1){
    hemv(uplo, N, alpha, A, x, beta, y, lda, incx, incy);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void sbmv(char uplo, int N, int k, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    valid::default_ld(k + 1, lda);
    sbmv<conjugate>(uplo, N, k, alpha, A, lda, x, incx, beta, y, incy);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void sbmv(cpu_engine const&, char uplo, int N, int k, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    sbmv<conjugate>(uplo, N, k, alpha, A, lda, x, incx, beta, y, incy);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void sbmv(cpu_engine const&, char uplo, int N, int k, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    sbmv<conjugate>(uplo, N, k, alpha, A, x, beta, y, lda, incx, incy);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hbmv(char uplo, int N, int k, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    valid::default_ld(k + 1, lda);
    hbmv(uplo, N, k, alpha, A, lda, x, incx, beta, y, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hbmv(cpu_engine const&, char uplo, int N, int k, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    hbmv(uplo, N, k, alpha, A, lda, x, incx, beta, y, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hbmv(cpu_engine const&, char uplo, int N, int k, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    hbmv(uplo, N, k, alpha, A, x, beta, y, lda, incx, incy);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void spmv(char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int incx = 1, int incy = 1){
    spmv<conjugate>(uplo, N, alpha, A, x, incx, beta, y, incy);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void spmv(cpu_engine const&, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    spmv<conjugate>(uplo, N, alpha, A, x, incx, beta, y, incy);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void spmv(cpu_engine const&, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int incx = 1, int incy = 1){
    spmv<conjugate>(uplo, N, alpha, A, x, beta, y, incx, incy);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hpmv(char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int incx = 1, int incy = 1){
    hpmv(uplo, N, alpha, A, x, incx, beta, y, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hpmv(cpu_engine const&, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    hpmv(uplo, N, alpha, A, x, incx, beta, y, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hpmv(cpu_engine const&, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int incx = 1, int incy = 1){
    hpmv(uplo, N, alpha, A, x, beta, y, incx, incy);
}

template<class VectorLikeA, class VectorLikeX>
inline void trmv(char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    valid::default_ld(N, lda);
    trmv(uplo, trans, diag, N, A, lda, x, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void trmv(cpu_engine const&, char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    trmv(uplo, trans, diag, N, A, x, lda, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void trmv(cpu_engine const&, char uplo, char trans, char diag, int N, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    trmv(uplo, trans, diag, N, A, lda, x, incx);
}

template<class VectorLikeA, class VectorLikeX>
inline void trsv(char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    valid::default_ld(N, lda);
    trsv(uplo, trans, diag, N, A, lda, x, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void trsv(cpu_engine const&, char uplo, char trans, char diag, int N, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    trsv(uplo, trans, diag, N, A, lda, x, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void trsv(cpu_engine const&, char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    trsv(uplo, trans, diag, N, A, x, lda, incx);
}

template<class VectorLikeA, class VectorLikeX>
inline void tbmv(char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    valid::default_ld(k+1, lda);
    tbmv(uplo, trans, diag, N, k, A, lda, x, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void tbmv(cpu_engine const&, char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    tbmv(uplo, trans, diag, N, k, A, x, lda, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void tbmv(cpu_engine const&, char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    tbmv(uplo, trans, diag, N, k, A, lda, x, incx);
}

template<class VectorLikeA, class VectorLikeX>
inline void tbsv(char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    valid::default_ld(k+1, lda);
    tbsv(uplo, trans, diag, N, k, A, lda, x, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void tbsv(cpu_engine const&, char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    tbsv(uplo, trans, diag, N, k, A, lda, x, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void tbsv(cpu_engine const&, char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    tbsv(uplo, trans, diag, N, k, A, x, lda, incx);
}

template<class VectorLikeA, class VectorLikeX>
inline void tpmv(cpu_engine const&, char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int incx = 1){
    tpmv(uplo, trans, diag, N, A, x, incx);
}
template<class VectorLikeA, class VectorLikeX>
inline void tpsv(cpu_engine const&, char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int incx = 1){
    tpsv(uplo, trans, diag, N, A, x, incx);
}

template<bool conjugate = true, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void ger(int M, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    valid::default_ld(M, lda);
    ger<conjugate>(M, N, alpha, x, incx, y, incy, A, lda);
}
template<bool conjugate = true, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void ger(cpu_engine const&, int M, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    ger<conjugate>(M, N, alpha, x, incx, y, incy, A, lda);
}
template<bool conjugate = true, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void ger(cpu_engine const&, int M, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    ger<conjugate>(M, N, alpha, x, y, A, incx, incy, lda);
}

template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void geru(int M, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    valid::default_ld(M, lda);
    ger<false>(M, N, alpha, x, incx, y, incy, A, lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void geru(cpu_engine const&, int M, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    ger<false>(M, N, alpha, x, incx, y, incy, A, lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void geru(cpu_engine const&, int M, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    ger<false>(M, N, alpha, x, y, A, incx, incy, lda);
}

template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void syr(char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1, int lda = -1){
    valid::default_ld(N, lda);
    syr<conjugate>(uplo, N, alpha, x, incx, A, lda);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void syr(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A, int lda){
    syr<conjugate>(uplo, N, alpha, x, incx, A, lda);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void syr(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1, int lda = -1){
    syr<conjugate>(uplo, N, alpha, x, A, incx, lda);
}

template<typename FSa, class VectorLikeX, class VectorLikeA>
inline void her(char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1, int lda = -1){
    valid::default_ld(N, lda);
    syr<true>(uplo, N, alpha, x, incx, A, lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeA>
inline void her(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A, int lda){
    syr<true>(uplo, N, alpha, x, incx, A, lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeA>
inline void her(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1, int lda = -1){
    syr<true>(uplo, N, alpha, x, A, incx, lda);
}

template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void spr(char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1){
    spr<conjugate>(uplo, N, alpha, x, incx, A);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void spr(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A){
    spr<conjugate>(uplo, N, alpha, x, incx, A);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void spr(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1){
    spr<conjugate>(uplo, N, alpha, x, A, incx);
}

template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void hpr(char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1){
    spr<true>(uplo, N, alpha, x, incx, A);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void hpr(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A){
    spr<true>(uplo, N, alpha, x, incx, A);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void hpr(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1){
    spr<true>(uplo, N, alpha, x, A, incx);
}

template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void syr2(char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    valid::default_ld(N, lda);
    syr2<conjugate>(uplo, N, alpha, x, incx, y, incy, A, lda);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void syr2(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    syr2<conjugate>(uplo, N, alpha, x, incx, y, incy, A, lda);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void syr2(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    syr2<conjugate>(uplo, N, alpha, x, y, A, incx, incy, lda);
}

template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void her2(char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    valid::default_ld(N, lda);
    syr2<true>(uplo, N, alpha, x, incx, y, incy, A, lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void her2(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    syr2<true>(uplo, N, alpha, x, incx, y, incy, A, lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void her2(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    syr2<true>(uplo, N, alpha, x, y, A, incx, incy, lda);
}

template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void spr2(char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1){
    spr2<conjugate>(uplo, N, alpha, x, incx, y, incy, A);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void spr2(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A){
    spr2<conjugate>(uplo, N, alpha, x, incx, y, incy, A);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void spr2(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1){
    spr2<conjugate>(uplo, N, alpha, x, y, A, incx, incy);
}

template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void hpr2(char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1){
    spr2<true>(uplo, N, alpha, x, incx, y, incy, A);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void hpr2(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A){
    spr2<true>(uplo, N, alpha, x, incx, y, incy, A);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void hpr2(cpu_engine const&, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1){
    spr2<true>(uplo, N, alpha, x, y, A, incx, incy);
}

/***********************************************************************************************************************/
/* BLAS level 3                                                                                                        */
/***********************************************************************************************************************/
template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trmm(char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, VectorLikeB &&B, int lda = -1, int ldb = -1){
    valid::default_ld(is_l(side), M, N, lda);
    valid::default_ld(M, ldb);
    trmm(side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb);
}
template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trmm(cpu_engine const&, char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, int lda, VectorLikeB &&B, int ldb){
    trmm(side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb);
}
template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trmm(cpu_engine const&, char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, VectorLikeB &&B, int lda = -1, int ldb = -1){
    trmm(side, uplo, trans, diag, M, N, alpha, A, B, lda, ldb);
}

template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trsm(char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, VectorLikeB &&B, int lda = -1, int ldb = -1){
    valid::default_ld(M, ldb);
    valid::default_ld(is_l(side), M, N, lda);
    trsm(side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb);
}
template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trsm(cpu_engine const&, char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, int lda, VectorLikeB &&B, int ldb){
    trsm(side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb);
}
template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trsm(cpu_engine const&, char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, VectorLikeB &&B, int lda = -1, int ldb = -1){
    trsm(side, uplo, trans, diag, M, N, alpha, A, B, lda, ldb);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void syrk(char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, FSb beta, VectorLikeC &&C, int lda = -1, int ldc = -1){
    valid::default_ld(is_n(trans), N, K, lda);
    valid::default_ld(N, ldc);
    syrk<conjugate>(uplo, trans, N, K, alpha, A, lda, beta, C, ldc);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void syrk(cpu_engine const&, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, int lda, FSb beta, VectorLikeC &&C, int ldc){
    syrk<conjugate>(uplo, trans, N, K, alpha, A, lda, beta, C, ldc);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void syrk(cpu_engine const&, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, FSb beta, VectorLikeC &&C, int lda = -1, int ldc = -1){
    syrk<conjugate>(uplo, trans, N, K, alpha, A, beta, C, lda, ldc);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void herk(char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, int lda, FSb beta, VectorLikeC &&C, int ldc){
    syrk<true>(uplo, trans, N, K, alpha, A, lda, beta, std::forward<VectorLikeC>(C), ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void herk(char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, FSb beta, VectorLikeC &&C, int lda = -1, int ldc = -1){
    valid::default_ld(is_n(trans), N, K, lda);
    valid::default_ld(N, ldc);
    syrk<true>(uplo, trans, N, K, alpha, A, lda, beta, std::forward<VectorLikeC>(C), ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void herk(cpu_engine const&, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, int lda, FSb beta, VectorLikeC &&C, int ldc){
    syrk<true>(uplo, trans, N, K, alpha, A, lda, beta, std::forward<VectorLikeC>(C), ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void herk(cpu_engine const&, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, FSb beta, VectorLikeC &&C, int lda = -1, int ldc = -1){
    syrk<true>(uplo, trans, N, K, alpha, A, beta, std::forward<VectorLikeC>(C), lda, ldc);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void syr2k(char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A,
                  VectorLikeB const &B, FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    valid::default_ld(is_n(trans), N, K, lda);
    valid::default_ld(is_n(trans), N, K, ldb);
    valid::default_ld(N, ldc);
    syr2k<conjugate>(uplo, trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void syr2k(cpu_engine const&, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, int lda,
                  VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    syr2k<conjugate>(uplo, trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void syr2k(cpu_engine const&, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A,
                  VectorLikeB const &B, FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    syr2k<conjugate>(uplo, trans, N, K, alpha, A, B, beta, C, lda, ldb, ldc);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void her2k(char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, int lda,
                  VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    syr2k<true>(uplo, trans, N, K, alpha, A, lda, B, ldb, beta, std::forward<VectorLikeC>(C), ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void her2k(char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A,
                  VectorLikeB const &B, FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    valid::default_ld(is_n(trans), N, K, lda);
    valid::default_ld(is_n(trans), N, K, ldb);
    valid::default_ld(N, ldc);
    syr2k<true>(uplo, trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void her2k(cpu_engine const&, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A, int lda,
                  VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    syr2k<true>(uplo, trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void her2k(cpu_engine const&, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A,
                  VectorLikeB const &B, FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    syr2k<true>(uplo, trans, N, K, alpha, A, B, beta, C, lda, ldb, ldc);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void symm(char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, VectorLikeB const &B,
                 FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    valid::default_ld(is_l(side), M, N, lda);
    valid::default_ld(M, ldb);
    valid::default_ld(M, ldc);
    symm<conjugate>(side, uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void symm(cpu_engine const&, char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    symm<conjugate>(side, uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
}
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void symm(cpu_engine const&, char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, VectorLikeB const &B,
                 FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    symm<conjugate>(side, uplo, M, N, alpha, A, B, beta, C, lda, ldb, ldc);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void hemm(char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    symm<true>(side, uplo, M, N, alpha, A, lda, B, ldb, beta, std::forward<VectorLikeC>(C), ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void hemm(char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, VectorLikeB const &B,
                 FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    valid::default_ld(is_l(side), M, N, lda);
    valid::default_ld(M, ldb);
    valid::default_ld(M, ldc);
    symm<true>(side, uplo, M, N, alpha, A, lda, B, ldb, beta, std::forward<VectorLikeC>(C), ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void hemm(cpu_engine const&, char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    symm<true>(side, uplo, M, N, alpha, A, lda, B, ldb, beta, std::forward<VectorLikeC>(C), ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void hemm(cpu_engine const&, char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, VectorLikeB const &B,
                 FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    symm<true>(side, uplo, M, N, alpha, A, B, beta, std::forward<VectorLikeC>(C), lda, ldb, ldc);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void gemm(char transa, char transb, int M, int N, int K, FSa alpha, VectorLikeA const &A,
                 VectorLikeB const &B, FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    valid::default_ld(is_n(transa), M, K, lda);
    valid::default_ld(is_n(transb), K, N, ldb);
    valid::default_ld(M, ldc);
    gemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void gemm(cpu_engine const&, char transa, char transb, int M, int N, int K, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    gemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void gemm(cpu_engine const&, char transa, char transb, int M, int N, int K, FSa alpha, VectorLikeA const &A,
                 VectorLikeB const &B, FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    gemm(transa, transb, M, N, K, alpha, A, B, beta, C, lda, ldb, ldc);
}

#endif

}

#endif
