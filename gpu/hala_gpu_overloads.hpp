#ifndef __HALA_GPU_OVERLOADS_HPP
#define __HALA_GPU_OVERLOADS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_gpu_plu.hpp"

/*!
 * \internal
 * \file hala_gpu_overloads.hpp
 * \ingroup HALACORE
 * \brief Contains the overloads for the CUDA methods.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 *
 * All overloads involving the cuBlas, cuSparse, rocblas and rocsparse with gpu_engine.
 * \endinternal
 */

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)

namespace hala{

template<typename FPa, class VectorLikeA, typename FPb, class VectorLikeB, class VectorLikeC>
inline void geam(gpu_engine const &engine, char transa, char transb, int M, int N, FPa alpha, VectorLikeA const &A,
                 FPb beta, VectorLikeB const &B, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    valid::default_ld(is_n(transa), M, N, lda);
    valid::default_ld(is_n(transb), M, N, ldb);
    valid::default_ld(M, ldc);
    geam(engine, transa, transb, M, N, alpha, A, lda, beta, B, ldb, C, ldc);
}

template<class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void dgmm(gpu_engine const &engine, char side, int M, int N, VectorLikeA const &A,
                 VectorLikeB const &x, VectorLikeC &&C, int lda = -1, int incx = 1, int ldc = -1){
    valid::default_ld(M, lda);
    valid::default_ld(M, ldc);
    dgmm(engine, side, M, N, A, lda, x, incx, C, ldc);
}

template<class VectorLikeA, class VectorLikeAP>
void tr2tp(gpu_engine const &engine, char uplo, int N, VectorLikeA const &A, VectorLikeAP &&AP, int lda = -1){
    valid::default_ld(N, lda);
    tr2tp(engine, uplo, N, A, lda, AP);
}

template<class vec>
auto gpu_engine::vcopy(vec const &x) const{
    auto y = new_vector(*this, x);
    auto xdevice = get_device(x);
    if ((xdevice > -1) && (xdevice != cgpu)){
        using standard_type = get_standard_type<vec>;
        std::vector<standard_type> cpuy(get_size(x));
        gpu_copy_n<copy_direction::device2host>(get_data(x), get_size(x), get_data(cpuy));
        y.load(cpuy);
    }else{
        hala::vcopy(*this, x, y);
    }
    return y;
}

template<class VectorLikeX, class VectorLikeY>
inline void vswap(gpu_engine const &engine, VectorLikeX &&x, VectorLikeY &&y, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    vswap(engine, N, x, incx, y, incy);
}

template<class VectorLikeX> inline auto norm2(gpu_engine const &engine, VectorLikeX const &x, int incx = 1, int N = -1){
    valid::default_size(x, incx, N);
    return norm2(engine, N, x, incx);
}

template<class VectorLikeX> inline auto asum(gpu_engine const &engine, VectorLikeX const &x, int incx = 1, int N = -1){
    valid::default_size(x, incx, N);
    return asum(engine, N, x, incx);
}

template<class VectorLikeX> inline int iamax(gpu_engine const &engine, VectorLikeX const &x, int incx = 1, int N = -1){
    valid::default_size(x, incx, N);
    return iamax(engine, N, x, incx);
}

template<bool conjugate = true, class VectorLikeX, class VectorLikeY>
auto dot(gpu_engine const &engine, VectorLikeX const &x, VectorLikeY const &y, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    return dot<conjugate>(engine, N, x, incx, y, incy);
}

template<class VectorLikeX, class VectorLikeY>
auto dotu(gpu_engine const &engine, VectorLikeX const &x, VectorLikeY const &y, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    return dot<false>(engine, N, x, incx, y, incy);
}
template<class VectorLikeX, class VectorLikeY>
auto dotu(gpu_engine const &engine, int N, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy){
    return dot<false>(engine, N, x, incx, y, incy);
}

template<typename FP, class VectorLikeX, class VectorLikeY>
void axpy(gpu_engine const &engine, FP alpha, VectorLikeX const &x, VectorLikeY &&y, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    axpy(engine, N, alpha, x, incx, y, incy);
}

template<typename FP, class VectorLike>
void scal(gpu_engine const &engine, FP alpha, VectorLike &&x, int incx = 1, int N = -1){
    valid::default_size(x, incx, N);
    scal(engine, N, alpha, x, incx);
}

template<typename FC, typename FS, class VectorLikeX, class VectorLikeY>
void rot(gpu_engine const &engine, VectorLikeX &x, VectorLikeY &&y, FC C, FS S, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    rot(engine, N, x, incx, y, incy, C, S);
}

template<class VectorLikeX, class VectorLikeY, class VectorLikeP>
void rotm(gpu_engine const &engine, VectorLikeX &x, VectorLikeY &&y, VectorLikeP const &param, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    rotm(engine, N, x, incx, y, incy, param);
}

template<typename FPA, typename FPB, class VectorLikeA, class VectorLikeX, class VectorLikeY>
void gemv(gpu_engine const &engine, char trans, int M, int N,
          FPA alpha, VectorLikeA const &A, VectorLikeX const &x, FPB beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    valid::default_ld(M, lda);
    gemv(engine, trans, M, N, alpha, A, lda, x, incx, beta, y, incy);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gbmv(gpu_engine const &engine, char trans, int M, int N, int kl, int ku, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    valid::default_ld(1 + kl + ku, lda);
    gbmv(engine, trans, M, N, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void symv(gpu_engine const &engine, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    valid::default_ld(N, lda);
    symv<conjugate>(engine, uplo, N, alpha, A, lda, x, incx, beta, y, incy);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hemv(gpu_engine const &engine, char uplo, int N, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    symv<true>(engine, uplo, N, alpha, A, lda, x, incx, beta, y, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hemv(gpu_engine const &engine, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    valid::default_ld(N, lda);
    hemv(engine, uplo, N, alpha, A, lda, x, incx, beta, y, incy);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void sbmv(gpu_engine const &engine, char uplo, int N, int k, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    valid::default_ld(k + 1, lda);
    sbmv<conjugate>(engine, uplo, N, k, alpha, A, lda, x, incx, beta, y, incy);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hbmv(gpu_engine const &engine, char uplo, int N, int k, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    valid::default_ld(k + 1, lda);
    sbmv<true>(engine, uplo, N, k, alpha, A, lda, x, incx, beta, y, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hbmv(gpu_engine const &engine, char uplo, int N, int k, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int lda = -1, int incx = 1, int incy = 1){
    valid::default_ld(k + 1, lda);
    hbmv(engine, uplo, N, k, alpha, A, lda, x, incx, beta, y, incy);
}


template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void spmv(gpu_engine const &engine, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int incx = 1, int incy = 1){
    spmv<conjugate>(engine, uplo, N, alpha, A, x, incx, beta, y, incy);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hpmv(gpu_engine const &engine, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    spmv<true>(engine, uplo, N, alpha, A, x, beta, y, incx, incy);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hpmv(gpu_engine const &engine, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, FSb beta, VectorLikeY &&y, int incx = 1, int incy = 1){
    spmv<true>(engine, uplo, N, alpha, A, x, beta, y, incx, incy);
}

template<class VectorLikeA, class VectorLikeX>
inline void trmv(gpu_engine const &engine, char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    valid::default_ld(N, lda);
    trmv(engine, uplo, trans, diag, N, A, lda, x, incx);
}

template<class VectorLikeA, class VectorLikeX>
inline void trsv(gpu_engine const &engine, char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    valid::default_ld(N, lda);
    trsv(engine, uplo, trans, diag, N, A, lda, x, incx);
}

template<class VectorLikeA, class VectorLikeX>
inline void tbmv(gpu_engine const &engine, char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    valid::default_ld(k+1, lda);
    tbmv(engine, uplo, trans, diag, N, k, A, lda, x, incx);
}

template<class VectorLikeA, class VectorLikeX>
inline void tbsv(gpu_engine const &engine, char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, VectorLikeX &&x, int lda = -1, int incx = 1){
    valid::default_ld(k+1, lda);
    tbsv(engine, uplo, trans, diag, N, k, A, lda, x, incx);
}

template<bool conjugate = true, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void ger(gpu_engine const &engine, int M, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    valid::default_ld(M, lda);
    ger<conjugate>(engine, M, N, alpha, x, incx, y, incy, A, lda);
}

template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void geru(gpu_engine const &engine, int M, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    valid::default_ld(M, lda);
    ger<false>(engine, M, N, alpha, x, incx, y, incy, A, lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void geru(gpu_engine const &engine, int M, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    ger<false>(engine, M, N, alpha, x, incx, y, incy, A, lda);
}

template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void syr(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1, int lda = -1){
    valid::default_ld(N, lda);
    syr<conjugate>(engine, uplo, N, alpha, x, incx, A, lda);
}

template<typename FSa, class VectorLikeX, class VectorLikeA>
inline void her(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A, int lda){
    syr<true>(engine, uplo, N, alpha, x, incx, A, lda);
}
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void her(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1, int lda = -1){
    valid::default_ld(N, lda);
    syr<true>(engine, uplo, N, alpha, x, incx, A, lda);
}

template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void spr(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeA &&A, int incx = 1){
    spr<conjugate>(engine, uplo, N, alpha, x, incx, A);
}

template<typename FSa, class VectorLikeX, class VectorLikeA>
inline void hpr(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A){
    spr<true>(engine, uplo, N, alpha, x, incx, A);
}

template<bool conjugate = true, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void syr2(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    valid::default_ld(N, lda);
    syr2<conjugate>(engine, uplo, N, alpha, x, incx, y, incy, A, lda);
}

template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void her2(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    syr2<true>(engine, uplo, N, alpha, x, incx, y, incy, std::forward<VectorLikeA>(A), lda);
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void her2(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1, int lda = -1){
    valid::default_ld(N, lda);
    syr2<true>(engine, uplo, N, alpha, x, incx, y, incy, std::forward<VectorLikeA>(A), lda);
}

template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void spr2(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1){
    spr2<conjugate>(engine, uplo, N, alpha, x, incx, y, incy, A);
}

template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void hpr2(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A){
    spr2<true>(engine, uplo, N, alpha, x, incx, y, incy, std::forward<VectorLikeA>(A));
}
template<typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void hpr2(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, VectorLikeY const &y, VectorLikeA &&A, int incx = 1, int incy = 1){
    spr2<true>(engine, uplo, N, alpha, x, incx, y, incy, std::forward<VectorLikeA>(A));
}

template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trmm(gpu_engine const &engine, char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, VectorLikeB &&B, int lda = -1, int ldb = -1){
    valid::default_ld(is_l(side), M, N, lda);
    valid::default_ld(M, ldb);
    trmm(engine, side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb);
}

template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trsm(gpu_engine const &engine, char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, VectorLikeB &&B, int lda = -1, int ldb = -1){
    valid::default_ld(M, ldb);
    valid::default_ld(is_l(side), M, N, lda);
    trsm(engine, side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb);
}

template<bool conjugate = false, typename FPA, typename FPB, class VectorLikeA, class VectorLikeC>
inline void syrk(gpu_engine const &engine, char uplo, char trans, int N, int K, FPA alpha, VectorLikeA const &A,
                 FPB beta, VectorLikeC &&C, int lda = -1, int ldc = -1){
    valid::default_ld(is_n(trans), N, K, lda);
    valid::default_ld(N, ldc);
    syrk<conjugate>(engine, uplo, trans, N, K, alpha, A, lda, beta, C, ldc);
}

template<typename FPA, typename FPB, class VectorLikeA, class VectorLikeC>
inline void herk(gpu_engine const &engine, char uplo, char trans, int N, int K, FPA alpha, VectorLikeA const &A, int lda,
                 FPB beta, VectorLikeC &&C, int ldc){
    syrk<true>(engine, uplo, trans, N, K, alpha, A, lda, beta, std::forward<VectorLikeC>(C), ldc);
}
template<typename FPA, typename FPB, class VectorLikeA, class VectorLikeC>
inline void herk(gpu_engine const &engine, char uplo, char trans, int N, int K, FPA alpha, VectorLikeA const &A,
                 FPB beta, VectorLikeC &&C, int lda = -1, int ldc = -1){
    valid::default_ld(is_n(trans), N, K, lda);
    valid::default_ld(N, ldc);
    syrk<true>(engine, uplo, trans, N, K, alpha, A, lda, beta, std::forward<VectorLikeC>(C), ldc);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void syr2k(gpu_engine const &engine, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A,
                  VectorLikeB const &B, FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    valid::default_ld(is_n(trans), N, K, lda);
    valid::default_ld(is_n(trans), N, K, ldb);
    valid::default_ld(N, ldc);
    syr2k<conjugate>(engine, uplo, trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<typename FPA, typename FPB, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void her2k(gpu_engine const &engine, char uplo, char trans, int N, int K, FPA alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FPB beta, VectorLikeC &&C, int ldc){
    syr2k<true>(engine, uplo, trans, N, K, alpha, A, lda, B, ldb, beta, std::forward<VectorLikeC>(C), ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void her2k(gpu_engine const &engine, char uplo, char trans, int N, int K, FSa alpha, VectorLikeA const &A,
                  VectorLikeB const &B, FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    valid::default_ld(is_n(trans), N, K, lda);
    valid::default_ld(is_n(trans), N, K, ldb);
    valid::default_ld(N, ldc);
    syr2k<true>(engine, uplo, trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void symm(gpu_engine const &engine, char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, VectorLikeB const &B,
                 FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    valid::default_ld(is_l(side), M, N, lda);
    valid::default_ld(M, ldb);
    valid::default_ld(M, ldc);
    symm<conjugate>(engine, side, uplo, M, N, alpha, A, lda, B, ldb, beta, std::forward<VectorLikeC>(C), ldc);
}

template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void symm(gpu_engine const &engine, char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    symm<true>(engine, side, uplo, M, N, alpha, A, lda, B, ldb, beta, std::forward<VectorLikeC>(C), ldc);
}
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void symm(gpu_engine const &engine, char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, VectorLikeB const &B,
                 FSb beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    valid::default_ld(is_l(side), M, N, lda);
    valid::default_ld(M, ldb);
    valid::default_ld(M, ldc);
    symm<true>(engine, side, uplo, M, N, alpha, A, lda, B, ldb, beta, std::forward<VectorLikeC>(C), ldc);
}

template<typename FPA, typename FPB, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void gemm(gpu_engine const &engine, char transa, char transb, int M, int N, int K, FPA alpha, VectorLikeA const &A, VectorLikeB const &B,
                 FPB beta, VectorLikeC &&C, int lda = -1, int ldb = -1, int ldc = -1){
    valid::default_ld(is_n(transa), M, K, lda);
    valid::default_ld(is_n(transb), K, N, ldb);
    valid::default_ld(M, ldc);
    gemm(engine, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, std::forward<VectorLikeC>(C), ldc);
}

/***********************************************************************************************************************/
/* SPARSE                                                                                                              */
/***********************************************************************************************************************/
template<typename FSA, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeB, typename FSB, class VectorLikeC>
inline void sparse_gemm(gpu_engine const &engine, char transa, char transb, int M, int N, int K,
                        FSA alpha, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                        VectorLikeB const &B, FSB beta, VectorLikeC &&C, int ldb = -1, int ldc = -1){
    valid::default_ld(is_n(transb), K, N, ldb);
    valid::default_ld(M, ldc);
    sparse_gemm(engine, transa, transb, M, N, K, alpha, pntr, indx, vals, B, ldb, beta, C, ldc);
}

#ifdef HALA_ENABLE_CUDA
/***********************************************************************************************************************/
/* SOLVER                                                                                                              */
/***********************************************************************************************************************/
template<class VectorLikeA>
inline void potrf(gpu_engine const &engine, char uplo, int N, VectorLikeA &A, int lda = -1){
    valid::default_ld(N, lda);
    potrf(engine, uplo, N, std::forward<VectorLikeA>(A), lda, new_vector(engine, A));
}
template<class VectorLikeA, class VectorLikeW>
inline void potrf(gpu_engine const &engine, char uplo, int N, VectorLikeA &A, int lda, VectorLikeW &&W){
    potrf(engine, uplo, N, std::forward<VectorLikeA>(A), lda, std::forward<VectorLikeW>(W));
}

template<class VectorLikeA, class VectorLikeB>
inline void potrs(gpu_engine const &engine, char uplo, int N, int nrhs, VectorLikeA const &A, VectorLikeB &&B, int lda = -1, int ldb = -1){
    valid::default_ld(N, lda);
    valid::default_ld(N, ldb);
    potrs(engine, uplo, N, nrhs, A, lda, std::forward<VectorLikeB>(B), ldb);
}

template<class VectorLikeA, class VectorLikeI>
inline void getrf(gpu_engine const &engine, int M, int N, VectorLikeA &&A, VectorLikeI &&ipiv, int lda = -1){
    valid::default_ld(M, lda);
    getrf(engine, M, N, std::forward<VectorLikeA>(A), lda, std::forward<VectorLikeI>(ipiv), new_vector(engine, A));
}
template<class VectorLikeA, class VectorLikeI, class VectorLikeW>
inline void getrf(gpu_engine const &engine, int M, int N, VectorLikeA &&A, VectorLikeI &&ipiv, int lda, VectorLikeW &&W){
    valid::default_ld(M, lda);
    getrf(engine, M, N, std::forward<VectorLikeA>(A), lda, std::forward<VectorLikeI>(ipiv), std::forward<VectorLikeW>(W));
}

template<class VectorLikeA, class VectorLikeI, class VectorLikeB>
inline void getrs(gpu_engine const &engine, char uplo, int N, int nrhs, VectorLikeA const &A, VectorLikeI const &ipiv, VectorLikeB &&B, int lda = -1, int ldb = -1){
    valid::default_ld(N, lda);
    valid::default_ld(N, ldb);
    getrs(engine, uplo, N, nrhs, A, lda, ipiv, std::forward<VectorLikeB>(B), ldb);
}

#endif

}

#endif

#endif
