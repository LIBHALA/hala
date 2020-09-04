#ifndef __HALA_GPU_BLAS2_HPP
#define __HALA_GPU_BLAS2_HPP
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
 * \internal
 * \file hala_gpu_blas2.hpp
 * \brief HALA wrappers to GPU-BLAS level 2 methods.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAGPUBLAS2
 *
 * \endinternal
*/

#include "hala_gpu_blas1.hpp"

namespace hala{

/*!
 * \ingroup HALAGPU
 * \addtogroup HALAGPUBLAS2 GPU-BLAS Level 2
 *
 * Wrappers to GPU-BLAS level 2 methods for vector operations.
 */

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to GPU-BLAS general matrix-vector multiply Xgemv().
 */
template<typename FPA, typename FPB, class VectorLikeA, class VectorLikeX, class VectorLikeY>
void gemv(gpu_engine const &engine, char trans, int M, int N, FPA alpha,
          VectorLikeA const &A, int lda, VectorLikeX const &x, int incx,
          FPB beta, VectorLikeY &&y, int incy){
    check_types(A, x, y);
    engine.check_gpu(A, x, y);
    pntr_check_set_size(beta, y, 1 + incy * ((is_n(trans) ? M : N) - 1), 1);
    assert( valid::gemv(trans, M, N, A, lda, x, incx, y, incy) );

    auto cuda_trans = trans_to_gpu(trans);

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto palpha = get_pointer<scalar_type>(alpha);
    auto pbeta  = get_pointer<scalar_type>(beta);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasSgemv, cublasDgemv, cublasCgemv, cublasZgemv,
                      "GPU-BLAS::Xgemv()", engine, cuda_trans, M, N, palpha, convert(A), lda, convert(x), incx, pbeta, convert(y), incy);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_sgemv, rocblas_dgemv, rocblas_cgemv, rocblas_zgemv,
                      "rocBlas::Xgemv()", engine, cuda_trans, M, N, palpha, convert(A), lda, convert(x), incx, pbeta, convert(y), incy);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to cuBLAS general banded matrix-vector multiply Xgbmv().
 */
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gbmv(gpu_engine const &engine, char trans, int M, int N, int kl, int ku, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    check_types(A, x, y);
    engine.check_gpu(A, x, y);
    pntr_check_set_size(beta, y, 1 + incy * ((is_n(trans) ? M : N) - 1), 1);
    assert( valid::gbmv(trans, M, N, kl, ku, A, lda, x, incx, y, incy) );

    auto cuda_trans = trans_to_gpu(trans);

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto palpha = get_pointer<scalar_type>(alpha);
    auto pbeta  = get_pointer<scalar_type>(beta);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasSgbmv, cublasDgbmv, cublasCgbmv, cublasZgbmv,
                      "GPU-BLAS::Xgbmv()", engine, cuda_trans, M, N, kl, ku, palpha, convert(A), lda, convert(x), incx, pbeta, convert(y), incy);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_sgbmv, rocblas_dgbmv, rocblas_cgbmv, rocblas_zgbmv,
                      "rocBlas::Xgbmv()", engine, cuda_trans, M, N, kl, ku, palpha, convert(A), lda, convert(x), incx, pbeta, convert(y), incy);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to cuBLAS symmetric or Hermitian matrix-vector multiply Xsymv().
 */
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void symv(gpu_engine const &engine, char uplo, int N, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    check_types(A, x, y);
    engine.check_gpu(A, x, y);
    pntr_check_set_size(beta, y, 1 + incy * (N - 1), 1);
    assert( valid::symv(uplo, N, A, lda, x, incx, y, incy) );

    auto cuda_uplo = uplo_to_gpu(uplo);

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto palpha = get_pointer<scalar_type>(alpha);
    auto pbeta  = get_pointer<scalar_type>(beta);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend6<conjugate, scalar_type>(cublasSsymv, cublasDsymv, cublasChemv, cublasZhemv, cublasCsymv, cublasZsymv,
                       "GPU-BLAS::Xsymv()", engine, cuda_uplo, N, palpha, convert(A), lda, convert(x), incx, pbeta, cconvert(y), incy);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend6<conjugate, scalar_type>(rocblas_ssymv, rocblas_dsymv, rocblas_chemv, rocblas_zhemv, rocblas_csymv, rocblas_zsymv,
                       "rocBlas::Xsymv()", engine, cuda_uplo, N, palpha, convert(A), lda, convert(x), incx, pbeta, cconvert(y), incy);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to BLAS symmetric or Hermitian banded matrix-vector multiply Xsbmv().
 */
template<bool conjugate = true, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void sbmv(gpu_engine const &engine, char uplo, int N, int k, FSa alpha, const VectorLikeA &A, int lda,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    check_types(A, x, y);
    engine.check_gpu(A, x, y);
    pntr_check_set_size(beta, y, 1 + incy * (N - 1), 1);
    assert( valid::sbmv(uplo, N, k, A, lda, x, incx, y, incy) );

    auto cuda_uplo = uplo_to_gpu(uplo);

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto palpha = get_pointer<scalar_type>(alpha);
    auto pbeta  = get_pointer<scalar_type>(beta);

    #ifdef HALA_ENABLE_CUDA
    if __HALA_CONSTEXPR_IF__ (conjugate || is_float<scalar_type>::value || is_double<scalar_type>::value){
        cuda_call_backend<scalar_type>(cublasSsbmv, cublasDsbmv, cublasChbmv, cublasZhbmv,
                        "GPU-BLAS::Xsbmv()", engine, cuda_uplo, N, k, palpha, convert(A), lda, convert(x), incx, pbeta, cconvert(y), incy);
    }else{
        // fallback mode, using the CPU version with data-movement
        auto cpuA = engine.unload(A);
        auto cpux = engine.unload(x);
        auto cpuy = engine.unload(y);
        hala::sbmv<conjugate>(uplo, N, k, *((scalar_type const*) palpha), cpuA, lda, cpux, incx, *((scalar_type const*) pbeta), cpuy, incy);
        check_cuda(cudaMemcpy(get_data(y), get_data(cpuy), get_size(y) * sizeof(scalar_type),
                              cudaMemcpyHostToDevice), "hala::sbmv::cudaMemcpy()");
    }
    #endif
    #ifdef HALA_ENABLE_ROCM
    if __HALA_CONSTEXPR_IF__ (is_float<scalar_type>::value || is_double<scalar_type>::value){
        rocm_call_backend<scalar_type>(rocblas_ssbmv, rocblas_dsbmv, rocblas_chbmv, rocblas_zhbmv,
                         "rocBlas::Xsbmv()", engine, cuda_uplo, N, k, palpha, convert(A), lda, convert(x), incx, pbeta, cconvert(y), incy);
    }else{
        // fallback mode, using the CPU version with data-movement
        auto cpuA = engine.unload(A);
        auto cpux = engine.unload(x);
        auto cpuy = engine.unload(y);
        hala::sbmv<conjugate>(uplo, N, k, *((scalar_type const*) palpha), cpuA, lda, cpux, incx, *((scalar_type const*) pbeta), cpuy, incy);
        gpu_copy_n<copy_direction::host2device>(get_data(cpuy), get_size(y), get_data(y));
    }
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to BLAS symmetric or Hermitian matrix-vector multiply Xspmv() in packed format.
 */
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void spmv(gpu_engine const &engine, char uplo, int N, FSa alpha, const VectorLikeA &A,
                 const VectorLikeX &x, int incx, FSb beta, VectorLikeY &&y, int incy){
    check_types(A, x, y);
    engine.check_gpu(A, x, y);
    pntr_check_set_size(beta, y, 1 + incy * (N - 1), 1);
    assert( valid::spmv(uplo, N, A, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeA>;

    #ifdef HALA_ENABLE_CUDA
    if __HALA_CONSTEXPR_IF__ (conjugate || !is_complex<scalar_type>::value){
        cublasFillMode_t cuda_uplo = uplo_to_gpu(uplo);

        auto palpha = get_pointer<scalar_type>(alpha);
        auto pbeta  = get_pointer<scalar_type>(beta);

        cuda_call_backend<scalar_type>(cublasSspmv, cublasDspmv, cublasChpmv, cublasZhpmv,
                          "GPU-BLAS::Xspmv()", engine, cuda_uplo, N, palpha, convert(A), convert(x), incx, pbeta, cconvert(y), incy);
    }else{
        // since Cspmv and Zspmv are missing from the GPU-BLAS standard, unpack A, form full matrix, use gemv
        auto full_matrix = new_vector(engine, A);
        { // respect the user selected pointer mode
            gpu_pntr<host_pntr> hold(engine);
            auto unpacked = new_vector(engine, A);
            force_size(unpacked, hala_size(N, N));
            scal(engine, 0.0, unpacked);
            tp2tr(engine, uplo, N, A, unpacked, N);
            force_size(full_matrix, hala_size(N, N));
            scal(engine, 0.0, full_matrix);
            geam(engine, 'N', 'T', N, N, 1.0, unpacked, N, 1.0, unpacked, N, full_matrix, N);
            scal(engine, N, 0.5, full_matrix, N+1);
        }
        gemv(engine, 'N', N, N, alpha, full_matrix, N, x, incx, beta, y, incy);
    }
    #endif
    #ifdef HALA_ENABLE_ROCM
    if __HALA_CONSTEXPR_IF__ (conjugate || !is_complex<scalar_type>::value){
        auto cuda_uplo = uplo_to_gpu(uplo);

        auto palpha = get_pointer<scalar_type>(alpha);
        auto pbeta  = get_pointer<scalar_type>(beta);

        rocm_call_backend<scalar_type>(rocblas_sspmv, rocblas_dspmv, rocblas_chpmv, rocblas_zhpmv,
                          "rocBlas::Xspmv()", engine, cuda_uplo, N, palpha, convert(A), convert(x), incx, pbeta, cconvert(y), incy);
    }else{
        // fallback, since cspmv and zspmv are missing from the standard, use the CPU
        auto cA = engine.unload(A);
        auto cx = engine.unload(x);
        auto cy = engine.unload(y);
        spmv<conjugate>(uplo, N, alpha, cA, cx, incx, beta, cy, incy);
        gpu_copy_n<copy_direction::host2device>(get_data(cy), get_size(y), get_data(y));
    }
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to cuBLAS triangular matrix-vector multiply Xtrmv().
 */
template<class VectorLikeA, class VectorLikeX>
inline void trmv(gpu_engine const &engine, char uplo, char trans, char diag, int N, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    check_types(A, x);
    engine.check_gpu(A, x);
    assert( valid::trmsv(uplo, trans, diag, N, A, lda, x, incx) );

    auto cuda_uplo  = uplo_to_gpu(uplo);
    auto cuda_trans = trans_to_gpu(trans);
    auto cuda_diag  = diag_to_gpu(diag);

    using scalar_type = get_scalar_type<VectorLikeA>;

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasStrmv, cublasDtrmv, cublasCtrmv, cublasZtrmv,
                      "GPU-BLAS::Xtrmv()", engine, cuda_uplo, cuda_trans, cuda_diag, N, convert(A), lda, cconvert(x), incx);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_strmv, rocblas_dtrmv, rocblas_ctrmv, rocblas_ztrmv,
                      "rocBlas::Xtrmv()", engine, cuda_uplo, cuda_trans, cuda_diag, N, convert(A), lda, cconvert(x), incx);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to BLAS triangular matrix-vector solve Xtrsv().
 */
template<class VectorLikeA, class VectorLikeX>
inline void trsv(gpu_engine const &engine, char uplo, char trans, char diag, int N, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    check_types(A, x);
    engine.check_gpu(A, x);
    assert( valid::trmsv(uplo, trans, diag, N, A, lda, x, incx) );

    auto cuda_uplo  = uplo_to_gpu(uplo);
    auto cuda_trans = trans_to_gpu(trans);
    auto cuda_diag  = diag_to_gpu(diag);

    using scalar_type = get_scalar_type<VectorLikeA>;

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasStrsv, cublasDtrsv, cublasCtrsv, cublasZtrsv,
                      "GPU-BLAS::Xtrsv()", engine, cuda_uplo, cuda_trans, cuda_diag, N, convert(A), lda, cconvert(x), incx);
    #endif
    #ifdef HALA_ENABLE_ROCM
    if __HALA_CONSTEXPR_IF__ (not is_complex<scalar_type>::value){
        rocm_call_backend<scalar_type>(rocblas_strsv, rocblas_dtrsv, rocblas_ctrsv, rocblas_ztrsv,
                        "rocBlas::Xtrsv()", engine, cuda_uplo, cuda_trans, cuda_diag, N, convert(A), lda, cconvert(x), incx);
    }else{
        auto cpuA = engine.unload(A);
        auto cpux = engine.unload(x);
        trsv(uplo, trans, diag, N, cpuA, lda, cpux, incx);
        gpu_copy_n<copy_direction::host2device>(get_data(cpux), get_size(x), get_data(x));
    }
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to BLAS triangular matrix-vector multiply in packed format Xtpmv().
 */
template<class VectorLikeA, class VectorLikeX>
inline void tpmv(gpu_engine const &engine, char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int incx = 1){
    check_types(A, x);
    engine.check_gpu(A, x);
    assert( valid::tpmsv(uplo, trans, diag, N, A, x, incx) );

    auto cuda_uplo  = uplo_to_gpu(uplo);
    auto cuda_trans = trans_to_gpu(trans);
    auto cuda_diag  = diag_to_gpu(diag);

    using scalar_type = get_scalar_type<VectorLikeA>;

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasStpmv, cublasDtpmv, cublasCtpmv, cublasZtpmv,
                      "GPU-BLAS::Xtpmv()", engine, cuda_uplo, cuda_trans, cuda_diag, N, convert(A), cconvert(x), incx);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_stpmv, rocblas_dtpmv, rocblas_ctpmv, rocblas_ztpmv,
                      "rocBlas::Xtpmv()", engine, cuda_uplo, cuda_trans, cuda_diag, N, convert(A), cconvert(x), incx);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to BLAS triangular matrix-vector solve in packed format Xtpsv().
 */
template<class VectorLikeA, class VectorLikeX>
inline void tpsv(gpu_engine const &engine, char uplo, char trans, char diag, int N, const VectorLikeA &A, VectorLikeX &&x, int incx = 1){
    check_types(A, x);
    engine.check_gpu(A, x);
    assert( valid::tpmsv(uplo, trans, diag, N, A, x, incx) );

    auto cuda_uplo  = uplo_to_gpu(uplo);
    auto cuda_trans = trans_to_gpu(trans);
    auto cuda_diag  = diag_to_gpu(diag);

    using scalar_type = get_scalar_type<VectorLikeA>;

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasStpsv, cublasDtpsv, cublasCtpsv, cublasZtpsv,
                      "GPU-BLAS::Xtpsv()", engine, cuda_uplo, cuda_trans, cuda_diag, N, convert(A), cconvert(x), incx);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_stpsv, rocblas_dtpsv, rocblas_ctpsv, rocblas_ztpsv,
                      "rocmBlas::Xtpsv()", engine, cuda_uplo, cuda_trans, cuda_diag, N, convert(A), cconvert(x), incx);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to cuBLAS triangular banded matrix-vector multiply Xtbmv().
 */
template<class VectorLikeA, class VectorLikeX>
inline void tbmv(gpu_engine const &engine, char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    check_types(A, x);
    engine.check_gpu(A, x);
    assert( valid::tbmsv(uplo, trans, diag, N, k, A, lda, x, incx) );

    auto cuda_uplo  = uplo_to_gpu(uplo);
    auto cuda_trans = trans_to_gpu(trans);
    auto cuda_diag  = diag_to_gpu(diag);

    using scalar_type = get_scalar_type<VectorLikeA>;

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasStbmv, cublasDtbmv, cublasCtbmv, cublasZtbmv,
                      "GPU-BLAS::Xtbmv()", engine, cuda_uplo, cuda_trans, cuda_diag, N, k, convert(A), lda, cconvert(x), incx);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_stbmv, rocblas_dtbmv, rocblas_ctbmv, rocblas_ztbmv,
                      "rocBlas::Xtbmv()", engine, cuda_uplo, cuda_trans, cuda_diag, N, k, convert(A), lda, cconvert(x), incx);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to cuBLAS triangular matrix-vector solve Xtbsv().
 */
template<class VectorLikeA, class VectorLikeX>
inline void tbsv(gpu_engine const &engine, char uplo, char trans, char diag, int N, int k, const VectorLikeA &A, int lda, VectorLikeX &&x, int incx){
    check_types(A, x);
    engine.check_gpu(A, x);
    assert( valid::tbmsv(uplo, trans, diag, N, k, A, lda, x, incx) );

    auto cuda_uplo  = uplo_to_gpu(uplo);
    auto cuda_trans = trans_to_gpu(trans);
    auto cuda_diag  = diag_to_gpu(diag);

    using scalar_type = get_scalar_type<VectorLikeA>;

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasStbsv, cublasDtbsv, cublasCtbsv, cublasZtbsv,
                      "GPU-BLAS::Xtbsv()", engine, cuda_uplo, cuda_trans, cuda_diag, N, k, convert(A), lda, cconvert(x), incx);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_stbsv, rocblas_dtbsv, rocblas_ctbsv, rocblas_ztbsv,
                      "rocBlas::Xtbsv()", engine, cuda_uplo, cuda_trans, cuda_diag, N, k, convert(A), lda, cconvert(x), incx);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to cuBLAS general rank-1 update Xgerx().
 */
template<bool conjugate = true, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void ger(gpu_engine const &engine, int M, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    check_types(x, y, A);
    engine.check_gpu(x, y, A);
    assert( valid::ger(M, N, x, incx, y, incy, A, lda) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto palpha = get_pointer<scalar_type>(alpha);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend6<conjugate, scalar_type>(cublasSger, cublasDger, cublasCgerc, cublasZgerc, cublasCgeru, cublasZgeru,
                       "GPU-BLAS::Xgerx()", engine, M, N, palpha, convert(x), incx, convert(y), incy, cconvert(A), lda);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend6<conjugate, scalar_type>(rocblas_sger, rocblas_dger, rocblas_cgerc, rocblas_zgerc, rocblas_cgeru, rocblas_zgeru,
                       "rocBlas::Xgerx()", engine, M, N, palpha, convert(x), incx, convert(y), incy, cconvert(A), lda);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to cuBLAS symmetric or Hermitian rank-1 update Xsyr() and Xher().
 */
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeA>
inline void syr(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A, int lda){
    check_types(x, A);
    engine.check_gpu(x, A);
    assert( valid::syr(uplo, N, x, incx, A, lda) );

    auto cuda_uplo  = uplo_to_gpu(uplo);

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto palpha = get_pointer<scalar_type>(alpha);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend6<conjugate, scalar_type>(cublasSsyr, cublasDsyr, cublasCher, cublasZher, cublasCsyr, cublasZsyr,
                       "GPU-BLAS::Xsyr()/Xher()", engine, cuda_uplo, N, palpha, convert(x), incx, cconvert(A), lda);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend6<conjugate, scalar_type>(rocblas_ssyr, rocblas_dsyr, rocblas_cher, rocblas_zher, rocblas_csyr, rocblas_zsyr,
                       "rocBlas::Xsyr()/Xher()", engine, cuda_uplo, N, palpha, convert(x), incx, cconvert(A), lda);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to cuBLAS symmetric or Hermitian rank-1 update using packed format Xspr() and Xhpr().
 */
template<bool conjugate = true, typename FSa, class VectorLikeX, class VectorLikeA>
inline void spr(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeA &&A){
    check_types(x, A);
    engine.check_gpu(x, A);
    assert( valid::spr(uplo, N, x, incx, A) );

    using scalar_type = get_scalar_type<VectorLikeA>;

    #ifdef HALA_ENABLE_CUDA
    if __HALA_CONSTEXPR_IF__ (conjugate || !is_complex<scalar_type>::value){
        cublasFillMode_t cuda_uplo  = uplo_to_gpu(uplo);
        auto palpha = get_pointer<scalar_type>(alpha);
        cuda_call_backend<scalar_type>(cublasSspr, cublasDspr, cublasChpr, cublasZhpr,
                          "GPU-BLAS::Xspr()", engine, cuda_uplo, N, palpha, convert(x), incx, cconvert(A));
    }else{
        // since Cspr and Zspr are missing from the GPU-BLAS standard, unpack A, use syr, pack A
        auto unpacked = new_vector(engine, A);
        tp2tr(engine, uplo, N, A, unpacked, N);
        syr<conjugate>(engine, uplo, N, alpha, x, incx, unpacked, N);
        tr2tp(engine, uplo, N, unpacked, N, A);
    }
    #endif
    #ifdef HALA_ENABLE_ROCM
    auto cuda_uplo  = uplo_to_gpu(uplo);
    auto palpha = get_pointer<scalar_type>(alpha);
    rocm_call_backend6<conjugate, scalar_type>(rocblas_sspr, rocblas_dspr, rocblas_chpr, rocblas_zhpr, rocblas_cspr, rocblas_zspr,
                       "rocBlas::Xspr()", engine, cuda_uplo, N, palpha, convert(x), incx, cconvert(A));
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to cuBLAS symmetric or Hermitian rank-2 update Xsyr2() and Xher2().
 */
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void syr2(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A, int lda){
    check_types(x, A);
    engine.check_gpu(x, A);
    assert( valid::syr2(uplo, N, x, incx, y, incy, A, lda) );

    auto cuda_uplo  = uplo_to_gpu(uplo);

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto palpha = get_pointer<scalar_type>(alpha);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend6<conjugate, scalar_type>(cublasSsyr2, cublasDsyr2, cublasCher2, cublasZher2, cublasCsyr2, cublasZsyr2,
                       "GPU-BLAS::Xsyr2()/Xher2", engine, cuda_uplo, N, palpha, convert(x), incx, convert(y), incy, cconvert(A), lda);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend6<conjugate, scalar_type>(rocblas_ssyr2, rocblas_dsyr2, rocblas_cher2, rocblas_zher2, rocblas_csyr2, rocblas_zsyr2,
                       "rocBlas::Xsyr2()/Xher2", engine, cuda_uplo, N, palpha, convert(x), incx, convert(y), incy, cconvert(A), lda);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS2
 * \brief Wrapper to cuBLAS symmetric or Hermitian rank-2 update in packed format Xspr2() and Xhpr2().
 */
template<bool conjugate = false, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void spr2(gpu_engine const &engine, char uplo, int N, FSa alpha, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy, VectorLikeA &&A){
    check_types(x, A);
    engine.check_gpu(x, A);
    assert( valid::spr2(uplo, N, x, incx, y, incy, A) );

    using scalar_type = get_scalar_type<VectorLikeA>;

    #ifdef HALA_ENABLE_CUDA
    if __HALA_CONSTEXPR_IF__ (conjugate || !is_complex<scalar_type>::value){
        auto cuda_uplo  = uplo_to_gpu(uplo);
        auto palpha = get_pointer<scalar_type>(alpha);
        cuda_call_backend<scalar_type>(cublasSspr2, cublasDspr2, cublasChpr2, cublasZhpr2,
                          "GPU-BLAS::Xspr2()", engine, cuda_uplo, N, palpha, convert(x), incx, convert(y), incy, cconvert(A));
    }else{
        // since Cspr2 and Zspr2 are missing from the GPU-BLAS standard, unpack A, use syr2, pack A
        auto unpacked = new_vector(engine, A);
        tp2tr(engine, uplo, N, A, unpacked, N);
        syr2<conjugate>(engine, uplo, N, alpha, x, incx, y, incy, unpacked, N);
        tr2tp(engine, uplo, N, unpacked, N, A);
    }
    #endif
    #ifdef HALA_ENABLE_ROCM
    if __HALA_CONSTEXPR_IF__ (conjugate || !is_complex<scalar_type>::value){
        auto cuda_uplo  = uplo_to_gpu(uplo);
        auto palpha = get_pointer<scalar_type>(alpha);
        rocm_call_backend<scalar_type>(rocblas_sspr2, rocblas_dspr2, rocblas_chpr2, rocblas_zhpr2,
                          "rocBlas::Xspr2()", engine, cuda_uplo, N, palpha, convert(x), incx, convert(y), incy, cconvert(A));
    }else{
        // since Cspr2 and Zspr2 are missing from the GPU-BLAS standard, unpack A, use syr2, pack A
        auto cx = engine.unload(x);
        auto cy = engine.unload(y);
        auto cA = engine.unload(A);
        spr2(uplo, N, alpha, cx, incx, cy, incy, cA);
        gpu_copy_n<copy_direction::host2device>(get_data(cA), ((N+1) * N) / 2, get_data(A));
    }
    #endif
}

}

#endif
