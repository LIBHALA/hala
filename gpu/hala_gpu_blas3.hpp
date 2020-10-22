#ifndef __HALA_GPU_BLAS3_HPP
#define __HALA_GPU_BLAS3_HPP
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
 * \file hala_gpu_blas3.hpp
 * \brief HALA wrappers to GPU-BLAS level 3 methods.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAGPUBLAS3
 *
 * \endinternal
*/

#include "hala_gpu_blas2.hpp"

namespace hala{

/*!
 * \ingroup HALAGPU
 * \addtogroup HALAGPUBLAS3 GPU-BLAS Level 3
 *
 * Wrappers to GPU-BLAS level 3 methods for vector operations.
 */

/*!
 * \ingroup HALAGPUBLAS3
 * \brief Wrapper to GPU triangular matrix-matrix multiply Xtrmm().
 */
template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trmm(gpu_engine const &engine, char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, int lda, VectorLikeB &&B, int ldb){
    check_types(A, B);
    engine.check_gpu(A, B);
    assert( valid::trmsm(side, uplo, trans, diag, M, N, A, lda, B, ldb) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto palpha = get_pointer<scalar_type>(alpha);

    auto gpu_side  = side_to_gpu(side);
    auto gpu_uplo  = uplo_to_gpu(uplo);
    auto gpu_trans = trans_to_gpu(trans);
    auto gpu_diag  = diag_to_gpu(diag);

    #ifdef HALA_ENABLE_CUDA
    auto C = new_vector(engine, B);
    force_size(hala_size(M, N), C);

    cuda_call_backend<scalar_type>(cublasStrmm, cublasDtrmm, cublasCtrmm, cublasZtrmm,
                      "GPU-BLAS::Xtrmm()", engine, gpu_side, gpu_uplo, gpu_trans, gpu_diag,
                      M, N, palpha, convert(A), lda, convert(B), ldb, cconvert(C), M);
    if (ldb == M){
        vcopy(engine, C, B); // copy back
    }else{ // copy only the block
        gpu_pntr<host_pntr> hold(engine); // in case device pointers were used in the call above
        geam(engine, 'N', 'N', M, N, 1, C, M, 0, C, M, B, ldb);
    }
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_strmm, rocblas_dtrmm, rocblas_ctrmm, rocblas_ztrmm,
                      "rocBlas::Xtrmm()", engine, gpu_side, gpu_uplo, gpu_trans, gpu_diag,
                      M, N, palpha, convert(A), lda, convert(B), ldb);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS3
 * \brief Wrapper to GPU triangular matrix-matrix solve Xtrsm().
 */
template<typename FS, class VectorLikeA, class VectorLikeB>
inline void trsm(gpu_engine const &engine, char side, char uplo, char trans, char diag, int M, int N, FS alpha, const VectorLikeA &A, int lda, VectorLikeB &&B, int ldb){
    check_types(A, B);
    assert( valid::trmsm(side, uplo, trans, diag, M, N, A, lda, B, ldb) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto palpha = get_pointer<scalar_type>(alpha);

    auto cuda_side  = side_to_gpu(side);
    auto cuda_uplo  = uplo_to_gpu(uplo);
    auto cuda_trans = trans_to_gpu(trans);
    auto cuda_diag  = diag_to_gpu(diag);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasStrsm, cublasDtrsm, cublasCtrsm, cublasZtrsm,
                      "GPU-BLAS::Xtrsm()", engine, cuda_side, cuda_uplo, cuda_trans, cuda_diag, M, N, palpha, convert(A), lda, cconvert(B), ldb);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_strsm, rocblas_dtrsm, rocblas_ctrsm, rocblas_ztrsm,
                      "rocBlas::Xtrsm()", engine, cuda_side, cuda_uplo, cuda_trans, cuda_diag, M, N, palpha, convert(A), lda, cconvert(B), ldb);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS3
 * \brief Wrapper to BLAS symmetric rank-k update Xsyrk() and Xherk().
 */
template<bool conjugate = false, typename FPA, typename FPB, class VectorLikeA, class VectorLikeC>
inline void syrk(gpu_engine const &engine, char uplo, char trans, int N, int K, FPA alpha, VectorLikeA const &A, int lda,
                 FPB beta, VectorLikeC &&C, int ldc){
    check_types(A, C);
    engine.check_gpu(A, C);
    pntr_check_set_size(beta, C, ldc, N);
    assert( valid::syrk(uplo, trans, N, K, A, lda, C, ldc) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto palpha = get_pointer<scalar_type>(alpha);
    auto pbeta  = get_pointer<scalar_type>(beta);

    auto gpu_uplo  = uplo_to_gpu(uplo);

    #ifdef HALA_ENABLE_CUDA
    auto gpu_trans = trans_to_gpu(trans);
    cuda_call_backend6<conjugate, scalar_type>(cublasSsyrk, cublasDsyrk, cublasCherk, cublasZherk, cublasCsyrk, cublasZsyrk,
                       "GPU-BLAS::Xsyrk()/Xherk()", engine, gpu_uplo, gpu_trans, N, K, palpha, convert(A), lda, pbeta, cconvert(C), ldc);
    #endif
    #ifdef HALA_ENABLE_ROCM
    auto gpu_trans = hermitian_trans<conjugate, scalar_type>(trans);
    rocm_call_backend6<conjugate, scalar_type>(rocblas_ssyrk, rocblas_dsyrk, rocblas_cherk, rocblas_zherk, rocblas_csyrk, rocblas_zsyrk,
                       "rocBlas::Xsyrk()/Xherk()", engine, gpu_uplo, gpu_trans, N, K, palpha, convert(A), lda, pbeta, cconvert(C), ldc);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS3
 * \brief Wrapper to BLAS symmetric rank-k update Xsyrk() and Xherk().
 */
template<bool conjugate = false, typename FPA, typename FPB, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void syr2k(gpu_engine const &engine, char uplo, char trans, int N, int K, FPA alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FPB beta, VectorLikeC &&C, int ldc){
    check_types(A, B, C);
    engine.check_gpu(A, B, C);
    pntr_check_set_size(beta, C, ldc, N);
    assert( valid::syr2k(uplo, trans, N, K, A, lda, B, ldb, C, ldc) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto palpha = get_pointer<scalar_type>(alpha);
    auto pbeta  = get_pointer<scalar_type>(beta);

    auto gpu_uplo  = uplo_to_gpu(uplo);

    #ifdef HALA_ENABLE_CUDA
    auto gpu_trans = trans_to_gpu(trans);
    cuda_call_backend6<conjugate, scalar_type>(cublasSsyr2k, cublasDsyr2k, cublasCher2k, cublasZher2k,
                                               cublasCsyr2k, cublasZsyr2k,
                       "GPU-BLAS::Xsyr2k()/Xher2k()", engine, gpu_uplo, gpu_trans, N, K, palpha, convert(A), lda,
                       convert(B), ldb, pbeta, cconvert(C), ldc);
    #endif
    #ifdef HALA_ENABLE_ROCM
    auto gpu_trans = hermitian_trans<conjugate, scalar_type>(trans);
    rocm_call_backend6<conjugate, scalar_type>(rocblas_ssyr2k, rocblas_dsyr2k, rocblas_cher2k, rocblas_zher2k,
                                               rocblas_csyr2k, rocblas_zsyr2k,
                       "rocBlas::Xsyr2k()/Xher2k()", engine, gpu_uplo, gpu_trans, N, K, palpha, convert(A), lda,
                       convert(B), ldb, pbeta, cconvert(C), ldc);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS3
 * \brief Wrapper to GPU symmetric or Hermitian matrix multiply Xsymm()/Xhemm().
 */
template<bool conjugate = false, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void symm(gpu_engine const &engine, char side, char uplo, int M, int N, FSa alpha, VectorLikeA const &A, int lda,
                 VectorLikeB const &B, int ldb, FSb beta, VectorLikeC &&C, int ldc){
    check_types(A, B, C);
    pntr_check_set_size(beta, C, ldc, N);
    assert( valid::symm(side, uplo, M, N, A, lda, B, ldb, C, ldc) );

    auto cuda_side  = side_to_gpu(side);
    auto cuda_uplo  = uplo_to_gpu(uplo);

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto palpha = get_pointer<scalar_type>(alpha);
    auto pbeta  = get_pointer<scalar_type>(beta);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend6<conjugate, scalar_type>(cublasSsymm, cublasDsymm, cublasChemm, cublasZhemm,
                                               cublasCsymm, cublasZsymm,
                       "GPU-BLAS::Xsymm()/Xhemm()", engine, cuda_side, cuda_uplo, M, N, palpha,
                       convert(A), lda, convert(B), ldb, pbeta, cconvert(C), ldc);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend6<conjugate, scalar_type>(rocblas_ssymm, rocblas_dsymm, rocblas_chemm, rocblas_zhemm,
                                               rocblas_csymm, rocblas_zsymm,
                       "rocBlas::Xsymm()/Xhemm()", engine, cuda_side, cuda_uplo, M, N, palpha,
                       convert(A), lda, convert(B), ldb, pbeta, cconvert(C), ldc);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS3
 * \brief Wrapper to GPU general matrix multiply Xgemm().
 */
template<typename FPA, typename FPB, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void gemm(gpu_engine const &engine, char transa, char transb, int M, int N, int K,
                 FPA alpha, VectorLikeA const &A, int lda, VectorLikeB const &B, int ldb,
                 FPB beta, VectorLikeC &&C, int ldc){
    check_types(A, B, C);
    engine.check_gpu(A, B, C);
    pntr_check_set_size(beta, C, ldc, N);
    assert( valid::gemm(transa, transb, M, N, K, A, lda, B, ldb, C, ldc) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto cuda_transa = trans_to_gpu(transa);
    auto cuda_transb = trans_to_gpu(transb);

    auto palpha = get_pointer<scalar_type>(alpha);
    auto pbeta  = get_pointer<scalar_type>(beta);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasSgemm, cublasDgemm, cublasCgemm, cublasZgemm,
                      "GPU-BLAS::Xgemm()", engine, cuda_transa, cuda_transb, M, N, K, palpha,
                      convert(A), lda, convert(B), ldb, pbeta, cconvert(C), ldc);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_sgemm, rocblas_dgemm, rocblas_cgemm, rocblas_zgemm,
                      "rocBlas::Xgemm()", engine, cuda_transa, cuda_transb, M, N, K, palpha,
                      convert(A), lda, convert(B), ldb, pbeta, cconvert(C), ldc);
    #endif
}

}

#endif
