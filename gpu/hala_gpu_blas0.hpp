#ifndef __HALA_GPU_BLAS0_HPP
#define __HALA_GPU_BLAS0_HPP
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
 * \file hala_gpu_blas0.hpp
 * \brief HALA wrappers to GPU-BLAS helpers.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAGPUBLAS0
 *
 * \endinternal
*/

#include "hala_gpu_engine.hpp"

namespace hala{


/*!
 * \ingroup HALAGPU
 * \addtogroup HALAGPUBLAS0 GPU-BLAS Extensions
 *
 * Several BLAS-like extensions included in the GPU-BLAS libraries.
 */

/*!
 * \ingroup HALAGPUBLAS0
 * \brief Wrapper to GPU-BLAS matrix addition Xgeam().
 *
 * The GPU-BLAS library includes the BLAS-like extension Xgeam() that computes
 * \f$ C = \alpha op(A) + \beta op(B) \f$.
 * - \b transa and \b transb are lower-upper case N/T/C indicating the \b op applied to \b A and \b B
 * - \b op can be identity (N), transpose (T), or conjigate-transpose (C)
 * - \b A and \b B are an \b M by \b N (or N by M) matrices with leading dimensions \b lda and \b ldb
 * - \b lda and \b ldb are minimum M for transa = N and N otherwise
 * - \b C is \b ldc by N with minimum \b ldc of \b M
 *
 * Overloads:
 * \code
 *  geam(engine, transa, transb, M, N, alpha, A, beta, B, C, lda=-1, ldb=-1, ldc=-1);
 *  geam(engine, transa, transb, M, N, alpha, A, lda, beta, B, ldb, C, ldc);
 * \endcode
 * The method is GPU-BLAS extension hence only the hala::gpu_engine and hala::mixed_engine overloads
 * will perform any action, the hala::cpu_engine overload will be a no-op.
 */
template<typename FPa, class VectorLikeA, typename FPb, class VectorLikeB, class VectorLikeC>
inline void geam(gpu_engine const &engine, char transa, char transb, int M, int N, FPa alpha, VectorLikeA const &A, int lda,
                 FPb beta, VectorLikeB const &B, int ldb, VectorLikeC &&C, int ldc){
    check_types(A, B, C);
    engine.check_gpu(A, B, C);
    assert( check_trans(transa) );
    assert( check_trans(transb) );
    assert( lda >= (is_n(transa) ? M : N) );
    assert( ldb >= (is_n(transb) ? M : N) );
    assert( ldc >= M );
    assert( check_size(A, lda, is_n(transa) ? N : M) );
    assert( check_size(B, ldb, is_n(transb) ? N : M) );
    check_set_size(assume_output, C, ldc, N);

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto gpu_transa = trans_to_gpu(transa);
    auto gpu_transb = trans_to_gpu(transb);

    auto palpha = get_pointer<scalar_type>(alpha);
    auto pbeta  = get_pointer<scalar_type>(beta);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasSgeam, cublasDgeam, cublasCgeam, cublasZgeam, "GPU-BLAS::Xgeam()",
           engine, gpu_transa, gpu_transb, M, N, palpha, convert(A), lda, pbeta, convert(B), ldb, cconvert(C), ldc);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_sgeam, rocblas_dgeam, rocblas_cgeam, rocblas_zgeam, "rocBlas::Xgeam()",
           engine, gpu_transa, gpu_transb, M, N, palpha, convert(A), lda, pbeta, convert(B), ldb, cconvert(C), ldc);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS0
 * \brief Wrapper to GPU-BLAS matrix addition Xdgmm().
 *
 * The GPU-BLAS library includes the BLAS-like extension Xdgmm() that computes \f$ C = A D \f$ or \f$ C = D A \f$,
 * i.e., a dense matrix multiplied by a diagonal matrix.
 * - \b side is upper/lower case L/R indicating whether D is on the left (L) or right (R) of \b A
 * - \b A is an \b M by \b N matrix with leading dimension \b lda (minimum M)
 * - \b x holds the diagonal entries of D in a stride of \b incx
 * - the active entries in \b x must be either M (for side left) or N (for side right)
 * - \b C is the resulting matrix with leading dimension \b ldc (minimum M)
 *
 * Overloads:
 * \code
 *  dgmm(engine, side, M, N, A, x, C, lda=-1, incx=-1, ldc=-1);
 *  dgmm(engine, side, M, N, A, lda, x, incx, C, ldc);
 * \endcode
 * The method is GPU-BLAS extension hence only the hala::gpu_engine and hala::mixed_engine overloads
 * will perform any action, the hala::cpu_engine overload will be a no-op.
 */
template<class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void dgmm(gpu_engine const &engine, char side, int M, int N, VectorLikeA const &A, int lda,
                 VectorLikeB const &x, int incx, VectorLikeC &&C, int ldc){
    check_types(A, x, C);
    engine.check_gpu(A, x, C);
    assert( check_side(side) );
    assert( lda >= M );
    assert( ldc >= M );
    assert( incx > 0 );
    assert( check_size(A, lda, N) );
    assert( check_size(x, (is_l(side) ? (1 + (M - 1) * incx) : (1 + (N - 1) * incx))) );
    check_set_size(assume_output, C, ldc, N);

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto gpu_side = side_to_gpu(side);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasSdgmm, cublasDdgmm, cublasCdgmm, cublasZdgmm,
        "GPU-BLAS::Xdgmm()", engine, gpu_side, M, N, convert(A), lda, convert(x), incx, cconvert(C), ldc);
    #endif
    #ifdef HALA_ENABLE_ROCM
    // as of rocblas 3.5 dgmm is bugged, see https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1160
    // fixed in develop, but the fix didn't make it to 3.7, keep fingers crosses for 3.9
//     rocm_call_backend<scalar_type>(rocblas_sdgmm, rocblas_ddgmm, rocblas_cdgmm, rocblas_zdgmm,
//         "GPU-BLAS::Xdgmm()", engine, gpu_side, M, N, convert(A), lda, convert(x), incx, cconvert(C), ldc);
    if (gpu_side == rocblas_side_left){
        if (incx == 1){
            for(int i=0; i<N; i++)
                gbmv(engine, 'N', M, M, 0, 0, 1.0, x, 1, get_data(A) + i * lda, 1, 0.0, get_data(C) + i * ldc, 1);
        }else{
            gpu_vector<scalar_type> xcont;
            vcopy(engine, M, x, incx, xcont, 1);
            for(int i=0; i<N; i++)
                gbmv(engine, 'N', M, M, 0, 0, 1.0, xcont, 1, get_data(A) + i * lda, 1, 0.0, get_data(C) + i * ldc, 1);
        }
    }else{
        gpu_pntr<device_pntr> using_device_pntr(engine);
        for(int i=0; i<N; i++){
            vcopy(engine, M, get_data(A) + i * lda, 1, get_data(C) + i * ldc, 1);
            scal(engine, M, get_data(x) + i * incx, get_data(C) + i * ldc, 1);
        }
    }
    #endif
}

/*!
 * \ingroup HALAGPUBLAS0
 * \brief Wrapper to GPU-BLAS matrix addition Xtpttr(), converts triangular-packed to triangular-regular formats.
 *
 * Takes a triangular packed matrix and converts it into a regular \b N  by \b N triangular matrix.
 * - \b uplo is lower/upper case L/U indicating whether to use the lower (L) or upper (U) part of \b A
 * - \b N is the number of rows and columns of the matrix
 * - \b AP has N (N+1) / 2 entries holding the packed matrix
 * - \b A is a matrix \b lda by \b N and the upper or lower part of the matrix will be rewritten
 * - \b lda is minimum \b N, and \b A is always resized (if the size is not sufficient already)
 * See also the inverse operation hala::tr2tp().
 *
 * Overloads: the method can be used with a hala::mixed_engine too.
 */
template<class VectorLikeA, class VectorLikeAP>
void tp2tr(gpu_engine const &engine, char uplo, int N, VectorLikeAP const &AP, VectorLikeA &&A, int lda = -1){
    check_types(A, AP);
    valid::default_ld(N, lda);
    assert( lda >= N );
    assert( check_size( AP, ( (N * (N+1)) / 2 )) );
    assert( check_uplo(uplo) );
    check_set_size(assume_output, A, lda, N);

    #ifdef HALA_ENABLE_CUDA
    auto cuda_uplo = uplo_to_gpu(uplo);

    using scalar_type = get_scalar_type<VectorLikeA>;

    cuda_call_backend<scalar_type>(cublasStpttr, cublasDtpttr, cublasCtpttr, cublasZtpttr,
        "GPU-BLAS::Xtpttr()", engine, cuda_uplo, N, convert(AP), cconvert(A), lda);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS0
 * \brief Wrapper to GPU-BLAS matrix addition Xttttp(), converts triangular-regular to triangular-packed formats.
 *
 * Takes a triangular matrix and converts it into packed format.
 * - \b uplo is lower/upper case L/U indicating whether to use the lower (L) or upper (U) part of \b A
 * - \b N is the number of rows and columns of the matrix
 * - \b A is a matrix \b lda by \b N, minimum \b lda is \b N
 * - \b AP will be resized and overwritten with the N (N+1) / 2 entries holding the packed matrix
 * See also the inverse operation hala::tp2tr().
 *
 * Overloads:
 * \code
 *  tr2tp(engine, uplo, N, A, lda, AP);
 *  tr2tp(engine, uplo, N, A, AP, lda = -1);
 * \endcode
 * The \b lda will assume the minimum value.
 */
template<class VectorLikeA, class VectorLikeAP>
void tr2tp(gpu_engine const &engine, char uplo, int N, VectorLikeA const &A, int lda, VectorLikeAP &&AP){
    check_types(A, AP);
    assert( lda >= N );
    assert( check_size(A, lda, N) );
    assert( check_uplo(uplo) );
    check_set_size(assume_output, AP, (N * (N+1)) / 2);

    #ifdef HALA_ENABLE_CUDA
    auto cuda_uplo = uplo_to_gpu(uplo);

    using scalar_type = get_scalar_type<VectorLikeA>;

    cuda_call_backend<scalar_type>(cublasStrttp, cublasDtrttp, cublasCtrttp, cublasZtrttp,
        "GPU-BLAS::Xtrttp()", engine, cuda_uplo, N, convert(A), lda, cconvert(AP));
    #endif
}

}

#endif
