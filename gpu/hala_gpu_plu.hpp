#ifndef __HALA_GPU_SPARSE_PLU_HPP
#define __HALA_GPU_SPARSE_PLU_HPP
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
 * \file hala_cuda_plu.hpp
 * \brief HALA wrappers to cuSolver methods.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACUDASOLVERPLU
 *
 * \endinternal
*/

#include "hala_gpu_ilu.hpp"

namespace hala{

#ifdef HALA_ENABLE_CUDA

/*!
 * \ingroup HALACUDA
 * \addtogroup HALACUDASOLVERPLU cuSolver PLU
 *
 * Wrappers to cuSolverDn methods for Cholesky and PLU factorization.
 */

/*!
 * \internal
 * \ingroup HALALAPACK
 * \brief If \b info is not zero, throws a runtime error including the \b name and \b info in the description.
 *
 * \endinternal
 */
inline void lapack_check(gpu_vector<int> const &info, const char *name){
    if (info.unload()[0] != 0)
        throw std::runtime_error(std::string(name) + " returned error code: " + std::to_string(info.unload()[0]));
}

/*!
 * \ingroup HALALAPACKPLU
 * \brief Returns the size of the workspace used by the potrf() method.
 */
template<class VectorLikeA>
size_t potrf_size(gpu_engine const &engine, char uplo, int N, VectorLikeA const &A, int lda = -1){
    valid::default_ld(N, lda);
    check_types(A);
    engine.check_gpu(A);
    assert( valid::potrf(uplo, N, A, lda) );

    auto gpu_uplo  = uplo_to_gpu(uplo);

    using scalar_type = get_scalar_type<VectorLikeA>;

    int size = 0;
    cuda_call_backend<scalar_type>(cusolverDnSpotrf_bufferSize, cusolverDnDpotrf_bufferSize,
                                   cusolverDnCpotrf_bufferSize, cusolverDnZpotrf_bufferSize,
                      "CUDA-SOLVER::potrf_size()", engine, gpu_uplo, N, convert(A), lda, &size);

    return static_cast<size_t>(size);
}

/*!
 * \ingroup HALALAPACKPLU
 * \brief Wrapper to LAPACK Cholesky decomposition of an s.p.d. matrix xpotrf()
 */
template<class VectorLikeA, class VectorLikeW>
inline void potrf(gpu_engine const &engine, char uplo, int N, VectorLikeA &&A, int lda, VectorLikeW &&W){
    valid::default_ld(N, lda);
    check_types(A, W);
    engine.check_gpu(A);
    assert( valid::potrf(uplo, N, A, lda) );
    int lwork = static_cast<int>(potrf_size(engine, uplo, N, A, lda));
    check_set_size(assume_output, W, lwork);

    auto gpu_uplo  = uplo_to_gpu(uplo);

    using scalar_type = get_scalar_type<VectorLikeA>;

    gpu_vector<int> info = engine.load(std::vector<int>(1, 0));
    cuda_call_backend<scalar_type>(cusolverDnSpotrf, cusolverDnDpotrf, cusolverDnCpotrf, cusolverDnZpotrf,
                      "CUDA-SOLVER::potrf()", engine, gpu_uplo, N, cconvert(A), lda, cconvert(W), lwork, info.data());

    lapack_check(info, "hala::gpu::potrf()");
}

/*!
 * \ingroup HALALAPACKPLU
 * \brief Wrapper to LAPACK solver using Cholesky decomposition of an s.p.d. matrix xpotrs()
 */
template<class VectorLikeA, class VectorLikeB>
inline void potrs(gpu_engine const &engine, char uplo, int N, int nrhs, VectorLikeA const &A, int lda, VectorLikeB &&B, int ldb){
    check_types(A, B);
    engine.check_gpu(A, B);
    assert( valid::potrs(uplo, N, nrhs, A, lda, B, ldb) );

    auto gpu_uplo  = uplo_to_gpu(uplo);

    using scalar_type = get_scalar_type<VectorLikeA>;

    gpu_vector<int> info = engine.load(std::vector<int>(1, 0));
    cuda_call_backend<scalar_type>(cusolverDnSpotrs, cusolverDnDpotrs, cusolverDnCpotrs, cusolverDnZpotrs,
                      "CUDA-SOLVER::potrs()", engine, gpu_uplo, N, nrhs, convert(A), lda, cconvert(B), ldb, info.data());

    lapack_check(info, "hala::gpu::potrs()");
}

/*!
 * \ingroup HALALAPACKPLU
 * \brief Returns the size of the workspace used by the getrf() method.
 */
template<class VectorLikeA>
size_t getrf_size(gpu_engine const &engine, int M, int N, VectorLikeA const &A, int lda = -1){
    valid::default_ld(M, lda);
    check_types(A);
    engine.check_gpu(A);
    assert( valid::getrf(M, N, A, lda) );

    using scalar_type = get_scalar_type<VectorLikeA>;

    int size = 0;
    cuda_call_backend<scalar_type>(cusolverDnSgetrf_bufferSize, cusolverDnDgetrf_bufferSize,
                                   cusolverDnCgetrf_bufferSize, cusolverDnZgetrf_bufferSize,
                      "CUDA-SOLVER::potrf_size()", engine, M, N, convert(A), lda, &size);

    return static_cast<size_t>(size);
}

/*!
 * \ingroup HALALAPACKPLU
 * \brief Wrapper to LAPACK general PLU factorization xgetrf()
 */
template<class VectorLikeA, class VectorLikeI, class VectorLikeW>
inline void getrf(gpu_engine const &engine, int M, int N, VectorLikeA &&A, int lda, VectorLikeI &&ipiv, VectorLikeW &&W){
    check_types(A, W);
    check_types_int(ipiv);
    assert( valid::getrf(M, N, A, lda) );
    check_set_size(assume_output, ipiv, std::min(M, N));

    check_set_size(assume_output, W, getrf_size(engine, M, N, A, lda));

    using scalar_type = get_scalar_type<VectorLikeA>;

    gpu_vector<int> info = engine.load(std::vector<int>(1, 0));
    cuda_call_backend<scalar_type>(cusolverDnSgetrf, cusolverDnDgetrf, cusolverDnCgetrf, cusolverDnZgetrf,
                      "CUDA-SOLVER::getrf()", engine, M, N, cconvert(A), lda, cconvert(W), cconvert(ipiv), info.data());

    lapack_check(info, "hala::gpu::getrf()");
}

/*!
 * \ingroup HALALAPACKPLU
 * \brief Wrapper to LAPACK solver using general PLU factorization, xgetrs()
 */
template<class VectorLikeA, class VectorLikeI, class VectorLikeB>
inline void getrs(gpu_engine const &engine, char trans, int N, int nrhs, VectorLikeA const &A, int lda, VectorLikeI const &ipiv,
                  VectorLikeB &&B, int ldb){
    check_types(A, B);
    check_types_int(ipiv);
    assert( valid::getrs(trans, N, nrhs, A, lda, ipiv, B, ldb) );

    auto gpu_trans  = trans_to_gpu(trans);

    using scalar_type = get_scalar_type<VectorLikeA>;

    gpu_vector<int> info = engine.load(std::vector<int>(1, 0));
    cuda_call_backend<scalar_type>(cusolverDnSgetrs, cusolverDnDgetrs, cusolverDnCgetrs, cusolverDnZgetrs,
                                   "CUDA-SOLVER::getrs()", engine, gpu_trans, N, nrhs,
                                   convert(A), lda, convert(ipiv), cconvert(B), ldb, info.data());

    lapack_check(info, "hala::gpu::getrs()");
}



#endif

}

#endif
