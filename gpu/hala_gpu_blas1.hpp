#ifndef __HALA_GPU_BLAS1_HPP
#define __HALA_GPU_BLAS1_HPP
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
 * \file hala_gpu_blas1.hpp
 * \brief HALA wrappers to GPU-BLAS level 1 methods.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAGPUBLAS1
 *
 * \endinternal
*/

#include "hala_gpu_blas0.hpp"

namespace hala{

/*!
 * \ingroup HALAGPU
 * \addtogroup HALAGPUBLAS1 GPU-BLAS Level 1
 *
 * Wrappers to GPU-BLAS level 1 methods for vector operations.
 * The operations are identical to the BLAS templates described in \ref HALABLAS1,
 * the difference is that the accepted vectors sit on the CUDA device
 * and the first parameter of the overloads is a hala::gpu_engine with
 * active GPU-BLAS handle.
 *
 * \b Note: the device ID of the vectors and the engine should match.
 */

/*!
 * \ingroup HALAGPUBLAS1
 * \brief Wrapper to cuBLAS copy.
 *
 * Copies \b N entries of \b x into \b y using strides \b incx and \b incy.
 * The name is chosen to be distinct from standard copy().
 */
template<class VectorLikeX, class VectorLikeY>
inline void vcopy(gpu_engine const &engine, int N, VectorLikeX const &x, int incx, VectorLikeY &&y, int incy){
    check_types(x, y);
    engine.check_gpu(x, y);
    check_set_size(assume_output, y, 1 + (N - 1) * incy);
    assert( valid::vcopy(N, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeX>;

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasScopy, cublasDcopy, cublasCcopy, cublasZcopy,
                                   "GPU-BLAS::Xcopy", engine, N, convert(x), incx, convert(y), incy);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_scopy, rocblas_dcopy, rocblas_ccopy, rocblas_zcopy,
                                   "rocBlas::Xcopy", engine, N, convert(x), incx, convert(y), incy);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS1
 * \brief Overload with default parameters.
 *
 * See hala::vcopy(), only the order of inputs is different.
 */
template<class VectorLikeX, class VectorLikeY>
inline void vcopy(gpu_engine const &engine, VectorLikeX const &x, VectorLikeY &&y, int incx = 1, int incy = 1, int N = -1){
    valid::default_size(x, incx, N);
    vcopy(engine, N, x, incx, y, incy);
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to cuBLAS Xswap() methods.
 */
template<class VectorLikeX, class VectorLikeY>
inline void vswap(gpu_engine const &engine, int N, VectorLikeX &&x, int incx, VectorLikeY &&y, int incy){
    check_types(x, y);
    engine.check_gpu(x, y);
    assert( valid::vswap(N, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasSswap, cublasDswap, cublasCswap, cublasZswap, "GPU-BLAS::Xswap", engine, N, cconvert(x), incx, cconvert(y), incy);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_sswap, rocblas_dswap, rocblas_cswap, rocblas_zswap, "rocBlas::Xswap", engine, N, cconvert(x), incx, cconvert(y), incy);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS1
 * \brief Wrapper to GPU-BLAS 2-norm Xnrm2().
 */
template<class VectorLike> auto norm2(gpu_engine const &engine, int N, VectorLike const &x, int incx){
    check_types(x);
    engine.check_gpu(x);
    assert( valid::norm2(N, x, incx) );

    using scalar_type    = get_scalar_type<VectorLike>;
    using precision_type = get_precision_type<VectorLike>;

    precision_type cpu_result;
    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasSnrm2, cublasDnrm2, cublasScnrm2, cublasDznrm2,
                                   "GPU-BLAS::Xnrm2", engine, N, convert(x), incx, pconvert(&cpu_result));
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_snrm2, rocblas_dnrm2, rocblas_scnrm2, rocblas_dznrm2,
                                   "rocBlas::Xnrm2", engine, N, convert(x), incx, pconvert(&cpu_result));
    #endif

    return cpu_result;
}

/*!
 * \ingroup HALAGPUBLAS1
 * \brief Wrapper to cuBLAS vector sum of absolute values of the entry components.
 */
template<class VectorLikeX>
inline auto asum(gpu_engine const &engine, int N, VectorLikeX const &x, int incx){
    check_types(x);
    engine.check_gpu(x);
    assert( valid::norm2(N, x, incx) ); // inputs are the same

    using scalar_type    = get_scalar_type<VectorLikeX>;
    using precision_type = get_precision_type<VectorLikeX>;

    precision_type cpu_result = get_cast<precision_type>(0.0);
    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasSasum, cublasDasum, cublasScasum, cublasDzasum,
                                   "GPU-BLAS::Xasum", engine, N, convert(x), incx, pconvert(&cpu_result));
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_sasum, rocblas_dasum, rocblas_scasum, rocblas_dzasum,
                                   "rocBlas::Xasum", engine, N, convert(x), incx, pconvert(&cpu_result));
    #endif

    return cpu_result;
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to cuBLAS find index of largest vector entry, Ixamax().
 */
template<class VectorLikeX>
inline int iamax(gpu_engine const &engine, int N, VectorLikeX const &x, int incx){
    check_types(x);
    engine.check_gpu(x);
    assert( valid::norm2(N, x, incx) ); // inputs are the same

    using scalar_type = get_scalar_type<VectorLikeX>;
    int cpu_result = 0;

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasIsamax, cublasIdamax, cublasIcamax, cublasIzamax,
                                   "GPU-BLAS::Ixamax", engine, N, convert(x), incx, pconvert(&cpu_result));
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_isamax, rocblas_idamax, rocblas_icamax, rocblas_izamax,
                                   "rocBlas::Ixamax", engine, N, convert(x), incx, pconvert(&cpu_result));
    #endif

    return cpu_result - 1;
}

/*!
 * \ingroup HALAGPUBLAS1
 * \brief Wrapper to GPU-BLAS vector dot Xdot().
 */
template<bool conjugate = true, class VectorLikeX, class VectorLikeY>
auto dot(gpu_engine const &engine, int N, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy){
    check_types(x, y);
    engine.check_gpu(x, y);
    assert( valid::dot(N, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeX>;

    scalar_type cpu_result = get_cast<scalar_type>(0.0);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend6<conjugate, scalar_type>(cublasSdot, cublasDdot, cublasCdotc, cublasZdotc, cublasCdotu, cublasZdotu,
                       "GPU-BLAS::Xdot()", engine, N, convert(x), incx, convert(y), incy, pconvert(&cpu_result)) ;
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend6<conjugate, scalar_type>(rocblas_sdot, rocblas_ddot, rocblas_cdotc, rocblas_zdotc, rocblas_cdotu, rocblas_zdotu,
                       "rocBlas::Xdot()", engine, N, convert(x), incx, convert(y), incy, pconvert(&cpu_result)) ;
    #endif

    return cpu_result;
}

/*!
 * \ingroup HALAGPUBLAS1
 * \brief Wrapper to GPU-BLAS vector scale-addition Xaxpy().
 */
template<typename FS, class VectorLikeX, class VectorLikeY>
void axpy(gpu_engine const &engine, int N, FS alpha,
          VectorLikeX const &x, int incx, VectorLikeY &&y, int incy){
    check_types(x, y);
    engine.check_gpu(x, y);
    assert( valid::axpy(N, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    auto palpha = get_pointer<scalar_type>(alpha);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasSaxpy, cublasDaxpy, cublasCaxpy, cublasZaxpy,
                      "GPU-BLAS::Xaxpy()", engine, N, palpha, convert(x), incx, cconvert(y), incy);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_saxpy, rocblas_daxpy, rocblas_caxpy, rocblas_zaxpy,
                      "rocBlas::Xaxpy()", engine, N, palpha, convert(x), incx, cconvert(y), incy);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS1
 * \brief Wrapper to GPU-BLAS vector scale Xscal().
 */
template<typename FS, class VectorLike>
void scal(gpu_engine const &engine, int N, FS alpha, VectorLike &&x, int incx){
    check_types(x);
    engine.check_gpu(x);
    assert( valid::scal(N, x, incx) );

    using scalar_type = get_scalar_type<VectorLike>;
    auto palpha = get_pointer<scalar_type>(alpha);

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<scalar_type>(cublasSscal, cublasDscal, cublasCscal, cublasZscal,
                      "GPU-BLAS::Xscal()", engine, N, palpha, convert(x), incx);
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<scalar_type>(rocblas_sscal, rocblas_dscal, rocblas_cscal, rocblas_zscal,
                      "rocBlas::Xscal()", engine, N, palpha, convert(x), incx);
    #endif
}

/*!
 * \ingroup HALAGPUBLAS1
 * \brief Wrapper to BLAS generate Givens rotation, xrotg(), (using only 4 variables on the cpu, no actual gpu work is done here).
 */
template<typename T>
void rotg(gpu_engine const &engine, T *SA, T *SB, typename define_standard_precision<T>::value_type *C, T *S){
    check_types(std::vector<T>());

    #ifdef HALA_ENABLE_CUDA
    cuda_call_backend<T>(cublasSrotg, cublasDrotg, cublasCrotg, cublasZrotg,
                     "GPU-BLAS::Xrotg()", engine, pconvert(SA), pconvert(SB), pconvert(C), pconvert(S));
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_call_backend<T>(rocblas_srotg, rocblas_drotg, rocblas_crotg, rocblas_zrotg,
                     "rocBlas::Xrotg()", engine, pconvert(SA), pconvert(SB), pconvert(C), pconvert(S));
    #endif
}

/*!
 * \ingroup HALAGPUBLAS1
 * \brief Wrapper to cuBLAS apply the Givens rotation, Xrot().
 */
template<typename FC, typename FS, class VectorLikeX, class VectorLikeY>
void rot(gpu_engine const &engine, int N, VectorLikeX &x, int incx, VectorLikeY &&y, int incy, FC C, FS S){
    check_types(x, y); // make sure x and y types match
    engine.check_gpu(x, y);
    assert( valid::rot(N, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    using precision_type = get_precision_type<VectorLikeX>;

    #ifdef HALA_ENABLE_CUDA
    if __HALA_CONSTEXPR_IF__ ((is_fcomplex<scalar_type>::value || is_dcomplex<scalar_type>::value) &&
        (std::is_same<FS, int>::value || is_float<FS>::value || is_double<FS>::value)){ // mix complex and real
            auto effc = get_pointer<precision_type>(C);
            auto effs = get_pointer<precision_type>(S);
            if __HALA_CONSTEXPR_IF__ (is_fcomplex<scalar_type>::value){
                cublasCsrot(engine, N, cconvert(x), incx, cconvert(y), incy, effc, effs);
            }else{
                cublasZdrot(engine, N, cconvert(x), incx, cconvert(y), incy, effc, effs);
            }
    }else{ // scaled by the same type
        auto effc = get_pointer<precision_type>(C);
        auto effs = get_pointer<scalar_type>(S);
        cuda_call_backend<scalar_type>(cublasSrot, cublasDrot, cublasCrot, cublasZrot,
                          "GPU-BLAS::Xrot()", engine, N, cconvert(x), incx, cconvert(y), incy, effc, effs);
    }
    #endif
    #ifdef HALA_ENABLE_ROCM
    if __HALA_CONSTEXPR_IF__ ((is_fcomplex<scalar_type>::value || is_dcomplex<scalar_type>::value) &&
        (std::is_same<FS, int>::value || is_float<FS>::value || is_double<FS>::value)){ // mix complex and real
            auto effc = get_pointer<precision_type>(C);
            auto effs = get_pointer<precision_type>(S);
            if __HALA_CONSTEXPR_IF__ (is_fcomplex<scalar_type>::value){
                rocblas_csrot(engine, N, cconvert(x), incx, cconvert(y), incy, effc, effs);
            }else{
                rocblas_zdrot(engine, N, cconvert(x), incx, cconvert(y), incy, effc, effs);
            }
    }else{ // scaled by the same type
        auto effc = get_pointer<precision_type>(C);
        auto effs = get_pointer<scalar_type>(S);
        rocm_call_backend<scalar_type>(rocblas_srot, rocblas_drot, rocblas_crot, rocblas_zrot,
                          "rocBlas::Xrot()", engine, N, cconvert(x), incx, cconvert(y), incy, effc, effs);
    }
    #endif
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to cuBLAS generate modified Givens rotation.
 */
template<typename T, class VectorLike>
void rotmg(gpu_engine const &engine, T &D1, T &D2, T &X, T const &Y, VectorLike &&param){
    check_types(param);
    engine.check_gpu(param);
    static_assert(is_float<T>::value || is_double<T>::value, "Givens rotations work only with real numbers.");

    using standard_type = get_standard_type<VectorLike>;
    static_assert(is_compatible<T, standard_type>::value, "rotmg() requires that the types of all inputs (vector and scalars) match");

    assert( check_size(param, 5) );

    #ifdef HALA_ENABLE_CUDA
    if __HALA_CONSTEXPR_IF__ (is_float<T>::value){
        check_cuda( cublasSrotmg(engine, pconvert(&D1), pconvert(&D2), pconvert(&X), pconvert(&Y), cconvert(param)), "GPU-BLAS::Xrotmg()" );
    }else{
        check_cuda( cublasDrotmg(engine, pconvert(&D1), pconvert(&D2), pconvert(&X), pconvert(&Y), cconvert(param)), "GPU-BLAS::Xrotmg()" );
    }
    #endif
    #ifdef HALA_ENABLE_ROCM
    if __HALA_CONSTEXPR_IF__ (is_float<T>::value){
        check_rocm( rocblas_srotmg(engine, pconvert(&D1), pconvert(&D2), pconvert(&X), pconvert(&Y), cconvert(param)), "rocBlas::Xrotmg()" );
    }else{
        check_rocm( rocblas_drotmg(engine, pconvert(&D1), pconvert(&D2), pconvert(&X), pconvert(&Y), cconvert(param)), "rocBlas::Xrotmg()" );
    }
    #endif
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS apply the Givens rotation, Xrotm().
 */
template<class VectorLikeX, class VectorLikeY, class VectorLikeP>
void rotm(gpu_engine const &engine, int N, VectorLikeX &&x, int incx, VectorLikeY &&y, int incy, VectorLikeP const &param){
    check_types(x, y, param); // make sure x and y types match
    engine.check_gpu(x, y, param);
    using scalar_type = get_scalar_type<VectorLikeX>;
    static_assert(is_float<scalar_type>::value || is_double<scalar_type>::value, "Givens rotations work only with real numbers.");

    assert( valid::rot(N, x, incx, y, incy) ); // test is the same with the addition of param having size 5
    assert( check_size(param, 5) );

    #ifdef HALA_ENABLE_CUDA
    if __HALA_CONSTEXPR_IF__ (is_float<scalar_type>::value){
        check_cuda( cublasSrotm(engine, N, cconvert(x), incx, cconvert(y), incy, convert(param)), "GPU-BLAS::Xrotm()");
    }else{
        check_cuda( cublasDrotm(engine, N, cconvert(x), incx, cconvert(y), incy, convert(param)), "GPU-BLAS::Xrotm()");
    }
    #endif
    #ifdef HALA_ENABLE_ROCM
    if __HALA_CONSTEXPR_IF__ (is_float<scalar_type>::value){
        check_rocm( rocblas_srotm(engine, N, cconvert(x), incx, cconvert(y), incy, convert(param)), "rocBlas::Xrotm()");
    }else{
        check_rocm( rocblas_drotm(engine, N, cconvert(x), incx, cconvert(y), incy, convert(param)), "rocBlas::Xrotm()");
    }
    #endif
}

}

#endif
