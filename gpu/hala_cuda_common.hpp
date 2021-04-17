#ifndef __HALA_CUDA_COMMON_HPP
#define __HALA_CUDA_COMMON_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cublas_v2.h>
#include <cusparse.h>

#include <cusolverDn.h>
#endif // __HALA_DOXYGEN_SKIP

#include "hala_input_checker.hpp"

/*!
 * \internal
 * \file hala_cuda_common.hpp
 * \brief HALA common CUDA templates and the CUDA headers.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACUDA
 *
 * \endinternal
*/

/*!
 * \internal
 * \ingroup HALACUDA
 * \addtogroup HALACUDACOMMON Helper Templates
 *
 * Helper templates used in many cuda methods.
 * \endinternal
 */

#ifndef __HALA_CUDA_API_VERSION__
/*!
 * \ingroup HALACUDACOMMON
 * \brief The effective CUDA API version that HALA will assume.
 *
 * Normally, this should be set to CUDART_VERSION, but it is good to have for testing.
 * For example, CUDA 10 deprecates cusparseScsrmv() but still has the method, then we can force
 * __HALA_CUDA_API_VERSION__ to 9000 (i.e. CUDA 9) and test both the new and old API implementations.
 *
 */
#define __HALA_CUDA_API_VERSION__ CUDART_VERSION
#endif

namespace hala{

/*!
 * \ingroup HALACTYPES
 * \brief CUDA libraries use types cuComplex for single precision complex numbers.
 */
template<> struct is_fcomplex<cuComplex> : std::true_type{};

/*!
 * \ingroup HALACTYPES
 * \brief CUDA libraries use types cuDoubleComplex for double precision complex numbers.
 */
template<> struct is_dcomplex<cuDoubleComplex> : std::true_type{};

/*!
 * \ingroup HALACUDACOMMON
 * \brief Check if cuda returns cudaSuccess; if not, throw \b runtime_error and append \b message to the cuda error info.
 *
 * Overloads are provided for cuBlas and cuSparse status messages.
 */
inline void check_cuda(cudaError_t status, std::string const &function_name){
    if (status != cudaSuccess)
        throw std::runtime_error(function_name + " failed with message: " + cudaGetErrorString(status));
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Check cuBlas status.
 */
inline void check_cuda(cublasStatus_t cublasStatus, std::string const &function_name){
    if (cublasStatus != CUBLAS_STATUS_SUCCESS){
        std::string message = "ERROR: cuBlas failed with code: ";
        if (cublasStatus == CUBLAS_STATUS_NOT_INITIALIZED){
            message += "CUBLAS_STATUS_NOT_INITIALIZED";
        }else if (cublasStatus == CUBLAS_STATUS_ALLOC_FAILED){
            message += "CUBLAS_STATUS_ALLOC_FAILED";
        }else if (cublasStatus == CUBLAS_STATUS_INVALID_VALUE){
            message += "CUBLAS_STATUS_INVALID_VALUE";
        }else if (cublasStatus == CUBLAS_STATUS_ARCH_MISMATCH){
            message += "CUBLAS_STATUS_ARCH_MISMATCH";
        }else if (cublasStatus == CUBLAS_STATUS_MAPPING_ERROR){
            message += "CUBLAS_STATUS_MAPPING_ERROR";
        }else if (cublasStatus == CUBLAS_STATUS_EXECUTION_FAILED){
            message += "CUBLAS_STATUS_EXECUTION_FAILED";
        }else if (cublasStatus == CUBLAS_STATUS_INTERNAL_ERROR){
            message += "CUBLAS_STATUS_INTERNAL_ERROR";
        }else if (cublasStatus == CUBLAS_STATUS_NOT_SUPPORTED){
            message += "CUBLAS_STATUS_NOT_SUPPORTED";
        }else if (cublasStatus == CUBLAS_STATUS_LICENSE_ERROR){
            message += "CUBLAS_STATUS_LICENSE_ERROR";
        }else{
            message += "UNKNOWN";
        }
        throw std::runtime_error(message + " at " + function_name);
    }
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Check cuSparse status.
 */
inline void check_cuda(cusparseStatus_t cusparseStatus, std::string const &function_name){
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS){
        std::string message = "ERROR: cuSparse failed with code: ";
        if (cusparseStatus == CUSPARSE_STATUS_NOT_INITIALIZED){
            message += "CUSPARSE_STATUS_NOT_INITIALIZED";
        }else if (cusparseStatus == CUSPARSE_STATUS_ALLOC_FAILED){
            message += "CUSPARSE_STATUS_ALLOC_FAILED";
        }else if (cusparseStatus == CUSPARSE_STATUS_INVALID_VALUE){
            message += "CUSPARSE_STATUS_INVALID_VALUE";
        }else if (cusparseStatus == CUSPARSE_STATUS_ARCH_MISMATCH){
            message += "CUSPARSE_STATUS_ARCH_MISMATCH";
        }else if (cusparseStatus == CUSPARSE_STATUS_INTERNAL_ERROR){
            message += "CUSPARSE_STATUS_INTERNAL_ERROR";
        }else if (cusparseStatus == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED){
            message += "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        }else if (cusparseStatus == CUSPARSE_STATUS_EXECUTION_FAILED){
            message += "CUSPARSE_STATUS_EXECUTION_FAILED";
        }else{
            message += "UNKNOWN";
        }
        throw std::runtime_error(message + " at " + function_name);
    }
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Check cuSolver status.
 */
inline void check_cuda(cusolverStatus_t status, std::string const &function_name){
    if (status != CUSOLVER_STATUS_SUCCESS){
        std::string message = "ERROR: cuSolver failed with code " + std::to_string(status) + " at " + function_name;
        throw std::runtime_error(message);
    }
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Very similar to hala::call_backend(), wraps all calls with check_cuda and accepts an engine and string with function name.
 */
template<typename criteria, typename Fsingle, typename Fdouble, typename Fcomplex, typename Fzomplex, class Engine, class... Inputs>
inline void cuda_call_backend(Fsingle &sfunc, Fdouble &dfunc, Fcomplex &cfunc, Fzomplex &zfunc, std::string const &function_name,
                              Engine const &engine, Inputs const &...args){
    if __HALA_CONSTEXPR_IF__ (is_float<criteria>::value){
        check_cuda( sfunc(engine, args...), function_name );
    }else if __HALA_CONSTEXPR_IF__ (is_double<criteria>::value){
        check_cuda( dfunc(engine, args...), function_name );
    }else if __HALA_CONSTEXPR_IF__ (is_fcomplex<criteria>::value){
        check_cuda( cfunc(engine, args...), function_name );
    }else if __HALA_CONSTEXPR_IF__ (is_dcomplex<criteria>::value){
        check_cuda( zfunc(engine, args...), function_name );
    }
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Very similar to hala::call_backend(), wraps all calls with check_cuda and accepts an engine and string with function name.
 */
template<bool flag, typename criteria, typename Fsingle, typename Fdouble, typename Fcomplex, typename Fzomplex,
         typename Frcomplex, typename Frzomplex, class Engine, class... Inputs>
inline void cuda_call_backend6(Fsingle &sfunc, Fdouble &dfunc, Fcomplex &cfunc, Fzomplex &zfunc, Frcomplex &crfunc, Frzomplex &zrfunc,
                               std::string const &function_name, Engine const &engine, Inputs const &...args){
    if __HALA_CONSTEXPR_IF__ (flag)
        return cuda_call_backend<criteria>(sfunc, dfunc, cfunc, zfunc, function_name, engine, args...);
    else
        return cuda_call_backend<criteria>(sfunc, dfunc, crfunc, zrfunc, function_name, engine, args...);
}

/*!
 * \ingroup HALAGPU
 * \brief Wrapper to CUDA API to read the number of visible devices.
 *
 * Return the number of available CUDA Devices on the system and
 * the admissible value of \b gpuid parameter of HALA CUDA functions.
 *
 * \b NOTE: devices are indexed from 0
 */
inline int gpu_device_count(){
    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    return gpu_count;
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Converts characters N, T, C (upper or lower) to the correct cublasOperation_t.
 */
inline cublasOperation_t trans_to_gpu(char trans){
    return ((trans == 'N') || (trans == 'n')) ? CUBLAS_OP_N : ((trans == 'T') || (trans == 't')) ? CUBLAS_OP_T : CUBLAS_OP_C;
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Converts characters N, T, C (upper or lower) to the correct cusparseOperation_t.
 */
template<typename scalar_type>
inline cusparseOperation_t trans_to_cuda_sparse(char trans){
    if __HALA_CONSTEXPR_IF__ (is_complex<scalar_type>::value){
        return (is_n(trans)) ? CUSPARSE_OPERATION_NON_TRANSPOSE :
                    ((is_c(trans)) ? CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE :
                        CUSPARSE_OPERATION_TRANSPOSE);
    }else{
        return (is_n(trans) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE);
    }
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Converts characters L, R (upper or lower) to the correct cublasSideMode_t.
 */
inline cublasSideMode_t side_to_gpu(char side){
    return ((side == 'L') || (side == 'l')) ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Converts characters U, L (upper or lower) to the correct cublasFillMode_t.
 */
inline cublasFillMode_t uplo_to_gpu(char uplo){
    return ((uplo == 'U') || (uplo == 'u')) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Converts characters U, N (upper or lower) to the correct cublasDiagType_t.
 */
inline cublasDiagType_t diag_to_gpu(char diag){
    return ((diag == 'U') || (diag == 'u')) ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Allocate memory for the given number of elements on the specified device.
 */
template<typename T>
T* gpu_allocate(int gpu_device, size_t num_elements){
    T* gpu_data;
    cudaSetDevice(gpu_device);
    check_cuda(cudaMalloc((void**) &gpu_data, num_elements * sizeof(T)), "hala::gpu_allocate()");
    return gpu_data;
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Free the allocated GPU memory.
 */
template<typename T>
void gpu_free(T *gpu_data){
    if (gpu_data != nullptr)
        check_cuda(cudaFree(gpu_data), "cudaFree()");
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Identical to std::copy_n() but works on cpu-gpu or gpu-gpu pair of arrays.
 */
template<copy_direction dir, typename T>
void gpu_copy_n(T const *source, size_t num_entries, T *destination){
    if __HALA_CONSTEXPR_IF__ (dir == copy_direction::host2device){
        check_cuda(cudaMemcpy(destination, source, num_entries * sizeof(T), cudaMemcpyHostToDevice), "hala::gpu_copy_n(cudaMemcpyHostToDevice)");
    }else if __HALA_CONSTEXPR_IF__ (dir == copy_direction::device2device){
        check_cuda(cudaMemcpy(destination, source, num_entries * sizeof(T), cudaMemcpyDeviceToDevice), "hala::gpu_copy_n(cudaMemcpyDeviceToDevice)");
    }else if __HALA_CONSTEXPR_IF__ (dir == copy_direction::device2host){
        check_cuda(cudaMemcpy(destination, source, num_entries * sizeof(T), cudaMemcpyDeviceToHost), "hala::gpu_copy_n(cudaMemcpyDeviceToHost)");
    }
}

}


#endif
