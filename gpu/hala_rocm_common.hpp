#ifndef __HALA_ROCM_COMMON_HPP
#define __HALA_ROCM_COMMON_HPP
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

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>

#endif // __HALA_DOXYGEN_SKIP

#include "hala_input_checker.hpp"

/*!
 * \internal
 * \file hala_rocm_common.hpp
 * \brief HALA common ROCM templates and the ROCM headers.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAROCM
 *
 * \endinternal
*/

/*!
 * \internal
 * \ingroup HALAROCM
 * \addtogroup HALAROCMCOMMON Helper Templates
 *
 * Helper templates used in many rocm methods.
 * \endinternal
 */


namespace hala{

/*!
 * \ingroup HALACTYPES
 * \brief CUDA libraries use types cuComplex for single precision complex numbers.
 */
template<> struct is_fcomplex<rocblas_float_complex> : std::true_type{};

/*!
 * \ingroup HALACTYPES
 * \brief CUDA libraries use types cuDoubleComplex for double precision complex numbers.
 */
template<> struct is_dcomplex<rocblas_double_complex> : std::true_type{};

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Converts rocblas status into a human readable string.
 */
inline std::string info_from(rocblas_status status){
    switch(status){
        case rocblas_status_success : return "success";
        case rocblas_status_invalid_handle : return "invlida handle";
        case rocblas_status_not_implemented : return "not implemented";
        case rocblas_status_invalid_pointer : return "invalid pointer";
        case rocblas_status_invalid_size : return "invalid size";
        case rocblas_status_memory_error : return "memory error";
        case rocblas_status_internal_error : return "internal error";
        case rocblas_status_perf_degraded : return "performance degraded due to low memory";
        case rocblas_status_size_query_mismatch : return "unmatched start/stop size query";
        case rocblas_status_size_increased : return "queried device memory size increased";
        case rocblas_status_size_unchanged : return "queried device memory size unchanged";
        case rocblas_status_invalid_value : return "passed argument not valid";
        case rocblas_status_continue : return "nothing preventing function to proceed";
        default:
            return "unknown";
    }
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Check if cuda returns cudaSuccess; if not, throw \b runtime_error and append \b message to the cuda error info.
 *
 * Overloads are provided for cuBlas and cuSparse status messages.
 */
inline void check_rocm(rocblas_status status, const char *function_name){
    if (status != rocblas_status_success)
        throw std::runtime_error(std::string("rocblas encountered an error with code: ") + info_from(status) + ", at " + function_name);
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Check AMD HIP status.
 */
inline void check_rocm(hipError_t status, const char *function_name){
   if (status != hipSuccess)
       throw std::runtime_error(std::string("rocm-hip encountered an error with code: ") + std::to_string(status) + " at " + function_name);
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Check AMD HIP status.
 */
inline void check_rocm(rocsparse_status status, const char *function_name){
   if (status != rocsparse_status_success)
       throw std::runtime_error(std::string("rocsparse encountered an error with code: ") + std::to_string(status) + " at " + function_name);
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Very similar to hala::call_backend(), wraps all calls with check_rocm and accepts an engine and string with function name.
 */
template<typename criteria, typename Fsingle, typename Fdouble, typename Fcomplex, typename Fzomplex, class Engine, class... Inputs>
inline void rocm_call_backend(Fsingle &sfunc, Fdouble &dfunc, Fcomplex &cfunc, Fzomplex &zfunc, const char *function_name,
                              Engine const &engine, Inputs const &...args){
    if __HALA_CONSTEXPR_IF__ (is_float<criteria>::value){
        check_rocm( sfunc(engine, args...), function_name );
    }else if __HALA_CONSTEXPR_IF__ (is_double<criteria>::value){
        check_rocm( dfunc(engine, args...), function_name );
    }else if __HALA_CONSTEXPR_IF__ (is_fcomplex<criteria>::value){
        check_rocm( cfunc(engine, args...), function_name );
    }else if __HALA_CONSTEXPR_IF__ (is_dcomplex<criteria>::value){
        check_rocm( zfunc(engine, args...), function_name );
    }
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Very similar to hala::call_backend(), wraps all calls with check_rocm and accepts an engine and string with function name.
 */
template<bool flag, typename criteria, typename Fsingle, typename Fdouble, typename Fcomplex, typename Fzomplex,
         typename Frcomplex, typename Frzomplex, class Engine, class... Inputs>
inline void rocm_call_backend6(Fsingle &sfunc, Fdouble &dfunc, Fcomplex &cfunc, Fzomplex &zfunc, Frcomplex &crfunc, Frzomplex &zrfunc,
                               const char *function_name, Engine const &engine, Inputs const &...args){
    if __HALA_CONSTEXPR_IF__ (flag)
        return rocm_call_backend<criteria>(sfunc, dfunc, cfunc, zfunc, function_name, engine, args...);
    else
        return rocm_call_backend<criteria>(sfunc, dfunc, crfunc, zrfunc, function_name, engine, args...);
}

#ifndef __HALA_DOXYGEN_SKIP
inline int gpu_device_count(){
    int gpu_count = 0;
    check_rocm(hipGetDeviceCount(&gpu_count), "hala::gpu_device_count()");
    return gpu_count;
}
#endif


#ifndef __HALA_DOXYGEN_SKIP
// skip the placeholders that will be removed later
using rocm_place_holder = int;
inline void rocm_ignore(rocm_place_holder){}
inline rocm_place_holder trans_to_cuda_sparse(char trans){
    return (is_n(trans)) ? 0 :
                ((is_c(trans)) ? 1 :
                    2);
}
#endif

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Converts characters N, T, C (upper or lower) to the correct rocblas_operation.
 */
inline rocblas_operation trans_to_gpu(char trans){
    return ((trans == 'N') || (trans == 'n')) ? rocblas_operation_none :
            ((trans == 'T') || (trans == 't')) ? rocblas_operation_transpose : rocblas_operation_conjugate_transpose;
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Converts trans character to one compatible with rocsparse hermitian operations.
 *
 * Several Hermitian operations, e.g., hala::syrk(), use a trans character to indicate
 * a side of an operation and not necessarily the conjugate vs. non-conjugate component of the transpose.
 * But this hinters writing generic code, since different character has to be passed in
 * based on the scalar-type. This method corrects for that.
 */
template<bool conjugate, typename scalar_type>
rocblas_operation hermitian_trans(char &trans){
    if __HALA_CONSTEXPR_IF__ (conjugate and is_complex<scalar_type>::value){
        return (is_n(trans)) ? rocblas_operation_none : rocblas_operation_conjugate_transpose;
    }else{
        return (is_n(trans)) ? rocblas_operation_none : rocblas_operation_transpose;
    }
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Converts characters L, R (upper or lower) to the correct rocblas_side.
 */
inline rocblas_side side_to_gpu(char side){
    return ((side == 'L') || (side == 'l')) ? rocblas_side_left : rocblas_side_right;
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Converts characters U, L (upper or lower) to the correct rocblas_fill.
 */
inline rocblas_fill uplo_to_gpu(char uplo){
    return ((uplo == 'U') || (uplo == 'u')) ? rocblas_fill_upper : rocblas_fill_lower;
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Converts characters U, N (upper or lower) to the correct rocblas_diagonal.
 */
inline rocblas_diagonal diag_to_gpu(char diag){
    return ((diag == 'U') || (diag == 'u')) ? rocblas_diagonal_unit : rocblas_diagonal_non_unit;
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Converts characters N, T, C (upper or lower) to the correct rocsparse_operation.
 */
template<typename scalar_type>
inline rocsparse_operation trans_to_rocm_sparse(char trans){
    if __HALA_CONSTEXPR_IF__ (is_complex<scalar_type>::value){
        return ((trans == 'N') || (trans == 'n')) ? rocsparse_operation_none :
                ((trans == 'T') || (trans == 't')) ? rocsparse_operation_transpose : rocsparse_operation_conjugate_transpose;
    }else{
        return ((trans == 'N') || (trans == 'n')) ? rocsparse_operation_none : rocsparse_operation_transpose;
    }
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Allocate memory for the given number of elements on the specified device.
 */
template<typename T>
T* gpu_allocate(int gpu_device, size_t num_elements){
    T* gpu_data;
    check_rocm(hipSetDevice(gpu_device), "hala::gpu_allocate() set device");
    check_rocm(hipMalloc((void**) &gpu_data, num_elements * sizeof(T)), "hala::gpu_allocate()");
    return gpu_data;
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Free the allocated GPU memory.
 */
template<typename T>
void gpu_free(T *gpu_data){
    if (gpu_data != nullptr)
        check_rocm(hipFree(gpu_data), "hala::gpu_free()");
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Deleter for the ROCm handles.
 */
struct rocm_deleter{
    //! \brief Initialize the deleter with the ownership.
    rocm_deleter(ptr_ownership const set_ownership) : ownership(set_ownership){}
    //! \brief Deleter rocBlas handle.
    void delete_handle(rocblas_handle h){ check_rocm( rocblas_destroy_handle(h), "rocblas_destroy_handle()" ); }
    //! \brief Deleter rocSparse handle.
    void delete_handle(rocsparse_handle h){ check_rocm( rocsparse_destroy_handle(h), "rocsparse_destroy_handle()" ); }
    //! \brief Deleter rocsparse_mat_descr handle.
    void delete_handle(rocsparse_mat_descr h){ check_rocm( rocsparse_destroy_mat_descr(h), "rocsparse_destroy_mat_descr()" ); }
    //! \brief Deleter rocsparse_mat_info handle.
    void delete_handle(rocsparse_mat_info h){ check_rocm( rocsparse_destroy_mat_info(h), "rocsparse_destroy_mat_info()" ); }
    //! \brief Delete a pointer, but only if owned.
    template<typename T>
    void operator() (T* p){
        if (ownership == ptr_ownership::own){ delete_handle(p); }
    }
private:
    //! \brief Remember the ownership of the handle.
    ptr_ownership ownership;
};

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Wraps a pointer handle into a std::unique_ptr variable with the hala::rocm_deleter in own mode.
 */
template<typename handle>
inline auto rocm_unique_ptr(handle h){
    return std::unique_ptr<typename std::remove_pointer<handle>::type, rocm_deleter>(h, rocm_deleter(ptr_ownership::own));
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Identical to std::copy_n() but works on cpu-gpu or gpu-gpu pair of arrays.
 */
template<copy_direction dir, typename T>
void gpu_copy_n(T const *source, size_t num_entries, T *destination){
    if __HALA_CONSTEXPR_IF__ (dir == copy_direction::host2device){
        check_rocm(hipMemcpy(destination, source, num_entries * sizeof(T), hipMemcpyHostToDevice), "hala::gpu_copy_n(hipMemcpyHostToDevice)");
    }else if __HALA_CONSTEXPR_IF__ (dir == copy_direction::device2device){
        check_rocm(hipMemcpy(destination, source, num_entries * sizeof(T), hipMemcpyDeviceToDevice), "hala::gpu_copy_n(hipMemcpyDeviceToDevice)");
    }else if __HALA_CONSTEXPR_IF__ (dir == copy_direction::device2host){
        check_rocm(hipMemcpy(destination, source, num_entries * sizeof(T), hipMemcpyDeviceToHost), "hala::gpu_copy_n(hipMemcpyDeviceToHost)");
    }
}

}

#endif
