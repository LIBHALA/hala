#ifndef __HALA_GPU_SPARSE_STRUCTS_HPP
#define __HALA_GPU_SPARSE_STRUCTS_HPP
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
 * \file hala_gpu_sparse_structs.hpp
 * \brief HALA wrappers to cuSparse helper wrappers.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACUDASPARSE
 *
 * \endinternal
 */

#include "hala_gpu_blas3.hpp"

namespace hala{

#if (__HALA_CUDA_API_VERSION__ >= 10000)
template<typename T>
cudaDataType_t get_cuda_dtype(){
    if __HALA_CONSTEXPR_IF__ (is_float<T>::value){
        return CUDA_R_32F;
    }else if __HALA_CONSTEXPR_IF__ (is_double<T>::value){
        return CUDA_R_64F;
    }else if __HALA_CONSTEXPR_IF__ (is_fcomplex<T>::value){
        return CUDA_C_32F;
    }else if __HALA_CONSTEXPR_IF__ (is_dcomplex<T>::value){
        return CUDA_C_64F;
    }
}
#if (__HALA_CUDA_API_VERSION__ >= 11020)
#define HALA_CUSPARSE_SPMV_ALG_DEFAULT CUSPARSE_SPMV_ALG_DEFAULT
#else
#define HALA_CUSPARSE_SPMV_ALG_DEFAULT CUSPARSE_MV_ALG_DEFAULT
#endif
#endif

/*!
 * \ingroup HALACUDACOMMON
 * \brief Wrapper around  used in sparse mat-vec and mat-mat operations.
 *
 * Accepted types are:
 * - cusparseSpMatDescr_t, cusparseDnVecDescr_t, cusparseDnMatDescr_t
 * - cusparseMatDescr_t, csrilu02Info_t, csrsv2Info_t
 */
template<typename T>
struct cuda_struct_description{
    //! \brief Construct a null description.
    cuda_struct_description() : desc(nullptr){
        static_assert(std::is_same<T, cusparseMatDescr_t>::value ||
                      std::is_same<T, csrilu02Info_t>::value ||
                      std::is_same<T, csrsv2Info_t>::value
                      #if (__HALA_CUDA_API_VERSION__ >= 10000)
                      || std::is_same<T, cusparseSpMatDescr_t>::value
                      || std::is_same<T, cusparseDnVecDescr_t>::value
                      || std::is_same<T, cusparseDnMatDescr_t>::value
                      #endif
                      , "cuda pointer holder with incompatible type");
        if __HALA_CONSTEXPR_IF__ (std::is_same<T, csrilu02Info_t>::value){
            cusparseCreateCsrilu02Info(pconvert(&desc));
        }else if __HALA_CONSTEXPR_IF__ (std::is_same<T, csrsv2Info_t>::value){
            cusparseCreateCsrsv2Info(pconvert(&desc));
        }
    };
    //! \brief Destructor, deletes the description.
    ~cuda_struct_description(){
        if __HALA_CONSTEXPR_IF__ (std::is_same<T, cusparseMatDescr_t>::value){
            if (desc != nullptr) cusparseDestroyMatDescr(pconvert(desc));
        }else if __HALA_CONSTEXPR_IF__ (std::is_same<T, csrilu02Info_t>::value){
            if (desc != nullptr) cusparseDestroyCsrilu02Info(pconvert(desc));
        }else if __HALA_CONSTEXPR_IF__ (std::is_same<T, csrsv2Info_t>::value){
            if (desc != nullptr) cusparseDestroyCsrsv2Info(pconvert(desc));
        }
        #if (__HALA_CUDA_API_VERSION__ >= 10000)
        else if __HALA_CONSTEXPR_IF__ (std::is_same<T, cusparseSpMatDescr_t>::value){
            if (desc != nullptr) cusparseDestroySpMat(pconvert(desc));
        }else if __HALA_CONSTEXPR_IF__ (std::is_same<T, cusparseDnVecDescr_t>::value){
            if (desc != nullptr) cusparseDestroyDnVec(pconvert(desc));
        }else if __HALA_CONSTEXPR_IF__ (std::is_same<T, cusparseDnMatDescr_t>::value){
            if (desc != nullptr) cusparseDestroyDnMat(pconvert(desc));
        }
        #endif
    }

    //! \brief Wrappers to pointer aliases cannot be copied.
    cuda_struct_description(cuda_struct_description<T> const &other) = delete;
    //! \brief Wrappers to pointer aliases cannot be copied.
    cuda_struct_description<T>& operator = (cuda_struct_description<T> const &other) = delete;

    //! \brief Trivially move the description.
    cuda_struct_description(cuda_struct_description<T> &&other) : desc(std::exchange(other.desc, nullptr)){}
    //! \brief Trivially move the description.
    cuda_struct_description<T>& operator = (cuda_struct_description<T> &&other){
        cuda_struct_description<T> temp(std::move(other));
        std::swap(desc, temp.desc);
        return *this;
    }

    //! \brief Returns the description, automatic conversion.
    operator T () const{ return desc; }
    //! \brief Holds the description pointer.
    T desc;
};

/*!
 * \ingroup HALACUDACOMMON
 * \brief Creates sparse matrix description.
 *
 * Note that this is internal API and doesn't come with assert() checks,
 * the uplo and diag can be other letters in which case they will not be set.
 * - \b type is 'G' for general or anything else for triangular, e.g., 'T'
 * - \b uplo sets upper or lower fill mode, but can be something else to skip the mode, e.g., 'S'
 * - \b diag sets unit/non-unit diagonal, but can be something else to skip the mode, e.g., 'S'
 */
inline auto make_cuda_mat_description(char type, char uplo, char diag){
    cuda_struct_description<cusparseMatDescr_t> result;
    check_cuda( cusparseCreateMatDescr(&result.desc), "cuSparse::CreateMatDescr()");
    cusparseSetMatType(result.desc, (type == 'G') ? CUSPARSE_MATRIX_TYPE_GENERAL : CUSPARSE_MATRIX_TYPE_TRIANGULAR);
    cusparseSetMatIndexBase(result.desc, CUSPARSE_INDEX_BASE_ZERO);
    if (check_diag(diag))
        cusparseSetMatDiagType(result.desc, (is_n(diag)) ? CUSPARSE_DIAG_TYPE_NON_UNIT : CUSPARSE_DIAG_TYPE_UNIT);
    if (check_uplo(uplo))
        cusparseSetMatFillMode(result.desc, (is_l(uplo)) ? CUSPARSE_FILL_MODE_LOWER : CUSPARSE_FILL_MODE_UPPER);
    return result;
}

#if (__HALA_CUDA_API_VERSION__ >= 10000)
/*!
 * \ingroup HALACUDACOMMON
 * \brief Create sparse matrix description, using 32bit and zero based indexing, and row-compressed format.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
inline auto make_cuda_mat_description(int rows, int cols, int nnz,
                                      VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){

    using scalar_type = get_scalar_type<VectorLikeV>;

    cudaDataType_t cuda_type = get_cuda_dtype<scalar_type>();

    void *upntr = const_cast<void*>(reinterpret_cast<void const*>(get_data(pntr)));
    void *uindx = const_cast<void*>(reinterpret_cast<void const*>(get_data(indx)));
    void *uvals = const_cast<void*>(reinterpret_cast<void const*>(get_data(vals)));

    cuda_struct_description<cusparseSpMatDescr_t> result;
    check_cuda(cusparseCreateCsr(&result.desc, (int64_t) rows, (int64_t) cols, (int64_t) nnz, upntr, uindx, uvals,
                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuda_type),
               "cuSparse::CreateCsr()");

    return result;
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Create dense vector description using the \b effective_size to avoid the max-int when working with pointers.
 */
template<class VectorLike>
inline auto make_cuda_dvec_description(VectorLike const &x, int effective_size){

    using scalar_type = get_scalar_type<VectorLike>;

    cudaDataType_t cuda_type = get_cuda_dtype<scalar_type>();

    void *ux = const_cast<void*>(reinterpret_cast<void const*>(get_data(x)));

    cuda_struct_description<cusparseDnVecDescr_t> result;
    cusparseStatus_t status = cusparseCreateDnVec(&result.desc, (int64_t) effective_size, ux, cuda_type);

    check_cuda(status, "cuSparse::CreateDnVec()");

    return result;
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Create dense matrix description using columns major format.
 */
template<class VectorLike>
inline auto make_cuda_dmat_description(VectorLike const &A, int rows, int cols, int lda){

    using scalar_type = get_scalar_type<VectorLike>;

    cudaDataType_t cuda_type = get_cuda_dtype<scalar_type>();

    void *ua = const_cast<void*>(reinterpret_cast<void const*>(get_data(A)));

    cuda_struct_description<cusparseDnMatDescr_t> result;
    check_cuda(cusparseCreateDnMat(&result.desc, (int64_t) rows, (int64_t) cols, (int64_t) lda, ua, cuda_type, CUSPARSE_ORDER_COL), "cuSparse::CreateDnMat()");

    return result;
}

#endif

/*!
 * \ingroup HALAGPU
 * \addtogroup HALAGPUSPARSE cuSparse Methods
 *
 * Wrappers to cuSparse methods for vector operations.
 */

/*!
 * \ingroup HALAGPU
 * \brief Wrapper around sparse matrix data associated with the GPU device.
 *
 * Takes non-owning references to a hala::gpu_engine and the triple (pntr, indx, vals) describing a sparse matrix.
 * The struct wraps around meta-data used by some versions of sparse GPU APIs and caches the meta values.
 * Using the hala::gpu_sparse_matrix class is more efficient when the same matrix will be used repeatedly
 * in multiple calls to matrix-vector or matrix-matrix multiplies.
 */
template<typename T>
struct gpu_sparse_matrix{
public:
    //! \brief The value type of the matrix.
    using value_type = std::remove_cv_t<T>;
    //! \brief Define the underlying engine.
    using engine_type = gpu_engine;
    //! \brief Constructor set the engine and data for the matrix using the given dimensions.
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    gpu_sparse_matrix(gpu_engine const &engine, int num_rows, int num_cols, int num_nz,
                      VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals)
    : rengine(engine), rpntr(get_data(pntr)), rindx(get_data(indx)), rvals(get_standard_data(vals)),
    rows(num_rows), cols(num_cols), nnz(num_nz)
    #if (__HALA_CUDA_API_VERSION__ >= 10000)
    , desc(make_cuda_mat_description(rows, cols, nnz, pntr, indx, vals)), size_buffer_n(0), size_buffer_t(0), size_buffer_c(0)
    #endif
    {
        check_types(vals);
        check_types_int(pntr, indx);
        assert( check_size(pntr, rows+1) );
        assert( check_size(indx, nnz) );
        assert( check_size(vals, nnz) );
        engine.check_gpu(pntr, indx, vals);
    }
    //! \brief Constructor set the engine and data for the matrix, infers the number of rows and num_nz from the inputs (cannot use raw-arrays).
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    gpu_sparse_matrix(gpu_engine const &engine, int num_cols,
                      VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals)
    : rengine(engine), rpntr(get_data(pntr)), rindx(get_data(indx)), rvals(get_standard_data(vals)),
    rows(get_size_int(pntr) - 1), cols(num_cols), nnz(get_size(indx))
    #if (__HALA_CUDA_API_VERSION__ >= 10000)
    , desc(make_cuda_mat_description(rows, cols, nnz, pntr, indx, vals)), size_buffer_n(0), size_buffer_t(0), size_buffer_c(0)
    #endif
    {
        check_types(vals);
        check_types_int(pntr, indx);
        assert( check_size(indx, nnz) );
        engine.check_gpu(pntr, indx, vals);
    }
    //! \brief Using default destructor.
    ~gpu_sparse_matrix() = default;

    //! \brief Cannot copy.
    gpu_sparse_matrix(gpu_sparse_matrix const&) = delete;
    //! \brief Can move.
    gpu_sparse_matrix(gpu_sparse_matrix &&) = default;

    //! \brief Cannot copy.
    gpu_sparse_matrix& operator = (gpu_sparse_matrix const&) = delete;
    //! \brief Can move.
    gpu_sparse_matrix& operator = (gpu_sparse_matrix &&) = default;

    //! \brief Returns the associated GPU engine.
    gpu_engine const& engine() const{ return rengine; }

    #if (__HALA_CUDA_API_VERSION__ >= 10000)
    //! \brief Returns the size of the workspace (in bytes) on the GPU device required for a gemv() operation.
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
    size_t gemv_buffer_size(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y) const{
        auto cuda_trans = trans_to_cuda_sparse<value_type>(trans);
        pntr_check_set_size(beta, y, (is_n(trans)) ? rows : cols, 1);
        if (cuda_trans == CUSPARSE_OPERATION_NON_TRANSPOSE){
            if (size_buffer_n > 0) return size_buffer_n;
            get_gemv_buffer(cuda_trans, alpha, x, beta, y, &size_buffer_n);
            return size_buffer_n;
        }else if (cuda_trans == CUSPARSE_OPERATION_TRANSPOSE){
            if (size_buffer_t > 0) return size_buffer_t;
            get_gemv_buffer(cuda_trans, alpha, x, beta, y, &size_buffer_t);
            return size_buffer_t;
        }else{
            if (size_buffer_c > 0) return size_buffer_c;
            get_gemv_buffer(cuda_trans, alpha, x, beta, y, &size_buffer_c);
            return size_buffer_c;
        }
    }
    //! \brief Performs matrix vector product with external workspace buffer, see hala::sparse_gemv().
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY, class VectorLikeBuff>
    void gemv(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y, VectorLikeBuff &&buff) const{
        check_types(rvals, x, y);
        rengine.check_gpu(x, y);
        pntr_check_set_size(beta, y, (is_n(trans)) ? rows : cols, 1);

        auto cuda_trans = trans_to_cuda_sparse<value_type>(trans);
        auto palpha = get_pointer<value_type>(alpha);
        auto pbeta  = get_pointer<value_type>(beta);

        auto xdesc = make_cuda_dvec_description(x, (cuda_trans == CUSPARSE_OPERATION_NON_TRANSPOSE) ? cols : rows);
        auto ydesc = make_cuda_dvec_description(y, (cuda_trans == CUSPARSE_OPERATION_NON_TRANSPOSE) ? rows : cols);
        check_cuda( cusparseSpMV(rengine, cuda_trans, palpha, desc, xdesc, pbeta, ydesc, get_cuda_dtype<value_type>(), HALA_CUSPARSE_SPMV_ALG_DEFAULT, convert(buff)),
                    "cusparseSpMV()");
    }
    //! \brief Performs matrix vector product, see hala::sparse_gemv().
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
    void gemv(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y) const{
        gemv(trans, alpha, x, beta, y, gpu_vector<value_type>(gemv_buffer_size(trans, alpha, x, beta, y), rengine.device()));
    }
    //! \brief Returns the workspace size (in bytes) for a matrix-matrix product.
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
    size_t gemm_buffer_size(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &C, int ldc) const{
        check_types(rvals, B, C);
        rengine.check_gpu(B, C);
        int M = (is_n(transa)) ? rows : cols;
        int N = (is_n(transb)) ? b_cols : b_rows;
        pntr_check_set_size(beta, C, ldc, N);

        #if (__HALA_CUDA_API_VERSION__ < 10020)
        int K = (is_n(transa)) ? cols : rows;
        if (is_n(transa))
            return (is_c(transb)) ? hala_size(K, N) * sizeof(value_type) : 0;
        #endif

        cuda_struct_description<cusparseDnMatDescr_t> bdesc = make_cuda_dmat_description(B, b_rows, b_cols, ldb);
        cuda_struct_description<cusparseDnMatDescr_t> cdesc = make_cuda_dmat_description(C, M, N, ldc);
        auto palpha = get_pointer<value_type>(alpha);
        auto pbeta  = get_pointer<value_type>(beta);
        cusparseOperation_t cuda_transa = trans_to_cuda_sparse<value_type>(transa);
        cusparseOperation_t cuda_transb = trans_to_cuda_sparse<value_type>(transb);

        return get_gemm_buffer(cuda_transa, cuda_transb, palpha, bdesc, pbeta, cdesc);
    }
    //! \brief Performs matrix-matrix product with external workspace buffer, see hala::sparse_gemv().
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC, class VectorLikeT>
    void gemm(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &C, int ldc, VectorLikeT &&temp) const{
        check_types(rvals, B, C);
        rengine.check_gpu(B, C);
        int M = (is_n(transa)) ? rows : cols;
        int N = (is_n(transb)) ? b_cols : b_rows;
        int K = (is_n(transa)) ? cols : rows;
        pntr_check_set_size(beta, C, ldc, N);
        assert( valid::sparse_gemm(transa, transb, M, N, K, nnz, rpntr, rindx, rvals, B, ldb, C, ldc) );

        auto palpha = get_pointer<value_type>(alpha);
        auto pbeta  = get_pointer<value_type>(beta);

        cusparseOperation_t cuda_transa = trans_to_cuda_sparse<value_type>(transa);
        cusparseOperation_t cuda_transb = trans_to_cuda_sparse<value_type>(transb);

        #if (__HALA_CUDA_API_VERSION__ < 10020)
        // prior to 10020 cuSparse has a bug when A is used in non-transpose mode, switch to the old API
        if (is_n(transa)){
            if (is_c(transb)){ // conjugate transpose not supported for cusparseScsrmm2
                {
                    gpu_pntr<host_pntr> hold(rengine);
                    geam(rengine, 'C', 'C', K, N, 1, B, ldb, 0, B, ldb, temp, K);
                }
                cuda_transb = trans_to_cuda_sparse<value_type>('N');
                cuda_call_backend<value_type>(cusparseScsrmm2, cusparseDcsrmm2, cusparseCcsrmm2, cusparseZcsrmm2,
                                "cusparseXcsrmm2 (B-conj-transpose)", rengine, cuda_transa, cuda_transb, M, N, K, nnz,
                                palpha, rengine.general_matrix(), pconvert(rvals), rpntr, rindx,
                                convert(temp), K, pbeta, cconvert(C), ldc);

            }else{
                cuda_call_backend<value_type>(cusparseScsrmm2, cusparseDcsrmm2, cusparseCcsrmm2, cusparseZcsrmm2,
                                "cusparseXcsrmm2", rengine, cuda_transa, cuda_transb, M, N, K, nnz,
                                palpha, rengine.general_matrix(), pconvert(rvals), rpntr, rindx,
                                convert(B), ldb, pbeta, cconvert(C), ldc);
            }
        }else{
        #endif
            auto bdesc = make_cuda_dmat_description(B, b_rows, b_cols, ldb);
            auto cdesc = make_cuda_dmat_description(C, M, N, ldc);

            cudaDataType_t cuda_type = get_cuda_dtype<value_type>();

            #if (__HALA_CUDA_API_VERSION__ < 11000)
            constexpr cusparseSpMMAlg_t spmm_alg = CUSPARSE_MM_ALG_DEFAULT;
            #else
            constexpr cusparseSpMMAlg_t spmm_alg = CUSPARSE_SPMM_ALG_DEFAULT;
            #endif

            check_cuda( cusparseSpMM(rengine, cuda_transa, cuda_transb, palpha, desc, bdesc, pbeta, cdesc,
                                     cuda_type, spmm_alg, convert(temp)),
                        "cusparseSpMM()");

        #if (__HALA_CUDA_API_VERSION__ < 10020)
        } // closes the pre 10020 api work-around
        #endif
    }
    #else
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
    size_t gemv_buffer_size(char, FPa, VectorLikeX const&, FPb, VectorLikeY&) const{ return 0; }
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
    void gemv(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &y) const{
        check_types(rvals, x, y);
        rengine.check_gpu(x, y);
        pntr_check_set_size(beta, y, (is_n(trans)) ? rows : cols, 1);

        auto cuda_trans = trans_to_cuda_sparse<value_type>(trans);
        auto palpha = get_pointer<value_type>(alpha);
        auto pbeta  = get_pointer<value_type>(beta);

        cuda_call_backend<value_type>(cusparseScsrmv, cusparseDcsrmv, cusparseCcsrmv, cusparseZcsrmv,
                      "cuSparse::Xcsrmv()", rengine, cuda_trans, rows, cols, nnz, palpha, rengine.general_matrix(),
                      pconvert(rvals), rpntr, rindx, convert(x), pbeta, convert(y));
    }
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY, class VectorLikeBuff>
    void gemv(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y, VectorLikeBuff &&) const{
        gemv(trans, alpha, x, beta, y);
    }
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
    size_t gemm_buffer_size(char transa, char transb, int b_rows, int b_cols, FSA, VectorLikeB const&, int, FSB, VectorLikeC &, int) const{
        int N = (is_n(transb)) ? b_cols : b_rows;
        int K = (is_n(transa)) ? cols : rows;
        if (is_n(transa))
            return (is_c(transb)) ? hala_size(N, K) * sizeof(value_type) : 0;
        else
            return (is_n(transb)) ? 0 : hala_size(N, K) * sizeof(value_type);
    }
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC, class VectorLikeT>
    void gemm(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &C, int ldc, VectorLikeT &&temp) const{
        check_types(rvals, B, C);
        rengine.check_gpu(B, C);
        int M = (is_n(transa)) ? rows : cols;
        int N = (is_n(transb)) ? b_cols : b_rows;
        int K = (is_n(transa)) ? cols : rows;
        pntr_check_set_size(beta, C, ldc, N );
        assert( valid::sparse_gemm(transa, transb, M, N, K, nnz, rpntr, rindx, rvals, B, ldb, C, ldc) );

        auto palpha = get_pointer<value_type>(alpha);
        auto pbeta  = get_pointer<value_type>(beta);

        cusparseOperation_t cuda_transa = trans_to_cuda_sparse<value_type>(transa);
        cusparseOperation_t cuda_transb = trans_to_cuda_sparse<value_type>(transb);

        if (is_n(transa)){
            if (is_c(transb)){ // conjugate transpose not supported for cusparseScsrmm2
                {
                    gpu_pntr<host_pntr> hold(rengine);
                    geam(rengine, 'C', 'C', K, N, 1, B, ldb, 0, B, ldb, temp, K);
                }
                cuda_transb = trans_to_cuda_sparse<value_type>('N');
                cuda_call_backend<value_type>(cusparseScsrmm2, cusparseDcsrmm2, cusparseCcsrmm2, cusparseZcsrmm2,
                                "cuSparse::Xcsrmm2() (B-conj-transpose)", rengine, cuda_transa, cuda_transb, M, N, K, nnz,
                                palpha, rengine.general_matrix(), pconvert(rvals), rpntr, rindx,
                                convert(temp), K, pbeta, cconvert(C), ldc);

            }else{
                cuda_call_backend<value_type>(cusparseScsrmm2, cusparseDcsrmm2, cusparseCcsrmm2, cusparseZcsrmm2,
                                "cuSparse::Xcsrmm2()", rengine, cuda_transa, cuda_transb, M, N, K, nnz,
                                palpha, rengine.general_matrix(), pconvert(rvals), rpntr, rindx,
                                convert(B), ldb, pbeta, cconvert(C), ldc);
            }
        }else{
            if (is_n(transb)){
                cuda_call_backend<value_type>(cusparseScsrmm, cusparseDcsrmm, cusparseCcsrmm, cusparseZcsrmm,
                                "cuSparse::Xcsrmm() (B-non-transpose)", rengine, cuda_transa, K, N, M, nnz,
                                palpha, rengine.general_matrix(), pconvert(rvals), rpntr, rindx,
                                convert(B), ldb, pbeta, cconvert(C), ldc);
            }else{ // manual transpose needed
                {
                    gpu_pntr<host_pntr> hold(rengine);
                    geam(rengine, transb, transb, K, N, 1, B, ldb, 0, B, ldb, temp, K);
                }
                cuda_call_backend<value_type>(cusparseScsrmm, cusparseDcsrmm, cusparseCcsrmm, cusparseZcsrmm,
                                "cuSparse::Xcsrmm() (B-transposed)", rengine, cuda_transa, K, N, M, nnz,
                                palpha, rengine.general_matrix(), pconvert(rvals), rpntr, rindx,
                                convert(temp), K, pbeta, cconvert(C), ldc);
            }
        }
    }
    #endif
    //! \brief Performs matrix-matrix product, see hala::sparse_gemv().
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
    void gemm(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &C, int ldc) const{
        size_t work = gemm_buffer_size(transa, transb, b_rows, b_cols, alpha, B, ldb, beta, C, ldc);
        gemm(transa, transb, b_rows, b_cols, alpha, B, ldb, beta, C, ldc, gpu_vector<value_type>(work / sizeof(value_type) + 1, rengine.device()));
    }

protected:
    #if (__HALA_CUDA_API_VERSION__ >= 10000)
    //! \brief Helper wrapper around cusparseSpMV_bufferSize().
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
    void get_gemv_buffer(cusparseOperation_t trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &y, size_t *size) const{
        auto xdesc = make_cuda_dvec_description(x, (trans == CUSPARSE_OPERATION_NON_TRANSPOSE) ? cols : rows);
        auto ydesc = make_cuda_dvec_description(y, (trans == CUSPARSE_OPERATION_NON_TRANSPOSE) ? rows : cols);

        auto palpha = get_pointer<value_type>(alpha);
        auto pbeta  = get_pointer<value_type>(beta);
        check_cuda(cusparseSpMV_bufferSize(rengine, trans, palpha, desc, xdesc, pbeta, ydesc, get_cuda_dtype<value_type>(), HALA_CUSPARSE_SPMV_ALG_DEFAULT, size),
                   "cusparseSpMV_bufferSize");
    }
    //! \brief Helper wrapper around cusparseSpMM_bufferSize().
    template<typename FSA, typename FSB>
    size_t get_gemm_buffer(cusparseOperation_t transa, cusparseOperation_t transb,
                           FSA alpha, cuda_struct_description<cusparseDnMatDescr_t> const &bdesc,
                           FSB beta, cuda_struct_description<cusparseDnMatDescr_t> const &cdesc) const{
        #if (__HALA_CUDA_API_VERSION__ < 11000)
        constexpr cusparseSpMMAlg_t spmm_alg = CUSPARSE_MM_ALG_DEFAULT;
        #else
        constexpr cusparseSpMMAlg_t spmm_alg = CUSPARSE_SPMM_ALG_DEFAULT;
        #endif
        size_t bsize = 0;
        check_cuda( cusparseSpMM_bufferSize(rengine, transa, transb, alpha, desc, bdesc, beta, cdesc, get_cuda_dtype<value_type>(), spmm_alg, &bsize),
                    "cusparseSpMM_bufferSize()");
        return bsize;
    }
    #endif

private:
    gpu_engine rengine;
    int const *rpntr, *rindx;
    T const* rvals;
    int rows, cols, nnz;
    #if (__HALA_CUDA_API_VERSION__ >= 10000)
    cuda_struct_description<cusparseSpMatDescr_t> desc;
    mutable size_t size_buffer_n, size_buffer_t, size_buffer_c;
    #endif
};

/*!
 * \brief Factory method, creates a hala::gpu_sparse_matrix wrapper with the given matrix parameters.
 *
 * This overload can be used with raw-array data.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_sparse_matrix(gpu_engine const &engine, int num_rows, int num_cols, int num_nz,
                        VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    check_types(vals);
    check_types_int(pntr, indx);
    using scalar_type = get_scalar_type<VectorLikeV>;
    return gpu_sparse_matrix<scalar_type>(engine, num_rows, num_cols, num_nz, pntr, indx, vals);
}
/*!
 * \brief Factory method, creates a hala::gpu_sparse_matrix wrapper where most of the dimensions are inferred from the vector sizes.
 *
 * This overload cannot be used with raw-arrays.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_sparse_matrix(gpu_engine const &engine, int num_cols, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    check_types(vals);
    check_types_int(pntr, indx);
    using scalar_type = get_scalar_type<VectorLikeV>;
    return gpu_sparse_matrix<scalar_type>(engine, num_cols, pntr, indx, vals);
}

/*!
 * \ingroup HALAGPUSPARSE
 * \brief Sparse matrix times a dense vector, GPU version.
 */
template<typename FPa, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, typename FPb, class VectorLikeY>
void sparse_gemv(gpu_engine const &engine, char trans, int M, int N,
                 FPa alpha, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, VectorLikeX const &x,
                 FPb beta, VectorLikeY &y){
    check_types(vals, x, y);
    check_types_int(pntr, indx);
    engine.check_gpu(pntr, indx, vals, x, y);
    pntr_check_set_size(beta, y, (is_n(trans)) ? M : N, 1);
    assert( valid::sparse_gemv(trans, M, N, 0, pntr, indx, vals, x, y) ); // using 0 for the nnz, cannot check on the gpu side

    int nnz = get_size_int(indx);
    make_sparse_matrix(engine, M, N, nnz, pntr, indx, vals).gemv(trans, alpha, x, beta, y);
}

/*!
 * \ingroup HALAGPUSPARSE
 * \brief Sparse matrix-matrix multiplication, GPU version.
 */
template<typename FSA, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeB, typename FSB, class VectorLikeC>
void sparse_gemm(gpu_engine const &engine, char transa, char transb, int M, int N, int K,
                 FSA alpha, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                 VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &C, int ldc){
    check_types(vals, B, C);
    check_types_int(pntr, indx);
    engine.check_gpu(pntr, indx, vals, B, C);
    pntr_check_set_size(beta, C, ldc, N );
    int nnz = get_size_int(indx);
    assert( valid::sparse_gemm(transa, transb, M, N, K, nnz, pntr, indx, vals, B, ldb, C, ldc) );

    make_sparse_matrix(engine, (is_n(transa)) ? M : K, (is_n(transa)) ? K : M, nnz, pntr, indx, vals)
        .gemm(transa, transb, (is_n(transb)) ? K : N, (is_n(transb)) ? N : K, alpha, B, ldb, beta, C, ldc);
}

}
#endif
