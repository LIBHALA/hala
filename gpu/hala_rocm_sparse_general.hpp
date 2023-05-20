#ifndef __HALA_GPU_SPARSE_HPP
#define __HALA_GPU_SPARSE_HPP
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
 * \file hala_gpu_sparse.hpp
 * \brief HALA wrappers to cuSparse methods.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAGPUSPARSE
 *
 * \endinternal
*/

#include "hala_gpu_blas3.hpp"

namespace hala{

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Factory method for a general matrix description.
 *
 * \param diag specifies whether the matrix is uint or non-unit diagonal
 */
inline auto make_rocsparse_general_description(char diag = 'N'){
    assert( check_diag(diag) );
    rocsparse_mat_descr result;
    check_rocm( rocsparse_create_mat_descr(&result), "rocsparse_create_mat_descr()");
    rocsparse_set_mat_type(result, rocsparse_matrix_type_general);
    rocsparse_set_mat_index_base(result, rocsparse_index_base_zero);
    rocsparse_set_mat_diag_type(result, (is_n(diag)) ? rocsparse_diag_type_non_unit : rocsparse_diag_type_unit);
    return rocm_unique_ptr(result);
}

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Factory method for a general matrix info.
 */
inline auto make_rocsparse_mat_info(){
    rocsparse_mat_info result;
    check_rocm( rocsparse_create_mat_info(&result), "rocsparse_create_mat_info()");
    return rocm_unique_ptr(result);
}
/*!
 * \ingroup HALAROCMCOMMON
 * \brief Factory method for a triangular matrix description.
 *
 * \param uplo specifies whether the matrix has upper or lower fill
 * \param diag specifies whether the matrix is uint or non-unit diagonal
 */
inline auto make_rocsparse_triangular_description(char uplo, char diag){
    assert( check_uplo(uplo) );
    assert( check_diag(diag) );
    rocsparse_mat_descr result;
    check_rocm( rocsparse_create_mat_descr(&result), "rocsparse_create_mat_descr()");
    rocsparse_set_mat_type(result, rocsparse_matrix_type_general);
    rocsparse_set_mat_index_base(result, rocsparse_index_base_zero);
    rocsparse_set_mat_fill_mode(result, (is_l(uplo)) ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper);
    rocsparse_set_mat_diag_type(result, (is_n(diag)) ? rocsparse_diag_type_non_unit : rocsparse_diag_type_unit);
    return rocm_unique_ptr(result);
}

#ifndef __HALA_DOXYGEN_SKIP
template<typename T>
struct gpu_sparse_matrix{
public:
    using value_type = std::remove_cv_t<T>;
    using engine_type = gpu_engine;
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    gpu_sparse_matrix(gpu_engine const &engine, int num_rows, int num_cols, int num_nz,
                      VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals)
    : rengine(engine), rpntr(get_data(pntr)), rindx(get_data(indx)), rvals(get_standard_data(vals)),
    rows(num_rows), cols(num_cols), nnz(num_nz), desc(make_rocsparse_general_description())
    {
        check_types(vals);
        check_types_int(pntr, indx);
        assert( check_size(pntr, rows+1) );
        assert( check_size(indx, nnz) );
        assert( check_size(vals, nnz) );
        engine.check_gpu(pntr, indx, vals);
    }
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    gpu_sparse_matrix(gpu_engine const &engine, int num_cols,
                      VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals)
    : rengine(engine), rpntr(get_data(pntr)), rindx(get_data(indx)), rvals(get_standard_data(vals)),
    rows(get_size_int(pntr) - 1), cols(num_cols), nnz(get_size(indx)), desc(make_rocsparse_general_description())
    {
        check_types(vals);
        check_types_int(pntr, indx);
        assert( check_size(indx, nnz) );
        engine.check_gpu(pntr, indx, vals);
    }
    ~gpu_sparse_matrix() = default;

    gpu_sparse_matrix(gpu_sparse_matrix const&) = delete;
    gpu_sparse_matrix(gpu_sparse_matrix &&) = default;

    gpu_sparse_matrix& operator = (gpu_sparse_matrix const&) = delete;
    gpu_sparse_matrix& operator = (gpu_sparse_matrix &&) = default;

    gpu_engine const& engine() const{ return rengine; }

    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
    size_t gemv_buffer_size(char, FPa, VectorLikeX const&, FPb, VectorLikeY &&) const{ return 0; }
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY, class VectorLikeBuff>
    void gemv(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y, VectorLikeBuff &&) const{
        check_types(rvals, x, y);
        rengine.check_gpu(x, y);
        pntr_check_set_size(beta, y, (is_n(trans)) ? rows : cols, 1);

        auto rocm_trans = trans_to_rocm_sparse<value_type>(trans);
        auto palpha = get_pointer<value_type>(alpha);
        auto pbeta  = get_pointer<value_type>(beta);

        rocsparse_mat_info info = nullptr;

        rocm_call_backend<value_type>(rocsparse_scsrmv, rocsparse_dcsrmv, rocsparse_ccsrmv, rocsparse_zcsrmv,
            "csrmv()", rengine, rocm_trans, rows, cols, nnz, palpha, desc.get(), pconvert(rvals), pconvert(rpntr), pconvert(rindx), info, convert(x), pbeta, cconvert(y));
    }
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
    void gemv(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y) const{
        gemv(trans, alpha, x, beta, y, gpu_vector<value_type>(rengine.device()));
    }
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
    size_t gemm_buffer_size(char, char, int, int, FSA, VectorLikeB const&, int, FSB, VectorLikeC&, int) const{
        return 0;
    }
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC, class VectorLikeT>
    void gemm(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &C, int ldc, VectorLikeT &&) const{
        check_types(rvals, B, C);
        rengine.check_gpu(B, C);
        int M = rows;
        int N = (is_n(transb)) ? b_cols : b_rows;
        int K = cols;
        pntr_check_set_size(beta, C, ldc, N);
        // rocsparse uses different definition for M and K that is fixed on A regardless of op(A)
        // error checking uses the standard format but the functon call uses the rocblas way
        assert( valid::sparse_gemm(transa, transb, (is_n(transa)) ? rows : cols, N, (is_n(transa)) ? cols : rows, nnz, rpntr, rindx, rvals, B, ldb, C, ldc) );

        auto palpha = get_pointer<value_type>(alpha);
        auto pbeta  = get_pointer<value_type>(beta);

        auto rocm_transa = trans_to_rocm_sparse<value_type>(transa);
        auto rocm_transb = trans_to_rocm_sparse<value_type>(transb);

		rocm_call_backend<value_type>(rocsparse_scsrmm, rocsparse_dcsrmm, rocsparse_ccsrmm, rocsparse_zcsrmm,
            "csrmm()", rengine, rocm_transa, rocm_transb, M, N, K, nnz, palpha, desc.get(), convert(rvals), convert(rpntr), convert(rindx), convert(B), ldb, pbeta, cconvert(C), ldc);
    }
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
    void gemm(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &C, int ldc) const{
        gemm(transa, transb, b_rows, b_cols, alpha, B, ldb, beta, C, ldc, 0);
    }

private:
    gpu_engine rengine;
    int const *rpntr, *rindx;
    T const* rvals;
    int rows, cols, nnz;
    std::unique_ptr<typename std::remove_pointer<rocsparse_mat_descr>::type, rocm_deleter> desc;
};

template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_sparse_matrix(gpu_engine const &engine, int num_rows, int num_cols, int num_nz,
                        VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    check_types(vals);
    check_types_int(pntr, indx);
    using scalar_type = get_scalar_type<VectorLikeV>;
    return gpu_sparse_matrix<scalar_type>(engine, num_rows, num_cols, num_nz, pntr, indx, vals);
}
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_sparse_matrix(gpu_engine const &engine, int num_cols, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    check_types(vals);
    check_types_int(pntr, indx);
    using scalar_type = get_scalar_type<VectorLikeV>;
    return gpu_sparse_matrix<scalar_type>(engine, num_cols, pntr, indx, vals);
}

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
#endif

}

#endif
