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
 * \internal
 * \ingroup HALAROCMCOMMON
 * \brief RAII wrapper around ROCM mat-description and mat-info.
 *
 * Creates an instance of rocsparse_mat_descr or rocsparse_mat_info and deletes with the destructor.
 * \endinternal
 */
template<typename T>
struct rocm_struct_description{
    //! \brief Construct an empty struct ready to be used.
    rocm_struct_description() : desc(nullptr){
        static_assert(std::is_same<T, rocsparse_mat_descr>::value or
                      std::is_same<T, rocsparse_mat_info>::value,
                      "Incorrect type T for rocm_struct_description<T>");
        if __HALA_CONSTEXPR_IF__ (std::is_same<T, rocsparse_mat_descr>::value){
            check_rocm( rocsparse_create_mat_descr(pconvert(&desc)), "rocsparse_create_mat_descr()");
        }else __HALA_CONSTEXPR_IF__ if (std::is_same<T, rocsparse_mat_info>::value){
            check_rocm( rocsparse_create_mat_info(pconvert(&desc)), "rocsparse_create_mat_info()");
        }
    }
    //! \brief Destructor, frees all used memory.
    ~rocm_struct_description(){
        if (desc != nullptr){
            if __HALA_CONSTEXPR_IF__ (std::is_same<T, rocsparse_mat_descr>::value){
                check_rocm( rocsparse_destroy_mat_descr(pconvert(desc)), "rocsparse_destroy_mat_descr()");
            }else if __HALA_CONSTEXPR_IF__ (std::is_same<T, rocsparse_mat_info>::value){
                check_rocm( rocsparse_destroy_mat_info(pconvert(desc)), "rocsparse_destroy_mat_info()");
            }
        }
    }

    //! \brief No copy constructor.
    rocm_struct_description(rocm_struct_description<T> const&) = delete;
    //! \brief Can move construct.
    rocm_struct_description(rocm_struct_description<T> &&other) : desc(std::exchange(other.desc, nullptr)){};

    //! \brief Cannot copy assign.
    rocm_struct_description<T>& operator = (rocm_struct_description<T> const&) = delete;
    //! \brief Can move assign.
    rocm_struct_description<T>& operator = (rocm_struct_description<T> &&other){
        rocm_struct_description<T> temp(std::move(other));
        std::swap(desc, temp.desc);
        return *this;
    }

    //! \brief Automatically convert to the wrapped type.
    operator T () const { return desc; }
    //! \brief Wrapper pointer.
    T desc;
};

/*!
 * \ingroup HALAROCMCOMMON
 * \brief Factory method for a general matrix description.
 *
 * \param diag specifies whether the matrix is uint or non-unit diagonal
 */
inline rocm_struct_description<rocsparse_mat_descr> make_rocsparse_general_description(char diag = 'N'){
    assert( check_diag(diag) );
    rocm_struct_description<rocsparse_mat_descr> desc;
    rocsparse_set_mat_type(desc, rocsparse_matrix_type_general);
    rocsparse_set_mat_index_base(desc, rocsparse_index_base_zero);
    rocsparse_set_mat_diag_type(desc, (is_n(diag)) ? rocsparse_diag_type_non_unit : rocsparse_diag_type_unit);
    return desc;
}
/*!
 * \ingroup HALAROCMCOMMON
 * \brief Factory method for a triangular matrix description.
 *
 * \param uplo specifies whether the matrix has upper or lower fill
 * \param diag specifies whether the matrix is uint or non-unit diagonal
 */
inline rocm_struct_description<rocsparse_mat_descr> make_rocsparse_triangular_description(char uplo, char diag){
    assert( check_uplo(uplo) );
    assert( check_diag(diag) );
    rocm_struct_description<rocsparse_mat_descr> desc;
    rocsparse_set_mat_type(desc, rocsparse_matrix_type_general);
    rocsparse_set_mat_index_base(desc, rocsparse_index_base_zero);
    rocsparse_set_mat_fill_mode(desc, (is_l(uplo)) ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper);
    rocsparse_set_mat_diag_type(desc, (is_n(diag)) ? rocsparse_diag_type_non_unit : rocsparse_diag_type_unit);
    return desc;
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
        runtime_assert(rocm_trans == rocsparse_operation_none, "as of rocsparse 3.5 there is no support for transpose and conjugate-transpose gemv operations.");
        auto palpha = get_pointer<value_type>(alpha);
        auto pbeta  = get_pointer<value_type>(beta);

        rocsparse_mat_info info = nullptr;

        rocm_call_backend<value_type>(rocsparse_scsrmv, rocsparse_dcsrmv, rocsparse_ccsrmv, rocsparse_zcsrmv,
            "csrmv()", rengine, rocm_trans, rows, cols, nnz, palpha, desc, pconvert(rvals), pconvert(rpntr), pconvert(rindx), info, convert(x), pbeta, cconvert(y));
    }
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
    void gemv(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y) const{
        gemv(trans, alpha, x, beta, y, gpu_vector<value_type>(rengine.device()));
    }
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
    size_t gemm_buffer_size(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &C, int ldc){
        if (is_complex<value_type>::value and trans_to_rocm_sparse<value_type>(transb) == rocsparse_operation_conjugate_transpose){
            return hala_size(b_rows, b_cols) * sizeof(value_type);
        }
        return 0;
    }
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC, class VectorLikeT>
    void gemm(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &C, int ldc, VectorLikeT &&temp){
        check_types(rvals, B, C);
        rengine.check_gpu(B, C);
        int M = (is_n(transa)) ? rows : cols;
        int N = (is_n(transb)) ? b_cols : b_rows;
        int K = (is_n(transa)) ? cols : rows;
        pntr_check_set_size(beta, C, ldc, N);
        assert( valid::sparse_gemm(transa, transb, M, N, K, nnz, rpntr, rindx, rvals, B, ldb, C, ldc) );

        auto palpha = get_pointer<value_type>(alpha);
        auto pbeta  = get_pointer<value_type>(beta);

        auto rocm_transa = trans_to_rocm_sparse<value_type>(transa);
        auto rocm_transb = trans_to_rocm_sparse<value_type>(transb);
        runtime_assert(rocm_transa == rocsparse_operation_none, "as of rocsparse 3.5 there is no support for transpose and conjugate-transpose gemm operations.");

        auto desc = make_rocsparse_general_description();

        if (is_complex<value_type>::value and rocm_transb == rocsparse_operation_conjugate_transpose){
            // employing manual out-of-place conj-transpose for B
            geam(rengine, 'C', 'C', K, N, 1.0, B, ldb, 0.0, B, ldb, temp, K);
            rocm_call_backend<value_type>(rocsparse_scsrmm, rocsparse_dcsrmm, rocsparse_ccsrmm, rocsparse_zcsrmm,
                "csrmm()", rengine, rocm_transa, rocsparse_operation_none, M, N, K, nnz, palpha, desc, convert(rvals), convert(rpntr), convert(rindx), convert(temp), K, pbeta, cconvert(C), ldc);
        }else{
        rocm_call_backend<value_type>(rocsparse_scsrmm, rocsparse_dcsrmm, rocsparse_ccsrmm, rocsparse_zcsrmm,
                "csrmm()", rengine, rocm_transa, rocm_transb, M, N, K, nnz, palpha, desc, convert(rvals), convert(rpntr), convert(rindx), convert(B), ldb, pbeta, cconvert(C), ldc);
        }
    }
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
    void gemm(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &C, int ldc){
        size_t work = gemm_buffer_size(transa, transb, b_rows, b_cols, alpha, B, ldb, beta, C, ldc);
        gemm(transa, transb, b_rows, b_cols, alpha, B, ldb, beta, C, ldc, gpu_vector<value_type>(work / sizeof(value_type) + 1, rengine.device()));
    }

private:
    gpu_engine rengine;
    int const *rpntr, *rindx;
    T const* rvals;
    int rows, cols, nnz;
    rocm_struct_description<rocsparse_mat_descr> desc;
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
