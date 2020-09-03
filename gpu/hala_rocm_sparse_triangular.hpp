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

#include "hala_rocm_sparse_general.hpp"

namespace hala{

template<typename T>
class gpu_triangular_matrix{
public:
    //! \brief The value type of the matrix.
    using value_type = std::remove_cv_t<T>;
    //! \brief Define the underlying engine.
    using engine_type = gpu_engine;

    //! \brief Create the triangular matrix, prefer using hala::make_triangular_matrix().
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    gpu_triangular_matrix(gpu_engine const &cengine, char uplo, char diag,
                           VectorLikeP const &pn, VectorLikeI const &in, VectorLikeV const &va,
                           char
                          )
        : rengine(cengine), rpntr(get_data(pn)), rindx(get_data(in)), rvals(get_standard_data(va)),
        nrows(get_size_int(pn) - 1), nz(get_size_int(in)), buff_size(0),
        cdesc(make_rocsparse_triangular_description(uplo, diag))
    {
        assert( get_size(in) == get_size(va) );

        rocm_call_backend<value_type>(rocsparse_scsrsv_buffer_size, rocsparse_dcsrsv_buffer_size, rocsparse_ccsrsv_buffer_size, rocsparse_zcsrsv_buffer_size,
            "csrsv_buffer_size", rengine, rocsparse_operation_none, nrows, nz, cdesc, pconvert(rvals), pconvert(rpntr), pconvert(rindx), info, pconvert(&buff_size));
        buff_size /= sizeof(value_type);
    }

    //! \brief The matrix is not copiable (the CUDA analysis cannot be copied).
    gpu_triangular_matrix(gpu_triangular_matrix const &other) = delete;
    //! \brief The matrix is not copiable (the CUDA analysis cannot be copied).
    gpu_triangular_matrix& operator = (gpu_triangular_matrix const &other) = delete;

    //! \brief The matrix is trivially movable.
    gpu_triangular_matrix& operator = (gpu_triangular_matrix &&other) = default;

    //! \brief The matrix is trivially movable.
    gpu_triangular_matrix(gpu_triangular_matrix &&other) = default;

    //! \brief Free the CUDA analysis, but not the aliased matrix values and indexes.
    ~gpu_triangular_matrix(){}

    //! \brief Return the matrix description.
    rocsparse_mat_descr description() const{ return cdesc.desc; }
    //! \brief Return the analysis info for vector solve.
    template<class VectorLike>
    rocsparse_mat_info analysis(rocsparse_operation trans, VectorLike &&buff) const{
        rocm_call_backend<value_type>(rocsparse_scsrsv_analysis, rocsparse_dcsrsv_analysis, rocsparse_ccsrsv_analysis, rocsparse_zcsrsv_analysis,
            "csrsv_analysis", rengine, trans, nrows, nz, cdesc, pconvert(rvals), pconvert(rpntr), pconvert(rindx), info,
            rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, cconvert(buff));
        return info.desc;
    }
    //! \brief Return the current analysis info.
    rocsparse_mat_info analysis() const{ return info.desc; }

    //! \brief Return the alias to the pntr data.
    int const* pntr() const{ return rpntr; }
    //! \brief Return the alias to the indx data.
    int const* indx() const{ return rindx; }
    //! \brief Return the alias to the vals data.
    auto vals() const{ return pconvert(rvals); }

    //! \brief Return the alias to the engine.
    gpu_engine const& engine() const{ return rengine; }

    //! \brief Return the number of rows/columns.
    int rows() const{ return nrows; }
    //! \brief Return the number of non-zeros.
    int nnz() const{ return nz; }
    //! \brief Returns the buffer size for vector solve.
    size_t buffer_size() const{ return buff_size; }

private:
    gpu_engine rengine;
    int const *rpntr, *rindx;
    value_type const *rvals;
    int nrows, nz;
    size_t buff_size;

    rocm_struct_description<rocsparse_mat_descr> cdesc;
    rocm_struct_description<rocsparse_mat_info> info;
    char rpolicy;
};

/*!
 * \ingroup HALACUDASPARSE
 * \brief Overload using the CUDA engine.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_triangular_matrix(gpu_engine const &engine, char uplo, char diag,
                            VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, char policy = 'N'){
    check_types(vals);
    check_types_int(pntr, indx);
    assert( get_size(indx) == get_size(vals) );
    assert( check_uplo(uplo) );
    assert( check_diag(diag) );
    using scalar_type = get_scalar_type<VectorLikeV>;
    return gpu_triangular_matrix<scalar_type>(engine, uplo, diag, pntr, indx, vals, policy);
}

/*!
 * \ingroup HALACUDACOMMON
 * \brief Switches between true-type and false-type depending whether \b TriangularMatrix::engine_type is cpu_engine.
 */
template<class TriangularMatrix, typename = void> struct is_on_gpu : std::false_type{};

/*!
 * \ingroup HALACUDACOMMON
 * \brief Specialization indicating that the engine is indeed the cuda_engine.
 */
template<class TriangularMatrix>
struct is_on_gpu<TriangularMatrix, std::enable_if_t<std::is_same<typename TriangularMatrix::engine_type, gpu_engine>::value, void>> : std::true_type{};

template<class TriangularMat, typename FPA, class VectorLikeB, class VectorLikeX>
std::enable_if_t<is_on_gpu<TriangularMat>::value>
sparse_trsv(char trans, TriangularMat const &tri, FPA alpha, VectorLikeB const &b, VectorLikeX &x){
    check_types(tri, b, x);
    assert( valid::sparse_trsv(trans, tri, b) );
    int const rows = tri.rows();
    check_set_size(assume_output, x, rows);

    using scalar_type = get_scalar_type<VectorLikeX>;
    auto rocm_trans = trans_to_rocm_sparse<scalar_type>(trans);
    auto palpha = get_pointer<scalar_type>(alpha);

    runtime_assert(rocm_trans != rocsparse_operation_conjugate_transpose, "as of rocsparse 3.5 there is no support for conjugate-transpose trsv operations.");

    gpu_vector<scalar_type> buff(tri.buffer_size(), tri.engine().device());
    rocm_call_backend<scalar_type>(rocsparse_scsrsv_solve, rocsparse_dcsrsv_solve, rocsparse_ccsrsv_solve, rocsparse_zcsrsv_solve,
        "csrsv_solve", tri.engine(), rocm_trans, tri.rows(), tri.nnz(), palpha, tri.description(), tri.vals(), tri.pntr(), tri.indx(),
        tri.analysis(rocm_trans, buff), convert(b), cconvert(x), rocsparse_solve_policy_auto, convert(buff));
}


template<class TriangularMat, typename FPA, class VectorLikeB>
std::enable_if_t<is_on_gpu<TriangularMat>::value>
sparse_trsm(char transa, char transb, int nrhs, TriangularMat &tri, FPA alpha, VectorLikeB &B, int ldb = -1){
    check_types(tri, B);
    valid::default_ld(is_n(transb), tri.rows(), nrhs, ldb);
    assert( valid::sparse_trsm(transa, transb, nrhs, tri, B, ldb) );

    using scalar_type = get_scalar_type<VectorLikeB>;
    auto rocm_transa = trans_to_rocm_sparse<scalar_type>(transa);
    auto rocm_transb = trans_to_rocm_sparse<scalar_type>(transb);
    auto palpha = get_pointer<scalar_type>(alpha);

    rocsparse_mat_info info = tri.analysis();

    runtime_assert(rocm_transa == rocsparse_operation_none, "as of rocsparse 3.5 there is no support for transpose and conjugate-transpose trsm operations on A.");
    runtime_assert(rocm_transb != rocsparse_operation_conjugate_transpose, "as of rocsparse 3.5 there is no support for conjugate-transpose trsm operations on B.");

    size_t buff_size = 0;
    rocm_call_backend<scalar_type>(rocsparse_scsrsm_buffer_size, rocsparse_dcsrsm_buffer_size, rocsparse_ccsrsm_buffer_size, rocsparse_zcsrsm_buffer_size,
        "csrsm_buffer_size", tri.engine(), rocm_transa, rocm_transb, tri.rows(), nrhs, tri.nnz(), palpha, tri.description(), tri.vals(), tri.pntr(), tri.indx(),
        convert(B), ldb, info, rocsparse_solve_policy_auto, pconvert(&buff_size));
    buff_size /= sizeof(scalar_type);

    gpu_vector<scalar_type> buff(buff_size, tri.engine().device());
    rocm_call_backend<scalar_type>(rocsparse_scsrsm_analysis, rocsparse_dcsrsm_analysis, rocsparse_ccsrsm_analysis, rocsparse_zcsrsm_analysis,
            "csrsm_analysis", tri.engine(), rocm_transa, rocm_transb, tri.rows(), nrhs, tri.nnz(), palpha, tri.description(), tri.vals(), tri.pntr(), tri.indx(),
            convert(B), ldb, info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, cconvert(buff));

    rocm_call_backend<scalar_type>(rocsparse_scsrsm_solve, rocsparse_dcsrsm_solve, rocsparse_ccsrsm_solve, rocsparse_zcsrsm_solve,
            "csrsm_solve", tri.engine(), rocm_transa, rocm_transb, tri.rows(), nrhs, tri.nnz(), palpha, tri.description(), tri.vals(), tri.pntr(), tri.indx(),
            convert(B), ldb, info, rocsparse_solve_policy_auto, cconvert(buff));
}

}
#endif
