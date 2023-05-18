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

#include "hala_cuda_sparse_general.hpp"

namespace hala{

/*!
 * \ingroup HALAGPUSPARSE
 * \brief Wrapper around a sparse triangular matrix and all associated analysis.
 *
 * Takes non-owning references to a gpu_engine and arrays (from vector like classes)
 * and managed all the associated CUDA analysis.
 *
 * In place of the cosntructor, it is better to use the factory method hala::make_triangular_matrix().
 */
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
                           char lpolicy
                          )
        : rengine(cengine), rpntr(get_data(pn)), rindx(get_data(in)), rvals(get_standard_data(va)),
        nrows(get_size_int(pn) - 1), nz(get_size_int(in)), cdesc(make_cuda_mat_description('G', uplo, diag)),
        rpolicy(lpolicy),
        infosv{nullptr, nullptr}, infosm{nullptr, nullptr, nullptr, nullptr}
    {
        assert( get_size(in) == get_size(va) );

        // make the matrix description
        assert( check_uplo(uplo) );
        assert( check_diag(diag) );
    }

    //! \brief The matrix is not copiable (the CUDA analysis cannot be copied).
    gpu_triangular_matrix(gpu_triangular_matrix const &other) = delete;
    //! \brief The matrix is not copiable (the CUDA analysis cannot be copied).
    gpu_triangular_matrix& operator = (gpu_triangular_matrix const &other) = delete;

    //! \brief The matrix is trivially movable.
    gpu_triangular_matrix& operator = (gpu_triangular_matrix &&other) = default;

    //! \brief The matrix is trivially movable.
    gpu_triangular_matrix(gpu_triangular_matrix &&other) :
        rengine(other.rengine), rpntr(other.rpntr), rindx(other.rindx), rvals(other.rvals),
        nrows(other.nrows), nz(other.nz), cdesc(std::move(other.cdesc)),
        rpolicy(other.rpolicy),
        infosv{std::exchange(other.infosv[0], nullptr), std::exchange(other.infosv[1], nullptr)},
        infosm{std::exchange(other.infosm[0], nullptr), std::exchange(other.infosm[1], nullptr),
               std::exchange(other.infosm[2], nullptr), std::exchange(other.infosm[3], nullptr)}
        {}

    //! \brief Free the CUDA analysis, but not the aliased matrix values and indexes.
    ~gpu_triangular_matrix(){
        for(size_t id=0; id<2; id++){
            if (infosv[id] != nullptr)
                check_cuda(cusparseDestroyCsrsv2Info(infosv[id]), "cuSparse::DestroyCsrsv2Info()");
        }
        for(size_t id=0; id<4; id++){
            if (infosm[id] != nullptr)
                check_cuda(cusparseDestroyCsrsm2Info(infosm[id]), "cuSparse::DestroyCsrsm2Info()");
        }
    }

    //! \brief Return the solver policy selected during construction.
    cusparseSolvePolicy_t policy() const{
        return ((rpolicy == 'L') || (rpolicy == 'l')) ?  CUSPARSE_SOLVE_POLICY_USE_LEVEL : CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    }
    //! \brief Perform analysis for vector solve, or use the cached analysis.
    csrsv2Info_t analyze_sv(char trans) const{
        assert( check_trans( trans ) );

        auto cuda_trans = trans_to_cuda_sparse<T>(trans);

        size_t const id = (is_n(trans)) ? 0 : 1;
        if (infosv[id] == nullptr){
            check_cuda(cusparseCreateCsrsv2Info(&infosv[id]), "cuSparse::CreateCsrsv2Info()");

            int buff_size = 0;
            cuda_call_backend<value_type>(
                cusparseScsrsv2_bufferSize, cusparseDcsrsv2_bufferSize, cusparseCcsrsv2_bufferSize, cusparseZcsrsv2_bufferSize,
                "cuSparse::Xcsrsv2_bufferSize()", engine(), cuda_trans, nrows, nz, cdesc.get(), pconvert(rvals), rpntr, rindx, infosv[id], &buff_size);

            auto tmp_buffer = make_gpu_vector<int>(buff_size / sizeof(int), engine().device());

            cuda_call_backend<value_type>(
                cusparseScsrsv2_analysis, cusparseDcsrsv2_analysis, cusparseCcsrsv2_analysis, cusparseZcsrsv2_analysis,
                "cuSparse::Xcsrsv2_analysis()", engine(), cuda_trans, nrows, nz, cdesc.get(), pconvert(rvals), rpntr, rindx, infosv[id], policy(), convert(tmp_buffer));
        }
        return infosv[id];
    }
    //! \brief Perform analysis for matrix solve, or use the cached analysis.
    template<typename FPa, class VectorLikeB>
    csrsm2Info_t analyze_sm(char transa, char transb, int nrhs, FPa alpha, VectorLikeB const &B, int ldb) const{ // transb is not used
        assert( check_trans( transa ) );
        assert( check_trans( transb ) );

        auto cuda_transa = trans_to_cuda_sparse<T>(transa);
        auto cuda_transb = trans_to_cuda_sparse<T>(transb);

        size_t const id = (is_n(transa)) ? ( (is_n(transb)) ? 0 : 1 ) : ( (is_n(transb)) ? 2 : 3 );

        if (infosm[id] == nullptr){
            cusparseStatus_t status = cusparseCreateCsrsm2Info(&infosm[id]);
            check_cuda(status, "cuSparse::CreateCsrsm2Info()");

            auto palpha = get_pointer<value_type>(alpha);

            size_t buff_size = 0;
            cuda_call_backend<value_type>(
                cusparseScsrsm2_bufferSizeExt, cusparseDcsrsm2_bufferSizeExt, cusparseCcsrsm2_bufferSizeExt, cusparseZcsrsm2_bufferSizeExt,
                "cuSparse::Xcsrsm2_bufferSizeExt()", engine(), 0, cuda_transa, cuda_transb, nrows, nrhs, nz, palpha, cdesc.get(),
                pconvert(rvals), rpntr, rindx, convert(B), ldb, infosm[id], policy(), &buff_size);

            auto f77_buffer = make_gpu_vector<int>(buff_size / sizeof(int), engine().device());

            cuda_call_backend<value_type>(
                cusparseScsrsm2_analysis, cusparseDcsrsm2_analysis, cusparseCcsrsm2_analysis, cusparseZcsrsm2_analysis,
                "cuSparse::Xcsrsm2_analysis()", engine(), 0, cuda_transa, cuda_transb, nrows, nrhs, nz,
                palpha, cdesc.get(), pconvert(rvals), rpntr, rindx, convert(B), ldb, infosm[id], policy(), convert(f77_buffer));
        }
        return infosm[id];
    }

    //! \brief Return the matrix description.
    cusparseMatDescr_t description() const{ return cdesc.get(); }

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

    //! \brief Returns the size required for the temporary buffers.
    template<typename FPA, class VectorLikeB, class VectorLikeX>
    size_t trsv_buffer_size(char trans, FPA, VectorLikeB const&, VectorLikeX &) const{
        auto cuda_trans = trans_to_cuda_sparse<value_type>(trans);

        int buff_size = 0;
        cuda_call_backend<value_type>(
                cusparseScsrsv2_bufferSize, cusparseDcsrsv2_bufferSize, cusparseCcsrsv2_bufferSize, cusparseZcsrsv2_bufferSize,
                "cusparseXcsrsv2_bufferSize", rengine, cuda_trans, nrows, nz, cdesc.get(),
                vals(), rpntr, rindx, analyze_sv(trans), pconvert(&buff_size));
        return static_cast<size_t>(buff_size);
    }
    //! \brief Matrix vector product with a user-provided buffer.
    template<typename FPA, class VectorLikeB, class VectorLikeX, class VectorLikeT>
    void trsv(char trans, FPA alpha, VectorLikeB const &b, VectorLikeX &&x, VectorLikeT &&temp) const{
        check_types(rvals, b, x);
        assert( valid::sparse_trsv(trans, *this, b) );
        check_set_size(assume_output, x, nrows);

        auto cuda_trans = trans_to_cuda_sparse<value_type>(trans);
        auto palpha = get_pointer<value_type>(alpha);

        cuda_call_backend<value_type>(
                cusparseScsrsv2_solve, cusparseDcsrsv2_solve, cusparseCcsrsv2_solve, cusparseZcsrsv2_solve,
                "cusparseXcsrsv2_solve", rengine, cuda_trans, nrows, nz, palpha, cdesc.get(),
                vals(), rpntr, rindx, analyze_sv(trans),
                convert(b), cconvert(x), policy(), cconvert(temp));
    }
    //! \brief Returns the size required for the temporary buffers.
    template<typename FPA, class VectorLikeB>
    size_t trsm_buffer_size(char transa, char transb, int nrhs, FPA alpha, VectorLikeB &&B, int ldb = -1) const{
        check_types(rvals, B);
        valid::default_ld(is_n(transb), nrows, nrhs, ldb);
        auto palpha = get_pointer<value_type>(alpha);

        auto cuda_transa = trans_to_cuda_sparse<value_type>(transa);
        if (is_c(transb)) transb = 'T'; // the conj operation cancels out, use simple transpose
        auto cuda_transb = trans_to_cuda_sparse<value_type>(transb);

        size_t buff_size = 0;
        cuda_call_backend<value_type>(
                cusparseScsrsm2_bufferSizeExt, cusparseDcsrsm2_bufferSizeExt, cusparseCcsrsm2_bufferSizeExt, cusparseZcsrsm2_bufferSizeExt,
                "cusparseXcsrsm2_bufferSizeExt", rengine, 0, cuda_transa, cuda_transb, nrows, nrhs, nz, palpha, cdesc.get(),
                vals(), rpntr, rindx, convert(B), ldb,
                analyze_sm(transa, transb, nrhs, alpha, B, ldb), policy(), pconvert(&buff_size));
        return buff_size;
    }
    //! \brief Matrix matrix product with a user-provided buffer.
    template<typename FPA, class VectorLikeB, class VectorLikeT>
    void trsm(char transa, char transb, int nrhs, FPA alpha, VectorLikeB &&B, int ldb, VectorLikeT &&temp) const{
        check_types(rvals, B);
        valid::default_ld(is_n(transb), nrows, nrhs, ldb);
        assert( valid::sparse_trsm(transa, transb, nrhs, *this, B, ldb) );

        auto palpha = get_pointer<value_type>(alpha);

        auto cuda_transa = trans_to_cuda_sparse<value_type>(transa);
        if (is_c(transb)) transb = 'T'; // the conj operation cancels out, use simple transpose
        auto cuda_transb = trans_to_cuda_sparse<value_type>(transb);

        cuda_call_backend<value_type>(
                cusparseScsrsm2_solve, cusparseDcsrsm2_solve, cusparseCcsrsm2_solve, cusparseZcsrsm2_solve,
                "cusparseXcsrsm2_solve", rengine, 0, cuda_transa, cuda_transb, nrows, nrhs, nz, palpha, cdesc.get(),
                vals(), rpntr, rindx, convert(B), ldb,
                analyze_sm(transa, transb, nrhs, alpha, B, ldb), policy(), cconvert(temp));
    }

    //! \brief Matrix vector product.
    template<typename FPA, class VectorLikeB, class VectorLikeX>
    void trsv(char trans, FPA alpha, VectorLikeB const &b, VectorLikeX &x) const{
        trsv(trans, alpha, b, x, gpu_vector<value_type>(trsv_buffer_size(trans, alpha, b, x), rengine.device()));
    }
    //! \brief Matrix matrix product.
    template<typename FPA, class VectorLikeB>
    void trsm(char transa, char transb, int nrhs, FPA alpha, VectorLikeB &B, int ldb = -1) const{
        valid::default_ld(is_n(transb), nrows, nrhs, ldb);
        trsm(transa, transb, nrhs, alpha, B, ldb,
             gpu_vector<value_type>(trsm_buffer_size(transa, transb, nrhs, alpha, B, ldb), rengine.device()));
    }

private:
    gpu_engine rengine;
    int const *rpntr, *rindx;
    T const *rvals;

    int nrows, nz;
    std::unique_ptr<typename std::remove_pointer<cusparseMatDescr_t>::type, cuda_deleter> cdesc;
    char rpolicy;

    // the new API has different analysis structs for vector and matrix solvers as well as combinations of trans operations
    csrsv2Info_t mutable infosv[2]; // non-transpose and transpose cases
    csrsm2Info_t mutable infosm[4]; // 4 way transpose non-transpose combinations between A and B

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

}

#endif
