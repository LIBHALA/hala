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
        #if (__HALA_CUDA_API_VERSION__ >= 10000)
        infosv{nullptr, nullptr}, infosm{nullptr, nullptr, nullptr, nullptr}
        #else
        info{nullptr, nullptr}
        #endif
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
        #if (__HALA_CUDA_API_VERSION__ >= 10000)
        infosv{std::exchange(other.infosv[0], nullptr), std::exchange(other.infosv[1], nullptr)},
        infosm{std::exchange(other.infosm[0], nullptr), std::exchange(other.infosm[1], nullptr),
               std::exchange(other.infosm[2], nullptr), std::exchange(other.infosm[3], nullptr)}
        #else
        info{std::exchange(other.info[0], nullptr), std::exchange(other.info[1], nullptr)}
        #endif
        {}

    //! \brief Free the CUDA analysis, but not the aliased matrix values and indexes.
    ~gpu_triangular_matrix(){
        #if (__HALA_CUDA_API_VERSION__ >= 10000)
        for(size_t id=0; id<2; id++){
            if (infosv[id] != nullptr)
                check_cuda(cusparseDestroyCsrsv2Info(infosv[id]), "cuSparse::DestroyCsrsv2Info()");
        }
        for(size_t id=0; id<4; id++){
            if (infosm[id] != nullptr)
                check_cuda(cusparseDestroyCsrsm2Info(infosm[id]), "cuSparse::DestroyCsrsm2Info()");
        }
        #else
        for(size_t id=0; id<2; id++){
            if (info[id] != nullptr)
                check_cuda(cusparseDestroySolveAnalysisInfo(info[id]), "cuSparse::DestroySolveAnalysisInfo()");
        }
        #endif
    }

    #if (__HALA_CUDA_API_VERSION__ >= 10000)
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
                "cuSparse::Xcsrsv2_bufferSize()", engine(), cuda_trans, nrows, nz, cdesc, pconvert(rvals), rpntr, rindx, infosv[id], &buff_size);

            auto f77_buffer = make_gpu_vector<int>(buff_size / sizeof(int), engine().device());

            cuda_call_backend<value_type>(
                cusparseScsrsv2_analysis, cusparseDcsrsv2_analysis, cusparseCcsrsv2_analysis, cusparseZcsrsv2_analysis,
                "cuSparse::Xcsrsv2_analysis()", engine(), cuda_trans, nrows, nz, cdesc, pconvert(rvals), rpntr, rindx, infosv[id], policy(), convert(f77_buffer));
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
                "cuSparse::Xcsrsm2_bufferSizeExt()", engine(), 0, cuda_transa, cuda_transb, nrows, nrhs, nz, palpha, cdesc,
                pconvert(rvals), rpntr, rindx, convert(B), ldb, infosm[id], policy(), &buff_size);

            auto f77_buffer = make_gpu_vector<int>(buff_size / sizeof(int), engine().device());

            cuda_call_backend<value_type>(
                cusparseScsrsm2_analysis, cusparseDcsrsm2_analysis, cusparseCcsrsm2_analysis, cusparseZcsrsm2_analysis,
                "cuSparse::Xcsrsm2_analysis()", engine(), 0, cuda_transa, cuda_transb, nrows, nrhs, nz,
                palpha, cdesc, pconvert(rvals), rpntr, rindx, convert(B), ldb, infosm[id], policy(), convert(f77_buffer));
        }
        return infosm[id];
    }
    #else
    //! \brief Perform analysis for vector or matrix solve, or use the cached analysis.
    cusparseSolveAnalysisInfo_t analyze_sv(char trans) const{
        assert( check_trans( trans ) );

        auto cuda_trans = trans_to_cuda_sparse<T>(trans);

        size_t const id = (is_n(trans)) ? 0 : 1;
        if (info[id] == nullptr){
            check_cuda( cusparseCreateSolveAnalysisInfo(&info[id]), "cuSparse::CreateSolveAnalysisInfo()" );

            cuda_call_backend<value_type>(
                cusparseScsrsm_analysis, cusparseDcsrsm_analysis, cusparseCcsrsm_analysis, cusparseZcsrsm_analysis,
                "cuSparse::Xcsrsm_analysis()", engine(), cuda_trans, nrows, nz, cdesc, pconvert(rvals), rpntr, rindx, info[id]);
        }
        return info[id];
    }
    #endif
    //! \brief Return the matrix description.
    cusparseMatDescr_t description() const{ return cdesc; }

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

private:
    gpu_engine rengine;
    int const *rpntr, *rindx;
    T const *rvals;

    int nrows, nz;
    cuda_struct_description<cusparseMatDescr_t> cdesc;
    char rpolicy;

    #if (__HALA_CUDA_API_VERSION__ >= 10000)
    // the new API has different analysis structs for vector and matrix solvers as well as combinations of trans operations
    csrsv2Info_t mutable infosv[2]; // non-transpose and transpose cases
    csrsm2Info_t mutable infosm[4]; // 4 way transpose non-transpose combinations between A and B
    #else
    // the old API as only one analysis info
    cusparseSolveAnalysisInfo_t mutable info[2]; // non-transpose and transpose cases
    #endif
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


/*!
 * \ingroup HALAGPUSPARSE
 * \brief Wrapper around solver of sparse triangular system of equations, Xcsrsv_solve() and Xcsrsv2_solve.
 */
template<class TriangularMat, typename FPA, class VectorLikeB, class VectorLikeX>
std::enable_if_t<is_on_gpu<TriangularMat>::value>
sparse_trsv(char trans, TriangularMat const &tri, FPA alpha, VectorLikeB const &b, VectorLikeX &x){
    check_types(tri, b, x);
    assert( valid::sparse_trsv(trans, tri, b) );
    int const rows = tri.rows();
    check_set_size(assume_output, x, rows);

    using scalar_type = get_scalar_type<VectorLikeX>;
    auto cuda_trans = trans_to_cuda_sparse<scalar_type>(trans);
    auto palpha = get_pointer<scalar_type>(alpha);

    #if (__HALA_CUDA_API_VERSION__ >= 10000)

    int buff_size = 0;
    cuda_call_backend<scalar_type>(
            cusparseScsrsv2_bufferSize, cusparseDcsrsv2_bufferSize, cusparseCcsrsv2_bufferSize, cusparseZcsrsv2_bufferSize,
            "cuSparse::Xcsrsv2_bufferSize()", tri.engine(), cuda_trans, tri.rows(), tri.nnz(), tri.description(),
            tri.vals(), tri.pntr(), tri.indx(), tri.analyze_sv(trans), pconvert(&buff_size));

    auto f77_buffer = make_gpu_vector<int>(buff_size / sizeof(int), tri.engine().device());

    cuda_call_backend<scalar_type>(
            cusparseScsrsv2_solve, cusparseDcsrsv2_solve, cusparseCcsrsv2_solve, cusparseZcsrsv2_solve,
            "cuSparse::Xcsrsv2_solve()", tri.engine(), cuda_trans, tri.rows(), tri.nnz(), palpha, tri.description(),
            tri.vals(), tri.pntr(), tri.indx(), tri.analyze_sv(trans),
            convert(b), cconvert(x), tri.policy(), cconvert(f77_buffer));

    #else
    cuda_call_backend<scalar_type>(cusparseScsrsv_solve, cusparseDcsrsv_solve, cusparseCcsrsv_solve, cusparseZcsrsv_solve,
                      "cuSparse::Xcsrsm_solve()", tri.engine(), cuda_trans, rows, palpha, tri.description(),
                      tri.vals(), tri.pntr(), tri.indx(), tri.analyze_sv(trans), convert(b), cconvert(x));
    #endif
}

/*!
 * \ingroup HALAGPUSPARSE
 * \brief Triangular matrix-matrix solve using the gpu_engine.
 */
template<class TriangularMat, typename FPA, class VectorLikeB>
std::enable_if_t<is_on_gpu<TriangularMat>::value>
sparse_trsm(char transa, char transb, int nrhs, TriangularMat &tri, FPA alpha, VectorLikeB &B, int ldb = -1){
    check_types(tri, B);
    valid::default_ld(is_n(transb), tri.rows(), nrhs, ldb);
    assert( valid::sparse_trsm(transa, transb, nrhs, tri, B, ldb) );

    using scalar_type = get_scalar_type<VectorLikeB>;
    auto palpha = get_pointer<scalar_type>(alpha);

    #if (__HALA_CUDA_API_VERSION__ >= 10000)

    auto cuda_transa = trans_to_cuda_sparse<scalar_type>(transa);
    if (is_c(transb)) transb = 'T'; // the conj operation cancels out, use simple transpose
    auto cuda_transb = trans_to_cuda_sparse<scalar_type>(transb);

    size_t buff_size = 0;
    cuda_call_backend<scalar_type>(
            cusparseScsrsm2_bufferSizeExt, cusparseDcsrsm2_bufferSizeExt, cusparseCcsrsm2_bufferSizeExt, cusparseZcsrsm2_bufferSizeExt,
            "cuSparse::Xcsrsm2_bufferSize()", tri.engine(), 0, cuda_transa, cuda_transb, tri.rows(), nrhs, tri.nnz(), palpha, tri.description(),
            tri.vals(), tri.pntr(), tri.indx(), convert(B), ldb,
            tri.analyze_sm(transa, transb, nrhs, alpha, B, ldb), tri.policy(), pconvert(&buff_size));

    auto f77_buffer = make_gpu_vector<int>(buff_size / sizeof(int), tri.engine().device());

    palpha = get_pointer<scalar_type>(alpha);
    cuda_call_backend<scalar_type>(
            cusparseScsrsm2_solve, cusparseDcsrsm2_solve, cusparseCcsrsm2_solve, cusparseZcsrsm2_solve,
            "cuSparse::Xcsrsm2_solve()", tri.engine(), 0, cuda_transa, cuda_transb, tri.rows(), nrhs, tri.nnz(), palpha, tri.description(),
            tri.vals(), tri.pntr(), tri.indx(), convert(B), ldb,
            tri.analyze_sm(transa, transb, nrhs, alpha, B, ldb), tri.policy(), cconvert(f77_buffer));

    #else

    auto cuda_transa = trans_to_cuda_sparse<scalar_type>(transa);

    auto C = new_vector(tri.engine(), B);
    force_size(C, hala_size(tri.rows(), nrhs));

    if (is_n(transb)){

        cuda_call_backend<scalar_type>(cusparseScsrsm_solve, cusparseDcsrsm_solve, cusparseCcsrsm_solve, cusparseZcsrsm_solve,
                          "cuSparse::Xcsrsm_solve()", tri.engine(), cuda_transa, tri.rows(), nrhs, palpha, tri.description(),
                          tri.vals(), tri.pntr(), tri.indx(), tri.analyze_sv(transa), convert(B), ldb, cconvert(C), tri.rows());

        if (ldb == tri.rows()){
            vcopy(tri.engine(), C, B); // copy back
        }else{ // copy only the block
            gpu_pntr<host_pntr> hold(tri.engine()); // in case device pointers were used in the call above
            geam(tri.engine(), 'N', 'N', tri.rows(), nrhs, 1, C, tri.rows(), 0, C, tri.rows(), B, ldb);
        }
    }else{

        auto X = new_vector(tri.engine(), B); // set X = B^T
        force_size(X, hala_size(tri.rows(), nrhs));
        {
            gpu_pntr<host_pntr> hold(tri.engine());
            geam(tri.engine(), 'T', 'T', tri.rows(), nrhs, 1, B, ldb, 0, B, ldb, X, tri.rows());
        }

        cuda_call_backend<scalar_type>(cusparseScsrsm_solve, cusparseDcsrsm_solve, cusparseCcsrsm_solve, cusparseZcsrsm_solve,
                          "cuSparse::Xcsrsm_solve()", tri.engine(), cuda_transa, tri.rows(), nrhs, palpha, tri.description(),
                          tri.vals(), tri.pntr(), tri.indx(), tri.analyze_sv(transa), convert(X), tri.rows(), cconvert(C), tri.rows());

        { // C holds the answer, write back to B in transposed order
            gpu_pntr<host_pntr> hold(tri.engine());
            geam(tri.engine(), 'T', 'T', nrhs, tri.rows(), 1, C, tri.rows(), 0, C, tri.rows(), B, ldb);
        }

    }
    #endif
}

}

#endif
