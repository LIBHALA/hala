#ifndef __HALA_SPARSE_STRUCTS_HPP
#define __HALA_SPARSE_STRUCTS_HPP
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
 * \file hala_sparse.hpp
 * \brief HALA Sparse Linear Algebra
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALASPARSE
 *
 * Contains the HALA Sparse Matrix structures for triangular solves.
 * \endinternal
 */

#include "hala_sparse_utils.hpp"

namespace hala{

/*!
 * \ingroup HALASPARSE
 * \brief Wrapper around a sparse matrix located in CPU memory.
 *
 * There is no sparse standard that is as universal as BLAS, many standards exist and several of them require that
 * data is wrapped around specific data-structures and comes with additional analysis meta-data.
 * HALA aims at providing a universal API and the sparse matrix classes all take non-owning references to
 * the data and seamlessly wrap around the required backend data-structures.
 */
template<typename T>
struct cpu_sparse_matrix{
    //! \brief The value type of the matrix.
    using value_type = std::remove_cv_t<T>;
    //! \brief Define the underlying engine.
    using engine_type = cpu_engine;
    //! \brief Constructor set the data for the matrix using the given dimensions.
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    cpu_sparse_matrix(int num_rows, int num_cols, int num_nz, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals)
    : rpntr(get_data(pntr)), rindx(get_data(indx)), rvals(get_standard_data(vals)), rows(num_rows), cols(num_cols), nnz(num_nz)
    {
        check_types(vals);
        check_types_int(pntr, indx);
        assert( check_size(pntr, rows+1) );
        assert( check_size(indx, nnz) );
        assert( check_size(vals, nnz) );
    }
    //! \brief Constructor set the data for the matrix, infers the number of rows and num_nz from the inputs (cannot use raw-arrays).
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    cpu_sparse_matrix(int num_cols, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals)
    : rpntr(get_data(pntr)), rindx(get_data(indx)), rvals(get_standard_data(vals)), rows(get_size_int(pntr) - 1), cols(num_cols), nnz(get_size(indx))
    {
        check_types(vals);
        check_types_int(pntr, indx);
        assert( check_size(indx, nnz) );
    }
    //! \brief Engine overload.
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    cpu_sparse_matrix(cpu_engine const&, int num_rows, int num_cols, int num_nz,
                      VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals)
    : cpu_sparse_matrix(num_rows, num_cols, num_nz, pntr, indx, vals){}
    //! \brief Engine overload.
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    cpu_sparse_matrix(cpu_engine const&, int num_cols, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals)
    : cpu_sparse_matrix(num_cols, pntr, indx, vals){}
    //! \brief Using default destructor.
    ~cpu_sparse_matrix() = default;

    //! \brief Cannot copy.
    cpu_sparse_matrix(cpu_sparse_matrix<T> const&) = delete;
    //! \brief Can move.
    cpu_sparse_matrix(cpu_sparse_matrix<T> &&) = default;

    //! \brief Cannot copy.
    cpu_sparse_matrix<T>& operator = (cpu_sparse_matrix<T> const&) = delete;
    //! \brief Can move.
    cpu_sparse_matrix<T>& operator = (cpu_sparse_matrix<T> &&) = default;

    //! \brief Returns 0, added for consistency with CUDA and ROCM.
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
    size_t gemv_buffer_size(char, FPa, VectorLikeX const&, FPb, VectorLikeY&&) const{ return 0; }
    //! \brief Matrix-vector product.
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
    void gemv(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y) const{
        check_types(rvals, x, y);
        check_set_size((hala_abs(beta) == 0.0), y, (is_n(trans)) ? rows : cols, 1);

        auto const fsalpha = get_cast<value_type>(alpha);
        auto const fsbeta  = get_cast<value_type>(beta);

        if (is_n(trans))
            sparse_gemv_array<value_type, 'N'>(rows, cols, fsalpha, rpntr, rindx, pconvert(rvals), convert(x), fsbeta, cconvert(y));
        else if (is_c(trans))
            sparse_gemv_array<value_type, 'C'>(rows, cols, fsalpha, rpntr, rindx, pconvert(rvals), convert(x), fsbeta, cconvert(y));
        else
            sparse_gemv_array<value_type, 'T'>(rows, cols, fsalpha, rpntr, rindx, pconvert(rvals), convert(x), fsbeta, cconvert(y));
    }
    //! \brief Matrix-vector product.
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY, class VectorLikeBuff>
    void gemv(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y, VectorLikeBuff &&) const{
        gemv(trans, alpha, x, beta, y);
    }

    //! \brief Returns 0, added for consistency with CUDA and ROCM.
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
    size_t gemm_buffer_size(char, char, int, int, FSA, VectorLikeB const&, int, FSB, VectorLikeC&&, int) const{ return 0; }
    //! \brief Matrix-matrix product.
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC, class VectorLikeT>
    void gemm(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &&C, int ldc, VectorLikeT &&) const{
        gemm(transa, transb, b_rows, b_cols, alpha, B, ldb, beta, C, ldc);
    }
    //! \brief Matrix-matrix product.
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
    void gemm(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &&C, int ldc) const{
        check_types(rvals, B, C);
        int M = (is_n(transa)) ? rows : cols;
        int N = (is_n(transb)) ? b_cols : b_rows;
        int K = (is_n(transa)) ? cols : rows;
        check_set_size((hala_abs(beta) == 0.0), C, ldc, N );
        assert( valid::sparse_gemm(transa, transb, M, N, K, rpntr[rows], rpntr, rindx, rvals, B, ldb, C, ldc) );

        auto const fsalpha = get_cast<value_type>(alpha);
        auto const fsbeta  = get_cast<value_type>(beta);

        sparse_gemm_array<value_type>(transa, transb, M, N, K, fsalpha, pconvert(rpntr), pconvert(rindx), pconvert(rvals), convert(B), ldb, fsbeta, cconvert(C), ldc);
    }

private:
    int const *rpntr, *rindx;
    value_type const* rvals;
    int rows, cols, nnz;
};

/*!
 * \ingroup HALASPARSE
 * \brief Factory method to construct a mixed sparse matrix.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_sparse_matrix(int num_rows, int num_cols, int num_nz, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    check_types(vals);
    check_types_int(pntr, indx);
    using scalar_type = get_scalar_type<VectorLikeV>;
    return cpu_sparse_matrix<scalar_type>(num_rows, num_cols, num_nz, pntr, indx, vals);
}
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_sparse_matrix(int num_cols, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    check_types(vals);
    check_types_int(pntr, indx);
    using scalar_type = get_scalar_type<VectorLikeV>;
    return cpu_sparse_matrix<scalar_type>(num_cols, pntr, indx, vals);
}
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_sparse_matrix(cpu_engine const&, int num_rows, int num_cols, int num_nz,
                        VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    return make_sparse_matrix(num_rows, num_cols, num_nz, pntr, indx, vals);
}
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_sparse_matrix(cpu_engine const&, int num_cols, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    return make_sparse_matrix(num_cols, pntr, indx, vals);
}

#ifndef __HALA_DOXYGEN_SKIP
template<typename MatrixType, typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
size_t sparse_gemv_buffer_size(MatrixType const &matrix, char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y){
    return matrix.gemv_buffer_size(trans, alpha, x, beta, std::forward<VectorLikeY>(y));
}
template<typename MatrixType, typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
void sparse_gemv(MatrixType const &matrix, char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y){
    matrix.gemv(trans, alpha, x, beta, std::forward<VectorLikeY>(y));
}
template<typename MatrixType, typename FPa, class VectorLikeX, typename FPb, class VectorLikeY, class VectorLikeBuff>
void sparse_gemv(MatrixType const &matrix, char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y, VectorLikeBuff &&temp){
    matrix.gemv(trans, alpha, x, beta, std::forward<VectorLikeY>(y), std::forward<VectorLikeBuff>(temp));
}
template<typename MatrixType, typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
size_t sparse_gemm_buffer_size(MatrixType const &matrix, char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &&C, int ldc){
    return matrix.gemm_buffer_size(transa, transb, b_rows, b_cols, alpha, B, ldb, beta, std::forward<VectorLikeC>(C), ldc);
}
template<typename MatrixType, typename FSA, class VectorLikeB, typename FSB, class VectorLikeC, class VectorLikeBuff>
void sparse_gemm(MatrixType const &matrix, char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &&C, int ldc, VectorLikeBuff &&temp){
    matrix.gemm(transa, transb, b_rows, b_cols, alpha, B, ldb, beta, std::forward<VectorLikeC>(C), ldc, std::forward<VectorLikeBuff>(temp));
}
template<typename MatrixType, typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
void sparse_gemm(MatrixType const &matrix, char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &&C, int ldc){
    matrix.gemm(transa, transb, b_rows, b_cols, alpha, B, ldb, beta, std::forward<VectorLikeC>(C), ldc);
}
#endif



/*!
 * \ingroup HALASPARSE
 * \brief Wrapper around triangular sparse matrix and the associated analysis.
 *
 * There is no sparse standard that is as universal as BLAS, there are multiple standards with pros and cons,
 * the HALA defined standard is designed as a generalization to the cuSparse library (much as it already includes BLAS and cuBLAS).
 * The cuSparse standard currently supports two APIs the old has been deprecated in CUDA 10 (will be removed in CUDA 11)
 * and the new standard that was introduced partially in CUDA 9 and more so in CUDA 10.
 * Both standards use a series of analysis (and Fortran77 style of external buffers) to perform triangular solves,
 * the HALA API wraps and caches all relevant analysis procedures in the triangular matrix classes,
 * one for each engine.
 *
 * The triangular matrix objects hold \b non-owning reference to the internal data of the three vector-like classes,
 * i.e., pntr, indx, and vals.
 *
 */
template<typename T>
class cpu_triangular_matrix{
public:
    //! \brief Floating point type used by the values.
    using value_type = std::remove_cv_t<T>;

    //! \brief The type of the associated engine.
    using engine_type = cpu_engine;

    //! \brief Constructor, favor using the factory method hala::make_triangular_matrix() (cpu_engine version).
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    cpu_triangular_matrix(cpu_engine const &cengine, char uplo, char diag,
                          VectorLikeP const &mpntr, VectorLikeI const &mindx, VectorLikeV const &mvals, char policy)
        : rengine(std::ref(cengine)), rpntr(get_data(mpntr)), rindx(get_data(mindx)), rvals(get_standard_data(mvals)),
        nrows(get_size_int(mpntr) - 1), nz(get_size_int(mindx)), ruplo(uplo), rdiag(diag), rpolicy(policy)
    {
        assert( check_uplo(uplo) );
        assert( check_diag(diag) );
    }

    //! \brief The class is non-copyable.
    cpu_triangular_matrix(cpu_triangular_matrix const &other) = delete;
    //! \brief The class is non-copyable.
    cpu_triangular_matrix& operator = (cpu_triangular_matrix const &other) = delete;

    //! \brief The class is trivially movable.
    cpu_triangular_matrix& operator = (cpu_triangular_matrix &&other) = default;
    //! \brief The class is trivially movable.
    cpu_triangular_matrix(cpu_triangular_matrix &&other) = default;

    //! \brief Destructor, nothing to do in the cpu case.
    ~cpu_triangular_matrix() = default;

    //! \brief Return the pntr array.
    int const* pntr() const{ return rpntr; }
    //! \brief Return the indx array.
    int const* indx() const{ return rindx; }
    //! \brief Return the vals array (wrapped for conversion).
    auto vals() const{ return pconvert(rvals); }

    //! \brief Return the instance of the engine.
    cpu_engine const& engine(){ return rengine.get(); }

    //! \brief Indicates whether the matrix is upper or lower triangular.
    char uplo() const{ return ruplo; }
    //! \brief Indicates whether the matrix is unit or non-unit diagonal.
    char diag() const{ return rdiag; }

    //! \brief Returns the number of rows.
    int rows() const{ return nrows; }
    //! \brief Returns the number of non-zeros.
    int nnz() const{ return nz; }
    //! \brief Returns the stored policy.
    char policy(){ return rpolicy; }

private:
    std::reference_wrapper<cpu_engine const> rengine;
    int const *rpntr, *rindx; // reference pntr, indx, and vals
    T const *rvals;

    int nrows, nz; // num rows and num-non-zeros
    char ruplo, rdiag, rpolicy; // reference uplo, diag and policy
};

/*!
 * \brief Factory method to construct a cpu variant of a triangular matrix.
 *
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_triangular_matrix(cpu_engine const &engine, char uplo, char diag,
                            VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, char policy = 'N'){
    check_types(vals);
    check_types_int(pntr, indx);
    assert( get_size(indx) == get_size(vals) );
    using scalar_type = get_scalar_type<VectorLikeV>;
    return cpu_triangular_matrix<scalar_type>(engine, uplo, diag, pntr, indx, vals, policy);
}

/*!
 * \internal
 * \ingroup HALACORE
 * \brief Switches between true-type and false-type depending whether \b TriangularMatrix::engine_type is cpu_engine.
 *
 * \endinternal
 */
template<class TriangularMatrix, typename = void> struct is_on_cpu : std::false_type{};

/*!
 * \internal
 * \ingroup HALACORE
 * \brief Specialization indicating that the engine is indeed the cpu_engine.
 *
 * \endinternal
 */
template<class TriangularMatrix>
struct is_on_cpu<TriangularMatrix, std::enable_if_t<std::is_same<typename TriangularMatrix::engine_type, cpu_engine>::value, void>> : std::true_type{};


}

#endif
