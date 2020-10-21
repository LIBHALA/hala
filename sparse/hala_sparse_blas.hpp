#ifndef __HALA_SPARSE_BLAS_HPP
#define __HALA_SPARSE_BLAS_HPP
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
 * \file hala_sparse_blas.hpp
 * \brief HALA Sparse Linear Algebra
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALASPARSECORE
 *
 * Contains the HALA Sparse matrix-vector and matrix-matrix operations.
 * \endinternal
 */

#include "hala_sparse_core.hpp"

namespace hala{

/*!
 * \ingroup HALASPARSE
 * \addtogroup HALASPARSEBLAS Sparse BLAS Multiply and Solve
 *
 * \par Multiply and Solve
 * Sparse matrix times dense matrix and vector multiplication and solve operations.
 */

/*!
 * \ingroup HALASPARSEBLAS
 * \brief Sparse matrix vector multiplication.
 *
 * Computes \f$ y = \alpha A x + \beta y \f$.
 * - The matrix is stored in row-compressed format and represented by the variables \b pntr, \b indx, \b vals
 * - \b x is a vector of the same type as \b vals
 * - \b y is a vector of the same type as \b x and with size matching the number of rows
 * - \b alpha and \b beta are scalars convertible to the type of the vectors
 *
 * \b Note: the number of columns of the matrix is implicit and the size of \b x cannot be checked.
 *
 * Overloads:
 * \code
 *  sparse_gemv(trans, M, N, alpha, pntr, indx, vals, x, beta, y);
 *  sparse_gemv(engine, trans, M, N, alpha, pntr, indx, vals, x, beta, y);
 *  sparse_gemv(sparse_matrix, trans, alpha, x, beta, y);
 * \endcode
 * The \b sparse_matrix can be either of the classes hala::cpu_sparse_matrix or hala::gpu_sparse_matrix.
 */
template<typename FSA, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, typename FSB, class VectorLikeY>
void sparse_gemv(char trans, int M, int N,
                 FSA alpha, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, VectorLikeX const &x,
                 FSB beta, VectorLikeY &&y){
    check_set_size((hala_abs(beta) == 0.0), y, (is_n(trans)) ? M : N );
    assert( valid::sparse_gemv(trans, M, N, get_back(pntr), pntr, indx, vals, x, y) );

    using scalar_type = get_standard_type<VectorLikeV>;
    auto const fsalpha = get_cast<scalar_type>(alpha);
    auto const fsbeta  = get_cast<scalar_type>(beta);

    if (is_n(trans))
        sparse_gemv_array<scalar_type, 'N'>(M, N, fsalpha, convert(pntr), convert(indx), convert(vals), convert(x), fsbeta, cconvert(y));
    else if (is_c(trans))
        sparse_gemv_array<scalar_type, 'C'>(M, N, fsalpha, convert(pntr), convert(indx), convert(vals), convert(x), fsbeta, cconvert(y));
    else
        sparse_gemv_array<scalar_type, 'T'>(M, N, fsalpha, convert(pntr), convert(indx), convert(vals), convert(x), fsbeta, cconvert(y));
}

/*!
 * \ingroup HALASPARSEBLAS
 * \brief Sparse matrix-matrix multiplication.
 *
 * Computes \f$ C = \alpha op(A) op(B) + \beta C \f$ where \b A is a sparse matrix.
 * - \b transa and \b transb are upper/lower case N/T/C indicating the \b op and \b A and \b B
 * - the possible \b op variants are identity (N), transpose (T), and conjugate-transpose (C)
 * - \b A is M by K for \b transa begin identity and K by M otherwise
 * - \b B is K by N for \b transb begin identity and N by K otherwise
 * - \b C is M by N
 * - \b K is always the middle dimension of the product
 * - \b ldc is minimum K when transb is N and minimum N when transb is T or C
 * - \b ldc is minimum M
 * - everything other than A is identical to hala::gemm()
 *
 * Overloads:
 * \code
 *  hala::sparse_gemm(transa, trnasb, M, N, K,
 *                    alpha, pntr, indx, vals, B, beta, C, ldb =-1, ldc=-1);
 *  hala::sparse_gemm(engine, transa, trnasb, M, N, K,
 *                    alpha, pntr, indx, vals, B, ldb, beta, C, ldc);
 *  hala::sparse_gemm(engine, transa, trnasb, M, N, K,
 *                    alpha, pntr, indx, vals, B, beta, C, ldb =-1, ldc=-1);
 * \endcode
 * The leading dimensions assume the minimum acceptable value, i.e., as if the matrices have no padding.
 */
template<typename FSA, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeB, typename FSB, class VectorLikeC>
void sparse_gemm(char transa, char transb, int M, int N, int K,
                 FSA alpha, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                 VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &&C, int ldc){

    check_types(vals, B, C);
    check_types_int(pntr, indx);
    check_set_size((hala_abs(beta) == 0.0), C, ldc, N );
    assert( valid::sparse_gemm(transa, transb, M, N, K, get_back(pntr), pntr, indx, vals, B, ldb, C, ldc) );

    using scalar_type = get_standard_type<VectorLikeV>;
    auto const fsalpha = get_cast<scalar_type>(alpha);
    auto const fsbeta  = get_cast<scalar_type>(beta);

    sparse_gemm_array<scalar_type>(transa, transb, M, N, K, fsalpha, convert(pntr), convert(indx), convert(vals), convert(B), ldb, fsbeta, cconvert(C), ldc);
}

/*!
 * \ingroup HALASPARSEBLAS
 * \brief Solves the triangular system of equations.
 *
 * Computes \f$ A x = \alpha b \f$, where \b A is a sparse triangular matrix, and \b x and \b b are dense vectors.
 * - \b trans is lower/upper case U/T/C indicating whether to take the transpose
 * - \b tri is a triangular matrix constructed with hala::make_triangular_matrix()
 * - \b b and \b x are the right-hand side and solution vectors
 * - \b b must have size at least tri.rows()
 * - \b x will be resized to tri.rows(), if the current size if not sufficient
 *
 * Overloads are provided for triangular matrices constructed with all compute engines,
 * note that x and b must be vectors located in CPU memory for the case when \b tri
 * is constructed using the cpu_engine or the mixed_engine, and GPU memory for the cuda_engine.
 */
template<class TriangularMat, typename FPA, class VectorLikeB, class VectorLikeX>
void sparse_trsv(char trans, TriangularMat const &tri, FPA alpha, VectorLikeB const &b, VectorLikeX &&x){
    tri.trsv(trans, alpha, b, x);
}

/*!
 * \ingroup HALASPARSEBLAS
 * \brief Solves the triangular system of equations with multiple right hand sides.
 *
 * Computes \f$ A X = \alpha B \f$, where \b A is a sparse triangular matrix, and \b X and \b B are general dense matrices.
 * - \b transa and \b transb are upper/lower case N/T/C indicating whether to use identity (N), transpose or conjugate transpose on \b A and \b B
 * - \b nrhs indicates the number of right-hand-sides to use
 * - \b tri is a triangular matrix constructed with hala::make_triangular_matrix()
 * - \b B (on entry) is the right-hand-side of the equation, and (on exit) it is overwritten with the solution
 * - \b ldb is the leading dimension of \b B, minimum the number of rows of \b tri (when transb is N) or \b nrhs (transb is T or C)
 *
 * Overloads are provided for triangular matrices constructed with all compute engines,
 * note that x and b must be vectors located in CPU memory for the case when \b tri
 * is constructed using the cpu_engine or the mixed_engine, and GPU memory for the cuda_engine.
 */
template<class TriangularMat, typename FPA, class VectorLikeB>
void sparse_trsm(char transa, char transb, int nrhs, TriangularMat const &tri, FPA alpha, VectorLikeB &&B, int ldb = -1){
    tri.trsm(transa, transb, nrhs, alpha, B, ldb);
}

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)

template<typename sp_matrix, typename FPa, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, typename FPb, class VectorLikeY>
void sparse_gemv(sp_matrix const &matrix, char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &y){
    matrix.gemv(trans, alpha, x, beta, y);
}

template<typename FSA, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, typename FSB, class VectorLikeY>
inline void sparse_gemv(cpu_engine const&,char trans, int M, int N,
                        FSA alpha, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, VectorLikeX const &x,
                        FSB beta, VectorLikeY &&y){
    sparse_gemv(trans, M, N, alpha, pntr, indx, vals, x, beta, y);
}

template<typename FSA, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeB, typename FSB, class VectorLikeC>
inline void sparse_gemm(char transa, char transb, int M, int N, int K,
                        FSA alpha, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                        VectorLikeB const &B, FSB beta, VectorLikeC &&C, int ldb = -1, int ldc = -1){
    valid::default_ld(is_n(transb), K, N, ldb);
    valid::default_ld(M, ldc);
    sparse_gemm(transa, transb, M, N, K, alpha, pntr, indx, vals, B, ldb, beta, C, ldc);
}

template<typename FSA, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeB, typename FSB, class VectorLikeC>
inline void sparse_gemm(cpu_engine const&, char transa, char transb, int M, int N, int K,
                        FSA alpha, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                        VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &&C, int ldc){
    sparse_gemm(transa, transb, M, N, K, alpha, pntr, indx, vals, B, ldb, beta, C, ldc);
}

template<typename FSA, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeB, typename FSB, class VectorLikeC>
inline void sparse_gemm(cpu_engine const&, char transa, char transb, int M, int N, int K,
                        FSA alpha, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                        VectorLikeB const &B, FSB beta, VectorLikeC &&C, int ldb = -1, int ldc = -1){
    sparse_gemm(transa, transb, M, N, K, alpha, pntr, indx, vals, B, beta, C, ldb, ldc);
}

#endif

}

#endif
