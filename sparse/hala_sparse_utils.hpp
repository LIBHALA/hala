#ifndef __HALA_SPARSE_UTILS_HPP
#define __HALA_SPARSE_UTILS_HPP
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
 * \file hala_sparse_utils.hpp
 * \brief HALA Sparse Linear Algebra
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALASPARSEUTILS
 *
 * Contains the HALA Sparse Matrix Operations.
 * \endinternal
 */

#include "hala_blas.hpp"

/*!
 * \defgroup HALASPARSE HALA Sparse Matrix Templates
 *
 * \par HALA-Sparse Matrix Templates
 * Templates operating with spares matrices in row-compressed format defined by
 * tripples of vectors \b pntr, \b indx and \b vals. The \b pntr vector defines
 * the number of rows and the starting offsets of each row (using zero indexing),
 * the corresponding strides of \b indx hold the indexes of the non-zero entries
 * while the floating point vector \b vals holds the actual values
 * (those could be zero).
 *
 * \par Vector Types
 * The indexing is done with 32-bit \b int values (ensures compatibility with
 * CUDA cuSparse methods). The floating point vectors have the same restrictions
 * as the ones used by BLAS and LAPACK.
 */

/*!
 * \internal
 * \ingroup HALASPARSE
 * \addtogroup HALASPARSEUTILS Utilities for the Sparse Module
 *
 * \par Utilities
 * Basic utility templates used through the sparse module,
 * including simple sparsity pattern analysis and reference implementaton
 * to the sparse multiplication and triangular solve.
 *
 * \endinternal
 */

namespace hala{

/*!
 * \ingroup HALASPARSEUTILS
 * \brief Converts spares matrix to dense format, used raw-arrays.
 *
 * Used by hala::dense_convert().
 */
template<typename T>
void dense_convert_array(int num_rows, int num_cols, int const pntr[], int const indx[], T const vals[], T result[]){
    std::fill_n(result, hala_size(num_rows, num_cols), get_cast<T>(0.0));
    for(int i=0; i<num_rows; i++)
        for(int j=pntr[i]; j<pntr[i+1]; j++)
            result[indx[j] * num_rows + i] = vals[j];
}

/*!
 * \ingroup HALASPARSEUTILS
 * \brief Converts spares matrix to dense format, useful for testing and debugging.
 *
 * Takes a sparse matrix defined by \b pntr, \b indx, and \b vals with given \b num_rows and \b num_cols
 * and converts it to a vector containing the dense matrix.
 * The result is the same type as \b VectorLikeV.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto dense_convert(int num_rows, int num_cols, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    check_types(vals);
    check_types_int(pntr, indx);
    assert( get_size(pntr) > 1 );
    assert( check_size(indx, pntr[num_rows]) );
    assert( check_size(vals, pntr[num_rows]) );

    using scalar_type = get_scalar_type<VectorLikeV>;
    std::vector<scalar_type> result(hala_size(num_rows, num_cols));

    dense_convert_array<scalar_type>(num_rows, num_cols, get_data(pntr), get_data(indx), get_data(vals), get_data(result));

    return result;
}

/*!
 * \ingroup HALASPARSEUTILS
 * \brief Performs sparse matrix-vector multiply using raw-arrays.
 *
 * Uses raw-array interface assuming the arrays have the right dimensions.
 */
template<typename T, char trans>
inline void sparse_gemv_array(int num_rows, int num_cols, T alpha, int const pntr[], int const indx[], T const vals[], T const x[], T beta, T y[]){
    if __HALA_CONSTEXPR_IF__ (trans == 'N'){
        for(int i=0; i<num_rows; i++){
            T sum = get_cast<T>(0.0);
            for(int j=pntr[i]; j<pntr[i+1]; j++)
                sum += vals[j] * x[ indx[j] ];
            y[i] = alpha * sum + beta * y[i];
        }
    }else{
        scal(num_cols, beta, y, 1);
        for(int i=0; i<num_rows; i++)
            for(int j=pntr[i]; j<pntr[i+1]; j++)
                y[indx[j]] += alpha * x[i] * ((trans == 'C') ? cconj(vals[j]) : vals[j]);
    }
}

/*!
 * \ingroup HALASPARSEUTILS
 * \brief Performs sparse matrix-matrix multiply using raw-arrays.
 *
 * Uses raw-array interface assuming the arrays have the right dimensions.
 */
template<typename T>
void sparse_gemm_array(char transa, char transb, int M, int N, int K,
                       T alpha, int const pntr[], int const indx[], T const vals[],
                       T const B[], int ldb, T beta, T C[], int ldc){
    if (is_n(transa)){
        if (is_n(transb)){
            for(int i=0; i<N; i++)
                sparse_gemv_array<T, 'N'>(M, K, alpha, pntr, indx, vals, &B[hala_size(i, ldb)], beta, &C[hala_size(i, ldc)]);
        }else{
            for(int i=0; i<N; i++){
                std::vector<T> x(K);
                vcopy(K, &B[i], ldb, x, 1);
                if (is_c(transb)) for(auto &v : x) v = cconj(v);
                sparse_gemv_array<T, 'N'>(M, K, alpha, pntr, indx, vals, x.data(), beta, &C[hala_size(i, ldc)]);
            }
        }
    }else if(is_c(transa)){
        if (is_n(transb)){
            for(int i=0; i<N; i++)
                sparse_gemv_array<T, 'C'>(K, M, alpha, pntr, indx, vals, &B[hala_size(i, ldb)], beta, &C[hala_size(i, ldc)]);
        }else{
            for(int i=0; i<N; i++){
                std::vector<T> x(K);
                vcopy(K, &B[i], ldb, x, 1);
                if (is_c(transb)) for(auto &v : x) v = cconj(v);
                sparse_gemv_array<T, 'C'>(K, M, alpha, pntr, indx, vals, x.data(), beta, &C[hala_size(i, ldc)]);
            }
        }
    }else{
        if (is_n(transb)){
            for(int i=0; i<N; i++)
                sparse_gemv_array<T, 'T'>(K, M, alpha, pntr, indx, vals, &B[hala_size(i, ldb)], beta, &C[hala_size(i, ldc)]);
        }else{
            for(int i=0; i<N; i++){
                std::vector<T> x(K);
                vcopy(K, &B[i], ldb, x, 1);
                if (is_c(transb)) for(auto &v : x) v = cconj(v);
                sparse_gemv_array<T, 'T'>(K, M, alpha, pntr, indx, vals, x.data(), beta, &C[hala_size(i, ldc)]);
            }
        }
    }
}

/*!
 * \ingroup HALASPARSEUTILS
 * \brief Split the patter into upper and lower portions using raw-arrays.
 *
 * Uses raw-array interface assuming the arrays have the right dimensions.
 */
template<bool set_pntrs, typename T = int>
void split_pattern_array(T num_rows,     T const pntr[], T const indx[], T const diag[],
                         T upper_pntr[], T upper_indx[], T lower_pntr[], T lower_indx[]){
    if (set_pntrs){
        upper_pntr[0] = 0;
        lower_pntr[0] = 0;

        for(T i=0; i<num_rows; i++){
            lower_pntr[i+1] = lower_pntr[i] + diag[i] - pntr[i] + 1;
            upper_pntr[i+1] = upper_pntr[i] + pntr[i+1] - diag[i];
        }
    }else{
        for(T i=0; i<num_rows; i++){
            std::copy_n(indx + pntr[i], lower_pntr[i+1] - lower_pntr[i], lower_indx + lower_pntr[i]);
            std::copy_n(indx + diag[i], upper_pntr[i+1] - upper_pntr[i], upper_indx + upper_pntr[i]);
        }
    }
}

/*!
 * \ingroup HALASPARSECORE
 * \brief Split the values assuming the pattern has been split already, uses raw-arrays.
 *
 * Uses raw-array interface assuming the arrays have the right dimensions.
 */
template<typename Tindex, typename T>
void split_values_array(Tindex num_rows, char uplo, char diag,
                        Tindex const pntr[],       T      const vals[],
                        Tindex const upper_pntr[], Tindex const lower_pntr[],
                        T upper_vals[],            T lower_vals[]){

    T alpha = is_n(diag) ? get_cast<T>(0.0) : get_cast<T>(1.0);

    for(Tindex i=0; i<num_rows; i++){
        Tindex num_lower = lower_pntr[i+1] - lower_pntr[i];
        std::copy_n(vals + pntr[i], num_lower, lower_vals + lower_pntr[i]);
        std::copy_n(vals + pntr[i] + num_lower -1,
                    upper_pntr[i+1] - upper_pntr[i],
                    upper_vals + upper_pntr[i]);

        if (is_l(uplo)) // set upper entry to alpha
            upper_vals[upper_pntr[i]] = alpha;
        else // set the lower entry to alpha
            lower_vals[lower_pntr[i+1] - 1] = alpha;
    }
}

/*!
 * \ingroup HALASPARSEUTILS
 * \brief Performs ILU factorization using raw-arrays.
 *
 * Uses raw-array interface assuming the arrays have the right dimensions.
 */
template<typename T>
inline void factorize_ilu_array(int num_rows, int const pntr[], int const indx[], int const diag[], T ilu[]){
    for(int i=0; i<num_rows; i++){
        auto u = ilu[diag[i]];
        //#pragma omp parallel for
        for(int j=i+1; j<num_rows; j++){
            auto jc = pntr[j];
            while(indx[jc] < i) jc++;
            if (indx[jc] == i){
                ilu[jc] /= u;
                auto l = ilu[jc];
                auto ik = diag[i]+1;
                auto jk = jc+1;
                while((ik<pntr[i+1]) && (jk<pntr[j+1])){
                    if (indx[ik] == indx[jk]){
                        ilu[jk] -= l * ilu[ik];
                        ik++; jk++;
                    }else if (indx[ik] < indx[jk]){
                        ik++;
                    }else{
                        jk++;
                    }
                }
            }
        }
    }
}

/*!
 * \ingroup HALASPARSEUTILS
 * \brief Applies ILU factorization on a vector.
 *
 * Uses raw-array interface assuming the arrays have the right dimensions.
 */
template<typename T>
inline void apply_ilu_array(int num_rows, int const pntr[], int const indx[], int const diag[], T const ilu[], T x[]){
    // action of the preconditioner
    for(int i=1; i<num_rows; i++)
        for(int j=pntr[i]; j<diag[i]; j++)
            x[i] -= ilu[j] * x[indx[j]];

    for(int i=num_rows-1; i>=0; i--){
        for(int j=diag[i]+1; j<pntr[i+1]; j++)
            x[i] -= ilu[j] * x[indx[j]];

        x[i] /= ilu[diag[i]];
    }
}

/*!
 * \ingroup HALASPARSEUTILS
 * \brief Solves a sparse triangular system of equations, raw-array version.
 *
 * Uses raw-array interface assuming the arrays have the right dimensions.
 */
template<char diag, typename T, typename Tindex = int>
void sparse_trsv_array(char uplo, char trans, Tindex num_rows,
                       T alpha, Tindex const pntr[], Tindex const indx[], T const vals[], T const b[], T x[]){
    if (is_n(trans)){
        if (uplo == 'L'){
            x[0] = (diag == 'U') ?  alpha * b[0] : alpha * b[0] / vals[0];
            for(int i=1; i<num_rows; i++){
                T s = get_cast<T>(0.0);
                for(int j=pntr[i]; j<pntr[i+1]-1; j++)
                    s += vals[j] * x[indx[j]];
                x[i] = (diag == 'U') ?  (alpha * b[i] - s) : (alpha * b[i] - s) / vals[ pntr[i+1]-1 ];
            }
        }else{
            x[num_rows - 1] = (diag == 'U') ?  alpha * b[num_rows - 1] : alpha * b[num_rows - 1] / vals[ pntr[num_rows] -1 ];
            for(int i=num_rows-2; i>=0; i--){
                T s = get_cast<T>(0.0);
                for(int j=pntr[i]+1; j<pntr[i+1]; j++)
                    s += vals[j] * x[indx[j]];
                x[i] = (diag == 'U') ?  (alpha * b[i] - s) : (alpha * b[i] - s) / vals[ pntr[i] ];
            }
        }
    }else if (is_c(trans)){ // conjugate transpose
        hala::scal(num_rows, 0.0, x, 1);
        if (uplo == 'L'){
            for(int i=num_rows-1; i>=0; i--){
                x[i] = (diag == 'U') ?  (alpha * b[i] - x[i]) : (alpha * b[i] - x[i]) / hala::cconj(vals[ pntr[i+1] - 1 ]);
                for(int j=pntr[i]; j<pntr[i+1] - 1; j++) // last entry is the diagonal, do not update
                    x[indx[j]] += x[i] * hala::cconj(vals[j]);
            }
        }else{
            for(int i=0; i<num_rows; i++){
                x[i] = (diag == 'U') ?  (alpha * b[i] - x[i]) : (alpha * b[i] - x[i]) / hala::cconj(vals[ pntr[i] ]);
                for(int j=pntr[i] + 1; j<pntr[i+1]; j++)
                    x[indx[j]] += x[i] * hala::cconj(vals[j]);
            }
        }
    }else{ // non-conj transpose
        hala::scal(num_rows, 0.0, x, 1);
        if (uplo == 'L'){
            for(int i=num_rows-1; i>=0; i--){
                x[i] = (diag == 'U') ?  (alpha * b[i] - x[i]) : (alpha * b[i] - x[i]) / vals[ pntr[i+1] - 1 ];
                for(int j=pntr[i]; j<pntr[i+1] - 1; j++) // last entry is the diagonal, do not update
                    x[indx[j]] += x[i] * vals[j];
            }
        }else{
            for(int i=0; i<num_rows; i++){
                x[i] = (diag == 'U') ?  (alpha * b[i] - x[i]) : (alpha * b[i] - x[i]) / vals[ pntr[i] ];
                for(int j=pntr[i] + 1; j<pntr[i+1]; j++)
                    x[indx[j]] += x[i] * vals[j];
            }
        }
    }
}

}

#endif
