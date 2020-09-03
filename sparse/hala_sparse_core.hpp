#ifndef __HALA_SPARSE_CORE_HPP
#define __HALA_SPARSE_CORE_HPP
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
 * \file hala_sparse_core.hpp
 * \brief HALA Sparse Linear Algebra
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALASPARSECORE
 *
 * Contains the HALA Sparse Matrix Operations.
 * \endinternal
 */

#include "hala_sparse_structs.hpp"

namespace hala{

/*!
 * \ingroup HALASPARSE
 * \addtogroup HALASPARSECORE Core Sparse Operations
 *
 * \par Core Operations
 * The core linear sparse operations include the matrix vector product, pattern partition and factorization.
 */

/*!
 * \ingroup HALASPARSECORE
 * \brief Fills \b diag with the indexes in \b indx of the diagonal entries of the matrix.
 *
 * On exit, indx[diag[i]] = i, and pntr[i] <= diag[i] < pntr[i+1].
 * - This assumes that the diagonal entry is included in the pattern (even if it is numerically zero)
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeD>
void get_diagonal_index(VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeD &diag){
    check_types_int(pntr, indx, diag);
    int num_rows = get_size_int(pntr) - 1;
    assert(num_rows > 0);
    assert( check_size(indx, get_back(pntr)) );
    check_set_size(assume_output, diag, num_rows);

    auto pntr_data = get_data(pntr);
    auto indx_data = get_data(indx);
    auto diag_data = get_data(diag);

    for(int i=0; i<num_rows; i++){
        diag_data[i] = pntr_data[i];
        while(indx_data[diag_data[i]] < i) diag_data[i]++;
    }
}

/*!
 * \ingroup HALASPARSECORE
 * \brief Split the patter into upper and lower portions, i.e., make two sparse matrices.
 *
 * Split the pattern defined by \b pntr, \b indx with pre-computed diagonal \b diag,
 * effectively makes two sparse matrices.
 * The diagonal entries will be included in both patterns.
 */
template<class VectorLikeP,  class VectorLikeI,  class VectorLikeD,
         class VectorLikePU, class VectorLikeIU, class VectorLikePL, class VectorLikeIL>
void split_pattern(VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeD const &diag,
                   VectorLikePU &upper_pntr, VectorLikeIU &upper_indx,
                   VectorLikePL &lower_pntr, VectorLikeIL &lower_indx){
    check_types_int(pntr, indx, diag, upper_pntr, upper_indx, lower_pntr, lower_indx);
    int num_rows = get_size_int(pntr) - 1;
    assert( num_rows > 0 );
    assert( check_size(indx, get_back(pntr)) );

    force_size(upper_pntr, num_rows + 1);
    force_size(lower_pntr, num_rows + 1);

    split_pattern_array<true>(num_rows, get_data(pntr), get_data(indx), get_data(diag),
                              get_data(upper_pntr), get_data(upper_indx), get_data(lower_pntr), get_data(lower_indx));

    force_size(upper_indx, get_back(upper_pntr));
    force_size(lower_indx, get_back(lower_pntr));

    split_pattern_array<false>(num_rows, get_data(pntr), get_data(indx), get_data(diag),
                               get_data(upper_pntr), get_data(upper_indx), get_data(lower_pntr), get_data(lower_indx));
}

/*!
 * \ingroup HALASPARSECORE
 * \brief Split the values assuming the pattern has been split already.
 *
 * Split the values in \b vals into the lower and upper portions of the patterns
 * defined by \b upper_pntr and \b lower_pntr.
 * Assumes the order of the indexes matches the order used in hala::split_pattern().
 * - \b uplo is lower or upper case 'U' or 'L' indicates which section should get the diagonal entry
 * - \b diag is 'U' or 'N' indicates whether the other entry should be set to 1 (Unit) or 0 (non-Unit)
 */
template<class VectorLikeP,  class VectorLikeV,
         class VectorLikePU, class VectorLikePL, class VectorLikeVU, class VectorLikeVL>
void split_values(char uplo, char diag,
                  VectorLikeP const &pntr, VectorLikeV const &vals,
                  VectorLikePU const &upper_pntr, VectorLikePL const &lower_pntr,
                  VectorLikeVU &upper_vals,       VectorLikeVL &lower_vals){
    check_types(vals, upper_vals, lower_vals);
    check_types_int(pntr, upper_pntr, lower_pntr);
    int num_rows = get_size_int(pntr) - 1;
    assert(num_rows > 0);
    assert( check_size(upper_pntr, num_rows + 1) );
    assert( check_size(lower_pntr, num_rows + 1) );
    assert( check_uplo(uplo) );
    assert( check_diag(diag) );

    force_size(upper_vals, get_back(upper_pntr));
    force_size(lower_vals, get_back(lower_pntr));

    using scalar_type = get_standard_type<VectorLikeV>;
    split_values_array<int, scalar_type>(num_rows, uplo, diag, convert(pntr), convert(vals),
                                         convert(upper_pntr), convert(lower_pntr),
                                         convert(upper_vals), convert(lower_vals));
}

/*!
 * \ingroup HALASPARSECORE
 * \brief Split both the pattern and values into two matrices, see \b hala::split_values().
 *
 * Converts a single matrix into two matrices containing the upper and lower parts.
 * The call effectively combines a call to hala::split_pattern() followed by hala::split_values().
 */
template<class VectorLikeP,  class VectorLikeI,  class VectorLikeV,
         class VectorLikePU, class VectorLikeIU, class VectorLikeVU,
         class VectorLikePL, class VectorLikeIL, class VectorLikeVL>
void split_matrix(char uplo, char diag,
                  VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                  VectorLikePU &upper_pntr, VectorLikeIU &upper_indx, VectorLikeVU &upper_vals,
                  VectorLikePL &lower_pntr, VectorLikeIL &lower_indx, VectorLikeVL &lower_vals){
    check_types(vals, upper_vals, lower_vals);
    check_types_int(pntr, indx, upper_pntr, upper_indx, lower_pntr, lower_indx);

    std::vector<int> ddiag;
    get_diagonal_index(pntr, indx, ddiag);
    split_pattern(pntr, indx, ddiag, upper_pntr, upper_indx, lower_pntr, lower_indx);
    split_values(uplo, diag, pntr, vals, upper_pntr, lower_pntr, upper_vals, lower_vals);
}

/*!
 * \ingroup HALASPARSECORE
 * \brief Computes the incomplete Lower-Upper factorization of a sparse matrix.
 *
 * Assuming that the matrix is a diagonally dominant and non-singular,
 * this will compute the incomplete factorization.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeD, class VectorLikeILU>
void factorize_ilu(VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, VectorLikeD const &diag, VectorLikeILU &ilu){
    check_types(vals, ilu);
    check_types_int(pntr, indx, diag);
    int num_rows = get_size_int(pntr) - 1;
    assert(num_rows > 0);
    assert( check_size(diag, num_rows) );
    force_size(ilu, get_size(vals));

    hala::vcopy(vals, ilu);

    using scalar_type = get_scalar_type<VectorLikeV>;
    factorize_ilu_array<scalar_type>(num_rows, get_data(pntr), get_data(indx), get_data(diag), get_data(ilu));
}

/*!
 * \ingroup HALASPARSECORE
 * \brief Overload that will call hala::get_diagonal_index() if the \b diag has incorrect size.
 *
 * If the \b diag is not a const vector and the values have not been computed already,
 * then call hala::get_diagonal_index() and then factorize as if using a const vector.
 * The test whether \b diag has been computed is done by checking the size.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeD, class VectorLikeILU>
void factorize_ilu(VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, VectorLikeD &diag, VectorLikeILU &ilu){
    assert(get_size(vals) > 0);
    auto num_rows = get_size(vals) - 1;
    if (get_size(diag) != num_rows)
        hala::get_diagonal_index(pntr, indx, diag);
    hala::factorize_ilu(pntr, indx, vals, *const_cast<VectorLikeD const*>(&diag), ilu); // add const to the diag
}

/*!
 * \ingroup HALASPARSECORE
 * \brief Overload that will call hala::get_diagonal_index() if the \b diag has incorrect size.
 *
 * If the \b diag is not a const vector and the values have not been computed already,
 * then call hala::get_diagonal_index() and then factorize as if using a const vector.
 * The test whether \b diag has been computed is done by checking the size.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeD, class VectorLikeILU, class VectorLikeX>
void apply_ilu(VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeD &diag, VectorLikeILU const &ilu, VectorLikeX &&x){
    check_types(ilu, x);
    check_types_int(pntr, indx, diag);
    int num_rows = get_size_int(pntr) - 1;
    assert(num_rows > 0);
    assert( check_size(diag, num_rows) );
    assert( check_size(x, num_rows) );
    assert( check_size(indx, get_size(ilu)) );

    using scalar_type = get_scalar_type<VectorLikeILU>;
    apply_ilu_array<scalar_type>(num_rows, get_data(pntr), get_data(indx), get_data(diag), get_data(ilu), get_data(x));
}


}

#endif
