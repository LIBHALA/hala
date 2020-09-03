#ifndef __HALA_SPARSE_ILU_HPP
#define __HALA_SPARSE_ILU_HPP
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
 * \file hala_sparse_ilu.hpp
 * \brief HALA Sparse Linear Algebra
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALASPARSEILU
 *
 * Contains the HALA Sparse ILU preconditioner.
 * \endinternal
 */

#include "hala_sparse_blas.hpp"

namespace hala{

/*!
 * \ingroup HALASPARSE
 * \addtogroup HALASPARSEILU Sparse ILU preconditioner
 *
 * \par Preconditioner
 * The incomplete lower-upper factorization is a method to approximate a sparse matrix
 * by the produce of two sparse triangular matrices with patterns very similar to the original.
 * The ILU method is a proven general preconditioner for Krylov iterative solvers,
 * such as those included in the \ref HALASOLVERS.
 * HALA provides a reference CPU implementation of the algorithm and wrappers
 * around the ILU methods implemented in the cuSparse library.
 */

/*!
 * \ingroup HALASPARSEILU
 * \brief A wrapper around an Incomplete Lower Upper (ILU) factorization.
 *
 * The HALA ILU structs take non-owning references to a compute engine and the row-pointer
 * and indexes vectors describing the non-zero pattern of a sparse matrix.
 * At the constructor phase, an internal copy of the matrix values will be stored inside
 * the struct and the values will be overwritten with those of the factors.
 * The class will also contain opaque mata-data specific to the compute engine
 * used in the factorization and solve.
 *
 * The ILU methods have an apply() method that will perform the solve phase
 * of the factorization. The solve can be performed on one or multiple right
 * hand sides at the same time.
 *
 * Variants:
 * \code
 *  hala::cpu_ilu
 *  hala::cuda_ilu
 *  hala::mixed_ilu
 * \endcode
 * The CPU and CUDA version differ only in some of the internal data structures and
 * the backend implementation, both classes have an apply() method that accepts
 * vectors on the corresponding memory space (CPU or CUDA).
 * The hala::mixed_ilu will take also create internal vectors and copy the sparsity
 * pattern over to the GPU. The apply() method accepts vectors in CPU memory space
 * but the mixed ILU will also hold a public member that is a cuda_ilu.
 * See the associated page for details.
 *
 * To maintain code generality, it is recommended to avoid the constructor directly
 * and use the overloaded factory method hala::make_ilu() with the appropriate compute engine.
 *
 */
template<typename T>
struct cpu_ilu{
    //! \brief Floating point type used by the values.
    using value_type = typename define_standard_type<T>::value_type;

    //! \brief The type of the associated engine.
    using engine_type = cpu_engine;

    //! \brief Alias the engine and sparsity pattern and creates the ILU factor.
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    cpu_ilu(cpu_engine const &cengine, VectorLikeP const &mpntr, VectorLikeI const &mindx, VectorLikeV const &mvals, char policy)
        : rengine(std::ref(cengine)), pntr(get_data(mpntr)), indx(get_data(mindx)), num_rows(get_size_int(mpntr)-1), rpolicy(policy){
        factorize_ilu(mpntr, mindx, mvals, diag, ilu);
    }

    //! \brief Destructor, nothing to do in the cpu case.
    ~cpu_ilu() = default;

    //! \brief The class is non-copyable.
    cpu_ilu(cpu_ilu const &other) = delete;
    //! \brief The class is non-copyable.
    cpu_ilu& operator = (cpu_ilu const &other) = delete;

    //! \brief The class is trivially movable.
    cpu_ilu& operator = (cpu_ilu &&other) = default;
    //! \brief The class is trivially movable.
    cpu_ilu(cpu_ilu &&other) = default;

    //! \brief Applies the inverse of the ILU factors to one or more right-hand-sides \b x and returns the result in \b r.
    template<class VectorLikeX, class VectorLikeR>
    void apply(VectorLikeX const &x, VectorLikeR &&r, int num_rhs = 1) const{
        check_types(x, r);
        check_set_size(assume_output, r, num_rhs, num_rows);
        vcopy(num_rhs * num_rows, x, 1, r, 1);
        assert( check_size(x, num_rows) );

        value_type *rdata = get_standard_data(r);

        for(int i=0; i<num_rhs; i++)
            apply_ilu_array<value_type>(num_rows, pntr, indx, diag.data(), ilu.data(), &rdata[hala_size(i, num_rows)]);
    }

private:
    std::reference_wrapper<cpu_engine const> rengine;
    std::vector<value_type> ilu;
    int const *pntr;
    int const *indx;
    std::vector<int> diag;

    int num_rows;
    char rpolicy;
};

/*!
 * \ingroup HALASPARSEILU
 * \brief Constructs an ILU preconditioner associated with a compute engine.
 *
 * Constructs an ILU factorization of a sparse matrix using the given policy.
 * - \b cengine is the compute engine associated with the matrix and factor
 * - \b pntr, \b indx and \b vals describe a sparse matrix in row-compressed format
 * - \b policy, if set to lower/upper case L, indicates whether to use the level-policy and algorithms
 *  when computing the hala::cuda_ilu factor; the variable has no effect when using the cpu_engine.
 *
 * Overloads:
 * \code
 *  make_ilu(engine, pntr, indx, vals, policy);
 * \endcode
 * An overload is provided for each compute engine, the intended use of this factory method
 * is in conjunction with the \b auto keyword:
 * \code
 *  auto ilu_factor = make_ilu(engine, pntr, indx, vals, policy);
 * \endcode
 * Thus the cpu/cuda/mixed variant of the factor will be inferred from the type of the engine.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_ilu(cpu_engine const &cengine, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, char policy){
    using standard_type = get_standard_type<VectorLikeV>;
    return cpu_ilu<standard_type>(cengine, pntr, indx, vals, policy);
}


}

#endif
