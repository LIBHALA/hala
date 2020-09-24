#ifndef __HALA_SOLVERS_CG_BATCH_HPP
#define __HALA_SOLVERS_CG_BATCH_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */
#include "hala_solvers_cg.hpp"

/*!
 * \internal
 * \file hala_solvers_cg_batch.hpp
 * \brief HALA Conjugate-Gradient Solver.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALASOLVECGBATCH
 *
 * Conjugate gradient solver with multiple simultaneous right-hand-sides.
 * \endinternal
 */

/*!
 * \ingroup HALASOLVERS
 * \addtogroup HALASOLVECGBATCH Batch Conjugate-Gradient Solver
 *
 * A series of templates that implement the preconditioned conjugate-gradient method
 * with multiple simultaneous right-hand-sides.
 */

namespace hala{

/*!
 * \ingroup HALASOLVECGBATCH
 * \brief Conjugate-Gradient method with multiple right-hand sides.
 *
 * Uses the same preconditioned CG method as hala::solve_cg() but solves multiple equations with the same
 * matrix and different right-hand-sides. The advantage of this method is that the matrix product
 * and the preconditioner use matrix-matrix operations which are significantly more efficient.
 * The scalars in the algorithm are represented by vectors and the vectors by matrices; thus,
 * the scalar-scalar operations now use BLAS level 1 (as opposed to scalar arithmetic),
 * the vector-scalar methods use BLAS level 2 (as opposed to level 1), and the matrix-vector methods rely
 * on the most efficient BLAS level 3. The total computing time for solving multiple right-hand-sides
 * is significantly less than solving the systems one at a time.
 * - The vectors are stored contiguously in the columns of the matrix \b B and \b X (using column major format).
 * - The number of right-hand-side equations is inferred from the size of \b pntr
 *  (indicating the number of rows of the matrix) and the size of \b b.
 * - If \b x is provided with sufficient size, it will be used as the initial iterate, otherwise \b x will
 *  be resize and scaled to zero.
 * - The tolerance used in the stopping criteria is used in comparison with the largest of the residual vectors,
 * using the hala::batch_max_norm2() method.
 *
 * This is the base method using a generic preconditioner, see hala::solve_batch_cg_ilu() for the variant
 * that directly uses an ILU preconditioner used by HALA.
 *
 * Overloads:
 * \code
 *  solve_batch_cg(stop, pntr, indx, vals, precon, B, X);
 *  solve_batch_cg(engine, stop, pntr, indx, vals, precon, B, X);
 * \endcode
 * Note that the signature of the preconditioner is slightly different when working with
 * the engine overload. See hala::preconditioner and hala::preconditioner_noe.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int solve_batch_cg(stop_criteria<get_precision_type<VectorLikeV>> const &stop,
                   VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                   preconditioner_noe<VectorLikeV> precon,
                   VectorLikeB const &B, VectorLikeX &X){

    check_types(vals, X, B);
    check_types_int(pntr, indx);

    int num_entries = get_size_int(pntr);
    assert( num_entries > 1 ); // smallest case has one entry
    int num_rhs = get_size_int(B) / (num_entries - 1);

    if (get_size(X) < get_size(B)){
        force_size(X, get_size(B));
        set_zero(get_size(B), X); // if defined on CUDA the entries will not be initialized
    }

    using temp_vector = decltype( new_vector( std::declval<VectorLikeV>() ) );
    using scalar_type = temp_vector;

    int nrows = get_size_int(pntr) - 1;

    auto matrix = make_sparse_matrix(nrows, pntr, indx, vals);
    size_t bsize =  sparse_gemm_buffer_size(matrix, 'N', 'N', nrows, num_rhs, 1.0, B, nrows, 0.0, X, nrows);
    auto work_buffer = new_vector(X);
    set_size(bsize, work_buffer);

    return solve_cg_core<scalar_type>(precon,
                [&](VectorLikeX const &v, temp_vector &r)->void{
                    sparse_gemm(matrix, 'N', 'N', nrows, num_rhs, 1.0, v, nrows, 0.0, r, nrows, work_buffer);
                },
                [&](temp_vector const &v, temp_vector &r)->void{
                    sparse_gemm(matrix, 'N', 'N', nrows, num_rhs, 1.0, v, nrows, 0.0, r, nrows, work_buffer);
                },
                B, X,
                [&]()->scalar_type{ return new_vector(X); },
                [&](scalar_type const &num, scalar_type const &denom, scalar_type &result)->void{ vdivide(num, denom, result); },
                [](scalar_type &src, scalar_type &dest)->void{ dest = std::move(src); },
                [&](scalar_type const &a, temp_vector &r)->void{ batch_scal(num_rhs, a, r); },
                [&](temp_vector const &v, temp_vector &r)->void{ axpy( 1.0, v, r); },
                [&](temp_vector const &v, temp_vector &r)->void{ axpy(-1.0, v, r); },
                [&](scalar_type const &a, temp_vector const &v, VectorLikeX &r)->void{ batch_axpy(num_rhs,  1.0, a, v, r); },
                [&](scalar_type const &a, temp_vector const &v, temp_vector &r)->void{ batch_axpy(num_rhs, -1.0, a, v, r); },
                [&](temp_vector const &v, temp_vector const &r, scalar_type &a)->void{ batch_dot(num_rhs, v, r, a); },
                [&](int i, temp_vector const &r)->bool{ return ((i == stop.max_iter) || (batch_max_norm2(num_rhs, r) < stop.tol)); }
        );
}

#ifndef __HALA_DOXYGEN_SKIP
template<class compute_engine, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
hala_int<compute_engine>
solve_batch_cg(compute_engine const &engine,
               stop_criteria<get_precision_type<VectorLikeV>> const &stop,
               VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
               preconditioner<compute_engine, VectorLikeX> precon,
               VectorLikeB const &B, VectorLikeX &X){
    auto p = bind_engine_vector(engine, pntr);
    auto i = bind_engine_vector(engine, indx);
    auto v = bind_engine_vector(engine, vals);
    auto BB = bind_engine_vector(engine, B);
    auto XX = bind_engine_vector(engine, X);
    return solve_batch_cg(stop, p, i, v,
                          [&](auto const &inx, auto &outr)->void{
                                precon(inx.vector(), outr.vector());
                          }, BB, XX);
}
#ifdef HALA_ENABLE_CUDA
template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int
solve_batch_cg(mixed_engine const &engine,
               stop_criteria<get_precision_type<VectorLikeV>> const &stop,
               VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
               preconditioner<mixed_engine, VectorLikeX> precon,
               VectorLikeB const &B, VectorLikeX &X){
    auto p = engine.gpu().load(pntr);
    auto i = engine.gpu().load(indx);
    auto v = engine.gpu().load(vals);
    auto BB = engine.gpu().load(B);
    auto XX = gpu_bind_vector(engine.gpu(), X);
    return solve_batch_cg(engine.gpu(), stop, p, i, v,
                          [&](auto const &inx, auto &outr)->void{
                                precon(inx, outr);
                          }, BB, XX);
}
#endif
#endif

/*!
 * \ingroup HALASOLVECGBATCH
 * \brief Batch Conjugate-Gradient method with ILU preconditioner.
 *
 * Equivalent to hala::solve_batch_cg() but using the ILU preconditioner \b ilu.
 *
 * Note that the engine used for the \b ilu must match the engine of the vectors.
 *
 * Overloads:
 * \code
 *  solve_batch_cg_ilu(stop, pntr, indx, vals, B, X);
 *  solve_batch_cg_ilu(stop, pntr, indx, vals, ilu, B, X);
 *  solve_batch_cg_ilu(engine, stop, pntr, indx, vals, B, X);
 *  solve_batch_cg_ilu(engine, stop, pntr, indx, vals, ilu, B, X);
 * \endcode
 * If an \b ilu factorization is not provided, it will be computed "on-the-fly"
 * and discarded after the call.
 */
template<class VectorLikeP,  class VectorLikeI, class VectorLikeV,
         class ILUclass, class VectorLikeX, class VectorLikeB>
int solve_batch_cg_ilu(stop_criteria<get_precision_type<VectorLikeV>> const &stop,
                       VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                       ILUclass const &ilu, VectorLikeB const &B, VectorLikeX &X){

    int num_entries = get_size_int(pntr);
    assert( num_entries > 1 ); // smallest case has one entry
    int num_rhs = get_size_int(B) / (num_entries - 1);

    return solve_batch_cg(stop, pntr, indx, vals,
                          [&](auto const &inx, auto &outr)->void{
                              ilu_apply(ilu, inx, outr, num_rhs);
                          }, B, X);
}

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)

template<class compute_engine, class VectorLikeP,  class VectorLikeI, class VectorLikeV,
         class ILUclass, class VectorLikeX, class VectorLikeB>
hala_int<compute_engine>
solve_batch_cg_ilu(compute_engine const &engine,
                   stop_criteria<get_precision_type<VectorLikeV>> const &stop,
                   VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                   ILUclass const &ilu, VectorLikeB const &B, VectorLikeX &X){

    static_assert(std::is_same<typename ILUclass::engine_type, compute_engine>::value,
                  "Using compatible compute engine and ILU preconditioner");

    auto p = bind_engine_vector(engine, pntr);
    auto i = bind_engine_vector(engine, indx);
    auto v = bind_engine_vector(engine, vals);
    auto BB = bind_engine_vector(engine, B);
    auto XX = bind_engine_vector(engine, X);
    return hala::solve_batch_cg_ilu(stop, p, i, v, ilu, BB, XX);
}

#ifdef HALA_ENABLE_CUDA
template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int solve_batch_cg_ilu(mixed_engine const &engine,
             stop_criteria<get_precision_type<VectorLikeV>> const &stop,
             VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
             mixed_ilu<get_standard_type<VectorLikeV>> const &ilu, VectorLikeB const &B, VectorLikeX &X){
    auto gpntr = engine.gpu().load(pntr);
    auto gindx = engine.gpu().load(indx);
    auto gvals = engine.gpu().load(vals);
    auto gB = engine.gpu().load(B);
    auto gX = gpu_bind_vector(engine.gpu(), X);
    return solve_batch_cg_ilu(engine.gpu(), stop, gpntr, gindx, gvals, ilu.cilu(), gB, gX);
}
#endif // end CUDA

template<class VectorLikeP, class VectorLikeI, class VectorLikeV,
         class VectorLikeX, class VectorLikeB>
int solve_batch_cg_ilu(stop_criteria<get_precision_type<VectorLikeV>> const &stop,
                       VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                       VectorLikeB const &B, VectorLikeX &X, char policy = 'N'){

    auto ilu = make_ilu(pntr, indx, vals, policy);

    return hala::solve_batch_cg_ilu(stop, pntr, indx, vals, ilu, B, X);
}

template<class compute_engine, class VectorLikeP,  class VectorLikeI, class VectorLikeV,
         class VectorLikeX, class VectorLikeB>
int solve_batch_cg_ilu(compute_engine const &engine,
                       stop_criteria<get_precision_type<VectorLikeV>> const &stop,
                       VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                       VectorLikeB const &B, VectorLikeX &X, char policy = 'N'){

    auto ilu = make_ilu(engine, pntr, indx, vals, policy);

    return hala::solve_batch_cg_ilu(engine, stop, pntr, indx, vals, ilu, B, X);
}

#endif // end skip docs

}

#endif
