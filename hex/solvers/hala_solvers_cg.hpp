#ifndef __HALA_SOLVERS_CG_HPP
#define __HALA_SOLVERS_CG_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */
#include "hala_solvers_core.hpp"

/*!
 * \internal
 * \file hala_solvers_cg.hpp
 * \brief HALA Conjugate-Gradient Solver.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALASOLVECG
 *
 * Conjugate gradient solver.
 * \endinternal
 */

/*!
 * \ingroup HALASOLVERS
 * \addtogroup HALASOLVECG Conjugate-Gradient Solver
 *
 * A series of templates that implement the preconditioned conjugate-gradient method
 * with several options to choose from different preconditioners.
 */

namespace hala{

/*!
 * \internal
 * \ingroup HALASOLVECG
 * \brief An overly-generalized template that implement the conjugate gradient method.
 *
 * The template implements the most generalized version of the conjugate-gradient method
 * with callable methods to represent the vector operations and complete disregard of
 * vector types or even dimensions.
 *
 * The reason for the over-generalization is two-fold:
 * -# keep the method agnostic with respect to the user-provided containers and to ensure
 *    interoperability with all containers with HALA-compatible specializations
 * -# allow for batch solves where multiple right-hand-sides and unknown vectors are
 *    resolved in one run of the algorithm taking advantage of the faster matrix-matrix
 *    methods, i.e., improving performance over multiple independent solves
 *
 * In the fist case, the signatures of the vector methods depends on too many potential parameters
 * depending on the storage matrix, right-hand-side, the unknown vector and the internal vector.
 * In the second case, the scalar is in fact a vector and vectors are matrices, thus the signatures
 * are even more complex and scalar operations have to be performed with vector methods too.
 * The scalar must be trivially movable or at least trivially copiable with the latter creating
 * a potential performance penalty.
 * Thus, to keep sanity, the signatures are removed and we document the actual action of
 * methods:
 *
 * - \b engine is a hala compute engine, e.g., hala::cpu_engine; the compute engine is used to create
 *      the temporary vectors with hala::vcopy (engine, x) signature and hala::get_vector().
 * - \b apply_preconditioner(x, r) takes input x and returns r, both vectors are internal
 * - \b apply_operator(x, r) takes input x and returns r; x is the initial guess vector r is internal
 * - \b apply_operator_internal(x, r) same as apply operator but both vectors are internal
 * - \b b and \b x are the right-hand-side and initial-iterate/final-solution, user defined vectors
 * - \b scalar_make() returns a default scalar, doesn't have to be initialized but must have the correct
 *      size and type for use, e.g., double in single solve mode and std::vector<double> in batch solve mode
 * - \b scalar_division(numerator, denominator, result) must compute result = numerator / denominator
 * - \b scalar_move(old, new) move the value of old into the new, old will leave scope right afterwards
 * - \b vector_copy(x, y), computes y = x, both vectors are internal
 * - \b vector_scale(scalar, vector), computes vector *= scalar, the vector is internal
 * - \b vector_add(x, y), computes y += x, both vectors are internal
 * - \b vector_sub(x, y), computes y -= x, both vectors are internal
 * - \b vector_sadd(alpha, internal, result), computes result += alpha * internal, alpha is a scalar,
 *      internal is an internal vector, result is actually the template input \b x (i.e., the solution)
 * - \b vector_ssub(scalar, x, y), computes y -= scalar * x, where x and y are internal vectors
 * - \b vector_dot(x, y, scalar), computes scalar = dot(x, y), where x and y are internal vectors
 * - \b check_converged(iterations, residual) returns whether the iteration should terminate or continue,
 *      the iterations is an int indicating the number of operator/preconditioner applications and
 *      residual is the vector with the current residual.
 *
 * - \b returns the total number of iterations.
 *
 * \endinternal
 */
template<typename hscalar,
         typename hvectorx, typename hvectorb,
         typename htypea, typename htypeb, typename htypec, typename htyped, typename htypef,
         typename htypeg, typename htypeh, typename htypei, typename htypej, typename htypek, typename htypel,
         typename htypem, typename htypen>
int solve_cg_core(htypea   apply_preconditioner,
                  htypeb   apply_operator,
                  htypen   apply_operator_internal,
                  hvectorx const &b, hvectorb &x,
                  htypem  scalar_make,     // return a scalar, useful with auto-initialize
                  htypec  scalar_division,
                  htyped  scalar_move,
                  htypef  vector_scale,
                  htypeg  vector_add,
                  htypeh  vector_sub,
                  htypei  vector_sadd,
                  htypej  vector_ssub,
                  htypek  vector_dot,
                  htypel  check_converged){
    auto r = new_vector(b);
    vcopy(b, r);

    auto p = new_vector(x);
    apply_operator(x, p);
    vector_sub(p, r); // r = b - A x

    auto Ap = new_vector(x);
    auto z  = new_vector(x);

    apply_preconditioner(r, z); // z = E^-T E^-1 r

    auto zr  = scalar_make();
    auto a   = scalar_make();

    vcopy(z, p); // p = z

    vector_dot(r, z, zr); // zr = <r, z>

    int iterations = 1; // applied one Ax with preconditioner
    bool iterate = true;

    while(iterate){
        apply_operator_internal(p, Ap); // Ap = A * p
        iterations++;

        auto nzr = scalar_make();
        vector_dot(p, Ap, nzr);
        scalar_division(zr, nzr, a); // the correction term alpha = <r, z> / <p, A p>

        vector_sadd(a,  p, x); // x += a * p, x is the new solution state
        vector_ssub(a, Ap, r); // r -= a * ap, r is the new residual

        iterate = !check_converged(iterations, r); // if not converged, keep going

        if (iterate){ // skip this part of done already
            apply_preconditioner(r, z); // z = E^-T E^-1 r

            // keep the new zr = dot(r, z), and get p = z - beta * p, where beta = new_zr / old_zr
            vector_dot(r, z, nzr);
            scalar_division(nzr, zr, a); // p-correction alpha = new <r, z> / old <r, z>

            vector_scale(a, p);
            vector_add(z, p); // p = z + p

            scalar_move(nzr, zr); // keep nzr = <z, r> so we do not recompute the dot-product
        }
    }

    return iterations;
}


/*!
 * \ingroup HALASOLVECG
 * \brief General preconditioned conjugate-gradient solver.
 *
 * Solves \f$ A x = b \f$ where \b A is a sparse symmetric-positive-definite matrix
 * and \b P is an appropriate preconditioner.
 * The iteration is terminated when either the residual drops below the tolerance or the maximum
 * number of iterations is reached.
 * - \b pntr, \b indx, \b vals describe the sparse matrix \b A in row compressed format
 * - \b b specifies the right-hand-side of the equation
 * - \b x is both the initial guess for the iteration and the final computed solution,
 *      if the size of \b x is incorrect it will be resized and reset to zero
 * - \b precon(p, r) takes two vectors of the internal type and overwrites \b r with
 *      the \f$ P^{-1} p \f$, see hala::preconditioner
 *
 * Overloads:
 * \code
 *  solve_cg(stop, pntr, indx, vals, precon, b, x);
 *  solve_cg(engine, stop, pntr, indx, vals, precon, b, x);
 * \endcode
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int solve_cg(stop_criteria<get_precision_type<VectorLikeV>> const &stop,
             VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
             preconditioner_noe<VectorLikeV> precon,
             VectorLikeB const &b, VectorLikeX &x){

    check_types(vals, x, b);
    check_types_int(pntr, indx);

    auto num_entries = get_size(pntr);
    assert( num_entries > 1 ); // smallest case has one entry
    assert( check_size(b, num_entries - 1) );

    if (get_size(x) < num_entries - 1){
        force_size(x, num_entries - 1);
        set_zero(num_entries - 1, x); // if using CUDA the entries will not be initialized
    }

    using temp_vector = decltype( new_vector( std::declval<VectorLikeV>() ) );
    using scalar_type = get_standard_type<VectorLikeX>;

    int nrows = get_size_int(pntr) - 1;

    auto matrix = make_sparse_matrix(nrows, pntr, indx, vals);
    size_t bsize =  sparse_gemv_buffer_size(matrix, 'N', 1.0, b, 0.0, x);
    auto work_buffer = new_vector(x);
    set_size(bsize, work_buffer);

    return solve_cg_core<scalar_type>(precon,
                [&](VectorLikeX const &v, temp_vector &r)->void{
                    sparse_gemv(matrix, 'N', 1.0, v, 0.0, hala::bind_engine_vector(vals.engine(), r.vector()), work_buffer);
                },
                [&](temp_vector const &v, temp_vector &r)->void{
                    sparse_gemv(matrix, 'N', 1.0, v, 0.0, r, work_buffer);
                },
                b, x,
                []()->scalar_type{ return get_cast<scalar_type>(0); },
                [](scalar_type const &num, scalar_type const &denom, scalar_type &result)->void{ result = num / denom; },
                [](scalar_type const &src, scalar_type &dest)->void{ dest = src; },
                [&](scalar_type const &a, temp_vector &r)->void{ scal(a, r); },
                [&](temp_vector const &v, temp_vector &r)->void{ axpy(1.0, v, r); },
                [&](temp_vector const &v, temp_vector &r)->void{ axpy(-1.0, v, r); },
                [&](scalar_type const &a, temp_vector const &v, VectorLikeX &r)->void{ axpy(a, v, r); },
                [&](scalar_type const &a, temp_vector const &v, temp_vector &r)->void{ axpy(-a, v, r); },
                [&](temp_vector const &v, temp_vector const &r, scalar_type &a)->void{ a = dot(v, r); },
                [&](int i, temp_vector const &r)->bool{ return ((i == stop.max_iter) || (norm2(r) < stop.tol)); }
        );
}

#ifndef __HALA_DOXYGEN_SKIP
template<class compute_engine, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
hala_int<compute_engine>
solve_cg(compute_engine const &engine,
         stop_criteria<get_precision_type<VectorLikeV>> const &stop,
         VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
         preconditioner<compute_engine, VectorLikeX> precon,
         VectorLikeB const &b, VectorLikeX &x){
    auto p = bind_engine_vector(engine, pntr);
    auto i = bind_engine_vector(engine, indx);
    auto v = bind_engine_vector(engine, vals);
    auto bb = bind_engine_vector(engine, b);
    auto xx = bind_engine_vector(engine, x);
    return solve_cg(stop, p, i, v,
                    [&](auto const &inx, auto &outr)->void{
                        precon(inx.vector(), outr.vector());
                    }, bb, xx);
}

#ifdef HALA_ENABLE_CUDA
template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int solve_cg(mixed_engine const &engine,
         stop_criteria<get_precision_type<VectorLikeV>> const &stop,
         VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
         preconditioner<mixed_engine, VectorLikeX> precon,
         VectorLikeB const &b, VectorLikeX &x){
    auto p = engine.gpu().load(pntr);
    auto i = engine.gpu().load(indx);
    auto v = engine.gpu().load(vals);
    auto bb = engine.gpu().load(b);
    auto xx = gpu_bind_vector(engine.gpu(), x);
    return solve_cg(engine.gpu(), stop, p, i, v,
                    [&](auto const &inx, auto &outr)->void{
                        precon(inx, outr);
                    }, bb, xx);
}
#endif
#endif

/*!
 * \ingroup HALASOLVECG
 * \brief Conjugate-gradient solver with ILU preconditioner, single right-hand-side.
 *
 * Solves \f$ A x = b \f$ using the conjugate-gradient method.
 * - \b stop defines the tolerance (measured as the norm of the residual) and the maximum number of iterations
 * - \b pntr, \b indx, and \b vals describe a symmetric positive definite sparse matrix in row-compressed format
 * - \b ilu is precomputed ILU factor matching the engine, e.g., cpu_ilu, cuda_ilu or mixed_ilu
 * - \b b is the right-hand-side of the equation
 * - \b x is the initial guess to start the iteration,
 *  if the size of \b x is insufficient it will be resized and scaled by zero
 * - \b returns the total number of iterations
 *
 * Overloads:
 * \code
 *  solve_cg_ilu(stop, pntr, indx, vals, b, x);
 *  solve_cg_ilu(stop, pntr, indx, vals, ilu, b, x);
 *  solve_cg_ilu(engine, stop, pntr, indx, vals, b, x);
 *  solve_cg_ilu(engine, stop, pntr, indx, vals, ilu, b, x);
 * \endcode
 * The methods work with all compute engines, if the ilu input is missing the overload will perform the ilu
 * factorization "on-the-fly" and discard it after the code has completed.
 *
 * \b Note: that the engine for the ILU factor and the rest of the vectors must match.
 */
template<class VectorLikeP,  class VectorLikeI, class VectorLikeV,
         class ILUclass, class VectorLikeX, class VectorLikeB>
int solve_cg_ilu(stop_criteria<get_precision_type<VectorLikeV>> const &stop,
                 VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                 ILUclass const &ilu, VectorLikeB const &b, VectorLikeX &x){

    return hala::solve_cg(stop, pntr, indx, vals,
                          [&](auto const &inx, auto &outr)->void{
                                ilu_apply(ilu, inx, outr);
                          }, b, x);
}

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)

template<class compute_engine, class VectorLikeP,  class VectorLikeI, class VectorLikeV,
         class ILUclass, class VectorLikeX, class VectorLikeB>
hala_int<compute_engine>
solve_cg_ilu(compute_engine const &engine,
             stop_criteria<get_precision_type<VectorLikeV>> const &stop,
             VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
             ILUclass const &ilu, VectorLikeB const &b, VectorLikeX &x){

    static_assert(std::is_same<typename ILUclass::engine_type, compute_engine>::value,
                  "Using compatible compute engine and ILU preconditioner");

    auto p = bind_engine_vector(engine, pntr);
    auto i = bind_engine_vector(engine, indx);
    auto v = bind_engine_vector(engine, vals);
    auto bb = bind_engine_vector(engine, b);
    auto xx = bind_engine_vector(engine, x);
    return solve_cg_ilu(stop, p, i, v, ilu, bb, xx);
}

#ifdef HALA_ENABLE_CUDA
template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int solve_cg_ilu(mixed_engine const &engine,
                 stop_criteria<get_precision_type<VectorLikeV>> const &stop,
                 VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                 mixed_ilu<get_standard_type<VectorLikeV>> const &ilu, VectorLikeB const &b, VectorLikeX &x){
    auto gpntr = engine.gpu().load(pntr);
    auto gindx = engine.gpu().load(indx);
    auto gvals = engine.gpu().load(vals);
    auto gb = engine.gpu().load(b);
    auto gx = gpu_bind_vector(engine.gpu(), x);
    return solve_cg_ilu(engine.gpu(), stop, gpntr, gindx, gvals, ilu.cilu(), gb, gx);
}
#endif // end CUDA

template<class VectorLikeP, class VectorLikeI, class VectorLikeV,
         class VectorLikeX, class VectorLikeB>
int solve_cg_ilu(stop_criteria<get_precision_type<VectorLikeV>> const &stop,
                 VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                 VectorLikeB const &b, VectorLikeX &x, char policy = 'N'){

    auto ilu = make_ilu(pntr, indx, vals, policy);

    return hala::solve_cg_ilu(stop, pntr, indx, vals, ilu, b, x);
}

template<class compute_engine, class VectorLikeP,  class VectorLikeI, class VectorLikeV,
         class VectorLikeX, class VectorLikeB>
int solve_cg_ilu(compute_engine const &engine,
                 stop_criteria<get_precision_type<VectorLikeV>> const &stop,
                 VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                 VectorLikeB const &b, VectorLikeX &x, char policy = 'N'){

    auto ilu = make_ilu(engine, pntr, indx, vals, policy);

    return hala::solve_cg_ilu(engine, stop, pntr, indx, vals, ilu, b, x);
}
#endif // end skip docs

}

#endif
