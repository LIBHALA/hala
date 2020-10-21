#ifndef __HALA_SOLVERS_GMRES_HPP
#define __HALA_SOLVERS_GMRES_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */
#include "hala_solvers_cg_batch.hpp"

/*!
 * \internal
 * \file hala_solvers_gmres.hpp
 * \brief HALA General Minimum RESidula solver.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALASOLVEGMRES
 *
 * General Minimum RESidula (GMRES) solver.
 * \endinternal
 */

namespace hala{

/*!
 * \ingroup HALASOLVERS
 * \addtogroup HALASOLVEGMRES GMRES Solver
 *
 * A series of templates that implement the preconditioned General Minimum RESidual method
 * with several options to choose from different preconditioners.
 */

/*!
 * \internal
 * \ingroup HALASOLVEGMRES
 * \brief Projects the vector \b r onto the basis \b W and returns the coefficients \b coeffs.
 *
 * The final coefficients always sit on the CPU regardless of which engine is being used.
 * Overloads exist for only CPU and CUDA engines with the latter pulling the data back
 * to the CPU before computing more.
 * \endinternal
 */
template<class VectorLike, class VectorLikeResult>
void krylov_project(cpu_engine const&, int num_rows, int num_basis, VectorLike &r, VectorLike const &W, VectorLikeResult &coeffs){
    gemv('T', num_rows, num_basis, 1.0, W, r, 0.0, coeffs);
    gemv('N', num_rows, num_basis, -1.0, W, coeffs, 1.0, r);
}
/*!
 * \internal
 * \ingroup HALASOLVEGMRES
 * \brief Increment the current solution vector \b x with the basis \b W times the coefficients \b coeffs.
 *
 * Overloads exist for only CPU and CUDA engines and the coefficients \b coeffs always sit on the CPU.
 * \endinternal
 */
template<class VectorLikeX, class VectorLikeW, class VectorLikeCoeff>
void krylov_combine(cpu_engine const&, int num_rows, int num_basis, VectorLikeW &W, VectorLikeCoeff const &coeffs, VectorLikeX &&x){
    gemv('N', num_rows, num_basis, 1.0, W, coeffs, 1.0, x);
}

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)
#ifdef HALA_ENABLE_GPU
template<class VectorLike, class VectorLikeResult>
void krylov_project(gpu_engine const &engine, int num_rows, int num_basis, VectorLike &r, VectorLike const &W, VectorLikeResult &coeffs){
    auto tmp = new_vector(engine, r);
    gemv(engine, 'T', num_rows, num_basis,  1.0, W, r, 0.0, tmp);
    gemv(engine, 'N', num_rows, num_basis, -1.0, W, tmp, 1.0, r);
    tmp.unload(coeffs);
}
template<class VectorLikeX, class VectorLikeW, class VectorLikeCoeff>
void krylov_combine(gpu_engine const &engine, int num_rows, int num_basis, VectorLikeW &W, VectorLikeCoeff const &coeffs, VectorLikeX &&x){
    auto tmp = engine.load(coeffs);
    gemv(engine, 'N', num_rows, num_basis, 1.0, W, tmp, 1.0, x);
}
#endif
#endif

/*!
 * \ingroup HALASOLVEGMRES
 * \brief General restarted preconditioned Minimum RESidula (GMRES) method, base template.
 *
 * Solves \f$ P^{-1} A x = P^{-1} b \f$ using the GMRES method for general non-singular matrices.
 * Compared to the Conjugate-Gradient method, GMRES handles much more general problems but comes at
 * the additional cost of explicitly storing the Krylov basis.
 * - \b engine is the compute engine associated with the matrix and input vectors
 * - \b stop defines the tolerance (measured as the norm of the residual) and the maximum number of (outer) iterations
 * - \b restart is the maximum size of the Krylov basis after which the method will restart with the current
 *  solution as a new initial guess;
 * - \b pntr, \b indx, and \b vals describe a sparse matrix in row-compressed format
 * - \b precon is a preconditioner associated either with the cpu_engine or the gpu_engine,
 *  when using the mixed_engine the preconditioner accepts vectors on the GPU
 * - \b b is the right hand side
 * - \b x is the initial iterate and the final solution, if the size of \b x is insufficient
 *  it will be resized and set to zero
 * - \b returns the total number of matrix-vector-preconditioner applications
 *
 * \b Restart frequency should be balanced, since frequent restarts will deteriorate the convergence
 * but larger basis is expensive to store and manipulate. The Krylov basis has size equal to
 * the number of basis vectors times the number of rows of the matrix.
 *
 * Overloads:
 * \code
 *  solve_gmres(stop, restart, pntr, indx, vals, precon, b, x);
 *  solve_gmres(engine, stop, restart, pntr, indx, vals, precon, b, x);
 * \endcode
 * The method can be called with any compute engine and no engine,
 * the type of the \b precon function changes accordingly,
 * see hala::preconditioner and hala::preconditioner_noe.
 *
 * \internal
 * \b Note: the CG implementation is written generically in a no-engine format
 * and the engine comes through the overloads where it gets bind to engined-vectors.
 * The GMRES implementation explicitly uses the engine and the engine overloads,
 * because the coefficients of the Krylov basis are stored and manipulated on the CPU.
 * Therefore, even in a GPU implementation, the method will use a mix of CPU and GPU
 * operations and the engine will be used to differentiate between the two modes.
 * This is in contrast to CG where all operations are done on either the CPU or GPU
 * and carrying the same engine to each and every call is redundant.
 * \endinternal
 *
 */
template<class compute_engine, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
hala_int<compute_engine>
solve_gmres(compute_engine const &engine,
            stop_criteria<get_precision_type<VectorLikeV>> const &stop, int restart,
            VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
            preconditioner<compute_engine, VectorLikeV> precon,
            VectorLikeB const &b, VectorLikeX &x){
    // sanity check
    check_types(vals, x, b);
    check_types_int(pntr, indx);

    int num_rows = get_size_int(pntr) -1;
    assert( num_rows > 0 ); // smallest case has one entry
    assert( check_size(b, num_rows) );

    if (get_size_int(x) < num_rows){
        force_size(num_rows, x);
        set_zero(engine, (size_t) num_rows, x);
    }

    // define the types
    using standart_type  = get_standard_type<VectorLikeV>;
    using precision_type = get_precision_type<VectorLikeV>;

    std::vector<standart_type> H, S, Z;
    std::vector<precision_type> C;
    H.reserve(restart * (restart + 1)); // transformation matrix, upper triangular packed
    S.reserve(restart + 1); // sin of Givens rotation
    C.reserve(restart + 1); // cos of Givens rotation
    Z.reserve(restart + 1); // holds the coefficients of the solution

    precision_type inner_res, outer_res = get_cast<precision_type>(stop.tol + 1.0); // inner and outer residuals
    int total_iterations = 0;
    int outer_iterations = 0;

    auto matrix = make_sparse_matrix(engine, num_rows, pntr, indx, vals);
    size_t bsize =  sparse_gemv_buffer_size(matrix, 'N', 1.0, b, 1.0, x);
    auto work_buffer = new_vector(x);
    set_size(bsize, work_buffer);

    while((outer_res > stop.tol) && (outer_iterations < stop.max_iter)){
        H.resize(0); S.resize(0); C.resize(0); Z.resize(0);

        auto t = new_vector(engine, vals);
        auto r = new_vector(engine, vals);

        vcopy(engine, b, t);
        matrix.gemv('N', -1.0, x, 1.0, t, work_buffer);
        precon(t, r);
        total_iterations++;

        inner_res = norm2(engine, r);
        scal(engine, 1.0 / inner_res, r);
        Z.push_back(get_cast<standart_type>(inner_res));

        auto W = new_vector(engine, vals);
        force_size(hala_size(num_rows, restart), W);
        vcopy(engine, r, W);

        int inner_iterations = 0;
        while ((inner_res > stop.tol) && (inner_iterations < restart)){

            matrix.gemv('N', 1.0, r, 0.0, t, work_buffer);
            precon(t, r);
            total_iterations++;

            std::vector<standart_type> coeffs;
            krylov_project(engine, num_rows, inner_iterations + 1, r, W, coeffs);
            auto nrm = norm2(engine, r);
            scal(engine, 1.0 / nrm, r);

            for(int i=0; i<inner_iterations; i++)
                rot(1, &coeffs[i], 1, &coeffs[i+1], 1, C[i], S[i]);

            standart_type isin, beta = get_cast<standart_type>(nrm);
            precision_type icos;
            rotg(coeffs[inner_iterations], beta, icos, isin);

            H.insert(H.end(), coeffs.begin(), coeffs.end());
            S.push_back(isin);
            C.push_back(icos);

            inner_res = hala_abs(S.back() * Z.back());
            inner_iterations++;

            if ((inner_res > stop.tol) && (inner_iterations < restart)){ // if still going ...
                // expand the Krylov basis
                vcopy(engine, num_rows, r, 1, get_data(W) + inner_iterations * num_rows, 1);

                Z.push_back(get_cast<standart_type>(0.0));
                rot(1, get_data(Z) + inner_iterations-1, 1, get_data(Z) + inner_iterations, 1, C.back(), S.back());
            }
        }

        if (H.size() > 0){
            // this happens when the last iteration overestimated the residual and the new iteration
            // simply declared "converged" at iteration 0, then there is no point to adjust x
            tpsv('U', 'N', 'N', (int) Z.size(), H, Z);
            krylov_combine(engine, num_rows, (int) Z.size(), W, Z, x);
        }
        outer_iterations++;
        outer_res = inner_res;
    }

    return total_iterations;
}

/*!
 * \ingroup HALASOLVEGMRES
 * \brief General restarted Minimum RESidula (GMRES) method with ILU preconditioner.
 *
 * Calls hala::solve_gmres() using the ILU preconditioner \b ilu.
 * The preconditioner has to be associated with the compute engine (e.g., mixed preconditioner
 * for the mixed_engine), the ILU preconditioner must also match the engine for
 * the vector in the no-engine case (e.g., hala::engined_vector).
 *
 * Overloads:
 * \code
 *  solve_gmres_ilu(stop, restart, pntr, indx, vals, b, x);
 *  solve_gmres_ilu(stop, restart, pntr, indx, vals, ilu, b, x);
 *  solve_gmres_ilu(engine, stop, restart, pntr, indx, vals, b, x);
 *  solve_gmres_ilu(engine, stop, restart, pntr, indx, vals, ilu, b, x);
 * \endcode
 * If the ILU is not provided, factorization will be computed, used and finally discarded on-the-fly.
 */
template<class compute_engine, class VectorLikeP, class VectorLikeI, class VectorLikeV, class ILUclass, class VectorLikeX, class VectorLikeB>
hala_int<compute_engine>
solve_gmres_ilu(compute_engine const &engine,
                stop_criteria<get_precision_type<VectorLikeV>> const &stop, int restart,
                VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                ILUclass const &ilu, VectorLikeB const &b, VectorLikeX &x){

    static_assert(std::is_same<typename ILUclass::engine_type, compute_engine>::value,
                  "Using compatible compute engine and ILU preconditioner");

    auto temp_buffer = ilu_temp_buffer(ilu, b, x, 1);

    return solve_gmres(engine, stop, restart, pntr, indx, vals,
                       [&](auto const &inx, auto &outr)->void{
                           ilu_apply(ilu, inx, outr, 1, temp_buffer);
                       },
                       b, x);
}

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)
#ifdef HALA_ENABLE_GPU
template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int solve_gmres(mixed_engine const &engine,
                stop_criteria<get_precision_type<VectorLikeV>> const &stop, int restart,
                VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                preconditioner<gpu_engine, VectorLikeX> precon,
                VectorLikeB const &b, VectorLikeX &x){
    auto gpntr = engine.gpu().load(pntr);
    auto gindx = engine.gpu().load(indx);
    auto gvals = engine.gpu().load(vals);
    auto gb = engine.gpu().load(b);
    auto gx = cuda_bind_vector(engine.gpu(), x);
    return solve_gmres(engine.gpu(), stop, restart, gpntr, gindx, gvals, precon, gb, gx);
}

template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class ILUclass, class VectorLikeX, class VectorLikeB>
int solve_gmres_ilu(mixed_engine const &engine,
                    stop_criteria<get_precision_type<VectorLikeV>> const &stop, int restart,
                    VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                    ILUclass const &ilu, VectorLikeB const &b, VectorLikeX &x){

    static_assert(std::is_same<typename ILUclass::engine_type, mixed_engine>::value,
                  "Using compatible compute engine and ILU preconditioner");

    auto gpntr = engine.gpu().load(pntr);
    auto gindx = engine.gpu().load(indx);
    auto gvals = engine.gpu().load(vals);
    auto gb = engine.gpu().load(b);
    auto gx = gpu_bind_vector(engine.gpu(), x);
    return solve_gmres(engine.gpu(), stop, restart, gpntr, gindx, gvals,
                       [&](auto const &inx, auto &outr)->void{
                           ilu_apply(ilu.cilu(), inx, outr);
                       },
                       gb, gx);
}
#endif

template<class cengine, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int solve_gmres(stop_criteria<get_precision_type<VectorLikeV>> const &stop, int restart,
                engined_vector<cengine, VectorLikeP> const &pntr, engined_vector<cengine, VectorLikeI> const &indx,
                engined_vector<cengine, VectorLikeV> const &vals,
                preconditioner_noe<VectorLikeV> precon,
                engined_vector<cengine, VectorLikeB> const &b, engined_vector<cengine, VectorLikeX> &x){
    return solve_gmres(vals.engine(), stop, restart, pntr.vector(), indx.vector(), vals.vector(), precon, b.vector(), x.vector());
}

template<class cengine, class VectorLikeP, class VectorLikeI, class VectorLikeV, class ILUclass, class VectorLikeX, class VectorLikeB>
int solve_gmres(stop_criteria<get_precision_type<VectorLikeV>> const &stop, int restart,
                engined_vector<cengine, VectorLikeP> const &pntr, engined_vector<cengine, VectorLikeI> const &indx,
                engined_vector<cengine, VectorLikeV> const &vals,
                ILUclass const &ilu,
                engined_vector<cengine, VectorLikeB> const &b, engined_vector<cengine, VectorLikeX> &x){
    return solve_gmres(vals.engine(), stop, restart, pntr.vector(), indx.vector(), vals.vector(), ilu, b.vector(), x.vector());
}

template<class compute_engine, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int solve_gmres_ilu(compute_engine const &engine,
                    stop_criteria<get_precision_type<VectorLikeV>> const &stop, int restart,
                    VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                    VectorLikeB const &b, VectorLikeX &x, char policy = 'N'){

    auto ilu = make_ilu(engine, pntr, indx, vals, policy);
    return solve_gmres_ilu(engine, stop, restart, pntr, indx, vals, ilu, b, x);
}

template<class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int solve_gmres_ilu(stop_criteria<get_precision_type<VectorLikeV>> const &stop, int restart,
                    VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
                    VectorLikeB const &b, VectorLikeX &x, char policy = 'N'){

    auto ilu = make_ilu(pntr, indx, vals, policy);
    return solve_gmres_ilu(stop, restart, pntr, indx, vals, ilu, b, x);
}
#endif


}

#endif
