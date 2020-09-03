#include <iostream>
#include <iomanip>

#include "hala.hpp"

using std::cout;
using std::endl;
using std::setw;

/*!
 * \file example_gauss_seidel.cpp
 * \brief Example of using the BLAS module of HALA.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAEXAMPLE1
 *
 * Example 1, Gauss-Seidel solver using BLAS.
 */


/*!
 * \defgroup HALAEXAMPLE HALA Examples
 *
 * Several examples that demonstrate the usage of the HALA capabilities.
 */

/*!
 * \ingroup HALAEXAMPLE
 * \addtogroup HALAEXAMPLE1 Example of BLAS usage
 *
 * \par Example 1
 * This example presents three implementations of an implicit time-stepping scheme using
 * the Gauss-Seidel method for systems of linear equations using dense matrices.
 * The example makes no claims regarding the overall best strategy to address such problem,
 * it only serves as a demonstration of how to combine different components of HALA to
 * build more complex methods while preserving the flexibility and exploiting
 * the optimized linear algebra libraries.
 *
 * \par
 * The example is divided into several sections:
 * - \ref Example1Problem "Problem Setup"
 * - \ref Example1VarA "Variant A using BLAS level 2"
 * - \ref Example1VarB "Variant B using BLAS level 3"
 * - \ref Example1VarC "Variant C tuned to use different levels of BLAS"
 * - \ref Example1Remarks "Remarks and Results"
 *
 * \anchor Example1Problem
 * \par Problem Setup
 * Consider the integro-differential equation with fractional Laplacian operator
 * \f$ \frac{\partial u}{\partial t} = - \left( - \Delta \right)^s u \f$
 * over the domain [0, 1] integrated from 0 to t = 1 with a quadratic profile
 * as initial condition.
 * The specific problem and discretization scheme are not of focus, it is sufficient
 * to say that the problem reduces to the system of ordinary differential equations
 * \f$ \frac{d}{dx} f(t) = A f(t) \f$ where the matrix is dense.
 * The time-stepping method used here is Crank-Nicholson implicit scheme
 * \f$ \left( I - \frac{\Delta t}{2} A \right) f^{n+1} = \left( I + \frac{\Delta t}{2} A \right) f^n \f$
 * where \f$ f^n \approx f(n \Delta t) \f$, \f$ I \f$ denotes the identity matrix,
 * and finding each successive f requires the solution of a (dense) linear system of equations.
 * The linear solver used here is the Gauss-Seidel iteration, namely
 * \f$ f_{k+1} = L^{-1} \left( b + I f_k - U f_k \right) \f$ where L is the lower
 * triangular portion of the left-hand matrix, U is the upper triangular portion with
 * unit diagonal, b is the right-hand side of the equation, the k=0 iterate is
 * the n-th time step and as k increases the k-th iterate converges to the n+1-th time step.
 * The stopping criteria uses the difference between two successive Gauss-Seidel iterates.
 *
 * \par
 * The two components of interest are the time-stepping scheme and the iterative solver.
 *
 * \anchor Example1VarA
 * \par Gauss-Seidel variant A, using BLAS level 2
 * See below the implementation of the Gauss-Seidel solver and note the templated interface.
 *
 * \par
 * Implementation overview:
 * - The assert() statements check for input size consistency, which is an important feature for C++ containers
 *   that "know" their own size and cannot be implemented with raw-arrays.
 * - The algorithm uses one temporary vector \b xnew which will be filled with the next iterate,
 *   the temporary vector does not have to be manually resized, this will be done automatically by hala::vcopy().
 * - The most performance critical steps are the BLAS level 2 operations, triangular multiply and solve
 *   using hala::trmv() and hala::trsv().
 * - One scale operation is needed to flip the direction of \b xnew, i.e., using hala::scal().
 * - The values of \b x and \b xnew are swapped using std::swap() which will exchange only the internal
 *   pointers and thus has complexity independent of the size of the vectors.
 *
 * \par
 * Other variants exist, e.g., the forward substitution variant of the Gauss-Seidel solver can yield convergence in fewer iterations
 * by using the newly computed values immediately, e.g., in the triangular solve stage,
 * and this variant does not need a temporary vector.
 * However, the approach is sequential and most BLAS implementations are parallel using all available
 * CPU cores; sequential algorithms are at a massive disadvantage when executed on modern multi-core processors
 * and unlike the forward substitution, our implementation requires only a few calls to HALA without the need of
 * complicated for-loops or indexing.
 *
 * \snippet examples/example_gauss_seidel.cpp Example_01 gsa
 *
 * \par Crank-Nicholson variant A
 * The first variant of the integration scheme is very similar to the solver, the left-hand matrix
 * is constructed explicitly in the \b update variable and the right-hand operator is treated implicitly,
 * i.e., the application of that onto the state is computed with two calls to HALA.
 * The hala::gemv() operation could in this case be replaced by hala::symv() since the specific operator
 * is symmetric, but the savings would be negligible compared to the cost of the Gauss-Seidel solver and
 * the hala::gemv() variant can handle a broader class of problems.
 *
 * \snippet examples/example_gauss_seidel.cpp Example_01 inta
 *
 * \anchor Example1VarB
 * \par Gauss-Seidel variant B, using BLAS level 3
 * Suppose that we are interested in observing how noise in the initial data propagates though
 * the solution to the integro-differential equation.
 * In such scenario, we would be performing multiple simulations with different initial conditions
 * and collecting statistical data from the final solutions.
 * Multiple simulations can be performed by wrapping the existing integrator in a for-loop solving
 * one problems at a time, but a much better approach is to rewrite the solver using BLAS level 3 methods.
 * The level 3 methods require the largest amount of computations vs. data-size and can
 * achieve highest performance with the aid of very advanced caching algorithms.
 *
 * \par
 * Implementation overview:
 * - The level 3 implementation accepts an additional variable indicating the number of simulations
 *   equations and the size asset statements are modified accordingly.
 * - The level 2 operations are replaced by level 3 hala::trmm() and hala::trsm() and the scale
 *   operation is assimilated into the product.
 * - The norm is replaced by hala::batch_max_norm2(), which is a HALA extension that uses a simple for-loop.
 * - The rest of the algorithm remains unchanged, largely due to the HALA overloads that automatically
 *   infer vector dimensions and thus save us from counting N times the number of right-hand-sides.
 *
 * \snippet examples/example_gauss_seidel.cpp Example_01 gsb
 *
 * \par Crank-Nicholson variant B
 * The corresponding time-integrator is just as easy to update replacing only the size assertions
 * and the update that now uses hala::gemm().
 *
 * \snippet examples/example_gauss_seidel.cpp Example_01 intb
 *
 * \anchor Example1VarC
 * \par Gauss-Seidel variant C, tuning a mix of level 2 and level 3
 * BLAS implementations use a wide range of algorithms for different cases and situations and it is
 * hard to predict the exact point where to switch between multiple simulations of using the level 2
 * implementation and using the level 3. For a single state the level 2 method is the clear winner,
 * and using many simulations (e.g., 100) the level 3 would be the clear choice.
 * But what about the case of just 2 right hand sides, or 7, or 50?
 * The \e best approach is empirical and tied to a specific machine and BLAS implementation,
 * but here we present several techniques that can be successfully utilized and also demonstrate
 * different aspects of the HALA API.
 *
 * \par
 * We consider three use cases:
 * - when the number of right-hand-sides is no more than 5, we apply BLAS level 2 operations in a for loop
 * - when the number of right-hand-sides is 6 to 10, we pad the problem to size 10
 * - when the problem has more than 10 right-hand-sides we use the BLAS level 3 directly
 *
 * \par
 * See the \ref Example1Results "Results" section.
 *
 * \par
 * The Gauss-Seidel variant C now implements a flexible algorithm that can use both level 2
 * and level 3 operations and work with flexible vector-like classes that can be slices of
 * larger vectors (e.g., the columns of a larger matrix).
 *
 * \par
 * Implementation overview:
 * - The first thing to note is the replacement of std::vector classes for b and x with a generic
 *   \b VectorLike class that can be any class with the appropriate HALA template specialization,
 *   e.g., std::vector, std::valarray, hala::wrapped_array, etc.
 * - The assert() statement now has to use the generic hala::get_size() method instead of relying
 *   on \b x.size().
 * - If-statements select between level 2 and level 3 methods on the fly.
 * - Note that even though \b x and \b b have the generic \b VectorLike types, the calls to HALA
 *   have not changed, it is up to the compiler to instantiate the corresponding templates
 *   with the correct types and handle all the necessary book-keeping.
 * - The only other change is the switch from std::swap() to hala::vswap() which makes a physical
 *   swap of the vector entries, since std::swap() cannot be called with different containers.
 *
 * \snippet examples/example_gauss_seidel.cpp Example_01 gsc
 *
 * \par Crank-Nicholson variant C
 * The variant C simulator now handles the padding and switches between different modes of the solver.
 * - Fewer than 6 states would trigger the use of level 2 solver with hala::wrapped_array inputs
 *   for \b x and \b b.
 * - Between 6 and 9 samples, the state and \b rhs will be padded; note that the extra padding
 *   does not interfere with the hala::gemv() and hala::gemm() calls.
 * - For 10 or more stages, the solver will use no padding and level 3 mode.
 *
 * \snippet examples/example_gauss_seidel.cpp Example_01 intc
 *
 * \anchor Example1Remarks
 * \par Remarks
 * The specific tuning parameters would be very different between different systems;
 * however, the general principles apply.
 * The example serves to demonstrate the flexibility of the HALA interface that
 * allows for a very general algorithm to be implemented and coupled with very specific
 * tuning criteria with only a few well placed if-statements.
 *
 * \anchor Example1Results
 * \par Results
 * Experiments were performed on an Intel i9-9920X CPU with 12-cores and using the OpenBLAS library,
 * on Ubuntu 18.04. The tuned implementation showed the greatest performance, although the
 * specific tuning parameters would be heavily dependent on the system.
 * Also note the inconsistencies in execution time when using different problem sizes,
 * this is due to OpenBLAS switching between algorithms with different caching and parallel
 * strategies. However, in each case, the implementation would most certainly outperform
 * any manual implementation of the corresponding operations, even the forward substitution
 * strategy often used in serial implementations of the Gauss-Seidel algorithm.
 *
 * \snippet examples/example_gauss_seidel.cpp Example_01 main
 *
 * \code
 *  Using BLAS level 2
 *   Integrated one in  2163 milliseconds.
 *  Using BLAS level 3
 *   Integrated   1 in 10745 milliseconds.
 *   Integrated   5 in 15074 milliseconds.
 *   Integrated  10 in 15019 milliseconds.
 *   Integrated 100 in 21984 milliseconds.
 *  Using BLAS dynamic
 *   Integrated   1 in  2189 milliseconds.
 *   Integrated   5 in 11076 milliseconds.
 *   Integrated  10 in 14948 milliseconds.
 *   Integrated 100 in 24128 milliseconds.
 * \endcode
 *
 */

/*!
 * \ingroup HALAEXAMPLE1
 * \brief Construct a vector that represents the domain.
 *
 * Divides the domain [0, 1] into N intervals and returns a vector with
 * the mid-points of all intervals.
 */
template<typename precision>
std::vector<precision> make_domain(int N);

/*!
 * \ingroup HALAEXAMPLE1
 * \brief Creates a simple discrete approximation to the fractional Laplacian operator.
 *
 * The numerical properties of the discretization scheme is not in scope of this example,
 * the point is that the resulting approximation uses a dense matrix of size N by N.
 */
template<typename precision>
std::vector<precision> make_flaplace(int N);

/*!
 * \ingroup HALAEXAMPLE1
 * \brief Gives a quadratic initial profile over the domain.
 */
template<typename precision>
std::vector<precision> make_initial(int N);

//! [Example_01 gsa]
template<typename precision>
void solve_gauss_seidel_a(int max_iter,
                          typename std::vector<precision>::value_type stop_tolerance,
                          std::vector<precision> const &A,
                          std::vector<precision> const &b,
                          std::vector<precision> &x){
    // solves A x = b
    assert( b.size() == x.size() );
    assert( A.size() == x.size() * x.size() );

    int N = (int) x.size();
    precision error = 0.0;
    int iterations = 0;

    do{
        std::vector<precision> xnew;
        hala::vcopy(x, xnew);

        hala::trmv('U', 'N', 'U', N, A, xnew); // xnew = U x
        hala::scal(-1.0, xnew);   // xnew = - U x

        hala::axpy(1.0, x, xnew);
        hala::axpy(1.0, b, xnew); // xnew = b + I x - U x

        hala::trsv('L', 'N', 'N', N, A, xnew); // xnew = L^{-1} ( b + I x - U x )

        std::swap(x, xnew);

        hala::axpy(-1.0, x, xnew); // xnew = x - xnew
        error = hala::norm2(xnew); // error = \| xnew - x \|

        iterations++;
    }while(error > stop_tolerance and iterations < max_iter);
}
//! [Example_01 gsa]

//! [Example_01 gsb]
template<typename precision>
void solve_gauss_seidel_b(int max_iter,
                          typename std::vector<precision>::value_type stop_tolerance,
                          int num_rhs,
                          std::vector<precision> const &A,
                          std::vector<precision> const &b,
                          std::vector<precision> &x){
    // solves A x_i = b_i, for i = 0, 1, ..., num_rhs - 1
    assert( num_rhs > 0 );
    assert( b.size() == x.size() );

    int N = (int) x.size() / num_rhs;
    assert( A.size() == (size_t) (N * N) );

    precision error = 0.0;
    int iterations = 0;

    do{
        std::vector<precision> xnew;
        hala::vcopy(x, xnew);

        hala::trmm('L', 'U', 'N', 'U', N, num_rhs, -1.0, A, xnew); // xnew = - U x

        hala::axpy(1.0, x, xnew);
        hala::axpy(1.0, b, xnew); // xnew = b - U x + I x

        hala::trsm('L', 'L', 'N', 'N', N, num_rhs, 1.0, A, xnew); // xnew = L^{-1} ( b - U x + I x )

        std::swap(x, xnew);

        hala::axpy(-1.0, x, xnew);
        error = hala::batch_max_norm2(num_rhs, xnew); // error = max \| xnew - x \|

        iterations++;
    }while(error > stop_tolerance and iterations < max_iter);
}
//! [Example_01 gsb]

//! [Example_01 gsc]
template<typename precision, class VectorLikeB, class VectorLikeX>
void solve_gauss_seidel_c(int max_iter,
                          typename std::vector<precision>::value_type stop_tolerance,
                          int num_rhs,
                          std::vector<precision> const &A,
                          VectorLikeB const &b,
                          VectorLikeX &&x){

    assert( num_rhs > 0 );
    assert( hala::get_size(x) == hala::get_size(b) );

    int N = static_cast<int>(hala::get_size(x)) / num_rhs;
    assert( A.size() == static_cast<size_t>(N * N) );

    precision error = 0.0;
    int iterations = 0;

    do{
        std::vector<precision> xnew;
        hala::vcopy(x, xnew);

        if (num_rhs == 1){
            hala::trmv('U', 'N', 'U', N, A, xnew);
            hala::scal(-1.0, xnew);
        }else{
            hala::trmm('L', 'U', 'N', 'U', N, num_rhs, -1.0, A, xnew);
        }

        hala::axpy(1.0, x, xnew);
        hala::axpy(1.0, b, xnew);

        if (num_rhs == 1)
            hala::trsv('L', 'N', 'N', N, A, xnew);
        else
            hala::trsm('L', 'L', 'N', 'N', N, num_rhs, 1.0, A, xnew);

        hala::vswap(x, xnew);

        hala::axpy(-1.0, x, xnew);
        error = hala::batch_max_norm2(num_rhs, xnew);

        iterations++;
    }while(error > stop_tolerance and iterations < max_iter);
}
//! [Example_01 gsc]

//! [Example_01 inta]
template<typename precision>
void integrate_a(int num_steps,
                 int max_iter,
                 typename std::vector<precision>::value_type stop_tolerance,
                 std::vector<precision> const &op,
                 std::vector<precision> &state){
    // integrate state from t = 0 till t = 1 using dt = 1.0 / num_steps and
    // Crank-Nicholson scheme (I - dt/2 * op) f^{n+1} = (I + dt/2 * op) f^n
    int size = (int) state.size();
    assert( op.size() == (size_t) size * size );

    precision dt = 1.0 / num_steps;

    // set update = ( I - dt/2 * op )
    std::vector<precision> update;
    hala::vcopy(op, update);
    hala::scal(-0.5 * dt, update); // update = -0.5 * dt * op

    std::vector<precision> ones(size, 1.0);
    hala::axpy(size, 1.0, ones, 1, update, size+1); // update += I

    for(int t=0; t<num_steps/10; t++){

        std::vector<precision> rhs;
        hala::gemv('N', size, size, 0.5 * dt, op, state, 0.0, rhs);
        hala::axpy(1.0, state, rhs);

        solve_gauss_seidel_a(max_iter, stop_tolerance, update, rhs, state);
    }
}
//! [Example_01 inta]

//! [Example_01 intb]
template<typename precision>
void integrate_b(int num_steps,
                 int max_iter,
                 typename std::vector<precision>::value_type stop_tolerance,
                 std::vector<precision> const &op,
                 std::vector<precision> &state,
                 int num_states = 1){
    // Crank-Nicholson scheme (I - dt/2 * op) f^{n+1} = (I + dt/2 * op) f^n

    int size = (int) state.size() / num_states;
    assert( op.size() == (size_t) size * size );
    assert( state.size() == (size_t) size * num_states );

    precision dt = 1.0 / num_steps;

    // set update = ( I - dt/2 * op )
    std::vector<precision> update;
    hala::vcopy(op, update);
    hala::scal(-0.5 * dt, update); // update = -0.5 * dt * op

    std::vector<precision> ones(size, 1.0);
    hala::axpy(size, 1.0, ones, 1, update, size+1); // update += I

    for(int t=0; t<num_steps/10; t++){

        std::vector<precision> rhs;
        hala::gemm('N', 'N', size, num_states, size, 0.5 * dt, op, state, 0.0, rhs);
        hala::axpy(1.0, state, rhs);

        solve_gauss_seidel_b(max_iter, stop_tolerance, num_states, update, rhs, state);
    }
}
//! [Example_01 intb]

//! [Example_01 intc]
template<typename precision>
void integrate_c(int num_steps,
                 int max_iter,
                 typename std::vector<precision>::value_type stop_tolerance,
                 std::vector<precision> const &op,
                 std::vector<precision> &state,
                 int num_states = 1){
    // Crank-Nicholson scheme (I - dt/2 * op) f^{n+1} = (I + dt/2 * op) f^n

    int size = (int) state.size() / num_states;
    assert( op.size() == (size_t) size * size );
    assert( state.size() == (size_t) size * num_states );

    precision dt = 1.0 / num_steps;

    // set update = ( I - dt/2 * op )
    std::vector<precision> update;
    hala::vcopy(op, update);
    hala::scal(-0.5 * dt, update); // update = -0.5 * dt * op

    std::vector<precision> ones(size, 1.0);
    hala::axpy(size, 1.0, ones, 1, update, size+1); // update += I

    for(int t=0; t<num_steps/10; t++){
        // pad rhs (if necessary)
        std::vector<precision> rhs = (num_states > 5 and num_states < 10) ?
                                      std::vector<precision>(10 * size) :
                                      std::vector<precision>();

        if (num_states == 1)
            hala::gemv('N', size, size, 0.5 * dt, op, state, 0.0, rhs);
        else
            hala::gemm('N', 'N', size, num_states, size, 0.5 * dt, op, state, 0.0, rhs);
        hala::axpy(1.0, state, rhs);

        if (num_states <= 5){
            for(int i=0; i<num_states; i++)
                solve_gauss_seidel_c(max_iter, stop_tolerance, 1, update,
                                     hala::wrap_array(&rhs[i * size], size),
                                     hala::wrap_array(&state[i * size], size));
        }else if (num_states <10){
            std::vector<precision> padded_state(10 * size);
            hala::vcopy(state, padded_state); // state determines the number of entries to copy
            solve_gauss_seidel_c(max_iter, stop_tolerance, 10, update, rhs, padded_state);
            hala::vcopy(num_states * size, padded_state, 1, state, 1);
        }else{
            solve_gauss_seidel_c(max_iter, stop_tolerance, num_states, update, rhs, state);
        }
    }
}
//! [Example_01 intc]

//! [Example_01 main]
int main(int argc, char**){
    if (argc > 1) return 0; // used to test the linker, ignore for the example
    int num_nodes = 1000;   // number of discrete nodes

    using precision = double;
    precision stop_tolerance = 1.E-3;
    int max_iter = 1000000;

    // initialize the state and the fractional-Laplacian operator
    auto laplacian = make_flaplace<precision>(num_nodes);
    auto state     = make_initial<precision>(num_nodes);

    hala::chronometer meter; // used for benchmarking

    cout << "Using BLAS level 2\n";
    meter.set_start();
    integrate_a(1000, max_iter, stop_tolerance, laplacian, state);
    long long duration = meter.get_end();
    cout << " Integrated" << setw(4) << "one" << " in" << setw(6) << duration << " milliseconds." << "\n";

    cout << "Using BLAS level 3\n";
    for(int num_states : std::vector<int>{1, 5, 10, 100}){
        // the multi-state is just multiple identical states lumped together
        // normally each state would be different, but that is beyond the scope here
        state = make_initial<precision>(num_nodes);
        std::vector<precision> multistate(num_nodes * num_states);
        for(int i=0; i<num_states; i++) hala::vcopy(state, &multistate[i * num_nodes]);

        meter.set_start();
        integrate_b(1000, max_iter, stop_tolerance, laplacian, multistate, num_states);
        duration = meter.get_end();
        cout << " Integrated" << setw(4) << num_states << " in" << setw(6) << duration << " milliseconds." << "\n";
    }

    cout << "Using BLAS dynamic\n";
    for(int num_states : std::vector<int>{1, 5, 10, 100}){
        state = make_initial<precision>(num_nodes);
        std::vector<precision> multistate(num_nodes * num_states);
        for(int i=0; i<num_states; i++) hala::vcopy(state, &multistate[i * num_nodes]);

        meter.set_start();
        integrate_c(1000, max_iter, stop_tolerance, laplacian, multistate, num_states);
        duration = meter.get_end();
        cout << " Integrated" << setw(4) << num_states << " in" << setw(6) << duration << " milliseconds." << "\n";
    }
    return 0;
//! [Example_01 main]
}


template<typename precision>
std::vector<precision> make_domain(int N){
    std::vector<precision> x(N);
    x[0] = precision(1.0) / precision(2 * N);
    for(int i=1; i<N; i++)
        x[i] = x[i-1] + precision(1.0) / precision(N);
    return x;
}

template<typename precision>
std::vector<precision> make_flaplace(int N){
    precision const s = 3.0 / 4.0;
    precision const alpha = 1.0 + 2 * s;

    auto x = make_domain<precision>(N);

    std::vector<precision> L(N * N);

    for(int i=0; i<N; i++){
        L[i * N + i] = 0.0;
        for(int j=0; j<N; j++){
            if (i != j){
                precision c = precision(1.0) / std::pow(std::abs(x[i] - x[j]), alpha);
                L[j * N + i]  = c;
                L[i * N + i] -= c;
            }
        }
    }
    for(auto &l : L) l *= 0.299206710301075 / static_cast<precision>(N);

    return L;
}

template<typename precision>
std::vector<precision> make_initial(int N){
    auto x = make_domain<precision>(N);
    for(auto &s : x) s = 1.0 - s*s;
    return x;
}
