#include <iostream>
#include <iomanip>
#include <valarray>
#include <numeric>

#include "hala.hpp"
#include "hala_solvers.hpp"

using std::cout;
using std::endl;
using std::setw;

/*!
 * \file example_gpu.cpp
 * \brief Example of using the BLAS module of HALA.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAEXAMPLE2
 *
 * Example 2, Using BLAS and CUDA/ROCM.
 */


/*!
 * \ingroup HALAEXAMPLE
 * \addtogroup HALAEXAMPLE2 Example of GPU usage
 *
 * \par Example 2
 * Example 1 was a demonstration of the dense (BLAS) module of HALA that implemented a time-stepping scheme and a linear solver.
 * The implementation used the optimized BLAS library on the CPU.
 * Example 2 demonstrates the sparse capabilities of HALA by implementing a preconditioned conjugate-gradient method
 * which can be used either on the CPU or GPU.
 *
 * \par
 * The example is divided into several sections:
 * - \ref Example2Problem "Problem Setup"
 * - \ref Example2PCG "Preconditioned Conjugate Gradient method"
 * - \ref Example2PCGPU "GPU Implementation"
 * - \ref Example2PCMIX "Mixed Engine Overload"
 * - \ref Example2Solvers "Native HALA Solvers"
 *
 * \anchor Example2Problem
 * \par Problem Setup
 * Consider the elliptic partial differential equation \f$ - \Delta u = f \f$ in two-dimensions  over the unit square
 * with homogeneous Dirichlet boundary conditions. The forcing term is \f$ f(x, y) = 12 x^2 (y - y^3) + 6 y (x - x^4) \f$
 * which was chosen to yield an analytic solution is \f$ u(x, y) = (x - x^4)(y - y^3) \f$.
 * We approximate the equation using a finite difference scheme using central difference for the second derivative.
 * The intricacies of the equation and discretization are not in scope of HALA,
 * it suffices to know that the two methods provided here discretize_pde() and compute_l2_error()
 * can be used to generate the sparse matrix and right-hand-side, and both work with std::valarray
 * containers to demonstrate the generality of HALA, i.e., the HALA templates are not restricted to std::vector
 * or other HALA specific containers.
 *
 * \anchor Example2PCG
 * \par Preconditioned Conjugate-Gradient method
 * Consider a sparse linear system of equations \f$ A u = f \f$, where \b A is the approximation to the Laplacian operator
 * described in the previous section, \b f is the approximate forcing term and \b u is the approximate solution.
 * The matrix \b A is symmetric-positive-definite and a good approach for solving such systems is the
 * <a href="https://en.wikipedia.org/wiki/Conjugate_gradient_method#:~:text=The%20preconditioned%20conjugate%20gradient%20method,-See%20also%3A%20Preconditioner&text=The%20preconditioner%20matrix%20M%20has,change%20from%20iteration%20to%20iteration.&text=An%20example%20of%20a%20commonly%20used%20preconditioner%20is%20the%20incomplete%20Cholesky%20factorization.">
 * \b preconditioned \b conjugate-gradient \b method. </a> The implementation is as follows:
 *
 * \snippet examples/example_gpu.cpp Example_02 pcg
 *
 * \par
 * The first thing to note is the generic \b VectorLike types for all the inputs, this is similar to
 * the solve_gauss_seidel_c() method in \ref Example1VarC "Example 1".
 * The full potential of this generality will become apparent later, for now we go over the section of the code
 * in a step by step manner:
 * - as an iterative method, the CG algorithm takes as a stopping criteria either an error tolerance or a maximum number of iteration
 * - couple of assertions check for proper sizes of the inputs, and note that HALA will perform additional type,
 *   e.g., hala::make_sparse_matrix() will check that the sizes of \b vals and \b indx match
 * - \b x is expected to be initialized with the zeroth iterate, e.g., zero vector
 * - the sparse matrix triple is wrapped in a generic matrix struct and external workspace is allocated for
 *   the matrix-vector multiplication
 * - the Incomplete Lower Upper factorization is computed with hala::make_ilu()
 * - temporary vectors are created with the hala::new_vector() command, which simply creates a default vector
 *   (e.g., std::vector) with scalar type matching x
 * - the rest is direct translation of the algorithm into a series of BLAS operations, as well as
 *   hala::sparse_gemv() and application of the ILU preconditioner.
 *
 * \par
 * Using the two discretization methods, we solve the PDE using different number of points for the grid in the uniform
 * discretization, and we plot the number of iterations, duration, degrees of freedom (problem size) and error.
 * As expected, we observe convergence that is approximately linear in the degrees of freedom or quadratic in
 * the size of the box discretization. Each time we assume that the initial guess is zero.
 *
 * \snippet examples/example_gpu.cpp Example_02 main_cpu
 *
 * \par Result
 * \code
 *  iter  duration      size          error
 *     9         0        16   1.000499e-03
 *    14         0        64   3.092774e-04
 *    23         0       256   8.669565e-05
 *    41         4      1024   2.300756e-05
 *    78        31      4096   5.930229e-06
 *   152       421     16384   1.505631e-06
 * \endcode
 *
 * \anchor Example2PCGPU
 * \par GPU Implementation
 * The most expensive part of the CG method is the matrix vector product and the preconditioner and while the BLAS operations
 * use the optimized backend library, the sparse operations are implementated native to HALA and are probably far from optimal.
 * Sparse operations depend primarily to the memory speed and GPU (even consumer grade or gaming cards) have memory that
 * is much faster than the main CPU one; and GPU frameworks such as CUDA and ROCm provide free accelerated libraries
 * with the needed sparse implementations (ROCm is even open source).
 * Thus, it is natural to seek a GPU accelerated version of the CG algorithm.
 *
 * \par
 * The accelerated libraries provided by both CUDA and ROCm require GPU handle variables that indicate
 * the current device in use and other implementation specific meta-data.
 * In HALA, those are wrapped in a hala::gpu_engine object,
 * and a straight forward way to port an algorithm to the GPU is to modify the code so that each call
 * to HALA would take an engine variable, including the call to solve_pcg().
 * However, that requires modification to the code and will result in errors if even one call is missed;
 * not to mention that the CPU implementations would also have to carry a hala::cpu_engine object
 * which is redundant.
 *
 * \par
 * An alternative approach is to use the generality of the \b VectorLike classes and control the GPU-CPU selection
 * by creating a special type of vectors, which bind a GPU or a CPU context to any user provided vector container.
 * Thus, the original code requires no modifications, but a simple additional overload:
 *
 * \snippet examples/example_gpu.cpp Example_02 pcge
 *
 * \par
 * The engine overload can be called with any of the three HALA engines, hala::cpu_engine, hala::gpu_engine
 * or hala::mixed_engine, and the original solve_pcg() will be instantiated in the appropriate context.
 *
 * \par
 * The main method needs to be modified to provide the appropriate engine context and to move the date from
 * the CPU over to the GPU side. Note that the entire block needs a pre-processor \b ifndef guard
 * since the hala::gpu_engine and the load and unload methods are available only when a GPU backend is enabled.
 *
 * \snippet examples/example_gpu.cpp Example_02 main_gpu
 *
 * \anchor Example2PCMIX
 * \par Mixed Engine Overload
 * The load and unload operations performed in the main are tedious and many applications follow a CPU bound
 * workflow with only some operations offloaded to the GPU, e.g., as it is in our case where we offload
 * the solver. HALA provides a hala::mixed_engine mode where the inputs sit in CPU memory and are being
 * transferred for a single call, only to be transferred back with the result. The mixed mode could be
 * beneficial in some operations, e.g., large dense hala::gemm() where the number of operations far exceeds
 * the data being transferred; however, in a memory bound context of a sparse linear solver where many
 * BLAS 1 and BLAS 2 operations are utilized, the the transfer between each operation can negate any
 * benefits from the GPU acceleration.
 *
 * \par
 * Avoiding repeated data-transfer can be achieved with a single additional overload that encompasses
 * all load/unload operations.
 *
 * \snippet examples/example_gpu.cpp Example_02 pcgm
 *
 * \par
 * The mixed overload will be favored over the use of the tempated overload due to the explicit use
 * of hala::mixed_engine. The standard load commands are used and the hala::gpu_bind_vector()
 * method is used to bind a CPU and GPU vectors together, so that when the GPU container is destroyed
 * the data will be written back to the CPU one. This enables RAII style of resource management
 * and prevents us from accidentally forgetting to call unload.
 * In addition, the main method call to the overload can be called just like the original
 * CPU-only implementation whith only the engine as an extra variable.
 *
 * \snippet examples/example_gpu.cpp Example_02 main_gpm
 *
 * The neede for pre-processor guard directives is also reduced to the definition of the two engines.
 *
 * \par Engine vs. Engined-Vector
 * The hala::bind_engine_vector() approach used in the example offers several advantages:
 * - the original CPU-only code does not need to be changed or updated, which is favorable
 *   when working with complex and already tested algorithms
 * - there is no need to carry the redundant engine variable to each call
 *
 * \par
 * On the other hand, the engine offers other advantages:
 * - the engine calls are more explicit which is advantageous in cases when CPU and GPU bound
 *   methods are mixed in a single algorithm, e.g., GMRES where the matrix-vector product
 *   is computed on the GPU while the orthogonalization coefficients are handled on the CPU
 * - the template overload resolution is simpler and less prone to compiler quirks
 *
 * \anchor Example2Solvers
 * \par Native HALA Solvers
 * HALA includes several sparse iterative solvers, including the preconditioned conjugate-gradient
 * with the options for a ILU and a user provided preconditioner.
 * See the \ref HALASOLVERS solvers documentation.
 */

/*!
 * \ingroup HALAEXAMPLE2
 * \brief Computes the discretization of the PDE and the right hand side.
 *
 * - \b num_side is the number of points to use on each side square grid,
 *               the size of the linear system is num_side squared by num_side squared
 * - \b pntr, \b indx, \b vals describe the sparse matrix that approximates the left-hand-side operator
 * - \b f is the discretization of the right-hand-side forcing term
 */
template<typename scalar_type>
void discretize_pde(int num_side, std::valarray<int> &pntr, std::valarray<int> &indx, std::valarray<scalar_type> &vals, std::valarray<scalar_type> &f);

/*!
 * \ingroup HALAEXAMPLE2
 * \brief Computes the l-2 error for the approximate solution \b u to the PDE.
 */
template<typename scalar_type>
scalar_type compute_l2_error(std::valarray<scalar_type> &u);

//! [Example_02 pcg]
template<class VectorLikeP,  class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int solve_pcg(int max_iter, hala::get_precision_type<VectorLikeV> stop_tolerance,
              VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
              VectorLikeB const &b, VectorLikeX &x){
    assert( b.size() == x.size() );

    int num_rows = hala::get_size_int(pntr) - 1;
    assert( b.size() == static_cast<size_t>(num_rows) );

    auto matrix = hala::make_sparse_matrix(num_rows, pntr, indx, vals);
    auto ilu = hala::make_ilu(pntr, indx, vals, 'N');

    size_t buffer_size = std::max(
                             ilu_buffer_size(ilu, b, x, 1),
                             sparse_gemv_buffer_size(matrix, 'N', 1.0, b, 0.0, x)
                              );
    auto work_buffer = hala::new_vector(x);
    hala::set_size(buffer_size, work_buffer);

    auto r = hala::new_vector(x);
    auto p = hala::new_vector(x);
    auto Ap = hala::new_vector(x);
    auto z  = hala::new_vector(x);

    hala::vcopy(b, r);
    hala::sparse_gemv(matrix, 'N', -1.0, x, 1.0, r, work_buffer);
    hala::ilu_apply(ilu, r, z); // in wikipedia notation, z = M^-1 r

    hala::vcopy(z, p);
    auto zr = hala::dot(r, z);

    int iterations = 1; // (above) applied one Ax with preconditioner
    bool iterate = true;

    while(iterate){
        hala::sparse_gemv(matrix, 'N', 1.0, p, 0.0, Ap, work_buffer);
        iterations++;

        auto nzr = hala::dot(Ap, p);
        auto alpha = zr / nzr;

        hala::axpy( alpha,  p, x);
        hala::axpy(-alpha, Ap, r);

        iterate = not ((iterations >= max_iter) or (hala::norm2(r) < stop_tolerance));

        if (iterate){ // skip this part of done already
            hala::ilu_apply(ilu, r, z, 1, work_buffer);

            nzr = hala::dot(r, z);

            hala::scal(nzr / zr, p);
            hala::axpy(1.0, z, p);

            zr = nzr;
        }
    }

    return iterations;
}
//! [Example_02 pcg]

//! [Example_02 pcge]
template<class EngineType,
         class VectorLikeP,  class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int solve_pcg(EngineType const &engine,
              int max_iter, hala::get_precision_type<VectorLikeV> stop_tolerance,
              VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
              VectorLikeB const &b, VectorLikeX &x){
    auto bind_pntr = bind_engine_vector(engine, pntr);
    auto bind_indx = bind_engine_vector(engine, indx);
    auto bind_vals = bind_engine_vector(engine, vals);
    auto bind_b = bind_engine_vector(engine, b);
    auto bind_x = bind_engine_vector(engine, x);
    return solve_pcg(max_iter, stop_tolerance,
                     bind_pntr, bind_indx, bind_vals, bind_b, bind_x);
}
//! [Example_02 pcge]

//! [Example_02 pcgm]
#ifdef HALA_ENABLE_GPU
template<class VectorLikeP,  class VectorLikeI, class VectorLikeV, class VectorLikeX, class VectorLikeB>
int solve_pcg(hala::mixed_engine const &engine,
              int max_iter, hala::get_precision_type<VectorLikeV> stop_tolerance,
              VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals,
              VectorLikeB const &b, VectorLikeX &x){
    auto gpu_pntr = engine.gpu().load(pntr);
    auto gpu_indx = engine.gpu().load(indx);
    auto gpu_vals = engine.gpu().load(vals);
    auto gpu_b = engine.gpu().load(b);
    auto gpu_x = hala::gpu_bind_vector(engine.gpu(), x);
    return solve_pcg(engine.gpu(), max_iter, stop_tolerance,
                     gpu_pntr, gpu_indx, gpu_vals, gpu_b, gpu_x);
}
#endif
//! [Example_02 pcgm]

//! [Example_02 main]
int main(int, char**){
    cout << std::scientific; cout.precision(6);

    //! [Example_02 main_cpu]
    double tolerance = 1.E-8;
    int max_iter = 100000; // just pick a large number
    hala::chronometer meter;

    cout << "\nCompute using the CPU\n";
    cout << setw(5) << "iter" << setw(10) << "duration"
         << setw(10) << "size" << setw(15) << "error" << "\n";
    for(int n : std::vector<int>{4, 8, 16, 32, 64, 128}){

        std::valarray<int> pntr, indx;
        std::valarray<double> vals, f;

        discretize_pde(n, pntr, indx, vals, f);

        std::valarray<double> u(0.0, f.size());

        meter.set_start();
        int iter = solve_pcg(max_iter, tolerance, pntr, indx, vals, f, u);
        long long duration = meter.get_end();
        // iter == max_iter means that the tolerance was not reached
        assert( iter < max_iter);

        cout << setw(5) << iter << setw(10) << duration
             << setw(10) << u.size() << setw(15) << compute_l2_error(u) << "\n";
    }
    //! [Example_02 main_cpu]

    #ifdef HALA_ENABLE_GPU
    cout << "\nCompute using the GPU\n";
    cout << setw(5) << "iter" << setw(10) << "duration"
         << setw(10) << "size" << setw(15) << "error" << "\n";

    //! [Example_02 main_gpu]
    hala::gpu_engine egpu(0);
    for(int n : std::vector<int>{4, 8, 16, 32, 64, 128}){

        std::valarray<int> pntr, indx;
        std::valarray<double> vals, f;

        discretize_pde(n, pntr, indx, vals, f);
        std::valarray<double> u(0.0, f.size());

        meter.set_start();
        auto gpu_pntr = egpu.load(pntr);
        auto gpu_indx = egpu.load(indx);
        auto gpu_vals = egpu.load(vals);
        auto gpu_f = egpu.load(f);
        auto gpu_u = egpu.load(u);
        int iter = solve_pcg(egpu, max_iter, tolerance, gpu_pntr, gpu_indx, gpu_vals,
                             gpu_f, gpu_u);
        gpu_u.unload(u);
        long long duration = meter.get_end();

        std::valarray<double> cpu_u(0.0, f.size());
        solve_pcg(hala::cpu_engine(), max_iter, tolerance, pntr, indx, vals, f, cpu_u);
        hala::axpy(-1.0, u, cpu_u);
        assert(hala::norm2(cpu_u) < 1.E-11);

        cout << setw(5) << iter << setw(10) << duration
             << setw(10) << u.size() << setw(15) << compute_l2_error(u) << "\n";
    }
    //! [Example_02 main_gpu]
    #endif

    cout << "\nCompute using the GPU and mixed-engine\n";
    cout << setw(5) << "iter" << setw(10) << "duration"
         << setw(10) << "size" << setw(15) << "error" << "\n";

    //! [Example_02 main_gpm]
    #ifdef HALA_ENABLE_GPU
    hala::mixed_engine engine(egpu);
    #else
    hala::cpu_engine engine;
    #endif
    for(int n : std::vector<int>{4, 8, 16, 32, 64, 128}){

        std::valarray<int> pntr, indx;
        std::valarray<double> vals, f;

        discretize_pde(n, pntr, indx, vals, f);

        std::valarray<double> u(0.0, f.size());

        meter.set_start();
        int iter = solve_pcg(engine, max_iter, tolerance, pntr, indx, vals, f, u);
        long long duration = meter.get_end();
        assert( iter < max_iter);

        cout << setw(5) << iter << setw(10) << duration
             << setw(10) << u.size() << setw(15) << compute_l2_error(u) << "\n";
    }
    //! [Example_02 main_gpm]

    return 0;
//! [Example_02 main]
}

template<typename scalar_type>
void discretize_pde(int num_side, std::valarray<int> &pntr, std::valarray<int> &indx, std::valarray<scalar_type> &vals, std::valarray<scalar_type> &f){
    // first form the operator
    int num_nnz = num_side * num_side + 4 * num_side * (num_side - 1);
    pntr = std::valarray<int>(0, num_side * num_side + 1);
    indx = std::valarray<int>(0, num_nnz);
    vals = std::valarray<scalar_type>(0.0, num_nnz);

    std::valarray<int> iota(0, pntr.size());
    std::iota(std::begin(iota), std::end(iota), 0);

    int line_begin = 0;
    for(int i=0; i<num_side; i++){
        int stride = (i == 0 or i == num_side-1) ? 4 : 5;
        int offset = 0;

        // low-neighbor
        if (i > 0){ // if there is a row of nodes below
            int next_low = (i == num_side -1) ? 3 : 4;
            vals[line_begin] = -1.0;
            indx[line_begin] = (i-1) * num_side;
            vals[std::slice(line_begin + next_low, num_side-1, stride)] = -1.0;
            indx[std::slice(line_begin + next_low, num_side-1, stride)] =  iota[std::slice((i-1)*num_side +1, num_side-1, 1)];
            offset++;
        }

        // left-neighbor, the +4 skips the left neighbor for the left-most node
        int next_left = (i == num_side -1) ? 2 : 3;
        vals[std::slice(line_begin + 2*offset + next_left, num_side-1, stride)] = -1.0;
        indx[std::slice(line_begin + 2*offset + next_left, num_side-1, stride)] =  iota[std::slice(i*num_side, num_side-1, 1)];

        // self-neighbor
        vals[std::slice(line_begin + offset, num_side, stride)] =  4.0;
        indx[std::slice(line_begin + offset, num_side, stride)] =  iota[std::slice(i*num_side, num_side, 1)];
        offset++;

        // right-neighbor
        vals[std::slice(line_begin + offset, num_side-1, stride)] = -1.0;
        indx[std::slice(line_begin + offset, num_side-1, stride)] =  iota[std::slice(i*num_side +1, num_side-1, 1)];
        offset++;

        // top-neighbor
        if (i < num_side - 1){ // if there is a row of nodes above
            vals[std::slice(line_begin + offset, num_side-1, stride)] = -1.0;
            indx[std::slice(line_begin + offset, num_side-1, stride)] =  iota[std::slice((i+1)*num_side, num_side-1, 1)];
            vals[line_begin + offset + stride *(num_side-1) -1] = -1.0;
            indx[line_begin + offset + stride *(num_side-1) -1] = (i+2)*num_side - 1;
        }
        line_begin += offset + stride *(num_side-1);

        // count number of neighbors per node, the first and last node on the row have one neighbor less
        pntr[std::slice(i*num_side, num_side, 1)] = stride;
        pntr[i*num_side] -= 1;
        pntr[(i+1)*num_side -1] -= 1;
    }

    int count = 0;
    for(auto &p : pntr)
        p = std::exchange(count, count + p); // replace pntr[i] with count, and increment count to count + pntr[i]

    scalar_type dx = 1.0 / static_cast<scalar_type>(num_side+1);
    vals /= dx * dx;

    // form the right-hand-side forcing vector
    f = std::valarray<scalar_type>(0.0, num_side * num_side);
    for(int i=0; i<num_side; i++){
        scalar_type y = dx * (i+1);
        for(int j=0; j<num_side; j++){
            scalar_type x = dx * (j+1);
            f[i * num_side + j] = 12.0 * x * x * (y - y * y * y) + 6 * y * (x - x * x * x * x);
        }
    }
}

template<typename scalar_type>
scalar_type compute_l2_error(std::valarray<scalar_type> &u){
    int num_side = 0;
    for(size_t i=1; i<=u.size(); i++){
        if (u.size() == i*i){
            num_side = static_cast<int>(i);
            break;
        }
    }
    hala::runtime_assert(u.size() == static_cast<size_t>(num_side * num_side), "invalid size for u");

    scalar_type dx = 1.0 / static_cast<scalar_type>(num_side+1);

    scalar_type err = 0.0;
    for(int i=0; i<num_side; i++){
        scalar_type y = dx * (i+1);
        for(int j=0; j<num_side; j++){
            scalar_type x = dx * (j+1);
            scalar_type e = u[i * num_side + j] - (x - x * x * x * x) * (y - y * y * y);
            err += e * e;
        }
    }

    return std::sqrt(err) / static_cast<scalar_type>(num_side + 1);
}
