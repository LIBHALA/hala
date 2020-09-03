#include <iostream>
#include <iomanip>

#include "hala.hpp"

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
 * The previous example demonstrated how to build a simple Gauss-Seidel solver and
 * a Crank-Nicholson time integrator coupled together to solve a simple
 * integro-differential equation.
 * The focus fell on generalizing the solvers for different types, problems sizes and containers.
 * This example builds on the previous algorithms and puts them on the GPU device.
 *
 * \par Increased Problem Size
 * The problem size in this scenario is bumped to 3000 nodes and the same discretization and
 * integration schemes are used.
 * We ignore the tuning for small sizes and consider only the cases of 1, 10, 100, and 300
 * simultaneous problems.
 * The results from using the different implementations is listed at the bottom of the example.
 *
 * \par
 * The example is divided into several sections:
 * - \ref Example2Solver        "Updated Solver"
 * - \ref Example2Integrator    "Updated Integrator"
 * - \ref Example2MakeOnes      "Make-Ones Overloads"
 * - \ref Example2EngineMode    "Engine Overload"
 * - \ref Example2MixedOverload "Mixed-Engine Overload"
 * - \ref Example2Remarks       "Remarks on Main"
 * - \ref Example2Results       "Remarks and Results"
 *
 *
 * \anchor Example2Solver
 * \par Updated Solver
 * The CPU and GPU methods operate on different memory spaces and call different backend libraries,
 * thus a general solver template has to "know" which implementation to use. Furthermore, the CUDA
 * libraries require an additional handle variable that contains implementation specific information.
 * HALA propagates such information in one of two ways:
 * -# explicitly pass an engine object to each hala call
 * -# call the method with hala::engined_vector objects that carry the engine context
 *
 * \par
 * The first approach is more suitable for algorithms that use both calls to the CPU and GPU backends,
 * e.g., GMRES where the matrix-vector product and the Krylov basis are computed on the GPU while
 * the coefficients of the basis expansion and the Givens rotations are handled strictly on the CPU.
 * The explicit engine approach is also good for sporadically making linear algebra calls within
 * a generally non-linear algorithm.
 * The engined-vector approach is better suited for algorithms that use a single backend by avoiding
 * the repetition of the additional variable.
 * This example uses the second approach.
 *
 * \par
 * Using example 1 variant C as a starting point, we make very few changes:
 * - We generalize every container to a vector-like class.
 * - All dimension checks are done with hala::get_size().
 * - The precision is determined using hala::get_precision_type.
 * - The type of the \b error variable can be set to the precision, or alternatively the \b decltype
 *   trick can be used to assign the type without making a real call to hala::norm2().
 * - The hala::new_vector() command is used to create an empty temporary vector consistent with the
 *   rest of the vectors (e.g., CPU bound, GPU bound, engined or not).
 * - The individual operations and methods remain the same.
 *
 * \snippet examples/example_gpu.cpp Example_02 gsa
 *
 * \anchor Example2Integrator
 * \par Updated Integrator
 * The time integrator has to be generalized in a similar way. The inputs are updated to general
 * vector-like classes, and the precision and temporary vector are set the same way. An additional
 * challenge is presented from the vector of ones which must be created in an appropriate context.
 *
 * \par
 * A set of overloads is needed to differentiate the context and create an appropriate vector of ones.
 * One approach would be to create an overloaded methods that create either a CPU bound vector
 * or a hala::engined_vector that \a owns a CPU or a GPU vector and takes a reference to an engine
 * using one of the existing vectors. In this approach, the overloads have to differentiate between
 * a hala::engined_vector, the different CPU/GPU engines bound to the vector, and a general vector.
 * While viable, it can be a bit tricky to get the compiler to pick the correct overload,
 * overloads using different compute engines are less ambiguous since they do not require
 * a parameter with a free auto-deduced type.
 * In this example, we use a simpler two stage approach that uses one extra variable which is
 * either a standard reference or a non-owning hala::engined_vector.
 *
 * \par
 * The no-engine method can be instantiated in one of two contexts, either using general
 * CPU bound vectors or engined-vectors. The hala::get_engine() method will either
 * create a new instance of hala::cpu_engine (in the general context), or return the engine
 * associated with the \b state variable (in the engined context).
 * The make_ones() methods are described in \ref Example2MakeOnes "Make-Ones Overloads"
 * section, here we note that result is a non-engined vector that sits either in CPU or GPU
 * memory depending on the engine.
 * The hala::bind_engine_vector() method is used to then either create a simple reference
 * to the \b ones vector or create an engined vector using the same engine as the \b state.
 *
 * \par
 * In the general context:
 * - \b ones is a std::vector with scalar type matching \b state
 * - \b rebind_ones is a reference to \b ones
 *
 * \par
 * In the engined vector context:
 * - \b ones is either a std::vector or hala::cuda_vector depending whether the \b state vector
 *  was created with a hala::cpu_engine or hala::gpu_engine
 * - \b rebind_ones is an engined vector with the same engine as \b state and holding a non-owning
 *  reference to \b ones
 *
 * \snippet examples/example_gpu.cpp Example_02 inta
 *
 * \anchor Example2MakeOnes
 * \par Make Ones Overloads
 * The make_ones() overloads take a compute engine and return an appropriate vector of ones:
 * - The scalar type is inferred from the \b VectorLike template parameter which is deduced from the input vector,
 *   even thought the actual vector object is not used.
 * - The hala::get_standard_type template is used to get one of the four standard types float, double,
 *   std::complex<float> or std::complex<double> that corresponds to the type of the vector-like;
 *   the standard type is used (as opposed to the scalar type) to ensure that the type would have a constructor
 *   that accepts the constant 1.0 as input.
 * - The mixed and cpu engines simply return std::vectors while the gpu engine loads the vectors and returns
 *   the results (by default a hala::cuda_vector).
 * - Note that the mixed engine overload is used only for testing, see further down.
 *
 * \snippet examples/example_gpu.cpp Example_02 mkveca
 * \snippet examples/example_gpu.cpp Example_02 mkvecb
 * \snippet examples/example_gpu.cpp Example_02 mkvecc
 *
 * \anchor Example2EngineMode
 * \par Engine Overload
 * The updated integrator and solver can be instantiated with either generic CPU vectors or with
 * hala::engined_vector that contain a reference to the appropriate engine.
 * However, both the engine and the engined vectors have to be created and bound together;
 * thus, we add a general integrator overloaded template that accepts CPU or GPU vectors and
 * an engine object.
 * The operator and state vectors are simply bound to the engine and the resulting engined vectors
 * are used in the call to the no-engine method.
 *
 * \par
 * Several notes:
 * - The vectors and engine must be compatible, e.g., hala::cuda_vector is not compatible with hala::cpu_engine;
 *  for general vectors there is no way to check whether they are bound to the CPU or GPU or whether
 *  the correct GPU device is used on a system with multiple devices.
 * - There is no need to add an overload to the solver method, since the solver is not called
 *  directly but only though the integrator. The solvers included in HALA have the corresponding
 *  overloads and it is good practice to add those in a general library, for a specific problem,
 *  it is sufficient to overload only the top method in a stack of calls.
 * - The compiler uses the number of input parameters to differentiate between the engine
 *  and no-engine templates, which is sufficient for this example.
 *  If additional template constraints are needed, the hala::is_engine template can be used to
 *  check whether the engine type is appropriate, e.g.,
 *  \code
 *      template<class compute_engine, class VectorLikeOp, class VectorLikeState>
 *      std::enable_if_t<hala::is_engine<compute_engine>::value>
 *      integrate(compute_engine const &engine, int num_steps, ...
 *  \endcode
 *
 * \par
 * Here is the engine overload template:
 *
 * \snippet examples/example_gpu.cpp Example_02 intb
 *
 *
 * \anchor Example2MixedOverload
 * \par Mixed-Engine Overload
 * The GPU can improve performance by as much as an order of magnitude; however, using the gpu_engine can be
 * disruptive in the code-flow when all the data sits on the CPU
 * since each use of the integrator would require the data to be pushed to the GPU and the result pulled back.
 * For this reason, HALA offers the hala::mixed_engine that accepts all input and outputs on the CPU,
 * but the computations are performed on the GPU.
 * The mixed engine can speedup level 3 methods since the data moved back and forth is much less than
 * the total amount of work;
 * but for levels 1 and 2 (or methods that use a mix of levels),
 * the overhead of data-transfer can negate all GPU advantages and even make things slower.
 * The proper way to mix CPU and GPU workflow is to add yet another overload where data is transferred once at
 * the beginning of the call, then retrieved back in the end.
 * - Note the use of hala::cuda_bind_vector that will create a hala::binded_cuda_vector, a struc that
 *   will work the same as any other vector-like class but will write all data back to the CPU
 *   vector when the struct leaves scope (i.e., using RAII style of resource management).
 * - The mixed_engine overload also removes the need to have a make_ones specific to this engine since
 *   the main integrator will be instantiated only with the CPU and GPU engines.
 *
 * \snippet examples/example_gpu.cpp Example_02 intc
 *
 * \anchor Example2Remarks
 * \par Remarks on Main
 * In main(), the operator and state are constructed using CPU bound non-linear methods and
 * the integrator and solver are implemented with the accelerated wrappers of HALA.
 * - The reference BLAS implementation is no different than before.
 * - The gpu_engine and mixed_engine are available only when HALA_ENABLE_GPU is defines, otherwise
 *  the implementation will default to the cpu_engine (which will default to the BLAS case
 *  with no run-time overhead associated with the cpu_engine variable).
 * - Falling back to the cpu_engine makes the code portable across platforms with different computing capabilities.
 * - In the gpu_engine case, there is the extra code to modify the load/unload methods
 *  into reference aliases, the mixed_engine requires no additional pre-processor macros.
 *
 * \snippet examples/example_gpu.cpp Example_02 main
 *
 * \anchor Example2Results
 * \par Results
 * The timing experiments were performed on an Intel i9-9920X CPU with 12-cores and using the OpenBLAS
 * library and Quadro GV100 with CUDA 10.2. The timings will be different for different pairs of
 * devices and perhaps even different on another system with the same hardware chips.
 * The purpose of the example is to demonstrate how HALA can be used to gain access to the potential of
 * different computing devices, and in no way a benchmark or comparison between specific components.
 *
 * \par
 * Note that the mixed no-overload corresponds to simply running the integrator with the mixed
 * engine without the overload described in \ref Example2MixedOverload "Mixed-Engine Overload".
 *
 * \code
 * Using BLAS
 *  Integrated   1 in   9152 milliseconds.
 *  Integrated  10 in  47817 milliseconds.
 *  Integrated 100 in  55078 milliseconds.
 *  Integrated 300 in  88754 milliseconds.
 * Using cuBlas
 *  Integrated   1 in   4297 milliseconds.
 *  Integrated  10 in   4972 milliseconds.
 *  Integrated 100 in   8346 milliseconds.
 *  Integrated 300 in  13360 milliseconds.
 * Using cuBlas-mixed (no overload)
 *  Integrated   1 in  83699 milliseconds.
 *  Integrated  10 in  89880 milliseconds.
 *  Integrated 100 in 120497 milliseconds.
 *  Integrated 300 in 168932 milliseconds.
 * Using cuBlas-mixed (with overload)
 *  Integrated   1 in   4248 milliseconds.
 *  Integrated  10 in   4961 milliseconds.
 *  Integrated 100 in   8483 milliseconds.
 *  Integrated 300 in  13273 milliseconds.
 * \endcode
 */

/*!
 * \ingroup HALAEXAMPLE2
 * \brief Construct a vector that represents the mid-points of the domain.
 */
template<typename precision>
std::vector<precision> make_domain(int N);

/*!
 * \ingroup HALAEXAMPLE2
 * \brief Creates a simple discrete approximation to the fractional Laplacian operator.
 */
template<typename precision>
std::vector<precision> make_flaplace(int N);

/*!
 * \ingroup HALAEXAMPLE2
 * \brief Gives a quadratic initial profile over the domain.
 */
template<typename precision>
std::vector<precision> make_initial(int N);

//! [Example_02 gsa]
template<class VectorLikeA, class VectorLikeB, class VectorLikeX>
void solve_gauss_seidel(hala::stop_criteria<hala::get_precision_type<VectorLikeA>> stop,
                        int num_rhs,
                        VectorLikeA const &A,
                        VectorLikeB const &b,
                        VectorLikeX &&x){

    assert( num_rhs > 0 );
    assert( hala::get_size(x) == hala::get_size(b) );

    int N = (int) hala::get_size(x) / num_rhs;
    assert( hala::get_size(A) == (size_t) (N * N) );

    auto error = static_cast<decltype( hala::norm2(A) )>(0.0);
    int iterations = 0;

    do{
        auto xnew = hala::new_vector(x);
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
    }while((error > stop.tol) && (iterations < stop.max_iter));
}
//! [Example_02 gsa]

//! [Example_02 mkveca]
template<class VectorLike>
inline auto make_ones(hala::cpu_engine const &, size_t num, VectorLike const&){
    return std::vector<hala::get_standard_type<VectorLike>>(num, 1.0);
}
//! [Example_02 mkveca]

#ifdef HALA_ENABLE_GPU
//! [Example_02 mkvecb]
template<class VectorLike>
inline auto make_ones(hala::gpu_engine const &engine, size_t num, VectorLike const&){
    std::vector<hala::get_standard_type<VectorLike>> ones(num, 1.0);
    return engine.load(ones);
}
//! [Example_02 mkvecb]

//! [Example_02 mkvecc]
template<class VectorLike>
inline auto make_ones(hala::mixed_engine const &, size_t num, VectorLike const&){
    return std::vector<hala::get_standard_type<VectorLike>>(num, 1.0);
}
//! [Example_02 mkvecc]
#endif

//! [Example_02 inta]
template<class VectorLikeOp, class VectorLikeState>
void integrate(int num_steps,
               hala::stop_criteria<hala::get_precision_type<VectorLikeOp>> stop,
               VectorLikeOp const &op,
               VectorLikeState &state,
               int num_states = 1){

    // Crank-Nicholson scheme (I - dt/2 * op) f^{n+1} = (I + dt/2 * op) f^n
    int size = (int) hala::get_size(state) / num_states;
    assert( hala::get_size(op)    == (size_t) size * size );
    assert( hala::get_size(state) == (size_t) size * num_states );

    hala::get_precision_type<VectorLikeOp> dt = 1.0 / num_steps;

    // set update = ( I - dt/2 * op )
    auto update = hala::new_vector(op);
    hala::vcopy(op, update);
    hala::scal(-0.5 * dt, update); // update = -0.5 * dt * op

    auto ones = make_ones(hala::get_engine(state), size, state);
    auto rebind_ones = hala::bind_engine_vector(state, ones);
    hala::axpy(size, 1.0, rebind_ones, 1, update, size+1); // update += I

    for(int t=0; t<num_steps/10; t++){
        // pad rhs (if necessary)
        auto rhs = hala::new_vector(state);

        if (num_states == 1)
            hala::gemv('N', size, size, 0.5 * dt, op, state, 0.0, rhs);
        else
            hala::gemm('N', 'N', size, num_states, size, 0.5 * dt, op, state, 0.0, rhs);
        hala::axpy(1.0, state, rhs);

        solve_gauss_seidel(stop, num_states, update, rhs, state);
    }
}
//! [Example_02 inta]

#ifdef HALA_ENABLE_GPU
//! [Example_02 intb]
template<class compute_engine, class VectorLikeOp, class VectorLikeState>
void integrate(compute_engine const &engine,
               int num_steps,
               hala::stop_criteria<hala::get_precision_type<VectorLikeOp>> stop,
               VectorLikeOp const &op,
               VectorLikeState &state,
               int num_states = 1){
    auto engined_op    = hala::bind_engine_vector(engine, op);
    auto engined_state = hala::bind_engine_vector(engine, state);
    integrate(num_steps, stop, engined_op, engined_state, num_states);
}
//! [Example_02 intb]

//! [Example_02 intc]
template<class VectorLikeOp, class VectorLikeState>
void integrate(hala::mixed_engine const &engine,
               int num_steps,
               hala::stop_criteria<hala::get_precision_type<VectorLikeOp>> stop,
               VectorLikeOp const &op,
               VectorLikeState &state,
               int num_states = 1){
    auto gpu_op = engine.gpu().load(op);
    auto gpu_state = hala::gpu_bind_vector(engine.gpu(), state);
    integrate(engine.gpu(), num_steps, stop, gpu_op, gpu_state, num_states);
}
//! [Example_02 intc]
#endif

//! [Example_02 main]
int main(int argc, char**){
    if (argc > 1) return 0;
    int num_nodes = 3000;

    using precision = double;
    hala::stop_criteria<precision> stop = {1.E-3};

    auto laplacian = make_flaplace<precision>(num_nodes);
    auto state     = make_initial<precision>(num_nodes);

    hala::chronometer meter;

    cout << "Using BLAS\n";
    for(int num_states : std::vector<int>{1, 10, 100, 300}){
        state = make_initial<precision>(num_nodes);
        std::vector<precision> multistate(num_nodes * num_states);
        for(int i=0; i<num_states; i++) hala::vcopy(state, &multistate[i * num_nodes]);

        meter.set_start();
        integrate(1000, stop, laplacian, multistate, num_states);
        long long duration = meter.get_end();
        cout << " Integrated" << setw(4) << num_states << " in " << setw(6) << duration << " milliseconds." << "\n";
    }

    #ifdef HALA_ENABLE_GPU
    hala::gpu_engine   const cengine(0);
    hala::mixed_engine const mengine(cengine);
    #else
    hala::cpu_engine const cengine;
    hala::cpu_engine const mengine;
    #endif

    cout << "Using cuBlas\n";
    for(int num_states : std::vector<int>{1, 10, 100, 300}){
        state = make_initial<precision>(num_nodes);
        std::vector<precision> multistate(num_nodes * num_states);
        for(int i=0; i<num_states; i++) hala::vcopy(state, &multistate[i * num_nodes]);

        #ifdef HALA_ENABLE_GPU
        auto gpu_lap   = cengine.load(laplacian);
        auto gpu_state = cengine.load(multistate);
        #else
        auto &gpu_lap   = laplacian;
        auto &gpu_state = multistate;
        #endif

        meter.set_start();
        integrate(cengine, 1000, stop, gpu_lap, gpu_state, num_states);
        long long duration = meter.get_end();

        #ifdef HALA_ENABLE_GPU
        gpu_state.unload(state);
        #endif

        cout << " Integrated" << setw(4) << num_states << " in " << setw(6) << duration << " milliseconds." << "\n";
    }

    cout << "Using cuBlas-mixed\n";
    for(int num_states : std::vector<int>{1, 10, 100, 300}){
        state = make_initial<precision>(num_nodes);
        std::vector<precision> multistate(num_nodes * num_states);
        for(int i=0; i<num_states; i++) hala::vcopy(state, &multistate[i * num_nodes]);

        meter.set_start();
        integrate(mengine, 1000, stop, laplacian, multistate, num_states);
        long long duration = meter.get_end();
        cout << " Integrated" << setw(4) << num_states << " in " << setw(6) << duration << " milliseconds." << "\n";
    }
    return 0;
//! [Example_02 main]
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
