#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>

#include "hala.hpp"
#include "hala_vex.hpp"

using std::cout;
using std::endl;
using std::setw;

/*!
 * \file example_vex.cpp
 * \brief Example of using the BLAS module of HALA.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAEXAMPLE3
 *
 * Example 3, Using the HALA vectorization extensions.
 */


/*!
 * \ingroup HALAEXAMPLE
 * \addtogroup HALAEXAMPLE3 Example of VEX usage
 *
 * \par Example 3
 * Example 3 demonstrates the usage of the \ref HALAVEX module that utilizes intrinsic functions
 * to perform multiple floating point operations in a single instruction.
 * The data for each operation has to be stored in packed structures using either hala::mmpack
 * or hala::bind_pack, and the standard arithmetic operations are supported.
 * The example uses ordinary differential equations with different random parameters,
 * multiple equations need to be integrated and statistics is collected from the data.
 * The vectorization operations are a natural way to lump several equations together and significantly
 * increase the performance.
 *
 * \par
 * The example is divided into several sections:
 * - \ref Example3Problem "Problem Setup"
 * - \ref Example3Basic "Basic Implementation"
 * - \ref Example3Vector "Vectorized Implementation"
 * - \ref Example3Perform "Performance Boost Comments"
 *
 * \anchor Example3Problem
 * \par Problem Setup
 * Consider the SIR model defined by the following system of differential equations:
 * \f{eqnarray*}{
 *     \frac{dS}{dt} &=& - \beta I S \\
 *     \frac{dI}{dt} &=& \beta I S - \gamma I \\
 *     \frac{dR}{dt} &=& \gamma I
 * \f}
 * We use the initial conditions \f$ S = 0.99, I = 0.01, R = 0 \f$ all defined at \f$ t = 0\f$.
 * The parameters \f$ \beta, \gamma \f$ are uniformly distributed over \f$ (0.1, 0.3) \f$.
 *
 * \par
 * Given a large sample of random values for \f$ \beta, \gamma \f$ we seek to compute
 * the percentage of samples that will result in \f$ I < 0.01 \f$ at \f$ t = 100 \f$.
 *
 * \par
 * The integration is done using Runge–Kutta order 4 method. For an autonomous equation
 * \f$ \frac{dx}{dt} = f(x) \f$, the method advances from \f$ x(t) \to x(t + \Delta t) \f$ using the 4 steps:
 * \f{eqnarray*}{
 *      k_1 &=& f(x) \\
 *      k_2 &=& f(x + 0.5 \Delta t k_1) \\
 *      k_3 &=& f(x + 0.5 \Delta t k_2) \\
 *      k_4 &=& f(x + \Delta t k_4) \\
 *      x(t + \Delta t) &\approx & x + \frac{\Delta t}{6} \left( k_1 + 2 k_2 + 2 k_3 + k_4 \right)
 * \f}
 * The specific interpretation of the equations and the properties of the integration scheme
 * are not of focus. This example demonstrates an efficient way to integrate the equations.
 *
 * \anchor Example3Basic
 * \par Basic Implementation
 * Vectors with random values for the parameters are generated in main() and passed into the compute function.
 * Two nested for-loops compute the result, the outer loop goes over all entries of the parameter vectors,
 * while the inner loop implements the Runge-Kutta methods.
 * At the end of the outer loop, the state for \b I is compared to the threshold and if lower the corresponding
 * count is incremented.
 *
 * \snippet examples/example_vex.cpp Example_03 basic
 *
 * \anchor Example3Vector
 * \par Vectorized Implementation
 * The vectorized implementation differs in only a few ways:
 * - The hala_vex.hpp header needs to be included in the code.
 * - The method is templated on the register type, although the \b reg parameter can be omitted from all calls
 * and HALA will default to the largest register that has been enabled by the current set of flags.
 * - The size of the \b beta and \b gamma vectors is restricted to a multiple of the vector stride,
 * i.e., the number of doubles that can fit in the selected register.
 * The restriction can be removed if we have a second loop that can handle the remainder,
 * in which case the body of the loop can be moved to a separate template to avoid repeating the Runge-Kutta code.
 * - The rest of the code is changed only slightly, the \b s, \b i and \b r variables
 * are replaced by packs of doubles, the hala::load method is used to load multiple entries
 * from the \b beta and \b gamma vectors, and all the \b k variables are defined as \b auto
 * type which will be resolved as instances of hala::mmpack.
 * - The if-condition to count the number of lower results has to loop over the entries
 * in the vector variables. The direct indexing of the entries in a hala::mmpack is relatively slow,
 * but the one time access is negligible in the overall cost.
 *
 * \snippet examples/example_vex.cpp Example_03 vector
 *
 * \anchor Example3Perform
 * \par Performance Boost Comments
 * Running the test on an AMD Ryzen 3900X CPU yields the following results:
 * \code
 *        Mode              Result        Milliseconds
 *      basic:        8.837891e-01                 186
 *        def:        8.837891e-01                  46
 *       none:        8.837891e-01                 181
 *    128-bit:        8.837891e-01                  90
 *    256-bit:        8.837891e-01                  46
 * \endcode
 * The specific results will be affected by the CPU type and frequency, memory speed, and other factors;
 * however, the observed behavior is what we expect.
 *
 * - The basic and vectorized implementations yield identical result, in some cases rounding error
 * may be different but it should not be significant for a given application.
 * - Compared the basis implementation, the 128-bit and 256-bit registers yield respective
 * speedup of 2 and 4 which is proportional to the register size (note that double has 64-bits).
 * - The default register (\b def) is selected as the largest 256-bit one.
 * - The implementation using hala::regtype::none is equivalent to the basic implementation,
 * thus only one implementation is needed even if extended registers are not available.
 *
 * \par
 * The timing is very sensitive to the optimization flags, make sure to use \b -O3 optimization
 * which will strip away all the fluff brought by the HALA templates and will leave only
 * the direct calls to the intrinsic functions. Without the \b -O3 optimization,
 * the vectorized implementation will be significantly slower.
 *
 * \snippet examples/example_vex.cpp Example_03 main
 */

//! [Example_03 basic]
double integrate_sir_basic(std::vector<double> const &beta, std::vector<double> const &gamma){
    size_t const num_samples = beta.size();
    size_t num_lower = 0;
    assert(beta.size() == gamma.size());

    for(size_t j=0; j<num_samples; j++){
        // set the initial conditions
        double r = 0.0, i = 0.01, s = 0.99;
        // load the current rates
        double b = beta[j], g = gamma[j];

        double dt = 1.E-2, dt2 = 0.5 * dt;
        for(size_t t = 0; t < 10000; t++){
            double k1_s = - b * i * s;
            double k1_i = b * i * s - g * i;
            double k1_r = g * i;

            double k2_s = - b * (i + dt2 * k1_i) * (s + dt2 * k1_s);
            double k2_i = b * (i + dt2 * k1_i) * (s + dt2 * k1_s) - g * (i + dt2 * k1_i);
            double k2_r = g * (i + dt2 * k1_i);

            double k3_s = - b * (i + dt2 * k2_i) * (s + dt2 * k2_s);
            double k3_i = b * (i + dt2 * k2_i) * (s + dt2 * k2_s) - g * (i + dt2 * k2_i);
            double k3_r = g * (i + dt2 * k2_i);

            double k4_s = - b * (i + dt * k3_i) * (s + dt * k3_s);
            double k4_i = b * (i + dt * k3_i) * (s + dt * k3_s) - g * (i + dt * k3_i);
            double k4_r = g * (i + dt * k3_i);

            s = s + dt * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s) / 6.0;
            i = i + dt * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) / 6.0;
            r = r + dt * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r) / 6.0;
        }

        if (i < 0.01) num_lower++;
    }
    return static_cast<double>(num_lower) / static_cast<double>(num_samples);
}
//! [Example_03 basic]

//! [Example_03 vector]
template<hala::regtype reg = hala::default_regtype>
double integrate_sir_vex(std::vector<double> const &beta, std::vector<double> const &gamma){
    using mmpack = hala::mmpack<double, reg>;
    size_t const num_samples = beta.size();
    size_t num_lower = 0;
    assert(beta.size() == gamma.size());
    hala::runtime_assert(num_samples % mmpack::stride == 0,
                         "the size of beta and gamma must be divisible by the pack stride");

    for(size_t j=0; j<num_samples; j += mmpack::stride){
        mmpack s = 0.99;
        mmpack i = 0.01;
        mmpack r = 0.0;

        auto b = hala::mmload<reg>(&beta[j]);
        auto g = hala::mmload<reg>(&gamma[j]);

        mmpack dt = 1.E-2;
        mmpack dt2 = 0.5 * dt;
        for(size_t t = 0; t < 10000; t++){
            auto k1_s = -1.0 * b * i * s;
            auto k1_i = b * i * s - g * i;
            auto k1_r = g * i;

            auto k2_s = -1.0 * b * (i + dt2 * k1_i) * (s + dt2 * k1_s);
            auto k2_i = b * (i + dt2 * k1_i) * (s + dt2 * k1_s) - g * (i + dt2 * k1_i);
            auto k2_r = g * (i + dt2 * k1_i);

            auto k3_s = -1.0 * b * (i + dt2 * k2_i) * (s + dt2 * k2_s);
            auto k3_i = b * (i + dt2 * k2_i) * (s + dt2 * k2_s) - g * (i + dt2 * k2_i);
            auto k3_r = g * (i + dt2 * k2_i);

            auto k4_s = -1.0 * b * (i + dt * k3_i) * (s + dt * k3_s);
            auto k4_i = b * (i + dt * k3_i) * (s + dt * k3_s) - g * (i + dt * k3_i);
            auto k4_r = g * (i + dt * k3_i);

            s += dt * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s) / 6.0;
            i += dt * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) / 6.0;
            r += dt * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r) / 6.0;
        }

        for(size_t t = 0; t < i.stride; t++)
            if (i[t] < 0.01) num_lower++;

    }
    return static_cast<double>(num_lower) / static_cast<double>(num_samples);
}
//! [Example_03 vector]

//! [Example_03 main]
int main(int, char**){
    cout << std::scientific; cout.precision(6);

    // generate random distribution for beta and gamma
    std::minstd_rand park_miller(42);
    std::uniform_real_distribution<double> unif_distribution(0.1, 0.3);

    constexpr size_t num_samples = 1024; // must be multiple of the default stride
    std::vector<double> beta(num_samples), gamma(num_samples);

    for(auto &b : beta)  b = unif_distribution(park_miller);
    for(auto &g : gamma) g = unif_distribution(park_miller);

    hala::chronometer meter; // times the calls to integrate the equations

    // basic implementation with no vectorization
    meter.set_start();
    double result_basic = integrate_sir_basic(beta, gamma);
    long long duration_basic = meter.get_end();

    // implementation with vectorization and default registers
    meter.set_start();
    double result_vector = integrate_sir_vex(beta, gamma);
    long long duration_vector = meter.get_end();

    cout << setw(10) << "Mode" << setw(20) << "Result" << setw(20) << "Milliseconds" << "\n";
    cout << setw(10) << "basic:" << setw(20) << result_basic
         << setw(20) << duration_basic << "\n";
    cout << setw(10) << "default:" << setw(20) << result_vector
         << setw(20) << duration_vector << "\n";

    // implementation using the templated vectorized code but removing actual vectorization
    // i.e., defaulting to an implementation that is equivalent to the basic mode
    meter.set_start();
    double result_none = integrate_sir_vex<hala::regtype::none>(beta, gamma);
    long long duration_none = meter.get_end();
    cout << setw(10) << "none:" << setw(20) << result_none
         << setw(20) << duration_none << "\n";

    #ifdef __SSE3__
    // the __SSE3__ macro is enabled by the -msse3 flag and enables the 128-bit registers
    meter.set_start();
    double result_sse = integrate_sir_vex<hala::regtype::sse>(beta, gamma);
    long long duration_sse = meter.get_end();
    cout << setw(10) << "128-bit:" << setw(20) << result_sse
         << setw(20) << duration_sse << "\n";
    #endif

    #ifdef __AVX__
    // the __SSE3__ macro is enabled by the -mavx flag and enables the 256-bit registers
    meter.set_start();
    double result_avx = integrate_sir_vex<hala::regtype::avx>(beta, gamma);
    long long duration_avx = meter.get_end();
    cout << setw(10) << "256-bit:" << setw(20) << result_avx
         << setw(20) << duration_avx << "\n";
    #endif

    return 0;
//! [Example_03 main]
}
