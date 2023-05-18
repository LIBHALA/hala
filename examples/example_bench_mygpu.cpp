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
 * \file example_bench_mygpu.cpp
 * \brief Example of using GEMM operations to benchmark the GPU
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAEXAMPLE4
 *
 * Example 4, Using GPU BLAS to benchmark the GPU
 */


/*!
 * \ingroup HALAEXAMPLE
 * \addtogroup HALAEXAMPLE4 Example of GPU benchmark
 *
 */


template<typename T>
std::vector<T> make_matrix(int n){
    std::vector<T> result(n*n);
    std::iota(result.begin(), result.end(), -std::floor( std::sqrt(n*n) ));
    return result;
}


//! [Example_04 main]
int main(int, char**){
    cout << std::fixed; cout.precision(2);

    int n = 0;
    hala::chronometer meter;

    using precision = float;

    hala::gpu_engine egpu(0);

    cout << "multiplying three matrices of size N by N leveraging BLAS level 3\n";
    cout << "   the flop count assumes (2N-1) * N * N flops\n";
    cout << "   depending on 'shortcuts' in the implementation that may be an overestimate\n";

    for(int iter=0; iter<10; iter++){
        n += 1000;
        auto A = make_matrix<precision>(n);
        auto B = make_matrix<precision>(n);
        std::vector<precision> C(n*n);

        auto gA = egpu.load(A);
        auto gB = egpu.load(B);
        auto gC = egpu.load(C);

        meter.set_start();
        hala::gemm(egpu, 'N', 'N', n, n, n, 1, gA, gB, 0, gC);
        egpu.synchronize();
        long long duration = meter.get_end();

        long long flops = (2 * static_cast<long long>(n) - 1)
                          * static_cast<long long>(n)
                          * static_cast<long long>(n);

        // Gflops applies scale of 1.E-9 while duration is in milliseconds with is 1.E+3
        cout << " size n = " << setw(6) << n
             << "   performance: " << setw(8) << 1.E-6 * static_cast<double>(flops) / static_cast<double>(duration)
             << " Gflops/second\n";
    }


    return 0;
//! [Example_04 main]
}

