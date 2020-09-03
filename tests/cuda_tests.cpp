/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */
#include "cuda_core_tests.hpp"
#include "cuda_blas0_tests.hpp"
#include "cuda_blas1_tests.hpp"
#include "cuda_blas2_tests.hpp"
#include "cuda_blas3_tests.hpp"
#include "cuda_sparse_tests.hpp"

std::vector<std::function<void(void)>> all_tests = {
    []()->void{
        load_unload<float>();
        load_unload<double>();
        load_unload<std::complex<float>>();
        load_unload<std::complex<double>>();
    },
    []()->void{
        test_geam<float>();
        test_geam<double>();
        test_geam<std::complex<float>>();
        test_geam<std::complex<double>>();
    },
    []()->void{
        test_dgmm<float>();
        test_dgmm<double>();
        test_dgmm<std::complex<float>>();
        test_dgmm<std::complex<double>>();
    },
    []()->void{
        test_trtp<float>();
        test_trtp<double>();
        test_trtp<std::complex<float>>();
        test_trtp<std::complex<double>>();
    },
    []()->void{
        transpose<float>();
        transpose<double>();
        transpose<std::complex<float>>();
        transpose<std::complex<double>>();
    },
    []()->void{
        norm2<float>();
        norm2<double>();
        norm2<std::complex<float>>();
        norm2<std::complex<double>>();
    },
    []()->void{
        dot<float>();
        dot<double>();
        dot<std::complex<float>>();
        dot<std::complex<double>>();
    },
    []()->void{
        axpy<float>();
        axpy<double>();
        axpy<std::complex<float>>();
        axpy<std::complex<double>>();
    },
    []()->void{
        scal<float>();
        scal<double>();
        scal<std::complex<float>>();
        scal<std::complex<double>>();
    },
    []()->void{
        gemv<float>();
        gemv<double>();
        gemv<std::complex<float>>();
        gemv<std::complex<double>>();
    },
    []()->void{
        tbsv<float>();
        tbsv<double>();
        tbsv<std::complex<float>>();
        tbsv<std::complex<double>>();
    },
    []()->void{
        gemm<float>();
        gemm<double>();
        gemm<std::complex<float>>();
        gemm<std::complex<double>>();
    },
    []()->void{
        syrk<float>();
        syrk<double>();
        syrk<std::complex<float>>();
        syrk<std::complex<double>>();
    },
    []()->void{
        sparse_matvec<float>();
        sparse_matvec<double>();
        sparse_matvec<std::complex<float>>();
        sparse_matvec<std::complex<double>>();
    },
};

int main(int argc, char**){

    cout << std::scientific;
    cout.precision(6);

    verbose = (argc > 1);
    if (verbose) cout << "using cuda api: " << __HALA_CUDA_API_VERSION__ << "\n";

    std::string name = "HALA CUDA TESTS";
    begin_report(name);

    for(auto const &test : all_tests) perform(test);

    end_report(name);

    if (verbose) cout << "used cuda api: " << __HALA_CUDA_API_VERSION__ << "\n";

    return test_result();
}
