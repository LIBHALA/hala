/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */
#include "vex_tests.hpp"

std::vector<std::function<void(void)>> all_tests = {
    []()->void{
        test_extended_registers<float,  hala::regtype::none>();
        test_extended_registers<double, hala::regtype::none>();
        test_extended_registers<std::complex<float>, hala::regtype::none>();
        test_extended_registers<std::complex<double>, hala::regtype::none>();
    },
    []()->void{
        test_extended_registers<float,  hala::default_regtype>();
        test_extended_registers<double, hala::default_regtype>();
        test_extended_registers<std::complex<float>,  hala::default_regtype>();
        test_extended_registers<std::complex<double>, hala::default_regtype>();
    },
    []()->void{
        #ifdef __SSE3__
        test_extended_registers<float,  hala::regtype::sse>();
        test_extended_registers<double, hala::regtype::sse>();
        test_extended_registers<std::complex<float>,  hala::regtype::sse>();
        test_extended_registers<std::complex<double>, hala::regtype::sse>();
        #endif
    },
    []()->void{
        #ifdef __AVX__
        test_extended_registers<float,  hala::regtype::avx>();
        test_extended_registers<double, hala::regtype::avx>();
        test_extended_registers<std::complex<float>, hala::regtype::avx>();
        test_extended_registers<std::complex<double>, hala::regtype::avx>();
        #endif
    },
    []()->void{
        #ifdef __AVX512F__
        test_extended_registers<float,  hala::regtype::avx512>();
        test_extended_registers<double, hala::regtype::avx512>();
        test_extended_registers<std::complex<float>, hala::regtype::avx512>();
        test_extended_registers<std::complex<double>, hala::regtype::avx512>();
        #endif
    },
    []()->void{
        test_aligned_allocator<std::vector, hala::regtype::none>();
        #ifdef __SSE3__
        test_aligned_allocator<std::vector, hala::regtype::sse>();
        #endif
        #ifdef __AVX__
        test_aligned_allocator<std::vector, hala::regtype::avx>();
        #endif
        #ifdef __AVX512F__
        test_aligned_allocator<std::vector, hala::regtype::avx512>();
        #endif
    },
    []()->void{
        test_default<float>();
        test_default<double>();
        test_default<std::complex<float>>();
        test_default<std::complex<double>>();
    }
};

int main(int argc, char**){

    cout << std::scientific;
    cout.precision(6);

    verbose = (argc > 1);
    std::string name = "HALA VECTORIZATION TESTS";
    begin_report(name);

    for(auto const &test : all_tests) perform(test);

    end_report(name);

    return test_result();
}
