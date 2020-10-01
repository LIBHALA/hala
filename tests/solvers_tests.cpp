/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "solvers_tests.hpp"

template<typename cengine>
std::vector<std::function<void(void)>> make_tests(cengine const &engine){
    return std::vector<std::function<void(void)>>{
        [&]()->void{
            test_cg<float>(engine);
            test_cg<double>(engine);
            test_cg<std::complex<float>>(engine);
            test_cg<std::complex<double>>(engine);
        },
        [&]()->void{
            test_cg_batch<float>(engine);
            test_cg_batch<double>(engine);
            test_cg_batch<std::complex<float>>(engine);
            test_cg_batch<std::complex<double>>(engine);
        },
        [&]()->void{
            test_gmres<float>(engine);
            test_gmres<double>(engine);
            test_gmres<std::complex<float>>(engine);
            test_gmres<std::complex<double>>(engine);
        },
    };
}

int main(int argc, char**){

    cout << std::scientific;
    cout.precision(6);

    verbose = (argc > 1);
    std::string name = "HALA SOLVER TESTS";
    begin_report(name);

    hala::cpu_engine const ecpu;
    auto all_tests = make_tests(ecpu);

    for(auto const &test : all_tests) perform(test);

    #ifdef HALA_ENABLE_GPU
    for(int gpuid=0; gpuid<hala::gpu_device_count(); gpuid++){
        begin_report(std::string("Testing GPU SOLVERS: device ") + std::to_string(gpuid));

        hala::gpu_engine ecuda(gpuid);
        hala::mixed_engine emixed(ecuda);

        all_tests = make_tests(emixed);
        for(auto const &test : all_tests) perform(test);
    }
    #endif

    end_report(name);

    return test_result();
}
