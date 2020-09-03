/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#define USE_ENGINE_INTERFACE // brings in the global variables as "extern"
#include "sparse_tests.hpp"

std::vector<std::function<void(void)>> make_tests_cpu(){
    return std::vector<std::function<void(void)>>{
        []()->void{
            test_split_matrix<float>();
            test_split_matrix<double>();
            test_split_matrix<std::complex<float>>();
            test_split_matrix<std::complex<double>>();
        },
        []()->void{
            test_ilu<float>();
            test_ilu<double>();
            test_ilu<std::complex<float>>();
            test_ilu<std::complex<double>>();
        },
    };
}

template<class cengine> std::vector<std::function<void(void)>> make_tests_wengine(cengine const &engine);
template<class cengine> std::vector<std::function<void(void)>> make_tests_noengine(cengine const &engine);

int main(int argc, char**){

    verbose = (argc > 1);
    std::string name = "HALA SPARSE TESTS";
    begin_report(name);

    hala::cpu_engine ecpu;
    auto all_tests = make_tests_wengine(ecpu);
    for(auto const &test : all_tests) perform(test);

    all_tests = make_tests_cpu();
    for(auto const &test : all_tests) perform(test);

    begin_report(std::string("Testing no-engine variants "));
    all_tests = make_tests_noengine(ecpu);
    for(auto const &test : all_tests) perform(test);

    #ifdef HALA_ENABLE_GPU
    for(int gpuid=0; gpuid<hala::gpu_device_count(); gpuid++){
        begin_report(std::string("Testing GPU SPARSE: device ") + std::to_string(gpuid) + " (with engine)");

        hala::gpu_engine ecuda(gpuid);
        hala::mixed_engine emixed(ecuda);

        all_tests = make_tests_wengine(emixed);
        for(auto const &test : all_tests) perform(test);

        begin_report(std::string("Testing GPU BLAS: device ") + std::to_string(gpuid) + " (no engine)");
        all_tests = make_tests_noengine(emixed);
        for(auto const &test : all_tests) perform(test);
    }
    #endif

    end_report(name);

    return test_result();
}
