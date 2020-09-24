/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */
#define USE_ENGINE_INTERFACE

#include "testing_common.hpp"

// instantiated explicitly in blas_with_noengine.cpp and bals_with_engine.cpp
// the actual list of tests is created in blas_tests.hpp
// the tests are defined in blas(1, 2, 3)_tests.hpp
// see the comments in blas_tests.hpp
template<class cengine> std::vector<std::function<void(void)>> make_tests_wengine(cengine const &engine);
template<class cengine> std::vector<std::function<void(void)>> make_tests_noengine(cengine const &engine);

int main(int argc, char**){

    cout << std::scientific;
    cout.precision(6);

    verbose = (argc > 1);
    std::string name = "HALA BLAS TESTS";
    begin_report(name);

    hala::cpu_engine ecpu;
    auto all_tests = make_tests_wengine(ecpu);

    for(auto const &test : all_tests) perform(test);

    begin_report(std::string("Testing no-engine variants "));
    all_tests = make_tests_noengine(ecpu);
    for(auto const &test : all_tests) perform(test);

    #if defined(HALA_ENABLE_CUDA) || defined(HALA_ENABLE_ROCM)
    for(int gpuid=0; gpuid<hala::gpu_device_count(); gpuid++){
        begin_report(std::string("Testing GPU BLAS: device ") + std::to_string(gpuid) + " (with engine)");

        hala::gpu_engine ecuda(gpuid);
        hala::mixed_engine emixed(ecuda);

        all_tests = make_tests_wengine(emixed);
        for(auto const &test : all_tests) perform(test);

        #ifdef HALA_ENABLE_ROCM
        hala::gpu_engine dummy(gpuid); // no clue why this is needed by ROCm, it is not the obvious
        #endif
        begin_report(std::string("Testing GPU BLAS: device ") + std::to_string(gpuid) + " (no engine)");
        all_tests = make_tests_noengine(emixed);
        for(auto const &test : all_tests) perform(test);
    }
    #endif

    end_report(name);

    return test_result();
}


