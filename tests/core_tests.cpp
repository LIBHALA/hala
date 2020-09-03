/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "core_tests.hpp"

std::vector<std::function<void(void)>> all_tests = {
    []()->void{
        type_casting();
    },
    []()->void{
        default_internal_vector<float>();
        default_internal_vector<double>();
    },
    []()->void{
        custom_complex();
    },
    []()->void{
        custom_vector();
    },
    []()->void{
        raw_arrays();
    },
};

int main(int argc, char**){

    cout << std::scientific;
    cout.precision(6);

    verbose = (argc > 1);
    std::string name = "HALA CORE FUNCTIONALITY TESTS";
    begin_report(name);

    for(auto const &test : all_tests) perform(test);

    end_report(name);

    return test_result();
}
