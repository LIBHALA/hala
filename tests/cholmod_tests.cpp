/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */
#include "testing_common.hpp"

template<typename T> void test_cholmod();

int main(int argc, char**){

    verbose = (argc > 1);
    std::string name = "HALA CHOLMOD TESTS";
    begin_report(name);

    perform(test_cholmod<double>);

    end_report(name);

    return 0;
}

template<typename T> void test_cholmod(){
    current_test<T> tests("cholmod");
    hala::cholmod_factor<T> f;
    hala::cholmod_factor<T> g = std::move(f);
    f = std::move(g);

    std::vector<int> pntr = {0, 2, 5, 8, 11, 13};
    std::vector<int> indx = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    std::vector<T> vals = {2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0};
    std::vector<T> xref = {1.0, 2.0, 3.0, 4.0, 5.0}, x, b;

    hala::sparse_gemv('N', 5, 5, 1.0, pntr, indx, vals, xref, 0.0, b);
    hala::cholmod_factorize_spd(pntr, indx, vals).solve(b, x);

    hassert(testvec(x, xref));

    // spd approximate
    pntr = {0, 3, 6, 9, 12, 15};
    indx = {0, 1, 4, 0, 1, 2, 1, 2, 3, 2, 3, 4, 0, 3, 4};
    vals = {2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0};

    x = std::vector<T>();
    hala::sparse_gemv('N', 5, 5, 1.0, pntr, indx, vals, xref, 0.0, b);
    auto spd_factor = hala::cholmod_factorize_spd(pntr, indx, vals);
    std::swap(f, spd_factor);
    g = std::move(f);
    std::swap(g, spd_factor);
    spd_factor.solve(b, x);

    hassert(testvec(x, xref));


    // non-spd exat
    pntr = {0, 2, 5, 8, 11, 13};
    indx = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    vals = {2.0, -1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, -1.0, 2.0};

    x = std::vector<T>();
    hala::sparse_gemv('N', 5, 5, 1.0, pntr, indx, vals, xref, 0.0, b);
    hala::cholmod_factorize_gen(pntr, indx, vals).solve(b, x);

    hassert(testvec(x, xref));


    // non-spd approximate
    pntr = {0, 3, 6, 9, 12, 15};
    indx = {0, 1, 4, 0, 1, 2, 1, 2, 3, 2, 3, 4, 0, 3, 4};
    vals = {2.0, 1.0, -1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, -1.0, 1.0, 1.0, 2.0};

    x = std::vector<T>();
    hala::sparse_gemv('N', 5, 5, 1.0, pntr, indx, vals, xref, 0.0, b);
    hala::cholmod_factorize_gen(pntr, indx, vals).solve(b, x);

    hassert(testvec(x, xref));
}
