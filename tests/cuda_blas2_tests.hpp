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

template<typename T> void gemv(){
    current_test<T> tests("gemv");
    int M = 3, N = 2;
    std::vector<T> A(M * N), x(N), y, yref;

    hala_rnd = -2;
    for(auto &a : A) a = getVal(a);
    for(auto &v : x) v = getVal(v);

    refgemm('N', 'N', M, 1, N, hala::get_cast<T>(2.0), A, x, hala::get_cast<T>(0.0), yref);

    hala::gpu_engine engine;
    auto gpuA = engine.load(A);
    auto gpux = engine.load(x);
    hala::gpu_vector<T> gpuy;
    gemv(engine, 'N', M, N, hala::get_cast<T>(2.0), gpuA, gpux, hala::get_cast<T>(0.0), gpuy);
    gpuy.unload(y);

    hassert(testvec(y, yref));
}
template<typename T> void tbsv(){
    current_test<T> tests("tbsv");
    int N = 3, K = 1;
    std::vector<T> A = {
        hala::get_cast<T>(2.0), hala::get_cast<T>(1.0),
        hala::get_cast<T>(2.0), hala::get_cast<T>(1.0),
        hala::get_cast<T>(2.0), hala::get_cast<T>(1.0)
    };
    std::vector<T> x    = { hala::get_cast<T>(2.0), hala::get_cast<T>(5.0), hala::get_cast<T>(8.0) };
    std::vector<T> xref = { hala::get_cast<T>(1.0), hala::get_cast<T>(2.0), hala::get_cast<T>(3.0) };

    hala::gpu_engine engine;
    auto gpua = engine.load(A);
    auto gpux = engine.load(x);

    tbsv(engine, 'L', 'N', 'N', N, K, gpua, gpux);
    x = gpux.unload();

    hassert(testvec(x, xref));
}
