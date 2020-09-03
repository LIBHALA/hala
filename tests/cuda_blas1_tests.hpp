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

float mconj(float x){ return x; }
double mconj(double x){ return x; }
std::complex<float> mconj(std::complex<float> x){ return std::conj(x); }
std::complex<double> mconj(std::complex<double> x){ return std::conj(x); }


template<typename T> void norm2(){
    current_test<T> tests("norm2");
    int N = 10;
    hala_rnd = 0;
    std::valarray<T> x(N);
    for(auto &v : x) v = getVal(v);

    T nref = hala::get_cast<T>(0.0);
    for(auto v : x) nref += mconj(v) * v;
    nref = std::sqrt(nref);

    hala::gpu_engine engine;
    auto gpux = engine.load(x);
    auto n = norm2(engine, gpux);
    hassert(testnum(n, nref));
}
template<typename T> void dot(){
    current_test<T> tests("dot");
    int N = 10;
    hala_rnd = 0;
    std::valarray<T> x(N);
    for(auto &v : x) v = getVal(v);
    std::vector<T> y(N);
    for(auto &v : y) v = getVal(v);

    T conj_ref = hala::get_cast<T>(0.0);
    for(int i=0; i<N; i++) conj_ref += mconj(x[i]) * y[i];

    T dot_ref = hala::get_cast<T>(0.0);
    for(int i=0; i<N; i++) dot_ref += x[i] * y[i];

    hala::gpu_engine engine;
    auto gpux = engine.load(x);
    auto gpuy = engine.load(y);
    hassert(testnum(conj_ref, dot(engine, gpux, gpuy)));
    hassert(testnum(dot_ref,  dotu(engine, gpux, gpuy)));
}
template<typename T> void axpy(){
    current_test<T> tests("axpy");
    int N = 8;
    hala_rnd = 0;
    std::valarray<T> x(N);
    for(auto &v : x) v = getVal(v);
    std::valarray<T> y(N);
    for(auto &v : y) v = getVal(v);

    T alpha = hala::get_cast<T>(2.0);
    auto yref = y;
    yref += alpha * x;

    hala::gpu_engine eng;
    auto gpux = eng.load(x);
    auto gpuy = eng.load(y);
    axpy(eng, alpha, gpux, gpuy);
    gpuy.unload(y);
    hassert(testvec(y, yref));
}
template<typename T> void scal(){
    current_test<T> tests("scal");
    int N = 8;
    hala_rnd = 0;
    std::valarray<T> x(N);
    for(auto &v : x) v = getVal(v);

    T alpha = hala::get_cast<T>(2.0);
    auto xref = x;
    xref *= alpha;

    hala::gpu_engine eng;
    auto gpux = eng.load(x);
    hala::scal(eng, alpha, gpux);
    gpux.unload(x);
    hassert(testvec(x, xref));
}
