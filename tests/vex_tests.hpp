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

template<typename T> T mabs2(T a){ return a * a; }
template<typename T> std::complex<T> mabs2(std::complex<T> a){
    T r = std::abs(a)*std::abs(a);
    return {r, r};
}
template<typename T> T mabs(T a){ return std::abs(a); }
template<typename T> std::complex<T> mabs(std::complex<T> a){
    T r = std::abs(a);
    return {r, r};
}

template<typename T, hala::regtype reg> void test_extended_registers(){
    current_test<T> tests(regname(reg));
    hala_rnd = 0;
    size_t N = 128;
    auto x = make_vector<T>(N);
    auto y = make_vector<T>(N);
    auto y2 = y;
    auto y_ref = y;

    T alpha = static_cast<T>(2.0);
    for(size_t i=0; i<N; i++) y_ref[i] += alpha * x[i];

    // using vectorization, test multiply
    hala::mmpack<T, reg> scalar(alpha);
    for(size_t i=0; i<N; i+=scalar.stride){
        hala::mmpack<T, reg> chunky = &y[i];
        chunky += scalar * hala::mmload<reg>(&x[i]);
        chunky.put(&y[i]);
    }

    hassert(testvec(y, y_ref));

    // test multiply add (fmadd)
    for(size_t i=0; i<N; i+=scalar.stride){
        hala::mmpack<T, reg> chunky = &y2[i];
        chunky.fmadd(scalar, &x[i]);
        chunky.put(&y2[i]);
    }

    hassert(testvec(y2, y_ref));
    y = y_ref;

    auto vv = hala::mmzero<T, reg>();
    std::stringstream ss;
    ss << vv; // covers the << overwrites

    // test divide
    scalar = static_cast<T>(4.0);

    hala::mmpack<T, reg> vexx;
    for(size_t i=0; i<N; i+=scalar.stride)
        hala::mmbind<reg>(&y[i]) /= scalar;

    for(size_t i=0; i<N; i++) y_ref[i] /= static_cast<T>(4.0);

    hassert(testvec(y, y_ref));

    // test scale by mmpack
    y = y_ref;

    scalar -= hala::mmload<reg>(static_cast<T>(2.0));
    for(size_t i=0; i<N; i+=scalar.stride){
        auto v = hala::mmbind<reg>(&y[i]);
        v *= scalar;
    }

    for(size_t i=0; i<N; i++) y_ref[i] *= static_cast<T>(2.0);

    hassert(testvec(y, y_ref));

    // test scale by value
    for(size_t i=0; i<N; i+=scalar.stride)
        hala::mmbind<reg>(&y[i]) *= scalar;

    for(size_t i=0; i<N; i++) y_ref[i] *= static_cast<T>(2.0);

    hassert(testvec(y, y_ref));

    // test square-root
    x = y;
    y_ref = y;
    for(size_t i=0; i<N; i+=scalar.stride)
        hala::mmload<reg>(&x[i]).sqrt().put(&y[i]);

    for(size_t i=0; i<N; i+=scalar.stride)
        hala::mmbind<reg>(&x[i]).set_sqrt();

    for(size_t i=0; i<N; i++) y_ref[i] = std::sqrt(y_ref[i]);

    hassert(testvec(y, y_ref, 5.E+01));
    hassert(testvec(x, y_ref, 5.E+01));

    // test abs() and abs2()
    hala_rnd = -20;
    y = make_vector<T>(N);
    y_ref = y;
    auto y_ref2 = y;
    auto nrm = hala::hala_abs(y[0]);
    for(auto v : y) nrm = std::max(nrm, hala::hala_abs(v));

    for(size_t i=0; i<N; i+=scalar.stride)
        hala::mmload<reg>(&y[i]).abs().put(&x[i]);
    for(size_t i=0; i<N; i+=scalar.stride)
        hala::mmload<reg>(&y[i]).abs2().put(&y[i]);
    for(size_t i=0; i<N; i++) y_ref2[i] = mabs2(y_ref2[i]);
    for(size_t i=0; i<N; i++) y_ref[i] = mabs(y_ref[i]);

    hassert(testvec(x, y_ref, nrm * nrm));
    hassert(testvec(y, y_ref2, nrm * nrm));

    // test real and imag
    hala_rnd = 1;
    y = make_vector<T>(hala::mmpack<T, reg>::stride);

    for(size_t i=0; i<hala::mmpack<T, reg>::stride; i++){
        auto yr  = hala::mmload<reg>(&y[0]).real(i);
        auto yrr = std::real(y[i]);
        hassert(testnum(yr, yrr));

        yr  = hala::mmload<reg>(&y[0]).imag(i);
        yrr = std::imag(y[i]);
        hassert(testnum(yr, yrr));
    }

    // test simple copy
    hala_rnd = 3;
    y = make_vector<T>(hala::mmpack<T, reg>::stride);
    hala::mmpack<T, reg> ytemp;
    for(size_t i=0; i<hala::mmpack<T, reg>::stride; i++) ytemp[i] = y[i];
    y_ref = make_vector<T>(hala::mmpack<T, reg>::stride);
    ytemp.put(y_ref.data());
    hassert(testvec(y, y_ref));

}

template<typename T, template<typename, typename> class vclass, hala::regtype reg> void test_type_aligned_allocator(){
    current_test<T> tests(std::string("alloc-") + regname(reg));
    std::vector<T> x_ref, x_test;
    vclass<T, hala::aligned_allocator<T, reg>> x = {1.0, 2.0, 3.0};
    hala::vcopy(x, x_ref);

    auto y = x;
    y.resize(11);
    std::swap(x, y);
    x = y;

    hala::vcopy(x, x_test);
    hassert(testvec(x_test, x_ref));

    {
        auto t = std::move(x);
        x = vclass<T, hala::aligned_allocator<T, reg>>(2);
        x = std::move(t);
    }

    hala::vcopy(x, x_test);
    hassert(testvec(x_test, x_ref));

    std::vector<vclass<T, hala::aligned_allocator<T, reg>>> fake_matrix(11, {17});
    for(auto const& r : fake_matrix){
        size_t pval = reinterpret_cast<size_t>(hala::get_data(r));
        constexpr size_t alignment = hala::mmpack<T, reg>::stride * sizeof(T);
        hassert(pval % alignment == 0); // proper alignment
    }

    hala_rnd = 3;
    hala::aligned_vector<T, reg> vv, ww, rref;
    hala::vcopy( make_vector<T>(hala::mmpack<T, reg>::stride * 3), vv );
    hala::vcopy( make_vector<T>(hala::mmpack<T, reg>::stride * 3), ww );
    hala::vcopy(vv, rref);
    hala::axpy(1, ww, rref);

    for(auto iv = vv.begin(), iw = ww.begin();
        iv != vv.end();
        iv += hala::mmpack<T, reg>::stride, iw += hala::mmpack<T, reg>::stride)
    {
        hala::mmpack<T, reg> mm = &*iv;
        mm += &*iw;
        mm.put(&*iv);
    }
    hassert(testvec(vv, rref));
}

template<template<typename, typename> class vclass, hala::regtype reg> void test_aligned_allocator(){
    test_type_aligned_allocator<float, vclass, reg>();
    test_type_aligned_allocator<double, vclass, reg>();
    test_type_aligned_allocator<std::complex<float>, vclass, reg>();
    test_type_aligned_allocator<std::complex<double>, vclass, reg>();
}

template<typename T> void test_default(){
    current_test<T> tests(std::string("alloc-default"));
    size_t num_entries = 256;
    hala_rnd = 3;
    // get some semi-random numbers
    auto rndx = make_vector<T>(num_entries);
    auto rndy = make_vector<T>(num_entries);

    // define two vectors aligned to the default register type
    hala::aligned_vector<T> x, y;
    // fill the vectors with the random data
    hala::vcopy(rndx, x);
    hala::vcopy(rndy, y);

    // perform operations using iterators
    for(auto ix = hala::mmbegin(x), iy = hala::mmbegin(y);
        ix < hala::mmend(x) && iy <= hala::mmend(y);
        hala::mmadvance(ix, iy))
    {
        auto mmx = hala::mmbind(ix); // tests mmbind(iterator)
        mmx += iy;                   // tests += iterator
        auto mmy = hala::mmload(iy); // tests mmload(iterator)
        mmx *= mmy;                  // tests *= mmpack()
    } // on exit, mmx will sync back with ix

    // perform the same operations using the simple vectors
    for(size_t i=0; i<num_entries; i++)
        rndx[i] = (rndx[i] + rndy[i]) * rndy[i];

    // copy the result to the rndy
    hala::vcopy(x, rndy);
    // compare the result computed with mmpack vs the regular one
    hassert(testvec(rndx, rndy));
}
