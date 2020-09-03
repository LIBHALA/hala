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

template<typename T> void load_unload(){
    current_test<T> tests("cuda_vec");
    hala_rnd = 0;
    auto x = make_valarray<T>(4);

    hala::gpu_engine engine;
    auto gpux = engine.load(x);

    auto y = gpux.unload_valarray();
    hassert(testvec(x, y));

    auto gpux2 = gpux;
    auto xvec = gpux.unload();
    auto yvec = gpux2.unload();
    hassert(xvec.size() == x.size());
    hassert(testvec(xvec, yvec));

    hala::gpu_vector<T> xvec2 = std::move(gpux);
    auto x2 = xvec2.unload();
    hassert(testvec(xvec, x2));

    auto h = make_vector<T>(5);
    auto gpu1 = engine.load(h);
    hala::gpu_vector<T> gpu2;
    gpu2 = std::move(gpu1);
    auto g = gpu2.unload();
    hassert(testvec(g, h));

    hala::gpu_vector<T> gpu3 = gpu2; // copy
    auto f = gpu3.unload();
    hassert(testvec(f, h));

    std::vector<T> ones(43, hala::get_cast<T>(1.0));
    hala::gpu_vector<T> gpu_ones(43, 0);
    gpu_ones.fill(hala::get_cast<T>(1.0));
    auto filled = gpu_ones.unload();
    hassert(testvec(filled, ones));

    // wrapper dynamics
    auto wx = make_vector<T>(20);
    auto wy = make_vector<T>(20);
    auto gpubasex = engine.load(wx);
    auto gpubasey = engine.load(wy);

    const auto wrappedx = hala::wrap_gpu_array((T const *) gpubasex.data(), gpubasex.size());
    auto wrappedy = hala::wrap_gpu_array(gpubasey.data(), gpubasey.size());
    hala::axpy(engine, hala::get_cast<T>(2.0), wrappedx, wrappedy);

    auto compy = wrappedy.unload();
    gpubasey = engine.load(wy); // reset wrappedy for the next test
    wrappedy = hala::wrap_gpu_array(gpubasey.data(), gpubasey.size());
    for(size_t i=0; i<wx.size(); i++) wy[i] += hala::get_cast<T>(2.0) * wx[i];

    hassert(testvec(compy, wy));

    hala::axpy(hala::gpu_engine(engine), hala::get_cast<T>(2.0), wrappedx, wrappedy);
    compy = wrappedy.unload();
    hassert(testvec(compy, wy));
}
