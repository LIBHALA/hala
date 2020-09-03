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

template<typename T> void gemm(){
    current_test<T> tests("gemm");
    int M = 3, N = 2, K = 4;
    //std::vector<T> A(M * K), B(K * N), C, Cref;

    hala_rnd = -4;
    std::vector<T> A = make_vector<T>(M * K);
    std::valarray<T> B = make_valarray<T>(K * N);
    std::vector<T> C, Cref;

    refgemm('N', 'N', M, N, K, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);

    hala::gpu_engine engine;
    auto gpuA = engine.load(A);
    auto gpuB = engine.load(B);
    auto gpuC = new_vector(engine, A);
    gemm(engine, 'N', 'N', M, N, K, hala::get_cast<T>(2.0), gpuA, gpuB, hala::get_cast<T>(0.0), gpuC);
    gpuC.unload(C);

    hassert(testvec(C, Cref));

    //engine.set_cublas_device_pntr();
    auto gputwo = engine.vector<T>(1, hala::get_cast<T>(2.0));
    auto gpuzero = engine.vector<T>(1, hala::get_cast<T>(0.0));
    auto gpuC2 = new_vector(engine, A);
    gpuC2.resize(hala::hala_size(M, N));
    {
        hala::gpu_pntr<hala::device_pntr> hold(engine);
        gemm(engine, 'N', 'N', M, N, K, gputwo.data(), gpuA, gpuB, gpuzero.data(), gpuC2);
    }
    C.clear();
    gpuC2.unload(C);
    hassert(testvec(C, Cref));
    //engine.reset_cublas_device_pntr();

    std::vector<T> vectwo(1, hala::get_cast<T>(2.0));
    std::vector<T> veczero(1, hala::get_cast<T>(0.0));
    auto gpuC3 = new_vector(engine, A);
    gpuC3.resize(hala::hala_size(M, N));
    gemm(engine, 'N', 'N', M, N, K, vectwo.data(), gpuA, gpuB, veczero.data(), gpuC3);
    C.clear();
    gpuC3.unload(C);
    hassert(testvec(C, Cref));
}

template<typename T> void syrk(){
    current_test<T> tests("syrk");
    int N = 4, K = 5;
    std::vector<T> C, Cref;

    hala_rnd = -4;
    auto A = make_vector<T>(N * K);

    std::vector<T> At(A.size());
    for(int i=0; i<K; i++)
        for(int j=0; j<N; j++)
            At[j * K + i] = A[i * N + j];

    refgemm('N', 'N', N, N, K, hala::get_cast<T>(2.0), A, At, hala::get_cast<T>(0.0), Cref);

    hala::gpu_engine engine;
    auto gpuA = engine.load(A);
    hala::gpu_vector<T> gpuC;

    syrk(engine, 'U', 'N', N, K, hala::get_cast<T>(2.0), gpuA, hala::get_cast<T>(0.0), gpuC);

    C = gpuC.unload();
    make_triangular('U', 'N', C);
    make_triangular('U', 'N', Cref);
    hassert(testvec(C, Cref));
}
