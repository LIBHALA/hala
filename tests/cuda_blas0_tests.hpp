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


template<typename T> void transpose(){
    current_test<T> tests("transpose");
    int M = 3, N = 4, lda = M + 1;
    hala_rnd = 0;
    std::valarray<T> A(lda * N);
    for(auto &v : A) v = getVal(v);

    std::vector<T> ref(M * N);
    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++)
            ref[i * N + j] = A[j * lda + i];

    hala::gpu_engine engine;
    auto gpuA = engine.load(A);

    hala::gpu_vector<T> gpuAt;
    transpose(engine, M, N, gpuA, gpuAt, lda);
    std::vector<T> res = gpuAt.unload();
    hassert(testvec(res, ref));
}

template<typename T> void test_geam(){
    current_test<T> tests("geam");
    hala::gpu_engine engine;
    size_t M = 4, N = 3, ldc = 5;
    hala_rnd = -4;
    auto A = make_vector<T>(M * N);
    auto B = make_vector<T>(M * N);
    std::vector<T> C(ldc * N), Cref(ldc * N);

    for(size_t j=0; j<N; j++)
        for(size_t i=0; i<M; i++)
            Cref[j*ldc + i] = hala::get_cast<T>(2.0) * A[j*M + i] + B[j*M + i];

    auto gA = engine.load(A);
    auto gB = engine.load(B);
    auto gC = engine.load(C);
    hala::geam(engine, 'N', 'N', M, N, 2.0, gA, M, 1, gB, M, gC, ldc);
    gC.unload(C);
    hassert(testvec(C, Cref));

    Cref = std::vector<T>(M * N);
    C = std::vector<T>();
    for(size_t j=0; j<N; j++)
        for(size_t i=0; i<M; i++)
            Cref[j*M + i] = hala::get_cast<T>(2.0) * A[j*M + i] + hala::get_cast<T>(3.0) * B[j*M + i];

    gC = hala::gpu_vector<T>(engine.device());
    hala::geam(engine, 'N', 'N', M, N, 2.0, gA, 3, gB, gC);
    gC.unload(C);
    hassert(testvec(C, Cref));

    hala::mixed_engine e(engine);
    C = std::vector<T>();
    hala::geam(e, 'N', 'N', M, N, 2.0, A, 3, B, C);
    hassert(testvec(C, Cref));
}

template<typename T> void test_dgmm(){
    current_test<T> tests("dgmm");
    hala::gpu_engine engine;
    size_t M = 4, N = 3;
    hala_rnd = -4;
    auto A = make_vector<T>(M * N);
    auto x = make_vector<T>(M);
    std::vector<T> C, Cref = A;

    for(size_t i=0; i<N; i++)
        for(size_t j=0; j<M; j++)
            Cref[i*M + j] *= x[j];

    hala::mixed_engine e(engine);
    hala::dgmm(e, 'L', M, N, A, x, C);
    hassert(testvec(C, Cref));
}

template<typename T> void test_trtp(){
    current_test<T> tests("trtp");
    hala::gpu_engine engine;
    hala::mixed_engine e(engine);
    size_t N = 6;
    hala_rnd = -4;
    auto A = make_vector<T>(N * N);
    make_triangular('U', 'N', A);
    std::vector<T> Ares, AP, refAP = pack_triangular('U', A);
    hala::tr2tp(e, 'U', N, A, AP);
    hassert(testvec(refAP, AP));

    hala::tp2tr(e, 'U', N, AP, Ares);
    make_triangular('U', 'N', Ares);
    hassert(testvec(A, Ares));
}
