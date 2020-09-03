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

template<typename T, int, class cengine> void test_gemv(cengine const &engine){
    current_test<T> tests("g(eb)mv");
    hala_rnd = -3;
    int M = 3, N = 2;
    auto A = make_vector<T>(M * N);
    auto x = make_vector<T>(N);
    std::vector<T> y, yref;
    refgemm('N', 'N', M, 1, N, 1.0, A, x, 0.0, yref);
    hala::gemv(mengine 'N', M, N, 1, bind(A), bind(x), 0.0f, bind(y));
    hassert(testvec(y, yref));

    N = 3; M = 4;
    A = make_vector<T>(M * N);
    x = make_vector<T>(N);
    y = make_vector<T>(M);
    yref = y;
    refgemm('T', 'N', M, 1, N, 1.0, A, x, 2.0, yref);
    hala::gemv(mengine 'T', N, M, 1, bind(A), bind(x), 2.0, bind(y));
    hassert(testvec(y, yref));

    hala_rnd = -3;
    M = 4; N = 4;
    A = make_vector<T>(M * N);
    x = make_vector<T>(N);
    y = std::vector<T>();
    yref = std::vector<T>();
    make_banded(1, 2, A);
    refgemm('N', 'N', M, 1, N, 2.0, A, x, 0.0, yref);
    auto Ap = pack_banded(1, 2, A);
    hala::gbmv(mengine 'N', M, N, 1, 2, 2, bind(Ap), bind(x), 0.0f, bind(y));
    hassert(testvec(y, yref));

    refgemm('T', 'N', M, 1, N, 2.0, A, x, 1.0, yref);
    hala::gbmv(mengine 'T', M, N, 1, 2, 2, bind(Ap), bind(x), 1.0f, bind(y));
    hassert(testvec(y, yref));
}

template<typename T, int, class cengine> void test_symv(cengine const &engine){
    current_test<T> tests("s(ybp)mv");
    hala_rnd = -3;
    int N = 4;
    auto A = make_symmetric<T>(N);
    auto x = make_vector<T>(N);
    std::vector<T> y, yref;
    refgemm('N', 'N', N, 1, N, 2.0, A, x, 0.0, yref);
    std::vector<T> As = A;
    make_triangular('U', 'N', As);
    hala::symv(mengine 'U', N, 2, bind(As), bind(x), 0.0f, bind(y));
    hassert(testvec(y, yref));

    y = std::vector<T>(yref.size());
    As = pack_triangular('U', A);
    hala::spmv(mengine 'U', N, 2, bind(As), bind(x), 0.0f, bind(y));
    hassert(testvec(y, yref));

    y = std::vector<T>();
    int k = 1;
    make_banded(k, k, A);
    refgemm('N', 'N', N, 1, N, 2.0, A, x, 0.0, yref);
    As = pack_banded(k, 0, A);
    hala::sbmv(mengine 'L', N, k, 2, bind(As), bind(x), 0.0f, bind(y));
    hassert(testvec(y, yref));

    A = make_hermitian<T>(N);
    y = yref;
    refgemm('N', 'N', N, 1, N, 2.0, A, x, 1.0, yref);
    As = A;
    make_triangular('L', 'N', As);
    hala::hemv(mengine 'L', N, 2, bind(As), N, bind(x), 1, 1.0f, bind(y), 1);
    hassert(testvec(y, yref));

    refgemm('N', 'N', N, 1, N, 2.0, A, x, 0.0, yref);
    y = std::vector<T>(yref.size());
    As = pack_triangular('L', A);
    hala::hpmv(mengine 'L', N, 2, bind(As), bind(x), 1, 0.0f, bind(y), 1);
    hassert(testvec(y, yref));

    y = std::vector<T>();
    k = 2;
    make_banded(k, k, A);
    refgemm('N', 'N', N, 1, N, 2.0, A, x, 0.0, yref);
    As = pack_banded(0, k, A);
    hala::hbmv(mengine 'U', N, k, 2, bind(As), 3, bind(x), 1, 0.0f, bind(y), 1);
    hassert(testvec(y, yref));
}

template<typename T, int, class cengine> void test_trmsv(cengine const &engine){
    current_test<T> tests("t(rp)(ms)v");
    int N = 6;
    hala_rnd = -4;
    auto A = make_vector<T>(N * N);
    make_triangular('L', 'U', A);
    auto ox = make_vector<T>(N);
    std::vector<T> refx, x = ox;
    refgemm('N', 'N', N, 1, N, 1.0, A, ox, 0.0, refx);
    hala::trmv(mengine 'L', 'N', 'U', N, bind(A), bind(x));
    hassert(testvec(refx, x));

    auto Ap = pack_triangular('L', A);
    x = ox;
    hala::tpmv(mengine 'L', 'N', 'U', N, bind(Ap), bind(x));
    hassert(testvec(refx, x));

    hala::trsv(mengine 'L', 'N', 'U', N, bind(A), bind(x));
    hassert(testvec(ox, x));

    x = refx;
    hala::tpsv(mengine 'L', 'N', 'U', N, bind(Ap), bind(x));
    hassert(testvec(ox, x));

    hala_rnd = -4;
    A = make_vector<T>(N * N);
    make_triangular('U', 'U', A);
    ox = make_vector<T>(N);
    refx = std::vector<T>();
    x = ox;
    refgemm('N', 'N', N, 1, N, 1.0, A, ox, 0.0, refx);
    hala::trmv(mengine 'U', 'N', 'U', N, bind(A), bind(x));
    hassert(testvec(refx, x));

    Ap = pack_triangular('U', A);
    x = ox;
    hala::tpmv(mengine 'U', 'N', 'U', N, bind(Ap), bind(x));
    hassert(testvec(refx, x));

    hala::trsv(mengine 'U', 'N', 'U', N, bind(A), bind(x));
    hassert(testvec(ox, x));

    x = refx;
    hala::tpsv(mengine 'U', 'N', 'U', N, bind(Ap), bind(x));
    hassert(testvec(ox, x));
}

template<typename T, int, class cengine> void test_tbmsv(cengine const &engine){
    current_test<T> tests("tb(ms)v");
    int N = 6, k = 2;
    hala_rnd = -4;
    auto A = make_vector<T>(N * N);
    make_triangular('L', 'U', A);
    make_banded(k, k, A);
    auto ox = make_vector<T>(N);
    std::vector<T> refx, x = ox;
    refgemm('N', 'N', N, 1, N, 1.0, A, ox, 0.0, refx);
    auto Ab = pack_banded(k, 0, A);
    hala::tbmv(mengine 'L', 'N', 'U', N, k, bind(Ab), bind(x));
    hassert(testvec(refx, x));

    hala::tbsv(mengine 'L', 'N', 'U', N, k, bind(Ab), bind(x));
    hassert(testvec(ox, x));

    hala_rnd = -4;
    A = make_vector<T>(N * N);
    make_triangular('U', 'U', A);
    make_banded(k, k, A);
    ox = make_vector<T>(N);
    refx = std::vector<T>();
    x = ox;
    refgemm('N', 'N', N, 1, N, 1.0, A, ox, 0.0, refx);
    Ab = pack_banded(0, k, A);
    hala::tbmv(mengine 'U', 'N', 'U', N, k, bind(Ab), bind(x));
    hassert(testvec(refx, x));

    hala::tbsv(mengine 'U', 'N', 'U', N, k, bind(Ab), bind(x));
    hassert(testvec(ox, x));
}

template<typename T, int, class cengine> void test_rank12(cengine const &engine){
    current_test<T> tests("rank(12)");
    int M = 5, N = 4;
    hala_rnd = -4;
    auto A = make_vector<T>(M * N);
    auto x = make_vector<T>(M);
    auto y = make_vector<T>(N);

    // testing the general complex and non-complex cases
    std::vector<T> Aref = A;
    refgemm('N', 'N', M, N, 1, 2.0, x, y, 1.0, Aref);
    hala::geru(mengine M, N, 2.0, bind(x), bind(y), bind(A));
    hassert(testvec(Aref, A));

    std::vector<T> conjy = y;
    for(auto &v : conjy) v = hala::cconj(v);
    refgemm('N', 'N', M, N, 1, 3.0, x, conjy, 1.0, Aref);
    hala::ger(mengine M, N, 3.0, bind(x), bind(y), bind(A));
    hassert(testvec(Aref, A));

    // testing the symmetric packed and non-packed cases
    auto Ao = make_symmetric<T>(N); // original A
    A = Ao;
    Aref = Ao;
    make_triangular('U', 'N', A);
    x = make_vector<T>(N);
    std::vector<T> conjx = x;
    for(auto &v : conjx) v = hala::cconj(v);

    refgemm('N', 'N', N, N, 1, 2.0, x, x, 1.0, Aref);
    make_triangular('U', 'N', Aref);
    hala::syr(mengine 'U', N, 2.0, bind(x), bind(A));
    hassert(testvec(Aref, A));

    A = pack_triangular('U', Ao);
    hala::spr(mengine 'U', N, 2.0, bind(x), bind(A));
    hassert(testvec(pack_triangular('U', Aref), A));

    // testing the symmetric packed and non-packed cases (Hermitian matrices)
    Ao = make_hermitian<T>(N);
    A = Ao;
    make_triangular('L', 'N', A);
    Aref = Ao;
    refgemm('N', 'N', N, N, 1, 3.0, x, conjx, 1.0, Aref);
    make_triangular('L', 'N', Aref);
    hala::her(mengine 'L', N, 3.0, bind(x), 1, bind(A), N);
    hassert(testvec(Aref, A));

    A = pack_triangular('L', Ao);
    hala::hpr(mengine 'L', N, 3.0, bind(x), 1, bind(A));
    hassert(testvec(pack_triangular('L', Aref), A));

    // testing symmetric packed and non-packed rank-2 update
    hala_rnd = -3;
    Ao = make_symmetric<T>(N); // original A
    A = Ao;
    Aref = Ao;
    make_triangular('U', 'N', A);
    x = make_vector<T>(N);
    y = make_vector<T>(N);

    refgemm('N', 'N', N, N, 1, 2.0, x, y, 1.0, Aref);
    refgemm('N', 'N', N, N, 1, 2.0, y, x, 1.0, Aref);
    make_triangular('U', 'N', Aref);
    hala::syr2(mengine 'U', N, 2.0, bind(x), bind(y), bind(A));
    hassert(testvec(Aref, A));

    A = pack_triangular('U', Ao);
    hala::spr2(mengine 'U', N, 2.0, bind(x), bind(y), bind(A));
    hassert(testvec(pack_triangular('U', Aref), A));

    // testing for the Hermitian case
    conjx = x;
    for(auto &v : conjx) v = hala::cconj(v);
    conjy = y;
    for(auto &v : conjy) v = hala::cconj(v);

    Ao = make_hermitian<T>(N);
    A = Ao;
    make_triangular('L', 'N', A);
    Aref = Ao;
    refgemm('N', 'N', N, N, 1, 3.0, x, conjy, 1.0, Aref);
    refgemm('N', 'N', N, N, 1, 3.0, y, conjx, 1.0, Aref);
    make_triangular('L', 'N', Aref);
    hala::her2(mengine 'L', N, 3.0, bind(x), 1, bind(y), 1, bind(A), N);
    hassert(testvec(Aref, A));

    A = pack_triangular('L', Ao);
    hala::hpr2(mengine 'L', N, 3.0, bind(x), 1, bind(y), 1, bind(A));
    hassert(testvec(pack_triangular('L', Aref), A));
}
