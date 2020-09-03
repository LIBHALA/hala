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

template<typename T, int, class cengine> void test_trmsm(cengine const &engine){
    current_test<T> tests("tr(ms)m");
    int N = 6, rhs = 2;
    hala_rnd = -2;
    auto A = make_vector<T>(N * N);
    make_triangular('L', 'U', A);
    auto oB = make_vector<T>(N * rhs);
    std::vector<T> refB, B = oB;
    refgemm('N', 'N', N, rhs, N, 2.0, A, oB, 0.0, refB);
    hala::trmm(mengine 'L', 'L', 'N', 'U', N, rhs, 2.0, bind(A), N, bind(B), N);
    hassert(testvec(refB, B));

    hala::trsm(mengine 'L', 'L', 'N', 'U', N, rhs, 0.5, bind(A), bind(B));
    hassert(testvec(oB, B));

    hala_rnd = -4;
    A = make_vector<T>(N * N);
    make_triangular('U', 'U', A);
    oB = make_vector<T>(N * rhs);
    refB = std::vector<T>();
    B = oB;
    refgemm('T', 'N', N, rhs, N, 2.0, A, oB, 0.0, refB);
    hala::trmm(mengine 'L', 'U', 'T', 'U', N, rhs, 2.0, bind(A), bind(B));
    hassert(testvec(refB, B));

    hala::trsm(mengine 'L', 'U', 'T', 'U', N, rhs, 0.5, bind(A), N, bind(B), N);
    hassert(testvec(oB, B));
}

template<typename T, int, class cengine> void test_syrk(cengine const &engine){
    current_test<T> tests("syr(2)k");
    int N = 6, K = 2;
    hala_rnd = -2;
    auto A = make_vector<T>(N * K);
    std::vector<T> refC, C;

    refgemm('N', 'T', N, N, K, 2.0, A, A, 0.0, refC);
    make_triangular('L', 'N', refC);
    hala::syrk(mengine 'L', 'N', N, K, 2.0, bind(A), 0.0, bind(C));
    make_triangular('L', 'N', C); // sometimes CUDA doesn't initialize the top to zero
    hassert(testvec(refC, C));

    auto B = make_vector<T>(N * K);
    refC = std::vector<T>();
    C = std::vector<T>();
    refgemm('T', 'N', K, K, N, 2.0, A, B, 0.0, refC);
    refgemm('T', 'N', K, K, N, 2.0, B, A, 1.0, refC);
    make_triangular('L', 'N', refC);
    hala::syr2k(mengine 'L', 'T', K, N, 2.0, bind(A), bind(B), 0.0, bind(C));
    make_triangular('L', 'N', C);
    hassert(testvec(refC, C));

    A = make_hermitian<T>(N);
    refgemm('N', 'N', N, N, N, 3.0, A, A, 0.0, refC);
    make_triangular('U', 'N', refC);
    hala::herk(mengine 'U', 'C', N, N, 3.0, bind(A), N, 0.0, bind(C), N);
    make_triangular('U', 'N', C); // remove old data from the previous run
    hassert(testvec(refC, C));

    refC = std::vector<T>();
    refgemm('N', 'N', N, N, N, 3.0, A, A, 0.0, refC);
    refgemm('N', 'N', N, N, N, 3.0, A, A, 1.0, refC);
    make_triangular('U', 'N', refC);
    hala::her2k(mengine 'U', 'C', N, N, 3.0, bind(A), bind(A), 0.0, bind(C));
    make_triangular('U', 'N', C);
    hassert(testvec(refC, C));
}

template<typename T, int, class cengine> void test_symm(cengine const &engine){
    current_test<T> tests("symm");
    int N = 6, K = 2;
    hala_rnd = -2;
    auto A = make_symmetric<T>(N);
    auto B = make_vector<T>(N * K);
    std::vector<T> refC, C;

    refgemm('N', 'N', N, K, N, 3.0, A, B, 0.0, refC);
    make_triangular('U', 'N', A);
    hala::symm(mengine 'L', 'U', N, K, 3.0, bind(A), bind(B), 0.0, bind(C));
    hassert(testvec(refC, C));

    K = 3;
    hala_rnd = 1;
    A = make_hermitian<T>(N);
    B = make_vector<T>(N * K);
    refgemm('N', 'N', N, K, N, 3.0, A, B, 0.0, refC);
    make_triangular('L', 'N', A);
    hala::hemm(mengine 'L', 'L', N, K, 3.0, bind(A), bind(B), 0.0, bind(C));
    hassert(testvec(refC, C));

    if (hala::is_complex<T>::value){
        hala::symm(mengine 'L', 'L', N, K, 3.0, bind(A), bind(B), 0.0, bind(C));
        hassert(!testvec(refC, C)); // non-conj should not work for complex Hermitian matrices
    }
}

template<typename T, int, class cengine> void test_gemm(cengine const &engine){
    current_test<T> tests("gemm");
    int M = 5, N = 6, K = 7;

    hala_rnd = -10;
    auto A = make_valarray<T>(M * K);
    auto B = make_vector<T>(K * N);
    std::vector<T> C(M * N), refC;
    refgemm('N', 'N', M, N, K, -1.0, A, B, 0.0, refC);
    auto wC = hala::wrap_array(C.data(), C.size());
    hala::gemm(mengine 'N', 'N', M, N, K, -1.0, bind(A), bind(B), 0.0, bind(wC));
    hassert(testvec(refC, C));

    refC = std::vector<T>();
    C = std::vector<T>();
    refgemm('T', 'N', M, N, K, 2.0, A, B, 0.0, refC);
    hala::gemm(mengine 'T', 'N', M, N, K, 2.0, bind(A), bind(B), 0.0, bind(C));
    hassert(testvec(refC, C));
}
