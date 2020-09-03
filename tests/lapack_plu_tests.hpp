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

template<typename T, int, class cengine> void test_potrfs(cengine const &engine){
    current_test<T> tests("potrf/potrs");
    int N = 3, K = 5, nrhs = 2;
    hala_rnd = -5;
    auto D = make_vector<T>(N * K);
    std::vector<T> Aref;
    hala::gemm('C', 'N', N, N, K, 1, D, D, 0, Aref); // A = D' * D which is an spd matrix
    for(int i=0; i<N; i++) Aref[i * N + i] += hala::get_cast<T>(2.0);

    hala_rnd = -1;
    auto Xref = make_vector<T>(N * nrhs);
    std::vector<T> B;
    hala::gemm('N', 'N', N, nrhs, N, 1, Aref, Xref, 0, B); // B = A * X

    std::vector<T> A = Aref;
    make_triangular('U', 'N', A);
    hala::potrf(mengine 'U', N, bind(A));

    std::vector<T> Aapprox;
    hala::gemm('C', 'N', N, N, N, 1, A, A, 0, Aapprox);
    hassert(testvec(Aref, Aapprox, 1.E+3));

    hala::potrs(mengine 'U', N, nrhs, bind(A), bind(B));
    hassert(testvec(B, Xref, 1.E+4))

    A = Aref;
    make_triangular('L', 'N', A);
    hala::potrf(mengine 'L', N, bind(A), N);
    hala::gemm('N', 'C', N, N, N, 1, A, A, 0, Aapprox);
    hassert(testvec(Aref, Aapprox, 1.E+3));

    hala::gemm('N', 'N', N, nrhs, N, 1, Aref, Xref, 0, B);
    hala::potrs(mengine 'L', N, nrhs, bind(A), N, bind(B), N);
    hassert(testvec(B, Xref, 1.E+4));
}

template<typename T, int, class cengine> void test_getrfs(cengine const &engine){
    current_test<T> tests("getrf/getrs");
    int N = 3, nrhs = 2;
    hala_rnd = -5;
    auto Aref = make_vector<T>(N * N);
    for(int i=0; i<N; i++) Aref[i * N + i] += hala::get_cast<T>(10.0);

    hala_rnd = -1;
    auto Xref = make_vector<T>(N * nrhs);
    std::vector<T> B;
    hala::gemm('N', 'N', N, nrhs, N, 1, Aref, Xref, 0, B); // B = A * X

    std::vector<T> A = Aref;
    std::vector<int> ipiv;
    hala::getrf(mengine N, N, bind(A), bind(ipiv));

    hala::getrs(mengine 'N', N, nrhs, bind(A), bind(ipiv), bind(B));
    hassert(testvec(B, Xref, 1.E+2));

    hala::gemm('T', 'N', N, nrhs, N, 1, Aref, Xref, 0, B); // B = A^T * X
    hala::getrs(mengine 'T', N, nrhs, bind(A), bind(ipiv), bind(B));
    hassert(testvec(B, Xref, 1.E+2));
}
