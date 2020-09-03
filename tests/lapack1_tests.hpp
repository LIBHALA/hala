/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */
#include "lapack_plu_tests.hpp"

template<typename T>
void test_solvel2(){
    current_test<T> tests("solvel2");
    int N = 20;
    std::vector<T> A(2 * N, 1.0), B(N);

    // line is a * x + b
    T a = hala::get_cast<T>(2.0);
    T b = hala::get_cast<T>(-3.0);

    T x = hala::get_cast<T>(-5.0);
    for(int i=0; i<N; i++){
        A[i] = x;
        B[i] = a * x + b;
        x += hala::get_cast<T>(1.0);
    }
    hala::gels('N', N, 2, 1, A, B);

    hassert(B.size() == (size_t) N);
    B.resize(2);
    hassert(testvec(B, std::vector<T>({a, b})));
}


template<typename T>
void test_svd(){
    current_test<T> tests("svd");
    int M = 4, N = 3;
    std::vector<T> A(M * N, hala::get_cast<T>(0.0));

    A[0 * M + 3] = hala::get_cast<T>(3.0);
    A[1 * M + 1] = hala::get_cast<T>(2.0);
    A[2 * M + 0] = hala::get_cast<T>(4.0);

    std::vector<typename hala::define_standard_precision<T>::value_type> S;
    hala::svd(M, N, A, S);

    hassert(testvec(S, std::vector<typename hala::define_standard_precision<T>::value_type>({4.0, 3.0, 2.0})));

    hala_rnd = -1;
    for(auto &a : A) a = getVal(a);
    auto Aref = A;

    S.resize(0);
    std::vector<T> U;
    std::valarray<T> VH;
    hala::svd(M, N, A, S, U, VH);
    A = Aref;

    for(int i=0; i<N; i++) for(int j=0; j<N; j++) VH[i * N + j] *= S[j];
    hala::gemm('N', 'n', M, N, N, hala::get_cast<T>(1.0), U, VH, hala::get_cast<T>(-1.0), A);

    for(auto &a : Aref) a = hala::get_cast<T>(0.0);
    hassert(testvec(A, Aref, 1.E+3));
}


template<typename T>
void test_dsolve_plu(){
    current_test<T> tests("solve plu");
    int N = 5, num_rhs = 4;
    std::valarray<T> A = {8.0, 4.0, 1.0, 1.0, 0.0, 4.0, 8.0, 4.0, 2.0, 1.0, 2.0, 4.0, 8.0, 3.0, 2.0, 1.0, 2.0, 4.0, 8.0, 4.0, 0.0, 1.0, 2.0, 4.0, 8.0};
    auto Aref = A;

    int x = 1;
    std::vector<T> B(N * num_rhs);
    for(auto &b : B) b = (T) x++;

    std::vector<T> Bref = B, C;

    std::valarray<int> ipiv;
    hala::gesv(N, num_rhs, A, ipiv, B);
    refgemm('N', 'N', N, num_rhs, N, 1.0, Aref, B, 0.0, C);

    hassert(ipiv.size() == (size_t) N);
    hassert(testvec(Bref, C, 1.E+0));
}
template<typename T>
void test_zsolve_plu(){
    current_test<T> tests("cholsolve");
    int N = 3, num_rhs = 2;
    std::vector<T> A, D = {{8.0, 1.0}, {4.0, 0.5}, {2.0, 1.0}, {4.0, -1.0}, {8.0, 0.5}, {4.0, 1.0}, {4.0, -1.0}, {8.0, -0.5}, {4.0, 1.0}};

    std::vector<T> X(N * num_rhs), B; // make up a solution
    int x = 0;
    for(auto &s : X){
        s = T(hala::get_cast<typename T::value_type>(x + 1), hala::get_cast<typename T::value_type>(x + 2));
        x += 3;
    }

    hala::gemm('C', 'N', N, N, N, hala::get_cast<T>(1.0), D, D, hala::get_cast<T>(0.0), A);
    for(int i=0; i<N; i++) A[i * N + i] += hala::get_cast<T>(5.0);
    hala::gemm('N', 'N', N, num_rhs, N, hala::get_cast<T>(1.0), A, X, hala::get_cast<T>(0.0), B);
    std::valarray<int> ipiv;
    hala::gesv(N, num_rhs, A, ipiv, B);

    std::vector<T> Ar;
    for(int i=0; i<N; i++)
        for(int j=i+1; j<N; j++)
            A[i*N + j] = hala::get_cast<T>(0.0);
    hala::gemm('C', 'N', N, N, N, hala::get_cast<T>(1.0), A, A, hala::get_cast<T>(0.0), Ar);

    hassert(testvec(B, X, 1.E+2));
}
template<typename T>
void test_general_qr(){
    current_test<T> tests("general qr");

    int M = 5, N = 3;
    hala_rnd = -2;
    auto A = make_vector<T>(M * N);
    std::vector<T> Amin = A;
    std::vector<T> Aref = A;
    std::vector<T> factor, factor_min;

    hala::geqr(M, N, A, factor);
    hala::geqr(M, N, Amin, factor_min);
    std::vector<T> upper(M * N);
    for(int i=0; i<N; i++)
        for(int j=0; j<i+1; j++)
            upper[i * M + j] = A[i * M + j];

    std::vector<T> solution = upper;
    hala::gemqr('L', 'N', M, N, N, A, factor, solution);
    hassert(testvec(Aref, solution, 1.E+2));

    solution = upper;
    hala::gemqr('L', 'N', M, N, N, Amin, factor_min, solution);
    hassert(testvec(Aref, solution, 1.E+2));

    M = 3; N = 5;
    A = make_vector<T>(M * N);
    Amin = A;
    Aref = A;
    factor = std::vector<T>();
    factor_min = std::vector<T>();
    upper = std::vector<T>(M * N);
    hala::geqr(M, N, A, factor);
    hala::geqr(M, N, Amin, factor_min);

    for(int i=0; i<N; i++)
        for(int j=0; j<std::min(i+1, M); j++)
            upper[i * M + j] = A[i * M + j];

    solution = upper;
    hala::gemqr('L', 'N', M, N, M, A, factor, solution);
    hassert(testvec(Aref, solution, 1.E+2));

    solution = upper;
    hala::gemqr('L', 'N', M, N, M, Amin, factor_min, solution);
    hassert(testvec(Aref, solution, 1.E+2));
}
