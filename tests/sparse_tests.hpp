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

template<typename T, int, class cengine> void test_batch_axpy(cengine const &engine){
    current_test<T> tests("batch_axpy");
    int N = 11, batch_size = 4;
    hala_rnd = -5;
    auto x = make_vector<T>(batch_size * N);
    auto y = make_vector<T>(batch_size * N);
    auto a = make_vector<T>(batch_size);
    T c = hala::get_cast<T>(3.0);

    auto yref = y;
    for(int b=0; b<batch_size; b++)
        for(int i=0; i<N; i++) yref[b*N + i] += c * a[b] * x[b*N + i];

    hala::batch_axpy(mengine batch_size, c, bind(a), bind(x), bind(y));
    hassert(testvec(y, yref));
}

template<typename T, int, class cengine> void test_batch_dot(cengine const &engine){
    current_test<T> tests("batch_dot");
    int N = 10, batch_size = 4;
    hala_rnd = 0;
    auto x = make_vector<T>(batch_size * N);
    auto y = make_vector<T>(batch_size * N);
    std::vector<T> result;

    std::vector<T> ref_result(batch_size, hala::get_cast<T>(0.0));
    for(int b=0; b<batch_size; b++)
        for(int i=0; i<N; i++) ref_result[b] += x[b*N + i] * y[b*N + i];

    hala::batch_dotu(mengine batch_size, bind(x), bind(y), bind(result));
    hassert(testvec(result, ref_result));

    for(int b=0; b<batch_size; b++){
        ref_result[b] = hala::get_cast<T>(0.0);
        for(int i=0; i<N; i++) ref_result[b] += hala::cconj( x[b*N + i] ) * y[b*N + i];
    }

    hala::batch_dot(mengine batch_size, bind(x), bind(y), bind(result));
    hassert(testvec(result, ref_result));
}

template<typename T, int, class cengine>
void test_batch_max_norm(cengine const &engine){
    current_test<T> tests("batch_norm");
    int N = 5, batch_size = 4;
    hala_rnd = -3;
    auto x = make_vector<T>(batch_size * N);

    using precision_type = hala::get_precision_type<std::vector<T>>;

    precision_type ref_norm = 0.0;
    for(int i=0; i<batch_size; i++)
        ref_norm = std::max(ref_norm, hala::norm2(N, &x[hala::hala_size(i, N)], 1));

    auto nrm = hala::batch_max_norm2(mengine batch_size, bind(x));
    hassert(testnum(nrm, ref_norm, 3.0)); // error is high in MKL, OK in OpenBLAS
}

template<typename T, int, class cengine>
void test_batch_scale(cengine const &engine){
    current_test<T> tests("batch_scale");
    int N = 5, batch_size = 4;
    hala_rnd = -3;
    auto x = make_vector<T>(batch_size * N);
    auto a = make_vector<T>(batch_size);

    std::vector<T> xref = x;
    for(int i=0; i<batch_size; i++)
        for(int j=0; j<N; j++)
            xref[i*N + j] *= a[i];

    hala::batch_scal(mengine batch_size, bind(a), bind(x));
    hassert(testvec(x, xref));
}

template<typename T, int, class cengine>
void test_vdivide(cengine const &engine){
    current_test<T> tests("vdivide");
    size_t N = 5;
    hala_rnd = 1;
    auto x = make_vector<T>(N);
    auto y = make_vector<T>(N);
    std::vector<T> z, zref(N);

    for(size_t i=0; i<N; i++)
        zref[i] = x[i] / y[i];

    hala::vdivide(mengine bind(x), bind(y), bind(z));
    hassert(testvec(z, zref));
}

template<typename T, int, class cengine>
void test_sparse_matrix(cengine const &engine){
    current_test<T> tests("sparse_matrix");
    hala_rnd = -1;
    int N = 5;
    std::vector<int> pntr = {0, 2, 5, 8, 11, 13};
    std::valarray<int> indx = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    std::vector<T> vals = {2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0};
    std::vector<T> x = make_vector<T>(N);

    auto A = hala::dense_convert(N, N, pntr, indx, vals);
    std::vector<T> y, yref;
    auto sp = hala::make_sparse_matrix(engine, N, pntr, indx, vals);

    refgemm('N', 'N', N, 1, N, hala::get_cast<T>(2.0), A, x, hala::get_cast<T>(0.0), yref);
    sp.gemv('N', 2.0, x, 0.0, y);
    hassert(testvec(y, yref));

    #ifdef HALA_ENABLE_GPU
    refgemm('T', 'N', N, 1, N, hala::get_cast<T>(2.0), A, x, hala::get_cast<T>(0.0), yref);
    if (std::is_same<cengine, hala::cpu_engine>::value){
        std::vector<T> work(sp.gemv_buffer_size('T', 2.0, x, 0.0, y) / sizeof(T) + 1);
        sp.gemv('T', 2.0, x, 0.0, y, work);
        hassert(testvec(y, yref));
    }else{
        hala::gpu_vector<T> work(sp.gemv_buffer_size('T', 2.0, x, 0.0, y) / sizeof(T) + 1, get_device(engine));
        sp.gemv('T', 2.0, x, 0.0, y, work);
        hassert(testvec(y, yref));
    }
    #endif

    refgemm('C', 'N', N, 1, N, hala::get_cast<T>(2.0), A, x, hala::get_cast<T>(0.0), yref);
    sp.gemv('C', 2, x, 0, y);
    hassert(testvec(y, yref));

    int M = 5, K = 6;
    N = 3;
    pntr = {0, 3, 7, 9, 13, 16};
    indx = {0, 2, 5, 0, 1, 2, 5, 2, 3,  0, 2, 3, 5, 1, 3, 4};
    vals = {2.0, 4.0, 5.0, 1.0, 3.0, 1.0, 5.0, 4.0, 1.0, 3.0, 4.0, 4.0, 1.0, 5.0, 1.0, 3.0};
    A = hala::dense_convert(M, K, pntr, indx, vals);

    hala_rnd = -1;
    std::vector<T> C, Cref, B = make_vector<T>(K * N);
    sp = hala::make_sparse_matrix(engine, K, pntr, indx, vals);

    refgemm('N', 'N', M, N, K, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);
    sp.gemm('N', 'N', K, N, 2.0, B, K, 0.0, C, M);
    hassert(testvec(C, Cref));
    return;

    Cref = C = std::vector<T>();
    B = make_vector<T>(M * N);
    refgemm('T', 'N', K, N, M, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);
    hala::sparse_gemm(mengine 'T', 'N', K, N, M, 2.0, bind(pntr), bind(indx), bind(vals), bind(B), M, 0.0, bind(C), K);
    hassert(testvec(C, Cref));

    refgemm('C', 'N', K, N, M, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);
    hala::sparse_gemm(mengine 'C', 'N', K, N, M, 2.0, bind(pntr), bind(indx), bind(vals), bind(B), 0.0, bind(C));
    hassert(testvec(C, Cref));
}

template<typename T, int, class cengine>
void test_sparse_gemv(cengine const &engine){
    current_test<T> tests("sparse_gemv");
    hala_rnd = -1;
    int N = 5;
    std::vector<int> pntr = {0, 2, 5, 8, 11, 13};
    std::valarray<int> indx = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    std::vector<T> vals = {2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0};
    std::vector<T> x = make_vector<T>(N);

    auto wB = hala::wrap_array(x.data(), x.size());

    auto A = hala::dense_convert(N, N, pntr, indx, vals);
    std::vector<T> y, yref;

    refgemm('N', 'N', N, 1, N, hala::get_cast<T>(2.0), A, x, hala::get_cast<T>(0.0), yref);
    hala::sparse_gemv(mengine 'N', 5, 5, hala::get_cast<T>(2.0), bind(pntr), bind(indx), bind(vals), bind(x), hala::get_cast<T>(0.0), bind(y));
    hassert(testvec(y, yref));

    refgemm('T', 'N', N, 1, N, hala::get_cast<T>(2.0), A, x, hala::get_cast<T>(0.0), yref);
    hala::sparse_gemv(mengine 'T', 5, 5, hala::get_cast<T>(2.0), bind(pntr), bind(indx), bind(vals), bind(x), hala::get_cast<T>(0.0), bind(y));
    hassert(testvec(y, yref));

    refgemm('C', 'N', N, 1, N, hala::get_cast<T>(2.0), A, x, hala::get_cast<T>(0.0), yref);
    hala::sparse_gemv(mengine 'C', 5, 5, hala::get_cast<T>(2.0), bind(pntr), bind(indx), bind(vals), bind(x), hala::get_cast<T>(0.0), bind(y));
    hassert(testvec(y, yref));
}

template<typename T, int, class cengine>
void test_sparse_gemm(cengine const &engine){
    current_test<T> tests("sparse_gemm");
    int M = 5, K = 6, N = 3;
    std::vector<int> pntr = {0, 3, 7, 9, 13, 16};
    std::valarray<int> indx = {0, 2, 5, 0, 1, 2, 5, 2, 3,  0, 2, 3, 5, 1, 3, 4};
    std::vector<T> vals = {2.0, 4.0, 5.0, 1.0, 3.0, 1.0, 5.0, 4.0, 1.0, 3.0, 4.0, 4.0, 1.0, 5.0, 1.0, 3.0};
    std::vector<T> A = {2.0, 1.0, 0.0, 3.0, 0.0,
                        0.0, 3.0, 0.0, 0.0, 5.0,
                        4.0, 1.0, 4.0, 4.0, 0.0,
                        0.0, 0.0, 1.0, 4.0, 1.0,
                        0.0, 0.0, 0.0, 0.0, 3.0,
                        5.0, 5.0, 0.0, 1.0, 0.0};

    hala_rnd = -1;
    std::vector<T> C, Cref, B = make_vector<T>(K * N);

    refgemm('N', 'N', M, N, K, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);
    hala::sparse_gemm(mengine 'N', 'N', M, N, K, 2.0, bind(pntr), bind(indx), bind(vals), bind(B), K, 0.0, bind(C), M);
    hassert(testvec(C, Cref));

    Cref = C = std::vector<T>();
    B = make_vector<T>(M * N);
    refgemm('T', 'N', K, N, M, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);
    hala::sparse_gemm(mengine 'T', 'N', K, N, M, 2.0, bind(pntr), bind(indx), bind(vals), bind(B), M, 0.0, bind(C), K);
    hassert(testvec(C, Cref));

    refgemm('C', 'N', K, N, M, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);
    hala::sparse_gemm(mengine 'C', 'N', K, N, M, 2.0, bind(pntr), bind(indx), bind(vals), bind(B), 0.0, bind(C));
    hassert(testvec(C, Cref));

    hala_rnd = -1;
    B = make_vector<T>(K * N);
    Cref = C = std::vector<T>();
    refgemm('N', 'T', M, N, K, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);
    hala::sparse_gemm(mengine 'N', 'T', M, N, K, 2.0, bind(pntr), bind(indx), bind(vals), bind(B), 0.0, bind(C));
    hassert(testvec(C, Cref));

    Cref = C = std::vector<T>();
    B = make_vector<T>(M * N);
    refgemm('T', 'T', K, N, M, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);
    hala::sparse_gemm(mengine 'T', 'T', K, N, M, 2.0, bind(pntr), bind(indx), bind(vals), bind(B), N, 0.0, bind(C), K);
    hassert(testvec(C, Cref));

    refgemm('C', 'T', K, N, M, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);
    hala::sparse_gemm(mengine 'C', 'T', K, N, M, 2.0, bind(pntr), bind(indx), bind(vals), bind(B), 0.0, bind(C));
    hassert(testvec(C, Cref));

    hala_rnd = -3;
    B = make_vector<T>(K * N);
    Cref = C = std::vector<T>();
    refgemm('N', 'C', M, N, K, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);
    hala::sparse_gemm(mengine 'N', 'C', M, N, K, 2.0, bind(pntr), bind(indx), bind(vals), bind(B), N, 0.0, bind(C), M);
    hassert(testvec(C, Cref));

    Cref = C = std::vector<T>();
    B = make_vector<T>(M * N);
    refgemm('T', 'C', K, N, M, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);
    hala::sparse_gemm(mengine 'T', 'C', K, N, M, 2.0, bind(pntr), bind(indx), bind(vals), bind(B), N, 0.0, bind(C), K);
    hassert(testvec(C, Cref));

    refgemm('C', 'C', K, N, M, hala::get_cast<T>(2.0), A, B, hala::get_cast<T>(0.0), Cref);
    hala::sparse_gemm(mengine 'C', 'C', K, N, M, 2.0, bind(pntr), bind(indx), bind(vals), bind(B), N, 0.0, bind(C), K);
    hassert(testvec(C, Cref));
}

template<typename T, int, class cengine>
void test_sparse_trsv(cengine const &engine){
    current_test<T> tests("sparse_trsv");
    int M = 3;
    std::vector<int> pntr = {0, 1, 3, 5};
    std::vector<int> indx = {0, 0, 1, 0, 2};
    std::vector<T> vals = {1.0, 4.0, 2.0, 5.0, 3.0};
    std::vector<T> A = {1.0, 4.0, 5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0};
    std::vector<T> B, X;

    hala_rnd = -4;
    auto Xref = make_vector<T>(M);
    refgemm('N', 'N', M, 1, M, hala::get_cast<T>(2.0), A, Xref, hala::get_cast<T>(0.0), B);

    auto tri = make_triangular_matrix(mengine 'L', 'N', bind(pntr), bind(indx), bind(vals));

    hala::sparse_trsv('N', tri, hala::get_cast<T>(0.5), B, X);
    hassert(testvec(X, Xref));

    refgemm('T', 'N', M, 1, M, hala::get_cast<T>(2.0), A, Xref, hala::get_cast<T>(0.0), B);
    X = std::vector<T>();
    hala::sparse_trsv('T', tri, hala::get_cast<T>(0.5), B, X);
    hassert(testvec(X, Xref));

    refgemm('C', 'N', M, 1, M, hala::get_cast<T>(2.0), A, Xref, hala::get_cast<T>(0.0), B);
    X = std::vector<T>();
    hala::sparse_trsv('C', tri, hala::get_cast<T>(0.5), B, X);
    hassert(testvec(X, Xref));

    // test the unit diagonal case
    std::vector<T> Au = {1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    refgemm('N', 'N', M, 1, M, hala::get_cast<T>(2.0), Au, Xref, hala::get_cast<T>(0.0), B);
    vals = {2.0, 1.0, 3.0, 2.0, 4.0};

    tri = make_triangular_matrix(mengine 'L', 'U', bind(pntr), bind(indx), bind(vals));
    hala::sparse_trsv('N', tri, hala::get_cast<T>(0.5), B, X);
    hassert(testvec(X, Xref));

    pntr = {0, 3, 4, 5};
    indx = {0, 1, 2, 1, 2};
    vals = {1.0, 4.0, 5.0, 2.0, 3.0};

    refgemm('T', 'N', M, 1, M, hala::get_cast<T>(2.0), A, Xref, hala::get_cast<T>(0.0), B);
    tri = make_triangular_matrix(mengine 'U', 'N', bind(pntr), bind(indx), bind(vals));
    X = std::vector<T>();
    hala::sparse_trsv('N', tri, hala::get_cast<T>(0.5), B, X);
    hassert(testvec(X, Xref));

    refgemm('N', 'N', M, 1, M, hala::get_cast<T>(2.0), A, Xref, hala::get_cast<T>(0.0), B);
    X = std::vector<T>();
    hala::sparse_trsv('T', tri, hala::get_cast<T>(0.5), B, X);
    hassert(testvec(X, Xref));
}

template<typename T, int, class cengine>
void test_sparse_trsm(cengine const &engine){
    current_test<T> tests("sparse_trsm");
    int M = 3, nrhs = 6;
    std::vector<int> pntr = {0, 1, 3, 5};
    std::vector<int> indx = {0, 0, 1, 0, 2};
    std::vector<T> vals = {1.0, 4.0, 2.0, 5.0, 3.0};
    std::vector<T> A = {1.0, 4.0, 5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0};
    std::vector<T> X;

    hala_rnd = -4;
    auto Xref = make_vector<T>(M * nrhs);
    refgemm('N', 'N', M, nrhs, M, hala::get_cast<T>(2.0), A, Xref, hala::get_cast<T>(0.0), X);

    auto tri = make_triangular_matrix(mengine 'L', 'N', bind(pntr), bind(indx), bind(vals));

    hala::sparse_trsm('N', 'N', nrhs, tri, hala::get_cast<T>(0.5), X);
    hassert(testvec(X, Xref));

    refgemm('T', 'N', M, nrhs, M, hala::get_cast<T>(2.0), A, Xref, hala::get_cast<T>(0.0), X);
    hala::sparse_trsm('T', 'N', nrhs, tri, hala::get_cast<T>(0.5), X);
    hassert(testvec(X, Xref));

    refgemm('N', 'T', nrhs, M, M, hala::get_cast<T>(2.0), Xref, A, hala::get_cast<T>(0.0), X);
    hala::sparse_trsm('N', 'T', nrhs, tri, hala::get_cast<T>(0.5), X);
    hassert(testvec(X, Xref));

#ifdef HALA_ENABLE_ROCM
    if (not std::is_same<cengine, hala::mixed_engine>::value){
#endif
    refgemm('C', 'N', M, nrhs, M, hala::get_cast<T>(2.0), A, Xref, hala::get_cast<T>(0.0), X);
    hala::sparse_trsm('C', 'N', nrhs, tri, hala::get_cast<T>(0.5), X);
    hassert(testvec(X, Xref));

    refgemm('N', 'C', nrhs, M, M, hala::get_cast<T>(2.0), Xref, A, hala::get_cast<T>(0.0), X);
    hala::sparse_trsm('N', 'C', nrhs, tri, hala::get_cast<T>(0.5), X);
    hassert(testvec(X, Xref));
#ifdef HALA_ENABLE_ROCM
    }
#endif

    pntr = {0, 3, 4, 5};
    indx = {0, 1, 2, 1, 2};
    vals = {1.0, 4.0, 5.0, 2.0, 3.0};

    tri = make_triangular_matrix(mengine 'U', 'N', bind(pntr), bind(indx), bind(vals));
    refgemm('T', 'N', M, nrhs, M, hala::get_cast<T>(2.0), A, Xref, hala::get_cast<T>(0.0), X);
    hala::sparse_trsm('N', 'N', nrhs, tri, hala::get_cast<T>(0.5), X);
    hassert(testvec(X, Xref));

    refgemm('N', 'N', M, nrhs, M, hala::get_cast<T>(2.0), A, Xref, hala::get_cast<T>(0.0), X);
    hala::sparse_trsm('T', 'N', nrhs, tri, hala::get_cast<T>(0.5), X);
    hassert(testvec(X, Xref));
}

template<typename T> void test_split_matrix(){
    current_test<T> tests("split");
    std::valarray<int> pntr = {0, 2, 5, 8, 11, 13, 17};
    std::vector<int> indx = {0, 3, 0, 1, 4, 2, 3, 4, 0, 3, 4, 2, 4, 0, 1, 3, 5};
    T a = hala::get_cast<T>(1.0), d = hala::get_cast<T>(2.0), z = hala::get_cast<T>(0.0);
    std::vector<T>   vals = {d, a, a, d, a, d, a, a, a, d, a, a, d, a, a, a, d};

    std::valarray<int> diag, diag_ref = {0, 3, 5, 9, 12, 16};
    hala::get_diagonal_index(pntr, indx, diag);

    hassert(testints(diag, diag_ref)); //cout << "diag" << endl;

    std::vector<int> up, ui, lp, li;
    std::vector<int> up_ref = {0, 2, 4, 7, 9, 10, 11};
    std::vector<int> ui_ref = {0, 3, 1, 4, 2, 3, 4, 3, 4, 4, 5};
    std::vector<int> lp_ref = {0, 1, 3, 4, 6, 8, 12};
    std::vector<int> li_ref = {0, 0, 1, 2, 0, 3, 2, 4, 0, 1, 3, 5};

    hala::split_pattern(pntr, indx, diag, up, ui, lp, li);

    //cout << up.size() << "  " << up_ref.size() << endl;
    hassert(testints(up, up_ref)); //cout << "up" << endl;
    hassert(testints(lp, lp_ref)); //cout << "lp" << endl;
    hassert(testints(ui, ui_ref)); //cout << "ui" << endl;
    hassert(testints(li, li_ref)); //cout << "li" << endl;

    std::valarray<T> uv, lv;
    std::valarray<T> uv_ref = {d, a, d, a, d, a, a, d, a, d, d};
    std::valarray<T> lv_ref = {z, a, z, z, a, z, a, z, a, a, a, z};

    hala::split_values('U', 'N', pntr, vals, up, lp, uv, lv);

    hassert(testvec(uv, uv_ref));
    hassert(testvec(lv, lv_ref));

    up = std::vector<int>();
    ui = std::vector<int>();
    lp = std::vector<int>();
    li = std::vector<int>();
    uv = std::valarray<T>();
    lv = std::valarray<T>();

    hala::split_matrix('U', 'N', pntr, indx, vals, up, ui, uv, lp, li, lv);

    hassert(testints(up, up_ref)); //cout << "up" << endl;
    hassert(testints(lp, lp_ref)); //cout << "lp" << endl;
    hassert(testints(ui, ui_ref)); //cout << "ui" << endl;
    hassert(testints(li, li_ref)); //cout << "li" << endl;

    hassert(testvec(uv, uv_ref));
    hassert(testvec(lv, lv_ref));

    if (verbose) report<T>("split");
}

template<typename T> void test_ilu(){
    current_test<T> tests("ilu");
    std::vector<int> pntr = {0, 3, 6, 9};
    std::vector<int> indx = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<T> vals = {3.0, 2.0, 1.0, 2.0, 4.0, 3.0, 2.0, 1.0, 6.0};

    std::vector <int> diag_ref = {0, 4, 8}, diag;
    hala::get_diagonal_index(pntr, indx, diag);
    hassert(testvec(diag, diag_ref));

    std::vector<T> ilu, ilu_ref = {3.0, 2.0, 1.0, 2.0/3.0, 8.0/3.0, 7.0/3.0, 2.0/3.0, -0.125, 5.625};
    hala::factorize_ilu(pntr, indx, vals, diag, ilu);
    hassert(testvec(ilu, ilu_ref));

    std::vector<T> x = {1.0, 2.0, 3.0}, xref = {1.0/9.0, 1.0/9.0, 4.0/9.0};
    hala::apply_ilu(pntr, indx, diag, ilu, x);
    hassert(testvec(x, xref));
}

#ifdef USE_ENGINE_INTERFACE
constexpr int variant = 1;
#else
constexpr int variant = 2;
#endif

template<int var, class cengine>
std::vector<std::function<void(void)>> make_tests(cengine const &engine){
    return std::vector<std::function<void(void)>>{
        [&]()->void{
            test_batch_axpy<float, variant>(engine);
            test_batch_axpy<double, variant>(engine);
            test_batch_axpy<std::complex<float>, variant>(engine);
            test_batch_axpy<std::complex<double>, variant>(engine);
        },
        [&]()->void{
            test_batch_dot<float, variant>(engine);
            test_batch_dot<double, variant>(engine);
            test_batch_dot<std::complex<float>, variant>(engine);
            test_batch_dot<std::complex<double>, variant>(engine);
        },
        [&]()->void{
            test_batch_max_norm<float, variant>(engine);
            test_batch_max_norm<double, variant>(engine);
            test_batch_max_norm<std::complex<float>, variant>(engine);
            test_batch_max_norm<std::complex<double>, variant>(engine);
        },
        [&]()->void{
            test_batch_scale<float, variant>(engine);
            test_batch_scale<double, variant>(engine);
            test_batch_scale<std::complex<float>, variant>(engine);
            test_batch_scale<std::complex<double>, variant>(engine);
        },
        [&]()->void{
            test_vdivide<float, variant>(engine);
            test_vdivide<double, variant>(engine);
            test_vdivide<std::complex<float>, variant>(engine);
            test_vdivide<std::complex<double>, variant>(engine);
        },
        [&]()->void{
            test_sparse_matrix<float, variant>(engine);
            test_sparse_matrix<double, variant>(engine);
            test_sparse_matrix<std::complex<float>, variant>(engine);
            test_sparse_matrix<std::complex<double>, variant>(engine);
        },
        [&]()->void{
            test_sparse_gemv<float, variant>(engine);
            test_sparse_gemv<double, variant>(engine);
            test_sparse_gemv<std::complex<float>, variant>(engine);
            test_sparse_gemv<std::complex<double>, variant>(engine);
        },
        [&]()->void{
            hala::cpu_engine ecpu;
            test_sparse_gemm<float, variant>(engine);
            test_sparse_gemm<double, variant>(engine);
            test_sparse_gemm<std::complex<float>, variant>(engine);
            test_sparse_gemm<std::complex<double>, variant>(engine);
        },
        [&]()->void{
            test_sparse_trsv<float, variant>(engine);
            test_sparse_trsv<double, variant>(engine);
            test_sparse_trsv<std::complex<float>, variant>(engine);
            test_sparse_trsv<std::complex<double>, variant>(engine);
        },
        [&]()->void{
            test_sparse_trsm<float, variant>(engine);
            test_sparse_trsm<double, variant>(engine);
            test_sparse_trsm<std::complex<float>, variant>(engine);
            test_sparse_trsm<std::complex<double>, variant>(engine);
        }
    };
}
