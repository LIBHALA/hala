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

template<typename T, class cengine>
void test_cg(cengine const &engine){
    current_test<T> tests("conj-grad");
    std::vector<int> pntr = {0, 2, 5, 8, 11, 13};
    std::vector<int> indx = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    std::vector<T> vals = {2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0};
    std::vector<T> b, x, xref = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto A = hala::dense_convert(5, 5, pntr, indx, vals);

    refgemm('N', 'N', 5, 1, 5, 1.0, A, xref, 0.0, b); // compute the RHS

    // using manual ilu factorization
    auto ilu = hala::make_ilu(engine, pntr, indx, vals, 'N');
    // exact solver test, matrix is tri-diagonal hence ILU is exact factorization and convergence is instant
    hala::solve_cg(engine, 2, pntr, indx, vals,
                   [&](auto const &inx, auto &outr)->void{
                       ilu_apply(ilu, inx, outr);
                   }, b, x);
    hassert(testvec(x, xref));

    // test approximate solution
    pntr = {0, 3, 6, 9, 12, 15};
    indx = {0, 1, 4, 0, 1, 2, 1, 2, 3, 2, 3, 4, 0, 3, 4};
    vals = {2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0};

    A = hala::dense_convert(5, 5, pntr, indx, vals);
    refgemm('N', 'N', 5, 1, 5, 1.0, A, xref, 0.0, b); // compute the RHS

    using precision_type = hala::get_precision_type<std::vector<T>>;

    ilu = hala::make_ilu(engine, pntr, indx, vals, 'N');
    x = {1.0, 3.0, -4.0, -7.0, 11.0}; // pick a random vector
    hala::solve_cg_ilu(engine, (std::is_same<precision_type, float>::value) ? 1.E-4 : 1.E-9,
                       pntr, indx, vals, ilu, b, x);

    hassert(testvec(x, xref, hala::norm2(xref))); // approximate solver

    x = {1.0, 3.0, -4.0, -7.0, 11.0}; // pick a random vector
    hala::solve_cg_ilu(engine, (std::is_same<precision_type, float>::value) ? 1.E-4 : 1.E-9,
                       pntr, indx, vals, b, x);

    hassert(testvec(x, xref, hala::norm2(xref))); // approximate solver (factorization behind the scene)
}

template<typename T, class cengine>
void test_cg_batch(cengine const &engine){
    current_test<T> tests("batch_cg");
    std::vector<int> pntr = {0, 2, 5, 8, 11, 13};
    std::vector<int> indx = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    std::vector<T> vals = {2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0};
    std::vector<T> b, x;

    int N = 5, num_rhs = 3;
    auto xref = make_vector<T>(N * num_rhs);

    auto A = hala::dense_convert(N, N, pntr, indx, vals);

    refgemm('N', 'N', N, num_rhs, N, 1.0, A, xref, 0.0, b); // compute the RHS

    // using manual ilu factorization
    auto ilu = hala::make_ilu(engine, pntr, indx, vals, 'N');
    // exact solver test, matrix is tri-diagonal hence ILU is exact factorization and convergence is instant
    hala::solve_batch_cg(engine, 2, pntr, indx, vals,
                         [&](auto const &inx, auto &outr)->void{
                            ilu_apply(ilu, inx, outr, num_rhs);
                         }, b, x);
    hassert(testvec(x, xref, hala::norm2(xref))); // correct for the relative error

    // test approximate solution
    pntr = {0, 3, 6, 9, 12, 15};
    indx = {0, 1, 4, 0, 1, 2, 1, 2, 3, 2, 3, 4, 0, 3, 4};
    vals = {2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0};

    A = hala::dense_convert(N, N, pntr, indx, vals);
    refgemm('N', 'N', N, num_rhs, N, 1.0, A, xref, 0.0, b); // compute the RHS

    using precision_type = hala::get_precision_type<std::vector<T>>;

    ilu = hala::make_ilu(engine, pntr, indx, vals, 'N');
    x = std::vector<T>(N * num_rhs);
    hala::solve_batch_cg_ilu(engine, {(std::is_same<precision_type, float>::value) ? 1.E-4 : 1.E-9, 6},
                             pntr, indx, vals, ilu, b, x);

    hassert(testvec(x, xref, hala::norm2(xref))); // approximate solver

    x = std::vector<T>(); // pick a random vector
    hala::solve_batch_cg_ilu(engine, (std::is_same<precision_type, float>::value) ? 1.E-4 : 1.E-9,
                             pntr, indx, vals, b, x);

    hassert(testvec(x, xref, hala::norm2(xref))); // approximate solver (factorization behind the scene)
}

template<typename T, class cengine>
void test_gmres(cengine const &engine){
    current_test<T> tests("gmres");
    std::vector<int> pntr = {0, 2, 5, 8, 11, 13};
    std::vector<int> indx = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    std::vector<T> vals = {2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0};
    std::vector<T> b, x, xref = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto A = hala::dense_convert(5, 5, pntr, indx, vals);

    refgemm('N', 'N', 5, 1, 5, 1.0, A, xref, 0.0, b); // compute the RHS

    // using manual ilu factorization
    auto ilu = hala::make_ilu(engine, pntr, indx, vals, 'N');
    // exact solver test, matrix is tri-diagonal hence ILU is exact factorization and convergence is instant
    hala::solve_gmres_ilu(engine, 2, 2, pntr, indx, vals, ilu, b, x);
    hassert(testvec(x, xref));

    // test approximate solution
    using precision_type = hala::get_precision_type<std::vector<T>>;

    pntr = {0, 2, 5, 8, 11, 12};
    indx = {0, 1, 0, 1, 3, 1, 2, 3, 0, 2, 3, 4};
    vals = {2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 5.0, 2.0};

    A = hala::dense_convert(5, 5, pntr, indx, vals);

    refgemm('N', 'N', 5, 1, 5, 1.0, A, xref, 0.0, b);

    x = std::vector<T>();
    hala::solve_gmres_ilu(engine, (std::is_same<precision_type, float>::value) ? 1.E-4 : 1.E-9, 6, pntr, indx, vals, b, x);
    hassert(testvec(x, xref));
}
