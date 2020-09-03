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

template<typename T>
void sparse_matvec(){
    current_test<T> tests("sp_gemv");
    std::vector<int> pntr = {0, 2, 5, 8, 11, 13};
    std::vector<int> indx = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    std::vector<T> vals = {2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0};
    std::vector<T> A(5 * 5, 0.0), x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<T> y, yref;

    for(int i=0; i<5; i++) for(int j=pntr[i]; j<pntr[i+1]; j++) A[i * 5 + indx[j]] = vals[j];

    hala::gpu_engine engine;
    hala::gpu_vector<T> gv = hala::make_gpu_vector(vals), gx = hala::make_gpu_vector(x), gy;
    hala::gpu_vector<int> gp = hala::make_gpu_vector(pntr), gi = hala::make_gpu_vector(indx);
    sparse_gemv(engine, 'N', 5, 5, hala::get_cast<T>(2.0), gp, gi, gv, gx, hala::get_cast<T>(0.0), gy);
    gy.unload(y);
    refgemm('N', 'N', 5, 1, 5, hala::get_cast<T>(2.0), A, x, hala::get_cast<T>(0.0), yref);
    hassert(testvec(y, yref));
}
