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

template<typename T, int, class cengine> void test_vcopy(cengine const &engine){
    current_test<T> tests("copy");
    hala_rnd = -1;
    auto x = make_vector<T>(3);
    std::vector<T> t, y;
    hala::vcopy(mengine bind(x), bind(t));
    hala::vcopy(mengine bind(t), bind(y));
    hassert(testvec(x, y));

    x = make_vector<T>(7);
    y = make_vector<T>(5);
    t = y;
    for(size_t i=0; i<3; i++) t[2*i] = x[3*i];

    hala::vcopy(mengine bind(x), bind(y), 3, 2);
    hassert(testvec(t, y));
}

template<typename T, int, class cengine> void test_vswap(cengine const &engine){
    current_test<T> tests("swap");
    hala_rnd = -3;
    auto x = make_vector<T>(3);
    auto y = make_vector<T>(5);
    auto refx = x;
    auto refy = y;
    for(size_t i=0; i<3; i++) std::swap(refx[i], refy[2*i]);
    hala::vswap(mengine bind(x), bind(y), 1, 2);

    hassert(testvec(x, refx));
    hassert(testvec(y, refy));
}

template<typename T, int, class cengine> void test_axpy(cengine const &engine){
    current_test<T> tests("axpy");
    hala_rnd = -3;
    auto x = make_valarray<T>(7);
    auto y = make_valarray<T>(7);
    std::valarray<T> refy = y + hala::get_cast<T>(2.0) * x;
    hala::axpy(mengine 2, bind(x), bind(y));
    hassert(testvec(y, refy));

    refy += hala::get_cast<T>(3.0) * x;
    hala::axpy(mengine 7, 3, bind(x), 1, bind(y), 1);
    hassert(testvec(y, refy));

    auto vx = make_vector<T>(7);
    auto vy = make_vector<T>(7);
    std::vector<T> t = vy;
    for(size_t i=0; i<3; i++) t[3*i] += vx[3*i];
    hala::axpy(mengine 1, bind(vx), bind(vy), 3, 3, 3);
    hassert(testvec(t, vy));
}

template<typename T, int, class cengine> void test_rscalar(cengine const &engine){
    current_test<T> tests("norm2/asum/dot");
    hala_rnd = -2;
    auto x = make_vector<T>(7);

    auto nrm = std::real(static_cast<T>(0.0));
    for(size_t i=0; i<x.size(); i+=3) nrm += std::real( hala::cconj(x[i]) * x[i] );
    hassert(testnum(hala::norm2(mengine bind(x), 3), std::sqrt(nrm)));

    nrm = std::real(static_cast<T>(0.0));
    for(size_t i=0; i<x.size(); i+=3) nrm += std::abs(std::real(x[i])) + std::abs(std::imag(x[i]));
    hassert(testnum(hala::asum(mengine bind(x), 3), nrm));

    nrm = std::real(static_cast<T>(0.0));
    for(auto v : x) nrm += std::real( hala::cconj(v) * v );
    hassert(testnum(hala::norm2(mengine bind(x)), std::sqrt(nrm)));

    nrm = std::real(static_cast<T>(0.0));
    for(auto v : x) nrm += std::abs(std::real(v)) + std::abs(std::imag(v));
    hassert(testnum(hala::asum(mengine bind(x)), nrm));

    auto y = make_vector<T>(7);
    T d = static_cast<T>(0.0);
    for(size_t i=0; i<x.size(); i += 4) d += hala::cconj(x[i]) * y[i];
    hassert(testnum(hala::dot(mengine bind(x), bind(y), 4, 4), d));

    d = static_cast<T>(0.0);
    for(size_t i=0; i<x.size(); i++) d += x[i] * y[i];
    hassert(testnum(hala::dotu(mengine bind(x), bind(y)), d));
}

template<typename T, int, class cengine> void test_scal(cengine const &engine){
    current_test<T> tests("scal");
    hala_rnd = -3;
    auto x = make_valarray<T>(11);
    auto refx = make_valarray<T>(1);
    refx = hala::get_cast<T>(3.0) * x;
    hala::scal(mengine 3.0, bind(x));
    hassert(testvec(x, refx));

    auto s = make_valarray<T>(1);
    refx *= s[0];
    hala::scal(mengine s[0], bind(x));
    hassert(testvec(x, refx));

    auto vx = make_vector<T>(3);
    auto rx = make_vector<T>(1);
    rx = vx;
    for(size_t i=0; i<vx.size(); i+=2) rx[i] *= hala::get_cast<T>(4.0);
    hala::scal(mengine 4.0, bind(vx), 2);
    hassert(testvec(rx, vx));
}

template<typename T, int, class cengine> void test_iamax(cengine const &engine){
    current_test<T> tests("iamax");
    hala_rnd = 1;
    auto x = make_valarray<T>(5); // note that the entries get larger

    // last entry is largest
    hassert(testnum(hala::iamax(mengine bind(x)), 4));
    // last entry, but there are only 3 active indexes
    hassert(testnum(hala::iamax(mengine bind(x), 2), 2));

    x[4] = hala::get_cast<T>(1.0); // last entry no longer largest
    hassert(testnum(hala::iamax(mengine bind(x)), 3));
}

template<typename T, int, class cengine> void test_rotate(cengine const &engine){
    current_test<T> tests("rotate");

    T c, s, sa = 2, sb = 1;
    hala::rotg(engine, sa, sb, c, s);
    hassert(testnum(c, 2.0 / std::sqrt(5.0)));
    hassert(testnum(s, 1.0 / std::sqrt(5.0)));

    std::vector<T> vx(5, 2.0);
    std::vector<T> vy(5, 1.0);
    hala::rot(mengine bind(vx), bind(vy), c, s);
    hassert(testvec(vx, std::vector<T>(5, std::sqrt(5.0))));
    hassert(testvec(vy, std::vector<T>(5, 0.0)));

    T d1 = 1.0, d2 = 1.0, x = 1.0, y = 0.0;
    std::vector<T> param(5);
    param[0] = static_cast<T>(-1.0);
    std::vector<T> ref_param = {-2.0, 0.0, 0.0, 0.0, 0.0};

    hala::rotmg(d1, d2, x, y, param);
    hassert(testvec(param, ref_param));

    hala_rnd = 4;
    vx = make_vector<T>(3);
    vy = make_vector<T>(3);
    std::vector<T> vxref = vx;
    std::vector<T> vyref = vy;

    hala::rotm(mengine bind(vx), bind(vy), bind(param)); // vx and vy are not modified, param is zero but with flag to assume identity
    hassert(testvec(vx, vxref));
    hassert(testvec(vy, vyref));
}
