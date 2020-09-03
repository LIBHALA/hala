/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */
#include "lapack1_tests.hpp"

#ifdef USE_ENGINE_INTERFACE
constexpr int variant = 1;
#else
constexpr int variant = 2;
#endif

template<int var, class cengine>
std::vector<std::function<void(void)>> make_tests(cengine const &engine){
    return std::vector<std::function<void(void)>>{
        [&](void)->void{
            test_potrfs<float, variant>(engine);
            test_potrfs<double, variant>(engine);
            test_potrfs<std::complex<float>, variant>(engine);
            test_potrfs<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_getrfs<float, variant>(engine);
            test_getrfs<double, variant>(engine);
            test_getrfs<std::complex<float>, variant>(engine);
            test_getrfs<std::complex<double>, variant>(engine);
        },
        [](void)->void{
            test_solvel2<float>();
            test_solvel2<double>();
            test_solvel2<std::complex<float>>();
            test_solvel2<std::complex<double>>();
        },
        [](void)->void{
            test_svd<float>();
            test_svd<double>();
            test_svd<std::complex<float>>();
            test_svd<std::complex<double>>();
        },
        [](void)->void{
            test_dsolve_plu<float>();
            test_dsolve_plu<double>();
            test_zsolve_plu<std::complex<float>>();
            test_zsolve_plu<std::complex<double>>();
        },
        [](void)->void{
            test_general_qr<float>();
            test_general_qr<double>();
            test_general_qr<std::complex<float>>();
            test_general_qr<std::complex<double>>();
        }
    };
}
