/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

/*
 * Testing multiple overloads and interfaces with the same testing code!
 *
 * The engine is the easy part, running the test with cpu_engine and mixed_engine
 * will guarantee that the BLAS and GPU codes get called.
 * Templates will be instantiated correctly.
 *
 * The interface variants with and without an engine are the harder problem.
 * The same test must be performed with either:
 * hala::copy(engine, x, y);
 * or
 * hala::copy(hala::bind_engine_vector(engine, x), hala::bind_engine_vector(engine, y));
 *
 * The tests are actually written with the form:
 * hala::copy(mengine bind(x), bind(y));
 * (note the lack of comma after mengine, the macro could be empty)
 *
 * The mengine and bind are macros that are defined in the appropriate way for the
 * engine and no-engine case. The macros are defined in the common testing header.
 *
 * The tests are instantiated (in macros and template parameters) twice in two separate
 * source .cpp files with and without the USE_ENGINE_INTERFACE macro.
 * The templates accept also the additional "variant" parameter so that the compiler
 * knows that using USE_ENGINE_INTERFACE or not actually creates different templates.
 *
 * - It is important to always call the engine variable "engine",
 *   otherwise the macros will fails.
 * - It is important to always call the bind(x) macro on all vectors.
 */

#include "blas1_tests.hpp"
#include "blas2_tests.hpp"
#include "blas3_tests.hpp"

#ifdef USE_ENGINE_INTERFACE
constexpr int variant = 1;
#else
constexpr int variant = 2;
#endif

template<int var, class cengine>
std::vector<std::function<void(void)>> make_tests(cengine const &engine){
    return std::vector<std::function<void(void)>>{
        [&](void)->void{
            test_vcopy<float, variant>(engine);
            test_vcopy<double, variant>(engine);
            test_vcopy<std::complex<float>, variant>(engine);
            test_vcopy<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_vswap<float, variant>(engine);
            test_vswap<double, variant>(engine);
            test_vswap<std::complex<float>, variant>(engine);
            test_vswap<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_axpy<float, variant>(engine);
            test_axpy<double, variant>(engine);
            test_axpy<std::complex<float>, variant>(engine);
            test_axpy<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_rscalar<float, variant>(engine);
            test_rscalar<double, variant>(engine);
            test_rscalar<std::complex<float>, variant>(engine);
            test_rscalar<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_scal<float, variant>(engine);
            test_scal<double, variant>(engine);
            test_scal<std::complex<float>, variant>(engine);
            test_scal<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_iamax<float, variant>(engine);
            test_iamax<double, variant>(engine);
            test_iamax<std::complex<float>, variant>(engine);
            test_iamax<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_rotate<float, variant>(engine);
            test_rotate<double, variant>(engine);
        },
        [&](void)->void{
            test_gemv<float, variant>(engine);
            test_gemv<double, variant>(engine);
            test_gemv<std::complex<float>, variant>(engine);
            test_gemv<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_symv<float, variant>(engine);
            test_symv<double, variant>(engine);
            test_symv<std::complex<float>, variant>(engine);
            test_symv<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_trmsv<float, variant>(engine);
            test_trmsv<double, variant>(engine);
            test_trmsv<std::complex<float>, variant>(engine);
            test_trmsv<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_tbmsv<float, variant>(engine);
            test_tbmsv<double, variant>(engine);
            test_tbmsv<std::complex<float>, variant>(engine);
            test_tbmsv<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_rank12<float, variant>(engine);
            test_rank12<double, variant>(engine);
            test_rank12<std::complex<float>, variant>(engine);
            test_rank12<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_trmsm<float, variant>(engine);
            test_trmsm<double, variant>(engine);
            test_trmsm<std::complex<float>, variant>(engine);
            test_trmsm<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_syrk<float, variant>(engine);
            test_syrk<double, variant>(engine);
            test_syrk<std::complex<float>, variant>(engine);
            test_syrk<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_symm<float, variant>(engine);
            test_symm<double, variant>(engine);
            test_symm<std::complex<float>, variant>(engine);
            test_symm<std::complex<double>, variant>(engine);
        },
        [&](void)->void{
            test_gemm<float, variant>(engine);
            test_gemm<double, variant>(engine);
            test_gemm<std::complex<float>, variant>(engine);
            test_gemm<std::complex<double>, variant>(engine);
        }
    };
}
