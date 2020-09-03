#define USE_ENGINE_INTERFACE

#include "blas_tests.hpp"

template<class cengine>
std::vector<std::function<void(void)>> make_tests_wengine(cengine const &engine){
    return make_tests<variant>(engine);
}

template std::vector<std::function<void(void)>> make_tests_wengine<hala::cpu_engine>(hala::cpu_engine const&);
#if defined(HALA_ENABLE_CUDA) || defined(HALA_ENABLE_ROCM)
template std::vector<std::function<void(void)>> make_tests_wengine<hala::mixed_engine>(hala::mixed_engine const&);
#endif
