#define USE_ENGINE_INTERFACE

#include "sparse_tests.hpp"

template<class cengine>
std::vector<std::function<void(void)>> make_tests_wengine(cengine const &engine){
    return make_tests<variant>(engine);
}

template std::vector<std::function<void(void)>> make_tests_wengine<hala::cpu_engine>(hala::cpu_engine const&);
#ifdef HALA_ENABLE_GPU
template std::vector<std::function<void(void)>> make_tests_wengine<hala::mixed_engine>(hala::mixed_engine const&);
#endif
