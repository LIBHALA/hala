#undef USE_ENGINE_INTERFACE

#include "lapack_tests.hpp"

template<class cengine>
std::vector<std::function<void(void)>> make_tests_noengine(cengine const &engine){
    return make_tests<variant>(engine);
}

template std::vector<std::function<void(void)>> make_tests_noengine<hala::cpu_engine>(hala::cpu_engine const&);
#if defined(HALA_ENABLE_CUDA)
template std::vector<std::function<void(void)>> make_tests_noengine<hala::mixed_engine>(hala::mixed_engine const&);
#endif
