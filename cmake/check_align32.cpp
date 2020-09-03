#include <immintrin.h>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <valarray>

// performs a series of AVX operations in an attempt to determine
// whether the std::vector is by default aligned to 32-byte boundaries
// will crash if not aligned

template<typename T>
std::vector<T> get_vector(size_t num_entries, T start, T jump){
    std::vector<T> result(num_entries);
    for(auto &r : result){
        r = start;
        start += jump;
    }
    return result;
}

void test_floats(){
    size_t num_entries = 8; // there are 8 floats in one __m256

    auto x = get_vector<float>(num_entries, 4.0f, 1.0f);
    auto y = get_vector<float>(num_entries, 3.0f, 2.0f);

    auto reference = get_vector<float>(num_entries, 7.0f, 3.0f);
    auto result    = get_vector<float>(num_entries, 0.0f, 0.0f);

    // result = (x + y);
    // x = result - reference;
    __m256 regx = _mm256_load_ps(x.data());
    __m256 regy = _mm256_load_ps(y.data());

    __m256 regr = _mm256_add_ps(regx, regy);
    _mm256_store_ps(result.data(), regr);

    regx = _mm256_load_ps(reference.data());
    regx = _mm256_sub_ps(regx, regr);

    _mm256_store_ps(x.data(), regx);

    float err = 0.0;
    for(size_t i=0; i<num_entries; i++) err += std::abs(x[i]);
    for(size_t i=0; i<num_entries; i++) err += std::abs(result[i] - reference[i]);

    if (err > 1.E-8)
        throw std::runtime_error("avx float operations failed");
}

void test_doubles(){
    size_t num_entries = 4; // there are 4 doubles in one __m256d

    auto x = get_vector<double>(num_entries, 4.0, 1.0);
    auto y = get_vector<double>(num_entries, 3.0, 2.0);

    auto reference = get_vector<double>(num_entries, 7.0, 3.0);
    auto result    = get_vector<double>(num_entries, 0.0, 0.0);

    // result = (x + y);
    // x = result - reference;
    __m256d regx = _mm256_load_pd(x.data());
    __m256d regy = _mm256_load_pd(y.data());

    __m256d regr = _mm256_add_pd(regx, regy);
    _mm256_store_pd(result.data(), regr);

    regx = _mm256_load_pd(reference.data());
    regx = _mm256_sub_pd(regx, regr);

    _mm256_store_pd(x.data(), regx);

    double err = 0.0;
    for(size_t i=0; i<num_entries; i++) err += std::abs(x[i]);
    for(size_t i=0; i<num_entries; i++) err += std::abs(result[i] - reference[i]);

    if (err > 1.E-15)
        throw std::runtime_error("avx double operations failed");
}

int main(void){

    test_floats();
    test_doubles();

    std::cout << "256bit alignment seems OK" << std::endl;

    return 0;
}
