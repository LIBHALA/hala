#ifndef __HALA_TESTING_COMMON_HPP
#define __HALA_TESTING_COMMON_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#undef NDEBUG
#include <iostream>
#include <iomanip>
#include <string>

#include "hala.hpp"
#include "hala_solvers.hpp"
#include "hala_vex.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::setw;

#ifndef USE_ENGINE_INTERFACE
int const wfirst =  40;
int const wsecond = 10;

bool verbose;                // indicate whether to use verbose reporting
std::string hala_test_name;  // the name of the currently running test
bool hala_test_pass  = true; // helps in reporting whether the last test passed
bool hala_all_tests = true;  // set to false if any test fails

// help generate matrices and vectors with data (not really random, just integers)
int hala_rnd;
#else
extern int const wfirst;
extern int const wsecond;
extern bool verbose;
extern std::string hala_test_name;
extern bool hala_test_pass;
extern bool hala_all_tests;
extern int hala_rnd;
#endif


// get a real value
template<typename T> T getVal(T const &){ hala_rnd++; return (T) hala_rnd; }
// get a complex value
template<typename T> std::complex<T> getVal(std::complex<T> const &){ hala_rnd += 3; return std::complex<T>((T) hala_rnd, (T)(hala_rnd + 1) ); }

// set x to a complex or real entry
template<typename T> void setVal(T &x){ x = getVal(x); }
// make a vector with N entries using hala_rnd
template<typename T, typename size_type> auto make_vector(size_type N){
    std::vector<T> x(static_cast<size_t>(N));
    for(auto &v : x) setVal(v);
    return x;
}
// make a valarray with N entries using hala_rnd
template<typename T, typename size_type> auto make_valarray(size_type N){
    std::valarray<T> x(static_cast<size_t>(N));
    for(auto &v : x) setVal(v);
    return x;
}
// make a symmetric (non-Hermitian) matrix with size N by N
template<typename T, typename size_type> auto make_symmetric(size_type N){
    size_t num_rows = static_cast<size_t>(N);
    std::vector<T> x(num_rows * num_rows);
    for(size_t i=0; i<num_rows; i++){
        setVal(x[i * num_rows + i]);
        for(size_t j=i+1; j<num_rows; j++){
            setVal(x[i * num_rows + j]);
            x[j * num_rows + i] = x[i * num_rows + j];
        }
    }
    return x;
}
// make a Hermitian matrix of size N by N
template<typename T, typename size_type> auto make_hermitian(size_type N){
    size_t num_rows = static_cast<size_t>(N);
    std::vector<T> x(num_rows * num_rows);
    for(size_t i=0; i<num_rows; i++){
        setVal(x[i * num_rows + i]);
        x[i * num_rows + i] = hala::creal(x[i * num_rows + i]);
        for(size_t j=i+1; j<num_rows; j++){
            setVal(x[i * num_rows + j]);
            x[j * num_rows + i] = hala::cconj( x[i * num_rows + j] );
        }
    }
    return x;
}
// find the square-root of an integer, i.e., the dimension of a square matrix based on the size
inline size_t sqrt_int(size_t x){ // lazy square root of an integer
    size_t i = 0;
    while(i*i != x) i++;
    return i;
}
// make a triangular matrix that matches uplo and diag, basically zero-out or reset to 1 the appropriate entries
template<typename VectorLike> void make_triangular(char uplo, char diag, VectorLike &mat){
    // zero out the top or bottom part and respect the diag
    using scalar_type = hala::get_scalar_type<VectorLike>;
    size_t num_rows = sqrt_int(hala::get_size(mat));
    if (hala::is_l(uplo)){
        for(size_t i=1; i<num_rows; i++){
            for(size_t j=0; j<i; j++){
                mat[i*num_rows + j] = hala::get_cast<scalar_type>(0.0);
            }
        }
    }else{
        for(size_t i=0; i<num_rows-1; i++)
            for(size_t j=i+1; j<num_rows; j++)
                mat[i*num_rows + j] = hala::get_cast<scalar_type>(0.0);
    }
    if (!hala::is_n(diag)) // unit diagonal
        for(size_t i=0; i<num_rows; i++)
            mat[i*num_rows + i] = hala::get_cast<scalar_type>(1.0);
}
// make a banded
template<typename VectorLike> void make_banded(int kl, int ku, VectorLike &mat){
    // zero out the top or bottom part and respect the diag
    using scalar_type = hala::get_scalar_type<VectorLike>;
    int num_rows = (int) sqrt_int(hala::get_size(mat));
    for(int i=0; i<num_rows; i++){
        for(int j=0; j<i-ku; j++)          mat[i*num_rows + j] = hala::get_cast<scalar_type>(0.0);
        for(int j=i+kl+1; j<num_rows; j++) mat[i*num_rows + j] = hala::get_cast<scalar_type>(0.0);
    }
}
// pack a banded matrix
template<typename VectorLike> auto pack_banded(int kl, int ku, VectorLike &mat){
    // zero out the top or bottom part and respect the diag
    using scalar_type = hala::get_scalar_type<VectorLike>;
    int num_rows = (int) sqrt_int(hala::get_size(mat));
    std::vector<scalar_type> result(num_rows * (ku + kl + 1));
    for(int i=0; i<num_rows; i++){
        int k = ku - i;
        for(int j=std::max(i-ku, 0); j<std::min(i+kl+1, num_rows); j++)
            result[ i * (ku + kl + 1) + j + k ] = mat[ i * num_rows + j ];
    }
    return result;
}
// pack a triangular matrix
template<typename VectorLike> auto pack_triangular(char uplo, VectorLike &mat){
    using scalar_type = hala::get_scalar_type<VectorLike>;
    size_t num_rows = sqrt_int(hala::get_size(mat));
    std::vector<scalar_type> result( (num_rows * (num_rows + 1)) / 2 );
    size_t c = 0;
    if (hala::is_l(uplo)){
        for(size_t i=0; i<num_rows; i++)
            for(size_t j=i; j<num_rows; j++)
                result[c++] = mat[i * num_rows + j];
    }else{
        for(size_t i=0; i<num_rows; i++)
            for(size_t j=0; j<=i; j++)
                result[c++] = mat[i * num_rows + j];
    }
    return result;
}
// print a matrix as 2D array
template<typename VectorLike> void dump(int M, int N, VectorLike const &x){
    cout << " ... dumping a matrix to cout ... \n";
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++) cout << x[j*M + i] << "  ";
        cout << "\n";
    }
}

// simple reference implementation of gemm, uses 3 for-loops, can be used for gemv and other types of matrices
template<typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
void refgemm(char transa, char transb, int M, int N, int K, FSa fa, VectorLikeA const &A,
          VectorLikeB const &B, FSb fb, VectorLikeC &C){
    using scalar_type = hala::get_scalar_type<VectorLikeA>;
    auto alpha = hala::get_cast<scalar_type>(fa);
    auto beta  = hala::get_cast<scalar_type>(fb);

    assert((std::abs(beta) == 0.0) || (C.size() == (size_t) N * M));
    if (std::abs(beta) == 0.0)
        C.resize((size_t) (M * N));

    auto opa = [&](scalar_type const &a)->
        scalar_type{
            return (hala::is_c(transa)) ? hala::cconj(a) : a;
        };
    auto opb = [&](scalar_type const &b)->
        scalar_type{
            return (hala::is_c(transb)) ? hala::cconj(b) : b;
        };

    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            scalar_type sum = static_cast<scalar_type>(0.0);
            if (hala::is_n(transa) && hala::is_n(transb)){
                for(int k=0; k<K; k++)
                    sum += A[k * M + j] * B[i * K + k];
            }else if (hala::is_n(transa) && !hala::is_n(transb)){
                for(int k=0; k<K; k++)
                    sum += A[k * M + j] * opb(B[i + k * N]);
            }else if (!hala::is_n(transa) && hala::is_n(transb)){
                for(int k=0; k<K; k++)
                    sum += opa(A[k + j * K]) * B[i * K + k];
            }else{
                for(int k=0; k<K; k++)
                    sum += opa(A[k + j * K]) * opb(B[i + k * N]);
            }
            C[i * M + j] = hala::get_cast<scalar_type>(alpha * sum + beta * C[i * M + j]);
        }
    }
}

template<class vec> void dump(vec const &x){
    for(size_t i=0; i<x.size(); i++) cout << x[i] << "  ";
    cout << endl;
}

/************************************/
/*    Report Helpers                */
/************************************/
inline void begin_report(std::string const &s){
    int wspaces = 31 + (int) s.size() / 2;
    cout << "\n ----------------------------------------------------------- \n";
    cout << setw(wspaces) << s << "\n";
    cout << " ----------------------------------------------------------- \n" << endl;
}
inline void end_report(std::string const &s){
    if (hala_all_tests){
        begin_report(s + " : ALL PASS");
    }else{
        begin_report(s + " : --- ERROR: SOME FAILED ---");
    }
}

template<typename T> // cout typename, name and pass
std::string generate_test_name(std::string const &name = std::string()){
    std::string s = (name.empty()) ? hala_test_name : name;
    if (std::is_same<T, float>::value){
        s += "<float>()";
    }else if (std::is_same<T, double>::value){
        s += "<double>()";
    }else if (std::is_same<T, std::complex<float>>::value){
        s += "<complex<float>>()";
    }else if (std::is_same<T, std::complex<double>>::value){
        s += "<complex<double>>()";
    }else{
        s += "<all>()";
    }
    return s;
}

inline std::string regname(hala::regtype reg){
    switch(reg){
        case hala::regtype::sse    : return "sse";
        case hala::regtype::avx    : return "avx";
        case hala::regtype::avx512 : return "avx512";
        default:
            return "none";
    }
}

template<typename T> // cout typename, name and pass
void report(std::string const &name = std::string(), bool pass = true){
    cout << setw(40) << generate_test_name<T>(name) << setw(10) << ((pass) ? "Pass" : "Fail") << endl;
}

template<typename T>
struct current_test{
    current_test(std::string const &name){
        hala_test_name = name;
        hala_test_pass = true;
    };
    ~current_test(){
        if (verbose || !hala_test_pass)
            report<T>(hala_test_name, hala_test_pass);
    };
};

#define hassert(_result_)       \
    if (!(_result_)){           \
        hala_all_tests = false;      \
        hala_test_pass = false; \
        throw std::runtime_error( std::string("test ") + hala_test_name + " in file: " + __FILE__ + " line: " + std::to_string(__LINE__) ); \
    }                           \

constexpr bool show_details = true;
constexpr bool no_details   = false;

template<bool dump_err = no_details, class vec, class vecy>
bool testvec(vec const &x, vecy const &y, double correction = 1.0){
    if (x.size() != y.size()){ cout << "ERROR: vector size mismatch " << x.size() << " and " << y.size() << endl; return false; };
    if (dump_err){ dump(x); dump(y); }
    double err = 0.0;
    bool isvalid = true;
    for(size_t i=0; i<x.size(); i++){
        err += hala::hala_abs(x[i] - y[i]);
        if (std::isnan(hala::creal(x[i])) || std::isnan(hala::creal(y[i])) ||
            std::isnan(hala::cimag(x[i])) || std::isnan(hala::cimag(y[i])) ||
            std::isinf(hala::creal(x[i])) || std::isinf(hala::creal(y[i])) ||
            std::isinf(hala::cimag(x[i])) || std::isinf(hala::cimag(y[i]))) isvalid = false;
    }
    if (dump_err) cout << "computed errors: " << err << " corrected error: " << err / correction << " valid: " << ((isvalid) ? "true" : "false") << endl;
    if (hala::is_float<typename vec::value_type>::value ||
        hala::is_fcomplex<typename vec::value_type>::value){
        return (isvalid && ((err / correction) < 1.E-5));
    }else{
        return (isvalid && ((err / correction) < 1.E-14));
    }
}

template<bool dump_err = false, class vec>
bool testvec(vec const &x, vec const &y, double correction = 1.0){
    return testvec<dump_err, vec, vec>(x, y, correction);
}

template<bool dump_err = no_details, class vec, class vecy>
bool testints(vec const &x, vecy const &y){
    hassert(x.size() == y.size());
    if (dump_err){ dump(x); dump(y); }
    for(size_t i=0; i<x.size(); i++) if (x[i] != y[i]) return false;
    return true;
}

template<bool dump_err = false, class vec>
bool testints(vec const &x, vec const &y){
    return testints<dump_err, vec, vec>(x, y);
}


template<bool dump_err = false, typename numa, typename numb>
bool testnum(numa const &x, numb const &y, double correction = 1.0){
    if (std::isnan(hala::creal(x)) || std::isnan(hala::creal(y)) ||
        std::isnan(hala::cimag(x)) || std::isnan(hala::cimag(y)) ||
        std::isinf(hala::creal(x)) || std::isinf(hala::creal(y)) ||
        std::isinf(hala::cimag(x)) || std::isinf(hala::cimag(y))) return false;
    std::complex<double> d = {hala::creal(x) - hala::creal(y), hala::cimag(x) - hala::cimag(y)};
    if (dump_err) cout << "computed errors: " << std::abs(d) << endl;
    if (hala::is_float<numa>::value ||
        hala::is_fcomplex<numa>::value){
        return ((std::abs(d) / correction) < 1.E-5);
    }else{
        return ((std::abs(d) / correction) < 1.E-14);
    }
}

template<bool dump_err = false, typename numa>
bool testnum(numa const &x, numa const &y, double correction = 1.0){
    return testnum<dump_err, numa, numa>(x, y, correction);
}

inline void perform(std::function<void(void)> test){
    try{
        test();
        if (!verbose) report<int>();
    }catch (std::runtime_error &e){
        hala_all_tests = false;
        hala_test_pass = false;
        cout << "Failed: " << e.what() << endl;
    }
}

inline int test_result(){ return (hala_all_tests) ? 0 : 1; }

template<typename engine_type> inline int get_device(engine_type const&){ return 0; }
#ifdef HALA_ENABLE_GPU
inline int get_device(hala::mixed_engine const &engine){ return engine.gpu().device(); }
#endif

#ifdef USE_ENGINE_INTERFACE
    #define mengine engine,
    #define bind(x) (x)
#else
    #define mengine
    #define bind(x) hala::bind_engine_vector(engine, (x))
#endif

#endif
