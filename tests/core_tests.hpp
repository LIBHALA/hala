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

struct CustomComplex{ float r, i; };
struct CustomZomplex{ double r, i; };

namespace hala{
    template<> struct is_fcomplex<CustomComplex> : std::true_type{};
    template<> struct is_dcomplex<CustomZomplex> : std::true_type{};
}

// convert float to all possible types,
// r and i indicate what the real and imag results should be
template<typename T>
void convert_to_all(T x, double r, double i){
    float fr = static_cast<float>(r), fi = static_cast<float>(i);

    hassert( hala::creal(x) == r );
    hassert( hala::cimag(x) == i );

    hassert( hala::get_cast<float>(x) == fr );
    hassert( hala::get_cast<double>(x) == r );

    hassert( hala::get_cast<std::complex<float>>(x).real() == fr );
    hassert( hala::get_cast<std::complex<float>>(x).imag() == fi );
    hassert( hala::get_cast<std::complex<double>>(x).real() == r );
    hassert( hala::get_cast<std::complex<double>>(x).imag() == i );

    hassert( hala::get_cast<CustomComplex>(x).r == fr );
    hassert( hala::get_cast<CustomComplex>(x).i == fi );
    hassert( hala::get_cast<CustomZomplex>(x).r == r );
    hassert( hala::get_cast<CustomZomplex>(x).i == i );

    hassert( hala::get_cast<std::complex<double>>(hala::cconj(x)).real() == r );
    hassert( hala::get_cast<std::complex<double>>(hala::cconj(x)).imag() == -i );
}

void type_casting(){
    current_test<int> tests("type_casting");
    convert_to_all(float(2.0), 2.0, 0.0);
    convert_to_all(double(3.0), 3.0, 0.0);

    convert_to_all(std::complex<float>(3.0, 2.0), 3.0, 2.0);
    convert_to_all(std::complex<double>(4.0, 5.0), 4.0, 5.0);

    convert_to_all(CustomComplex{6.0, 7.0}, 6.0, 7.0);
    convert_to_all(CustomZomplex{8.0, 9.0}, 8.0, 9.0);
}

namespace hala{
    template<typename T> struct define_vdefault<hala::cpu_engine, T>{
        using vector = typename std::valarray<T>;
    };
}

template<typename T>
void default_internal_vector(){
    current_test<T> tests("dfvectors");
    std::vector<int> pntr = {0, 2, 5, 8, 11, 13};
    std::vector<int> indx = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    std::vector<T> vals = {3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0};
    std::vector<T> b, x, xref = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto A = hala::dense_convert(5, 5, pntr, indx, vals);

    refgemm('N', 'N', 5, 1, 5, 1.0, A, xref, 0.0, b); // compute the RHS

    std::valarray<T> diag(3.0, 5); // the diagonal entries
    hala::cpu_engine engine;

    hala::solve_cg(engine, 6,
                   pntr, indx, vals,
                   [&](auto const &inx, auto &outr)->void{ // the point here is that inx and outr must be valarrays
                       outr = inx / diag; // the inputs must be valarrays
                   }, b, x);
    hassert(testvec(x, xref));
}

void custom_complex(){
    current_test<int> tests("user-complex");
    std::vector<int> pntr = {0, 2, 4};
    std::vector<int> indx = {0, 1, 0, 1};
    std::vector<CustomZomplex> vals = {{3.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}, {2.0, 1.0}};
    std::vector<CustomZomplex> x = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::complex<double>> r, rref = {{-3.0, 10.0}, {3.0, 13.0}};

    hala::sparse_gemv('N', 2, 2, 1.0, pntr, indx, vals, x, 0.0, r);
    hassert(testvec(r, rref));

    size_t num_entries = 16;
    hala_rnd = 3;
    // get some semi-random numbers
    auto rndx = make_vector<std::complex<float>>(num_entries);
    std::vector<CustomComplex> xx;
    hala::vcopy(rndx, xx);

    for(size_t i=0; i<num_entries; i += hala::mmpack<std::complex<float>>::stride)
        hala::mmbind(&xx[i]) *= 2.0;

    for(auto &c : rndx) c *= 2.0;
    std::valarray<std::complex<float>> val1, val2;
    hala::vcopy(rndx, val1);
    hala::vcopy(  xx, val2);

    hassert(testvec(val1, val2));

    { // test interoperability between CustomComplex and std::complex<float>
        auto mmx = hala::mmbind(&xx[0]);   // bind to x with type CustomComplex
        auto mm2 = hala::mmload(&val2[0]); // load with type std::complex<float>
        mmx -= mm2; // first "stride" entries of x must be zero
    }
    val2 = std::valarray<std::complex<float>>(0.0, hala::mmpack<std::complex<float>>::stride);
    val1 = std::valarray<std::complex<float>>(3.0, hala::mmpack<std::complex<float>>::stride);
    for(size_t i=0; i<val1.size(); i++) val1[i] = hala::get_cast< std::complex<float> >(xx[i]);

    hassert(testvec(val1, val2));
}

template<typename T>
struct silly_vector{
    silly_vector(int num) : size(num) { pntr = new T[num]; }
    ~silly_vector(){ delete[] pntr; }
    T* pntr;
    int size;

    silly_vector<T>(silly_vector<T> const&) = delete;
    silly_vector<T>(silly_vector<T> &&) = delete;
    silly_vector<T>& operator =(silly_vector<T> const&) = delete;
    silly_vector<T>& operator =(silly_vector<T> &&) = delete;
};

namespace hala{
    template<typename T> struct define_type<silly_vector<T>>{
        using value_type = T;
    };
    template<typename T> struct define_type<silly_vector<T> const>{
        using value_type = T;
    };
    template<typename T> struct define_size<silly_vector<T>>{
        static size_t size(silly_vector<T> const &x){ return (size_t) x.size; }
    };
    template<typename T> struct define_data<const silly_vector<T>>{
        static T const* data(silly_vector<T> const &x){ return x.pntr; }
    };
    template<typename T> struct define_data<silly_vector<T>>{
        static T* data(silly_vector<T> &x){ return x.pntr; }
    };
}

void custom_vector(){
    current_test<int> tests("user-vector");
    static_assert(!hala::has_resize<silly_vector<double>>::value, "thinking silly vector can be resized");
    static_assert(hala::has_resize<std::vector<double>>::value, "thinking std::vector cannot be resized");

    silly_vector<int> pntr(3);
    pntr.pntr[0] = 0;
    pntr.pntr[1] = 2;
    pntr.pntr[2] = 4;
    silly_vector<int> indx(4);
    indx.pntr[0] = 0;
    indx.pntr[1] = 1;
    indx.pntr[2] = 0;
    indx.pntr[3] = 1;
    silly_vector<double> vals(4);
    vals.pntr[0] = 1.0;
    vals.pntr[1] = 2.0;
    vals.pntr[2] = 3.0;
    vals.pntr[3] = 4.0;
    std::vector<double> x = {5.0, 7.0};
    auto xw = hala::wrap_array(x.data(), 2);
    std::valarray<double> r, ref = {19.0, 43.0};

    hala::sparse_gemv('N', 2, 2, 1.0, pntr, indx, vals, xw, 0.0, r);
    hassert(testvec(r, ref));

    silly_vector<double> bad_vals(1);
    bool pass = [&](void)->
        bool{
            try{
                hala::sparse_gemv('N', 2, 2, 1.0, pntr, indx, vals, xw, 0.0, bad_vals);
                return false;
            }catch(std::runtime_error&){
                // OK we generated an exception for bad_vals which is too short and cannot be resized
                return true;
            }
        }(); // evaluate the block to get the result

    hala::runtime_assert(pass, "failing to raise exception on bad resize");
}

void raw_arrays(){
    current_test<int> tests("raw-arrays");
    std::vector<float> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> x = {1.0, 2.0};
    std::vector<float> y, yref = {7.0, 10.0};

    float *pA = A.data();
    float *px = x.data();

    hala::gemv('N', 2, 2, 1.0, pA, px, 0, y);
    hassert(testvec(y, yref));

    hala::scal(2, yref);
    float *py = y.data();
    hala::gemv('N', 2, 2, 1.0, pA, px, 1.0, py);
    hassert(testvec(y, yref));
}
