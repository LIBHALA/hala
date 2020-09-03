#ifndef __HALA_INTRINSICS_HPP
#define __HALA_INTRINSICS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_common.hpp"

/*!
 * \internal
 * \file hala_intrinsics.hpp
 * \brief HALA wrapper to the intrinsic types.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAINTR
 *
 * Contains the overload wrappers around intrinsic types used for manual vectorization.
 * \endinternal
 */

/*!
 * \internal
 * \ingroup HALAVEX
 * \addtogroup HALAINTR Intrinsic Overloads
 *
 * \par Vector Extensions
 *
 * SSE and AVX refer to special register types supported on most CPUs
 * that allow for multiple simultaneous floating point operations.
 * The capability is accessible through special C-style functions,
 * this module provides C++-style overloads that facilitate generic
 * template programming.
 *
 * The most notable feature in this module is the special overloads that
 * handle complex number multiplication, division, and square-root.
 * \endinternal
 */

#if defined(__SSE3__) || defined(__AVX__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace hala{

/*!
 * \ingroup HALAVEX
 * \brief Defines the type of register to use for vectorization.
 *
 * Each options requires compatible compiler flags:
 * - \b sse requires HALA_ENABLE_SSE and the -msse2 and -mss3 flags
 * - \b avx requires HALA_ENABLE_AVX and the -mavx flag
 * - \b avx512 requires HALA_ENABLE_AVX512 and the -mavx512f flag
 *
 * In addition, using HALA_ENABLE_FMA and the -mfma flag will enable extra templates
 * for the sse and avx register modes (the fma instructions for avx512 are included
 * by the -mavx512f flag).
 */
enum class regtype{
    //! \brief No vectorization, i.e., generate code equivalent to the serial code.
    none,
    //! \brief Using the 128 bit SSE registers.
    sse,
    //! \brief Using the 256 bit AVX registers.
    avx,
    //! \brief Using the 512 bit AVX-512 registers.
    avx512
};

#ifndef __HALA_DOXYGEN_SKIP

template<typename T> struct is_simple{
    static constexpr bool value = is_float<T>::value || is_double<T>::value || is_fcomplex<T>::value || is_dcomplex<T>::value;
};

template<typename T> struct is_pfloat{ // check if precision is float
    static constexpr bool value = is_float<T>::value || is_fcomplex<T>::value;
};

template<typename T> struct is_pdouble{ // check if precision is double
    static constexpr bool value = is_double<T>::value || is_dcomplex<T>::value;
};

// make a zero of the appropriate type
template<typename T, regtype, typename = void> struct mm_zero{
    static inline T get(){ return get_cast<T>(0); }
};

// load aligned or unaligned
template<typename T, regtype, typename = void> struct mm_load{
    using standard_type = typename define_standard_type<T>::value_type;
    static inline auto align(T const *a){ return get_cast<standard_type>(a[0]); }
    static inline auto unali(T const *a){ return get_cast<standard_type>(a[0]); }
};

// store aligned or unaligned
template<typename T, regtype, typename = void> struct mm_store{
    using standard_type = typename define_standard_type<T>::value_type;
    static inline void align(T *a, standard_type const &b){ a[0] = get_cast<T>(b); }
    static inline void unali(T *a, standard_type const &b){ a[0] = get_cast<T>(b); }
};

// set the value using zeros, a, or combination of multiple values
template<typename T, regtype> inline auto mm_set(T a){ return a; }


// real basic arithmetic for the "none" case
template<typename T> inline T mm_fmadd(T const &a, T const &b, T const &c){ return a * b + c; }
template<typename T> inline T mm_sqrt(T const &a){ return std::sqrt(a); }
template<typename T> inline T mm_abs(T const &a){ return std::abs(a); }
// note that in the complex case the result from abs2 should be a complex number
// with both real and complex parts set to the value of the real abs
template<typename T> inline T mm_abs2(T const &a){ return a * a; }
template<typename T> inline std::complex<T> mm_abs2(std::complex<T> const &a){
    T r = std::real(a) * std::real(a) + std::imag(a) * std::imag(a);
    return {r, r};
}

template<typename T>
inline std::enable_if_t<is_simple<T>::value, T> mm_complex_mul(T const &a, T const &b){ return a * b; }
template<typename T>
inline std::enable_if_t<is_simple<T>::value, T> mm_complex_fmadd(T const &a, T const &b, T const &c){ return a * b + c; }
template<typename T>
inline std::enable_if_t<is_simple<T>::value, T> mm_complex_div(T const &a, T const &b){ return a / b; }
template<typename T>
inline std::enable_if_t<is_simple<T>::value, T> mm_complex_conj(T const &a){ return hala_conj(a); }
template<typename T>
inline std::enable_if_t<is_simple<T>::value, T> mm_complex_abs(T const &a){ return std::abs(a); }
template<typename T>
inline std::enable_if_t<is_simple<T>::value, std::complex<T>> mm_complex_abs(std::complex<T> const &a){
    T r = std::abs(a);
    return {r, r};
}
template<typename T, typename std::enable_if_t<is_simple<T>::value>* = nullptr>
inline auto mm_complex_abs2(T const &a){ return mm_abs2(a); }
template<typename T>
inline std::enable_if_t<is_simple<T>::value, T> mm_complex_sqrt(T const &a){ return std::sqrt(a); }


#ifdef __SSE3__

// common I/O, set zero, set one, load/store
template<typename T> struct mm_zero<T, regtype::sse, std::enable_if_t<is_pfloat<T>::value>>{
    static inline auto get(){ return _mm_setzero_ps(); }
};
template<typename T> struct mm_zero<T, regtype::sse, std::enable_if_t<is_pdouble<T>::value>>{
    static inline auto get(){ return _mm_setzero_pd(); }
};

template<typename T> struct mm_load<T, regtype::sse, std::enable_if_t<is_pfloat<T>::value>>{
    static inline auto align(T const *a){ return _mm_load_ps(pconvert(a)); }
    static inline auto unali(T const *a){ return _mm_loadu_ps(pconvert(a)); }
};
template<typename T> struct mm_load<T, regtype::sse, std::enable_if_t<is_pdouble<T>::value>>{
    static inline auto align(T const *a){ return _mm_load_pd(pconvert(a)); }
    static inline auto unali(T const *a){ return _mm_loadu_pd(pconvert(a)); }
};

template<typename T> struct mm_store<T, regtype::sse, std::enable_if_t<is_pfloat<T>::value>>{
    static inline void align(T *a, __m128 const &b){ _mm_store_ps(pconvert(a), b); }
    static inline void unali(T *a, __m128 const &b){ _mm_storeu_ps(pconvert(a), b); }
};
template<typename T> struct mm_store<T, regtype::sse, std::enable_if_t<is_pdouble<T>::value>>{
    static inline void align(T *a, __m128d const &b){ _mm_store_pd(pconvert(a), b); }
    static inline void unali(T *a, __m128d const &b){ _mm_storeu_pd(pconvert(a), b); }
};

template<> inline auto mm_set<float,  regtype::sse>(float  a){ return _mm_set_ps1(a); }
template<> inline auto mm_set<double, regtype::sse>(double a){ return _mm_set_pd1(a); }
template<> inline auto mm_set<std::complex<float>,  regtype::sse>(std::complex<float> a){ return _mm_setr_ps(a.real(), a.imag(), a.real(), a.imag()); }
template<> inline auto mm_set<std::complex<double>, regtype::sse>(std::complex<double> a){ return _mm_setr_pd(a.real(), a.imag()); }


// common arithmetic

inline __m128  mm_sqrt(__m128  const &a){ return _mm_sqrt_ps(a); }
inline __m128d mm_sqrt(__m128d const &a){ return _mm_sqrt_pd(a); }
#ifdef __FMA__
inline __m128  mm_fmadd(__m128  const &a, __m128  const &b, __m128  const &c){ return _mm_fmadd_ps(a, b, c); }
inline __m128d mm_fmadd(__m128d const &a, __m128d const &b, __m128d const &c){ return _mm_fmadd_pd(a, b, c); }
#endif

// complex<float> helper functions
inline __m128 mm_complex_real(__m128 const &a){ return _mm_shuffle_ps(a, a, 160); } // 0, 1, 2, 3 -> 0, 0, 2, 2
inline __m128 mm_complex_imag(__m128 const &a){ return _mm_shuffle_ps(a, a, 245); } // 0, 1, 2, 3 -> 1, 1, 3, 3
inline __m128 mm_complex_swap(__m128 const &a){ return _mm_shuffle_ps(a, a, 177); } // 0, 1, 2, 3 -> 1, 0, 3, 2
inline __m128 mm_complex_two1(__m128 const &a){ return _mm_shuffle_ps(a, a, 240); } // 0, 1, 2, 3 -> 0, 0, 3, 3

inline __m128 mm_hadd(__m128 const &a){ return _mm_hadd_ps(a, a); } // 0, 1, 2, 3 -> 0 + 1, 2 + 3, 0 + 1, 2 + 3

inline __m128 mm_sign(__m128 const &a){ // 0, 1, 2, 3 -> sign(0), sign(1), sign(2), sign(3)
    return _mm_or_ps(_mm_set1_ps(1.0f), _mm_and_ps(_mm_set1_ps(-0.0f), a));
}
inline __m128 mm_sign_imag(__m128 const &a){ // 0, 1, 2, 3 -> 1.0, sign(1), 1.0, sign(3)
    return _mm_or_ps(_mm_set1_ps(1.0f), _mm_and_ps(_mm_set_ps(-0.0f, 0.0f, -0.0f, 0.0f), a));
}

// complex arithmetic
inline __m128 mm_abs(__m128 const &a){ return a * mm_sign(a); }
inline __m128 mm_complex_mul(__m128 const &a, __m128 const &b){
    #ifdef __FMA__
    return _mm_fmaddsub_ps( mm_complex_real(a), b, mm_complex_imag(a) * mm_complex_swap(b) );
    #else
    return _mm_addsub_ps(mm_complex_real(a) * b, mm_complex_imag(a) * mm_complex_swap(b));
    #endif
}
inline __m128 mm_complex_conj(__m128 const &a){ // 0, 1, 2, 3 -> 0, -1, 2, -3
    return _mm_xor_ps(a, _mm_set_ps(-0.0f, 0.0f, -0.0f, 0.0f));
}
inline __m128 mm_complex_abs2(__m128 const &a){ // 0, 1, 2, 3 -> 0^2 + 1^2, 0^2 + 1^2, 2^2 + 3^2, 2^2 + 3^2 (this is abs^2)
    auto d = mm_hadd(a * a);
    return mm_complex_two1(d);
}
inline __m128 mm_complex_abs(__m128 const &a){ // 0, 1, 2, 3 -> abs(0, 1), abs(0, 1), abs(2, 3), abs(2, 3)
    return mm_sqrt(mm_complex_abs2(a));
}
inline __m128 mm_complex_div(__m128 const &a, __m128 const &b){
    return mm_complex_mul(a, b / mm_complex_conj(mm_complex_abs2(b)));
}
inline __m128 mm_complex_sqrt(__m128  const &a){
    return mm_sign_imag(a) * mm_sqrt( _mm_set_ps1(0.5f) * ( mm_sqrt( mm_complex_abs2(a) ) + mm_complex_conj( mm_complex_real(a) ) ) );
}

// complex<double> helpers
inline __m128d mm_complex_real(__m128d const &a){ return _mm_shuffle_pd(a, a, 0); }
inline __m128d mm_complex_imag(__m128d const &a){ return _mm_shuffle_pd(a, a, 3); }
inline __m128d mm_complex_swap(__m128d const &a){ return _mm_shuffle_pd(a, a, 1); }
inline __m128d mm_sign(__m128d const &a){
    return _mm_or_pd(_mm_set1_pd(1.0), _mm_and_pd(_mm_set1_pd(-0.0), a));
}
inline __m128d mm_sign_imag(__m128d const &a){
    return _mm_or_pd(_mm_set1_pd(1.0), _mm_and_pd(_mm_set_pd(-0.0, 0.0), a));
}

// complex arithmetic
inline __m128d mm_abs(__m128d const &a){ return a * mm_sign(a); }
inline __m128d mm_complex_mul(__m128d const &a, __m128d const &b){
    #ifdef __FMA__
    return _mm_fmaddsub_pd( mm_complex_real(a), b, mm_complex_imag(a) * mm_complex_swap(b) );
    #else
    return _mm_addsub_pd(mm_complex_real(a) * b, mm_complex_imag(a) * mm_complex_swap(b));
    #endif
}
inline __m128d mm_complex_conj(__m128d const &a){
    return _mm_xor_pd(a, _mm_set_pd(-0.0, 0.0));
}
inline __m128d mm_complex_abs2(__m128d const &a){
    __m128d nrm = _mm_mul_pd(a, a);
    return _mm_hadd_pd(nrm, nrm);
}
inline __m128d mm_complex_abs(__m128d const &a){
    return mm_sqrt(mm_complex_abs2(a));
}
inline __m128d mm_complex_div(__m128d const &a, __m128d const &b){
    return mm_complex_mul(a, b / mm_complex_conj(mm_complex_abs2(b)));
}
inline __m128d mm_complex_sqrt(__m128d  const &a){
    return mm_sign_imag(a) * mm_sqrt( mm_set<double, regtype::sse>(0.5) * ( mm_sqrt( mm_complex_abs2(a) ) + mm_complex_conj( mm_complex_real(a) ) ) );
}

#endif // __SSE3__

#ifdef __AVX__

// common I/O, set zero, set one, load/store
template<typename T> struct mm_zero<T, regtype::avx, std::enable_if_t<is_pfloat<T>::value>>{
    static inline auto get(){ return _mm256_setzero_ps(); }
};
template<typename T> struct mm_zero<T, regtype::avx, std::enable_if_t<is_pdouble<T>::value>>{
    static inline auto get(){ return _mm256_setzero_pd(); }
};

template<typename T> struct mm_load<T, regtype::avx, std::enable_if_t<is_pfloat<T>::value>>{
    static inline auto align(T const *a){ return _mm256_load_ps(pconvert(a)); }
    static inline auto unali(T const *a){ return _mm256_loadu_ps(pconvert(a)); }
};
template<typename T> struct mm_load<T, regtype::avx, std::enable_if_t<is_pdouble<T>::value>>{
    static inline auto align(T const *a){ return _mm256_load_pd(pconvert(a)); }
    static inline auto unali(T const *a){ return _mm256_loadu_pd(pconvert(a)); }
};

template<typename T> struct mm_store<T, regtype::avx, std::enable_if_t<is_pfloat<T>::value>>{
    static inline void align(T *a, __m256 const &b){ _mm256_store_ps(pconvert(a), b); }
    static inline void unali(T *a, __m256 const &b){ _mm256_storeu_ps(pconvert(a), b); }
};
template<typename T> struct mm_store<T, regtype::avx, std::enable_if_t<is_pdouble<T>::value>>{
    static inline void align(T *a, __m256d const &b){ _mm256_store_pd(pconvert(a), b); }
    static inline void unali(T *a, __m256d const &b){ _mm256_storeu_pd(pconvert(a), b); }
};

template<> inline auto mm_set<float,  regtype::avx>(float  a){ return _mm256_set_ps(a, a, a, a, a, a, a, a); }
template<> inline auto mm_set<double, regtype::avx>(double a){ return _mm256_set_pd(a, a, a, a); }

template<> inline auto mm_set<std::complex<float>,  regtype::avx>(std::complex<float>  a){
    return _mm256_setr_ps(a.real(), a.imag(), a.real(), a.imag(), a.real(), a.imag(), a.real(), a.imag());
}
template<> inline auto mm_set<std::complex<double>, regtype::avx>(std::complex<double> a){
    return _mm256_setr_pd(a.real(), a.imag(), a.real(), a.imag());
}


// arithmetic operations
inline __m256  mm_sqrt(__m256  const &a){ return _mm256_sqrt_ps(a); }
inline __m256d mm_sqrt(__m256d const &a){ return _mm256_sqrt_pd(a); }
#ifdef __FMA__
inline __m256  mm_fmadd(__m256  const &a, __m256  const &b, __m256  const &c){ return _mm256_fmadd_ps(a, b, c); }
inline __m256d mm_fmadd(__m256d const &a, __m256d const &b, __m256d const &c){ return _mm256_fmadd_pd(a, b, c); }
#endif

// complex<float> helpers (same as 128 case)
inline __m256 mm_complex_real(__m256 const &a){ return _mm256_permute_ps(a, 160); }
inline __m256 mm_complex_imag(__m256 const &a){ return _mm256_permute_ps(a, 245); }
inline __m256 mm_complex_swap(__m256 const &a){ return _mm256_permute_ps(a, 177); }
inline __m256 mm_complex_two1(__m256 const &a){ return _mm256_permute_ps(a, 240); }

inline __m256 mm_hadd(__m256 const &a, __m256 const &b){ return _mm256_hadd_ps(a, b); }

inline __m256 mm_sign(__m256 const &a){
    return _mm256_or_ps(_mm256_set1_ps(1.0f), _mm256_and_ps(_mm256_set1_ps(-0.0f), a));
}
inline __m256 mm_sign_imag(__m256 const &a){
    return _mm256_or_ps(_mm256_set1_ps(1.0f), _mm256_and_ps(_mm256_set_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f), a));
}

// complex<float> arithmetic
inline __m256 mm_abs(__m256 const &a){ return a * mm_sign(a); }
inline __m256 mm_complex_mul(__m256 const &a, __m256 const &b){
    return _mm256_addsub_ps( mm_complex_real(a) * b, mm_complex_imag(a) * mm_complex_swap(b));
}
inline __m256 mm_complex_conj(__m256 const &a){
    return a * _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
}
inline __m256 mm_complex_abs2(__m256 const &a){
    auto m = a * a;
    auto d = mm_hadd(m, m);
    return mm_complex_two1(d);
}
inline __m256 mm_complex_abs(__m256 const &a){
    return mm_sqrt(mm_complex_abs2(a));
}
inline __m256 mm_complex_div(__m256 const &a, __m256 const &b){
    return mm_complex_mul(a, b / mm_complex_conj(mm_complex_abs2(b)));
}
inline __m256 mm_complex_sqrt(__m256  const &a){
    return mm_sign_imag(a) * mm_sqrt( ( mm_set<float, regtype::avx>(0.5) * ( mm_sqrt( mm_complex_abs2(a) ) + mm_complex_conj( mm_complex_real(a) ) ) ) );
}

// complex<double> helpers (same as 128 case)
inline __m256d mm_complex_real(__m256d const &a){ return _mm256_permute_pd(a, 0); }
inline __m256d mm_complex_imag(__m256d const &a){ return _mm256_permute_pd(a, 15); }
inline __m256d mm_complex_swap(__m256d const &a){ return _mm256_permute_pd(a, 5); }

inline __m256d mm_hadd(__m256d const &a, __m256d const &b){ return _mm256_hadd_pd(a, b); }

inline __m256d mm_sign(__m256d const &a){
    return _mm256_or_pd(_mm256_set1_pd(1.0), _mm256_and_pd(_mm256_set1_pd(-0.0), a));
}
inline __m256d mm_sign_imag(__m256d const &a){
    return _mm256_or_pd(_mm256_set1_pd(1.0), _mm256_and_pd(_mm256_set_pd(-0.0, 0.0, -0.0, 0.0), a));
}

// complex<double> arithmetic (same as 128) case
inline __m256d mm_abs(__m256d const &a){ return a * mm_sign(a); }
inline __m256d mm_complex_mul(__m256d const &a, __m256d const &b){
    return _mm256_addsub_pd( mm_complex_real(a) * b, mm_complex_imag(a) * mm_complex_swap(b) );
}
inline __m256d mm_complex_conj(__m256d const &a){
    return a * _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
}
inline __m256d mm_complex_abs2(__m256d const &a){
    auto nrm = a * a;
    return mm_hadd(nrm, nrm);
}
inline __m256d mm_complex_abs(__m256d const &a){
    return mm_sqrt(mm_complex_abs2(a));
}
inline __m256d mm_complex_div(__m256d const &a, __m256d const &b){
    return mm_complex_mul(a, b / mm_complex_conj(mm_complex_abs2(b)));
}
inline __m256d mm_complex_sqrt(__m256d const &a){
    return mm_sign_imag(a) * mm_sqrt( mm_set<double, regtype::avx>(0.5) * ( mm_sqrt( mm_complex_abs2(a) ) + mm_complex_conj( mm_complex_real(a) ) ) );
}

#endif // __AVX__

#ifdef __AVX512F__

// common I/O, set zero, set one, load/store
template<typename T> struct mm_zero<T, regtype::avx512, std::enable_if_t<is_pfloat<T>::value>>{
    static inline auto get(){ return _mm512_setzero_ps(); }
};
template<typename T> struct mm_zero<T, regtype::avx512, std::enable_if_t<is_pdouble<T>::value>>{
    static inline auto get(){ return _mm512_setzero_pd(); }
};

template<typename T> struct mm_load<T, regtype::avx512, std::enable_if_t<is_pfloat<T>::value>>{
    static inline auto align(T const *a){ return _mm512_load_ps(pconvert(a)); }
    static inline auto unali(T const *a){ return _mm512_loadu_ps(pconvert(a)); }
};
template<typename T> struct mm_load<T, regtype::avx512, std::enable_if_t<is_pdouble<T>::value>>{
    static inline auto align(T const *a){ return _mm512_load_pd(pconvert(a)); }
    static inline auto unali(T const *a){ return _mm512_loadu_pd(pconvert(a)); }
};

template<typename T> struct mm_store<T, regtype::avx512, std::enable_if_t<is_pfloat<T>::value>>{
    static inline void align(T *a, __m512 const &b){ _mm512_store_ps(pconvert(a), b); }
    static inline void unali(T *a, __m512 const &b){ _mm512_storeu_ps(pconvert(a), b); }
};
template<typename T> struct mm_store<T, regtype::avx512, std::enable_if_t<is_pdouble<T>::value>>{
    static inline void align(T *a, __m512d const &b){ _mm512_store_pd(pconvert(a), b); }
    static inline void unali(T *a, __m512d const &b){ _mm512_storeu_pd(pconvert(a), b); }
};

template<> inline auto mm_set<float,  regtype::avx512>(float  a){ return _mm512_set1_ps(a); }
template<> inline auto mm_set<double, regtype::avx512>(double a){ return _mm512_set1_pd(a); }

template<> inline auto mm_set<std::complex<float>,  regtype::avx512>(std::complex<float>  a){
    return _mm512_setr4_ps(a.real(), a.imag(), a.real(), a.imag());
}
template<> inline auto mm_set<std::complex<double>, regtype::avx512>(std::complex<double> a){
    return _mm512_setr4_pd(a.real(), a.imag(), a.real(), a.imag());
}

// arithmetic operations
inline __m512  mm_sqrt(__m512  const &a){ return _mm512_sqrt_ps(a); }
inline __m512d mm_sqrt(__m512d const &a){ return _mm512_sqrt_pd(a); }
inline __m512  mm_fmadd(__m512  const &a, __m512  const &b, __m512  const &c){ return _mm512_fmadd_ps(a, b, c); }
inline __m512d mm_fmadd(__m512d const &a, __m512d const &b, __m512d const &c){ return _mm512_fmadd_pd(a, b, c); }

// // complex<float> helpers (same as 128 case)
inline __m512 mm_complex_real(__m512 const &a){ return _mm512_permute_ps(a, 160); } // r i, r i, ... becomes r r, r r ...
inline __m512 mm_complex_imag(__m512 const &a){ return _mm512_permute_ps(a, 245); } // r i, r i, ... becomes i i, i i ...
inline __m512 mm_complex_swap(__m512 const &a){ return _mm512_permute_ps(a, 177); } // r i, r i, ... becomes i r, i r ...
inline __m512 mm_complex_two1(__m512 const &a){ return _mm512_permute_ps(a, 240); } // 0, 1, 2, 3 -> 0, 0, 3, 3, used after hadd


inline __m512 mm_sign(__m512 const &a){
    //#ifdef __AVX512DQ__
    //return _mm512_or_ps(_mm512_set1_ps(1.0f), _mm512_and_ps(_mm512_set1_ps(-0.0f), a));
    return a / _mm512_abs_ps(a);

}
inline __m512 mm_sign_imag(__m512 const &a){
    //#ifdef __AVX512DQ__
    //return _mm512_or_ps(_mm512_set1_ps(1.0f), _mm512_and_ps(_mm512_setr4_ps(-0.0f, 0.0f, -0.0f, 0.0f), a));
    return _mm512_mask_abs_ps( a, 43690, a ) / a;
}

// complex<float> arithmetic
inline __m512 mm_abs(__m512 const &a){ return _mm512_abs_ps(a); }
inline __m512 mm_complex_mul(__m512 const &a, __m512 const &b){
    return _mm512_fmaddsub_ps( mm_complex_real(a), b, mm_complex_imag(a) * mm_complex_swap(b));
}
inline __m512 mm_complex_conj(__m512 const &a){ // should probably do xor here
    return a * _mm512_setr4_ps(1.0f, -1.0f, 1.0f, -1.0f);
}
inline __m512 mm_complex_abs2(__m512 const &a){
    auto m = a * a;
    return mm_complex_real(m) + mm_complex_imag(m);
}
inline __m512 mm_complex_abs(__m512 const &a){
    return mm_sqrt(mm_complex_abs2(a));
}
inline __m512 mm_complex_div(__m512 const &a, __m512 const &b){
    return mm_complex_mul(a, b / mm_complex_conj(mm_complex_abs2(b)));
}
inline __m512 mm_complex_sqrt(__m512  const &a){
    return mm_sign_imag(a) * mm_sqrt( ( _mm512_set1_ps(0.5f) * ( mm_sqrt( mm_complex_abs2(a) ) + mm_complex_conj( mm_complex_real(a) ) ) ) );
}

// // complex<double> helpers (same as 128 case)
inline __m512d mm_complex_real(__m512d const &a){ return _mm512_permute_pd(a, 0); }
inline __m512d mm_complex_imag(__m512d const &a){ return _mm512_permute_pd(a, 255); }
inline __m512d mm_complex_swap(__m512d const &a){ return _mm512_permute_pd(a, 85); }

inline __m512d mm_sign(__m512d const &a){
    //#ifdef __AVX512DQ__
    //return _mm512_or_pd(_mm512_set1_pd(1.0), _mm512_and_pd(_mm512_set1_pd(-0.0), a));
    return a / _mm512_abs_pd(a);
}
inline __m512d mm_sign_imag(__m512d const &a){
    //#ifdef __AVX512DQ__
    //return _mm512_or_pd(_mm512_set1_pd(1.0), _mm512_and_pd(_mm512_setr4_pd(-0.0, 0.0, -0.0, 0.0), a));
    return _mm512_mask_abs_pd( a, 170, a ) / a;
}

// // complex<double> arithmetic (same as 128) case
inline __m512d mm_abs(__m512d const &a){ return _mm512_abs_pd(a); }
inline __m512d mm_complex_mul(__m512d const &a, __m512d const &b){
    return _mm512_fmaddsub_pd( mm_complex_real(a), b, mm_complex_imag(a) * mm_complex_swap(b) );
}
inline __m512d mm_complex_conj(__m512d const &a){ // should probably do xor here
    return a * _mm512_setr4_pd(1.0, -1.0, 1.0, -1.0);
}
inline __m512d mm_complex_abs2(__m512d const &a){
    auto m = a * a;
    return mm_complex_real(m) + mm_complex_imag(m);
}
inline __m512d mm_complex_abs(__m512d const &a){
    return mm_sqrt(mm_complex_abs2(a));
}
inline __m512d mm_complex_div(__m512d const &a, __m512d const &b){
    return mm_complex_mul(a, b / mm_complex_conj(mm_complex_abs2(b)));
}
inline __m512d mm_complex_sqrt(__m512d const &a){
    return mm_sign_imag(a) * mm_sqrt( _mm512_set1_pd(0.5) * ( mm_sqrt( mm_complex_abs2(a) ) + mm_complex_conj( mm_complex_real(a) ) ) );
}

#endif


#endif // __HALA_DOXYGEN_SKIP

}

#endif
