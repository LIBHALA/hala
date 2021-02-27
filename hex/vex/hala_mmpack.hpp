#ifndef __HALA_MMPACK_HPP
#define __HALA_MMPACK_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_aligned_allocator.hpp"

/*!
 * \internal
 * \file hala_mmpack.hpp
 * \brief HALA wrapper to the vector extension types.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAVEX
 *
 * Contains the wrappers around vector extension types used for manual vectorization.
 * \endinternal
 */

/*!
 * \defgroup HALAVEX HALA Advanced Vector Extensions
 *
 * \par SSE and AVX
 *
 * SSE and AVX refer to special register standards supported on many CPUs that allow
 * for multiple floating point operations at once. The use of the registers can
 * dramatically improve performance and while compilers strive to utilize the
 * full potential of the CPU vectorization, they often fall short.
 * Manual vectorization is doable, i.e., process a long vector in strides of
 * 2/4/8/16 entries at a time, but the implementation is cumbersome at best.
 * Gaining access to the intrinsic capabilities requires heavy use of C,
 * C-style functions, and C macros, especially when dealing with complex numbers.
 *
 * HALA provides a wrapper to the vector types and methods that uses C++ templates
 * and operator overloads, which significantly improves readability and accessibility.
 */

namespace hala{

/*!
 * \ingroup HALAVEX
 * \brief Wrapper for the aligned memory used by intrinsic methods.
 *
 * \par Wrapper to Intrinsic Methods
 * Intrinsic functions relate to the extended registers found on many processors that
 * allow multiple floating point operations to be executed simultaneously.
 * The use of these registers can yield a huge boost in performance;
 * however, utilizing the potential can be difficult.
 *
 * Intrinsic methods use multiple floating point variables that have been packed
 * in a memory location, the pack is aligned to the correct memory address and
 * can be read in/out of a CPU register in a single operation.
 * First, data has to be loaded in such packets, then the arithmetic operations
 * can be performed using a combination of operator overloads and C-style
 * of functions with somewhat cryptic names, e.g., \b _mm256_mul_ps().
 * See <a href="https://software.intel.com/sites/landingpage/IntrinsicsGuide/">
 * Intel Intrinsics Guide</a> for more details.
 *
 * HALA provides the \b mmpack struct that wraps around a \b data member which
 * has the proper size and alignment for the specified floating point type \b T.
 * The \b data can be loaded and stored from/to another memory location,
 * e.g., a std::vector, which in turn may or may-not be aligned. Each slice
 * of \b data can hold multiple floating point values, the number is defined
 * by the static constant \b stride. For example, the SSE registers have width
 * of 128 bits and can hold 4 x 32 bit numbers of type \b float,
 * or 2 x 64 bit numbers of type \b double; thus the \b stride for
 * \b T \b = \b float is 4 while the stride for \b T \b = \b std::complex<double>
 * is 1 (complex numbers require two floating point numbers).
 *
 * Arithmetic is performed on the data via expressive and convenient operator
 * overloads, which cover + - * and / for float, double, std::complex-float
 * and std::complex-double types with additional methods for
 * retrieving real/imaginary parts of numbers, absolute values and square-roots.
 *
 * The class also contains a fall-back mode of \b regtype \b none, which is nothing
 * more than a wrapper around a single variable of type \b T.
 *
 * Achieving maximum throughput of reading from/to a vector data-structure
 * requires that the vector memory is aligned to the corresponding register width.
 * Using the aligned versions of the \b load() and \b put() methods with unaligned
 * memory will lead to a segfault. The default alignment mode for each register type
 * will be determined during the CMake configuration stage, where tests will be
 * performed with instances of std::vector and the default C++ allocator.
 * If data is loaded and stored in vector-like classes with the same alignment
 * and if the data is accessed in multiples of the \b stride, the default mode will work.
 * However, in some circumstances, the compiler can decide to change
 * the alignment when dealing with vector-of-vectors or mixing vectors and other
 * data-structures and containers. The default alignment can be overwritten
 * by adding a corresponding HALA keyword (really just a constexpr variable)
 * to the \b load() and \b put() methods.
 * See also the hala::aligned_allocator which will ensure that every vector is aligned.
 *
 * Example, scaling a vector by a number using HALA intrinsic templates
 * \code
 * template<typename T, hala::regtype R>
 * void scale(typename std::vector<T>::value_type alpha, std::vector<T> &x){
 *     hala::mmpack<T, R> alpha(alpha);
 *     for(size_t i=0; i<N; i += alpha.stride){
 *         auto vx = hala::load<R>(&x[i]) * alpha;
 *         vx.put(&x[i]);
 *     }
 * }
 * \endcode
 * The example above does not outperform BLAS and a simple for-loop will be
 * optimized by the compiler to perform about the same. However, the compiler
 * struggles with std::complex numbers and many problems require non-linear
 * arithmetic that cannot be efficiently expressed with BLAS. HALA templates
 * generalize easily and can handle with efficiency complicated non-linear
 * real and complex valued operations.
 */
template<typename T, regtype R = default_regtype>
struct mmpack{
    //! \brief The type of floating point numbers stored.
    using value_type = T;
    //! \brief The type of the aligned memory.
    typedef typename vex_regtype<T, R>::mm_type mm_type;
    //! \brief The complex template parameter if T is of type std::complex, T otherwise.
    using precision_type = typename define_standard_precision<T>::value_type;
    //! \brief Defines the standard type.
    using standard_type = typename define_standard_type<T>::value_type;
    //! \brief The number of entries of type \b T stored in the aligned memory.
    static constexpr size_t stride = vex_stride<standard_type, R>::stride;

    //! \brief The only data member, represents a slice of aligned memory.
    mm_type data;

    //! \brief Default copy-constructor.
    mmpack(mmpack<T, R> const &) = default;
    //! \brief Default move-constructor.
    mmpack(mmpack<T, R> &&) = default;
    //! \brief Default copy-operator.
    mmpack<T, R>& operator = (mmpack<T, R> const &) = default;
    //! \brief Default move-operator.
    mmpack<T, R>& operator = (mmpack<T, R> &&) = default;

    //! \brief Copy-constructor from compatible type.
    template<typename U, typename = std::enable_if_t<is_compatible<T, U>::value && !std::is_same<T, U>::value>>
    mmpack(mmpack<U, R> const &x) : data(x.data) {}

    //! \brief Constructor, initialize the data to zero.
    mmpack(){ data = mm_zero<standard_type, R>::get(); }
    //! \brief Constructor, initialize all entries to the given number \b a.
    template<typename std::enable_if_t<(sizeof(T), true)>* = nullptr>
    mmpack(T const a) : data(mm_set<standard_type, R>(get_cast<standard_type>(a))){}
    //! \brief Constructor, initialize all entries to the given number \b a with different type.
    template<typename U, typename std::enable_if_t<!std::is_same<T, U>::value && !std::is_pointer<U>::value>* = nullptr>
    mmpack(U const a) : data(mm_set<standard_type, R>(get_cast<standard_type>(a))) {}
    //! \brief Constructor, read from a memory location using the default alignment mode.
    mmpack(T const *a) : data((default_alignment<T, R>::aligned) ? mm_load<T, R>::align(a) : mm_load<T, R>::unali(a)) {}
    //! \brief Constructor, read from a memory location using bad alignment mode.
    mmpack(T const *a, define_unaligned) : data(mm_load<T, R>::unali(a)) {}
    //! \brief Constructor, read from a memory location using good alignment mode.
    mmpack(T const *a, define_aligned) : data(mm_load<T, R>::align(a)) {}
    //! \brief Constructor, read from an aligned vector.
    template<class vec>
    mmpack(aligned_iterator<T, R, vec> const &a) : data(mm_load<T, R>::align(&*a)) {}

    //! \brief Constructor, copy from another slice of memory.
    mmpack(mm_type const &a) : data(a){}

    //! \brief The struct can be used interchangeably with the \b data and passed directly to the intrinsic methods.
    operator mm_type (){ return data; }
    //! \brief The struct can be used interchangeably with the \b data and passed directly to the intrinsic methods.
    operator mm_type const () const{ return data; }

    //! \brief Addition overload.
    template<typename dummy = int>
    std::enable_if_t<(sizeof(dummy), R != regtype::none or is_complex<T>::value), mmpack<T, R>>
    operator + (mmpack<T, R> const &other) const{ return data + other.data; }
    //! \brief Subtraction overload.
    template<typename dummy = int>
    std::enable_if_t<(sizeof(dummy), R != regtype::none or is_complex<T>::value), mmpack<T, R>>
    operator - (mmpack<T, R> const &other) const{ return data - other.data; }
    //! \brief Multiplication overload, also handles complex numbers.
    template<typename dummy = int>
    std::enable_if_t<(sizeof(dummy), R != regtype::none or is_complex<T>::value), mmpack<T, R>>
    operator * (mmpack<T, R> const &other) const{
        if (is_complex<T>::value)
            return mm_complex_mul(data, other.data);
        else
            return data * other.data;
    }
    //! \brief Division overload, also handles complex numbers.
    template<typename dummy = int>
    std::enable_if_t<(sizeof(dummy), R != regtype::none or is_complex<T>::value), mmpack<T, R>>
    operator / (mmpack<T, R> const &other) const{
        if (is_complex<T>::value)
            return mm_complex_div(data, other.data);
        else
            return data / other.data;
    }
    //! \brief Increment overload.
    mmpack<T, R> operator += (mmpack<T, R> const &other){
        data += other.data;
        return *this;
    }
    //! \brief Decrement overload.
    mmpack<T, R> operator -= (mmpack<T, R> const &other){
        data -= other.data;
        return *this;
    }
    //! \brief Scale overload.
    mmpack<T, R> operator *= (mmpack<T, R> const &other){
        if (is_complex<T>::value)
            data = mm_complex_mul(data, other.data);
        else
            data *= other.data;
        return *this;
    }
    //! \brief Scale-divide overload.
    mmpack<T, R> operator /= (mmpack<T, R> const &other){
        if (is_complex<T>::value)
            data = mm_complex_div(data, other.data);
        else
            data /= other.data;
        return *this;
    }

    //! \brief Fused multiply add (+= a * b), if the instruction is available.
    void fmadd(mmpack<T, R> const &a, mmpack<T, R> const &b){
        if (is_complex<T>::value)
            data += mm_complex_mul(a.data, b.data);
        else
            data = mm_fmadd(a.data, b.data, data);
    }

    //! \brief Return the square-root of the numbers, also handles complex numbers.
    mmpack<T, R> sqrt() const{
        if (is_complex<T>::value)
            return mm_complex_sqrt(data);
        else
            return mm_sqrt(data);
    }

    //! \brief Replace the internal data with the square-root of the data.
    void set_sqrt(){
        if (is_complex<T>::value)
            data = mm_complex_sqrt(data);
        else
            data = mm_sqrt(data);
    }

    //! \brief Return the absolute-value-squared (i.e., without the square-root), also handles complex numbers.
    mmpack<T, R> abs2() const{
        if (is_complex<T>::value)
            return mm_complex_abs2(data);
        else
            return data * data;
    }

    //! \brief Return the absolute-value of the numbers, also handles complex numbers.
    mmpack<T, R> abs() const{
        if (is_complex<T>::value)
            return mm_complex_abs(data);
        else
            return mm_abs(data);
    }

    //! \brief Return the conjugate of the numbers.
    mmpack<T, R> conj() const{
        if (is_complex<T>::value)
            return mm_complex_conj(data);
        else
            return data;
    }

    //! \brief Load from the pointer using the default alignment.
    void load(T const* a){
        if (default_alignment<T, R>::aligned){
            data = mm_load<T, R>::align(a);
        }else{
            data = mm_load<T, R>::unali(a);
        }
    }
    //! \brief Load from the pointer, assume bad alignment.
    void load(T const* a, define_unaligned){ data = mm_load<T, R>::unali(a); }
    //! \brief Load from the pointer, assume good alignment.
    void load(T const* a, define_aligned){ data = mm_load<T, R>::align(a); }

    //! \brief Put the \b data into \b a, assume default alignment.
    void put(T *a) const{
        if (default_alignment<T, R>::aligned){
            mm_store<T, R>::align(a, data);
        }else{
            mm_store<T, R>::unali(a, data);
        }
    }
    //! \brief Put the \b data into \b a, assume good alignment.
    void put(T *a, define_aligned) const{ mm_store<T, R>::align(a, data); }
    //! \brief Put the \b data into \b a, assume bad alignment.
    void put(T *a, define_unaligned) const{ mm_store<T, R>::unali(a, data); }

    //! \brief Return the real part of the \b i-th number.
    template<typename IndexType, typename std::enable_if_t<(sizeof(IndexType), R != regtype::none)>* = nullptr>
    auto real(IndexType i){
        if (is_complex<T>::value)
            return data[static_cast<size_t>(2*i)];
        else
            return data[static_cast<size_t>(i)];
    }
    //! \brief Return the complex part of the \b i-th number.
    template<typename IndexType, typename std::enable_if_t<(sizeof(IndexType), R != regtype::none)>* = nullptr>
    auto imag(IndexType i){
        if (is_complex<T>::value)
            return data[static_cast<size_t>(2*i + 1)];
        else
            return std::imag(static_cast<T>(0));
    }

    //! \brief Return the real part of the single stored number.
    template<typename IndexType, typename std::enable_if_t<(sizeof(IndexType), R == regtype::none)>* = nullptr>
    auto real(IndexType){ return std::real(data); }
    //! \brief Return the complex part of the single stored number.
    template<typename IndexType, typename std::enable_if_t<(sizeof(IndexType), R == regtype::none)>* = nullptr>
    auto imag(IndexType){ return std::imag(data); }

    //! \brief Load from the pointer \b a using default alignment.
    mmpack<T, R>& operator = (T const *a){ load(a); return *this; }
    //! \brief Set all \b data entries to \b a.
    template<typename std::enable_if_t<(sizeof(T),true)>* = nullptr>
    mmpack<T, R>& operator = (T a){ data = mm_set<T, R>(a); return *this; }
    //! \brief Copy from another aligned memory location.
    mmpack<T, R>& operator = (mm_type const &a){ data = a; return *this; }

    //! \brief Load from an aligned vector.
    mmpack<T, R>& operator = (typename aligned_vector<T, R>::iterator const &a){ data = mm_load<T, R>::align(&*a); return *this; }
    //! \brief Load from an aligned vector.
    mmpack<T, R>& operator = (typename aligned_vector<T, R>::const_iterator const &a){ data = mm_load<T, R>::align(&*a); return *this; }

    //! \brief Returns reference to the \b i-th element.
    template<typename IndexType>
    T& operator [] (IndexType i){
        return *(reinterpret_cast<T*>(&data) + static_cast<size_t>(i));
    }
    //! \brief Returns const reference to the \b i-th element.
    template<typename IndexType>
    T const& operator [] (IndexType i) const{
        return *(reinterpret_cast<T const*>(&data) + static_cast<size_t>(i));
    }
};

/*!
 * \ingroup HALAVEX
 * \brief Outputs all entries of the vector to the stream.
 */
template<typename T, regtype R>
std::ostream& operator<<(std::ostream& os, const mmpack<T, R>& s){
    os << "(" << s[0];
    for(size_t j=1; j<s.stride; j++) os << "," << s[j];
    return (os << ")");
}

/*!
 * \ingroup HALAVEX
 * \brief Make a slice filled with zeros.
 */
template<typename T, regtype R = default_regtype>
inline mmpack<T, R> mmzero(){ return mmpack<T, R>(); }

/*!
 * \ingroup HALAVEX
 * \brief Make a slice filled with the data from \b a, multiple modes are supported.
 *
 * Creates a pack of entries using the data in \b a. Overloads are provided
 * to handle pointers, values, and iterators, and to manually specify the alignment.
 *
 * Overloads:
 * \code
 *  T const *pntr = ...;          // pointer
 *  T const value = ...;          // value
 *  auto iter = hala::mmbegin(x); // aligned iterator
 *
 *  auto mmx = hala::mmload(pntr);            // use default alignment
 *  auto mmx = hala::mmload(pntr, aligned);   // assume aligned read
 *  auto mmx = hala::mmload(pntr, unaligned); // assume unaligned read
 *  auto mmx = hala::mmload(pntr, single);    // set all pack entries to the pntr[0]
 *  auto mmx = hala::mmload(value);           // set all pack entries to the value
 *  auto mmx = hala::mmload(iter);            // read from a hala::aligned_iterator
 * \endcode
 * In all cases, \b mmx will be an instance of hala::mmpack with the specified type and register type.
 */
template<regtype R = default_regtype, typename T>
inline mmpack<T, R> mmload(T const *a){ return mmpack<T, R>(a); }


#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)

template<typename T, regtype R>
std::enable_if_t<R != regtype::none or is_complex<T>::value, mmpack<T, R>> operator +(T const a, mmpack<T, R> const &b){ return b + a; }
template<typename T, regtype R>
std::enable_if_t<R != regtype::none or is_complex<T>::value, mmpack<T, R>> operator -(T const a, mmpack<T, R> const &b){ return mmpack<T, R>(a) - b; }
template<typename T, regtype R>
std::enable_if_t<R != regtype::none or is_complex<T>::value, mmpack<T, R>> operator *(T const a, mmpack<T, R> const &b){ return b * a; }
template<typename T, regtype R>
std::enable_if_t<R != regtype::none or is_complex<T>::value, mmpack<T, R>> operator /(T const a, mmpack<T, R> const &b){ return mmpack<T, R>(a) / b; }

template<regtype R = default_regtype, typename T,
         std::enable_if_t<
                !std::is_pointer<T>::value &&
                (is_float<T>::value || is_double<T>::value || is_fcomplex<T>::value || is_dcomplex<T>::value)
            >* = nullptr>
inline mmpack<T, R> mmload(T const a){ return mmpack<T, R>(a); }

template<regtype R = default_regtype, typename T>
inline mmpack<T, R> mmload(T const *a, define_unaligned){ return mmpack<T, R>(a, unaligned); }

template<regtype R = default_regtype, typename T>
inline mmpack<T, R> mmload(T const *a, define_aligned){ return mmpack<T, R>(a, aligned); }

template<regtype R = default_regtype, typename T>
inline mmpack<T, R> mmload(T const *a, define_single){ return mmpack<T, R>(a, single); }

template<regtype R = default_regtype, typename T, class vec>
inline mmpack<T, R> mmload(aligned_iterator<T, R, vec> const &i){ return mmpack<T, R>(i); }
#endif

/*!
 * \ingroup HALAVEX
 * \brief Shortcut alias to \b mmpack<T, \b sse>.
 */
template<typename T> using mmpack_sse = mmpack<T, regtype::sse>;

/*!
 * \ingroup HALAVEX
 * \brief Shortcut alias to \b mmpack<T, \b avx>.
 */
template<typename T> using mmpack_avx = mmpack<T, regtype::avx>;

/*!
 * \ingroup HALAVEX
 * \brief Shortcut alias to \b mmpack<T, \b avx512>.
 */
template<typename T> using mmpack_avx512 = mmpack<T, regtype::avx512>;

}

#endif
