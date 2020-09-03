#ifndef __HALA_INTRINSICS_MACROS_HPP
#define __HALA_INTRINSICS_MACROS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_intrinsics.hpp"

/*!
 * \internal
 * \file hala_intrinsics_macros.hpp
 * \brief HALA macros to handle the C-style of intrinsics.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAINTRMACROS
 *
 * Macros to handle the C-style of vector extensions.
 * \endinternal
 */

/*!
 * \internal
 * \ingroup HALAVEX
 * \addtogroup HALAINTRMACROS Intrinsic Macros
 *
 * The core of the intrinsics vector extension methods is written in C,
 * which requires the use of series of macros that cause all sorts of havoc
 * when coupled with templates and "pure" C++.
 * The purpose of this module is to contain the macros to a single header and
 * to translate define pre-processor directives to constexpr default variables
 * suitable for C++ template interfacing.
 * \endinternal
 */

namespace hala{

/*!
 * \internal
 * \ingroup HALAVEX
 * \brief Returns the default register type.
 *
 * The avx512 registers require macro __AVX512F__, the avx (256bit) require __AVX__ and the sse (128-bit) require __SSE3__.
 * Note that __SSE3__ requires __SSE2__ and others.
 * If a macro is not available, some of the arithmetic operations are not supported in the current build mode and
 * the corresponding register type will be disabled.
 * If none of the macros is present, the default is hala::regtype::none.
 * \endinternal
 */
constexpr regtype get_default_regtype(){
    #ifdef __AVX512F__
    return regtype::avx512;
    #else
    #ifdef __AVX__
    return regtype::avx;
    #else
    #ifdef __SSE3__
    return regtype::sse;
    #else
    return regtype::none;
    #endif
    #endif
    #endif
}


/*!
 * \ingroup HALAVEX
 * \brief Default register type for vectorization, different for each system.
 *
 * By default, HALA will use the widest available register, which is determined form the compiler macros
 * that are in turn controlled by the compiler options, e.g., -mtune=native.
 */
constexpr regtype default_regtype = get_default_regtype();


/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines the variable used in the intrinsic functions, defaults to \b T, e.g., \b double.
 */
template<typename T, regtype, typename = void> struct vex_regtype{
    //! \brief Default uses a single variable of type \b T.
    typedef typename define_standard_type<T>::value_type mm_type;
};

/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines the stride for the vectorization, e.g., \b regtype::sse with \b double gives \b stride = 2; defaults to 1 unless specialized.
 */
template<typename,   regtype> struct vex_stride{
    //! \brief Default stride is 1.
    static constexpr size_t stride = 1;
};

/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines the size of the types, just a helper to define the stride.
 */
template<typename> constexpr size_t get_size(){ return 1; }
/*!
 * \ingroup HALAINTRMACROS
 * \brief Float is considered one unit of size.
 */
template<> constexpr size_t get_size<float>(){ return 1; }
/*!
 * \ingroup HALAINTRMACROS
 * \brief Double is two units.
 */
template<> constexpr size_t get_size<double>(){ return 2; }
/*!
 * \ingroup HALAINTRMACROS
 * \brief Complex-float float is two units.
 */
template<> constexpr size_t get_size<std::complex<float>>(){ return 2; }
/*!
 * \ingroup HALAINTRMACROS
 * \brief Complex-double if 4 units.
 */
template<> constexpr size_t get_size<std::complex<double>>(){ return 4; }

#ifdef __SSE3__
/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines __m128.
 */
template<typename T> struct vex_regtype<T,  regtype::sse, typename std::enable_if_t<is_pfloat<T>::value>>{
    //! \brief Defines 16-byte vector of floats.
    typedef __m128  mm_type;
};

/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines __m128d.
 */
template<typename T> struct vex_regtype<T, regtype::sse, typename std::enable_if_t<is_pdouble<T>::value>>{
    //! \brief Defines 16-byte vector of doubles.
    typedef __m128d mm_type;
};

/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines \b stride 4.
 */
template<typename T> struct vex_stride<T,  regtype::sse>{
    //! \brief The 128-bit registers can load 4 float values.
    static constexpr size_t stride = 4 / get_size<T>();
};

#endif
#ifdef __AVX__
/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines __m256.
 */
template<typename T> struct vex_regtype<T,  regtype::avx, typename std::enable_if_t<is_pfloat<T>::value>>{
    //! \brief Defines 32-byte vector of floats.
    typedef __m256 mm_type;
};

/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines __m256d.
 */
template<typename T> struct vex_regtype<T, regtype::avx, typename std::enable_if_t<is_pdouble<T>::value>>{
    //! \brief Defines 32-byte vector of doubles.
    typedef __m256d mm_type;
};

/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines \b stride 8.
 */
template<typename T> struct vex_stride<T,  regtype::avx>{
    //! \brief The 128-bit registers can load 4 float values.
    static constexpr size_t stride = 8 / get_size<T>();
};
#endif
#ifdef __AVX512F__
/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines __m512.
 */
template<typename T> struct vex_regtype<T,  regtype::avx512, typename std::enable_if_t<is_pfloat<T>::value>>{
    //! \brief Defines 64-byte vector of floats.
    typedef __m512 mm_type;
};

/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines __m512d.
 */
template<typename T> struct vex_regtype<T, regtype::avx512, typename std::enable_if_t<is_pdouble<T>::value>>{
    //! \brief Defines 64-byte vector of doubles.
    typedef __m512d mm_type;
};

/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines \b stride 16.
 */
template<typename T> struct vex_stride<T,  regtype::avx512>{
    //! \brief The 512-bit registers can load 4 float values.
    static constexpr size_t stride = 16 / get_size<T>();
};
#endif

/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines the type for bad alignment.
 */
struct define_unaligned{};

/*!
 * \ingroup HALAVEX
 * \brief Allows for expressive calls to hala::vex when reading/writing from/to unaligned memory.
 *
 * Example:
 * \code
 * hala::vex::mmpack<float> v1(x); // using default behavior based on known compiler features
 * hala::vex::mmpack<float> v2(y, unaligned); // y is known to have improper alignment
 * hala::vex::mmpack<float> v3(z, aligned);   // z is known to have proper alignment
 * \endcode.
 */
constexpr define_unaligned unaligned;

/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines the type for good alignment.
 */
struct define_aligned{};

/*!
 * \ingroup HALAVEX
 * \brief Allows for expressive calls to hala::vex when reading/writing from/to unaligned memory, see hala::unaligned.
 */
constexpr define_aligned aligned;

/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines the type for reading single entry and setting it to all registers.
 */
struct define_single{};

/*!
 * \ingroup HALAVEX
 * \brief Allows for expressive calls to hala::vex when reading/writing a single entry from an array.
 */
constexpr define_single single;

/*!
 * \ingroup HALAINTRMACROS
 * \brief Defines whether the \b hala::vex constructor, \b load, \b store and \b operator \b = default to aligned or unaligned mode.
 *
 * During build, CMake will attempt to detect whether the compiler defaults to vectors with 16, 32, 64 or some other alignment.
 * The information is propagated to partial specializations of this class.
 * The partial specialization can be overwritten either in CMake or with a full specialization,
 * but note that assuming the incorrect alignment usually results in \b segfault.
 */
template<typename, regtype>
struct default_alignment{
    //! \brief Assumes no alignment by default.
    static constexpr bool aligned = false;
};

/*!
 * \ingroup HALAINTRMACROS
 * \brief Partial specialization for \b regtype::sse registers.
 */
template<typename T> struct default_alignment<T, regtype::sse>{
    //! \brief Defines whether the vectors are set to 128-bit alignment, using \b regtype::sse registers.
#ifdef HALA_ALIGN128
    static constexpr bool aligned = true;
#else
    static constexpr bool aligned = false;
#endif
};

/*!
 * \ingroup HALAINTRMACROS
 * \brief Partial specialization for \b regtype::avx registers.
 */
template<typename T> struct default_alignment<T, regtype::avx>{
    //! \brief Defines whether the vectors are set to 256-bit alignment, using \b regtype::avx registers.
#ifdef HALA_ALIGN256
    static constexpr bool aligned = true;
#else
    static constexpr bool aligned = false;
#endif
};

/*!
 * \ingroup HALAINTRMACROS
 * \brief Partial specialization for \b regtype::avx512 registers.
 */
template<typename T> struct default_alignment<T, regtype::avx512>{
    //! \brief Defines whether the vectors are set to 512-bit alignment, using \b regtype::avx512 registers.
#ifdef HALA_ALIGN512
    static constexpr bool aligned = true;
#else
    static constexpr bool aligned = false;
#endif
};

}

#endif
