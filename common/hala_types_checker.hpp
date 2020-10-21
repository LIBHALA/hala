#ifndef __HALA_TYPES_CHECKER_HPP
#define __HALA_TYPES_CHECKER_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_types_caster.hpp"

/*!
 * \internal
 * \file hala_types_checker.hpp
 * \brief HALA templates to assert type safely and const correctness.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACHECKTYPES
 *
 * Contains a series of templates that are used for type safety
 * and const correctness.
 * \endinternal
*/

namespace hala{

/*!
 * \internal
 * \ingroup HALACORE
 * \addtogroup HALACHECKTYPES Type Consistency Checks
 *
 * The BLAS and LAPACK standards use different names to select the precision of the method,
 * e.g., float or double. Furthermore, little or no verification is performed to ensure
 * that all input types are consistent, which is particularly frustrating when inputs can be
 * automatically converted. In contrast, HALA provides a consistent naming convention,
 * types correctness is checked at compile time, and automatic conversion is done whenever reasonable.
 * For example:
 * \code
 *      hala::axpy(-1.0, x, y); // computes y = y - x
 * \endcode
 * The method will fail to compile if the types of x and y do not match, but will work
 * regardless whether x and y have entries that are float, double, std::complex<float>,
 * and std::complex<double> even though -1.0 is technically a double.
 *
 * HALA achieves the type correctness goals using a series of templates and overloads
 * some of which (internally) stretch the limits of type safety and sanity.
 * This module build upon \ref HALACUSTOM.
 * \endinternal
 */

/*!
 * \ingroup HALACUSTOM
 * \brief Alias to the scalar type of a vector-like class.
 *
 * Helper template that accepts a vector-like class and aliases to the scalar type,
 * e.g., double for std::vector<double>.
 */
template<class VectorLike> using get_scalar_type = typename define_type<std::remove_reference_t<VectorLike>>::value_type;


/*!
 * \ingroup HALACUSTOM
 * \brief Alias to the standard equivalent for the scalar type of a vector-like class.
 *
 * Helper template that accepts a vector-like class and aliases to the equivalent standard scalar type,
 * e.g., if using a custom double precision complex type the template will alias to std::complex<double>.
 * If the vector is already using one of the C++ standard types, this is equivalent to hala::get_scalar_type.
 *
 * The standard type can always be used in algebraic expressions since all arithmetic operators are overloaded.
 */
template<class VectorLike> using get_standard_type = typename define_standard_type<typename define_type<VectorLike>::value_type>::value_type;


/*!
 * \ingroup HALACUSTOM
 * \brief Alias to the base precision type of a vector, e.g., \b float and \b std::complex<float> map to float.
 *
 * Helper template that accepts a vector-like class and aliases to the equivalent precision or real scalar type,
 * e.g., both float and std::complex<float> will alias to float.
 */
template<class VectorLike> using get_precision_type = typename define_standard_precision<typename define_type<VectorLike>::value_type>::value_type;


/*!
 * \ingroup HALACHECKTYPES
 * \brief Using reinterpret_cast, get a raw-array of type matching the base.
 */
template<class VectorLike>
auto get_standard_data(VectorLike &x){
    using standard_type  = get_standard_type<VectorLike>;
    return reinterpret_cast<standard_type*>(get_data(x));
}

/*!
 * \ingroup HALACHECKTYPES
 * \brief Using reinterpret_cast, get a raw-array of type matching the base (const overload).
 */
template<class VectorLike>
auto get_standard_data(VectorLike const &x){
    using standard_type  = get_standard_type<VectorLike>;
    return reinterpret_cast<standard_type const*>(get_data(x));
}


/*!
 * \ingroup HALACHECKTYPES
 * \brief Cast a number into a new type, or copy if the type is the same.
 */
template<typename T, typename U> inline T get_cast(U x){ return number_caster<T, U>::cast(x); }

/*!
 * \ingroup HALACHECKTYPES
 * \brief Return true if \b get_size(x) is no less than \b required_size, also calls static cast to \b size_t to avoid extraneous warnings.
 */
template<class VectorLike, typename SizeType>
bool check_size(VectorLike &x, SizeType required_size){ return (get_size(x) >= static_cast<size_t>(required_size)); }

/*!
 * \ingroup HALACHECKTYPES
 * \brief Return true if \b get_size(x) is no less than the product of \b required_sizeA and \b required_sizeB, also calls static cast to \b size_t to avoid extraneous warnings.
 */
template<class VectorLike, typename SizeTypeA, typename SizeTypeB>
bool check_size(VectorLike &x, SizeTypeA required_sizeA, SizeTypeB required_sizeB){
    return (get_size(x) >= hala_size(required_sizeA, required_sizeB));
}

/*!
 * \ingroup HALACHECKTYPES
 * \brief If \b x is strictly used as output and if the size is insufficient, then resize it; otherwise call assert check_size().
 *
 * In many cases a vector is used purely as output, e.g., when calling hala::gemm() with beta parameter equal to zero.
 * If the input entries of the vector will not be used, then the vector is strictly an output,
 * even if the vector has insufficient size it can be simply resized.
 * If the vector is not strictly and output, then this is equivalent to hala::check_size().
 */
template<class VectorLike, typename SizeType>
inline void check_set_size(bool strict_output, VectorLike &x, SizeType required_size){
    if (strict_output){
        if (get_size(x) < static_cast<size_t>(required_size)) set_size(static_cast<size_t>(required_size), x);
    }else{
        assert( check_size(x, required_size) );
    }
}

/*!
 * \ingroup HALACHECKTYPES
 * \brief Overload using required size as the product of two numbers.
 *
 * Identical to hala::check_set_size() with hala::hala_size() in place of the required size.
 */
template<class VectorLike, typename SizeTypeA, typename SizeTypeB>
inline void check_set_size(bool strict_output, VectorLike &x, SizeTypeA required_sizeA, SizeTypeB required_sizeB){
    check_set_size(strict_output, x, hala_size(required_sizeA, required_sizeB));
}

/*!
 * \ingroup HALACHECKTYPES
 * \brief Marks a variable as output only, allows for expressive calls to \b check_set_size().
 */
constexpr bool assume_output = true;

/*!
 * \ingroup HALACHECKTYPES
 * \brief If \b get_size(x) is not equal to the \b required_size, then call \b hala::set_size().
 */
template<class VectorLike, typename SizeType>
void force_size(SizeType required_size, VectorLike &x){
    if (get_size(x) != static_cast<size_t>(required_size)) set_size(static_cast<size_t>(required_size), x);
}

/*!
 * \ingroup HALACHECKTYPES
 * \brief Defines \b value to be true if \b T and \b U are compatible, e.g., complex with double precision.
 */
template<typename T, typename U, typename = void> struct is_compatible : std::false_type{};

/*!
 * \ingroup HALACHECKTYPES
 * \brief Specialization for the true case.
 */
template<typename T, typename U>
struct is_compatible<T, U, std::enable_if_t<
        (is_float<T>::value && is_float<U>::value) ||
        (is_double<T>::value && is_double<U>::value) ||
        (is_fcomplex<T>::value && is_fcomplex<U>::value) ||
        (is_dcomplex<T>::value && is_dcomplex<U>::value)
    >> : std::true_type{};


/*!
 * \ingroup HALACHECKTYPES
 * \brief Static asserts that the value_type of \b VectorLike is one of float, double, complex<float>, complex<double>.
 */
template<class VectorLike> void check_types(VectorLike const &){
    using scalar_type = get_scalar_type<VectorLike>;
    static_assert(is_float<scalar_type>::value ||
                  is_double<scalar_type>::value ||
                  is_fcomplex<scalar_type>::value ||
                  is_dcomplex<scalar_type>::value,
                  "HALA BLAS and LAPACK templates can only be instantiated with float, double, complex<float> and complex<double>");
}

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Terminates variadric recustion, runs a static assert that the types of \b VectorLike1 and \b VectorLike2 match.
 */
template<class VectorLike1, class VectorLike2> void match_types(VectorLike1 const &, VectorLike2 const &){
    using x = get_scalar_type<VectorLike1>;
    using y = get_scalar_type<VectorLike2>;
    static_assert(is_compatible<x, y>::value, "vector value mismatch");
}

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Effectively instantiates \b match_types with \b VectorLike1 and every vector type \b VectorLike2, \b VectorLike3 and \b MoreVectorLikes.
 */
template<class VectorLike1, class VectorLike2, class VectorLike3, class... MoreVectorLikes>
void match_types(VectorLike1 const &x, VectorLike2 const &y, VectorLike3 const &z, MoreVectorLikes const &... more_vecs){
    match_types(x, y);
    match_types(x, z, more_vecs...);
}

/*!
 * \ingroup HALACHECKTYPES
 * \brief Calls \b check_vector on \b VectorLike and then uses \b match_types() to recursively check is all vectors in \b MoreVectorLikes match.
 */
template<class VectorLike, class... MoreVectorLikes>
void check_types(VectorLike const &x, MoreVectorLikes const &...more_vecs){
    check_types(x);
    match_types(x, more_vecs...);
}

/*!
 * \ingroup HALACHECKTYPES
 * \brief Terminates variadric recustion, runs a static assert that the types of \b VectorLike is \b int.
 */
template<class VectorLike>
void check_types_int(VectorLike const &){
    static_assert(std::is_same<get_scalar_type<VectorLike>, int>::value,
                  "LAPACK and cuSparse indexing works only with int vectors");
}

/*!
 * \ingroup HALACHECKTYPES
 * \brief Checks with \b static_assert if every vector has type \b int.
 */
template<class VectorLike1, class VectorLike2, class... MoreVectorLikes>
void check_types_int(VectorLike1 const &x, VectorLike2 const &y, MoreVectorLikes const &... more_vecs){
    check_types_int(x);
    check_types_int(y, more_vecs...);
}


/*!
 * \ingroup HALACHECKTYPES
 * \brief Convert the pointer using \b hala::convert_pointer.
 */
template<typename T>
convert_pointer<T> pconvert(T *pntr){ return convert_pointer<T>(pntr); }

/*!
 * \ingroup HALACHECKTYPES
 * \brief Convert the pointer using \b hala::convert_pointer.
 */
template<class VectorLike>
auto convert(VectorLike &x){ return pconvert(get_data(x)); }

/*!
 * \ingroup HALACHECKTYPES
 * \brief Convert the non-const pointer using \b hala::convert_pointer.
 */
template<class VectorLike>
auto cconvert(VectorLike &&x){
    static_assert(!std::is_const< std::remove_pointer_t< decltype( get_data(x) ) > >::value,
                  "Container with const data cannot be used as output!");
    return pconvert(get_data(x));
}

/*!
 * \ingroup HALACHECKTYPES
 * \brief Defines \b value \b = \b true if the type \b T is \b std::complex<float> or \b std::complex<double>.
 */
template<typename T> struct is_complex{
    //! \brief Identifies whether the type \b T defines a complex number (pointers are not complex numbers).
    static const bool value = (is_fcomplex<T>::value || is_dcomplex<T>::value);
};

/*!
 * \ingroup HALACHECKTYPES
 * \brief Performs \b static_assert on T, permits int in addition to the floating point types.
 */
template<typename T> void check_gpu_type(){
    static_assert(is_float<T>::value || is_double<T>::value || is_fcomplex<T>::value || is_dcomplex<T>::value ||
                  std::is_same<T, int>::value, "HALA GPU vectors and array wrappers work only with int, float, double, or complex<float/double> types");
}

/*!
 * \ingroup HALACHECKTYPES
 * \brief If given a pointer, simply use the pointer.
 */
template<typename T> auto get_pointer(T const *p){ return pconvert(p); }
/*!
 * \ingroup HALACHECKTYPES
 * \brief If given a value, convert it to a host pointer.
 */
template<typename T, typename U>
std::enable_if_t< !std::is_pointer<U>::value, gpu_pconvert<T>> get_pointer(U x){
    return gpu_pconvert<T>{get_cast<T>(x)};
}

/*!
 * \ingroup HALACHECKTYPES
 * \brief If \b beta is not a pointer, then check if it is zero and resize C if necessary.
 *
 * Resize \b C to \b M by \b N if beta is zero, i.e., \b C is pure output.
 * Handles the case when \b beta is not a pointer and can be used to compare to zero.
 */
template<typename FPB, typename VectorLike, typename IntA, typename IntB>
inline typename std::enable_if_t< !std::is_pointer<FPB>::value > pntr_check_set_size(FPB beta, VectorLike &C, IntA M, IntB N){
    check_set_size((hala_abs(beta) == 0.0), C, M, N);
}
/*!
 * \ingroup HALACHECKTYPES
 * \brief If \b beta is a pointer, do nothing.
 */
#ifndef NDEBUG
template<typename FPB, typename VectorLike, typename IntA, typename IntB>
inline typename std::enable_if_t< std::is_pointer<FPB>::value > pntr_check_set_size(FPB, VectorLike &C, IntA M, IntB N){
    assert( check_size(C, M, N) );
}
#else
template<typename FPB, typename VectorLike, typename IntA, typename IntB>
inline typename std::enable_if_t< std::is_pointer<FPB>::value > pntr_check_set_size(FPB, VectorLike, IntA, IntB){
}
#endif

/*!
 * \ingroup HALACHECKTYPES
 * \brief Struct the reads the GPU device ID associated with the \b VectorLike, result will be ignored unless specialized.
 */
template<class VectorLike>
struct deviceid_extractor{
    //! \brief Return the CUDA device associated with the vector (dummy method, does nothing).
    static int device(VectorLike const&){ return -1; }
};

/*!
 * \ingroup HALACHECKTYPES
 * \brief Returns the GPU device ID associated with the \b VectorLike, result will be ignored unless this or \b hala::deviceid_extractor are specialized.
 */
template<class VectorLike>
int get_device(VectorLike &x){ return deviceid_extractor<VectorLike>::device(x); }

}

#endif
