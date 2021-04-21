#ifndef __HALA_TYPES_HELPERS_HPP
#define __HALA_TYPES_HELPERS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_vector_defines.hpp"

/*!
 * \internal
 * \file hala_types_helpers.hpp
 * \brief HALA templates to assert type safely and const correctness.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACHECKTYPES
 *
 * Contains a series of templates that are used for type safety
 * and const correctness.
 * \endinternal
*/

/*!
 * \internal
 * \ingroup HALACORE
 * \addtogroup HALACHECKTYPESHELP Type Consistency Helper Templates
 *
 * Type consistency is done with may specializations and overloads,
 * the details are documented here while the \ref HALACHECKTYPES contains the useful wrappers.
 * \endinternal
 */

namespace hala{

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Enums to expressively indicate ownership of a pointer.
 */
enum class ptr_ownership{
    //! \brief Does own the pointer.
    own,
    //! \brief Does not own the pointer.
    not_own
};

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Struct to specialize that returns the C++ standard equivalent of each type.
 */
template<typename, typename = void> struct define_standard_type{};

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Specialization for float.
 */
template<typename scalar_type>
struct define_standard_type<scalar_type, typename std::enable_if_t<is_float<scalar_type>::value>>{
    //! \brief Equivalent to standard float.
    using value_type = float;
};

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Specialization for double.
 */
template<typename scalar_type>
struct define_standard_type<scalar_type, typename std::enable_if_t<is_double<scalar_type>::value>>{
    //! \brief Equivalent to standard double.
    using value_type = double;
};

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Specialization for float-complex.
 */
template<typename scalar_type>
struct define_standard_type<scalar_type, typename std::enable_if_t<is_fcomplex<scalar_type>::value>>{
    //! \brief Equivalent to standard float complex.
    using value_type = std::complex<float>;
};

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Specialization for double-complex.
 */
template<typename scalar_type>
struct define_standard_type<scalar_type, typename std::enable_if_t<is_dcomplex<scalar_type>::value>>{
    //! \brief Equivalent to standard double complex.
    using value_type = std::complex<double>;
};

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Specialization for index types.
 */
template<typename scalar_type>
struct define_standard_type<scalar_type, typename std::enable_if_t<std::is_integral<scalar_type>::value>>{
    //! \brief Just keep the type.
    using value_type = scalar_type;
};

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Struct to specialize that returns the precision, e.g., float for float and complex-float.
 */
template<typename, typename = void> struct define_standard_precision{};

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Specialization for float and float-complex.
 */
template<typename scalar_type>
struct define_standard_precision<scalar_type, typename std::enable_if_t<is_float<scalar_type>::value || is_fcomplex<scalar_type>::value>>{
    //! \brief Define single precision.
    using value_type = float;
};

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Specialization for double and double-complex.
 */
template<typename scalar_type>
struct define_standard_precision<scalar_type, typename std::enable_if_t<is_double<scalar_type>::value || is_dcomplex<scalar_type>::value>>{
    //! \brief Define double precision.
    using value_type = double;
};


/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Returns the last entry of the vector, e.g., useful when the last entry is the number of non-zeros.
 */
template<class VectorLike>
auto get_back(VectorLike const &x){ return get_data(x)[get_size(x)-1]; }

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Structure that wraps around a pointer and automatically casts it to any other pointer type.
 *
 * The template can be dangerous as it effectively strips away type-safety, but it allows one template
 * to wrap all function calls associated with different precision and real/complex types.
 */
template<typename T>
struct convert_pointer{
    //! \brief Constructor that sets the pointer.
    convert_pointer(T* input) : pntr(input){}
    //! \brief User conversion, non-const case, convert the pointer into a pointer of type \b F*.
    template<typename F> operator F*(){ return reinterpret_cast<F*>(pntr); }
    //! \brief User conversion, const case, convert the pointer into a pointer of type \b F*.
    template<typename F> operator F*() const{ return const_cast<F*>(reinterpret_cast<F const*>(pntr)); }
    //! \brief The pointer to be converted.
    T* pntr;
};

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Converts a CPU bound value into a host pointer.
 */
template<typename T> struct gpu_pconvert{
    //! \brief The value to be converted.
    T vals;
    //! \brief Converts the value to a const pointer that can be passed to the GPU method.
    template<typename U> operator U const* () const{ return reinterpret_cast<U const*>(&vals); }
};




#ifdef __cpp_if_constexpr
/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Enable support for if-constexpr whenever supported in the context.
 */
#define __HALA_CONSTEXPR_IF__ constexpr
#else
/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Enable support for if-constexpr whenever supported in the context.
 */
#define __HALA_CONSTEXPR_IF__
#endif

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Call one of the 4 possible functions according to the \b criteria.
 *
 * The 4 inputs Xfunc correspond to BLAS naming conventions for single, double, complex and double-complex precision.
 * The \b criteria is tested for each type using is_float() ... is_dcomplex() and the corresponding function
 * is called with \b args.
 */
template<typename criteria, typename Fsingle, typename Fdouble, typename Fcomplex, typename Fzomplex, class... Inputs>
inline void call_backend(Fsingle &sfunc, Fdouble &dfunc, Fcomplex &cfunc, Fzomplex &zfunc, Inputs &&...args){
    if __HALA_CONSTEXPR_IF__ (is_float<criteria>::value){
        sfunc(args...);
    }else if __HALA_CONSTEXPR_IF__ (is_double<criteria>::value){
        dfunc(args...);
    }else if __HALA_CONSTEXPR_IF__ (is_fcomplex<criteria>::value){
        cfunc(args...);
    }else if __HALA_CONSTEXPR_IF__ (is_dcomplex<criteria>::value){
        zfunc(args...);
    }
}

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Call one of the 2 possible functions according to the \b criteria.
 *
 * Call the first or second function depending whether \b criteria is \b float or \b double.
 */
template<typename criteria, typename Fsingle, typename Fdouble, class... Inputs>
inline void call_backend2(Fsingle &sfunc, Fdouble &dfunc, Inputs &&...args){
    if __HALA_CONSTEXPR_IF__ (is_float<criteria>::value){
        sfunc(args...);
    }else if __HALA_CONSTEXPR_IF__ (is_double<criteria>::value){
        dfunc(args...);
    }
}

/*!
 * \ingroup HALACHECKTYPESHELP
 * \brief Call one of the 6 possible functions according to the \b criteria.
 *
 * Generates two call to the backend corresponding to the symmetric vs. conjugate-symmetric (Hermitian) cases.
 * - \b flags is true, then use the first 4 functions
 * - \b flags is false, then use the first 2 and the last 2 functions
 */
template<bool flag, typename criteria, typename Fsingle, typename Fdouble,
         typename Fcomplex, typename Fzomplex, typename Frcomplex, typename Frzomplex, class... Inputs>
inline void call_backend6(Fsingle &sfunc, Fdouble &dfunc, Fcomplex &cfunc, Fzomplex &zfunc,
                          Frcomplex &crfunc, Frzomplex &zrfunc, Inputs &&...args){
    if __HALA_CONSTEXPR_IF__ (flag)
        call_backend<criteria>(sfunc, dfunc, cfunc, zfunc, args...);
    else
        call_backend<criteria>(sfunc, dfunc, crfunc, zrfunc, args...);
}

}

#endif
