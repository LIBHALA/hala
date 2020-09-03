#ifndef __HALA_TYPES_CASTER_HPP
#define __HALA_TYPES_CASTER_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_types_helpers.hpp"

/*!
 * \internal
 * \file hala_types_caster.hpp
 * \brief HALA templates to assert type safely and const correctness.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACHECKTYPES
 *
 * Contains a series of templates that are used for type casting.
 * \endinternal
*/

namespace hala{

/*!
 * \internal
 * \ingroup HALACORE
 * \addtogroup HALACASTTYPES Type Conversion Helper Templates
 *
 * HALA is written for the C++-14 standard, which does not support constexpr-if statement (unlike C++-17),
 * which means that code that will never be executed should still compile even if it makes no sense.
 * Unfortunately, the static_cast() method cannot convert from complex to real numbers and cannot handle
 * user defined complex classes, thus HALA provides a substitution hala::
 * \endinternal
 */

/*!
 * \ingroup HALACASTTYPES
 * \brief Wrapper around standard real that can be specialized.
 */
template<typename T, typename = void> struct number_real{
    //! \brief Using static cast, convert x of type U into a number of type T.
    static inline auto get(T const x){ return std::real(x); }
};

/*!
 * \ingroup HALACASTTYPES
 * \brief Specialization for compatible custom complex class.
 */
template<typename T> struct number_real<T, std::enable_if_t<
        (is_fcomplex<T>::value || is_dcomplex<T>::value) &&
        !std::is_same<T, std::complex<float>>::value &&
        !std::is_same<T, std::complex<double>>::value
    >>{
    //! \brief Returns the real type of a complex number using reinterpret cast.
    static inline auto get(T const x){
        using real_type = typename define_standard_precision<T>::value_type;
        return *reinterpret_cast<real_type const*>(&x);
    }
};

/*!
 * \ingroup HALACTYPES
 * \brief Returns the real part of \b x.
 */
template<typename T> auto creal(T const x){ return number_real<T>::get(x); }


/*!
 * \ingroup HALACASTTYPES
 * \brief Wrapper around standard imag() that can be specialized.
 */
template<typename T, typename = void> struct number_imag{
    //! \brief Using static cast, convert x of type U into a number of type T.
    static inline auto get(T const x){ return std::imag(x); }
};

/*!
 * \ingroup HALACASTTYPES
 * \brief Specialization for compatible custom complex class.
 */
template<typename T> struct number_imag<T, std::enable_if_t<
        (is_fcomplex<T>::value || is_dcomplex<T>::value) &&
        !std::is_same<T, std::complex<float>>::value &&
        !std::is_same<T, std::complex<double>>::value
    >>{
    //! \brief Returns the real type of a complex number using reinterpret cast.
    static inline auto get(T const x){
        using real_type = typename define_standard_precision<T>::value_type;
        return reinterpret_cast<real_type const*>(&x)[1];
    }
};

/*!
 * \ingroup HALACTYPES
 * \brief Returns the complex part of \b x.
 */
template<typename T> auto cimag(T const x){ return number_imag<T>::get(x); }


/*!
 * \ingroup HALACASTTYPES
 * \brief Returns the conjugate of a number, handles the real case (i.e., just return the number).
 */
template<typename T, typename = void> struct number_conj{
    //! \brief Returns x.
    static inline T get(T const x){ return x; }
};

/*!
 * \ingroup HALACASTTYPES
 * \brief Returns the conjugate of a number, handles the std::complex case (i.e., just use std::conj).
 */
template<typename T> struct number_conj<T, std::enable_if_t<
        std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value
    >>{
    //! \brief Returns std::conj(x).
    static inline T get(T const x){ return std::conj(x); }
};

/*!
 * \ingroup HALACASTTYPES
 * \brief Returns the conjugate of a number, handles the non-standard complex case (i.e., using reinterpret cast).
 */
template<typename T> struct number_conj<T, std::enable_if_t<
        (is_fcomplex<T>::value || is_dcomplex<T>::value) &&
        !std::is_same<T, std::complex<float>>::value &&
        !std::is_same<T, std::complex<double>>::value
    >>{
    //! \brief Returns the conjugate of a complex number.
    static inline T get(T const x){
        using real_type = typename define_standard_precision<T>::value_type;
        T result;
        reinterpret_cast<real_type*>(&result)[0] =  reinterpret_cast<real_type const*>(&x)[0];
        reinterpret_cast<real_type*>(&result)[1] = -reinterpret_cast<real_type const*>(&x)[1];
        return result;
    }
};

/*!
 * \ingroup HALACTYPES
 * \brief Returns the conjugate of a number, note that the returned type always matches \b x.
 */
template<typename T> T cconj(T const x){ return number_conj<T>::get(x); }


/*!
 * \ingroup HALACASTTYPES
 * \brief Returns the absolute value of a number, handles the non-custom complex case.
 */
template<typename T, typename = void> struct number_abs{
    //! \brief Returns std::abs(x).
    static inline auto get(T const x){ return std::abs(x); }
};

/*!
 * \ingroup HALACASTTYPES
 * \brief Returns the absolute value of a number, handles the custom complex case.
 */
template<typename T> struct number_abs<T, std::enable_if_t<
        (is_fcomplex<T>::value || is_dcomplex<T>::value) &&
        !std::is_same<T, std::complex<float>>::value &&
        !std::is_same<T, std::complex<double>>::value
    >>{
    //! \brief Returns the square-root of real times real plus complex times complex.
    static inline auto get(T const x){ return std::sqrt(hala_real(x)*hala_real(x) + hala_imag(x) * hala_imag(x)); }
};

/*!
 * \ingroup HALACTYPES
 * \brief Returns the absolute value of a number, real or complex, standard or custom.
 */
template<typename T> auto hala_abs(T const x){ return number_abs<T>::get(x); }


/*!
 * \ingroup HALACASTTYPES
 * \brief Wrapper around static cast that can be specialized.
 */
template<typename T, typename U, typename = void> struct number_caster{
    //! \brief Using static cast, convert x of type U into a number of type T.
    static inline T cast(U x){ return static_cast<T>(x); }
};

/*!
 * \ingroup HALACASTTYPES
 * \brief Specialization when the type is the same.
 */
template<typename T> struct number_caster<T, T, void>{
    //! \brief Just return x, nothing to convert.
    static inline T cast(T x){ return x; }
};

/*!
 * \ingroup HALACASTTYPES
 * \brief Specialization when converting standard complex to real.
 */
template<typename T, typename U> struct number_caster<T, U,
        std::enable_if_t<
            (is_float<T>::value    || is_double<T>::value)   &&  // if destination is real
            (is_fcomplex<U>::value || is_dcomplex<U>::value)     // and the source is complex
        >>{
    //! \brief Converting complex to real, use std::real().
    static inline T cast(U x){ return static_cast<T>(creal(x)); }
};

/*!
 * \ingroup HALACASTTYPES
 * \brief Specialization when converting standard real to custom complex.
 */
template<typename T, typename U> struct number_caster<T, U,
        std::enable_if_t<
            (is_fcomplex<T>::value || is_dcomplex<T>::value) &&  // if the destination is complex
             !std::is_same<T, std::complex<float>>::value    &&
             !std::is_same<T, std::complex<double>>::value   &&  // and not a standard complex
            (is_float<U>::value    || is_double<U>::value)       // and the source is real
        >>{
    //! \brief Converting complex to real, use std::real().
    static inline T cast(U x){
        using real_type = typename define_standard_precision<T>::value_type;
        T result;
        reinterpret_cast<real_type*>(&result)[0] = static_cast<real_type>(x);
        reinterpret_cast<real_type*>(&result)[1] = static_cast<real_type>(0.0);
        return result;
    }
};

/*!
 * \ingroup HALACASTTYPES
 * \brief Specialization when converting complex to complex, at least one is not standard.
 */
template<typename T, typename U> struct number_caster<T, U,
        std::enable_if_t<
            (is_fcomplex<T>::value || is_dcomplex<T>::value) &&  // if the destination is complex
            (is_fcomplex<U>::value || is_dcomplex<U>::value) &&  // and the source is complex
            !(std::is_same<T, std::complex<float>>::value   && // are we are not converting standard to standard
              std::is_same<U, std::complex<double>>::value) &&
            !(std::is_same<T, std::complex<double>>::value  &&
              std::is_same<U, std::complex<float>>::value)  &&
             !std::is_same<T, U>::value                        // and not all are the same
        >>{
    //! \brief Converting complex to real, use std::real().
    static inline T cast(U x){
        using real_typet = typename define_standard_precision<T>::value_type;
        T result;
        reinterpret_cast<real_typet*>(&result)[0] = static_cast<real_typet>(creal(x));
        reinterpret_cast<real_typet*>(&result)[1] = static_cast<real_typet>(cimag(x));
        return result;
    }
};

/*!
 * \ingroup HALACASTTYPES
 * \brief Specialization when converting standard complex to complex.
 */
template<typename T, typename U> struct number_caster<T, U,
        std::enable_if_t<
            (std::is_same<T, std::complex<float>>::value  &&
             std::is_same<U, std::complex<double>>::value)  ||  // if both are standard complex
            (std::is_same<U, std::complex<float>>::value  &&
             std::is_same<T, std::complex<double>>::value)
        >>{
    //! \brief Converting complex to real, use std::real().
    static inline T cast(U x){
        return {static_cast<typename T::value_type>(x.real()), static_cast<typename T::value_type>(x.imag())};
    }
};

}

#endif
