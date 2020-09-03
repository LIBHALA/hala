#ifndef __HALA_BLAS_1_WRAPPERS_HPP
#define __HALA_BLAS_1_WRAPPERS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_blas_external.hpp"

namespace hala{
/*!
 * \internal
 * \file hala_blas_1.hpp
 * \brief HALA wrapper for BLAS level 1 functions.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALABLAS1
 *
 * Contains the HALA-BLAS level 1 wrapper templates.
 * \endinternal
*/

/*!
 * \ingroup HALABLAS
 * \addtogroup HALABLAS1 BLAS level 1 templates
 *
 * \par BLAS Level 1
 * Level 1 operations are defines as those that act on vectors, i.e., without matrices.
 * Examples of such operations are norm, scale, and vector addition.
 */

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS xcopy() methods.
 *
 * Copies \b N entries of \b x into \b y using strides \b incx and \b incy.
 * The name is chosen to be distinct from standard copy().
 *
 * Overloads:
 * \code
 *  hala::vcopy(x, y, incx = 1, incy = 1, N = -1);
 *  hala::vcopy(engine, N, x, incx, y, incy);
 *  hala::vcopy(engine, x, y, incx = 1, incy = 1, N = -1);
 *  default_vector_type y = hala::vcopy(engine, x);
 * \endcode
 * The increments default to 1 and the correct value of \b N is inferred from the size of \b x and \b incx.
 */
template<class VectorLikeX, class VectorLikeY>
inline void vcopy(int N, VectorLikeX const &x, int incx, VectorLikeY &&y, int incy){
    check_types(x, y);
    check_set_size(assume_output, y, 1 + (N - 1) * incy);
    assert( valid::vcopy(N, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    call_backend<scalar_type>(scopy_, dcopy_, ccopy_, zcopy_, &N, convert(x), &incx, cconvert(y), &incy);
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS xswap() methods.
 *
 * Swaps \b N entries between \b x and \b y using strides \b incx and \b incy.
 * The name is chosen to be distinct from standard swap().
 *
 * Overloads:
 * \code
 *  hala::vswap(x, y, incx = 1, incy = 1, N = -1);
 *  hala::vswap(engine, N, x, incx, y, incy);
 *  hala::vswap(engine, x, y, incx = 1, incy = 1, N = -1);
 * \endcode
 * The increments default to 1 and the correct value of \b N is inferred from the size of \b x and \b incx.
 */
template<class VectorLikeX, class VectorLikeY>
inline void vswap(int N, VectorLikeX &&x, int incx, VectorLikeY &&y, int incy){
    check_types(x, y);
    assert( valid::vswap(N, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    call_backend<scalar_type>(sswap_, dswap_, cswap_, zswap_, &N, cconvert(x), &incx, cconvert(y), &incy);
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS vector 2-norm.
 *
 * Computes the vector 2-norm, i.e., \f$ \left( \sum_{i=1}^N |x_i|^2 \right)^{1/2} \f$.
 * - \b N is the total number of entries
 * - \b incx is the distance between vector entries
 * - \b returns a real number with the same precision as \b x, i.e., float for complex<float>
 *
 * Overloads:
 * \code
 *  auto x_norm = hala::norm2(x, incx = 1, N = -1);
 *  auto x_norm = hala::norm2(engine, N, x, incx);
 *  auto x_norm = hala::norm2(engine, x, incx = 1, N = -1);
 * \endcode
 * The increment defaults to 1 and the correct value of \b N is inferred from the size of \b x and \b incx.
 */
template<class VectorLikeX>
inline auto norm2(int N, VectorLikeX const &x, int incx){
    check_types(x);
    assert( valid::norm2(N, x, incx) );

    using scalar_type    = get_scalar_type<VectorLikeX>;
    using precision_type = get_precision_type<VectorLikeX>;

    if __HALA_CONSTEXPR_IF__ (is_float<scalar_type>::value){
        return get_cast<precision_type>(snrm2_(&N, convert(x), &incx));
    }else if __HALA_CONSTEXPR_IF__ (is_double<scalar_type>::value){
        return get_cast<precision_type>(dnrm2_(&N, convert(x), &incx));
    }else if __HALA_CONSTEXPR_IF__ (is_fcomplex<scalar_type>::value){
        return get_cast<precision_type>(scnrm2_(&N, convert(x), &incx));
    }else if __HALA_CONSTEXPR_IF__ (is_dcomplex<scalar_type>::value){
        return get_cast<precision_type>(dznrm2_(&N, convert(x), &incx));
    }
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS vector sum of absolute values of the entry components.
 *
 * Computes the sum of absolute value of the entry components, i.e., \f$ \sum_{i=1}^N |real(x_i)| + |imag(x_i)| \f$.
 * - \b N is the total number of entries
 * - \b incx is the distance between vector entries
 * - \b returns a real number with the same precision as \b x, i.e., float for complex<float>
 *
 * Overloads:
 * \code
 *  auto x_sum = hala::asum(x, incx = 1, N = -1);
 *  auto x_sum = hala::asum(engine, N, x, incx);
 *  auto x_sum = hala::asum(engine, x, incx = 1, N = -1);
 * \endcode
 * The increment defaults to 1 and the correct value of \b N is inferred from the size of \b x and \b incx.
 */
template<class VectorLikeX>
inline auto asum(int N, VectorLikeX const &x, int incx){
    check_types(x);
    assert( valid::norm2(N, x, incx) ); // inputs are the same

    using scalar_type    = get_scalar_type<VectorLikeX>;
    using precision_type = get_precision_type<VectorLikeX>;

    if __HALA_CONSTEXPR_IF__ (is_float<scalar_type>::value){
        return get_cast<precision_type>(sasum_(&N, convert(x), &incx));
    }else if __HALA_CONSTEXPR_IF__ (is_double<scalar_type>::value){
        return get_cast<precision_type>(dasum_(&N, convert(x), &incx));
    }else if __HALA_CONSTEXPR_IF__ (is_fcomplex<scalar_type>::value){
        return get_cast<precision_type>(scasum_(&N, convert(x), &incx));
    }else if __HALA_CONSTEXPR_IF__ (is_dcomplex<scalar_type>::value){
        return get_cast<precision_type>(dzasum_(&N, convert(x), &incx));
    }
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS find index of largest vector entry, ixamax().
 *
 * Returns the index of the vector entry with the largest sum of absolute value of the components \f$ |real(x_i)| + |imag(x_i)| \f$.
 * - \b N is the total number of entries
 * - \b incx is the distance between vector entries
 * - \b returns a zero-index (unlike standard BLAS) of the largest entry, the index is between 0 and N-1
 *
 * Overloads:
 * \code
 *  int max = hala::iamax(x, incx = 1, N = -1);
 *  int max = hala::iamax(engine, N, x, incx);
 *  int max = hala::iamax(engine, x, incx = 1, N = -1);
 * \endcode
 * The increment defaults to 1 and the correct value of \b N is inferred from the size of \b x and \b incx.
 */
template<class VectorLikeX>
inline int iamax(int N, VectorLikeX const &x, int incx){
    check_types(x);
    assert( valid::norm2(N, x, incx) ); // inputs are the same

    using scalar_type    = get_scalar_type<VectorLikeX>;
    if __HALA_CONSTEXPR_IF__ (is_float<scalar_type>::value){
        return isamax_(&N, convert(x), &incx) - 1;
    }else if __HALA_CONSTEXPR_IF__ (is_double<scalar_type>::value){
        return idamax_(&N, convert(x), &incx) - 1;
    }else if __HALA_CONSTEXPR_IF__ (is_fcomplex<scalar_type>::value){
        return icamax_(&N, convert(x), &incx) - 1;
    }else if __HALA_CONSTEXPR_IF__ (is_dcomplex<scalar_type>::value){
        return izamax_(&N, convert(x), &incx) - 1;
    }
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS vector scaled addition xaxpy().
 *
 * Computes \f$ y = \alpha x + y \f$, i.e., scale \b x and add to \b y.
 * - \b N is the total number of entries
 * - \b incx and \b incy are the distances between vector entries
 *
 * Overloads:
 * \code
 *  hala::axpy(x, y, incx = 1, incy = 1, N = -1);
 *  hala::axpy(engine, N, x, incx, y, incy);
 *  hala::axpy(engine, x, y, incx = 1, incy = 1, N = -1);
 * \endcode
 * The increments default to 1 and the correct value of \b N is inferred from the effective size of \b x.
 */
template<typename FS, class VectorLikeX, class VectorLikeY>
inline void axpy(int N, FS alpha, VectorLikeX const &x, int incx, VectorLikeY &&y, int incy){
    check_types(x, y);
    assert( valid::axpy(N, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    auto fsalpha = get_cast<scalar_type>(alpha);
    call_backend<scalar_type>(saxpy_, daxpy_, caxpy_, zaxpy_, &N, pconvert(&fsalpha), convert(x), &incx, cconvert(y), &incy);
}


/*!
 * \internal
 * \ingroup HALABLAS1
 * \brief Manual implementation of dot-product to be used in the complex case.
 *
 * Returning complex numbers from Fortran to C++ is platform dependent,
 * the complex case is handled manually using raw-arrays.
 * \endinternal
 */
template<bool conjugate, typename T>
inline T dot_array(int N, T const x[], int incx, T const y[], int incy){
    T sum = get_cast<T>(0.0);
    if ((incx == 1) && (incy == 1)){
        if __HALA_CONSTEXPR_IF__ (conjugate){
            for(int i=0; i<N; i++) sum += cconj(x[i]) * y[i];
        }else{
            for(int i=0; i<N; i++) sum += x[i] * y[i];
        }
    }else{
        if __HALA_CONSTEXPR_IF__ (conjugate){
            for(int i=0; i<N; i++) sum += cconj(x[i * incx]) * y[i * incy];
        }else{
            for(int i=0; i<N; i++) sum += x[i * incx] * y[i * incy];
        }
    }
    return sum;
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS vector dot-product xdot().
 *
 * Computes the dot product between two real or complex valued vectors:
 * \f$ \sum_{i=1}^N op(x_i) y_i \f$,
 * where \b op is either the identity or the conjugate operation.
 *
 * - \b conjugate indicates whether to use the conjugate (\b true) or identity (\b false)
 * - \b N is the total number of entries to use
 * - \b incx and \b incy are the positive strides between entries
 * - \b returns a complex number that matches the entreis in \b x
 *
 * Overloads:
 * \code
 *  auto dot_product = hala::dot(x, y, incx = 1, incy = 1, N = -1);
 *  auto dot_product = hala::dot(engine, N, x, incx, y, incy);
 *  auto dot_product = hala::dot(engine, x, y, incx = 1, incy = 1, N = -1);
 *  auto dot_product = hala::dotu(x, y, incx = 1, incy = 1, N = -1);
 *  auto dot_product = hala::dotu(engine, N, x, incx, y, incy);
 *  auto dot_product = hala::dotu(engine, x, y, incx = 1, incy = 1, N = -1);
 * \endcode
 * The increments default to 1 and the correct value of \b N is inferred from the effective size of \b x.
 * The dotu() variant defaults to \b conjugate being \b non_conj.
 *
 * \b Note: The BLAS functions that implement complex dot-product return complex numbers which are incompatible with C++ "return" statement.
 * Instantiation with complex numbers (single or double precision) will call "manual" implementation with a simple for-loop,
 * and will (probably) be slower than BLAS.
 */
template<bool conjugate = true, class VectorLikeX, class VectorLikeY>
inline auto dot(int N, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy){
    check_types(x, y);
    assert( valid::dot(N, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    if __HALA_CONSTEXPR_IF__ (is_float<scalar_type>::value){
        return get_cast<scalar_type>(sdot_(&N, convert(x), &incx, convert(y), &incy));
    }else if __HALA_CONSTEXPR_IF__ (is_double<scalar_type>::value){
        return get_cast<scalar_type>(ddot_(&N, convert(x), &incx, convert(y), &incy));
    }else{
        return get_cast<scalar_type>(dot_array<conjugate>(N, get_standard_data(x), incx, get_standard_data(y), incy));
    }
}

#ifndef __HALA_DOXYGEN_SKIP
template<class VectorLikeX, class VectorLikeY>
inline auto dotu(int N, VectorLikeX const &x, int incx, VectorLikeY const &y, int incy){
    return dot<false>(N, x, incx, y, incy);
}
#endif

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS vector scale by constant xscal().
 *
 * Scales the vector by a constant \b alpha.
 * - real valued vectors can only be scaled by a real \b alpha
 * - complex valued vectors can be scaled by either real or complex scalar
 *   using an appropriately optimized algorithm at the back-end.
 *
 * Overloads:
 * \code
 *  hala::scal(alpha, x, incx = 1, N = -1);
 *  hala::scal(engine, N, alpha, x, incx);
 *  hala::scal(engine, alpha, x, incx = 1, N = -1);
 * \endcode
 * The increments default to 1 and the correct value of \b N is inferred from the effective size of \b x.
 */
template<typename FS, class VectorLikeX>
inline void scal(int N, FS alpha, VectorLikeX &&x, int incx){
    check_types(x);
    assert( valid::scal(N, x, incx) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    using precision_type = get_precision_type<VectorLikeX>;
    if __HALA_CONSTEXPR_IF__ ((is_fcomplex<scalar_type>::value || is_dcomplex<scalar_type>::value) &&
        (std::is_same<FS, int>::value || is_float<FS>::value || is_double<FS>::value)){ // mix complex and real
            auto fsalpha = get_cast<precision_type>(alpha);
            if __HALA_CONSTEXPR_IF__ (is_fcomplex<scalar_type>::value){
                csscal_(&N, pconvert(&fsalpha), cconvert(x), &incx);
            }else{
                zdscal_(&N, pconvert(&fsalpha), cconvert(x), &incx);
            }
    }else{ // scaled by the same type
        auto falpha = get_cast<scalar_type>(alpha);
        call_backend<scalar_type>(sscal_, dscal_, cscal_, zscal_, &N, pconvert(&falpha), cconvert(x), &incx);
    }
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS generate Givens rotation, xrotg().
 *
 * Computes the sine and cosine of a Givens rotation so that
 * \f$ \left( \begin{array}{cc} c & s \\ - \bar s & c \end{array} \right)
 *     \left( \begin{array}{cc} a \\  b \end{array} \right) =
 *     \left( \begin{array}{cc} n \\  0 \end{array} \right) \f$
 * - \b SA and \b SB are the starting vector entries \b a and \b b
 * - \b SA will be overwritten with the value of \b n
 * - some BLAS implementations (e.g., MKL) will also overwrite \b SB with additional flags
 *  thus all values are considered inputs and outputs by HALA
 * - \b C and \b S are the cos() and sin() of the rotation
 * - \b C is always a real numbers, the other three can be real or complex
 *
 * Overloads:
 * \code
 *  hala::rotg(engine, SA, SB, C, S);
 * \endcode
 * The cuda_engine overload will work with pointers only so that the values can be given on the GPU,
 * if the four constants are bound to the CPU then this method is preferable since the GPU cannot
 * really accelerate this very simple computation.
 */
template<typename T>
void rotg(T &SA, T &SB, typename define_standard_precision<T>::value_type &C, T &S){
    check_types(std::vector<T>());

    call_backend<T>(srotg_, drotg_, crotg_, zrotg_, pconvert(&SA), pconvert(&SB), pconvert(&C), pconvert(&S));
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS apply the Givens rotation, xrot().
 *
 * Applies the Givens rotation to the vectors \b x and \b y.
 * - the vectors have \b N entries in strides \b incx and \b incy
 * - \b C is always a real number, defines the cosine of the rotation
 * - \b S can be either real or complex (if \b x and \b y are complex) and defines the sine of the rotation
 * - the \b C and \b S are best computed by hala::rotg()
 *
 * Overloads:
 * \code
 *  hala::rot(x, y, C, S, incx = 1, incy = 1, N = -1);
 *  hala::rot(engine, N, x, incx, y, incy, C, S);
 *  hala::rot(engine, x, y, C, S, incx = 1, incy = 1, N = -1);
 * \endcode
 */
template<typename FC, typename FS, class VectorLikeX, class VectorLikeY>
void rot(int N, VectorLikeX &&x, int incx, VectorLikeY &&y, int incy, FC C, FS S){
    check_types(x, y); // make sure x and y types match
    assert( valid::rot(N, x, incx, y, incy) );

    using scalar_type = get_scalar_type<VectorLikeX>;
    using precision_type = get_precision_type<VectorLikeX>;

    if __HALA_CONSTEXPR_IF__ ((is_fcomplex<scalar_type>::value || is_dcomplex<scalar_type>::value) &&
        (std::is_same<FS, int>::value || is_float<FS>::value || is_double<FS>::value)){ // mix complex and real
            auto effc = get_cast<precision_type>(C);
            auto effs = get_cast<precision_type>(S);
            if __HALA_CONSTEXPR_IF__ (is_fcomplex<scalar_type>::value){
                csrot_(&N, cconvert(x), &incx, cconvert(y), &incy, pconvert(&effc), pconvert(&effs));
            }else{
                zdrot_(&N, cconvert(x), &incx, cconvert(y), &incy, pconvert(&effc), pconvert(&effs));
            }
    }else{ // scaled by the same type
        auto effc = get_cast<precision_type>(C);
        auto effs = get_cast<scalar_type>(S);
        call_backend<scalar_type>(srot_, drot_, crot_, zrot_,
                     &N, cconvert(x), &incx, cconvert(y), &incy, pconvert(&effc), pconvert(&effs));
    }
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS generate modified Givens rotation.
 *
 * Only works with real numbers.
 *
 * The modified Givens matrix is a 2 by 2 matrix that zeros out the second
 * component of the vector \f$ (\sqrt(d_1) x, \sqrt(d_2) y) \f$.
 * - \b D1, \b D2, \b X, and \b Y are the scalars
 * - \b param is a vector with 5 entries, first is a flag the rest are overwritten with the Givens matrix.
 * - since pretty much all parameters are both input and output, this works only if the types match
 *
 * Overloads:
 * \code
 *  hala::rotmg(engine, SA, SB, C, S);
 * \endcode
 */
template<typename T, class VectorLike>
void rotmg(T &D1, T &D2, T &X, T const &Y, VectorLike &&param){
    static_assert(is_float<T>::value || is_double<T>::value, "Givens rotations work only with real numbers.");

    using standard_type = get_standard_type<VectorLike>;
    static_assert(is_compatible<T, standard_type>::value, "rotmg() requires that the types of all inputs (vector and scalars) match");

    assert( check_size(param, 5) );

    call_backend2<T>(srotmg_, drotmg_, pconvert(&D1), pconvert(&D2), pconvert(&X), pconvert(&Y), cconvert(param));
}

/*!
 * \ingroup HALABLAS1
 * \brief Wrapper to BLAS apply the Givens rotation, xrotm().
 *
 * Only works with real numbers.
 * - applies the modified Givens rotation to the vectors \b x and \b y
 * - the vectors have \b N entries in strides \b incx and \b incy
 * - \b param must have size at least 5 and was generated by hala::rotmg()
 *
 * Overloads:
 * \code
 *  hala::rotm(x, y, param, incx = 1, incy = 1, N = -1);
 *  hala::rotm(engine, N, x, incx, y, incy, param);
 *  hala::rotm(engine, x, y, param, incx = 1, incy = 1, N = -1);
 * \endcode
 */
template<class VectorLikeX, class VectorLikeY, class VectorLikeP>
void rotm(int N, VectorLikeX &&x, int incx, VectorLikeY &&y, int incy, VectorLikeP const &param){
    check_types(x, y, param); // make sure x and y types match
    using scalar_type = get_scalar_type<VectorLikeX>;
    static_assert(is_float<scalar_type>::value || is_double<scalar_type>::value, "Givens rotations work only with real numbers.");

    assert( valid::rot(N, x, incx, y, incy) ); // test is the same with the addition of param having size 5
    assert( check_size(param, 5) );

    call_backend2<scalar_type>(srotm_, drotm_, &N, cconvert(x), &incx, cconvert(y), &incy, convert(param));
}

}

#endif
