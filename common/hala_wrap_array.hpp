#ifndef __HALA_WRAP_ARRAY_HPP
#define __HALA_WRAP_ARRAY_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_input_checker.hpp"

/*!
 * \file hala_wrap_array.hpp
 * \brief HALA wrapper simple arrays.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAARRAY
 *
 * Contains the HALA array wrapper.
*/

/*!
 * \brief Master namespace encapsulating all HALA capabilities.
 */
namespace hala{

/*!
 * \ingroup HALACUSTOM
 * \addtogroup HALAARRAY Array Wrapper
 *
 * \par Arrays and External API
 * C-style arrays are seldom used in C++ since they lack RAII capabilities and do not include meta-data such as size.
 * However, arrays are very useful when interfacing with other programming languages.
 * For example, a C++ function can be declared \b extern \b "C" and then allowed to accept an array as input,
 * then the function can be called with native data-structures from C,
 * Fortran (2003 or even 95 if the name ends with underscore), or Python (using ctypes and numpy arrays).
 *
 * \par
 * HALA templates require C++-style of containers in order to be instantiated.
 * In order to allow other languages to access HALA functionality, e.g., HALA iterative solvers and preconditioners,
 * HALA provides a tin-and-light array wrapper
 */

/*!
 * \ingroup HALAARRAY
 * \brief Wrapper around C-style arrays that can be passed to HALA templates.
 *
 * This wrapper accepts an externally allocated array and size and then provide
 * an interface that mimics containers such as std::valarray and std::vector.
 * The wrapper (and the wrapped array) can be passed to any HALA template.
 * However, the wrapper does not provide full container interface (only the bare minimum),
 * and the array will \b NOT be deleted by the destructor.
 *
 * The constructor accepts an array and the size of the array,
 * the provided size will be used for all operators that need to infer the number of entries, e.g., hala::norm2().
 * The resize method will never change the size of the array,
 * but only assert() that the requested new size matches the old (only in debug mode).
 *
 * Example usage:
 * \code
 * extern "C" double foo(int N, double *x){
 *     auto xvec = wrap_array(x, N);
 *     std::cout << "Norm of x is: " << hala::norm2(xvec) << "\n";
 *
 *     std::valarray<double> y(N, 1.0);
 *     std::cout << "Sum of entries in x: " << hala::dot(xvec, y) << "\n";
 * }
 * \endcode
 */
template<typename ScalarType>
class wrapped_array{
public:
    //! \brief Keeps track of the array type for sanity check and static_assert().
    using value_type = std::remove_const_t<ScalarType>;
    //! \brief Type of the index.
    using size_type = std::size_t;
    //! \brief Difference between pointers.
    using difference_type = std::ptrdiff_t;
    //! \brief Reference.
    using reference = value_type&;
    //! \brief Const reference.
    using const_reference = value_type const&;
    //! \brief Pointer.
    using pointer = value_type*;
    //! \brief Const pointer.
    using const_pointer = value_type const*;
    //! \brief Iterator (really just a pointer).
    using iterator = pointer;
    //! \brief Const iterator (really just a const pointer).
    using const_iterator = const_pointer;

    //! \brief Wraps the array \b arr and assumes the size is \b num_entries.
    wrapped_array(ScalarType *arr, size_t num_entries) : x(arr), num(num_entries){
        static_assert(is_float<value_type>::value ||
                      is_double<value_type>::value ||
                      is_fcomplex<value_type>::value ||
                      is_dcomplex<value_type>::value, "wrap_array can use float, double, or complex numbers only");
    }
    //! \brief Delete copy constructor to avoid an alias.
    wrapped_array(wrapped_array<ScalarType> const &) = delete;
    //! \brief Delete copy constructor to avoid an alias.
    void operator =(wrapped_array<ScalarType> const &) = delete;
    //! \brief Move constructor.
    wrapped_array(wrapped_array<ScalarType> &&other) : x(std::exchange(other.x, nullptr)), num(std::exchange(other.num, 0)) {}
    //! \brief Move assignemnt.
    void operator =(wrapped_array<ScalarType> &&other){ x = other.x; num = other.num; other.x = nullptr; other.num = 0; }
    //! \brief Destructor, does not delete the array.
    ~wrapped_array(){}

    //! \brief Return a const reference to the array entry at \b position.
    const ScalarType& operator[](size_t position) const{ return x[position]; }
    //! \brief Return a reference to the array entry at \b position.
    ScalarType& operator[](size_t position){ return x[position]; }

    //! \brief Returns the assumed size.
    size_t size() const{ return num; }

    //! \brief Return an alias to the wrapper array.
    ScalarType* data(){ return x; }
    //! \brief Return a const alias to the wrapper array.
    ScalarType const* data() const{ return x; }

    //! \brief Start of the array.
    iterator begin(){ return x; }
    //! \brief Const start of the array.
    const_iterator begin() const{ return x; }
    //! \brief Const start of the array.
    const_iterator cbegin() const{ return x; }
    //! \brief End of the array (not the last entry but one after).
    iterator end(){ return x + num; }
    //! \brief Const end of the array.
    const_iterator end() const{ return x + num; }
    //! \brief Const end of the array.
    const_iterator cend(){ return x + num; }

private:
    ScalarType *x;
    size_t num;
};

/*!
 * \ingroup HALAARRAY
 * \brief Creates a wrapper around the C-style arrays that can be passed to HALA templates.
 */
template<typename ArrayType>
wrapped_array<ArrayType> wrap_array(ArrayType arr[], size_t num_entries){
    return wrapped_array<ArrayType>(arr, num_entries);
}

}

#endif
