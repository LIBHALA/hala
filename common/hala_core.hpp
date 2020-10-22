#ifndef __HALA_CORE_HPP
#define __HALA_CORE_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include <cmath>
#include <vector>
#include <valarray>
#include <complex>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include <utility>
#include <functional>
#include <sstream>
#include <chrono>
#include <limits>
#include <memory>
#include <typeinfo>

/*!
 * \internal
 * \file hala_core.hpp
 * \brief HALA helper templates.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACORE
 *
 * Contains includes and helper macros used throughout HALA.
 * \endinternal
*/

/*!
 * \defgroup HALACORE HALA Helper Templates
 *
 * Several macros that do common tasks, e.g., assert that the vector \b value_type is
 * float, double, std::complex<float> or std::complex<double>.
 *
 */

#ifndef HALA_VERSION_MAJOR
/*!
 * \ingroup HALACORE
 * \brief The major component of the version, standalone without CMake.
 */
#define HALA_VERSION_MAJOR 1

/*!
 * \ingroup HALACORE
 * \brief The major component of the version, standalone without CMake.
 */
#define HALA_VERSION_MINOR 0

/*!
 * \ingroup HALACORE
 * \brief The major component of the version, standalone without CMake.
 */
#define HALA_VERSION_PATCH 0
#endif

namespace hala{

/*!
 * \internal
 * \ingroup HALACORE
 * \addtogroup HALACORECHAR Character Consistency Checks
 *
 * Inline functions to help assert whether char inputs have the correct value.
 *
 * \endinternal
 */

/*!
 * \ingroup HALACORECHAR
 * \brief Returns true if \b trans is either lower or upper case N, T, or C.
 */
inline bool check_trans(char trans){ return ((trans == 'T') || (trans == 't') || (trans == 'N') || (trans == 'n') || (trans == 'C') || (trans == 'c')); }

/*!
 * \ingroup HALACORECHAR
 * \brief Returns true if \b uplo is either lower or upper case U or L.
 */
inline bool check_uplo(char uplo){ return ((uplo == 'U') || (uplo == 'u') || (uplo == 'L') || (uplo == 'l')); }

/*!
 * \ingroup HALACORECHAR
 * \brief Returns true if \b diag is either lower or upper case U or N.
 */
inline bool check_diag(char diag){ return ((diag == 'U') || (diag == 'u') || (diag == 'N') || (diag == 'n')); }

/*!
 * \ingroup HALACORECHAR
 * \brief Returns true if \b side is either lower or upper case L or R.
 */
inline bool check_side(char side){ return ((side == 'R') || (side == 'r') || (side == 'L') || (side == 'l')); }

/*!
 * \ingroup HALACORECHAR
 * \brief Returns true if \b trans is upper or lower case 'N'
 */
inline bool is_n(char trans){ return ((trans == 'n') || (trans == 'N')); }

/*!
 * \ingroup HALACORECHAR
 * \brief Returns true if \b trans is upper or lower case 'C'
 */
inline bool is_c(char trans){ return ((trans == 'c') || (trans == 'C')); }

/*!
 * \ingroup HALACORECHAR
 * \brief Returns \b true if \b side is upper or lower case 'L'
 */
inline bool is_l(char side){ return ((side == 'l') || (side == 'L')); }

/*!
 * \internal
 * \ingroup HALACORE
 * \brief Casts two variables into \b size_t and multiplies them.
 *
 * Helps guard against overflow, usually each dimension of a matrix can be as large as the maximum
 * value of the signed int, but the product may exceed the limits fast.
 * This template helps guards against such overflow.
 * \endinternal
 */
template<typename IntA, typename IntB>
inline size_t hala_size(IntA a, IntB b){ return static_cast<size_t>(a) * static_cast<size_t>(b); }

/*!
 * \ingroup HALACORE
 * \brief If \b pass is false, it throws \b runtime_error() with the given message.
 */
inline void runtime_assert(bool pass, std::string message){ if (!pass) throw std::runtime_error(message); }

/*!
 * \defgroup HALAMISC HALA Miscellaneous Methods
 *
 * Useful structs and templates that don't necessarily fit in any particular category,
 * but are useful nevertheless.
 */

/*!
 * \ingroup HALAMISC
 * \brief Simple \b chronometer with start and stop functions that returns the milliseconds as a \b long \b long.
 *
 * A struct that simplifies the verbose type-casts from the chrono module.
 */
struct chronometer{
    //! \brief Stores the starting point of the \b set_start().
    std::chrono::time_point<std::chrono::system_clock> time_start;
    //! \brief Start measuring time.
    void set_start(){ time_start = std::chrono::system_clock::now(); }
    //! \brief End measuring time, return the duration in milliseconds since the last \b set_start() call.
    long long get_end(){ return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - time_start).count(); }
};

}

#endif
