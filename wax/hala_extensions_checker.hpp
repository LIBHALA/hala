#ifndef __HALA_EXTENSION_CHECKER_HPP
#define __HALA_EXTENSION_CHECKER_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_mixed_overloads.hpp"

namespace hala{

/*!
 * \internal
 * \file hala_extensions_checker.hpp
 * \ingroup HALABLASCHECKER
 * \brief Check methods for common api calls.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 *
 * Checks for common API calls, e.g., batch_axpy().
 * \endinternal
 */

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)

#ifndef NDEBUG
namespace valid{

template<class VectorLikeA, class VectorLikeX, class VectorLikeY>
bool batch_axpy(int batch_size, VectorLikeA const &alpha, VectorLikeX const &x, VectorLikeY const &y){
    assert( get_size(x) == get_size(y) );
    assert( check_size(alpha, batch_size) );
    assert( check_size(x, get_size_int(x) / batch_size, batch_size) );
    return true;
}

template<class VectorLikeX, class VectorLikeY>
bool batch_dot(int batch_size, VectorLikeX const &x, VectorLikeY const &y){
    assert( get_size(x) == get_size(y) );
    assert( check_size(x, get_size_int(x) / batch_size, batch_size) );
    return true;
}

template<class VectorLikeA, class VectorLikeX>
bool batch_scal(int batch_size, VectorLikeA const &alpha, VectorLikeX &&x){
    assert( batch_size > 0 );
    assert( check_size(alpha, batch_size) );
    assert( check_size(x, batch_size) );
    return true;
}

}
#endif

}

#endif

#endif
