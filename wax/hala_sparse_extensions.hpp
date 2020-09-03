#ifndef __HALA_SPARSE_EXTENSIONS_HPP
#define __HALA_SPARSE_EXTENSIONS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_blas_extensions.hpp"

namespace hala{

/*!
 * \internal
 * \file hala_sparse_extensions.hpp
 * \ingroup HALACORE
 * \brief Extension for sparse linear algebra methods.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \endinternal
 */

/*!
 * \ingroup HALAWAXEXT
 * \addtogroup HALAWAXSPARSE Sparse extensions
 *
 * Methods that combine multiple dense and sparse calls into handy utilities,
 * e.g., call sparse matrix-matrix multiply with different combinations of row/column formats.
 */

/*!
 * \ingroup HALAWAXSPARSE
 * \brief Functional overload of ILU apply.
 *
 * Equivalent to ilu.apply() but using functional-style of a call.
 */
template<class ClassILU, class VectorLikeX, class VectorLikeR>
inline void ilu_apply(ClassILU const &ilu, VectorLikeX const &x, VectorLikeR &&r, int num_rhs = 1){
    ilu.apply(x, r, num_rhs);
}

}

#endif
