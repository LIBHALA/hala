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
 * Equivalent to ilu.apply() but using functional-style of a call and can be used with hala::engined_vector.
 *
 * Overloads:
 * \code
 *  hala::ilu_apply(ilu, x, r);
 *  hala::ilu_apply(ilu, x, r, num_rhs);
 *  hala::ilu_apply(ilu, x, r, num_rhs, temp_buffer);
 * \endcode
 * The first overload assumes that the number of right-hand-sides is 1 and the last one uses a user
 * allocated buffer scratch space.
 */
template<class ClassILU, class VectorLikeX, class VectorLikeR>
inline void ilu_apply(ClassILU const &ilu, VectorLikeX const &x, VectorLikeR &&r, int num_rhs = 1){
    ilu.apply(x, r, num_rhs);
}

/*!
 * \ingroup HALAWAXSPARSE
 * \brief Functional overload of ILU that creates a temporary buffer.
 *
 * Returns a buffer allocated for the temporary work-space.
 *
 * Overloads:
 * \code
 *  auto workspace = hala::ilu_temp_buffer(ilu, x, r);
 *  auto workspace = hala::ilu_temp_buffer(ilu, x, r, 1);
 * \endcode
 * The fist overload assumes that num_rhs is one.
 * Overloads are provided for cpu and gpu preconditioners and hala::engined_vector.
 */
template<typename T, class VectorLikeX, class VectorLikeR>
std::vector<T> ilu_temp_buffer(cpu_ilu<T> const &, VectorLikeX const&, VectorLikeR &&, int = 1){
    return std::vector<T>();
}

/*!
 * \ingroup HALAWAXSPARSE
 * \brief Functional overload of ILU that returns the buffer size.
 *
 * Overloads:
 * \code
 *  size_t bsize = hala::ilu_buffer_size(ilu, x, r);
 *  size_t bsize = hala::ilu_buffer_size(ilu, x, r, 1);
 * \endcode
 * The fist overload assumes that num_rhs is one.
 * Overloads are provided for cpu and gpu preconditioners and hala::engined_vector.
 */
template<typename T, class VectorLikeX, class VectorLikeR>
size_t ilu_buffer_size(cpu_ilu<T> const &ilu, VectorLikeX const&x, VectorLikeR &&r, int num_rhs = 1){
    return ilu.buffer_size(x, r, num_rhs);
}

#ifndef __HALA_DOXYGEN_SKIP

template<class ClassILU, class VectorLikeX, class VectorLikeR, class VectorLikeT>
inline void ilu_apply(ClassILU const &ilu, VectorLikeX const &x, VectorLikeR &&r, int num_rhs, VectorLikeT &&temp){
    ilu.apply(x, r, num_rhs, temp);
}

#ifdef HALA_ENABLE_GPU
template<typename T, class VectorLikeX, class VectorLikeR>
gpu_vector<T> ilu_temp_buffer(gpu_ilu<T> const &ilu, VectorLikeX const &x, VectorLikeR &&r, int num_rhs = 1){
    return gpu_vector<T>(ilu.buffer_size(x, r, num_rhs), ilu.engine().device());
}
template<typename T, class VectorLikeX, class VectorLikeR>
size_t ilu_buffer_size(gpu_ilu<T> const &ilu, VectorLikeX const&x, VectorLikeR &&r, int num_rhs = 1){
    return ilu.buffer_size(x, r, num_rhs);
}
#endif

#endif

}

#endif
