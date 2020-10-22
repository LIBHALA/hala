#ifndef __HALA_NOENGINE_OVERLOADS_HPP
#define __HALA_NOENGINE_OVERLOADS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_sparse_extensions.hpp"

/*!
 * \internal
 * \file hala_noengine_overloads.hpp
 * \ingroup HALAWAXENGINED
 * \brief Overloads for the engined-vector class.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 *
 * The no-engine overloads rely on the compiler to pick one of the overloads given here
 * every time a HALA method is called with an instance of engined_vector.
 * However, technically, the engined_vector is a VectorLike and the engined_vector can
 * be passed to either the no-engine overloads here or the original BLAS methods that
 * assume no engine at all (even in the vector).
 * The compiler has to select the correct template to instantiate, but relies
 * on strict adherence to the standard of finding the "most-specialized" among
 * the overloads.
 * Even with strict standard adherence, there are issues when dealing with temporary
 * variables (r-value references) when passing them to the output vectors.
 *
 * There are several possible solution to this:
 * - The best solution is to constrain the original templates with \b concepts which
 *  is a construct introduced in the C++-20 standard but still missing from main-stream compilers.
 * - Using the C++-14 enable-if construct, constrain the generic templates and ensure that
 *  those cannot be instantiated with engined vectors; however, that would require significant
 *  changes into the code structure and breaking the class definitions and overloads into
 *  multiple files which will make things hard to maintain and harder to document
 *  (each method in doxygen will have to hold an enable-if clause).
 *
 * The solution chosen here will include this file twice where each overload will be present
 * once with l-value reference for the output vector and once with an r-value reference.
 * The __HALAREFREF macro is used to mark the one or two refs, and to count the pass.
 * The __HALA_NOENGINE_OVERLOADS_HPP_PASS1 macro is used to guard certain blocks of code
 * where all vectors are const (e.g., dot, norm2, iamax) and places where two output
 * vectors require additional ref and ref-ref combination of overloads (e.g., rot and rotm).
 *
 * The file contains a macro to guard against multiple inclusions, but the macro is removed
 * in the hala_noengine_structs.hpp file between the two inclusions of this header.
 * \endinternal
 */

#ifndef __HALA_DOXYGEN_SKIP

#ifndef __HALA_REFREF
#define __HALA_REFREF &
#define __HALA_NOENGINE_OVERLOADS_HPP_PASS1
#else
#undef __HALA_REFREF
#undef __HALA_NOENGINE_OVERLOADS_HPP_PASS1
#define __HALA_REFREF &&
#endif

namespace hala{

/***********************************************************************************************************************/
/* BLAS level 1                                                                                                        */
/***********************************************************************************************************************/
template<class cengine, class VectorLikeX, class VectorLikeY>
inline void vcopy(int N, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy){
    assert( check_engines(x, y) );
    vcopy(x.engine(), x.vector(), y.vector(), incx, incy, N);
}
template<class cengine, class VectorLikeX, class VectorLikeY>
inline void vcopy(engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incx = 1, int incy = 1, int N = -1){
    vcopy(N, x, incx, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy);
}

template<class cengine, class VectorLikeX, class VectorLikeY>
inline void vswap(int N, engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int incx, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy){
    assert( check_engines(x, y) );
    vswap(x.engine(), x.vector(), y.vector(), incx, incy, N); // handles the defaults
}
template<class cengine, class VectorLikeX, class VectorLikeY>
inline void vswap(engined_vector<cengine, VectorLikeX> __HALA_REFREF x, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incx = 1, int incy = 1, int N = -1){
    vswap(N, std::forward<engined_vector<cengine, VectorLikeX>>(x), incx, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy);
}

#ifdef __HALA_NOENGINE_OVERLOADS_HPP_PASS1
template<class cengine, class VectorLikeX> inline auto asum(int N, engined_vector<cengine, VectorLikeX> const &x, int incx){
    return asum(x.engine(), N, x.vector(), incx);
}
template<class cengine, class VectorLikeX> inline auto asum(engined_vector<cengine, VectorLikeX> const &x, int incx = 1, int N = -1){
    return asum(x.engine(), x.vector(), incx, N);
}

template<class cengine, class VectorLikeX> inline int iamax(engined_vector<cengine, VectorLikeX> const &x, int incx = 1, int N = -1){
    return iamax(x.engine(), x.vector(), incx, N);
}
template<class cengine, class VectorLikeX> inline int iamax(int N, engined_vector<cengine, VectorLikeX> const &x, int incx){
    return iamax(x.engine(), N, x.vector(), incx);
}

template<bool conjugate = true, class cengine, class VectorLikeX, class VectorLikeY>
inline auto dot(int N, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeY> const &y, int incy){
    assert( check_engines(x, y) );
    return dot<conjugate>(x.engine(), x.vector(), y.vector(), incx, incy, N); // handles the defaults
}
template<bool conjugate = true, class cengine, class VectorLikeX, class VectorLikeY>
inline auto dot(engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> const &y, int incx = 1, int incy = 1, int N = -1){
    return dot<conjugate>(N, x, incx, y, incy);
}

template<class cengine, class VectorLikeX, class VectorLikeY>
inline auto dotu(int N, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeY> const &y, int incy){
    assert( check_engines(x, y) );
    return dot<false>(x.engine(), x.vector(), y.vector(), incx, incy, N); // handles the defaults
}
template<class cengine, class VectorLikeX, class VectorLikeY>
inline auto dotu(engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> const &y, int incx = 1, int incy = 1, int N = -1){
    return dot<false>(N, x, incx, y, incy);
}

template<class cengine, class VectorLikeX>
inline auto norm2(int N, engined_vector<cengine, VectorLikeX> const &x, int incx){
    return norm2(x.engine(), N, x.vector(), incx);
}
template<class cengine, class VectorLikeX>
inline auto norm2(engined_vector<cengine, VectorLikeX> const &x, int incx = 1, int N = -1){
    return norm2(x.engine(), x.vector(), incx, N);
}

template<class cengine, typename T, class VectorLike>
inline void rotmg(T &D1, T &D2, T &X, T const &Y, engined_vector<cengine, VectorLike> &param){ rotmg(param.engine(), D1, D2, X, Y, param.vector()); }
#endif

template<class cengine, typename FS, class VectorLikeX, class VectorLikeY>
inline void axpy(FS alpha, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incx = 1, int incy = 1, int N = -1){
    assert( check_engines(x, y) );
    axpy(x.engine(), alpha, x.vector(), y.vector(), incx, incy, N);
}
template<class cengine, typename FS, class VectorLikeX, class VectorLikeY>
inline void axpy(int N, FS alpha, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy){
    axpy(alpha, x, std::forward<engined_vector<cengine, VectorLikeY>>(y), incx, incy, N);
}

template<class cengine, typename FS, class VectorLikeX>
inline void scal(int N, FS alpha, engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int incx){
    scal(x.engine(), N, alpha, x.vector(), incx);
}
template<class cengine, typename FS, class VectorLikeX>
inline void scal(FS alpha, engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int incx = 1, int N = -1){
    scal(x.engine(), alpha, x.vector(), incx, N);
}

template<class cengine, typename FC, typename FS, class VectorLikeX, class VectorLikeY>
inline void rot(int N, engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int incx, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy, FC C, FS S){
    assert( check_engines(x, y) );
    rot(x.engine(), x.vector(), y.vector(), C, S, incx, incy, N);
}
template<class cengine, typename FC, typename FS, class VectorLikeX, class VectorLikeY>
inline void rot(engined_vector<cengine, VectorLikeX> __HALA_REFREF x, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, FC C, FS S, int incx = 1, int incy = 1, int N = -1){
    rot(N, std::forward<engined_vector<cengine, VectorLikeX>>(x), incx, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy, C, S);
}

template<class cengine, class VectorLikeX, class VectorLikeY, class VectorLikeP>
inline void rotm(int N, engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int incx, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy,
                 engined_vector<cengine, VectorLikeP> const &param){
    assert( check_engines(x, y) );
    rotm(x.engine(), x.vector(), y.vector(), param.vector(), incx, incy, N);
}
template<class cengine, class VectorLikeX, class VectorLikeY, class VectorLikeP>
inline void rotm(engined_vector<cengine, VectorLikeX> __HALA_REFREF x, engined_vector<cengine, VectorLikeY> __HALA_REFREF y,
                 engined_vector<cengine, VectorLikeP> const &param, int incx = 1, int incy = 1, int N = -1){
    rotm(N, std::forward<engined_vector<cengine, VectorLikeX>>(x), incx, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy, param);
}

#ifdef __HALA_NOENGINE_OVERLOADS_HPP_PASS1
template<class cengine, typename FC, typename FS, class VectorLikeX, class VectorLikeY>
inline void rot(int N, engined_vector<cengine, VectorLikeX> &&x, int incx, engined_vector<cengine, VectorLikeY> &y, int incy, FC C, FS S){
    assert( check_engines(x, y) );
    rot(x.engine(), x.vector(), y.vector(), C, S, incx, incy, N);
}
template<class cengine, typename FC, typename FS, class VectorLikeX, class VectorLikeY>
inline void rot(engined_vector<cengine, VectorLikeX> &x, engined_vector<cengine, VectorLikeY> &&y, FC C, FS S, int incx = 1, int incy = 1, int N = -1){
    rot(N, std::forward<engined_vector<cengine, VectorLikeX>>(x), incx, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy, C, S);
}

template<class cengine, class VectorLikeX, class VectorLikeY, class VectorLikeP>
inline void rotm(int N, engined_vector<cengine, VectorLikeX> &&x, int incx, engined_vector<cengine, VectorLikeY> &y, int incy,
                 engined_vector<cengine, VectorLikeP> const &param){
    assert( check_engines(x, y) );
    rotm(x.engine(), x.vector(), y.vector(), param.vector(), incx, incy, N);
}
template<class cengine, class VectorLikeX, class VectorLikeY, class VectorLikeP>
inline void rotm(engined_vector<cengine, VectorLikeX> &x, engined_vector<cengine, VectorLikeY> &&y,
                 engined_vector<cengine, VectorLikeP> const &param, int incx = 1, int incy = 1, int N = -1){
    rotm(N, std::forward<engined_vector<cengine, VectorLikeX>>(x), incx, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy, param);
}
#endif

/***********************************************************************************************************************/
/* BLAS level 2                                                                                                        */
/***********************************************************************************************************************/
template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gemv(char trans, int M, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeX> const &x, int incx, FSb beta,
                 engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy){
    assert( check_engines(A, x, y) );
    gemv(A.engine(), trans, M, N, alpha, A.vector(), x.vector(), beta, y.vector(), lda, incx, incy);
}
template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gemv(char trans, int M, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> const &x, FSb beta,
                 engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int lda = -1, int incx = 1, int incy = 1){
    gemv(trans, M, N, alpha, A, lda, x, incx, beta, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy);
}

template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gbmv(char trans, int M, int N, int kl, int ku, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeX> const &x, int incx, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy){
    assert( check_engines(A, x, y) );
    gbmv(A.engine(), trans, M, N, kl, ku, alpha, A.vector(), x.vector(), beta, y.vector(), lda, incx, incy);
}
template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void gbmv(char trans, int M, int N, int kl, int ku, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> const &x, FSb beta,
                 engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int lda = -1, int incx = 1, int incy = 1){
    gbmv(trans, M, N, kl, ku, alpha, A, lda, x, incx, beta, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy);
}

template<bool conjugate = false, class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void symv(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeX> const &x, int incx, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy){
    assert( check_engines(A, x, y) );
    symv<conjugate>(A.engine(), uplo, N, alpha, A.vector(), x.vector(), beta, y.vector(), lda, incx, incy);
}
template<bool conjugate = false, class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void symv(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> const &x, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int lda = -1, int incx = 1, int incy = 1){
    symv<conjugate>(uplo, N, alpha, A, lda, x, incx, beta, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy);
}

template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hemv(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeX> const &x, int incx, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy){
    assert( check_engines(A, x, y) );
    hemv(A.engine(), uplo, N, alpha, A.vector(), x.vector(), beta, y.vector(), lda, incx, incy);
}
template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hemv(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> const &x, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int lda = -1, int incx = 1, int incy = 1){
    hemv(uplo, N, alpha, A, lda, x, incx, beta, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy);
}

template<bool conjugate = false, class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void sbmv(char uplo, int N, int k, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeX> const &x, int incx, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy){
    assert( check_engines(A, x, y) );
    sbmv<conjugate>(A.engine(), uplo, N, k, alpha, A.vector(), x.vector(), beta, y.vector(), lda, incx, incy);
}
template<bool conjugate = false, class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void sbmv(char uplo, int N, int k, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> const &x, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int lda = -1, int incx = 1, int incy = 1){
    sbmv<conjugate>(uplo, N, k, alpha, A, lda, x, incx, beta, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy);
}

template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hbmv(char uplo, int N, int k, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeX> const &x, int incx, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy){
    assert( check_engines(A, x, y) );
    hbmv(A.engine(), uplo, N, k, alpha, A.vector(), x.vector(), beta, y.vector(), lda, incx, incy);
}
template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hbmv(char uplo, int N, int k, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> const &x, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int lda = -1, int incx = 1, int incy = 1){
    hbmv(uplo, N, k, alpha, A, lda, x, incx, beta, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy);
}

template<bool conjugate = false, class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void spmv(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> const &x, int incx, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy){
    assert( check_engines(A, x, y) );
    spmv<conjugate>(A.engine(), uplo, N, alpha, A.vector(), x.vector(), beta, y.vector(), incx, incy);
}
template<bool conjugate = false, class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void spmv(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> const &x, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incx = 1, int incy = 1){
    spmv<conjugate>(uplo, N, alpha, A, x, incx, beta, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy);
}

template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hpmv(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> const &x, int incx, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incy){
    spmv<true>(A.engine(), uplo, N, alpha, A.vector(), x.vector(), beta, y.vector(), incx, incy);
}
template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeX, class VectorLikeY>
inline void hpmv(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> const &x, FSb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, int incx = 1, int incy = 1){
    spmv<true>(uplo, N, alpha, A, x, incx, beta, std::forward<engined_vector<cengine, VectorLikeY>>(y), incy);
}

template<class cengine, class VectorLikeA, class VectorLikeX>
inline void trmv(char uplo, char trans, char diag, int N, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int incx){
    assert( check_engines(A, x) );
    trmv(A.engine(), uplo, trans, diag, N, A.vector(), x.vector(), lda, incx);
}
template<class cengine, class VectorLikeA, class VectorLikeX>
inline void trmv(char uplo, char trans, char diag, int N, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int lda = -1, int incx = 1){
    trmv(uplo, trans, diag, N, A, lda, std::forward<engined_vector<cengine, VectorLikeX>>(x), incx);
}

template<class cengine, class VectorLikeA, class VectorLikeX>
inline void trsv(char uplo, char trans, char diag, int N, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int incx){
    assert( check_engines(A, x) );
    trsv(A.engine(), uplo, trans, diag, N, A.vector(), x.vector(), lda, incx);
}
template<class cengine, class VectorLikeA, class VectorLikeX>
inline void trsv(char uplo, char trans, char diag, int N, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int lda = -1, int incx = 1){
    trsv(uplo, trans, diag, N, A, lda, std::forward<engined_vector<cengine, VectorLikeX>>(x), incx);
}

template<class cengine, class VectorLikeA, class VectorLikeX>
inline void tbmv(char uplo, char trans, char diag, int N, int k, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int incx){
    assert( check_engines(A, x) );
    tbmv(A.engine(), uplo, trans, diag, N, k, A.vector(), x.vector(), lda, incx);
}
template<class cengine, class VectorLikeA, class VectorLikeX>
inline void tbmv(char uplo, char trans, char diag, int N, int k, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int lda = -1, int incx = 1){
    tbmv(uplo, trans, diag, N, k, A, lda, std::forward<engined_vector<cengine, VectorLikeX>>(x), incx);
}

template<class cengine, class VectorLikeA, class VectorLikeX>
inline void tbsv(char uplo, char trans, char diag, int N, int k, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int incx){
    assert( check_engines(A, x) );
    tbsv(A.engine(), uplo, trans, diag, N, k, A.vector(), x.vector(), lda, incx);
}
template<class cengine, class VectorLikeA, class VectorLikeX>
inline void tbsv(char uplo, char trans, char diag, int N, int k, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int lda = -1, int incx = 1){
    tbsv(uplo, trans, diag, N, k, A, lda, std::forward<engined_vector<cengine, VectorLikeX>>(x), incx);
}

template<class cengine, class VectorLikeA, class VectorLikeX>
inline void tpmv(char uplo, char trans, char diag, int N, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int incx = 1){
    assert( check_engines(A, x) );
    tpmv(A.engine(), uplo, trans, diag, N, A.vector(), x.vector(), incx);
}
template<class cengine, class VectorLikeA, class VectorLikeX>
inline void tpsv(char uplo, char trans, char diag, int N, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeX> __HALA_REFREF x, int incx = 1){
    tpsv(A.engine(), uplo, trans, diag, N, A.vector(), x.vector(), incx);
}

template<bool conjugate = true, class cengine, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void ger(int M, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeY> const &y, int incy,
                engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int lda){
    assert( check_engines(A, x, y) );
    ger<conjugate>(x.engine(), M, N, alpha, x.vector(), y.vector(), A.vector(), incx, incy, lda);
}
template<bool conjugate = true, class cengine, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void ger(int M, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> const &y,
                engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int incx = 1, int incy = 1, int lda = -1){
    ger<conjugate>(M, N, alpha, x, incx, y, incy, std::forward<engined_vector<cengine, VectorLikeA>>(A), lda);
}

template<class cengine, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void geru(int M, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeY> const &y, int incy,
                engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int lda){
    assert( check_engines(A, x, y) );
    ger<false>(x.engine(), M, N, alpha, x.vector(), y.vector(), A.vector(), incx, incy, lda);
}
template<class cengine, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void geru(int M, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> const &y,
                engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int incx = 1, int incy = 1, int lda = -1){
    ger<false>(M, N, alpha, x, incx, y, incy, std::forward<engined_vector<cengine, VectorLikeA>>(A), lda);
}

template<bool conjugate = false, class cengine, typename FSa, class VectorLikeX, class VectorLikeA>
inline void syr(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int lda){
    assert( check_engines(A, x) );
    syr<conjugate>(x.engine(), uplo, N, alpha, x.vector(), A.vector(), incx, lda);
}
template<bool conjugate = false, class cengine, typename FSa, class VectorLikeX, class VectorLikeA>
inline void syr(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x,
                engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int incx = 1, int lda = -1){
    syr<conjugate>(uplo, N, alpha, x, incx, std::forward<engined_vector<cengine, VectorLikeA>>(A), lda);
}

template<class cengine, typename FSa, class VectorLikeX, class VectorLikeA>
inline void her(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int lda){
    assert( check_engines(A, x) );
    syr<true>(x.engine(), uplo, N, alpha, x.vector(), A.vector(), incx, lda);
}
template<class cengine, typename FSa, class VectorLikeX, class VectorLikeA>
inline void her(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x,
                engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int incx = 1, int lda = -1){
    syr<true>(uplo, N, alpha, x, incx, std::forward<engined_vector<cengine, VectorLikeA>>(A), lda);
}

template<bool conjugate = false, class cengine, typename FSa, class VectorLikeX, class VectorLikeA>
inline void spr(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeA> __HALA_REFREF A){
    assert( check_engines(A, x) );
    spr<conjugate>(x.engine(), uplo, N, alpha, x.vector(), A.vector(), incx);
}
template<bool conjugate = false, class cengine, typename FSa, class VectorLikeX, class VectorLikeA>
inline void spr(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int incx = 1){
    spr<conjugate>(uplo, N, alpha, x, incx, std::forward<engined_vector<cengine, VectorLikeA>>(A));
}

template<class cengine, typename FSa, class VectorLikeX, class VectorLikeA>
inline void hpr(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeA> __HALA_REFREF A){
    assert( check_engines(A, x) );
    spr<true>(x.engine(), uplo, N, alpha, x.vector(), A.vector(), incx);
}
template<class cengine, typename FSa, class VectorLikeX, class VectorLikeA>
inline void hpr(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int incx = 1){
    spr<true>(uplo, N, alpha, x, incx, std::forward<engined_vector<cengine, VectorLikeA>>(A));
}

template<bool conjugate = false, class cengine, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void syr2(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeY> const &y, int incy,
                 engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int lda){
    assert( check_engines(A, x, y) );
    syr2<conjugate>(x.engine(), uplo, N, alpha, x.vector(), y.vector(), A.vector(), incx, incy, lda);
}
template<bool conjugate = false, class cengine, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void syr2(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> const &y,
                 engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int incx = 1, int incy = 1, int lda = -1){
    syr2<conjugate>(uplo, N, alpha, x, incx, y, incy, std::forward<engined_vector<cengine, VectorLikeA>>(A), lda);
}

template<class cengine, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void her2(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeY> const &y, int incy,
                 engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int lda){
    assert( check_engines(A, x, y) );
    syr2<true>(x.engine(), uplo, N, alpha, x.vector(), y.vector(), A.vector(), incx, incy, lda);
}
template<class cengine, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void her2(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> const &y,
                 engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int incx = 1, int incy = 1, int lda = -1){
    syr2<true>(uplo, N, alpha, x, incx, y, incy, std::forward<engined_vector<cengine, VectorLikeA>>(A), lda);
}

template<bool conjugate = false, class cengine, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void spr2(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeY> const &y, int incy,
                 engined_vector<cengine, VectorLikeA> __HALA_REFREF A){
    assert( check_engines(A, x, y) );
    spr2<conjugate>(x.engine(), uplo, N, alpha, x.vector(), y.vector(), A.vector(), incx, incy);
}
template<bool conjugate = false, class cengine, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void spr2(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> const &y,
                 engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int incx = 1, int incy = 1){
    spr2<conjugate>(uplo, N, alpha, x, incx, y, incy, std::forward<engined_vector<cengine, VectorLikeA>>(A));
}

template<class cengine, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void hpr2(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, int incx, engined_vector<cengine, VectorLikeY> const &y, int incy,
                 engined_vector<cengine, VectorLikeA> __HALA_REFREF A){
    assert( check_engines(A, x, y) );
    spr2<true>(x.engine(), uplo, N, alpha, x.vector(), y.vector(), A.vector(), incx, incy);
}
template<class cengine, typename FSa, class VectorLikeX, class VectorLikeY, class VectorLikeA>
inline void hpr2(char uplo, int N, FSa alpha, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> const &y,
                 engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int incx = 1, int incy = 1){
    spr2<true>(uplo, N, alpha, x, incx, y, incy, std::forward<engined_vector<cengine, VectorLikeA>>(A));
}

/***********************************************************************************************************************/
/* BLAS level 3                                                                                                        */
/***********************************************************************************************************************/
template<class cengine, typename FS, class VectorLikeA, class VectorLikeB>
inline void trmm(char side, char uplo, char trans, char diag, int M, int N, FS alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeB> __HALA_REFREF B, int ldb){
    assert( check_engines(A, B) );
    trmm(A.engine(), side, uplo, trans, diag, M, N, alpha, A.vector(), B.vector(), lda, ldb);
}
template<class cengine, typename FS, class VectorLikeA, class VectorLikeB>
inline void trmm(char side, char uplo, char trans, char diag, int M, int N, FS alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeB> __HALA_REFREF B, int lda = -1, int ldb = -1){
    trmm(side, uplo, trans, diag, M, N, alpha, A, lda, std::forward<engined_vector<cengine, VectorLikeB>>(B), ldb);
}

template<class cengine, typename FS, class VectorLikeA, class VectorLikeB>
inline void trsm(char side, char uplo, char trans, char diag, int M, int N, FS alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeB> __HALA_REFREF B, int ldb){
    assert( check_engines(A, B) );
    trsm(A.engine(), side, uplo, trans, diag, M, N, alpha, A.vector(), B.vector(), lda, ldb);
}
template<class cengine, typename FS, class VectorLikeA, class VectorLikeB>
inline void trsm(char side, char uplo, char trans, char diag, int M, int N, FS alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeB> __HALA_REFREF B, int lda = -1, int ldb = -1){
    trsm(side, uplo, trans, diag, M, N, alpha, A, lda, std::forward<engined_vector<cengine, VectorLikeB>>(B), ldb);
}

template<bool conjugate = false, class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void syrk(char uplo, char trans, int N, int K, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldc){
    assert( check_engines(A, C) );
    syrk<conjugate>(A.engine(), uplo, trans, N, K, alpha, A.vector(), beta, C.vector(), lda, ldc);
}
template<bool conjugate = false, class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void syrk(char uplo, char trans, int N, int K, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int lda = -1, int ldc = -1){
    syrk<conjugate>(uplo, trans, N, K, alpha, A, lda, beta, std::forward<engined_vector<cengine, VectorLikeC>>(C), ldc);
}

template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void herk(char uplo, char trans, int N, int K, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldc){
    assert( check_engines(A, C) );
    syrk<true>(A.engine(), uplo, trans, N, K, alpha, A.vector(), beta, C.vector(), lda, ldc);
}
template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeC>
inline void herk(char uplo, char trans, int N, int K, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int lda = -1, int ldc = -1){
    syrk<true>(uplo, trans, N, K, alpha, A, lda, beta, std::forward<engined_vector<cengine, VectorLikeC>>(C), ldc);
}

template<bool conjugate = false, class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void syr2k(char uplo, char trans, int N, int K, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                  engined_vector<cengine, VectorLikeB> const &B, int ldb,
                  FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldc){
    assert( check_engines(A, B, C) );
    syr2k<conjugate>(A.engine(), uplo, trans, N, K, alpha, A.vector(), B.vector(), beta, C.vector(), lda, ldb, ldc);
}
template<bool conjugate = false, class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void syr2k(char uplo, char trans, int N, int K, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                  engined_vector<cengine, VectorLikeB> const &B,
                  FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int lda = -1, int ldb = -1, int ldc = -1){
    syr2k<conjugate>(uplo, trans, N, K, alpha, A, lda, B, ldb, beta, std::forward<engined_vector<cengine, VectorLikeC>>(C), ldc);
}

template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void her2k(char uplo, char trans, int N, int K, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                  engined_vector<cengine, VectorLikeB> const &B, int ldb,
                  FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldc){
    assert( check_engines(A, B, C) );
    syr2k<true>(A.engine(), uplo, trans, N, K, alpha, A.vector(), B.vector(), beta, C.vector(), lda, ldb, ldc);
}
template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void her2k(char uplo, char trans, int N, int K, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                  engined_vector<cengine, VectorLikeB> const &B,
                  FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int lda = -1, int ldb = -1, int ldc = -1){
    syr2k<true>(uplo, trans, N, K, alpha, A, lda, B, ldb, beta, std::forward<engined_vector<cengine, VectorLikeC>>(C), ldc);
}

template<bool conjugate = false, class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void symm(char side, char uplo, int M, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeB> const &B, int ldb,
                 FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldc){
    assert( check_engines(A, B, C) );
    symm<conjugate>(A.engine(), side, uplo, M, N, alpha, A.vector(), B.vector(), beta, C.vector(), lda, ldb, ldc);
}
template<bool conjugate = false, class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void symm(char side, char uplo, int M, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeB> const &B,
                 FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int lda = -1, int ldb = -1, int ldc = -1){
    symm<conjugate>(side, uplo, M, N, alpha, A, lda, B, ldb, beta, std::forward<engined_vector<cengine, VectorLikeC>>(C), ldc);
}

template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void hemm(char side, char uplo, int M, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeB> const &B, int ldb,
                 FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldc){
    assert( check_engines(A, B, C) );
    symm<true>(A.engine(), side, uplo, M, N, alpha, A.vector(), B.vector(), beta, C.vector(), lda, ldb, ldc);
}
template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void hemm(char side, char uplo, int M, int N, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeB> const &B,
                 FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int lda = -1, int ldb = -1, int ldc = -1){
    symm<true>(side, uplo, M, N, alpha, A, lda, B, ldb, beta, std::forward<engined_vector<cengine, VectorLikeC>>(C), ldc);
}

template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void gemm(char transa, char transb, int M, int N, int K, FSa alpha, engined_vector<cengine, VectorLikeA> const &A, int lda,
                 engined_vector<cengine, VectorLikeB> const &B, int ldb,
                 FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldc){
    assert( check_engines(A, B, C) );
    gemm(A.engine(), transa, transb, M, N, K, alpha, A.vector(), B.vector(), beta, C.vector(), lda, ldb, ldc);
}
template<class cengine, typename FSa, typename FSb, class VectorLikeA, class VectorLikeB, class VectorLikeC>
inline void gemm(char transa, char transb, int M, int N, int K, FSa alpha, engined_vector<cengine, VectorLikeA> const &A,
                 engined_vector<cengine, VectorLikeB> const &B,
                 FSb beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int lda = -1, int ldb = -1, int ldc = -1){
    gemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, std::forward<engined_vector<cengine, VectorLikeC>>(C), ldc);
}

/***********************************************************************************************************************/
/* SPARSE                                                                                                              */
/***********************************************************************************************************************/
template<class cengine, typename FPa, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeX, typename FPb, class VectorLikeY>
void sparse_gemv(char trans, int M, int N,
                 FPa alpha, engined_vector<cengine, VectorLikeP> const &pntr, engined_vector<cengine, VectorLikeI> const &indx,
                 engined_vector<cengine, VectorLikeV> const &vals,
                 engined_vector<cengine, VectorLikeX> const &x, FPb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y){
    assert( check_engines(pntr, indx, vals, x, y) );
    sparse_gemv(vals.engine(), trans, M, N, alpha, pntr.vector(), indx.vector(), vals.vector(), x.vector(), beta, y.vector());
}
template<class cengine, typename FSA, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeB, typename FSB, class VectorLikeC>
inline void sparse_gemm(char transa, char transb, int M, int N, int K,
                        FSA alpha, engined_vector<cengine, VectorLikeP> const &pntr, engined_vector<cengine, VectorLikeI> const &indx,
                        engined_vector<cengine, VectorLikeV> const &vals,
                        engined_vector<cengine, VectorLikeB> const &B, int ldb, FSB beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldc){
    assert( check_engines(pntr, indx, vals, B, C) );
    sparse_gemm(vals.engine(), transa, transb, M, N, K, alpha, pntr.vector(), indx.vector(), vals.vector(), B.vector(), beta, C.vector(), ldb, ldc);
}

template<class cengine, typename FSA, class VectorLikeP, class VectorLikeI, class VectorLikeV, class VectorLikeB, typename FSB, class VectorLikeC>
inline void sparse_gemm(char transa, char transb, int M, int N, int K,
                        FSA alpha, engined_vector<cengine, VectorLikeP> const &pntr, engined_vector<cengine, VectorLikeI> const &indx,
                        engined_vector<cengine, VectorLikeV> const &vals,
                        engined_vector<cengine, VectorLikeB> const &B, FSB beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldb = -1, int ldc = -1){
    sparse_gemm(transa, transb, M, N, K, alpha, pntr, indx, vals, B, ldb, beta, std::forward<engined_vector<cengine, VectorLikeC>>(C), ldc);
}

template<class cengine, typename MatrixType, typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
size_t sparse_gemv_buffer_size(MatrixType const &matrix, char trans, FPa alpha, engined_vector<cengine, VectorLikeX> const &x, FPb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y){
    return matrix.gemv_buffer_size(trans, alpha, x.vector(), beta, y.vector());
}
template<class cengine, typename MatrixType, typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
void sparse_gemv(MatrixType const &matrix, char trans, FPa alpha, engined_vector<cengine, VectorLikeX> const &x, FPb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y){
    matrix.gemv(trans, alpha, x.vector(), beta, y.vector());
}
template<class cengine, typename MatrixType, typename FPa, class VectorLikeX, typename FPb, class VectorLikeY, class VectorLikeBuff>
void sparse_gemv(MatrixType const &matrix, char trans, FPa alpha, engined_vector<cengine, VectorLikeX> const &x, FPb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, engined_vector<cengine, VectorLikeBuff> &temp){
    matrix.gemv(trans, alpha, x.vector(), beta, y.vector(), temp.vector());
}
template<class cengine, typename MatrixType, typename FPa, class VectorLikeX, typename FPb, class VectorLikeY, class VectorLikeBuff>
void sparse_gemv(MatrixType const &matrix, char trans, FPa alpha, engined_vector<cengine, VectorLikeX> const &x, FPb beta, engined_vector<cengine, VectorLikeY> __HALA_REFREF y, engined_vector<cengine, VectorLikeBuff> &&temp){
    matrix.gemv(trans, alpha, x.vector(), beta, y.vector(), temp.vector());
}
template<class cengine, typename MatrixType, typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
size_t sparse_gemm_buffer_size(MatrixType const &matrix, char transa, char transb, int b_rows, int b_cols, FSA alpha, engined_vector<cengine, VectorLikeB> const &B, int ldb, FSB beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldc){
    return matrix.gemm_buffer_size(transa, transb, b_rows, b_cols, alpha, B.vector(), ldb, beta, C.vector(), ldc);
}
template<class cengine, typename MatrixType, typename FSA, class VectorLikeB, typename FSB, class VectorLikeC, class VectorLikeBuff>
void sparse_gemm(MatrixType const &matrix, char transa, char transb, int b_rows, int b_cols, FSA alpha, engined_vector<cengine, VectorLikeB> const &B, int ldb, FSB beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldc, engined_vector<cengine, VectorLikeBuff> &temp){
    matrix.gemm(transa, transb, b_rows, b_cols, alpha, B.vector(), ldb, beta, C.vector(), ldc, temp.vector());
}
template<class cengine, typename MatrixType, typename FSA, class VectorLikeB, typename FSB, class VectorLikeC, class VectorLikeBuff>
void sparse_gemm(MatrixType const &matrix, char transa, char transb, int b_rows, int b_cols, FSA alpha, engined_vector<cengine, VectorLikeB> const &B, int ldb, FSB beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldc, engined_vector<cengine, VectorLikeBuff> &&temp){
    matrix.gemm(transa, transb, b_rows, b_cols, alpha, B.vector(), ldb, beta, C.vector(), ldc, temp.vector());
}
template<class cengine, typename MatrixType, typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
void sparse_gemm(MatrixType const &matrix, char transa, char transb, int b_rows, int b_cols, FSA alpha, engined_vector<cengine, VectorLikeB> const &B, int ldb, FSB beta, engined_vector<cengine, VectorLikeC> __HALA_REFREF C, int ldc){
    matrix.gemm(transa, transb, b_rows, b_cols, alpha, B.vector(), ldb, beta, C.vector(), ldc);
}

#ifdef __HALA_NOENGINE_OVERLOADS_HPP_PASS1
template<class cengine, class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_sparse_matrix(int num_cols, engined_vector<cengine, VectorLikeP> const &pntr,
                                      engined_vector<cengine, VectorLikeI> const &indx, engined_vector<cengine, VectorLikeV> const &vals){
    return make_sparse_matrix(vals.engine(), num_cols, pntr.vector(), indx.vector(), vals.vector());
}
template<class cengine, class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_triangular_matrix(char uplo, char diag,
                            engined_vector<cengine, VectorLikeP> const &pntr, engined_vector<cengine, VectorLikeI> const &indx,
                            engined_vector<cengine, VectorLikeV> const &vals, char policy = 'N'){
    assert( check_engines(pntr, indx, vals) );
    return make_triangular_matrix(vals.engine(), uplo, diag, pntr.vector(), indx.vector(), vals.vector(), policy);
}
#endif

/***********************************************************************************************************************/
/* LAPACK                                                                                                              */
/***********************************************************************************************************************/
#ifdef __HALA_NOENGINE_OVERLOADS_HPP_PASS1
template<class cengine, class VectorLikeA>
size_t potrf_size(char uplo, int N, engined_vector<cengine, VectorLikeA> const &A, int lda = -1){
    return potrf_size(A.engine(), uplo, N, A, lda);
}
template<class cengine, class VectorLikeA>
size_t getrf_size(int M, int N, engined_vector<cengine, VectorLikeA> const &A, int lda = -1){
    return getrf_size(A.engine(), M, N, A, lda);
}
#endif
template<class cengine, class VectorLikeA, class VectorLikeW>
inline void potrf(char uplo, int N, engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int lda,
                  engined_vector<cengine, VectorLikeW> __HALA_REFREF W){
    potrf(A.engine(), uplo, N, A.vector(), lda, W.vector());
}
template<class cengine, class VectorLikeA, class VectorLikeW = std::vector<get_scalar_type<VectorLikeA>>>
inline void potrf(char uplo, int N, engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int lda = -1){
    potrf(A.engine(), uplo, N, A.vector(), lda);
}
template<class cengine, class VectorLikeA, class VectorLikeB>
inline void potrs(char uplo, int N, int nrhs, engined_vector<cengine, VectorLikeA> const &A, int lda,
                  engined_vector<cengine, VectorLikeB> __HALA_REFREF B, int ldb){
    check_engines(A, B);
    potrs(A.engine(), uplo, N, nrhs, A.vector(), B.vector(), lda, ldb);
}
template<class cengine, class VectorLikeA, class VectorLikeB>
inline void potrs(char uplo, int N, int nrhs, engined_vector<cengine, VectorLikeA> const &A,
                  engined_vector<cengine, VectorLikeB> __HALA_REFREF B, int lda = -1, int ldb = -1){
    potrs(uplo, N, nrhs, A, lda, std::forward<engined_vector<cengine, VectorLikeB>>(B), ldb);
}

template<class cengine, class VectorLikeA, class VectorLikeI, class VectorLikeW>
inline void getrf(int M, int N, engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int lda,
                  engined_vector<cengine, VectorLikeW> __HALA_REFREF ipiv,
                  engined_vector<cengine, VectorLikeW> __HALA_REFREF W){
    getrf(A.engine(), M, N, A.vector(), ipiv.vector(), lda, W.vector());
}
template<class cengine, class VectorLikeA, class VectorLikeI>
inline void getrf(int M, int N, engined_vector<cengine, VectorLikeA> __HALA_REFREF A, int lda,
                  engined_vector<cengine, VectorLikeI> __HALA_REFREF ipiv){
    getrf(A.engine(), M, N, A.vector(), ipiv.vector(), lda);
}
template<class cengine, class VectorLikeA, class VectorLikeI, class VectorLikeW>
inline void getrf(int M, int N, engined_vector<cengine, VectorLikeA> __HALA_REFREF A,
                  engined_vector<cengine, VectorLikeI> __HALA_REFREF ipiv, int lda,
                  engined_vector<cengine, VectorLikeW> __HALA_REFREF W){
    getrf(A.engine(), M, N, A.vector(), ipiv.vector(), lda, W.vector());
}
template<class cengine, class VectorLikeA, class VectorLikeI>
inline void getrf(int M, int N, engined_vector<cengine, VectorLikeA> __HALA_REFREF A,
                  engined_vector<cengine, VectorLikeI> __HALA_REFREF ipiv, int lda = -1){
    getrf(A.engine(), M, N, A.vector(), ipiv.vector(), lda);
}
template<class cengine, class VectorLikeA, class VectorLikeI, class VectorLikeB>
inline void getrs(char trans, int N, int nrhs, engined_vector<cengine, VectorLikeA> const &A, int lda,
                  engined_vector<cengine, VectorLikeI> const &ipiv,
                  engined_vector<cengine, VectorLikeB> __HALA_REFREF B, int ldb){
    check_engines(A, ipiv, B);
    getrs(A.engine(), trans, N, nrhs, A.vector(), ipiv.vector(), B.vector(), lda, ldb);
}
template<class cengine, class VectorLikeA, class VectorLikeI, class VectorLikeB>
inline void getrs(char trans, int N, int nrhs, engined_vector<cengine, VectorLikeA> const &A,
                  engined_vector<cengine, VectorLikeI> const &ipiv,
                  engined_vector<cengine, VectorLikeB> __HALA_REFREF B, int lda = -1, int ldb = -1){
    getrs(trans, N, nrhs, A, lda, ipiv, std::forward<engined_vector<cengine, VectorLikeB>>(B), ldb);
}


/***********************************************************************************************************************/
/* EXTENSIONS                                                                                                          */
/***********************************************************************************************************************/
template<class cengine, typename FPA, class VectorLikeA, class VectorLikeX, class VectorLikeY>
void batch_axpy(int batch_size, FPA alphac, engined_vector<cengine, VectorLikeA> const &alpha,
                engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> __HALA_REFREF y){
    assert( check_engines(alpha, x, y) );
    batch_axpy(alpha.engine(), batch_size, alphac, alpha.vector(), x.vector(), y.vector());
}
template<bool conjugate = true, class cengine, class VectorLikeX, class VectorLikeY, class VectorLikeR>
void batch_dot(int batch_size, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> const &y,
               engined_vector<cengine, VectorLikeR> __HALA_REFREF result){
    assert( check_engines(x, y, result) );
    batch_dot<conjugate>(x.engine(), batch_size, x.vector(), y.vector(), result.vector());
}
template<class cengine, class VectorLikeX, class VectorLikeY, class VectorLikeR>
void batch_dotu(int batch_size, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> const &y,
               engined_vector<cengine, VectorLikeR> __HALA_REFREF result){
    assert( check_engines(x, y, result) );
    batch_dot<false>(x.engine(), batch_size, x.vector(), y.vector(), result.vector());
}
template<class cengine, class VectorLikeA, class VectorLikeX>
void batch_scal(int batch_size, engined_vector<cengine, VectorLikeA> const &alpha, engined_vector<cengine, VectorLikeX> __HALA_REFREF x){
    assert( check_engines(alpha, x) );
    batch_scal(alpha.engine(), batch_size, alpha.vector(), x.vector());
}
template<class cengine, class VectorLikeX, class VectorLikeY, class VectorLikeZ>
void vdivide(engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeY> const &y,
             engined_vector<cengine, VectorLikeZ> __HALA_REFREF z){
    assert( check_engines(x, y, z) );
    vdivide(x.engine(), x.vector(), y.vector(), z.vector());
}

#ifdef __HALA_NOENGINE_OVERLOADS_HPP_PASS1
template<class cengine, class VectorLike>
auto batch_max_norm2(int batch_size, engined_vector<cengine, VectorLike> const &x){
    return batch_max_norm2(x.engine(), batch_size, x.vector());
}

template<class cengine, class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_ilu(engined_vector<cengine, VectorLikeP> const &pntr, engined_vector<cengine, VectorLikeI> const &indx,
              engined_vector<cengine, VectorLikeV> const &vals, char policy){
    assert( check_engines(pntr, indx, vals) );
    return make_ilu(pntr.engine(), pntr.vector(), indx.vector(), vals.vector(), policy);
}
#endif

template<class cengine, class ClassILU, class VectorLikeX, class VectorLikeR>
inline void ilu_apply(ClassILU const &ilu, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeR> __HALA_REFREF r, int num_rhs = 1){
    assert( check_engines(x, r) );
    static_assert(std::is_same<typename ClassILU::engine_type, cengine>::value,
                  "Using compatible compute engines in ilu_apply()");
    ilu.apply(x.vector(), r.vector(), num_rhs);
}

template<class cengine, class ClassILU, class VectorLikeX, class VectorLikeR, class VectorLikeT>
inline void ilu_apply(ClassILU const &ilu, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeR> __HALA_REFREF r, int num_rhs, engined_vector<cengine, VectorLikeT> &temp){
    assert( check_engines(x, r) );
    static_assert(std::is_same<typename ClassILU::engine_type, cengine>::value,
                  "Using compatible compute engines in ilu_apply()");
    ilu.apply(x.vector(), r.vector(), num_rhs, temp.vector());
}
template<class cengine, class ClassILU, class VectorLikeX, class VectorLikeR, class VectorLikeT>
inline void ilu_apply(ClassILU const &ilu, engined_vector<cengine, VectorLikeX> const &x, engined_vector<cengine, VectorLikeR> __HALA_REFREF r, int num_rhs, engined_vector<cengine, VectorLikeT> &&temp){
    assert( check_engines(x, r) );
    static_assert(std::is_same<typename ClassILU::engine_type, cengine>::value,
                  "Using compatible compute engines in ilu_apply()");
    ilu.apply(x.vector(), r.vector(), num_rhs, temp.vector());
}

template<typename T, class VectorLikeX, class VectorLikeR>
auto ilu_temp_buffer(cpu_ilu<T> const &, engined_vector<cpu_engine, VectorLikeX> const &x, engined_vector<cpu_engine, VectorLikeR> __HALA_REFREF, int = 1){
    return engined_vector<cpu_engine, std::vector<T>>(x.engine());
}
template<typename T, class VectorLikeX, class VectorLikeR>
size_t ilu_buffer_size(cpu_ilu<T> const &ilu, engined_vector<cpu_engine, VectorLikeX> const&x, engined_vector<cpu_engine, VectorLikeR> __HALA_REFREF r, int num_rhs = 1){
    return ilu_buffer_size(ilu, x.vector(), r.vector(), num_rhs);
}

#ifdef HALA_ENABLE_GPU
template<typename T, class VectorLikeX, class VectorLikeR>
auto ilu_temp_buffer(gpu_ilu<T> const &ilu, engined_vector<gpu_engine, VectorLikeX> const &x, engined_vector<gpu_engine, VectorLikeR> __HALA_REFREF r, int num_rhs = 1){
    engined_vector<gpu_engine, gpu_vector<T>> temp(x.engine());
    size_t bsize = ilu.buffer_size(x.vector(), r.vector(), num_rhs);
    force_size(bsize, temp.vector());
    return temp;
}
template<typename T, class VectorLikeX, class VectorLikeR>
size_t ilu_buffer_size(gpu_ilu<T> const &ilu, engined_vector<gpu_engine, VectorLikeX> const&x, engined_vector<gpu_engine, VectorLikeR> __HALA_REFREF r, int num_rhs = 1){
    return ilu_buffer_size(ilu, x.vector(), r.vector(), num_rhs);
}
#endif

}

#endif

#endif
