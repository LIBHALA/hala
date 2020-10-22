#ifndef __HALA_LAPACK_SOLVER_WRAPPERS_HPP
#define __HALA_LAPACK_SOLVER_WRAPPERS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

/*!
 * \internal
 * \ingroup HALALAPACK
 * \file hala_lapack_plu.hpp
 * \brief HALA wrapper for LAPACK methods.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 *
 * Contains the HALA-LAPACK wrapper templates for PLU and Cholesky factorization.
 * \endinternal
 */

#define HALA_ENABLE_LAPACK

#include "hala_lapack_external.hpp"

/*!
 * \ingroup HALALAPACK
 * \addtogroup HALALAPACKPLU LAPACK solver templates
 *
 * \par Solvers
 * Covers the general factorization \f$ A = PLU \f$ and the symmetric positive definite Cholesky \f$ A = L L^T \f$.
 * Both the factorization and the solve stages are covered here.
 */

namespace hala{

/*!
 * \internal
 * \ingroup HALALAPACK
 * \brief If \b info is not zero, throws a runtime error including the \b name and \b info in the description.
 *
 * \endinternal
 */
inline void lapack_check(int info, std::string const &name){
    if (info != 0)
        throw std::runtime_error(name + " returned error code: " + std::to_string(info));
}

/*!
 * \ingroup HALALAPACKPLU
 * \brief Returns the size of the workspace used by the potrf() method.
 *
 * Overloads:
 * \code
 *  hala::portf_size(uplo, N, A, lda);
 *  hala::portf_size(uplo, N, A, lda = -1);
 *  hala::portf_size(engine, uplo, N, A, lda);
 *  hala::portf_size(engine, uplo, N, A, lda = -1);
 * \endcode
 * Note that lda of -1 will be overwritten with N.
 */
template<class VectorLikeA>
size_t potrf_size(char, int, VectorLikeA const&, int = -1){ return 0; }

/*!
 * \ingroup HALALAPACKPLU
 * \brief Wrapper to LAPACK Cholesky decomposition of an s.p.d. matrix xpotrf()
 *
 * Computes \f$ A = U^T U \f$, or \f$ A = L L^T \f$, \b A is overwritten with \b U/L
 * - \b uplo must lower or upper case U or L, indicates whether to work with the upper (U) or lower (L) part of \b A
 * - \b A must be a \b symmetric \b positive \b definite \b matrix of size \b lda by \b N
 * - \b lda is at least \b N
 *
 * Overloads:
 * \code
 *  hala::portf(uplo, N, A, lda);
 *  hala::portf(uplo, N, A, lda = -1);
 *  hala::portf(engine, uplo, N, A, lda);
 *  hala::portf(engine, uplo, N, A, lda = -1);
 * \endcode
 * All overloads accept an optional last argument a workspace vector that will be resized to hala::portf_size().
 * In addition, lda of -1 will be overwritten with N.
 */
template<class VectorLikeA, class VectorLikeW>
inline void potrf(char uplo, int N, VectorLikeA &&A, int lda, VectorLikeW &&){
    valid::default_ld(N, lda);
    check_types(A);
    assert( valid::potrf(uplo, N, A, lda) );

    using scalar_type = get_scalar_type<VectorLikeA>;

    int info = 0;
    call_backend<scalar_type>(spotrf_, dpotrf_, cpotrf_, zpotrf_, &uplo, &N, convert(A), &lda, &info);

    lapack_check(info, "hala::potrf()");
}

/*!
 * \ingroup HALALAPACKPLU
 * \brief Wrapper to LAPACK solver using Cholesky decomposition of an s.p.d. matrix xpotrs()
 *
 * Solves the linear system of equations \f$ A X = B \f$, where B is overwritten with the solution.
 * - \b uplo, \b N, \b A, and \b lda must already come from a call to hala::potrf().
 * - \b B is an \b nrhs by \b ldb matrix
 * - \b ldb is at least \b N
 * - \b B is overwritten with the solution
 *
 * Overloads:
 * \code
 *  hala::potrs(uplo, N, nrhs, A, lda, B, ldb);
 *  hala::potrs(uplo, N, nrhs, A, B, lda = -1, ldb = -1);
 *  hala::potrs(engine, uplo, N, nrhs, A, lda, B, ldb);
 *  hala::potrs(engine, uplo, N, nrhs, A, B, lda = -1, ldb = -1);
 * \endcode
 * Note, lda or ldb of -1 will be overwritten with N.
 */
template<class VectorLikeA, class VectorLikeB>
inline void potrs(char uplo, int N, int nrhs, VectorLikeA const &A, int lda, VectorLikeB &&B, int ldb){
    check_types(A, B);
    assert( valid::potrs(uplo, N, nrhs, A, lda, B, ldb) );

    using scalar_type = get_scalar_type<VectorLikeA>;

    int info = 0;
    call_backend<scalar_type>(spotrs_, dpotrs_, cpotrs_, zpotrs_, &uplo, &N, &nrhs, convert(A), &lda, convert(B), &ldb, &info);

    lapack_check(info, "hala::potrs()");
}

/*!
 * \ingroup HALALAPACKPLU
 * \brief Returns the size of the workspace used by the getrf() method.
 *
 * Overloads:
 * \code
 *  hala::gertf_size(M, N, A, lda);
 *  hala::gertf_size(M, N, A, lda = -1);
 *  hala::gertf_size(engine, M, N, A, lda);
 *  hala::gertf_size(engine, M, N, A, lda = -1);
 * \endcode
 * Note that lda of -1 will be overwritten with M.
 */
template<class VectorLikeA>
size_t getrf_size(int, int, VectorLikeA const&, int = -1){ return 0; }

/*!
 * \ingroup HALALAPACKPLU
 * \brief Wrapper to LAPACK general PLU factorization xgetrf()
 *
 * Computes \f$ P A = L U \f$, \b A is overwritten with \b U/L and the permutation matrix \b P is stored in \b ipiv
 * - \b A must be \b M by \b N matrix with full column/row rank (depending on the smaller of M/N)
 * - \b A has size \b lda by \b N with \b lda begin minimum \b M
 * - \b ipiv is a vector of integers holding the permutations, will be resized to match min M, N
 *
 * Overloads:
 * \code
 *  hala::gertf(M, N, A, lda, ipiv);
 *  hala::gertf(M, N, A, ipiv, lda = -1);
 *  hala::gertf(engine, M, N, A, lda, ipiv);
 *  hala::gertf(engine, M, N, A, ipiv, lda = -1);
 * \endcode
 * All overloads accept an optional last argument a workspace vector that will be resized to hala::gertf_size().
 * In addition, lda of -1 will be overwritten with M.
 */
template<class VectorLikeA, class VectorLikeI, class VectorLikeW>
inline void getrf(int M, int N, VectorLikeA &&A, int lda, VectorLikeI &&ipiv, VectorLikeW &&){
    check_types(A);
    check_types_int(ipiv);
    assert( valid::getrf(M, N, A, lda) );
    check_set_size(assume_output, ipiv, std::min(M, N));

    using scalar_type = get_scalar_type<VectorLikeA>;

    int info = 0;
    call_backend<scalar_type>(sgetrf_, dgetrf_, cgetrf_, zgetrf_, &M, &N, convert(A), &lda, convert(ipiv), &info);

    lapack_check(info, "hala::getrf()");
}

/*!
 * \ingroup HALALAPACKPLU
 * \brief Wrapper to LAPACK solver using general PLU factorization, xgetrs()
 *
 * Solves the linear system of equations \f$ A X = B \f$, where B is overwritten with the solution.
 * - \b N, \b A, \b lda, and \b ipiv must already come from a call to hala::getrf() with M = N
 * - \b B is an \b nrhs by \b ldb matrix
 * - \b ldb is at least \b N
 * - \b B is overwritten with the solution
 *
 * Overloads:
 * \code
 *  hala::getrs(trans, N, nrhs, A, lda, ipiv, B, ldb);
 *  hala::getrs(trans, N, nrhs, A, ipiv, B, lda = -1, ldb = -1);
 *  hala::getrs(engine, trans, N, nrhs, A, ipiv, lda, B, ldb);
 *  hala::getrs(engine, trans, N, nrhs, A, ipiv, B, lda = -1, ldb = -1);
 * \endcode
 * Note, lda or ldb of -1 will be overwritten with N.
 */
template<class VectorLikeA, class VectorLikeI, class VectorLikeB>
inline void getrs(char trans, int N, int nrhs, VectorLikeA const &A, int lda, VectorLikeI const &ipiv, VectorLikeB &&B, int ldb){
    check_types(A, B);
    check_types_int(ipiv);
    assert( valid::getrs(trans, N, nrhs, A, lda, ipiv, B, ldb) );

    using scalar_type = get_scalar_type<VectorLikeA>;

    int info = 0;
    call_backend<scalar_type>(sgetrs_, dgetrs_, cgetrs_, zgetrs_, &trans, &N, &nrhs, convert(A), &lda, convert(ipiv), convert(B), &ldb, &info);

    lapack_check(info, "hala::getrs()");
}

#ifndef __HALA_DOXYGEN_SKIP
template<class VectorLikeA> size_t potrf_size(cpu_engine const&, char, int, VectorLikeA const&, int = -1){ return 0; }
template<class VectorLikeA, class VectorLikeW>
inline void potrf(cpu_engine const&, char uplo, int N, VectorLikeA &A, int lda, VectorLikeW &&W){
    potrf(uplo, N, std::forward<VectorLikeA>(A), lda, std::forward<VectorLikeW>(W));
}
template<class VectorLikeA>
inline void potrf(cpu_engine const&, char uplo, int N, VectorLikeA &A, int lda = -1){
    potrf(uplo, N, std::forward<VectorLikeA>(A), lda, std::vector<get_scalar_type<VectorLikeA>>());
}

template<class VectorLikeA, class VectorLikeB>
inline void potrs(char uplo, int N, int nrhs, VectorLikeA const &A, VectorLikeB &&B, int lda = -1, int ldb = -1){
    valid::default_ld(N, lda);
    valid::default_ld(N, ldb);
    potrs(uplo, N, nrhs, A, lda, std::forward<VectorLikeB>(B), ldb);
}
template<class VectorLikeA, class VectorLikeB>
inline void potrs(cpu_engine const&, char uplo, int N, int nrhs, VectorLikeA const &A, int lda, VectorLikeB &&B, int ldb){
    potrs(uplo, N, nrhs, A, lda, std::forward<VectorLikeB>(B), ldb);
}
template<class VectorLikeA, class VectorLikeB>
inline void potrs(cpu_engine const&, char uplo, int N, int nrhs, VectorLikeA const &A, VectorLikeB &&B, int lda = -1, int ldb = -1){
    potrs(uplo, N, nrhs, A, std::forward<VectorLikeB>(B), lda, ldb);
}

template<class VectorLikeA> size_t getrf_size(cpu_engine const&, int, int, VectorLikeA const&, int = -1){ return 0; }

template<class VectorLikeA, class VectorLikeI, class VectorLikeW>
inline void getrf(int M, int N, VectorLikeA &&A, VectorLikeI &&ipiv, int lda, VectorLikeW &&W){
    valid::default_ld(M, lda);
    getrf(M, N, std::forward<VectorLikeA>(A), lda, std::forward<VectorLikeI>(ipiv), std::forward<VectorLikeW>(W));
}
template<class VectorLikeA, class VectorLikeI>
inline void getrf(int M, int N, VectorLikeA &&A, VectorLikeI &&ipiv, int lda = -1){
    valid::default_ld(M, lda);
    getrf(M, N, std::forward<VectorLikeA>(A), lda, std::forward<VectorLikeI>(ipiv), std::vector<get_scalar_type<VectorLikeA>>());
}

template<class VectorLikeA, class VectorLikeI, class VectorLikeW>
inline void getrf(cpu_engine const&, int M, int N, VectorLikeA &&A, int lda, VectorLikeI &&ipiv, VectorLikeW &&W){
    getrf(M, N, std::forward<VectorLikeA>(A), lda, std::forward<VectorLikeI>(ipiv), std::forward<VectorLikeW>(W));
}
template<class VectorLikeA, class VectorLikeI>
inline void getrf(cpu_engine const&, int M, int N, VectorLikeA &&A, VectorLikeI &&ipiv, int lda = -1){
    getrf(M, N, std::forward<VectorLikeA>(A), std::forward<VectorLikeI>(ipiv), lda);
}

template<class VectorLikeA, class VectorLikeI, class VectorLikeB>
inline void getrs(char uplo, int N, int nrhs, VectorLikeA const &A, VectorLikeI const &ipiv, VectorLikeB &&B, int lda = -1, int ldb = -1){
    valid::default_ld(N, lda);
    valid::default_ld(N, ldb);
    getrs(uplo, N, nrhs, A, lda, ipiv, std::forward<VectorLikeB>(B), ldb);
}
template<class VectorLikeA, class VectorLikeI, class VectorLikeB>
inline void getrs(cpu_engine const&, char uplo, int N, int nrhs, VectorLikeA const &A, int lda, VectorLikeI const &ipiv,
                  VectorLikeB &&B, int ldb){
    getrs(uplo, N, nrhs, A, lda, ipiv, std::forward<VectorLikeB>(B), ldb);
}
template<class VectorLikeA, class VectorLikeI, class VectorLikeB>
inline void getrs(cpu_engine const&, char uplo, int N, int nrhs, VectorLikeA const &A, VectorLikeI const &ipiv,
                  VectorLikeB &&B, int lda = -1, int ldb = -1){
    getrs(uplo, N, nrhs, A, ipiv, std::forward<VectorLikeB>(B), lda, ldb);
}

#endif

}

#endif
