#ifndef __HALA_LAPACK_WRAPPERS_HPP
#define __HALA_LAPACK_WRAPPERS_HPP
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
 * \ingroup HALALAPACK
 * \file hala_lapack.hpp
 * \brief HALA wrapper for LAPACK methods.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 *
 * Contains the HALA-LAPACK wrapper templates.
 */

#define HALA_ENABLE_LAPACK

#include "hala_lapack_plu.hpp"

namespace hala{

/*!
 * \defgroup HALALAPACK HALA LAPACK Wrapper Templates
 *
 * \par HALA LAPACK Wrapper Templates
 * A set of templates that wrap around LAPACK functions providing generality, overloads and error
 * checking similar to the \ref HALABLAS. Currently, only a small subset of LAPACK methods
 * is included in HALA and there is no support for engine overloads or GPUs.
 *
 */

/*!
 * \ingroup HALALAPACK
 * \brief Wrapper to LAPACK linear least-squares solver xGELS().
 *
 * \par Type 'N', with M >= N
 * Solves the over-determined system of equations \f$ \min_{X} \| B - A X \|_2 \f$
 * - \b A is column-major of size M (rows) by N (columns), or lda by N
 * - \b A is overwritten by the matrix QR factorization (see LAPACK dgeqrf function)
 * - \b B is column-major of size M (rows) by num_rhs (columns), or ldb by num_rhs
 * - \b B is overwritten with the solution and residual norm for each column
 *     - rows 1:N hold the solution to the minimization problem
 *     - the 2-norm of rows N+1:M is the residual norm for the corresponding column
 *
 * \par Type 'T', with M < N
 * Solves the over-determined system of equations \f$ \min_{X} \| B - A^T X \|_2 \f$
 * - \b A is column-major of size M (rows) by N (columns), or lda by N
 * - \b A is overwritten by the matrix LQ factorization (see LAPACK dgelqf function)
 * - \b B is column-major of size N (rows) by num_rhs (columns), or ldb by num_rhs
 * - \b B is overwritten with the solution and residual norm for each column
 *     - rows 1:M hold the solution to the minimization problem
 *     - the 2-norm of rows M+1:N is the residual norm for the corresponding column
 *
 * \par Type 'N', with M < N
 * Find the minimum norm solution to under-determined system \f$ A X = B \f$
 * - \b A is column-major of size M (rows) by N (columns), or lda by N
 * - \b A is overwritten by the matrix LQ factorization (see LAPACK dgelqf function)
 * - \b B is column-major of size N (rows) by num_rhs (columns), or ldb by num_rhs
 * - \b B is overwritten with the minimum norm solution (solution has size N)
 *
 * \par Type 'T', with M >= N
 * Find the minimum norm solution to under-determined system \f$ A^T X = B \f$
 * - \b A is column-major of size M (rows) by N (columns), or lda by N
 * - \b A is overwritten by the matrix QR factorization (see LAPACK dgeqrf function)
 * - \b B is column-major of size M (rows) by num_rhs (columns), or ldb by num_rhs
 * - \b B is overwritten with the minimum norm solution (solution has size M)
 */
template <class VectorLikeA, class VectorLikeB>
inline void gels(char type, int M, int N, int num_rhs, VectorLikeA &A, VectorLikeB &B, int lda = -1, int ldb = -1){
    check_types(A, B);
    assert( check_trans(type) );
    if (lda == -1) lda = M;
    if (ldb == -1) ldb = std::max(M, N);
    assert(M > 0);
    assert(N > 0);
    assert(num_rhs > 0);
    assert( check_size(A, lda, N) );
    assert( check_size(B, ldb, num_rhs) );

    int worksize = -1;
    int info = 0;

    using scalar_type = get_scalar_type<VectorLikeA>;
    std::vector<scalar_type> workspace(2, 0.0);

    call_backend<scalar_type>(sgels_, dgels_, cgels_, zgels_,
                 &type, &M, &N, &num_rhs, convert(A), &lda, convert(B), &ldb, pconvert(workspace.data()), &worksize, &info);
    runtime_assert((info == 0), "hala::solvel2() called lapack for work-space and got error code: " + std::to_string(info));

    worksize = static_cast<int>(creal(workspace[0]));
    workspace.resize(worksize);

    call_backend<scalar_type>(sgels_, dgels_, cgels_, zgels_,
                 &type, &M, &N, &num_rhs, convert(A), &lda, convert(B), &ldb, pconvert(workspace.data()), &worksize, &info);
    runtime_assert((info == 0), "hala::solvel2() lapack xGELS() returned error code "
                           + std::to_string(info) + ((info < 0) ? " (invalid input)" : " (matrix has insufficient rank)"));
}

/*!
 * \ingroup HALALAPACK
 * \brief Wrapper to LAPACK singular value decomposition xGESVD(), compute the singular values only.
 *
 * Computes the singular values of the general \b M by \b N matrix \b A.
 * - \b S is a vector which is resized to min(\b M, \b N) and holds the singular vectors
 * - \b Note: the content of \b A is destoryed in the factorization process
 */
template<class VectorLikeA, class VectorLikeS>
void svd(int M, int N, VectorLikeA &A, VectorLikeS &S, int lda = -1){
    check_types(A);
    using scalar_type   = get_scalar_type<VectorLikeA>;
    using standard_type = get_standard_type<VectorLikeA>;
    using precision_type = get_precision_type<VectorLikeA>;

    static_assert(std::is_same<typename define_type<VectorLikeS>::value_type, precision_type>::value,
                  "svd() called with mismatching precision of A and S, correct options, e.g., float-float, complex<double>-double");
    assert(std::min(M, N) >= 1);
    if (lda == -1) lda = M;
    assert(lda >= M);
    assert( check_size(A, lda, N) );

    size_t minmn = (size_t) std::min(M, N);
    check_set_size(assume_output, S, minmn);

    char jobn = 'N';
    int ldu = 1, ldvt = 1;
    std::vector<standard_type> dummyu = {get_cast<standard_type>(0.0)};
    auto dummyvt = dummyu; // cannot pass nullptr even if LAPACK ignores the vectors

    int worksize = -1;
    std::vector<standard_type> workspace = dummyu; // will resize later
    std::vector<precision_type> rwork(5 * minmn);

    int info = 0;
    for(int i=0; i<2; i++){
        if (is_float<scalar_type>::value){
            sgesvd_(&jobn, &jobn, &M, &N, convert(A), &lda, convert(S), convert(dummyu),
                    &ldu, convert(dummyvt), &ldvt, convert(workspace), &worksize, &info);
        }else if (is_double<scalar_type>::value){
            dgesvd_(&jobn, &jobn, &M, &N, convert(A), &lda, convert(S), convert(dummyu),
                    &ldu, convert(dummyvt), &ldvt, convert(workspace), &worksize, &info);
        }else if (is_fcomplex<scalar_type>::value){
            cgesvd_(&jobn, &jobn, &M, &N, convert(A), &lda, convert(S), convert(dummyu),
                    &ldu, convert(dummyvt), &ldvt, convert(workspace), &worksize, convert(rwork), &info);
        }else if (is_dcomplex<scalar_type>::value){
            zgesvd_(&jobn, &jobn, &M, &N, convert(A), &lda, convert(S), convert(dummyu),
                    &ldu, convert(dummyvt), &ldvt, convert(workspace), &worksize, convert(rwork), &info);
        }

        if (i == 0)
            runtime_assert((info == 0), "hala::svd() called lapack for work-space and got error code: " + std::to_string(info));
        else
            runtime_assert((info == 0), "hala::svd() called lapack for factorize and got error code: " + std::to_string(info));

        worksize = static_cast<int>(std::real(workspace[0]));
        workspace.resize(worksize);
    }
}

/*!
 * \ingroup HALALAPACK
 * \brief Wrapper to LAPACK singular value decomposition xGESVD(), compute the singular values only.
 *
 * Computes the singular values AND singular vectors of the general \b M by \b N matrix \b A.
 * - \b S is a vector which is resized to min(\b M, \b N) and holds the singular vectors
 * - \b U is a matrix of size \b ldu by min(\b M, \b N), holds the left singular vectors
 * - \b VH is a matrix of size \b ldvh by \b N, holds the complex transpose of the right singular vectors
 * - \b ldu is at least \b M, \b ldvh is at least min(\b M, \b N)
 * - if any of \b lda, \b ldu or \b ldvh are set to default -1, then the minimum allowed values are assumed
 * - \b Note: the content of \b A is destoryed in the factorization process
 */
template<class VectorLikeA, class VectorLikeS, class VectorLikeU, class VectorLikeUH>
void svd(int M, int N, VectorLikeA &A, VectorLikeS &S, VectorLikeU &U, VectorLikeUH &VH, int lda = -1, int ldu = -1, int ldvh = -1){
    check_types(A, U, VH);
    using scalar_type   = get_scalar_type<VectorLikeA>;
    using standard_type = get_standard_type<VectorLikeA>;
    using precision_type = get_precision_type<VectorLikeA>;

    static_assert(std::is_same<typename define_type<VectorLikeS>::value_type, precision_type>::value,
                  "svd() called with mismatching precision of A and S, correct options, e.g., float-float, complex<double>-double");

    assert(std::min(M, N) >= 1);
    size_t minmn = (size_t) std::min(M, N);

    if (lda == -1) lda = M;
    assert(lda >= M);
    assert( check_size(A, lda, N) );

    if (ldu == -1) ldu = M;
    assert(ldu >= M);
    check_set_size(assume_output, U, ldu, minmn);

    if (ldvh == -1) ldvh = (int) minmn;
    assert(ldvh >= (int) minmn);
    check_set_size(assume_output, VH, ldvh, N);

    check_set_size(assume_output, S, minmn);

    char jobs = 'S';

    int worksize = -1;
    std::vector<standard_type> workspace = {get_cast<standard_type>(0.0)}; // will resize later
    std::vector<precision_type> rwork(5 * minmn);

    int info = 0;
    for(int i=0; i<2; i++){
        if (is_float<scalar_type>::value){
            sgesvd_(&jobs, &jobs, &M, &N, convert(A), &lda, convert(S), convert(U), &ldu, convert(VH), &ldvh, convert(workspace), &worksize, &info);
        }else if (is_double<scalar_type>::value){
            dgesvd_(&jobs, &jobs, &M, &N, convert(A), &lda, convert(S), convert(U), &ldu, convert(VH), &ldvh, convert(workspace), &worksize, &info);
        }else if (is_fcomplex<scalar_type>::value){
            cgesvd_(&jobs, &jobs, &M, &N, convert(A), &lda, convert(S), convert(U), &ldu, convert(VH), &ldvh, convert(workspace), &worksize, convert(rwork), &info);
        }else if (is_dcomplex<scalar_type>::value){
            zgesvd_(&jobs, &jobs, &M, &N, convert(A), &lda, convert(S), convert(U), &ldu, convert(VH), &ldvh, convert(workspace), &worksize, convert(rwork), &info);
        }

        if (i == 0)
            runtime_assert((info == 0), "hala::svd() called lapack for work-space and got error code: " + std::to_string(info));
        else
            runtime_assert((info == 0), "hala::svd() called lapack for factorize and got error code: " + std::to_string(info));

        worksize = static_cast<int>(std::real(workspace[0]));
        workspace.resize(worksize);
    }
}

/*!
 * \ingroup HALALAPACK
 * \brief Wrapper to LAPACK general solver using PLU factorization xGESV().
 *
 * Solves the linear system of equations \f$ A X = B \f$, where B is overwritten with the solution
 * - \b A is a column major matrix of size N by N (or lda by N)
 * - \b A is overwritten with the LU part of the PLU factorization
 * - \b B is a column major matrix of size N by num_rhs (or ldb by num_rhs)
 * - \b B is overwritten with the solution
 * - \b ipiv must be a vector of integers, will be resized to \b N (if smaller) and overwritten with \b P
 */
template<class VectorLikeA, class VectorLikeP, class VectorLikeB>
inline void gesv(int N, int num_rhs, VectorLikeA &A, VectorLikeP &ipiv, VectorLikeB &B, int lda = -1, int ldb = -1){
    check_types(A, B);
    check_types_int(ipiv);

    valid::default_ld(N, lda);
    valid::default_ld(N, ldb);
    assert(lda >= N);
    assert(ldb >= N);
    assert( check_size(A, lda, N) );
    assert( check_size(B, ldb, num_rhs) );
    check_set_size(assume_output, ipiv, N);

    using scalar_type = get_scalar_type<VectorLikeA>;

    int info = 0;
    call_backend<scalar_type>(sgesv_, dgesv_, cgesv_, zgesv_,
                 &N, &num_rhs, convert(A), &lda, convert(ipiv), convert(B), &ldb, &info);

    runtime_assert((info == 0), "hala::solve_plu() lapack xGESV() returned error code "
                   + std::to_string(info) + ((info < 0) ? " (invalid input)" : " (matrix is singular)"));
}

/*!
 * \ingroup HALALAPACK
 * \brief Returns the condition number of a \b N by \b N non-singular matrix, requires an internal copy of \b A.
 */
template<class VectorLikeA>
inline auto condition_number(int N, VectorLikeA const &A, int lda = -1){
    check_types(A);
    valid::default_ld(N, lda);
    assert(lda >= N);
    assert( check_size(A, lda, N) );

    using scalar_type = get_scalar_type<VectorLikeA>;
    using precision_type = get_precision_type<VectorLikeA>;

    std::vector<scalar_type> Acopy;
    hala::vcopy(A, Acopy);

    std::vector<precision_type> S;
    hala::svd(N, N, Acopy, S, lda);

    return S[0] / S.back();
}

/*!
 * \internal
 * \ingroup HALALAPACK
 * \brief Query the required size of the Q factor and workspace used by xgeqr().
 *
 * The inputs are the same as hala::geqr() with the query type being either -1 or -2
 * corresponding to an optimal speed or minimal memory usage according to LAPACK.
 * Value -1 checks for the optimal performance, value -2 checks for minimum size.
 * \endinternal
 */
template<class VectorLikeA>
std::pair<int, int> geqr_query_size(int query_type, int M, int N, VectorLikeA &&A, int lda){
    check_types(A);
    assert( std::min(M, N) >= 1 );
    assert(lda >= M);
    assert( check_size(A, lda, N) );

    using scalar_type = get_scalar_type<VectorLikeA>;

    std::vector<scalar_type> T(5);
    std::vector<scalar_type> work(1);
    int info = 0;

    int tsize = query_type;
    int wsize = query_type;

    call_backend<scalar_type>(sgeqr_, dgeqr_, cgeqr_, zgeqr_,
                 &M, &N, convert(A), &lda, convert(T), &tsize, convert(work), &wsize, &info);

    runtime_assert((info == 0), "hala::geqr() (query sizes) lapack xGEQR() returned error code "
                   + std::to_string(info) + ((info < 0) ? " (invalid input)" : " (unknown)"));

    return std::make_pair(static_cast<int>(std::real(T[0])), static_cast<int>(std::real(work[0])));
}

/*!
 * \ingroup HALALAPACK
 * \brief Query the required size of the Q factor and workspace used by xgeqr().
 *
 * Returns the size required for an optimal performance QR decomposition LAPACK xgeqr.
 * The first entry in the pair represents the size of the factor, the second entry
 * is the size of the temporary work-space.
 */
template<class VectorLikeA>
std::pair<int, int> geqr_opt_size(int M, int N, VectorLikeA &&A, int lda){
    return geqr_query_size(-1, M, N, std::forward<VectorLikeA>(A), lda);
}

/*!
 * \ingroup HALALAPACK
 * \brief Query the required size of the Q factor and workspace used by xgeqr().
 *
 * Returns the size required for an optimal performance QR decomposition LAPACK xgeqr.
 * The first entry in the pair represents the size of the factor, the second entry
 * is the size of the temporary work-space.
 */
template<class VectorLikeA>
std::pair<int, int> geqr_min_size(int M, int N, VectorLikeA &&A, int lda){
    return geqr_query_size(-2, M, N, std::forward<VectorLikeA>(A), lda);
}

/*!
 * \ingroup HALALAPACK
 * \brief Wrapper around Lapack xgeqr() for QR factorization.
 *
 * Performs the factorization \f$ A = Q R \f$ where \b Q has orthonormal columns
 * and \b R is either upper triangular (M = N), upper triangular with padded zeros (M > N)
 * or trapezoidal (M < N). The matrix \b A will be overwritten with \b R and
 * the matrix \b Q will be stored in the zero section of \b R (below the diagonal)
 * and the vector \b T. Follow on multiplications of \b Q can be done with hala::gemqr().
 *
 * - \b M is positive integer indicating the number of rows of \b A
 * - \b N is positive integer indicating the number of columns of \b A
 * - \b lda is the leading dimension of \b A, must be minimum of \b M
 * - \b A must have size at least \b lda by \b N
 * - \b T holds a portion of the matrix \b Q, will be resized if the size is insufficient
 *
 * The size of the \b T will be determined by a call to:
 * \code
 *  auto sizes = hala::geqr_opt_size(M, N, A, lda);
 *  // T must have size sizes.first
 *  // additional workspace will be used with size sizes.second
 * \endcode
 *
 * Overloads:
 * \code
 *  hala::geqr(M, N, A, lda, T);
 *  hala::geqr(M, N, A, T, lda = -1);
 *  hala::geqr_min(M, N, A, lda, T);
 *  hala::geqr_min(M, N, A, T, lda = -1);
 * \endcode
 * Negative leading dimensions are overwritten with the minimum acceptable size.
 * The _min variants use minimum memory variants and hala::geqr_min_size().
 */
template<bool use_opt = true, class VectorLikeA, class VectorLikeT>
inline void geqr(int M, int N, VectorLikeA &&A, int lda, VectorLikeT &&T){
    check_types(A, T);

    using scalar_type = get_scalar_type<VectorLikeA>;

    std::pair<int, int> sizes = (use_opt) ? geqr_opt_size(M, N, std::forward<VectorLikeA>(A), lda)
                                          : geqr_min_size(M, N, std::forward<VectorLikeA>(A), lda);
    int tsize = sizes.first;
    int wsize = sizes.second;

    check_set_size(assume_output, T, sizes.first);
    std::vector<scalar_type> work(static_cast<size_t>(sizes.second));
    int info = 0;

    call_backend<scalar_type>(sgeqr_, dgeqr_, cgeqr_, zgeqr_,
                 &M, &N, convert(A), &lda, convert(T), &tsize, convert(work), &wsize, &info);

    runtime_assert((info == 0), "hala::geqr() lapack xGEQR() returned error code "
                   + std::to_string(info) + ((info < 0) ? " (invalid input)" : " (unknown)"));
}

/*!
 * \ingroup HALALAPACK
 * \brief Wrapper around Lapack xgemqr() for multiplication by a QR factor.
 *
 * Performs multiplication \f$ C = op(Q) C \f$ or \f$ C = C op(Q) \f$
 * where \b Q is a matrix computed and stored in the format used by hala::gerq().
 * Specifically, the matrix is defined by the two vectors \b A and \b T
 * that are used in the call to hala::geqr().
 *
 * - \b side is lower/upper case L or R indicating whether \b Q is on the left (L) or right (R) side
 * - \b trans is lower/upper case N (non-transpose) or T (conjugate transpose), indicating \b op
 * - \b C is an \b M by \b N matrix (or \b ldc by \b N)
 * - \b A and \b T are the vectors that come from a call to hala::geqr() with sizes given by geqr_M and geqr_N
 * - \b K is the number of factors used to define \b Q, namely the smaller of geqr_M and geqr_N
 * - \b lda is as in the call to hala::geqr()
 * - \b ldc is the leading dimension of \b C, minimum \b M
 * - \b tsize is best inferred from the size of \b T, if not, it is taken from the second output of hala::geqr_opt_size()
 *
 * Overloads:
 * \code
 *  hala::gemqr(side, trans, M, N, K, A, lda, T, tsize, C, ldc);
 *  hala::gemqr(side, trans, M, N, K, A, lda, T, C, ldc);
 *  hala::gemqr(side, trans, M, N, K, A, T, C, lda = -1, ldc = -1);
 * \endcode
 * Negative leading dimensions are overwritten with the minimum acceptable size.
 */
template<class VectorLikeA, class VectorLikeT, class VectorLikeC>
inline void gemqr(char side, char trans, int M, int N, int K,
                  VectorLikeA const &A, int lda, VectorLikeT const &T, int tsize, VectorLikeC &&C, int ldc){
    check_types(A, T, C);
    assert( check_side(side) );
    assert( check_trans(trans) );
    assert( K <= (is_l(side) ? M : N) );
    assert( lda >= (is_l(side) ? M : N) );
    assert( ldc >= M );
    assert( check_size(C, ldc, N) );

    using scalar_type = get_scalar_type<VectorLikeA>;

    int lwork = -1;
    std::vector<scalar_type> work(1);

    int info = 0;
    call_backend<scalar_type>(sgemqr_, dgemqr_, cgemqr_, zgemqr_,
                 &side, &trans, &M, &N, &K, convert(A), &lda, convert(T), &tsize, convert(C), &ldc, convert(work), &lwork, &info);

    runtime_assert((info == 0), "hala::geqr() lapack xGEMQR() returned error code "
                   + std::to_string(info) + ((info < 0) ? " (invalid input)" : " (unknown)"));

    lwork = static_cast<int>(std::real(work[0]));
    work.resize(lwork);

    call_backend<scalar_type>(sgemqr_, dgemqr_, cgemqr_, zgemqr_,
                 &side, &trans, &M, &N, &K, convert(A), &lda, convert(T), &tsize, convert(C), &ldc, convert(work), &lwork, &info);

    runtime_assert((info == 0), "hala::geqr() lapack xGEMQR() returned error code "
                   + std::to_string(info) + ((info < 0) ? " (invalid input)" : " (unknown)"));
}

#ifndef __HALA_DOXYGEN_SKIP
template<bool use_opt = true, class VectorLikeA, class VectorLikeT>
inline void geqr(int M, int N, VectorLikeA &&A, VectorLikeT &&T, int lda = -1){
    valid::default_ld(M, lda);
    geqr<use_opt>(M, N, A, lda, T);
}
template<class VectorLikeA, class VectorLikeT>
inline void geqr_min(int M, int N, VectorLikeA &&A, int lda, VectorLikeT &&T){
    geqr<false>(M, N, A, lda, T);
}
template<class VectorLikeA, class VectorLikeT>
inline void geqr_min(int M, int N, VectorLikeA &&A, VectorLikeT &&T, int lda = -1){
    geqr<false>(M, N, A, T, lda);
}

template<class VectorLikeA, class VectorLikeT, class VectorLikeC>
inline void gemqr(char side, char trans, int M, int N, int K,
                  VectorLikeA const &A, int lda, VectorLikeT const &T, VectorLikeC &&C, int ldc){
    int tsize = valid::infer_size(T);
    gemqr(side, trans, M, N, K, A, lda, T, tsize, C, ldc);
}

template<class VectorLikeA, class VectorLikeT, class VectorLikeC>
inline void gemqr(char side, char trans, int M, int N, int K,
                  VectorLikeA const &A, VectorLikeT const &T, VectorLikeC &&C, int lda = -1, int ldc = -1){
    valid::default_ld(M, ldc);
    valid::default_ld(is_l(side), M, N, lda);
    gemqr(side, trans, M, N, K, A, lda, T, C, ldc);
}
#endif

}

#endif
