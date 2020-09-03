#ifndef __HALA_LAPACK_EXTERNAL_HPP
#define __HALA_LAPACK_EXTERNAL_HPP
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
 * \file hala_lapack_external.hpp
 * \brief HALA wrapper for LAPACK methods.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 *
 * Contains the LAPACK header definitions.
 * \endinternal
 */

#include "hala_blas.hpp"

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section
#ifndef HALA_SKIP_DEFINES

extern "C"{

#if !defined(gesv_) // check if LAPACK header is already present

// general least-squares
void sgels_(const char *TRANS, const int *M, const int *N, const int *NRHS, float *A, const int *lda,
            float *B, const int *ldb, float *WORK, int *LWORK, int *info);
void dgels_(const char *TRANS, const int *M, const int *N, const int *NRHS, double *A, const int *lda,
            double *B, const int *ldb, double *WORK, int *LWORK, int *info);
void cgels_(const char *TRANS, const int *M, const int *N, const int *NRHS, std::complex<float> *A, const int *lda,
            std::complex<float> *B, const int *ldb, std::complex<float> *WORK, int *LWORK, int *info);
void zgels_(const char *TRANS, const int *M, const int *N, const int *NRHS, std::complex<double> *A, const int *lda,
            void *B, const int *ldb, std::complex<double> *WORK, int *LWORK, int *info);

// general SVD
void sgesvd_(char const *JOBU, char const *JOBV, int const *M, int const *N, float *A, int const *lda, float *S,
             float *U, int const *ldu, float *VT, int const *ldvt, float *workspace, const int *worksize, int *info);
void dgesvd_(char const *JOBU, char const *JOBV, int const *M, int const *N, double *A, int const *lda, double *S,
             double *U, int const *ldu, double *VT, int const *ldvt, double *workspace, const int *worksize, int *info);
void cgesvd_(char const *JOBU, char const *JOBV, int const *M, int const *N, std::complex<float> *A, int const *lda, float *S,
             std::complex<float> *U, int const *ldu, std::complex<float> *VT, int const *ldvt, std::complex<float> *workspace, const int *worksize, float *rwork, int *info);
void zgesvd_(char const *JOBU, char const *JOBV, int const *M, int const *N, std::complex<double> *A, int const *lda, std::complex<double> *S,
             std::complex<double> *U, int const *ldu, std::complex<double> *VT, int const *ldvt, std::complex<double> *workspace, const int *worksize, double *rwork, int *info);

// Cholesky SPD factorize
void spotrf_(const char *uplo, const int *N, float *A, const int *lda, int *info);
void dpotrf_(const char *uplo, const int *N, double *A, const int *lda, int *info);
void cpotrf_(const char *uplo, const int *N, std::complex<float> *A, const int *lda, int *info);
void zpotrf_(const char *uplo, const int *N, std::complex<double> *A, const int *lda, int *info);

void spotrs_(const char *uplo, const int *N, const int *nrhs, const float *A, const int *lda, float *B, const int *ldb, int *info);
void dpotrs_(const char *uplo, const int *N, const int *nrhs, const double *A, const int *lda, double *B, const int *ldb, int *info);
void cpotrs_(const char *uplo, const int *N, const int *nrhs, const std::complex<float> *A, const int *lda,
             std::complex<float> *B, const int *ldb, int *info);
void zpotrs_(const char *uplo, const int *N, const int *nrhs, const std::complex<double> *A, const int *lda,
             std::complex<double> *B, const int *ldb, int *info);

void sgetrf_(const int *M, const int *N, float *A, const int *lda, int *ipiv, int *info);
void dgetrf_(const int *M, const int *N, double *A, const int *lda, int *ipiv, int *info);
void cgetrf_(const int *M, const int *N, std::complex<float> *A, const int *lda, int *ipiv, int *info);
void zgetrf_(const int *M, const int *N, std::complex<double> *A, const int *lda, int *ipiv, int *info);

void sgetrs_(const char *trans, const int *N, const int *nrhs, const float *A, const int *lda, int const *ipiv,
             float *B, const int *ldb, int *info);
void dgetrs_(const char *trans, const int *N, const int *nrhs, const double *A, const int *lda, int const *ipiv,
             double *B, const int *ldb, int *info);
void cgetrs_(const char *trans, const int *N, const int *nrhs, const std::complex<float> *A, const int *lda, int const *ipiv,
             std::complex<float> *B, const int *ldb, int *info);
void zgetrs_(const char *trans, const int *N, const int *nrhs, const std::complex<double> *A, const int *lda, int const *ipiv,
             std::complex<double> *B, const int *ldb, int *info);

void dgeev_(const char *JOBVL, const char *JOBVR, const int *N, const double *A, const int *lda,
            double *WR, double *WI, double *VL, const int *ldvl, double *VR, const int *ldvr, double *WORK, int *LWORK, int *info);

// general solve using PLU
void sgesv_(const int *N, const int *NRHS, float *A, const int *lda, int *IPIV, float *B, const int *ldb, int *info);
void dgesv_(const int *N, const int *NRHS, double *A, const int *lda, int *IPIV, double *B, const int *ldb, int *info);
void cgesv_(const int *N, const int *NRHS, std::complex<float> *A, const int *lda, int *IPIV, std::complex<float> *B, const int *ldb, int *info);
void zgesv_(const int *N, const int *NRHS, std::complex<double> *A, const int *lda, int *IPIV, std::complex<double> *B, const int *ldb, int *info);

// general QR
void sgeqr_(const int *M, const int *N, float *A, const int *lda, float *T, const int *tsize, float *work, int const *lwork, int *info);
void dgeqr_(const int *M, const int *N, double *A, const int *lda, double *T, const int *tsize, double *work, int const *lwork, int *info);
void cgeqr_(const int *M, const int *N, std::complex<float> *A, const int *lda, std::complex<float> *T, const int *tsize, std::complex<float> *work, int const *lwork, int *info);
void zgeqr_(const int *M, const int *N, std::complex<double> *A, const int *lda, std::complex<double> *T, const int *tsize, std::complex<double> *work, int const *lwork, int *info);

void sgemqr_(const char* side, const char* trans, const int *M, const int *N, const int *K, float const *A, int const *lda, float const *T, int const *tsize, float *C, const int *ldac, float *work, const int *lwork, int *info);
void dgemqr_(const char* side, const char* trans, const int *M, const int *N, const int *K, double const *A, int const *lda, double const *T, int const *tsize, double *C, const int *ldac, double *work, const int *lwork, int *info);
void cgemqr_(const char* side, const char* trans, const int *M, const int *N, const int *K, std::complex<float> const *A, int const *lda, std::complex<float> const *T, int const *tsize, std::complex<float> *C, const int *ldac, std::complex<float> *work, const int *lwork, int *info);
void zgemqr_(const char* side, const char* trans, const int *M, const int *N, const int *K, std::complex<double> const *A, int const *lda, std::complex<double> const *T, int const *tsize, std::complex<double> *C, const int *ldac, std::complex<double> *work, const int *lwork, int *info);

#endif

}
#endif
#endif

#endif
