#ifndef __HALA_BLAS_EXTERNAL_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_common.hpp"

/*!
 * \internal
 * \file hala_blas_external.hpp
 * \brief HALA BALAS and LAPACK function definitions.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALABLAS
 *
 * Contains the header definitions of the BLAS and LAPACK methods (using extern C).
 * \endinternal
*/

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section
#ifndef HALA_SKIP_DEFINES

extern "C"{

// check if we appear to be using an external BLAS header already
#if !defined(sgemm_)

// BLAS Level 1
void scopy_(const int *N, const float *x, const int *incx, float *y, const int *incy);
void dcopy_(const int *N, const double *x, const int *incx, double *y, const int *incy);
void ccopy_(const int *N, const std::complex<float> *x, const int *incx, std::complex<float> *y, const int *incy);
void zcopy_(const int *N, const std::complex<double> *x, const int *incx, std::complex<double> *y, const int *incy);

float snrm2_(const int *N, const float *x, const int *incx);
double dnrm2_(const int *N, const double *x, const int *incx);
float scnrm2_(const int *N, const std::complex<float> *x, const int *incx);
double dznrm2_(const int *N, const std::complex<double> *x, const int *incx);

float sasum_(const int *N, const float *x, const int *incx);
double dasum_(const int *N, const double *x, const int *incx);
float scasum_(const int *N, const std::complex<float> *x, const int *incx);
double dzasum_(const int *N, const std::complex<double> *x, const int *incx);

void srotg_(float *SA, float *SB, float *C, float *S);
void drotg_(double *SA, double *SB, double *C, double *S);
void crotg_(std::complex<float> *SA, std::complex<float> *SB, std::complex<float> *C, std::complex<float> *S);
void zrotg_(std::complex<double> *SA, std::complex<double> *SB, std::complex<double> *C, std::complex<double> *S);

void srot_(const int *N, float *x, const int *incx, float *y, const int *incy, float *C, float *S);
void drot_(const int *N, double *x, const int *incx, double *y, const int *incy, double *C, double *S);
void csrot_(const int *N, std::complex<float> *x, const int *incx, std::complex<float> *y, const int *incy, float const *C, float const *S);
void zdrot_(const int *N, std::complex<double> *x, const int *incx, std::complex<double> *y, const int *incy, double const *C, double const *S);
void crot_(const int *N, std::complex<float> *x, const int *incx, std::complex<float> *y, const int *incy, float const *C, std::complex<float> const *S);
void zrot_(const int *N, std::complex<double> *x, const int *incx, std::complex<double> *y, const int *incy, double const *C, std::complex<double> const *S);

void srotmg_(float *D1, float *D2, float *X, const float *Y, float *param);
void drotmg_(double *D1, double *D2, double *X, const double *Y, double *param);

void srotm_(const int *N, float *x, const int *incx, float *y, const int *incy, const float *param);
void drotm_(const int *N, double *x, const int *incx, double *y, const int *incy, const double *param);

int isamax_(const int *N, const float *x, const int *incx);
int idamax_(const int *N, const double *x, const int *incx);
int icamax_(const int *N, const std::complex<float> *x, const int *incx);
int izamax_(const int *N, const std::complex<double> *x, const int *incx);

void saxpy_(const int *N, const float *alpha, const float *x, const int *incx, const float *y, const int *incy);
void daxpy_(const int *N, const double *alpha, const double *x, const int *incx, const double *y, const int *incy);
void caxpy_(const int *N, const std::complex<float> *alpha, const std::complex<float> *x, const int *incx, const std::complex<float> *y, const int *incy);
void zaxpy_(const int *N, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx, const std::complex<double> *y, const int *incy);

float sdot_(const int *N, const float *x, const int *incx, const float *y, const int *incy);
double ddot_(const int *N, const double *x, const int *incx, const double *y, const int *incy);

void sscal_(const int *N, const float *alpha, const float *x, const int *incx);
void dscal_(const int *N, const double *alpha, const double *x, const int *incx);
void cscal_(const int *N, const std::complex<float> *alpha, const std::complex<float> *x, const int *incx);
void csscal_(const int *N, const float *alpha, const std::complex<float> *x, const int *incx);
void zscal_(const int *N, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx);
void zdscal_(const int *N, const double *alpha, const std::complex<double> *x, const int *incx);

void sswap_(const int *N, const float *x, const int *incx, const float *y, const int *incy);
void dswap_(const int *N, const double *x, const int *incx, const double *y, const int *incy);
void cswap_(const int *N, const std::complex<float> *x, const int *incx, const std::complex<float> *y, const int *incy);
void zswap_(const int *N, const std::complex<double> *x, const int *incx, const std::complex<double> *y, const int *incy);

// BLAS Level 2
void sgemv_(const char *transa, const int *M, const int *N, const float *alpha, const float *A, const int *lda,
            const float *x, const int *incx, const float *beta, float *y, const int *incy);
void dgemv_(const char *transa, const int *M, const int *N, const double *alpha, const double *A, const int *lda,
            const double *x, const int *incx, const double *beta, double *y, const int *incy);
void cgemv_(const char *transa, const int *M, const int *N, const std::complex<float> *alpha, const std::complex<float> *A, const int *lda,
            const std::complex<float> *x, const int *incx, const std::complex<float> *beta, std::complex<float> *y, const int *incy);
void zgemv_(const char *transa, const int *M, const int *N, const std::complex<double> *alpha, const std::complex<double> *A, const int *lda,
            const std::complex<double> *x, const int *incx, const std::complex<double> *beta, std::complex<double> *y, const int *incy);

void sgbmv_(const char *transa, const int *M, const int *N, const int *kl, const int *ku,
            const float *alpha, const float *A, const int *lda,
            const float *x, const int *incx, const float *beta, float *y, const int *incy);
void dgbmv_(const char *transa, const int *M, const int *N, const int *kl, const int *ku,
            const double *alpha, const double *A, const int *lda,
            const double *x, const int *incx, const double *beta, double *y, const int *incy);
void cgbmv_(const char *transa, const int *M, const int *N, const int *kl, const int *ku,
            const std::complex<float> *alpha, const std::complex<float> *A, const int *lda,
            const std::complex<float> *x, const int *incx, const std::complex<float> *beta, std::complex<float> *y, const int *incy);
void zgbmv_(const char *transa, const int *M, const int *N, const int *kl, const int *ku,
            const std::complex<double> *alpha, const std::complex<double> *A, const int *lda,
            const std::complex<double> *x, const int *incx, const std::complex<double> *beta, std::complex<double> *y, const int *incy);

void ssymv_(const char *uplo, const int *N, const float *alpha, const float *A, const int *lda,
            const float *x, const int *incx, const float *beta, float *y, const int *incy);
void dsymv_(const char *uplo, const int *N, const double *alpha, const double *A, const int *lda,
            const double *x, const int *incx, const double *beta, double *y, const int *incy);
void csymv_(const char *uplo, const int *N, const std::complex<float> *alpha, const std::complex<float> *A, const int *lda,
            const std::complex<float> *x, const int *incx, const std::complex<float> *beta, std::complex<float> *y, const int *incy);
void zsymv_(const char *uplo, const int *N, const std::complex<double> *alpha, const std::complex<double> *A, const int *lda,
            const std::complex<double> *x, const int *incx, const std::complex<double> *beta, std::complex<double> *y, const int *incy);
void chemv_(const char *uplo, const int *N, const std::complex<float> *alpha, const std::complex<float> *A, const int *lda,
            const std::complex<float> *x, const int *incx, const std::complex<float> *beta, std::complex<float> *y, const int *incy);
void zhemv_(const char *uplo, const int *N, const std::complex<double> *alpha, const std::complex<double> *A, const int *lda,
            const std::complex<double> *x, const int *incx, const std::complex<double> *beta, std::complex<double> *y, const int *incy);

void ssbmv_(const char *uplo, const int *N, const int *k, const float *alpha, const float *A, const int *lda,
            const float *x, const int *incx, const float *beta, float *y, const int *incy);
void dsbmv_(const char *uplo, const int *N, const int *k, const double *alpha, const double *A, const int *lda,
            const double *x, const int *incx, const double *beta, double *y, const int *incy);
#ifdef HALA_HAS_ZSBMV
void csbmv_(const char *uplo, const int *N, const int *k, const std::complex<float> *alpha, const std::complex<float> *A, const int *lda,
            const std::complex<float> *x, const int *incx, const std::complex<float> *beta, std::complex<float> *y, const int *incy);
void zsbmv_(const char *uplo, const int *N, const int *k, const std::complex<double> *alpha, const std::complex<double> *A, const int *lda,
            const std::complex<double> *x, const int *incx, const std::complex<double> *beta, std::complex<double> *y, const int *incy);
#endif
void chbmv_(const char *uplo, const int *N, const int *k, const std::complex<float> *alpha, const std::complex<float> *A, const int *lda,
            const std::complex<float> *x, const int *incx, const std::complex<float> *beta, std::complex<float> *y, const int *incy);
void zhbmv_(const char *uplo, const int *N, const int *k, const std::complex<double> *alpha, const std::complex<double> *A, const int *lda,
            const std::complex<double> *x, const int *incx, const std::complex<double> *beta, std::complex<double> *y, const int *incy);

void sspmv_(const char *uplo, const int *N, const float *alpha, const float *A,
            const float *x, const int *incx, const float *beta, float *y, const int *incy);
void dspmv_(const char *uplo, const int *N, const double *alpha, const double *A,
            const double *x, const int *incx, const double *beta, double *y, const int *incy);
void cspmv_(const char *uplo, const int *N, const std::complex<float> *alpha, const std::complex<float> *A,
            const std::complex<float> *x, const int *incx, const std::complex<float> *beta, std::complex<float> *y, const int *incy);
void zspmv_(const char *uplo, const int *N, const std::complex<double> *alpha, const std::complex<double> *A,
            const std::complex<double> *x, const int *incx, const std::complex<double> *beta, std::complex<double> *y, const int *incy);
void chpmv_(const char *uplo, const int *N, const std::complex<float> *alpha, const std::complex<float> *A,
            const std::complex<float> *x, const int *incx, const std::complex<float> *beta, std::complex<float> *y, const int *incy);
void zhpmv_(const char *uplo, const int *N, const std::complex<double> *alpha, const std::complex<double> *A,
            const std::complex<double> *x, const int *incx, const std::complex<double> *beta, std::complex<double> *y, const int *incy);

void strmv_(const char *uplo, const char *transa, const char *diag, const int *N, const float *A, const int *lda,
            const float *x, const int *incx);
void dtrmv_(const char *uplo, const char *transa, const char *diag, const int *N, const double *A, const int *lda,
            const double *x, const int *incx);
void ctrmv_(const char *uplo, const char *transa, const char *diag, const int *N, const std::complex<float> *A, const int *lda,
            const std::complex<float> *x, const int *incx);
void ztrmv_(const char *uplo, const char *transa, const char *diag, const int *N, const std::complex<double> *A, const int *lda,
            const std::complex<double> *x, const int *incx);

void strsv_(const char *uplo, const char *transa, const char *diag, const int *N, const float *A, const int *lda,
            const float *x, const int *incx);
void dtrsv_(const char *uplo, const char *transa, const char *diag, const int *N, const double *A, const int *lda,
            const double *x, const int *incx);
void ctrsv_(const char *uplo, const char *transa, const char *diag, const int *N, const std::complex<float> *A, const int *lda,
            const std::complex<float> *x, const int *incx);
void ztrsv_(const char *uplo, const char *transa, const char *diag, const int *N, const std::complex<double> *A, const int *lda,
            const std::complex<double> *x, const int *incx);

void stbmv_(const char *uplo, const char *transa, const char *diag, const int *N, const int *k, const float *A, const int *lda,
            const float *x, const int *incx);
void dtbmv_(const char *uplo, const char *transa, const char *diag, const int *N, const int *k, const double *A, const int *lda,
            const double *x, const int *incx);
void ctbmv_(const char *uplo, const char *transa, const char *diag, const int *N, const int *k, const std::complex<float> *A, const int *lda,
            const std::complex<float> *x, const int *incx);
void ztbmv_(const char *uplo, const char *transa, const char *diag, const int *N, const int *k, const std::complex<double> *A, const int *lda,
            const std::complex<double> *x, const int *incx);

void stbsv_(const char *uplo, const char *transa, const char *diag, const int *N, const int *k, const float *A, const int *lda,
            const float *x, const int *incx);
void dtbsv_(const char *uplo, const char *transa, const char *diag, const int *N, const int *k, const double *A, const int *lda,
            const double *x, const int *incx);
void ctbsv_(const char *uplo, const char *transa, const char *diag, const int *N, const int *k, const std::complex<float> *A, const int *lda,
            const std::complex<float> *x, const int *incx);
void ztbsv_(const char *uplo, const char *transa, const char *diag, const int *N, const int *k, const std::complex<double> *A, const int *lda,
            const std::complex<double> *x, const int *incx);

void stpmv_(const char *uplo, const char *transa, const char *diag, const int *N, const float *A,
            const float *x, const int *incx);
void dtpmv_(const char *uplo, const char *transa, const char *diag, const int *N, const double *A,
            const double *x, const int *incx);
void ctpmv_(const char *uplo, const char *transa, const char *diag, const int *N, const std::complex<float> *A,
            const std::complex<float> *x, const int *incx);
void ztpmv_(const char *uplo, const char *transa, const char *diag, const int *N, const std::complex<double> *A,
            const std::complex<double> *x, const int *incx);

void stpsv_(const char *uplo, const char *transa, const char *diag, const int *N, const float *A,
            const float *x, const int *incx);
void dtpsv_(const char *uplo, const char *transa, const char *diag, const int *N, const double *A,
            const double *x, const int *incx);
void ctpsv_(const char *uplo, const char *transa, const char *diag, const int *N, const std::complex<float> *A,
            const std::complex<float> *x, const int *incx);
void ztpsv_(const char *uplo, const char *transa, const char *diag, const int *N, const std::complex<double> *A,
            const std::complex<double> *x, const int *incx);

void sger_(const int *M, const int *N, const float *alpha, const float *x, const int *incx, const float *y, const int *incy, float *A, const int *lda);
void dger_(const int *M, const int *N, const double *alpha, const double *x, const int *incx, const double *y, const int *incy, double *A, const int *lda);
void cgeru_(const int *M, const int *N, const std::complex<float> *alpha, const std::complex<float> *x, const int *incx,
            const std::complex<float> *y, const int *incy, std::complex<float> *A, const int *lda);
void zgeru_(const int *M, const int *N, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx,
            const std::complex<double> *y, const int *incy, std::complex<double> *A, const int *lda);
void cgerc_(const int *M, const int *N, const std::complex<float> *alpha, const std::complex<float> *x, const int *incx,
            const std::complex<float> *y, const int *incy, std::complex<float> *A, const int *lda);
void zgerc_(const int *M, const int *N, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx,
            const std::complex<double> *y, const int *incy, std::complex<double> *A, const int *lda);

void ssyr_(const char *uplo, const int *N, const float *alpha, const float *x, const int *incx, float *A, const int *lda);
void dsyr_(const char *uplo, const int *N, const double *alpha, const double *x, const int *incx, double *A, const int *lda);
void csyr_(const char *uplo, const int *N, const std::complex<float> *alpha, const std::complex<float> *x, const int *incx, std::complex<float> *A, const int *lda);
void zsyr_(const char *uplo, const int *N, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx, std::complex<double> *A, const int *lda);
void cher_(const char *uplo, const int *N, const std::complex<float> *alpha, const std::complex<float> *x, const int *incx, std::complex<float> *A, const int *lda);
void zher_(const char *uplo, const int *N, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx, std::complex<double> *A, const int *lda);

void ssyr2_(const char *uplo, const int *N, const float *alpha, const float *x, const int *incx, const float *y, const int *incy, float *A, const int *lda);
void dsyr2_(const char *uplo, const int *N, const double *alpha, const double *x, const int *incx, const double *y, const int *incy, double *A, const int *lda);
#ifdef HALA_HAS_ZSYR2
void csyr2_(const char *uplo, const int *N, const std::complex<float> *alpha, const std::complex<float> *x, const int *incx,
            const std::complex<float> *y, const int *incy, std::complex<float> *A, const int *lda);
void zsyr2_(const char *uplo, const int *N, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx,
            const std::complex<double> *y, const int *incy, std::complex<double> *A, const int *lda);
#endif
void cher2_(const char *uplo, const int *N, const std::complex<float> *alpha, const std::complex<float> *x, const int *incx,
            const std::complex<float> *y, const int *incy, std::complex<float> *A, const int *lda);
void zher2_(const char *uplo, const int *N, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx,
            const std::complex<double> *y, const int *incy, std::complex<double> *A, const int *lda);

void sspr_(const char *uplo, const int *N, const float *alpha, const float *x, const int *incx, float *A);
void dspr_(const char *uplo, const int *N, const double *alpha, const double *x, const int *incx, double *A);
void cspr_(const char *uplo, const int *N, const std::complex<float> *alpha, const std::complex<float> *x, const int *incx, std::complex<float> *A);
void zspr_(const char *uplo, const int *N, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx, std::complex<double> *A);
void chpr_(const char *uplo, const int *N, const std::complex<float> *alpha, const std::complex<float> *x, const int *incx, std::complex<float> *A);
void zhpr_(const char *uplo, const int *N, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx, std::complex<double> *A);

void sspr2_(const char *uplo, const int *N, const float *alpha, const float *x, const int *incx, const float *y, const int *incy, float *A);
void dspr2_(const char *uplo, const int *N, const double *alpha, const double *x, const int *incx, const double *y, const int *incy, double *A);
#ifdef HALA_HAS_ZSPR2
void cspr2_(const char *uplo, const int *N, const std::complex<float> *alpha, const std::complex<float> *x, const int *incx,
            const std::complex<float> *y, const int *incy, std::complex<float> *A);
void zspr2_(const char *uplo, const int *N, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx,
            const std::complex<double> *y, const int *incy, std::complex<double> *A);
#endif
void chpr2_(const char *uplo, const int *N, const std::complex<float> *alpha, const std::complex<float> *x, const int *incx,
            const std::complex<float> *y, const int *incy, std::complex<float> *A);
void zhpr2_(const char *uplo, const int *N, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx,
            const std::complex<double> *y, const int *incy, std::complex<double> *A);

// BLAS Level 3
void strmm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *M, const int *N,
            const float *alpha, const float *A, const int *lda, float *B, const int *ldb);
void dtrmm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *M, const int *N,
            const double *alpha, const double *A, const int *lda, double *B, const int *ldb);
void ctrmm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *M, const int *N,
            const std::complex<float> *alpha, const std::complex<float> *A, const int *lda, std::complex<float> *B, const int *ldb);
void ztrmm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *M, const int *N,
            const std::complex<double> *alpha, const std::complex<double> *A, const int *lda, std::complex<double> *B, const int *ldb);

void strsm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *M, const int *N,
            const float *alpha, const float *A, const int *lda, float *B, const int *ldb);
void dtrsm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *M, const int *N,
            const double *alpha, const double *A, const int *lda, double *B, const int *ldb);
void ctrsm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *M, const int *N,
            const std::complex<float> *alpha, const std::complex<float> *A, const int *lda, std::complex<float> *B, const int *ldb);
void ztrsm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *M, const int *N,
            const std::complex<double> *alpha, const std::complex<double> *A, const int *lda, std::complex<double> *B, const int *ldb);

void ssyrk_(const char *uplo, const char *trans, const int *N, const int *K, const float *alpha,
            const float *A, const int *lda, const float *beta, float *C, const int *ldc);
void dsyrk_(const char *uplo, const char *trans, const int *N, const int *K, const double *alpha,
            const double *A, const int *lda, const double *beta, double *C, const int *ldc);
void csyrk_(const char *uplo, const char *trans, const int *N, const int *K, const std::complex<float> *alpha,
            const std::complex<float> *A, const int *lda, const std::complex<float> *beta, std::complex<float> *C, const int *ldc);
void zsyrk_(const char *uplo, const char *trans, const int *N, const int *K, const std::complex<double> *alpha,
            const std::complex<double> *A, const int *lda, const std::complex<double> *beta, std::complex<double> *C, const int *ldc);
void cherk_(const char *uplo, const char *trans, const int *N, const int *K, const std::complex<float> *alpha,
            const std::complex<float> *A, const int *lda, const std::complex<float> *beta, std::complex<float> *C, const int *ldc);
void zherk_(const char *uplo, const char *trans, const int *N, const int *K, const std::complex<double> *alpha,
            const std::complex<double> *A, const int *lda, const std::complex<double> *beta, std::complex<double> *C, const int *ldc);

void ssyr2k_(const char *uplo, const char *trans, const int *N, const int *K, const float *alpha,
             const float *A, const int *lda, const float *B, const int *ldb, const float *beta, float *C, const int *ldc);
void dsyr2k_(const char *uplo, const char *trans, const int *N, const int *K, const double *alpha,
             const double *A, const int *lda, const double *B, const int *ldb, const double *beta, double *C, const int *ldc);
void csyr2k_(const char *uplo, const char *trans, const int *N, const int *K, const std::complex<float> *alpha,
             const std::complex<float> *A, const int *lda, const std::complex<float> *B, const int *ldb,
             const std::complex<float> *beta, std::complex<float> *C, const int *ldc);
void zsyr2k_(const char *uplo, const char *trans, const int *N, const int *K, const std::complex<double> *alpha,
             const std::complex<double> *A, const int *lda, const std::complex<double> *B, const int *ldb,
             const std::complex<double> *beta, std::complex<double> *C, const int *ldc);
void cher2k_(const char *uplo, const char *trans, const int *N, const int *K, const std::complex<float> *alpha,
             const std::complex<float> *A, const int *lda, const std::complex<float> *B, const int *ldb,
             const std::complex<float> *beta, std::complex<float> *C, const int *ldc);
void zher2k_(const char *uplo, const char *trans, const int *N, const int *K, const std::complex<double> *alpha,
             const std::complex<double> *A, const int *lda, const std::complex<double> *B, const int *ldb,
             const std::complex<double> *beta, std::complex<double> *C, const int *ldc);

void ssymm_(const char *side, const char *uplo, const int *M, const int *N, const float *alpha,
            const float *A, const int *lda, const float *B, const int *ldb, const float *beta, float *C, const int *ldc);
void dsymm_(const char *side, const char *uplo, const int *M, const int *N, const double *alpha,
            const double *A, const int *lda, const double *B, const int *ldb, const double *beta, double *C, const int *ldc);
void csymm_(const char *side, const char *uplo, const int *M, const int *N, const std::complex<float> *alpha,
            const std::complex<float> *A, const int *lda, const std::complex<float> *B, const int *ldb, const std::complex<float> *beta, std::complex<float> *C, const int *ldc);
void zsymm_(const char *side, const char *uplo, const int *M, const int *N, const std::complex<double> *alpha,
            const std::complex<double> *A, const int *lda, const std::complex<double> *B, const int *ldb, const std::complex<double> *beta, std::complex<double> *C, const int *ldc);
void chemm_(const char *side, const char *uplo, const int *M, const int *N, const std::complex<float> *alpha,
            const std::complex<float> *A, const int *lda, const std::complex<float> *B, const int *ldb, const std::complex<float> *beta, std::complex<float> *C, const int *ldc);
void zhemm_(const char *side, const char *uplo, const int *M, const int *N, const std::complex<double> *alpha,
            const std::complex<double> *A, const int *lda, const std::complex<double> *B, const int *ldb, const std::complex<double> *beta, std::complex<double> *C, const int *ldc);

void sgemm_(const char *transa, const char *transb, const int *M, const int *N,  const int *K, const float *alpha,
            const float *A, const int *lda, const float *B, const int *ldb, const float *beta, float *C, const int *ldc);
void dgemm_(const char *transa, const char *transb, const int *M, const int *N,  const int *K, const double *alpha,
            const double *A, const int *lda, const double *B, const int *ldb, const double *beta, double *C, const int *ldc);
void cgemm_(const char *transa, const char *transb, const int *M, const int *N,  const int *K, const std::complex<float> *alpha,
            const std::complex<float> *A, const int *lda, const std::complex<float> *B, const int *ldb, const std::complex<float> *beta, std::complex<float> *C, const int *ldc);
void zgemm_(const char *transa, const char *transb, const int *M, const int *N,  const int *K, const std::complex<double> *alpha,
            const std::complex<double> *A, const int *lda, const std::complex<double> *B, const int *ldb, const std::complex<double> *beta, std::complex<double> *C, const int *ldc);
#endif

}
#endif // __HALA_DOXYGEN_SKIP
#endif

#endif
