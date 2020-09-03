#ifndef __HALA_BLAS_EXTENSIONS_HPP
#define __HALA_BLAS_EXTENSIONS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_extensions_checker.hpp"

namespace hala{

/*!
 * \internal
 * \file hala_blas_extensions.hpp
 * \ingroup HALACORE
 * \brief Extension for dense linear algebra methods.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \endinternal
 */

/*!
 * \ingroup HALAWAXEXT
 * \addtogroup HALAWAXBLAS BLAS-like extensions
 *
 * Methods that combine multiple BLAS, cuBLAS and/or manually implemented methods
 * the perform some simple operations, such as batch dot-product or matrix transpose.
 */

/*!
 * \ingroup HALAWAXBLAS
 * \brief Compute the scaled vector sum of a batch of one dimensional vectors and scalars, where the vectors are stored contiguously.
 *
 * Computes \f$ y_i = \alpha_c \alpha_i x_i + y_i \f$ for a set of vectors.
 * - \b x and \b y have the same size and store a batch of vectors contiguously,
 *   i.e., the first vector is in the first strip of length x.size() / batch_size
 * - \b alpha is a vector of length \b batch_size indicating the scalars for each vector pair, i.e., alpha_i
 * - \b alphac is a constant that multiplies all scalars
 *
 * Overloads
 * \code
 *  batch_axpy(engine, batch_size, alphac, alpha, x, y);
 * \endcode
 */
template<typename FPA, class VectorLikeA, class VectorLikeX, class VectorLikeY>
void batch_axpy(int batch_size, FPA alphac, VectorLikeA const &alpha, VectorLikeX const &x, VectorLikeY &&y){
    check_types(alpha, x, y);
    assert( valid::batch_axpy(batch_size, alpha, x, y) );
    int N = get_size_int(x) / batch_size;

    using standard_type = get_standard_type<VectorLikeA>;
    auto fpalpha = get_cast<standard_type>(alphac);

    auto salphs = get_standard_data(alpha);
    auto sx = get_standard_data(x);
    auto sy = get_standard_data(y);

    for(int i=0; i<batch_size; i++){
        auto wrap = wrap_array(&sy[hala_size(i, N)], N);
        axpy(N, fpalpha * salphs[i], &sx[hala_size(i, N)], 1, wrap, 1);
    }
}

/*!
 * \ingroup HALAWAXBLAS
 * \brief Computes the dot-product for a batch of one dimensional vectors.
 *
 * Computes \f$ result_i = dot( x_i, y_i ) \f$ in either real or conjugate modes.
 * - \b x and \b y have the same size and store a batch of vectors contiguously,
 *   i.e., the first vector is in the first strip of length x.size() / batch_size
 * - \b result will be resized to at least \b batch_size and will hold the dot-products
 *
 * Overloads
 * \code
 *  batch_dot(engine, batch_size, alphac, alpha, x, y);
 * \endcode
 */
template<bool conjugate = true, class VectorLikeX, class VectorLikeY, class VectorLikeR>
void batch_dot(int batch_size, VectorLikeX const &x, VectorLikeY const &y, VectorLikeR &&result){
    check_types(x, y, result);
    assert( valid::batch_dot(batch_size, x, y) );
    int N = get_size_int(x) / batch_size;
    check_set_size(assume_output, result, batch_size);

    auto sx = get_standard_data(x);
    auto sy = get_standard_data(y);
    auto sresult = get_standard_data(result);

    for(int i=0; i<batch_size; i++)
        sresult[i] = dot<conjugate>(N, &sx[hala_size(i, N)], 1, &sy[hala_size(i, N)], 1);
}

/*!
 * \ingroup HALAWAXBLAS
 * \brief Scales a batch of vectors by a set of constants.
 *
 * Assuming that \b x has a batch of vectors stored contiguously and that \b alpha
 * holds a set of scalars, this scales each vector by the corresponding scalar.
 * It is equivalent to calling hala::scal() on each individual vector,
 * but the CUDA version of the operation is performed with hala::dgmm().
 *
 * Overloads:
 * \code
 *  batch_scal(batch_size, alpha, x);
 *  batch_scal(engine, batch_size, alpha, x);
 * \endcode
 */
template<class VectorLikeA, class VectorLikeX>
void batch_scal(int batch_size, VectorLikeA const &alpha, VectorLikeX &&x){
    check_types(alpha, x);
    assert( valid::batch_scal(batch_size, alpha, x) );
    int N = get_size_int(x) / batch_size;

    auto sx = get_standard_data(x);
    auto sa = get_standard_data(alpha);

    for(int i=0; i<batch_size; i++)
        scal(N, sa[i], &sx[hala_size(i, N)], 1);
}

/*!
 * \ingroup HALAWAXBLAS
 * \brief Divide the two vectors component-wise.
 *
 * Computes \f$ z_i = x_i / y_i \f$, component-wise division of the vectors.
 *
 * Overloads:
 * \code
 *  vdivide(x, y, z);
 *  vdivide(engine, x, y, z);
 * \endcode
 */
template<class VectorLikeX, class VectorLikeY, class VectorLikeZ>
void vdivide(VectorLikeX const &x, VectorLikeY const &y, VectorLikeZ &&z){
    check_types(x, y, z);
    assert( get_size(x) == get_size(y) );
    size_t N = get_size(x);
    check_set_size(assume_output, z, N);

    auto sx = get_standard_data(x);
    auto sy = get_standard_data(y);
    auto sz = get_standard_data(z);

    for(size_t i=0; i<N; i++)
        sz[i] = sx[i] / sy[i];
}


#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)

template<typename FPA, class VectorLikeA, class VectorLikeX, class VectorLikeY>
void batch_axpy(cpu_engine const&, int batch_size, FPA alphac, VectorLikeA const &alpha, VectorLikeX const &x, VectorLikeY &&y){
    batch_axpy(batch_size, alphac, alpha, x, y);
}
template<bool conjugate = true, class VectorLikeX, class VectorLikeY, class VectorLikeR>
void batch_dot(cpu_engine const&, int batch_size, VectorLikeX const &x, VectorLikeY const &y, VectorLikeR &&result){
    batch_dot<conjugate>(batch_size, x, y, result);
}
template<class VectorLikeX, class VectorLikeY, class VectorLikeR>
void batch_dotu(int batch_size, VectorLikeX const &x, VectorLikeY const &y, VectorLikeR &&result){
    batch_dot<false>(batch_size, x, y, std::forward<VectorLikeR>(result));
}
template<class VectorLikeX, class VectorLikeY, class VectorLikeR>
void batch_dotu(cpu_engine const&, int batch_size, VectorLikeX const &x, VectorLikeY const &y, VectorLikeR &&result){
    batch_dot<false>(batch_size, x, y, std::forward<VectorLikeR>(result));
}
template<class VectorLikeA, class VectorLikeX>
void batch_scal(cpu_engine const&, int batch_size, VectorLikeA const &alpha, VectorLikeX &&x){
    batch_scal(batch_size, alpha, x);
}
template<class VectorLikeX, class VectorLikeY, class VectorLikeZ>
void vdivide(cpu_engine const&, VectorLikeX const &x, VectorLikeY const &y, VectorLikeZ &&z){
    vdivide(x, y, z);
}

#endif

#ifdef HALA_ENABLE_GPU

/*!
 * \ingroup HALAWAXBLAS
 * \brief Transposes the matrix.
 *
 * Rearranges the memory to form the transpose of the matrix.
 * - \b A is an \b M by \b N matrix in column major format.
 * - \b At is an \b N by \b M matrix holding the transpose.
 * - \b lda is minimum M, \b ldat is minimum N (define the leading dimensions)
 * Uses hala::geam() at the back-end.
 *
 * Overloads:
 * \code
 *  transpose(engine, M, N, A, lda, At, ldat);
 *  transpose(engine, M, N, A, At, lda = -1, ldat = -1);
 * \endcode
 * The leading dimensions take the minimum acceptable values.
 */
template<class VectorLikeA, class VectorLikeAt>
void transpose(gpu_engine const &engine, int M, int N, VectorLikeA const &A, int lda, VectorLikeAt &At, int ldat){
    gpu_pntr<host_pntr> hold(engine);
    geam(engine, 'T', 'T', N, M, 1, A, lda, 0, A, lda, At, ldat);
}

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)

template<typename FPA, class VectorLikeA, class VectorLikeX, class VectorLikeY>
void batch_axpy(gpu_engine const &engine, int batch_size, FPA alphac, VectorLikeA const &alpha, VectorLikeX const &x, VectorLikeY &&y){
    check_types(alpha, x, y);
    engine.check_gpu(x, y, alpha);
    assert( valid::batch_axpy(batch_size, alpha, x, y) );
    int N = get_size_int(x) / batch_size;

    using scalar_type = get_scalar_type<VectorLikeA>;
    auto fpalpha = get_cast<scalar_type>(alphac);

    auto scaledx = new_vector(engine, x);
    dgmm(engine, 'R', N, batch_size, x, alpha, scaledx);
    axpy(engine, fpalpha, scaledx, y);
}

template<bool conjugate = true, class VectorLikeX, class VectorLikeY, class VectorLikeR>
void batch_dot(gpu_engine const &engine, int batch_size, VectorLikeX const &x, VectorLikeY const &y, VectorLikeR &&result){
    check_types(x, y, result);
    engine.check_gpu(x, y, result);
    assert( valid::batch_dot(batch_size, x, y) );
    int N = get_size_int(x) / batch_size;
    check_set_size(assume_output, result, batch_size);
    gpu_pntr<host_pntr> hold(engine);

    using scalar_type = get_scalar_type<VectorLikeX>;
    auto product = new_vector(engine, x); // engine.vector<scalar_type>(get_size(x));
    if __HALA_CONSTEXPR_IF__ (conjugate && is_complex<scalar_type>::value){
        // need to manually conjugate x
        auto x_conj = new_vector(engine, x);
        geam(engine, 'C', 'C', get_size_int(x), 1, 1.0, x, 1, 0, x, 1, x_conj, get_size_int(x));
        dgmm(engine, 'L', get_size_int(x), 1, x_conj, y, product);
    }else{
        dgmm(engine, 'L', get_size_int(x), 1, x, y, product);
    }

    auto gpu_ones = engine.vector<scalar_type>(static_cast<size_t>(N), get_cast<scalar_type>(1.0));
    gemv(engine, 'T', N, batch_size, 1.0, product, gpu_ones, 0.0, result);
}
template<class VectorLikeX, class VectorLikeY, class VectorLikeR>
void batch_dotu(gpu_engine const &engine, int batch_size, VectorLikeX const &x, VectorLikeY const &y, VectorLikeR &&result){
    batch_dot<false>(engine, batch_size, x, y, std::forward<VectorLikeR>(result));
}

template<class VectorLikeA, class VectorLikeX>
void batch_scal(gpu_engine const &engine, int batch_size, VectorLikeA const &alpha, VectorLikeX &&x){
    check_types(alpha, x);
    engine.check_gpu(alpha, x);
    assert( valid::batch_scal(batch_size, alpha, x) );

    int N = get_size_int(x) / batch_size;

    auto tmp = new_vector(engine, x);
    dgmm(engine, 'R', N, batch_size, x, N, alpha, 1, tmp, N);
    vcopy(engine, tmp, x);
}

template<class VectorLikeX, class VectorLikeY, class VectorLikeZ>
void vdivide(gpu_engine const &engine, VectorLikeX const &x, VectorLikeY const &y, VectorLikeZ &&z){
    check_types(x, y, z);
    engine.check_gpu(x, y, z);
    assert( get_size(x) == get_size(y) );
    int N = get_size_int(x);

    hala::vcopy(engine, x, z);
    hala::tbsv(engine, 'U', 'N', 'N', N, 0, y, z);
}

template<class VectorLikeA, class VectorLikeAt>
void transpose(gpu_engine const &engine, int M, int N, VectorLikeA const &A, VectorLikeAt &At, int lda = -1, int ldat = -1){
    valid::default_ld(M, lda);
    valid::default_ld(N, ldat);
    transpose(engine, M, N, A, lda, At, ldat);
}
template<class VectorLikeA, class VectorLikeAt>
void transpose(mixed_engine const &e, int M, int N, VectorLikeA const &A, int lda, VectorLikeAt &At, int ldat){
    auto gA = e.gpu().load(A);
    auto gAt = gpu_bind_vector(e.gpu(), At);
    transpose(e.gpu(), M, N, gA, gAt, lda, ldat);
}
template<class VectorLikeA, class VectorLikeAt>
void transpose(mixed_engine const &e, int M, int N, VectorLikeA const &A, VectorLikeAt &At, int lda = -1, int ldat = -1){
    transpose(e, M, N, A, At, lda, ldat);
}

template<typename FPA, class VectorLikeA, class VectorLikeX, class VectorLikeY>
void batch_axpy(mixed_engine const &e, int batch_size, FPA alphac, VectorLikeA const &alpha, VectorLikeX const &x, VectorLikeY &y){
    gpu_pntr<host_pntr> hold(e.gpu());
    auto galpha = e.gpu().load(alpha);
    auto gx = e.gpu().load(x);
    auto gy = gpu_bind_vector(e.gpu(), y);
    batch_axpy(e.gpu(), batch_size, alphac, galpha, gx, gy);
}

template<bool conjugate = true, class VectorLikeX, class VectorLikeY, class VectorLikeR>
void batch_dot(mixed_engine const &e, int batch_size, VectorLikeX const &x, VectorLikeY const &y, VectorLikeR &result){
    auto gx = e.gpu().load(x);
    auto gy = e.gpu().load(y);
    auto gresult = gpu_bind_vector(e.gpu(), result);
    batch_dot<conjugate>(e.gpu(), batch_size, gx, gy, gresult);
}

template<class VectorLikeX, class VectorLikeY, class VectorLikeR>
void batch_dotu(mixed_engine const &e, int batch_size, VectorLikeX const &x, VectorLikeY const &y, VectorLikeR &result){
    batch_dot<false>(e, batch_size, x, y, result);
}

template<class VectorLikeA, class VectorLikeX>
void batch_scal(mixed_engine const &e, int batch_size, VectorLikeA const &alpha, VectorLikeX &&x){
    auto ga = e.gpu().load(alpha);
    auto gx = gpu_bind_vector(e.gpu(), x);
    batch_scal(e.gpu(), batch_size, ga, gx);
}

template<class VectorLikeX, class VectorLikeY, class VectorLikeZ>
void vdivide(mixed_engine const &e, VectorLikeX const &x, VectorLikeY const &y, VectorLikeZ &&z){
    auto gx = e.gpu().load(x);
    auto gy = e.gpu().load(y);
    auto gz = gpu_bind_vector(e.gpu(), z);
    vdivide(e.gpu(), gx, gy, gz);
}

#endif // end overloads

#endif // end of CUDA

/*!
 * \ingroup HALAWAXBLAS
 * \brief Returns the norm of the largest vector in a batch.
 *
 * Here \b x represents a batch of vectors stored contiguously, this method returns
 * the norm of the largest vector.
 * - \b batch_size is a non-negative number indicating the size of the batch
 * - \b x is the batch of vectors, there is no check if the size is an even multiple of the batch size,
 *  but it is assumed that the size of each vector is size-of-x divided by batch_size
 *
 * Overloads:
 * \code
 *  batch_max_norm2(batch_size, x);
 *  batch_max_norm2(engine, batch_size, x);
 * \endcode
 * Implementations are provided for all compute engines, when called without an engine
 * the cpu_engine will be assumed.
 */
template<class cengine, class VectorLike>
auto batch_max_norm2(cengine const &engine, int batch_size, VectorLike const &x){
    check_types(x);
    assert( batch_size > 0 );
    assert( check_size(x, batch_size, get_size_int(x) / batch_size) );

    auto norms = new_vector(engine, x);
    batch_dot(engine, batch_size, x, x, norms);
    int nmax = iamax(engine, norms);

    auto nd = get_standard_data(norms);
    return std::sqrt( norm2(engine, 1, &nd[nmax], 1) );
}

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)
template<class VectorLike>
auto batch_max_norm2(int batch_size, VectorLike const &x){
    cpu_engine const ecpu;
    return batch_max_norm2(ecpu, batch_size, x);
}
#ifdef HALA_ENABLE_GPU
template<class VectorLike>
auto batch_max_norm2(mixed_engine const &e, int batch_size, VectorLike const &x){
    auto gx = e.gpu().load(x);
    return batch_max_norm2(e.gpu(), batch_size, gx);
}
#endif
#endif

}

#endif
