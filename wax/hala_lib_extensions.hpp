#ifndef __HALA_LIB_EXTENSIONS_HPP
#define __HALA_LIB_EXTENSIONS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_sparse.hpp" // also brings in blas

#include "hala_lapack.hpp"

#ifdef HALA_ENABLE_GPU
#include "hala_gpu.hpp"
#endif


namespace hala{

/*!
 * \internal
 * \file hala_lib_extensions.hpp
 * \ingroup HALACORE
 * \brief Extension methods that combine multiple hala modules.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \endinternal
 */

/*!
 * \defgroup HALAWAXEXT HALA Wrapped Accelerated Extensions
 *
 * Defines common methods that require interoperability between multiple HALA modules.
 * - hala::mixed_engine allows for all BLAS and Sparse methods to be used with input
 *  vectors on the CPU while the operations are performed on the GPU (data is moved
 *  behind the scenes)
 * - \ref HALAWAXBLAS that utilize one or more of the existing BLAS methods (and sometimes
 *  simple additional algorithms) that perform extra functionality, e.g., batch dot and
 *  axpy operations
 */

/*!
 * \ingroup HALAWAXEXT
 * \brief Return a copy of the vector using an member function for the engine.
 */
template<typename VectorLike>
auto cpu_engine::vcopy(VectorLike const &x) const{
    auto y = new_vector(*this, x);
    hala::vcopy(x, y);
    return y;
}

#ifdef HALA_ENABLE_GPU

/*!
 * \ingroup HALAWAXEXT
 * \brief A mixed type engine that performs computations on the GPU but takes inputs on the CPU.
 *
 * The mixed engine holds a single non-owning const reference to a gpu engine
 * and indicates that methods should accept input on the CPU side, push the data to the GPU,
 * perform the work and pull the data back.
 * BLAS level 3 methods can benefit from such approach (for large matrices) and the methods
 * simplify testing for levels 1 and 2.
 * Higher level (e.g., solver) methods can also benefit from the approach, e.g.,
 * load data once, perform numerous computations with different levels of BLAS
 * and finally bring the result back to the CPU.
 */
struct mixed_engine{
    //! \brief Constructor initializing the reference.
    explicit mixed_engine(gpu_engine const &ingpu) : cengine(ingpu){}
    //! \brief The reference to the GPU engine.
    gpu_engine const& gpu() const{ return cengine; }

    //! \brief Default vector is std::vector
    template<typename T> using dfvector = typename std::vector<T>;

    //! \brief Holds a reference to a gpu_engine.
    gpu_engine const &cengine;

    //! \brief Return a copy of the vector using an member function for the engine.
    template<typename VectorLike> auto vcopy(VectorLike const &x) const;
};

/*!
 * \ingroup HALAWAXEXT
 * \brief Make a vector in CPU memory with type matching the input \b x.
 */
template<class VectorLike>
auto new_vector(mixed_engine const &engine, VectorLike const &){
    return vector_constructor<get_vdefault<mixed_engine, VectorLike>>::make_one(engine);
}

template<typename VectorLike> auto mixed_engine::vcopy(VectorLike const &x) const{
    auto y = new_vector(*this, x);
    hala::vcopy(x, y);
    return y;
}

/*!
 * \ingroup HALAWAXEXT
 * \brief Fill the first \b num_entries of the vector with zeros.
 */
template<class VectorLike>
inline void set_zero(mixed_engine const&, size_t num_entries, VectorLike &&x){
    cpu_engine const ecpu;
    set_zero(ecpu, num_entries, x);
}

/*!
 * \ingroup HALAWAXEXT
 * \brief Create an array wrapper for the mixed_engine, actually this is the same as the cpu_engine.
 */
template<typename ArrayType>
auto wrap_array(mixed_engine const&, ArrayType arr[], size_t num_entries){
    return wrap_array(arr, num_entries);
}

/*!
 * \ingroup HALAWAXEXT
 * \brief Sparse matrix associated with the mixed-engine.
 *
 * The matrix is constructed from vectors on the CPU, the vectors are loaded
 * in hala::gpu_vector containers and the loaded data is used to construct
 * a sparse matrix bound on the GPU.
 * The struct owns the CUDA data but does not keep references to the CPU data.
 */
template<typename T>
struct mixed_sparse_matrix{
    //! \brief The value type of the matrix.
    using value_type = std::remove_cv_t<T>;
    //! \brief Defines the associated engine type.
    using engine_type = mixed_engine;

    //! \brief Constructor from either vectors or raw-arrays, all dimensions are specified explicitly.
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    mixed_sparse_matrix(mixed_engine const &engine, int num_rows, int num_cols, int num_nz,
                        VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals)
    : rpntr(engine.gpu().load(pntr, num_rows+1)), rindx(engine.gpu().load(indx, num_nz)), rvals(engine.gpu().load(vals, num_nz)),
      mat(make_sparse_matrix(engine.gpu(), num_rows, num_cols, num_nz, rpntr, rindx, rvals))
    {}
    //! \brief Constructor that infers dimensions from the sizes of the vectors.
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    mixed_sparse_matrix(mixed_engine const &engine, int num_cols,
                        VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals)
    : rpntr(engine.gpu().load(pntr)), rindx(engine.gpu().load(indx)), rvals(engine.gpu().load(vals)),
      mat(make_sparse_matrix(engine.gpu(), num_cols, rpntr, rindx, rvals))
    {}
    //! \brief Using default destructor.
    ~mixed_sparse_matrix() = default;

    //! \brief Cannot copy.
    mixed_sparse_matrix(mixed_sparse_matrix const&) = delete;
    //! \brief Can move.
    mixed_sparse_matrix(mixed_sparse_matrix &&) = default;

    //! \brief Cannot copy.
    mixed_sparse_matrix& operator = (mixed_sparse_matrix const&) = delete;
    //! \brief Can move.
    mixed_sparse_matrix& operator = (mixed_sparse_matrix &&) = default;

    //! \brief Returns the workspace (in bytes) requires on the GPU device.
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
    size_t gemv_buffer_size(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y) const{
        auto gx = mat.engine().load(x);
        auto gy = mat.engine().load(y);
        return mat.gemv_buffer_size(trans, alpha, x, beta, y);
    }
    //! \brief Matrix vector product, see hala::sparse_gemv().
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY>
    void gemv(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y){
        check_types(rvals, x, y);
        auto gx = mat.engine().load(x);
        auto gy = gpu_bind_vector(mat.engine(), y);
        mat.gemv(trans, alpha, gx, beta, gy);
    }
    //! \brief Matrix vector product with external workspace buffer, see hala::sparse_gemv().
    template<typename FPa, class VectorLikeX, typename FPb, class VectorLikeY, class VectorLikeBuff>
    void gemv(char trans, FPa alpha, VectorLikeX const &x, FPb beta, VectorLikeY &&y, VectorLikeBuff &&work){
        check_types(rvals, x, y);
        auto gx = mat.engine().load(x);
        auto gy = gpu_bind_vector(mat.engine(), y);
        mat.gemv(trans, alpha, gx, beta, gy, work);
    }
    //! \brief Returns the workspace (in bytes) requires on the GPU device.
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
    size_t gemm_buffer_size(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &&C, int ldc){
        auto gB = mat.engine().load(B);
        auto gC = mat.engine().load(C);
        return mat.gemm_buffer_size(transa, transb, b_rows, b_cols, alpha, gB, ldb, beta, gC, ldc);
    }
    //! \brief Matrix-matrix product with external workspace buffer, see hala::sparse_gemm().
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC, class VectorLikeT>
    void gemm(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &&C, int ldc, VectorLikeT &&work){
        check_types(rvals, B, C);
        auto gB = mat.engine().load(B);
        auto gC = gpu_bind_vector(mat.engine(), C);
        mat.gemm(transa, transb, b_rows, b_cols, alpha, gB, ldb, beta, gC, ldc, work);
    }
    //! \brief Matrix-matrix product, see hala::sparse_gemm().
    template<typename FSA, class VectorLikeB, typename FSB, class VectorLikeC>
    void gemm(char transa, char transb, int b_rows, int b_cols, FSA alpha, VectorLikeB const &B, int ldb, FSB beta, VectorLikeC &&C, int ldc){
        check_types(rvals, B, C);
        auto gB = mat.engine().load(B);
        auto gC = gpu_bind_vector(mat.engine(), C);
        mat.gemm(transa, transb, b_rows, b_cols, alpha, gB, ldb, beta, gC, ldc);
    }

    //! \brief The GPU pointer array.
    gpu_vector<int> rpntr;
    //! \brief The GPU index array.
    gpu_vector<int> rindx;
    //! \brief The GPU values array.
    gpu_vector<value_type> rvals;
    //! \brief The GPU matrix.
    gpu_sparse_matrix<value_type> mat;
};

/*!
 * \ingroup HALAWAXEXT
 * \brief Factory method to construct a mixed sparse matrix.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_sparse_matrix(mixed_engine const &engine, int num_rows, int num_cols, int num_nz,
                        VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    check_types(vals);
    check_types_int(pntr, indx);
    using scalar_type = get_scalar_type<VectorLikeV>;
    return mixed_sparse_matrix<scalar_type>(engine, num_rows, num_cols, num_nz, pntr, indx, vals);
}

/*!
 * \ingroup HALAWAXEXT
 * \brief Factory method to construct a mixed sparse matrix (infers sizes from the vector sizes).
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_sparse_matrix(mixed_engine const &engine, int num_cols, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    check_types(vals);
    check_types_int(pntr, indx);
    assert( get_size(indx) == get_size(vals) );
    using scalar_type = get_scalar_type<VectorLikeV>;
    return mixed_sparse_matrix<scalar_type>(engine, num_cols, pntr, indx, vals);
}

/*!
 * \ingroup HALAWAXEXT
 * \brief Triangular matrix associated with the mixed-engine.
 *
 * The matrix is constructed from vectors on the CPU, the vectors are loaded
 * in hala::gpu_vector containers and the loaded data is used to construct
 * a triangular matrix bound on the GPU.
 * The struct owns the CUDA data but does not keep references to the CPU data.
 */
template<typename T>
struct mixed_tiangular_matrix{
    //! \brief The value type of the matrix.
    using value_type = std::remove_cv_t<T>;
    //! \brief Defines the associated engine type.
    using engine_type = mixed_engine;

    //! \brief Constructor, prefer using the hala::make_triangular_matrix() factory method.
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    mixed_tiangular_matrix(mixed_engine const &eng, char uplo, char diag,
                           VectorLikeP const &pn, VectorLikeI const &in, VectorLikeV const &va, char policy)
        : gpu_pntr(eng.gpu().load(pn)), gpu_indx(eng.gpu().load(in)), gpu_vals(eng.gpu().load(va)),
          tri(eng.gpu(), uplo, diag, gpu_pntr, gpu_indx, gpu_vals, policy){}

    //! \brief The matrix is cannot be copied.
    mixed_tiangular_matrix(mixed_tiangular_matrix const &other) = delete;
    //! \brief The matrix is cannot be copied.
    mixed_tiangular_matrix& operator = (mixed_tiangular_matrix const &other) = delete;

    //! \brief The matrix is trivially movable.
    mixed_tiangular_matrix& operator = (mixed_tiangular_matrix &&other) = default;
    //! \brief The matrix is trivially movable.
    mixed_tiangular_matrix(mixed_tiangular_matrix &&other) = default;

    //! \brief Get the CUDA triangular matrix.
    gpu_triangular_matrix<T> const& gpu_tri() const{ return tri; }

    //! \brief Holds the loaded gpu data for the pntr (the cuda matrix alias this vector).
    gpu_vector<int> gpu_pntr;
    //! \brief Holds the loaded gpu data for the indx (the cuda matrix alias this vector).
    gpu_vector<int> gpu_indx;
    //! \brief Holds the loaded gpu data for the vals (the cuda matrix alias this vector).
    gpu_vector<T> gpu_vals;

    //! \brief The CUDA triangular matrix.
    gpu_triangular_matrix<T> tri;
};

/*!
 * \ingroup HALAWAXEXT
 * \brief Factory method to create a triangular matrix associated with the mixed-engine.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_triangular_matrix(mixed_engine const &engine, char uplo, char diag,
                            VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, char policy = 'N'){
    check_types(vals);
    check_types_int(pntr, indx);
    assert( get_size(indx) == get_size(vals) );
    using scalar_type = get_scalar_type<VectorLikeV>;
    return mixed_tiangular_matrix<scalar_type>(engine, uplo, diag, pntr, indx, vals, policy);
}

/*!
 * \ingroup HALAWAXEXT
 * \brief Mixed implementation of the incomplete Lower-Upper factorization.
 *
 * The main difference between the hala::cpu_ilu and this class is that the
 * hala::mixed_ilu will hold (own) copies of the pattern vectors moved to the GPU memory.
 * The apply() method works with vectors in the CPU memory space, but there is
 * an internal instance of hala::cuda_ilu that can be accessed though the
 *
 */
template<typename T> struct mixed_ilu{
    //! \brief Floating point type used by the values.
    using value_type = typename define_standard_type<T>::value_type;

    //! \brief The type of the associated engine.
    using engine_type = mixed_engine;

    //! \brief Constructor, performs the lower-upper factorization.
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    mixed_ilu(mixed_engine const &eng, VectorLikeP const &mpntr, VectorLikeI const &mindx, VectorLikeV const &mvals, char policy)
        : rengine(std::ref(eng)), num_rows(get_size_int(mpntr) - 1),
        gpu_pntr(eng.gpu().load(mpntr)), gpu_indx(eng.gpu().load(mindx)),
        gilu(eng.gpu(), gpu_pntr, gpu_indx, eng.gpu().load(mvals), policy){}

    //! \brief Destructor, nothing to do in the cpu case.
    ~mixed_ilu() = default;

    //! \brief The class is non-copyable.
    mixed_ilu(mixed_ilu<T> const &other) = delete;
    //! \brief The class is non-copyable.
    gpu_ilu<T>& operator = (gpu_ilu<T> const &other) = delete;

    //! \brief The class is trivially movable.
    mixed_ilu<T>& operator = (mixed_ilu<T> &&other) = default;
    //! \brief The class is trivially movable.
    mixed_ilu(mixed_ilu<T> &&other) = default;

    //! \brief Applies the preconditioner, the vectors holding data in CPU memory.
    template<class VectorLikeX, class VectorLikeR>
    void apply(VectorLikeX const &x, VectorLikeR &&r, int num_rhs = 1) const{
        check_types(x, r);
        check_set_size(assume_output, r, num_rhs, num_rows);

        auto gx = engine().gpu().load(x);
        auto gr = new_vector(engine().gpu(), r);
        gilu.apply(gx, gr, num_rhs);
        gr.unload(r);
    }

    //! \brief Returns the associated engine reference.
    mixed_engine const& engine() const{ return rengine.get(); }

    //! \brief Returns a reference to the internal hala::gpu_ilu.
    gpu_ilu<value_type> const& cilu() const{ return gilu; }

private:
    std::reference_wrapper<mixed_engine const> rengine;

    int num_rows;
    gpu_vector<int> gpu_pntr, gpu_indx;

    gpu_ilu<value_type> gilu;
};

/*!
 * \ingroup HALAWAXEXT
 * \brief Creates an ILU factorization using the mixed_engine.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_ilu(mixed_engine const &cengine, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, char policy = 'N'){
    using standard_type = get_standard_type<VectorLikeV>;
    return mixed_ilu<standard_type>(cengine, pntr, indx, vals, policy);
}

#else

#ifndef __HALA_DOXYGEN_SKIP
using mixed_engine = void;
#endif

#endif


}

#endif
