#ifndef __HALA_CHOLMOD_WRAPPERS_HPP
#define __HALA_CHOLMOD_WRAPPERS_HPP
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
 * \file hala_cholmod.hpp
 * \brief HALA wrapper for Cholmod functions
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACHOLMOD
 *
 * Contains the HALA-Cholmod wrapper template.
 */

/*!
 * \defgroup HALACHOLMOD HALA Cholmod Wrapper Templates
 *
 * Cholmod wrapper templates allow easy integration between
 * C++ containers and the direct sparse solver capabilities of
 * the cholmod library.
 */

#define HALA_ENABLE_CHOLMOD

#include "hala_cholmod_defines.hpp"

namespace hala{
/*!
 * \ingroup HALACHOLMOD
 * \brief Wrapper around a factorization of a sparse matrix using the Cholmod library.
 *
 * Handle the background memory management, but unfortunately requires that data is copied unnecessarily.
 * A copy of the sparse matrix is needed for factorization and the copy is preserved (in the non-spd matrix).
 * A copy of each right-hand-side vector is also needed to solve any type of system.
 * Nevertheless, the performance of the Cholmod library compensates for this relatively small overhead.
 */
template<typename T = double> // Cholmod only works with double precision and complex numbers
class cholmod_factor{
public:
    //! \brief Default constructor, makes an empty factor.
    cholmod_factor() : num_rows(0), chol_sparse(nullptr), chol_factor(nullptr), spd(false){}
    //! \brief Construct a factor and factorize either symmetric-positive-definite or general matrix.
    template<typename VectorLikeP, typename VectorLikeI, typename VectorLikeV>
    cholmod_factor(bool is_spd, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals)
        : cholmod_factor(){
        static_assert(std::is_same<get_scalar_type<VectorLikeV>, double>::value, "Cholmod works only with double-precision real numbers.");
        check_types_int(pntr, indx);
        num_rows = get_size_int(pntr) - 1;
        assert(num_rows > 0); // there is at least 1 row

        chol_common.resize(3 * 1024); // guess the size of the common block
        cholmod_start(convert(chol_common));

        if (is_spd){
            factorise_spd(pntr, indx, vals);
        }else{
            factorise_gen(pntr, indx, vals);
        }
    }
    //! \brief Default destructor.
    ~cholmod_factor(){
        if (chol_sparse != nullptr){
            cholmod_free_sparse(pconvert(&chol_sparse), convert(chol_common));
            chol_sparse = nullptr;
        }
        if (chol_factor != nullptr){
            cholmod_free_factor(pconvert(&chol_factor), convert(chol_common));
            chol_factor = nullptr;
        }
        cholmod_finish(convert(chol_common));
    }

    //! \brief Class cholmod_factor cannot be copied.
    cholmod_factor(cholmod_factor<T> const &) = delete;
    //! \brief Class cholmod_factor cannot be copied.
    cholmod_factor<T>& operator = (cholmod_factor<T> const &) = delete;

    //! \brief Move constructor.
    cholmod_factor(cholmod_factor<T> &&) = default;
    //! \brief Move assignment.
    cholmod_factor<T>& operator = (cholmod_factor<T> &&) = default;

    //! \brief Solve \f$ A x = b \f$ using the existing factorization, must be preceded by a call to **factoriseSPD()** or **factoriseGeneral()**
    template<typename VectorLikeB, typename VectorLikeX>
    void solve(VectorLikeB const &b, VectorLikeX &x){
        check_size(b, num_rows);
        check_set_size(assume_output, x, num_rows);
        check_types(x, b);
        static_assert(std::is_same<get_scalar_type<VectorLikeX>, T>::value, "cholmod_factor<T>::solve() can be called only with vectors with scalar type T");

        auto chol_b = make_cholmod_dense(b);

        if (spd){
            make_cholmod_dense( cholmod_solve(solve_all, pconvert(chol_factor), chol_b, convert(chol_common)) ).put(x);
        }else{
            T alpha[2] = {1.0, 0.0}, beta[2] = {0.0, 0.0};
            auto chol_z = make_cholmod_dense(num_rows);

            cholmod_sdmult(pconvert(chol_sparse), non_tansp, pconvert(alpha), pconvert(beta), chol_b, chol_z, convert(chol_common));

            make_cholmod_dense( cholmod_solve(solve_all, pconvert(chol_factor), chol_z, convert(chol_common)) ).put(x);
        }
    }

    //! \brief Cholmod definition of real (i.e., double).
    static constexpr int real = 1;
    //! \brief Cholmod definition of symmetric.
    static constexpr int symmetric = -1;
    //! \brief Cholmod definition of general.
    static constexpr int general   =  0;
    //! \brief Cholmod definition of solve all parts of the factorization.
    static constexpr int solve_all   = 0;
    //! \brief Cholmod definition of multiply by non-transpose of a matrix.
    static constexpr int non_tansp   = 0;

protected:
    /*!
     * \internal
     * \brief RAII management of Cholmod dense vectors.
     *
     * \endinternal
     */
    class chol_dense_vector{
    public:
        //! \brief Make empty vector of size \b nrows.
        chol_dense_vector(int nrows, std::vector<char>&h)
            : handler(h), _data(cholmod_allocate_dense(nrows, 1, nrows, real, convert(handler))){}
        //! \brief Copy \b x into a cholmod dense vector.
        template<class VectorLike>
        chol_dense_vector(VectorLike const &x, std::vector<char>&h)
            : chol_dense_vector(get_size_int(x), h){
            std::copy_n(get_data(x), get_size(x), cheat_cholmod::get_dense_x<T>(_data));
        }
        //! \brief Assume control of a cholmod dense vector (when returned from a solve call).
        chol_dense_vector(void *data, std::vector<char> &h) : handler(h), _data(data){}
        //! \brief Destrucor, deletes all memory.
        ~chol_dense_vector(){ cholmod_free_dense(&_data, convert(handler)); }

        //! \brief Copy the cholmod dense vector into \b x, note that \b x \b must \b have \b correct \b size.
        template<class VectorLike>
        void put(VectorLike &x){
            std::copy_n(cheat_cholmod::get_dense_x<T>(_data), get_size(x), get_data(x));
        }
        //! \brief Handle to easily pass to functions.
        template<typename U> operator U* (){ return reinterpret_cast<U*>(_data); }
    private:
        std::vector<char> &handler;
        void *_data;
    };

    //! \brief Creates a new dense vector and passes the chol_common.
    auto make_cholmod_dense(int nrows){ return chol_dense_vector(nrows, chol_common); }
    //! \brief Assumes control of the dense vector and passes the chol_common.
    auto make_cholmod_dense(void *x){ return chol_dense_vector(x, chol_common); }
    //! \brief Copies \b x into the dense vector and passes the chol_common.
    template<class VectorLike>
    auto make_cholmod_dense(VectorLike const &x){ return chol_dense_vector(x, chol_common); }

    //! \brief Copies the sparse matrix into the \b chol_sparse data structure.
    template<int mat_type, typename VectorLikeP, typename VectorLikeI, typename VectorLikeV>
    void set_sparse(VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
        check_types_int(pntr, indx);
        int num_nnz = pntr[num_rows];

        chol_sparse = (void*) cholmod_allocate_sparse(num_rows, num_rows, num_nnz, 1, 1, mat_type, real, convert(chol_common)); // last 1 is CHOLMOD_REAL

        std::copy_n(get_data(pntr), get_size(pntr), cheat_cholmod::get_sparse_pntr(chol_sparse));
        std::copy_n(get_data(indx), num_nnz,        cheat_cholmod::get_sparse_indx(chol_sparse));
        std::copy_n(get_data(vals), num_nnz,        cheat_cholmod::get_sparse_vals<T>(chol_sparse));
    }

    //! \brief Factorize a symmetric-positive-define matrix.
    template<typename VectorLikeP, typename VectorLikeI, typename VectorLikeV>
    void factorise_spd(VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){

        std::vector<int> upper_pntr, lower_pntr, upper_indx, lower_indx;
        std::vector<T> upper_vals, lower_vals;

        split_matrix('U', 'N', pntr, indx, vals, upper_pntr, upper_indx, upper_vals, lower_pntr, lower_indx, lower_vals);

        set_sparse<symmetric>(upper_pntr, upper_indx, upper_vals);

        chol_factor = cholmod_analyze(pconvert(chol_sparse), convert(chol_common));
        cholmod_factorize(pconvert(chol_sparse), pconvert(chol_factor), convert(chol_common));

        cholmod_free_sparse(pconvert(&chol_sparse), convert(chol_common));
        chol_sparse = nullptr;

        spd = true;
    }

    //! \brief Factorize a general matrix.
    template<typename VectorLikeP, typename VectorLikeI, typename VectorLikeV>
    void factorise_gen(VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){

        set_sparse<general>(pntr, indx, vals);

        chol_factor = cholmod_analyze(pconvert(chol_sparse), convert(chol_common));
        cholmod_factorize(pconvert(chol_sparse), pconvert(chol_factor), convert(chol_common));

        spd = false;
    }

private:
    int num_rows;
    void *chol_sparse, *chol_factor;
    std::vector<char> chol_common;
    bool spd;
};

/*!
 * \ingroup HALACHOLMOD
 * \brief Factorize the symmetric positive definite sparse matrix and return appropriate \b hala::cholmod_factor.
 *
 */
template<typename VectorLikeP, typename VectorLikeI, typename VectorLikeV>
auto cholmod_factorize_spd(VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    return cholmod_factor<get_scalar_type<VectorLikeV>>(true, pntr, indx, vals);
}

/*!
 * \ingroup HALACHOLMOD
 * \brief Factorize the general sparse matrix and return appropriate \b hala::cholmod_factor.
 *
 */
template<typename VectorLikeP, typename VectorLikeI, typename VectorLikeV>
auto cholmod_factorize_gen(VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals){
    return cholmod_factor<get_scalar_type<VectorLikeV>>(false, pntr, indx, vals);
}

}

#endif
