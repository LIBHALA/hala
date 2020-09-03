#ifndef __HALA_GPU_SPARSE_ILU_HPP
#define __HALA_GPU_SPARSE_ILU_HPP
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
 * \file hala_cuda_ilu.hpp
 * \brief HALA wrappers to cuSparse methods.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACUDASPARSEILU
 *
 * \endinternal
*/

#ifdef HALA_ENABLE_CUDA
#include "hala_cuda_sparse_triangular.hpp"
#endif

#ifdef HALA_ENABLE_ROCM
#include "hala_rocm_sparse_triangular.hpp"
#endif

namespace hala{

#ifdef HALA_ENABLE_CUDA

/*!
 * \ingroup HALACUDA
 * \addtogroup HALACUDASPARSEILU cuSparse ILU
 *
 * Wrappers to cuSparse methods for incomplete Lower-Upper factorization.
 */

/*!
 * \ingroup HALACUDASPARSEILU
 * \brief CUDA implementation of the incomplete Lower-Upper factorization.
 */
template<typename T> struct gpu_ilu{
    //! \brief Floating point type used by the values.
    using value_type = typename define_standard_type<T>::value_type;

    //! \brief The type of the associated engine.
    using engine_type = gpu_engine;

    //! \brief Constructor, performs the lower-upper factorization.
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    gpu_ilu(gpu_engine const &cengine, VectorLikeP const &mpntr, VectorLikeI const &mindx, VectorLikeV const &mvals, char policy)
        : rengine(std::ref(cengine)), ilu(rengine.get().device()),
        num_rows(get_size_int(mpntr)-1), nnz(get_size_int(mindx)){

        cusparseSolvePolicy_t rpolicy = ((policy == 'L') || (policy == 'l')) ? CUSPARSE_SOLVE_POLICY_USE_LEVEL : CUSPARSE_SOLVE_POLICY_NO_LEVEL;

        auto descM = make_cuda_mat_description('G', 'S', 'S');

        hala::vcopy(engine(), mvals, ilu);

        int buff_size = 0;
        cuda_call_backend<value_type>(cusparseScsrilu02_bufferSize, cusparseDcsrilu02_bufferSize, cusparseCcsrilu02_bufferSize, cusparseZcsrilu02_bufferSize,
                          "cuSparse::Xcsrilu02_bufferSize()", engine(), num_rows, nnz, descM, convert(ilu), convert(mpntr), convert(mindx), infoM, &buff_size);

        auto f77_buffer = make_gpu_vector<int>(buff_size / sizeof(int), engine().device());

        cuda_call_backend<value_type>(cusparseScsrilu02_analysis, cusparseDcsrilu02_analysis, cusparseCcsrilu02_analysis, cusparseZcsrilu02_analysis,
                          "cuSparse::Xcsrilu02_analysis()", engine(), num_rows, nnz, descM,
                          convert(ilu), convert(mpntr), convert(mindx), infoM, rpolicy, convert(f77_buffer));

        cuda_call_backend<value_type>(cusparseScsrilu02, cusparseDcsrilu02, cusparseCcsrilu02, cusparseZcsrilu02,
                          "cuSparse::Xcsrilu02()", engine(), num_rows, nnz, descM,
                          convert(ilu), convert(mpntr), convert(mindx), infoM, rpolicy, convert(f77_buffer));

        upper = std::make_unique<gpu_triangular_matrix<value_type>>(cengine, 'U', 'N', mpntr, mindx, ilu, policy);
        lower = std::make_unique<gpu_triangular_matrix<value_type>>(cengine, 'L', 'U', mpntr, mindx, ilu, policy);
    }

    //! \brief Destructor, nothing to do in the cpu case.
    ~gpu_ilu() = default;

    //! \brief The class is non-copyable.
    gpu_ilu(gpu_ilu const &other) = delete;
    //! \brief The class is non-copyable.
    gpu_ilu& operator = (gpu_ilu const &other) = delete;

    //! \brief The class is trivially movable.
    gpu_ilu& operator = (gpu_ilu &&other) = default;
    //! \brief The class is trivially movable.
    gpu_ilu(gpu_ilu &&other) = default;

    /*!
     * \brief Applies the preconditioner to the vector x and returns the result in r.
     *
     * The preconditioner can be applied either to a single vector \b x
     * or to a set of vectors in a batch command.
     * The \b num_rhs indicates the number of right-hand-sides.
     */
    template<class VectorLikeX, class VectorLikeR>
    void apply(VectorLikeX const &x, VectorLikeR &&r, int num_rhs = 1) const{
        check_types(x, r);
        check_set_size(assume_output, r, num_rhs, num_rows);
        gpu_pntr<host_pntr> hold(engine());

        if (num_rhs == 1){
            auto tmp = new_vector(engine(), r);
            sparse_trsv('N', *lower.get(), 1.0, x, tmp);
            sparse_trsv('N', *upper.get(), 1.0, tmp, r);
        }else{
            vcopy(engine(), x, r);
            sparse_trsm('N', 'N', num_rhs, *lower.get(), 1.0, r);
            sparse_trsm('N', 'N', num_rhs, *upper.get(), 1.0, r);
        }
    }

    //! \brief Returns the associated engine reference.
    gpu_engine const& engine() const{ return rengine.get(); }

private:
    std::reference_wrapper<gpu_engine const> rengine;

    cuda_struct_description<csrilu02Info_t> infoM;
    cuda_struct_description<csrsv2Info_t> infoL, infoU;

    gpu_vector<value_type> ilu;

    int num_rows, nnz;

    std::unique_ptr<gpu_triangular_matrix<value_type>> upper, lower;
};

#endif

#ifdef HALA_ENABLE_ROCM
/*!
 * \ingroup HALACUDA
 * \addtogroup HALACUDASPARSEILU cuSparse ILU
 *
 * Wrappers to cuSparse methods for incomplete Lower-Upper factorization.
 */

/*!
 * \ingroup HALACUDASPARSEILU
 * \brief CUDA implementation of the incomplete Lower-Upper factorization.
 */
template<typename T> struct gpu_ilu{
    //! \brief Floating point type used by the values.
    using value_type = typename define_standard_type<T>::value_type;

    //! \brief The type of the associated engine.
    using engine_type = gpu_engine;

    //! \brief Constructor, performs the lower-upper factorization.
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    gpu_ilu(gpu_engine const &cengine, VectorLikeP const &mpntr, VectorLikeI const &mindx, VectorLikeV const &mvals, char policy)
        : rengine(std::ref(cengine))
        //, ilu(rengine.get().device()),
//         num_rows(get_size_int(mpntr)-1), nnz(get_size_int(mindx))
        {

//         cusparseSolvePolicy_t rpolicy = ((policy == 'L') || (policy == 'l')) ? CUSPARSE_SOLVE_POLICY_USE_LEVEL : CUSPARSE_SOLVE_POLICY_NO_LEVEL;
//
//         auto descM = make_cuda_mat_description('G', 'S', 'S');
//
//         hala::vcopy(engine(), mvals, ilu);
//
//         int buff_size = 0;
//         cuda_call_backend<value_type>(cusparseScsrilu02_bufferSize, cusparseDcsrilu02_bufferSize, cusparseCcsrilu02_bufferSize, cusparseZcsrilu02_bufferSize,
//                           "cuSparse::Xcsrilu02_bufferSize()", engine(), num_rows, nnz, descM, convert(ilu), convert(mpntr), convert(mindx), infoM, &buff_size);
//
//         auto f77_buffer = make_gpu_vector<int>(buff_size / sizeof(int), engine().device());
//
//         cuda_call_backend<value_type>(cusparseScsrilu02_analysis, cusparseDcsrilu02_analysis, cusparseCcsrilu02_analysis, cusparseZcsrilu02_analysis,
//                           "cuSparse::Xcsrilu02_analysis()", engine(), num_rows, nnz, descM,
//                           convert(ilu), convert(mpntr), convert(mindx), infoM, rpolicy, convert(f77_buffer));
//
//         cuda_call_backend<value_type>(cusparseScsrilu02, cusparseDcsrilu02, cusparseCcsrilu02, cusparseZcsrilu02,
//                           "cuSparse::Xcsrilu02()", engine(), num_rows, nnz, descM,
//                           convert(ilu), convert(mpntr), convert(mindx), infoM, rpolicy, convert(f77_buffer));
//
//         upper = std::make_unique<gpu_triangular_matrix<value_type>>(cengine, 'U', 'N', mpntr, mindx, ilu, policy);
//         lower = std::make_unique<gpu_triangular_matrix<value_type>>(cengine, 'L', 'U', mpntr, mindx, ilu, policy);
    }

    //! \brief Destructor, nothing to do in the cpu case.
    ~gpu_ilu() = default;

    //! \brief The class is non-copyable.
    gpu_ilu(gpu_ilu const &other) = delete;
    //! \brief The class is non-copyable.
    gpu_ilu& operator = (gpu_ilu const &other) = delete;

    //! \brief The class is trivially movable.
    gpu_ilu& operator = (gpu_ilu &&other) = default;
    //! \brief The class is trivially movable.
    gpu_ilu(gpu_ilu &&other) = default;

    /*!
     * \brief Applies the preconditioner to the vector x and returns the result in r.
     *
     * The preconditioner can be applied either to a single vector \b x
     * or to a set of vectors in a batch command.
     * The \b num_rhs indicates the number of right-hand-sides.
     */
    template<class VectorLikeX, class VectorLikeR>
    void apply(VectorLikeX const &x, VectorLikeR &&r, int num_rhs = 1) const{
//         check_types(x, r);
//         check_set_size(assume_output, r, num_rhs, num_rows);
//         gpu_pntr<host_pntr> hold(engine());
//
//         if (num_rhs == 1){
//             auto tmp = new_vector(engine(), r);
//             sparse_trsv('N', *lower.get(), 1.0, x, tmp);
//             sparse_trsv('N', *upper.get(), 1.0, tmp, r);
//         }else{
//             vcopy(engine(), x, r);
//             sparse_trsm('N', 'N', num_rhs, *lower.get(), 1.0, r);
//             sparse_trsm('N', 'N', num_rhs, *upper.get(), 1.0, r);
//         }
    }

    //! \brief Returns the associated engine reference.
    gpu_engine const& engine() const{ return rengine.get(); }

private:
    std::reference_wrapper<gpu_engine const> rengine;

//     cuda_struct_description<csrilu02Info_t> infoM;
//     cuda_struct_description<csrsv2Info_t> infoL, infoU;
//
//     gpu_vector<value_type> ilu;
//
//     int num_rows, nnz;
//
//     std::unique_ptr<gpu_triangular_matrix<value_type>> upper, lower;
};
#endif

/*!
 * \ingroup HALACUDASPARSEILU
 * \brief Creates an ILU factorization using the cuda_engine.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_ilu(gpu_engine const &cengine, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, char policy){
    using standard_type = get_standard_type<VectorLikeV>;
    return gpu_ilu<standard_type>(cengine, pntr, indx, vals, policy);
}

}

#endif
