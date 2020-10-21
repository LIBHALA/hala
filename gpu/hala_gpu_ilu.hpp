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

/*!
 * \ingroup HALAGPU
 * \addtogroup HALAGPUSPARSEILU Sparse incomplete Lower-Upper (ILU) factorization
 *
 * Wrappers to GPU methods for incomplete Lower-Upper factorization.
 */

/*!
 * \ingroup HALACUDASPARSEILU
 * \brief GPU implementation of the incomplete Lower-Upper factorization.
 */
template<typename T> struct gpu_ilu{
    //! \brief Floating point type used by the values.
    using value_type = typename define_standard_type<T>::value_type;

    //! \brief The type of the associated engine.
    using engine_type = gpu_engine;

    //! \brief Constructor, performs the lower-upper factorization.
    template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
    gpu_ilu(gpu_engine const &cengine, VectorLikeP const &mpntr, VectorLikeI const &mindx, VectorLikeV const &mvals, char policy)
        : rengine(cengine), ilu(rengine.device()),
        num_rows(get_size_int(mpntr)-1), nnz(get_size_int(mindx)){

        #ifdef HALA_ENABLE_CUDA
        cusparseSolvePolicy_t rpolicy = ((policy == 'L') || (policy == 'l')) ? CUSPARSE_SOLVE_POLICY_USE_LEVEL : CUSPARSE_SOLVE_POLICY_NO_LEVEL;

        auto descM = make_cuda_mat_description('G', 'S', 'S');

        hala::vcopy(rengine, mvals, ilu);

        int buff_size = 0;
        cuda_call_backend<value_type>(cusparseScsrilu02_bufferSize, cusparseDcsrilu02_bufferSize, cusparseCcsrilu02_bufferSize, cusparseZcsrilu02_bufferSize,
                          "cuSparse::Xcsrilu02_bufferSize()", rengine, num_rows, nnz, descM, convert(ilu), convert(mpntr), convert(mindx), infoM, &buff_size);

        auto work_buffer = make_gpu_vector<int>(buff_size / sizeof(int), rengine.device());

        cuda_call_backend<value_type>(cusparseScsrilu02_analysis, cusparseDcsrilu02_analysis, cusparseCcsrilu02_analysis, cusparseZcsrilu02_analysis,
                          "cuSparse::Xcsrilu02_analysis()", rengine, num_rows, nnz, descM,
                          convert(ilu), convert(mpntr), convert(mindx), infoM, rpolicy, convert(work_buffer));

        cuda_call_backend<value_type>(cusparseScsrilu02, cusparseDcsrilu02, cusparseCcsrilu02, cusparseZcsrilu02,
                          "cuSparse::Xcsrilu02()", rengine, num_rows, nnz, descM,
                          convert(ilu), convert(mpntr), convert(mindx), infoM, rpolicy, convert(work_buffer));
        #endif

        #ifdef HALA_ENABLE_ROCM
        auto mdesc = make_rocsparse_general_description('N');

        hala::vcopy(rengine, mvals, ilu);

        size_t buffer_size = 0;
        rocm_call_backend<value_type>(rocsparse_scsrilu0_buffer_size, rocsparse_dcsrilu0_buffer_size,
                                      rocsparse_ccsrilu0_buffer_size, rocsparse_zcsrilu0_buffer_size,
                                      "csric0_buffer_size()",
                                      rengine, num_rows, nnz, mdesc, convert(ilu), convert(mpntr), convert(mindx), info, &buffer_size);

        auto work_buffer = make_gpu_vector<int>(buffer_size / sizeof(int), rengine.device());

        rocm_call_backend<value_type>(rocsparse_scsrilu0_analysis, rocsparse_dcsrilu0_analysis,
                                      rocsparse_ccsrilu0_analysis, rocsparse_zcsrilu0_analysis,
                                      "csric0_analysis()",
                                      rengine, num_rows, nnz, mdesc, convert(ilu), convert(mpntr), convert(mindx),
                                      info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, convert(work_buffer));

        rocm_call_backend<value_type>(rocsparse_scsrilu0, rocsparse_dcsrilu0, rocsparse_ccsrilu0, rocsparse_zcsrilu0,
                                      "csric0()",
                                      rengine, num_rows, nnz, mdesc, convert(ilu), convert(mpntr), convert(mindx),
                                      info, rocsparse_solve_policy_auto, convert(work_buffer));
        #endif

        upper = std::make_unique<gpu_triangular_matrix<value_type>>(rengine, 'U', 'N', mpntr, mindx, ilu, policy);
        lower = std::make_unique<gpu_triangular_matrix<value_type>>(rengine, 'L', 'U', mpntr, mindx, ilu, policy);
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

    //! \brief Returns the buffer size used in the solve stage.
    template<class VectorLikeX, class VectorLikeR>
    size_t buffer_size(VectorLikeX const &x, VectorLikeR &&r, int num_rhs) const{
        gpu_pntr<host_pntr> hold(engine());
        if (num_rhs == 1){
            auto tmp = new_vector(engine(), r);
            return std::max( lower->trsv_buffer_size('N', 1.0, x, tmp), upper->trsv_buffer_size('N', 1.0, x, tmp) );
        }else{
            return std::max( lower->trsm_buffer_size('N', 'N', num_rhs, 1.0, x), upper->trsm_buffer_size('N', 'N', num_rhs, 1.0, x) );
        }
    }

    //! \brief Returns a container with size matching gpu_ilu::buffer_size.
    template<class VectorLikeX, class VectorLikeR>
    gpu_vector<value_type> get_temp_buffer(VectorLikeX const &x, VectorLikeR &&r, int num_rhs) const{
        return gpu_vector<value_type>(buffer_size(x, r, num_rhs), rengine.device());
    }

    /*!
     * \brief Applies the preconditioner to the vector x and returns the result in r.
     *
     * The preconditioner can be applied either to a single vector \b x
     * or to a set of vectors in a batch command.
     * The \b num_rhs indicates the number of right-hand-sides.
     */
    template<class VectorLikeX, class VectorLikeR, class VectorLikeT>
    void apply(VectorLikeX const &x, VectorLikeR &&r, int num_rhs, VectorLikeT &&temp_buff) const{
        check_types(x, r);
        check_set_size(assume_output, r, num_rhs, num_rows);
        gpu_pntr<host_pntr> hold(engine());

        if (num_rhs == 1){
            auto tmp = new_vector(engine(), r);
            lower->trsv('N', 1.0, x, tmp, temp_buff);
            upper->trsv('N', 1.0, tmp, r, temp_buff);
        }else{
            vcopy(engine(), x, r);
            lower->trsm('N', 'N', num_rhs, 1.0, r, num_rows, temp_buff);
            upper->trsm('N', 'N', num_rhs, 1.0, r, num_rows, temp_buff);
        }
    }

    /*!
     * \brief Applies the preconditioner to the vector x and returns the result in r.
     */
    template<class VectorLikeX, class VectorLikeR>
    void apply(VectorLikeX const &x, VectorLikeR &&r, int num_rhs = 1) const{
        apply(x, r, num_rhs, get_temp_buffer(x, r, num_rhs));
    }

    //! \brief Returns the associated engine reference.
    gpu_engine const& engine() const{ return rengine; }

private:
    gpu_engine rengine;

    #ifdef HALA_ENABLE_CUDA
    cuda_struct_description<csrilu02Info_t> infoM;
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocm_struct_description<rocsparse_mat_info> info;
    #endif

    gpu_vector<value_type> ilu;

    int num_rows, nnz;

    std::unique_ptr<gpu_triangular_matrix<value_type>> upper, lower;
};

/*!
 * \ingroup HALAGPUSPARSEILU
 * \brief Creates an ILU factorization using the gpu_engine.
 */
template<class VectorLikeP, class VectorLikeI, class VectorLikeV>
auto make_ilu(gpu_engine const &cengine, VectorLikeP const &pntr, VectorLikeI const &indx, VectorLikeV const &vals, char policy){
    using standard_type = get_standard_type<VectorLikeV>;
    return gpu_ilu<standard_type>(cengine, pntr, indx, vals, policy);
}

}

#endif
