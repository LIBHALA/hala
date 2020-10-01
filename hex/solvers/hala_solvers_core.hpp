#ifndef __HALA_SOLVERS_CORE_HPP
#define __HALA_SOLVERS_CORE_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */
#include "hala_wrapped_extensions.hpp"

/*!
 * \internal
 * \file hala_solvers_core.hpp
 * \brief HALA solver core.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALASOLVERS
 *
 * Core and helpers for HALA solvers module.
 * \endinternal
 */

/*!
 * \defgroup HALASOLVERS HALA Solver Algorithms
 *
 * HALA provides several (mostly iterative) solver algorithms that build upon
 * the sparse, BLAS and CUDA modules.
 */

namespace hala{

/*!
 * \ingroup HALASOLVERS
 * \brief Struct that describes the convergence criteria of an iterative solver.
 *
 * Two conditions are used to determine whether an iterative solver should terminate
 * or continue: tolerance and total number of iterations. The tolerance is the threshold
 * at which the solution is considered sufficient (usually when the norm of the residual
 * drops below the tolerance). The limit on the total number of iteration guards against
 * stagnation.
 *
 * The struct helps specify tolerance and maximum number of iterations in a convenient
 * way to that either one or both can be specified without creating a large number of
 * method overloads.
 *
 * - \b precision is the precision of the scalar type, e.g., float for floats and std::complex<float>
 */
template<typename precision>
struct stop_criteria{
    //! \brief Specifies the maximum number of iterations.
    int max_iter;
    //! \brief Specifies the tolerance.
    precision tol;
    //! \brief Initializer with just the max number of iterations, tolerance is zero (good for testing).
    stop_criteria(int max_iterations) :
            max_iter(max_iterations), tol(0.0){}
    //! \brief Initializer with just tolerance, the maximum number of iterations is the maximum of the int.
    template<typename U>
    stop_criteria(U const &tolerance) :
            max_iter(std::numeric_limits<int>::max()),
            tol(get_cast<precision>(tolerance)){}
    //! \brief Specify both tolerance and number of iterations.
    template<typename U>
    stop_criteria(U const &tolerance, int max_iterations) :
            max_iter(max_iterations),
            tol(get_cast<precision>(tolerance)){}
};

/*!
 * \internal
 * \ingroup HALASOLVERS
 * \brief Preconditioner type, uses two default vectors.
 *
 * \endinternal
 */
template<class engine, class VectorLike>
struct define_preconditioner{
    //! \brief The type.
    using type = std::function<void(
                        get_vdefault<engine, VectorLike> const &x,
                        get_vdefault<engine, VectorLike> &r
                                   )>;
};

#ifdef HALA_ENABLE_GPU
/*!
 * \internal
 * \ingroup HALASOLVERS
 * \brief Partial specialization for the mixed-engine type so it uses the same type as the cuda-engine.
 *
 * \endinternal
 */
template<class VectorLike>
struct define_preconditioner<mixed_engine, VectorLike>{
    //! \brief The type.
    using type = typename define_preconditioner<gpu_engine, VectorLike>::type;
};
#endif

/*!
 * \ingroup HALASOLVERS
 * \brief Type that describes the preconditioner for an iterative solver, engine case.
 *
 * The preconditioners are cheap to compute approximate inverses that can significantly
 * accelerate the convergence of an iterative solver.
 * The preconditioners used in HALA are lambda objects with signature that takes
 * two internal vectors (based on the compute engine) and return \f$ r = P^{-1} x \f$
 * where \b P is the approximate inverse.
 * See the documentation of hala::cpu_engine for the internal vector type and how
 * it can be modified.
 * The \b VectorLike class will be used only to infer the scalar type for the default
 * vectors.
 *
 * General signature:
 * \code
 *  preconditioner = std::function<void(vector_type const &intput, vector_type &result)>;
 * \endcode
 * In the CPU engine case, the \b vector_type is the default CPU vector. However,
 * both the hala::gpu_engine and hala::mixed_engine use the default GPU vector type,
 * since the mixed solver is implemented to move all data onto the GPU once and then
 * retrieve the result at the very end (avoiding data movement in the mean time).
 */
template<class engine, class VectorLike>
using preconditioner = typename define_preconditioner<engine, VectorLike>::type;

/*!
 * \ingroup HALASOLVERS
 * \brief Type that describes the preconditioner for an iterative solver, no-engine case.
 *
 * The no-engine preconditioner is very similar to the hala::preconditioner,
 * the difference is that the vectors have the hala::engined_vector type
 * matching engine and scalar type of \b VectorLike and the using the default
 * vector type.
 */
template<class VectorLike>
using preconditioner_noe = std::function<void(
                        decltype( new_vector( std::declval<VectorLike>() ) ) const &x,
                        decltype( new_vector( std::declval<VectorLike>() ) ) &r
                                   )>;

/*!
 * \internal
 * \ingroup HALASOLVERS
 * \brief Enables a template if engine is a valid engine but not the mixed_engine.
 *
 * This template serves multiple purposes:
 * - constrains templates that must be instantiated with a compute engine, avoiding
 *  the ambiguity of converting an engine to a vector like class
 * - provides a short name better suited for the doxygen page formats
 * - differentiates the mixed-engine overloads, where the mixed-engine mode
 *  works better to push date onto the GPU once in the beginning of an algorithm
 *  then do all operations in GPU mode and pull back the data in the very end
 *
 * \endinternal
 */
#ifdef HALA_ENABLE_CUDA
template<class compute_engine>
using hala_int = typename std::enable_if_t<std::is_same<compute_engine, cpu_engine>::value || std::is_same<compute_engine, gpu_engine>::value, int>;
#else
template<class compute_engine>
using hala_int = int;
#endif

}

#endif
