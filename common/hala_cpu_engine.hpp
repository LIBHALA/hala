#ifndef __HALA_CPU_ENGINE_HPP
#define __HALA_CPU_ENGINE_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_wrap_array.hpp"

/*!
 * \internal
 * \file hala_cpu_engine.hpp
 * \brief Defines a core engine that indicates a CPU-only algorithm.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACORE
 *
 * Contains the CPU engine class which facilitates generality, e.g., when switching between BLAS and CUDA.
 * \endinternal
*/

/*!
 * \ingroup HALACORE
 * \addtogroup HALACPUENGINE The CPU Engine Class
 *
 * Define the CPU engine, see hala::cpu_engine
 */

namespace hala{

/*!
 * \internal
 * \ingroup HALACPUENGINE
 * \brief Defines the direction of CPU-GPU memory copy.
 *
 * \endinternal
 */
enum class copy_direction{
    //! \brief Copy from host to device.
    host2device,
    //! \brief Copy from device to device.
    device2device,
    //! \brief Copy from device to host.
    device2host
};

/*!
 * \ingroup HALACPUENGINE
 * \brief The CPU Engine
 *
 * \par HALA Compute Engines
 * The CPU and GPU linear algebra libraries often provide the same set of algorithms (e.g., the original BLAS),
 * but require handles, device ID or other simple meta-data.
 * In order to facilitate portability, HALA offers overloads to the common methods that accept one
 * of a set of compute engines, e.g., the hala::cuda_engine holds the cuBlas and cuSparse handles
 * while the hala::cpu_engine is an empty struct.
 *
 * Example, generic method to compute the distance between two vectors:
 * \code
 *  template<class compute_engine, class vecx, class vecy>
 *  auto distance(compute_engine &engine, vecx const &x, vecy const &y){
 *      auto temp = get_vector(engine, x);
 *      hala::vcopy(engine, y, temp);
 *      hala::axpy(engine, -1, x, temp);
 *      return hala::norm2(engine, temp);
 *  }
 *
 *  template<class vecx, class vecy>
 *  auto distance(vecx const &x, vecy const &y){
 *      return distance(hala::cpu_engine(), x, y);
 *  }
 * \endcode
 * The \b distance template now can be used for vectors both on the CUDA device and CPU:
 * \code
 *  hala::cuda_engine engine;
 *  hala::cuda_vector<T> x, y;
 *  // initialize x, y
 *  auto d = distance(engine, x, y);
 * \endcode
 * Or on the CPU side:
 * \code
 *  hala::cpu_engine engine;
 *  std::vector<T> x, y;
 *  // initialize x, y
 *  auto d = distance(engine, x, y);
 *  // or simply call distance(x, y);
 * \endcode
 *
 * \par Default Internal Vector
 * Several of the solver algorithms require temporary vectors to store intermediate states,
 * therefore HALA needs a default vector class for such internal needs.
 * The default vectors are also used in user-provided call-backs, e.g.,
 * to Krylov preconditioning methods, and thus the default can be overwritten by the user.
 * The default internal vectors are the standard vector std::vector and the hala::cuda_vector
 * for the cpu and CUDA engines respectively.
 *
 * \par Overwriting the Default
 * In addition to the standard definitions in \ref HALACUSTOM, the default vector must
 * satisfy these additional criteria:
 * -# the vector must be trivially movable, will also work with trivially copiable but
 *    there may be a performance penalty
 * -# the vector must be resizable, i.e., see hala::set_size() and hala::define_resize
 * -# the vector must have either a default empty constructor or a specialization to
 *    hala::vector_constructor
 */
struct cpu_engine{
    //! \brief Creates a new engine, mimics specifying a device ID.
    cpu_engine(int = 0){}
    //! \brief Mimics returning the active GPU device ID.
    int device() const{ return 0; }
    //! \brief Mimics setting the active GPU device.
    void set_active_device(){}
    //! \brief Wraps an array; see hala::wrapped_array()
    template<typename T>
    auto wrap_array(T *array, size_t num_entries){ return wrap_array(array, num_entries); }
    //! \brief Default vector is std::vector
    template<typename T> using dfvector = typename std::vector<T>;
    //! \brief Creates a copy of the vector using the default vector.
    template<typename VectorLike> auto vcopy(VectorLike const &x) const;
};

/*!
 * \ingroup HALACPUENGINE
 * \brief Specialization of this struct are used to define the default vector classes.
 *
 * The struct simply assumes the default types defined in the engine classes,
 * but can be specialized to change the default.
 */
template<class engine, typename scalar_type> struct define_vdefault{
    //! \brief Takes the default vector from the engine.
    using vector = typename engine::template dfvector<scalar_type>;
};


/*!
 * \internal
 * \ingroup HALACPUENGINE
 * \brief Helper alias to the default vector type that matches the \b VectorLike for the given \b engine.
 *
 * \endinternal
 */
template<class engine, class VectorLike>
using get_vdefault = typename define_vdefault<engine, typename define_type<VectorLike>::value_type>::vector;

/*!
 * \ingroup HALACPUENGINE
 * \brief Assumes the vector has a default constructor \b VectorLike(), otherwise must specialize.
 *
 * Must be specialized for each default vector that does not have a \b VectorLike() constructor.
 */
template<class VectorLike>
struct vector_constructor{
    /*!
     * \brief Optionally, the vector could use information from the compute engine, e.g., device ID.
     */
    template<class engine>
    static auto make_one(engine &){ return VectorLike(); }
};

/*!
 * \ingroup HALACPUENGINE
 * \brief Returns a default vector with the default constructor.
 *
 * Uses both hala::define_vdefault and hala::vector_constructor to make a default (probably) empty vector.
 * The call will instantiate to return an vector for the appropriate engine and the value type
 * will be copied from the provided vector-like.
 * Note that the data is not copied, the input vector is used to infer only the type.
 *
 * Overloads:
 * \code
 *  auto x = new_vector(engine, t);
 *  auto y = new_vector(t); // t is ordinary vector-like class
 *  auto z = new_vector(t); // t is hala::engined_vector
 * \endcode
 * Overloads are provided for all engine types, for hala::engined_vectors and for vector only inputs.
 * In the examples above, \b x will have the default vector type for the engine, see hala::define_vdefault.
 * The \b y vector will have the default vector type associated with the hala::cpu_engine.
 * The \b z will have the default type associated with the engine of \b t and \b z will alias
 * the engine of \b t.
 */
template<class VectorLike>
auto new_vector(cpu_engine const &engine, VectorLike const &){
    return vector_constructor<get_vdefault<cpu_engine, VectorLike>>::make_one(engine);
}

/*!
 * \ingroup HALACPUENGINE
 * \brief Fills the first \b num_entries of a vector with zeros.
 *
 * Used to set the initial iterate of solver algorithms.
 *
 * Overloads:
 * \code
 *  set_zero(num_entries, x);
 *  set_zero(engine, num_entries, x);
 * \endcode
 * The overloads include all compute engines and the engined_vector class.
 */
template<class VectorLike>
inline void set_zero(size_t num_entries, VectorLike &&x){
    check_set_size(assume_output, x, num_entries);
    std::fill_n(get_standard_data(x), num_entries, get_cast<get_standard_type<VectorLike>>(0.0));
}

#ifndef __HALA_DOXYGEN_SKIP
template<class VectorLike>
inline void set_zero(cpu_engine const&, size_t num_entries, VectorLike &&x){
    set_zero(num_entries, x);
}
#endif

/*!
 * \ingroup HALACPUENGINE
 * \brief Create an array wrapper for the corresponding engine.
 *
 * Creates a wrapper around an array so that the wrapper "knows" the array size.
 * This makes it safer to call HALA methods directly with raw-arrays.
 *
 * Overloads:
 * \code
 *  wrap_array(arr, num_entries);
 *  wrap_array(engine, arr, num_entries);
 * \endcode
 * Variants of this method are provided for each engine class resulting in appropriate corresponding wrappers.
 * If called without an engine, the method assumes the cpu_engine.
 */
template<typename ArrayType>
auto wrap_array(cpu_engine const&, ArrayType arr[], size_t num_entries){
    return wrap_array(arr, num_entries);
}


}

#endif
