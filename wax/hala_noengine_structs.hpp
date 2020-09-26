#ifndef __HALA_NOENGINE_STRUCTS_HPP
#define __HALA_NOENGINE_STRUCTS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_sparse_extensions.hpp"

namespace hala{

/*!
 * \internal
 * \file hala_noengine_structs.hpp
 * \ingroup HALAWAXENGINED
 * \brief Wrappers that use engined vectors and bypass carrying the engine to each call.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 *
 * \endinternal
 */

/*!
 * \ingroup HALAWAXEXT
 * \addtogroup HALAWAXENGINED Engined vectors
 *
 * Writing a generalized template while carrying the engine variable to each call
 * can get repetitive and tedious. HALA offers an engined-vector overload that
 * binds an engine and a vector using either owning or non-owning references so that the binded pair
 * can be used in hala calls without the need to explicitly carry the engine object.
 */

/*!
 * \internal
 * \ingroup HALAWAXENGINED
 * \brief If the class \b a is one of the engines, the struct defaults to std::true_type and std::false_type otherwise.
 *
 * \endinternal
 */
template<class a> struct is_engine{
    //! \brief Set to true or false if the engine type is one of the compatible engine types.
    #ifdef HALA_ENABLE_GPU
    static const bool value = std::is_same<a, cpu_engine>::value || std::is_same<a, gpu_engine>::value || std::is_same<a, mixed_engine>::value;
    #else
    static const bool value = std::is_same<a, cpu_engine>::value;
    #endif
};

/*!
 * \internal
 * \ingroup HALAWAXENGINED
 * \brief Defines the types of the vector and the internal variable of the hala::engined_vector.
 *
 * Allows for switching between owning and non-owning modes with specialization to this struct.
 * The owning mode (when \b vec is not a reference) this will assume both the internal variable
 * and the returned variables as the vector itself, i.e., this is an identity template.
 * In non-owning mode (when \b vec is a reference) this will use a reference wrapper for
 * the internal variable and will strip away the reference for the return types.
 * \endinternal
 */
template<class vec, typename = void> struct engined_vector_binding{
    //! \brief The type of the internal variable used in the engined_vector class.
    using variable = vec;
    //! \brief The type of the returned variable used in the engined_vector class.
    using type = vec;
};

#ifndef __HALA_DOXYGEN_SKIP
template<class vec> struct engined_vector_binding<vec, std::enable_if_t<std::is_reference<vec>::value>>{
    using variable = std::reference_wrapper<std::remove_reference_t<vec>>;
    using type = std::remove_reference_t<vec>;
};
#endif

/*!
 * \internal
 * \ingroup HALAWAXENGINED
 * \brief Allows the distinction between a vector and reference-wrapper to a vector, either way returns a reference to the vector.
 *
 * \endinternal
 */
template<class vec> vec& get_vec(vec &x){ return x; }
/*!
 * \internal
 * \ingroup HALAWAXENGINED
 * \brief Allows the distinction between a vector and reference-wrapper to a vector, either way returns a reference to the vector.
 *
 * \endinternal
 */
template<class vec> vec& get_vec(std::reference_wrapper<vec> &x){ return x.get(); }

/*!
 * \ingroup HALAWAXENGINED
 * \brief A struct that binds an engine and a vector so both can be carried in one context.
 *
 * The purpose of the construct is to create a new entity that carries both a vector and
 * an engine in one context. That way long algorithms do not have to constantly repeat
 * the engine variable into each call to HALA. Instead the engine will be bounded to
 * all input/output vectors and be carried out through the computations.
 *
 * Do not call the constructor directly, use the factory methods hala::bind_engine_vector
 * and hala::new_vector.
 *
 * \internal
 * The owning vs. non-owning mode is determined by the type \b vec, which must be either
 * a vector like (owning) or reference to a vector-like (non-owning).
 * The factory methods take care of the distinction depending whether calling
 * hala::bind_engine_vector or hala::new_vector.
 * \endinternal
 *
 * For example:
 * \code
 *  template<class VectorLikeA, class VectorLikeB, , class VectorLikeX>
 *  void advanced_solver(VectorLikeA const &A, VectorLikeB const &b, VectorLikeX &x){
 *      auto t = hala::new_vector(x);
 *      hala::vcopy(x, t);
 *      hala::axpy(-1.0, b, x);
 *      // ...
 *      // imagine many more HALA calls used to solve Ax = b
 *      // ...
 *  }
 * \endcode
 * The advanced solver works by default on the CPU, but what if we want to perform the solve
 * on the GPU. If we use the engine mechanics, we would have to add another variable to
 * the template indicating the engine type and then pass the variable to every HALA call:
 * \code
 *      ...
 *      auto t = hala::new_vector(engine, x);
 *      hala::vcopy(engine, x, t);
 *      ...
 * \endcode
 * If the advanced solver has many lines of code, the above will change a lot of code
 * and can easily lead to an error if one of the calls misses the engine.
 * In contrast, the hala::engined_vector allows to instantiate the solver with the GPU backend
 * and without making any changes to the solver code:
 * \code
 *  template<class cengine, class VectorLikeA, class VectorLikeB, , class VectorLikeX>
 *  void advanced_solver(cengine const&engine, VectorLikeA const &A, VectorLikeB const &b, VectorLikeX &x){
 *      auto bindA = hala::bind_engine_vector(engine, A);
 *      auto bindb = hala::bind_engine_vector(engine, b);
 *      auto bindx = hala::bind_engine_vector(engine, x);
 *      advanced_solver(bindA, bindb, bindx);
 * }
 * \endcode
 * The above small piece of code will overload the advanced solver and allow for calls with any engine type
 * while making no changes to the solver itself.
 */
template<class cengine, class vec>
struct engined_vector{
    //! \brief The type of the vector used in the binding, removes any reference qualifies but leaves const.
    using vtype = typename engined_vector_binding<vec>::type;
    //! \brief Defines the scalar type of the underlying vector, mostly used for the precision of the stopping criteria.
    using value_type = get_standard_type<vtype>;

    //! \brief Reference to the engine, it is recommended to use the engine() method in place of direct access.
    std::reference_wrapper<cengine const> e;
    //! \brief Either owning a vector variable or non-owning reference to a vector variable.
    typename engined_vector_binding<vec>::variable v;

    //! \brief Initialize the engined_vector with non-owning references to the engine and \b x.
    engined_vector(cengine const &engine, vtype &x) : e(engine), v(x){
        static_assert(is_engine<cpu_engine>::value, "engined_vector constructed with cengine that is not an engine");
        static_assert(std::is_reference<vec>::value, "engined_vector cannot take a reference in non-referenced mode, instantiate with vec& instead");
    }
    //! \brief Initialize the engined_vector with ownership to an internal vector.
    engined_vector(cengine const &engine) : e(engine), v(new_vector(engine, v)){
        static_assert(is_engine<cpu_engine>::value, "engined_vector constructed with cengine that is not an engine");
        static_assert(!std::is_reference<vec>::value, "engined_vector in non-owning mode must be initialized with the vector (use the other constructor)");
    }

    /*!
     * \brief Returns \b true if the other engined_vector has a compatible engine.
     *
     * Engines are compatible if
     * -# they have the same type, e.g., both cpu_engine
     * -# if using hala::gpu_engine or hala::mixed_engine, then the device ids of the engines should also match.
     *
     * Overloads are provided to handle all possible pairs of engine types.
     */
    template<class oengine, class vec2>
    bool engine_match(engined_vector<oengine, vec2> const &) const{ return false; }

    #ifndef __HALA_DOXYGEN_SKIP
    template<class vec2, class ce = cengine>
    std::enable_if_t<std::is_same<ce, cpu_engine>::value, bool>
    engine_match(engined_vector<cengine, vec2> const &) const{
        return true;
    }

    #if defined(HALA_ENABLE_CUDA) || defined(HALA_ENABLE_ROCM)
    template<class vec2, class ce = cengine>
    std::enable_if_t<std::is_same<ce, gpu_engine>::value, bool>
    engine_match(engined_vector<cengine, vec2> const &other) const{
        return (engine().device() == other.engine().device());
    }
    template<class vec2, class ce = cengine>
    std::enable_if_t<std::is_same<ce, mixed_engine>::value, bool>
    engine_match(engined_vector<cengine, vec2> const &other) const{
        return (engine().gpu().device() == other.engine().gpu().device());
    }
    #endif
    #endif

    //! \brief Return a reference to the engine.
    cengine const& engine() const{ return e.get(); }
    //! \brief Return a const reference to the vector.
    vtype const& vector() const{ return get_vec(v); }
    //! \brief Return a reference to the vector.
    vtype& vector(){ return get_vec(v); }

    //! \brief Returns the size of the vector.
    size_t size() const{ return get_size(vector()); }
    //! \brief Resizes the vector.
    void resize(size_t new_size){ set_size(new_size, vector()); }
};

/*!
 * \internal
 * \ingroup HALAWAXENGINED
 * \brief Runs a sanity check whether the classes have the same engine.
 *
 * Checks the engine type and the GPU device id.
 * There is a variadric overload to check multiple vectors.
 * \endinternal
 */
template<class ev1, class ev2>
bool check_engines(ev1 const &x, ev2 const &y){ return x.engine_match(y); }

#ifndef __HALA_DOXYGEN_SKIP
template<class ev1, class ev2, class ev3, class... evs>
bool check_engines(ev1 const &x, ev2 const &y, ev3 const &z, evs const&... more){
    if (!check_engines(x, y)) return false;
    return check_engines(y, z, more...);
}
#endif

/*!
 * \ingroup HALAWAXENGINED
 * \brief Creates a bind between the engine and the vector.
 *
 * Factory method to construct a bind between the vector and the engine.
 *
 * Overloads:
 * \code
 * auto x1 = bind_engine_vector(engine, x);
 * auto y1 = bind_engine_vector(engined_vector, y);
 * auto z1 = bind_engine_vector(vector, z);
 * \endcode
 * Overloads are provided for all engine types (case x1), the engine reference can be taken from an existing
 * hala::engined_vector, or the call can be made with a generic vector-like class in which case
 * z1 will be just a reference to z for calls that do not use binded engines.
 */
template<class cengine, class vec>
std::enable_if_t<is_engine<cengine>::value, engined_vector<cengine, vec&>> bind_engine_vector(cengine const &e, vec &x){
    return engined_vector<cengine, vec&>(e, x);
}

/*!
 * \ingroup HALAWAXENGINED
 * \brief Returns the engine associated with the vector.
 *
 * The method is an easy way to obtain the engine from the engined vector,
 * and the method is overloaded for general vector-like classes to return an instance of the cpu_engine.
 *
 * \b Note: do not use this to differentiate between GPU and CPU vectors,
 * i.e., this will return hala::cpu_engine even if \b vec is hala::cuda_vector.
 * The intended use is within the no-engine context where
 * an engined_vector and a CPU vectors are the only acceptable inputs.
 */
template<class cengine, class vec>
auto get_engine(engined_vector<cengine, vec> const &x){ return x.engine(); }

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)

template<class vec>
auto get_engine(vec const &){ return cpu_engine(); }

template<class cengine, class vec>
auto new_vector(engined_vector<cengine, vec> const &x){
    return engined_vector<cengine, get_vdefault<cengine, typename engined_vector<cengine, vec>::vtype>>(x.engine());
}

template<class cengine, class vec>
void set_zero(size_t num_entries, engined_vector<cengine, vec> &x){
    set_zero(x.engine(), num_entries, x.vector());
}
template<class cengine, class vec>
void set_size(size_t num_entries, engined_vector<cengine, vec> &x){
    set_size(num_entries, x.vector());
}

template<class vec>
auto new_vector(vec const &x){
    return new_vector(cpu_engine(), x);
}

template<class cengine, class vec>
std::enable_if_t<is_engine<cengine>::value, engined_vector<cengine, vec const&>> bind_engine_vector(cengine const &e, vec const &x){
    return engined_vector<cengine, vec const&>(e, x);
}

template<class cvec, class vec> std::enable_if_t<!is_engine<cvec>::value, vec&> bind_engine_vector(cvec const &, vec &x){ return x; }
template<class cvec, class vec> std::enable_if_t<!is_engine<cvec>::value, vec const&> bind_engine_vector(cvec const &, vec const &x){ return x; }

template<class cengine, class cvec, class vec>
auto bind_engine_vector(engined_vector<cengine, cvec> const &s, vec &x){ return bind_engine_vector(s.engine(), x); }

#endif

}

#include "hala_noengine_overloads.hpp"
#undef __HALA_NOENGINE_OVERLOADS_HPP
#include "hala_noengine_overloads.hpp"

#endif
