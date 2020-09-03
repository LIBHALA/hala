#ifndef __HALA_BINDPACK_HPP
#define __HALA_BINDPACK_HPP
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
 * \file hala_bindpack.hpp
 * \brief HALA wrappers to bind a \b hala::mmpack to a specific memory address.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAVEX
 *
 * The hala::bind_pack struct offers RAII style of memory management for a hala::mmpack.
 * \endinternal
 */

#include "hala_mmpack.hpp"

namespace hala{

/*!
 * \ingroup HALAVEX
 * \brief Binds a memory slice to an address, constructor reads from the address, destructor writes back.
 *
 * Provides RAII style of management for aligned memory and an memory address.
 * The alignment is stored as a template parameter to avoid the extra runtime variable; however,
 * this causes potential conflicts with the constructor. To avoid redundancy in syntax, use
 * the factory methods hala::mmbind().
 *
 * The class wrapps around an hala::mmpack instance and provides overloads for the standard arithmetic operators,
 * the mmpack can be accessed directly under the name \b packed_data.
 */
template<typename T, regtype R = default_regtype, typename alignment = default_alignment<T, R>, std::enable_if_t<!std::is_const<T>::value>* = nullptr>
struct bind_pack{
    //! \brief Bind the memory to the \b address using the default alignment.
    bind_pack(T *address) : packed_data(address), binded_address(address){
        static_assert(std::is_same<alignment, default_alignment<T, R>>::value, "this constructor requires alignment == default_alignment");
    }
    //! \brief Bind the memory to the \b address using good alignment.
    bind_pack(T *address, define_aligned) : packed_data(address, aligned), binded_address(address){
        static_assert(std::is_same<alignment, define_aligned>::value, "this constructor requires alignment == define_aligned");
    }
    //! \brief Bind the memory to the \b address using good alignment.
    bind_pack(T *address, define_unaligned) : packed_data(address, unaligned), binded_address(address){
        static_assert(std::is_same<alignment, define_unaligned>::value, "this constructor requires alignment == define_unaligned");
    }
    //! \brief Destructor, sync back with the \b address.
    ~bind_pack(){ sync(); }

    //! \brief Sync back with the \b address, in case periodic read/write is needed.
    void sync(){
        if __HALA_CONSTEXPR_IF__ (std::is_same<alignment, default_alignment<T, R>>::value){
            packed_data.put(binded_address);
        }else if __HALA_CONSTEXPR_IF__ (std::is_same<alignment, define_aligned>::value){
            packed_data.put(binded_address, aligned);
        }else if __HALA_CONSTEXPR_IF__ (std::is_same<alignment, define_unaligned>::value){
            packed_data.put(binded_address, unaligned);
        }else{
            runtime_assert(false, "bind_pack with unknown alignment type, how was this even constructed?"); // should never happen
        }
    }

    //! \brief Delete default constructor, must bind to memory address.
    bind_pack() = delete;
    //! \brief Delete copy constructor, should not bind memory twice.
    bind_pack(bind_pack<T, R, alignment, nullptr> const &) = delete;
    //! \brief Delete copy assignment, should not bind memory twice.
    bind_pack<T, R, alignment, nullptr>& operator = (bind_pack<T, R, alignment, nullptr> const &) = delete;
    //! \brief Move constructor.
    bind_pack(bind_pack<T, R, alignment, nullptr> &&other) : packed_data(other.packed_data), binded_address(other.binded_address){}
    //! \brief Move operator.
    bind_pack<T, R, alignment, nullptr>& operator = (bind_pack<T, R, alignment, nullptr> &&other){
        packed_data = other.packed_data;
        binded_address = other.binded_address;
    };

    //! \brief Automatically return the internal data.
    operator mmpack<T, R>& (){ return packed_data; }
    //! \brief Automatically return the internal data.
    operator mmpack<T, R> const& () const{ return packed_data; }

    //! \brief Addition overload.
    inline mmpack<T, R> operator + (mmpack<T, R> const &other) const{ return packed_data + other; }
    //! \brief Subtraction overload.
    inline mmpack<T, R> operator - (mmpack<T, R> const &other) const{ return packed_data - other; }
    //! \brief Multiplication overload, also handles complex numbers.
    inline mmpack<T, R> operator * (mmpack<T, R> const &other) const{ return packed_data * other; }
    //! \brief Division overload, also handles complex numbers.
    inline mmpack<T, R> operator / (mmpack<T, R> const &other) const{ return packed_data / other; }
    //! \brief Increment overload.
    inline mmpack<T, R> operator += (mmpack<T, R> const &other){
        return packed_data += other;
    }
    //! \brief Decrement overload.
    inline mmpack<T, R> operator -= (mmpack<T, R> const &other){
        return packed_data -= other;
    }
    //! \brief Scale overload.
    inline mmpack<T, R> operator *= (mmpack<T, R> const &other){
        return packed_data *= other;
    }
    //! \brief Scale-divide overload.
    inline mmpack<T, R> operator /= (mmpack<T, R> const &other){
        return packed_data /= other;
    }
    //! \brief Fused multiply add (+= a * b), if the instruction is available.
    inline void fmadd(mmpack<T, R> const &a, mmpack<T, R> const &b){
        packed_data.fmadd(a, b);
    }

    //! \brief Return the square-root of the data.
    inline mmpack<T, R> sqrt() const{ return packed_data.sqrt(); }
    //! \brief Set the data to the square-root.
    inline void set_sqrt(){ packed_data.set_sqrt(); }

    //! \brief Return the square-root of the data.
    inline mmpack<T, R> abs() const{ return packed_data.abs(); }
    //! \brief Set the data to the square-root.
    inline mmpack<T, R> abs2(){ return packed_data.abs2(); }

    //! \brief Returns reference to the \b i-th element.
    inline T& operator [] (size_t i){ return packed_data[i]; }
    //! \brief Returns const-reference to the \b i-th element
    inline T const& operator [] (size_t i) const{ return packed_data[i]; }

    //! \brief The packed numbers.
    mmpack<T, R> packed_data;
    //! \brief The bind address to write back to in the destructor.
    T* const binded_address;
};

/*!
 * \ingroup HALAVEX
 * \brief Bind the address to a hala::bind_pack, multiple overloads are provided.
 *
 * Creates a binding between a hala::mmpack and the specified address.
 * Overloads are provided to handle different alignment modes and aligned iterators,
 * but neither the \b address nor the iterator can be const since the
 * hala::bind_pack destructor will write to the address.
 *
 * Overloads:
 * \code
 *  T *pntr = ...;                // pointer
 *  auto iter = hala::mmbegin(x); // non-const aligned iterator
 *
 *  auto mmb = hala::mmbind(pntr);            // use default alignment
 *  auto mmb = hala::mmbind(pntr, aligned);   // assume aligned read/write
 *  auto mmb = hala::mmbind(pntr, unaligned); // assume unaligned read/write
 *  auto mmb = hala::mmbind(iter);            // read/write from/to hala::aligned_iterator
 * \endcode
 * In all cases, \b mmb will be an instance of hala::bind_pack with the specified type and register type.
 */
template<regtype R = default_regtype, typename T>
inline bind_pack<T, R> mmbind(T *address){
    static_assert(!std::is_const<T>::value, "cannot bind to a const address");
    return bind_pack<T, R>(address);
}

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)
template<regtype R = default_regtype, typename T>
inline bind_pack<T, R, define_aligned> mmbind(T *address, define_aligned){
    static_assert(!std::is_const<T>::value, "cannot bind to a const address");
    return bind_pack<T, R, define_aligned>(address, aligned);
}

template<regtype R = default_regtype, typename T>
inline bind_pack<T, R, define_unaligned> mmbind(T *address, define_unaligned){
    static_assert(!std::is_const<T>::value, "cannot bind to a const address");
    return bind_pack<T, R, define_unaligned>(address, unaligned);
}

template<regtype R = default_regtype, typename T>
inline bind_pack<T, R, define_aligned> mmbind(aligned_iterator<T, R, aligned_vector<T, R>> &i){ return bind_pack<T, R, define_aligned>(&*i, aligned); }
#endif

}

#endif
