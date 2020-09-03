#ifndef __HALA_ALLALOCATOR_HPP
#define __HALA_ALLALOCATOR_HPP
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
 * \file hala_aligned_allocator.hpp
 * \brief HALA custom allocator that aligns memory for extended register use.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAALALLOC
 *
 * Contains the definition of the custom allocator.
 * \endinternal
 */

#include <stdlib.h>
#include "hala_intrinsics_macros.hpp"

namespace hala{

/*!
 * \ingroup HALAVEX
 * \addtogroup HALAALALLOC Aligned Vector
 *
 * \par Allocators
 * The C++ allocators provide an abstraction to memory management used by most C++ containers,
 * specifically the std::vector container that is most relevant to HALA.
 * The default allocator will align the vectors to a bit that is decided by the compiler
 * and sometimes controlled with compiler flags.
 * HALA provides a custom allocator that guarantees alignment for a desired register type.
 * The allocator is implemented with a single wrapper around the default allocator
 * and is compatible only with the std::vector container.
 */

/*!
 * \ingroup HALAALALLOC
 * \brief Allocator that aligns memory to the selected register.
 *
 * Very simple allocator that uses the C11 \b aligned_alloc to make an aligned memory segment.
 * The method part of the C (not C++) 2011 standard and required the inclusion of stdlib.h header.
 */
template <class T, regtype R = default_regtype>
class aligned_allocator{
public:
    //! \brief Type of the allocated entries.
    using value_type = T;
    //! \brief Type of the pointers.
    using pointer = T*;
    //! \brief Type of the const pointers.
    using const_pointer = T const*;
    //! \brief Type of the index size.
    using size_type = std::size_t;

    //! \brief Do not propagate on copy, simply create a new allocator.
    using propagate_on_container_copy_assignment = std::false_type;
    //! \brief Do not propagate the allocator on container move, there is no state.
    using propagate_on_container_move_assignment = std::false_type;
    //! \brief Do not propagate the allocator on container swap, there is no state.
    using propagate_on_container_swap            = std::false_type;

    //! \brief The beginning of an aligned block with within a \b stride of any array, matches \b hala::vex:mem_slice::stride.
    static constexpr size_t extra_elements = vex_stride<T, R>::stride;
    //! \brief Number of bytes to use for the alignment.
    static constexpr size_t alignment_byte = extra_elements * sizeof(T);

    //! \brief Default constructor, do not use the offsets.
    aligned_allocator(){
        static_assert(is_float<T>::value    || is_double<T>::value ||
                      is_fcomplex<T>::value || is_dcomplex<T>::value,
                      "Aligned vectors work only with floating point numbers, float, double, std::complex<float> or std::complex<double> and without cv qualifiers");
    }

    //! \brief Returns an aligned pointer to an allocated block.
    pointer allocate(size_type n, const_pointer = 0){
        return reinterpret_cast<pointer>(aligned_alloc(extra_elements * sizeof(T), (n + extra_elements - n % extra_elements) * sizeof(T)));
    }

    //! \brief Deletes the entire block associated with \b p.
    void deallocate(pointer p, size_type){ free(p); }

    //! \brief Shrink the number of template parameters.
    template<class U> struct rebind{
        //! \brief Defines the reduced parameter template parameter alias, per standard requirements.
        using other = aligned_allocator<U, R>;
    };

    //! \brief Do not propagate on copy, this allocator cannot deallocate memory from another allocator.
    aligned_allocator<T, R> select_on_container_copy_construction() const{
        return aligned_allocator<T, R>();
    }

    //! \brief Allocators for regtype none have no effective state, all other allocators are different.
    constexpr bool operator == (aligned_allocator<T, R> const &) noexcept{
        return (R == hala::regtype::none); // the none allocator has no real state
    }

    //! \brief Allocators for regtype none have no effective state, all other allocators are different.
    constexpr bool operator != (aligned_allocator<T, R> const &) noexcept{
        return (R != hala::regtype::none); // the none allocator has no real state
    }
};

/*!
 * \ingroup HALAALALLOC
 * \brief Aligned vector alias.
 */
template<typename T, regtype R = default_regtype>
using aligned_vector = std::vector<T, aligned_allocator<T, R>>;


/*!
 * \internal
 * \ingroup HALAALALLOC
 * \brief Defines whether the aligned vector should work with const or non-const iterators (non-const case).
 *
 * \endinternal
 */
template<typename T, regtype R, typename = void> struct aligned_iterator_traits{
    //! \brief Defines the non-const iterator.
    using iterator = typename aligned_vector<T, R>::iterator;
    //! \brief Defines the non-const reference.
    using reference = typename aligned_vector<T, R>::reference;
};
/*!
 * \internal
 * \ingroup HALAALALLOC
 * \brief Defines whether the aligned vector should work with const or regular iterators (const case).
 *
 * \endinternal
 */
template<typename T, regtype R> struct aligned_iterator_traits<T, R, aligned_vector<T, R> const>{
    //! \brief Defines the const iterator.
    using iterator = typename aligned_vector<T, R>::const_iterator;
    //! \brief Defines the const reference.
    using reference = typename aligned_vector<T, R>::const_reference;
};

/*!
 * \ingroup HALAALALLOC
 * \brief Aligned random iterator that always moves in strides of the corresponding hala::mmpack.
 *
 * Wraps around an iterator to a hala::aligned_vector that can be advanced only in full strides
 * defined by the register type. The iterator can be either const or non-const depending on the
 * type of \b vec and the corresponding pointer and reference types will have corresponding const
 * qualifiers. The purpose of the iterator is to simplify processing of a vector stride-by-stride.
 *
 * The constructor is not intended for direct use, instead use the hala::mmbbegin() and hala::mmend()
 * methods. For example:
 * \code
 *  hala::aligned_vector<double> x(256); // number of entries is evenly divisible by all strides
 *  std::iota(x.begin(), x.end(), 0); // fill the vector, note that x is a std::vector and can be used as such
 *
 * for(auto ix = hala::mmbegin(x); ix < hala::mmend(x); ix++)
 *      hala::mmbind(ix) *= 2.0;
 * \endcode
 * The code will scale the vector x by 2.0 using only aligned memory read/write operations.
 *
 * The hala::mmend(x) is equivalent to x.end() so long as the number of entries is evenly divisible
 * by the stride; however, this cannot always be guaranteed in practice.
 * In such scenarios, the hala::mmend() will mark the end of the full strides, the remainder of
 * the vector can be processed by hala::mmrem(x) and x.end() both of which return regular iterators.
 * \code
 *  hala::aligned_vector<double> x(257); // number of entries is no divisible by any stride
 *  std::iota(x.begin(), x.end(), 0); // fill the vector, note that x is a std::vector and can be used as such
 *
 *  for(auto ix = hala::mmbegin(x); ix < hala::mmend(x); ix++)
 *      hala::mmbind(ix) *= 2.0;
 *  for(auto ix = hala::mmrem(x): ix < x.end(); ix++)
 *      *ix *= 2.0;
 * \endcode
 */
template<typename T, regtype R = default_regtype, class vec = aligned_vector<T, R>>
struct aligned_iterator{
    //! \brief The type of wrapped iterator.
    using iterator = typename aligned_iterator_traits<T, R, vec>::iterator;
    //! \brief The type of returned reference.
    using reference = typename aligned_iterator_traits<T, R, vec>::reference;
    //! \brief Defines the value type.
    using value_type = std::remove_cv_t<T>;
    //! \brief Difference between pointer types.
    using difference_type = std::ptrdiff_t;
    //! \brief Iterator category.
    using iterator_category = std::random_access_iterator_tag;
    //! \brief The internal iterator, do not access directly.
    iterator i;

    //! \brief Initialize from the beginning of the array.
    aligned_iterator(vec &x) : i(x.begin()){}
    //! \brief Initialize from a regular iterator, \b must be aligned.
    aligned_iterator(iterator x) : i(x){}

    //! \brief Dereference the iterator.
    reference operator *() const{ return *i; }

    //! \brief Advance the iterator by one stride.
    aligned_iterator<T, R, vec>& operator ++(){
        i += vex_stride<T, R>::stride;
        return *this;
    }
    //! \brief Move the iterator backwards by one stride.
    aligned_iterator<T, R, vec>& operator --(){
        i -= vex_stride<T, R>::stride;
        return *this;
    }
    //! \brief Advance the iterator by one stride, right operator.
    aligned_iterator<T, R, vec> operator ++(int){
        aligned_iterator<T, R, vec> t(*this);
        i += vex_stride<T, R>::stride;
        return t;
    }
    //! \brief Move the iterator backwards by one stride, right operator.
    aligned_iterator<T, R, vec> operator --(int){
        aligned_iterator<T, R, vec> t(*this);
        i -= vex_stride<T, R>::stride;
        return t;
    }

    //! \brief Advance the iterator by \b n full strides.
    aligned_iterator<T, R, vec> operator += (difference_type const n){
        i += n * vex_stride<T, R>::stride;
        return *this;
    }

    //! \brief Move the iterator backwards by \b n full strides.
    aligned_iterator<T, R, vec> operator -= (difference_type const n){
        i -= n * vex_stride<T, R>::stride;
        return *this;
    }

    //! \brief Compare the location addressed by the iterator, can compare to const or non-const iterators.
    template<class othervec> bool operator == (aligned_iterator<T, R, othervec> const &other) const{ return i == other.i; }
    //! \brief Compare the location addressed by the iterator, can compare to const or non-const iterators.
    template<class othervec> bool operator != (aligned_iterator<T, R, othervec> const &other) const{ return i != other.i; }
    //! \brief Compare the location addressed by the iterator, can compare to const or non-const iterators.
    template<class othervec> bool operator <  (aligned_iterator<T, R, othervec> const &other) const{ return i <  other.i; }
    //! \brief Compare the location addressed by the iterator, can compare to const or non-const iterators.
    template<class othervec> bool operator <= (aligned_iterator<T, R, othervec> const &other) const{ return i <= other.i; }
    //! \brief Compare the location addressed by the iterator, can compare to const or non-const iterators.
    template<class othervec> bool operator >  (aligned_iterator<T, R, othervec> const &other) const{ return i >  other.i; }
    //! \brief Compare the location addressed by the iterator, can compare to const or non-const iterators.
    template<class othervec> bool operator >= (aligned_iterator<T, R, othervec> const &other) const{ return i >= other.i; }
};

/*!
 * \ingroup HALAALALLOC
 * \brief Create a new aligned iterator starting from the beginning.
 *
 * Overloads are included for const and non-const vector \b x.
 * See hala::aligned_iterator for details.
 */
template<typename T, regtype R>
inline aligned_iterator<T, R, aligned_vector<T, R>> mmbegin(aligned_vector<T, R> &x){
    return aligned_iterator<T, R, aligned_vector<T, R>>(x);
}

/*!
 * \ingroup HALAALALLOC
 * \brief Create a new non-const aligned iterator pointing to the first non-aligned strip.
 *
 * Overloads are included for const and non-const vector \b x.
 * See hala::aligned_iterator for details.
 */
template<typename T, regtype R>
inline aligned_iterator<T, R, aligned_vector<T, R>> mmend(aligned_vector<T, R> &x){
    return aligned_iterator<T, R, aligned_vector<T, R>>(x.end() - x.size() % vex_stride<T, R>::stride);
}

/*!
 * \ingroup HALAALALLOC
 * \brief Create a new ordinary iterator pointing to the first of the remaining non-aligned entries.
 *
 * This method handles the case when working with an aligned vector with number of entries that is not
 * evenly divisible by the stride. In this case, there is a set of entries that cannot be handles
 * in vector operations due to not having a sufficient count.
 * The hala::mmrem() method works with both const and non-const vectors and will create an
 * iterator to the first of the remainder entries. See hala::aligned_iterator.
 */
template<typename T, regtype R>
inline auto mmrem(aligned_vector<T, R> &x){
    return x.end() - x.size() % vex_stride<T, R>::stride;
}

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section (overloads are documented with the main methods)
template<typename T, regtype R>
inline aligned_iterator<T, R, aligned_vector<T, R> const> mmbegin(aligned_vector<T, R> const &x){
    return aligned_iterator<T, R, aligned_vector<T, R> const>(x);
}
template<typename T, regtype R>
inline aligned_iterator<T, R, aligned_vector<T, R> const> mmend(aligned_vector<T, R> const &x){
    return aligned_iterator<T, R, aligned_vector<T, R> const>(x.end() - x.size() % vex_stride<T, R>::stride);
}
template<typename T, regtype R>
inline auto mmrem(aligned_vector<T, R> const &x){
    return x.end() - x.size() % vex_stride<T, R>::stride;
}
#endif


/*!
 * \ingroup HALAALALLOC
 * \brief Advance an iterator by one stride.
 *
 * Overloads:
 * \code
 *  hala::mmadvance(iterator);         // same as iterator += stride
 *  hala::mmadvance(aligned_iterator); // same as iterator++
 *  hala::mmadvance(ix, iy, iz);       // calls either += stride or ++
 * \endcode
 * Each iterator type is advance by exactly one stride, the variadric overload will advance
 * each iterator in turn.
 */
template<regtype R = default_regtype, class VectorIterator,
         typename T = typename std::iterator_traits<VectorIterator>::value_type>
inline void mmadvance(VectorIterator &a){ a += vex_stride<T, R>::stride; }

#ifndef __HALA_DOXYGEN_SKIP
template<regtype R = default_regtype, class vec, typename T>
inline void mmadvance(aligned_iterator<T, R, vec> &a){ ++a; }

template<regtype R = default_regtype, typename iter1, typename iter2, typename... more_iters>
void mmadvance(iter1 &x, iter2 &y, more_iters &... z){
    mmadvance<R>(x);
    mmadvance<R>(y, z...);
}
#endif

/*!
 * \ingroup HALAALALLOC
 * \brief Copy \b x into a std::vector using the \b aligned_allocator, copy is done either with BLAS xcopy or std::copy.
 */
template<regtype R = default_regtype, class VectorLike>
auto make_aligned(VectorLike const &x){
    check_types(x);
    using scalar_type = typename define_type<VectorLike>::value_type;
    aligned_vector<scalar_type, R> y(get_size(x));
    hala::vcopy(x, y);
    return y;
}

}

#endif
