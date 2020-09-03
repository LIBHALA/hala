#ifndef __HALA_VECTOR_DEFINES_HPP
#define __HALA_VECTOR_DEFINES_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_core.hpp"

/*!
 * \internal
 * \file hala_vector_defines.hpp
 * \brief HALA templates to extend interoperability with custom vectors.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACUSTOM
 *
 * Structs and functions for extending HALA templates to
 * user defined vector classes.
 * \endinternal
*/

/*!
 * \defgroup HALACUSTOM HALA Support for Custom Vector Class
 *
 * Several templates that can be fully or partially specialized to allow
 * interoperability between HALA wrappers and a user defined vector class.
 * - HALA provides builtin support for std::vector, std::valarray containers
 *   as well as the included hala::cuda_vector and the variations of HALA
 *   raw-array wrappers.
 * - Classes that define \b value_type, and have \b size() and \b data()
 *   methods similar to std::vector will be supported automatically.
 * - If a vector has a \b resize() method, it will be used to adjust the
 *   size of the vector in calls where the vector is a \a pure output,
 *   for example
 *   \code
 *   hala::vcopy(x, y);             // y is purely an output
 *   hala::gemv('N', M, N, 1.0, A, x, 0.0, y); // y is purely an output
 *   hala::gemv('N', M, N, 1.0, A, x, 1.0, y); // y is used as both input and output
 *   \endcode
 * - Vector classes that lack the three mandatory methods/types
 *   or the methods do something else (e.g., resize() returns a vector with
 *   entries in a different precision or value type has cv qualifiers),
 *   require specialization of several HALA templates.
 *
 *
 * \par Templates to Specialize
 *
 * In order to call the method for the correct scalar type or precision,
 * the struct hala::define_type must be specialized.
 * The struct has a single type member \b value_type which should specify the
 * content of the vector.
 *
 * Sanity checks in debug mode and inferring the vector sizes is done with
 * struct hala::define_size.
 * The struct has a single member hala::define_size::size() that accepts a const vector and
 * returns the number of elements. The struct allows for either partial or
 * full specialization; alternatively, the hala::get_size() function can be
 * fully specialized to avoid dealing with the struct.
 *
 * In some cases, vectors serve as pure output parameters, e.g., computing
 * \f$ y = \alpha A x + \beta y \f$ with \b beta equal to zero.
 * In such scenario, if the size of the output vector size is insufficient,
 * HALA will resize the vector to the minimum required.
 * The resize algorithm can be done with either specialization of
 * struct hala::define_resize or function hala::set_size().
 * \b Note: that if the provided size is sufficient, resize will never
 * be called.
 *
 * The libraries wrapped by HALA accept raw pointers to data,
 * which is accessed through the struct hala::define_data, which contains
 * a single method hala::define_data::data().
 * The struct has to be specialized for any vector class that does not have
 * a \b data() method.
 * As an alternative, the \b hala::get_data() function can also be specialized.
 *
 * \par Examples
 *
 * Specialization for a very simple vector class:
 * \code
 * #include "hala.hpp"
 *
 * // simple class that automatically allocates and deallocates data
 * // this is the most basic example, anything less will be just a raw pointer
 * // raw-pointers are handled with hala::wrap_array
 * struct FooVector{
 *     FooVector(size_t allocate_size) :
 *         pointer(new double[allocate_size]), num_entries(allocate_size){}
 *     ~FooVector(){ delete[] pointer; }
 *     double *pointer;
 *     size_t num_entries;
 * };
 *
 * namespace hala{
 *     // specializations needed to use FooVector in HALA templates
 *     // note that some specializations require both const and non-const FooVector
 *     template<> struct define_type<FooVector>{
 *         using value_type = double;
 *     };
 *     template<> struct define_type<FooVector const>{
 *         using value_type = double;
 *     };
 *     template<> size_t get_size<FooVector>(FooVector const &x)
 *         { return x.num_entries; }
 *     template<> size_t get_size<FooVector const>(FooVector const &x)
 *         { return x.num_entries; }
 *     template<> auto get_data<FooVector>(FooVector &x)
 *         { return x.pointer; }
 *     template<> auto get_data<FooVector const>(FooVector const &x)
 *         { return x.pointer; }
 *     // optional specialization that changes the size
 *     // template<> void set_size<FooVector>(size_t new_size, FooVector &x){}
 * }
 *
 * int main(void){
 *     FooVector x(3);
 *     x.pointer[0] = 0;
 *     x.pointer[1] = 1;
 *     x.pointer[2] = 2;
 *     std::cout << "norm computed by HALA: " << hala::norm2(x) << '\n';
 *     std::cout << "      expected answer: " << std::sqrt(5.0) << '\n';
 *     return 0;
 * }
 * \endcode
 *
 * The simple example above is using specializations of the functions,
 * which is easy enough since \b FooVector can only have scalar type \b double.
 * However, if the custom vector class is a template, specializing the
 * functions for every possible parameter can be tedious and partial
 * specialization of structs is a much easier approach.
 *
 * \code
 * #include "hala.hpp"
 *
 * // extension of FooVector to other types
 * template<typename T>
 * struct FooVector{
 *     FooVector(size_t allocate_size) :
 *         pointer(new T[allocate_size]), num_entries(allocate_size){}
 *     ~FooVector(){ delete[] pointer; }
 *     T *pointer;
 *     size_t num_entries;
 * };
 *
 * namespace hala{
 *     // specializations needed to use template FooVector in HALA templates
 *     // note that some specializations require both const and non-const FooVector
 *     template<typename T> struct define_type<FooVector<T>>{
 *         using value_type = std::remove_cv_t<T>;
 *     };
 *     template<typename T> struct define_type<FooVector<T> const>{
 *         using value_type = std::remove_cv_t<T>;
 *     };
 *     template<typename T> struct define_size<FooVector<T>>{
 *         static auto size(FooVector<T> const &x){ return x.num_entries; }
 *     };
 *     template<typename T> struct define_size<FooVector<T> const>{
 *         static auto size(FooVector<T> const &x){ return x.num_entries; }
 *     };
 *     template<typename T> struct define_data<FooVector<T>>{
 *         static auto data(FooVector<T> &x){ return x.pointer; }
 *     };
 *     template<typename T> struct define_data<FooVector<T> const>{
 *         static auto data(FooVector<T> const &x){ return x.pointer; }
 *     };
 *     // optional resize struct
 *     // template<typename T> struct define_resize<FooVector<T>>{
 *     //     static void resize(size_t new_size, FooVector<T> &x){
 *     //         delete[] x.pointer;
 *     //         x.points = new T[new_size];
 *     //         x.num_entries = new_size;
 *     //     }
 *     // };
 * }
 *
 * template<typename T>
 * void test(){
 *     FooVector<T> x(3);
 *     x.pointer[0] = 0;
 *     x.pointer[1] = 1;
 *     x.pointer[2] = 2;
 *     std::cout << "norm computed by HALA: " << hala::norm2(x) << '\n';
 *     std::cout << "      expected answer: " << std::sqrt(5.0) << '\n';
 * }
 *
 * int main(void){
 *
 *     std::cout << "Test with T = float\n";
 *     test<float>();
 *     std::cout << "\nTest with T = double\n";
 *     test<double>();
 *
 *     return 0;
 * }
 * \endcode
 *
 * The two simple implementations of \b FooVector required specialization for
 * the four types, size, resize, and data. However, if a vector already
 * provides some of those methods in a manner consistent with std::vector,
 * then only the missing cases have to be handled.
 * For example, the std::valarray container satisfies the first three cases,
 * but lacks the \b .data() method. Thus, only the \b define_data struct
 * requires a specialization and HALA already includes the necessary code.
 * Similarly, if we redefine \b FooVector we can remove the need to specialize
 * several of the structs:
 *
 * \code
 * #include "hala.hpp"
 *
 * // extension of FooVector to other types
 * template<typename T>
 * struct FooVector{
 *     using value_type = T;
 *     FooVector(size_t allocate_size) :
 *         pointer(new T[allocate_size]), num_entries(allocate_size){}
 *     ~FooVector(){ delete pointer; }
 *     T* data(){ return pointer; }
 *     T const* data() const{ return pointer; }
 *     size_t size() const{ return num_entries; }
 *     T *pointer;
 *     size_t num_entries;
 * };
 *
 * namespace hala{
 *     // define_type, define_size, and define_data will automatically
 *     // pick up the corresponding methods of \b FooVector<T>
 *     template<typename T> struct define_resize<FooVector<T>>{
 *         static void resize(size_t, FooVector<T> &){
 *             // even this is not necessary, by default HALA will trow
 *             // a runtime_error if a resize is attempted.
 *         }
 *     };
 * }
 *
 * template<typename T>
 * void test(){
 *     FooVector<T> x(3);
 *     x.pointer[0] = 0;
 *     x.pointer[1] = 1;
 *     x.pointer[2] = 2;
 *     std::cout << "norm computed by HALA: " << hala::norm2(x) << '\n';
 *     std::cout << "      expected answer: " << std::sqrt(5.0) << '\n';
 * }
 *
 * int main(void){
 *
 *     std::cout << "Test with T = float\n";
 *     test<float>();
 *     std::cout << "\nTest with T = double\n";
 *     test<double>();
 *
 *     return 0;
 * }
 * \endcode
 *
 */

namespace hala{

/*!
 * \ingroup HALACUSTOM
 * \brief Template that defined the type of the vector variable.
 *
 * Given the \b VectorLike class, the structure must define a \b value_type
 * member type, i.e., \b using \b value_type \b = \b scalar_type,
 * where \b scalar_type is the type of the vector entries (not including qualifiers).
 * The included implementation will automatically handle any \b VectorLike class
 * which defines a member \b typedef \b value_type in the same way as the \b std::vector class.
 * Must be specialized to handle \b VectorLike classes that do not provide \b value_type.
 */
template<class VectorLike> struct define_type{
    //! \brief The scalar type of the \b VectorLike class without any const qualifiers.
    using value_type = std::remove_cv_t<typename std::remove_reference_t<VectorLike>::value_type>;
};

/*!
 * \internal
 * \ingroup HALACUSTOM
 * \brief Specialization handling the type of a pointer.
 *
 * \endinternal
 */
template<typename T> struct define_type<T*>{
    //! \brief The scalar type of the \b VectorLike class without any const qualifiers.
    using value_type = std::remove_cv_t<T>;
};

/*!
 * \internal
 * \ingroup HALACUSTOM
 * \brief Specialization handling the type of an output pointer.
 *
 * \endinternal
 */
template<typename T> struct define_type<T*&>{
    //! \brief The scalar type of the \b VectorLike class without any const qualifiers.
    using value_type = std::remove_cv_t<T>;
};

/*!
 * \internal
 * \ingroup HALACUSTOM
 * \brief Specialization handling the type of a const pointer.
 *
 * \endinternal
 */
template<typename T> struct define_type<T* const>{
    //! \brief The scalar type of the \b VectorLike class without any const qualifiers.
    using value_type = std::remove_cv_t<T>;
};

/*!
 * \ingroup HALACUSTOM
 * \brief Template that defines the size of the vector.
 *
 * A struct with a single method that returns the size of the vector.
 * The struct allows for partial specialization of the way the vector size is computed, i.e.,
 * a template class of \b VectorLike objects that lack the \b .size() method.
 */
template<class VectorLike> struct define_size{
    //! \brief Simple method that returns the size of the \b VectorLike, allows partial and full specializations for custom vector classes.
    static auto size(VectorLike const &x){ return x.size(); }
};

/*!
 * \internal
 * \ingroup HALACUSTOM
 * \brief Always assume the raw pointer has enough space allocated to it.
 *
 * \endinternal
 */
template<typename T> struct define_size<T*>{
    //! \brief Gets called through hala::get_size(), does not need other specializations.
    static size_t size(T const *){ return std::numeric_limits<size_t>::max(); }
};

/*!
 * \ingroup HALACUSTOM
 * \brief Returns the number of entries in the vector, if \b VectorLike does not have \b .size() method, then the template must be specialized.
 *
 * The function uses \b hala::define_size which in turn uses \b VectorLike::size().
 * If the \b VectorLike class doesn't have \b .size(), then either this function
 * or struct \b \b hala::define_size have to be specialized.
 */
template<class VectorLike>
size_t get_size(VectorLike const &x){ return static_cast<size_t>(define_size<VectorLike>::size(x)); }

/*!
 * \ingroup HALACHECKTYPES
 * \brief Take the output of \b get_size() and type-cast it to \b int.
 */
template<class VectorLike>
int get_size_int(VectorLike const &x){ return static_cast<int>(get_size(x)); }


/*!
 * \ingroup HALACHECKTYPES
 * \brief Struct to check if a \b VectorLike class has a resize method.
 *
 * Defines two template methods that depend on the VectorLike. One takes two template typename parameters
 * where the second defaults to the return type of the vector resize() method (possibly void), note the use of
 * decltype and declval. The second method the other takes a typename and a variadric pack as inputs.
 * The two methods return true and false types.
 * When a call to the method is generated, the first method will be selected if the first method has a proper
 * default parameter (i.e., the vector has resize() method), since variadric templates always have the lowest priority.
 * Otherwise, the variadric method will be chosen.
 *
 * \b Note: other common methods use typeof() which is a gcc extension, and/or taking a pointer to the method which fails
 *          if the method belong to an overload class.
 */
template<typename VectorLike> struct has_resize{
    //! \brief Template that has proper default only if the vector has a resize method.
    template <typename T, typename S = decltype( std::declval<T>().resize( std::declval<size_t>() ) )> static std::true_type check(int);
    //! \brief Default fallback meaning that the vector has not type.
    template <typename T> static std::false_type check(...);

    //! \brief Define value to be true/false based on the choice of an overload.
    static constexpr bool value = decltype( check<VectorLike>(0) )::value;
};

/*!
 * \ingroup HALACUSTOM
 * \brief Template that defines how to resize of the vector.
 *
 * A struct with a single method that resizes the vector.
 * The struct allows for partial specialization of the way the vector is resized, i.e.,
 * a templated class of \b VectorLike objects that lack the \b .resize(\b size_t \b x) method.
 * While the default (non-specialized) struct defines two methods, only one is ever enabled depending
 * on whether the vector has a resize method.
 */
template<class VectorLike> struct define_resize{
    //! \brief Simple method that generates a runtime exception for vectors without a resize() method.
    template<typename VectorTest = VectorLike>
    std::enable_if_t< !has_resize<VectorTest>::value > static resize(size_t, VectorTest &x){
        throw std::runtime_error(std::string("Output has insufficient size and ") + typeid(x).name() + " cannot be resized.");
    }
    //! \brief Simple method that resizes a vector using the resize method.
    template<typename VectorTest = VectorLike>
    std::enable_if_t< has_resize<VectorTest>::value > static resize(size_t new_size, VectorTest &x){
        x.resize(new_size);
    }
};


/*!
 * \ingroup HALACUSTOM
 * \brief Set new size for the \b VectorLike, called only if \b get_size(x) is not sufficient and it uses \b .resize() method.
 *
 * HALA will never call \b set_size() on a \b VectorLike that has sufficient size for the requested operation.
 * If called, the included implementation will use the \b .resize() method and will automatically work
 * for any \b VectorLike class that implements the method similar to \b std::vector and \b std::valarray.
 * This template must be specialized to use classes that do not have \b .resize().
 */
template<class VectorLike>
void set_size(size_t new_size, VectorLike &x){ define_resize<VectorLike>::resize(new_size, x); }

/*!
 * \ingroup HALACUSTOM
 * \brief Struct with a single function that returns a raw pointer to the internal contiguous data stored in the vector, uses \b .data() method.
 *
 * Unlike the \b get_data() function, the \b data_extractor class can be partially specialized,
 * so a single template can be used to cover many cases, e.g., \b std::valarray<float> and \b std::valarray<double>.
 */
template<class VectorLike> struct define_data{
    //! \brief Returns a raw pointer to the internal data of the \b VectorLike.
    static auto data(VectorLike &x){ return x.data(); }
};

/*!
 * \internal
 * \ingroup HALACUSTOM
 * \brief Specialization for \b std::valarray containers, uses \b &x[0] go get the data.
 *
 * \endinternal
 */
template<typename T> struct define_data<typename std::valarray<T>>{
    //! \brief Returns a raw pointer to the internal data of the \b valarray.
    static auto data(typename std::valarray<T> &x){ return &x[0]; }
};
/*!
 * \internal
 * \ingroup HALACUSTOM
 * \brief Specialization for \b const \b std::valarray containers, uses \b &x[0] go get the data.
 *
 * \endinternal
 */
template<typename T> struct define_data<typename std::valarray<T> const>{
    //! \brief Returns a raw pointer to the internal data of the \b valarray.
    static auto data(typename std::valarray<T> const &x){ return &x[0]; }
};

/*!
 * \internal
 * \ingroup HALACUSTOM
 * \brief Specialization for pointers.
 *
 * \endinternal
 */
template<typename T> struct define_data<T*>{
    //! \brief Returns back the pointer.
    static T* data(T *x){ return x; }
};

/*!
 * \internal
 * \ingroup HALACUSTOM
 * \brief Specialization for const pointers.
 *
 * \endinternal
 */
template<typename T> struct define_data<T* const>{
    //! \brief Returns back the const pointer.
    static T const* data(T const *x){ return x; }
};

/*!
 * \ingroup HALACUSTOM
 * \brief Returns a raw pointer to the internal contiguous array stored in the \b VectorLike.
 *
 * The function uses the \b data_extractor struct and will automatically if the \b VectorLike
 * has the \b .data() method similar to \b std::vector, or if the \b data_extractor has
 * been specialized for the specific \b VectorLike.
 * Otherwise, this function has to be specialized to handle a user provide \b VectorLike.
 */
template<class VectorLike>
auto get_data(VectorLike &x){ return define_data<VectorLike>::data(x); }

/*!
 * \ingroup HALACUSTOM
 * \addtogroup HALACTYPES Custom Type Definitions
 *
 * BLAS and LAPACK standards works with real and complex floating point numbers
 * with single and double precision. The corresponding equivalents C++ standard types are:
 * \code
 *      float, double, std::complex<float>, std::complex<double>
 * \endcode
 * By default, HALA works with these types; however, some libraries define their own.
 * This is often true in the case of complex numbers and
 * to facilitate interfacing with custom complex types, HALA provides several templates that can be
 * specialized to help identify the correct native equivalents.
 *
 * Note that C++ is good at automatically identifying the native aliases with
 * the std::is_same template. Specialization here is needed only if the std::is_same
 * does not correctly identify the types.
 *
 * Example of custom double-complex:
 * \code
 *  struct CustomComplex{
 *      double r, i;
 *      // define just multiplication for example
 *      CustomComplex operator *= (CustomComplex const &other){
 *          double temp = r;
 *          r = r * other.r - i * other.i;
 *          i = i * other.r + temp * other.i;
 *          return *this;
 *      }
 *  };
 *
 *  std::vector<CustomComplex> x = {{1.0, 1.0}, {1.0, 2.0}};
 *
 *  // std::cout << hala::norm2(x) << std::endl; // will fail because CustomComplex is unknown
 *
 *  template<> struct hala::is_dcomplex<CustomComplex> : std::true_type{};
 *
 *  std::cout << hala::norm2(x) << std::endl; // now this works
 *
 * \endcode
 *
 * The C++ standard also includes methods such as std::real() and std::imag() that return the real
 * and complex components of C++ standard types and have appropriate overloads for float and double.
 * However, the conjugate operation in the standard is defined as std::conj() and always returns
 * a complex number even if the input is real, which makes it hard to generalize operations of the
 * form (e.g., used often in algorithms related to Hilbert space geometry):
 * \code
 *  auto z = std::conj(x) * y; // z will be complex even if x and y are real float or double
 * \endcode
 *
 * HALA provides a set of complex methods that return real numbers for real inputs and work
 * for custom complex numbers as defined above:
 * - hala::hala_real()
 * - hala::hala_imag()
 * - hala::hala_conj()
 * - hala::hala_abs()
 * The standard STL real, imag, conj, and abs methods will be used whenever appropriate,
 * reinterpret-cast and other tricks will be employed to handle the custom complex cases.
 * The methods are designed to work with standard types, custom complex types, and custom real
 * types that are aliases to float or double (or int), there is no guarantee that the methods
 * would work for some exotic double precision standard.
 */

/*!
 * \ingroup HALACTYPES
 * \brief Struct that defines whether \b candidate is effectively the same as \b float.
 */
template<typename candidate>
struct is_float{
    //! \brief A single static member, by default equal to std::is_same<candidate, float>::value.
    static const bool value = std::is_same<candidate, float>::value;
};

/*!
 * \ingroup HALACTYPES
 * \brief Struct that defines whether \b candidate is effectively the same as \b double.
 */
template<typename candidate>
struct is_double{
    //! \brief A single static member, by default equal to std::is_same<candidate, double>::value.
    static const bool value = std::is_same<candidate, double>::value;
};

/*!
 * \ingroup HALACTYPES
 * \brief Struct that defines whether \b candidate is effectively the same as \b std::complex<float>.
 */
template<typename candidate>
struct is_fcomplex{
    //! \brief A single static member, by default equal to std::is_same of candidate and std::complex<float>.
    static const bool value = std::is_same<candidate, std::complex<float>>::value;
};

/*!
 * \ingroup HALACTYPES
 * \brief Struct that defines whether \b candidate is effectively the same as \b std::complex<double>.
 */
template<typename candidate>
struct is_dcomplex{
    //! \brief A single static member, by default equal to std::is_same of candidate and std::complex<double>.
    static const bool value = std::is_same<candidate, std::complex<double>>::value;
};

}

#endif
