#ifndef __HALA_GPU_WRAP_ARRAY_HPP
#define __HALA_GPU_WRAP_ARRAY_HPP
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
 * \file hala_gpu_wrap_array.hpp
 * \brief HALA wrapper to simple GPU arrays.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAGPUWRAP
 *
 * Contains the HALA GPU array wrapper.
*/

#ifdef HALA_ENABLE_CUDA
#include "hala_cuda_common.hpp"
#endif

#ifdef HALA_ENABLE_ROCM
#include "hala_rocm_common.hpp"
#endif

namespace hala{

/*!
 * \ingroup HALAGPU
 * \addtogroup HALAGPUWRAP GPU Array Wrapper
 *
 * \par Wrapper to Raw GPU Arrays
 * Wrapper to GPU arrays similar to the \b hala::wrapped_array.
 */


/*!
 * \ingroup HALAGPUWRAP
 * \brief Wrapper around C-style GPU arrays that can be passed to hala::gpu_engine templates.
 *
 * This wrapper accepts an externally allocated array and size and then provide
 * an interface that mimics containers.
 * However, the wrapper does not provide full container interface (only the bare minimum),
 * and the array will \b NOT be deleted by the destructor.
 *
 * The constructor accepts an array and the size of the array,
 * the provided size will be used for all operators that need to infer the number of entries.
 * The resize method will never change the size of the array,
 * any request for change of size to new size will cause a run-time exception.
 */
template<typename ScalarType>
class gpu_wrapped_array{
public:
    //! \brief Keeps track of the array type for sanity check and static_assert().
    using value_type = std::remove_const_t<ScalarType>;

    //! \brief Wraps the array \b arr and assumes the size is \b num_entries.
    gpu_wrapped_array(ScalarType *arr, size_t num_entries) : gpu_data(arr), num(num_entries){
        check_gpu_type<value_type>();
    }
    //! \brief Delete copy constructor to avoid an alias.
    gpu_wrapped_array(gpu_wrapped_array<ScalarType> const &) = delete;
    //! \brief Delete copy constructor to avoid an alias.
    void operator =(gpu_wrapped_array<ScalarType> const &) = delete;
    //! \brief Move constructor.
    gpu_wrapped_array(gpu_wrapped_array<ScalarType> &&other)
         : gpu_data(std::exchange(other.gpu_data, nullptr)), num(std::exchange(other.num, 0)){}
    //! \brief Move assignemnt.
    void operator =(gpu_wrapped_array<ScalarType> &&other){
        gpu_wrapped_array<ScalarType> temp(std::move(other));
        std::swap(num, temp.num);
        std::swap(gpu_data, temp.gpu_data);
    }
    //! \brief Destructor, does not delete the array.
    ~gpu_wrapped_array(){}

    //! \brief Returns the assumed size.
    size_t size() const{ return num; }

    //! \brief Return an alias to the wrapper array.
    ScalarType* data(){ return gpu_data; }
    //! \brief Return a const alias to the wrapper array.
    ScalarType const* data() const{ return gpu_data; }

    //! \brief Copy the \b cpu_data from the cpu vector to the internal gpu array.
    template<class VectorLike>
    void load(const VectorLike &cpu_data){
        static_assert(std::is_same<value_type, typename define_type<VectorLike>::value_type>::value, "mismatch in the type of gpu_wrapped_array and cpu_data when calling load()");
        assert(num == get_size(cpu_data));
        gpu_copy_n<copy_direction::host2device>(get_data(cpu_data), num, gpu_data);
    }
    //! \brief Copy the data currently held on the device to the vector, the vector is resized if the current size is too small.
    template<class VectorLike>
    void unload(VectorLike &cpu_data) const{
        static_assert(std::is_same<value_type, typename define_type<VectorLike>::value_type>::value, "mismatch in the type of gpu_wrapped_array and cpu_data when calling unload()");
        check_set_size(assume_output, cpu_data, num);
        gpu_copy_n<copy_direction::device2host>(gpu_data, num, get_data(cpu_data));
    }
    //! \brief Unload the data into a std::vector and return the result.
    std::vector<value_type> unload() const{
        std::vector<value_type> cpu_data(num);
        unload(cpu_data);
        return cpu_data;
    }
    //! \brief Unload the data into a std::valarray and return the result.
    std::valarray<value_type> unload_valarray() const{
        std::valarray<value_type> cpu_data(num);
        unload(cpu_data);
        return cpu_data;
    }

private:
    ScalarType *gpu_data;
    size_t num;
};

/*!
 * \ingroup HALAGPUWRAP
 * \brief Creates a wrapper around the C-style GPU arrays that can be passed to HALA templates.
 */
template<typename ArrayType>
gpu_wrapped_array<ArrayType> wrap_gpu_array(ArrayType arr[], size_t num_entries){
    return gpu_wrapped_array<ArrayType>(arr, num_entries);
}

}

#endif
