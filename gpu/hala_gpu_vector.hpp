#ifndef __HALA_GPU_VECTOR_HPP
#define __HALA_GPU_VECTOR_HPP
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
 * \file hala_gpu_vector.hpp
 * \brief HALA defined gpu_vector class.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAGPUVECTOR
 *
 */

#include "hala_gpu_wrap_array.hpp"

namespace hala{

/*!
 * \ingroup HALAGPU
 * \addtogroup HALAGPUVECTOR GPU Vector Class
 *
 * \par GPU Vector Class
 * The hala::gpu_vector class provides C++ container-style of API for
 * GPU memory management.
 */

/*!
 * \ingroup HALAGPUVECTOR
 * \brief Templated class providing functionality similar to C++ containers.
 *
 * A class that provides a C++ container-style of API that wraps around a raw pointer to a GPU array.
 * The template can be instantiated with float, double, std::complex<float>, std::complex<double>, and int.
 * The type can also be inferred using hala::make_gpu_vector().
 *
 * Each vector is associated with a device when constructed and all memory operations are carried on that device.
 *
 * The vector can also be used to wrap around a simple device array (preserving const correctness),
 * wrapping should be done with gpu_wrap_array() or gpu_engine::wrap_array().
 */
template<typename T>
class gpu_vector{
public:
    //! \brief Default constructor, make an empty vector and associate with device \b deviceid.
    gpu_vector(int deviceid = 0) : gpu(deviceid), num(0), gpu_data(nullptr){
        check_gpu_type<T>();
        assert((gpu >= 0) && (gpu < gpu_device_count()));
    }
    //! \brief Constructor, make a vector with size \b num_entries and associate with \b deviceid.
    gpu_vector(size_t num_entries, int deviceid) : gpu(deviceid), gpu_data(nullptr){
        check_gpu_type<T>();
        alloc(num_entries);
        assert((gpu >= 0) && (gpu < gpu_device_count()));
    }
    //! \brief Copy-constructor, copy the device and the data of the \b other vector.
    gpu_vector(const gpu_vector<T>& other) : gpu(other.gpu), num(0), gpu_data(nullptr){
        alloc(other.num);
        gpu_copy_n<copy_direction::device2device>(other.gpu_data, num, gpu_data);
    }
    //! \brief Move constructor.
    gpu_vector(gpu_vector<T> &&other) : gpu(other.gpu), num(std::exchange(other.num, 0)), gpu_data(std::exchange(other.gpu_data, nullptr)){}
    //! \brief Destructor, free all resources.
    ~gpu_vector(){ clear(); }

    //! \brief Free all data and reset the vector to empty.
    void clear(){
        gpu_free(gpu_data);
        gpu_data = nullptr;
        num = 0;
    }

    //! \brief Copy assignment, copy all data from \b other to this vector; if devices don't match the data will be moved through the cpu.
    void operator =(gpu_vector<T> const &other){
        if (other.gpu == gpu){
            *this = gpu_vector<T>(*this);
        }else{
            load(other.unload());
        }
    }

    //! \brief Move assignment.
    void operator =(gpu_vector<T>&& other){
        gpu_vector<T> temp(std::move(other));
        std::swap(gpu, temp.gpu);
        std::swap(num, temp.num);
        std::swap(gpu_data, temp.gpu_data);
    }

    //! \brief If \b new_size doesn't match \b size(), then delete the current array and allocate a new one; all current data is \b lost.
    void resize(size_t new_size){
        if (new_size == num) return;
        gpu_free(gpu_data);
        alloc(new_size);
    }
    //! \brief Copy the \b cpu_data from the cpu vector to the internal gpu array.
    template<class VectorLike>
    void load(const VectorLike &cpu_data){
        static_assert(std::is_same<T, typename define_type<VectorLike>::value_type>::value, "mismatch in the type of gpu_vector and cpu_data when calling load()");
        resize(get_size(cpu_data));
        gpu_copy_n<copy_direction::host2device>(get_data(cpu_data), num, gpu_data);
    }
    //! \brief Copy the data currently held on the device to the vector, the vector is resized if the current size is too small.
    template<class VectorLike>
    void unload(VectorLike &cpu_data) const{
        static_assert(std::is_same<T, typename define_type<VectorLike>::value_type>::value, "mismatch in the type of gpu_vector and cpu_data when calling unload()");
        check_set_size(assume_output, cpu_data, num);
        gpu_copy_n<copy_direction::device2host>(gpu_data, num, get_data(cpu_data));
    }
    //! \brief Unload the data into a std::vector and return the result.
    std::vector<T> unload() const{
        std::vector<T> cpu_data(num);
        unload(cpu_data);
        return cpu_data;
    }
    //! \brief Unload the data into a std::valarray and return the result.
    std::valarray<T> unload_valarray() const{
        std::valarray<T> cpu_data(num);
        unload(cpu_data);
        return cpu_data;
    }

    //! \brief Give reference to the array, can be passed directly into cuBlas/cuSparse calls or custom kernels.
    T* data(){ return gpu_data; }
    //! \brief Give const reference to the array, can be passed directly into cuBlas/cuSparse calls or custom kernels.
    const T* data() const{ return gpu_data; }

    //! \brief Allows the vector to be used directly in place of a raw pointer.
    operator T *(){ return gpu_data; }
    //! \brief Allows the vector to be used directly in place of a raw const-pointer.
    operator T const *() const{ return gpu_data; }

    //! \brief Return the current size of the array, i.e., the number of elements.
    size_t size() const{ return num; }
    //! \brief Return the GPU device associated with this vector.
    int device() const{ return gpu; }
    //! \brief Return \b true if the vector is has zero size.
    bool empty() const{ return (num == 0); }

    //! \brief Fill the vector with the given \b value using log(size()) calls to gpu_copy_n().
    void fill(T value){
        if (num == 0) return;
        gpu_copy_n<copy_direction::host2device>(&value, 1, gpu_data);
        size_t loaded = 1;
        while(loaded <= num){
            gpu_copy_n<copy_direction::device2device>(gpu_data, std::min(loaded, num - loaded), gpu_data + loaded);
            loaded *= 2;
        }
    }

    //! \brief The value of the array, used for static error checking.
    using value_type = T;

protected:
    //! \brief Allocate a new array with the size \b num; does not free an existing array.
    void alloc(size_t new_size){
        num = new_size;
        gpu_data = gpu_allocate<T>(gpu, num);
    }

private:
    int gpu;
    size_t num;
    T *gpu_data;
};

/*!
 * \ingroup HALAGPUVECTOR
 * \brief Returns a hala::gpu_vector with type matching \b VectorLike, load the data from \b VectorLike into the array.
 */
template<class VectorLike>
inline auto make_gpu_vector(const VectorLike &cpu_data, int gpuid = 0){
    gpu_vector<typename define_type<VectorLike>::value_type> gpu_data(gpuid);
    gpu_data.load(cpu_data);
    return gpu_data;
}

/*!
 * \ingroup HALAGPUVECTOR
 * \brief Returns a hala::gpu_vector with given type and \b num_entries on the given GPU device.
 */
template<typename T>
inline auto make_gpu_vector(size_t num_entries, int gpuid = 0){
    return gpu_vector<T>(num_entries, gpuid);
}

/*!
 * \ingroup HALAGPUVECTOR
 * \brief Template specialization for GPU device ID for non-const hala::gpu_vector.
 */
template<typename T>
struct deviceid_extractor<gpu_vector<T>>{
    //! \brief Return the device associated with the vector (non-const case).
    static int device(gpu_vector<T> const &x){ return x.device(); }
};

/*!
 * \ingroup HALAGPUVECTOR
 * \brief Template specialization for GPU device ID for const hala::gpu_vector.
 */
template<typename T>
struct deviceid_extractor<const gpu_vector<T>>{
    //! \brief Return the device associated with the vector (const case).
    static int device(gpu_vector<T> const &x){ return x.device(); }
};

/*!
 * \ingroup HALAGPUVECTOR
 * \brief A struct that wraps around a gpu_vector and a reference to a cpu vector-like.
 *
 * At the constructor stage, the data from the cpu-vector will be loaded into the gpu vector,
 * when the destructor is called, the data currently set in the gpu vector will be written back.
 * This allows for RAII style of management of temporary data that needs to be placed
 * back on the CPU after the call.
 *
 * The struct can be used as a gpu vector like class.
 */
template<typename T, class VectorLike> struct binded_gpu_vector{
    //! \brief Defines the type.
    using value_type = T;
    //! \brief Constructor, load the CPU data into the gpu vector.
    binded_gpu_vector(int gpuid, VectorLike &v) : vec(v), cvec(gpuid){
        static_assert(!std::is_const<VectorLike>::value, "Cannot bind to a const vector!");
        cvec.load(v);
    }
    //! \brief Destructor, sync-back with the CPU.
    ~binded_gpu_vector(){ cvec.unload(vec); }

    //! \brief Return the internal data.
    T* data(){ return cvec.data(); }
    //! \brief Return the internal const data.
    T const* data() const{ return cvec.data(); }
    //! \brief Resize the internal gpu vector.
    void resize(size_t new_size){ cvec.resize(new_size); }
    //! \brief Return the size of the internal gpu vector.
    size_t size() const{ return cvec.size(); }
    //! \brief Resize the GPU device id of the gpu vector.
    int device() const{ return cvec.device(); }

private:
    VectorLike &vec;
    gpu_vector<T> cvec;
};

}

#endif
