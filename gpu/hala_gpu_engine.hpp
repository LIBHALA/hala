#ifndef __HALA_GPU_ENGINE_HPP
#define __HALA_GPU_ENGINE_HPP
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
 * \file hala_gpu_engine.hpp
 * \brief HALA gpu_engine.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAGPUENGINE
 *
 * \endinternal
*/

#include "hala_gpu_vector.hpp"

namespace hala{

/*!
 * \ingroup HALAGPU
 * \addtogroup HALAGPUENGINE Engine Class
 *
 * An engine class that indicates the use of the gpu libraries and provides a container
 * for the cuBlas, cuSparse, rocblas and rocsparse handles.
 */

/*!
 * \ingroup HALAGPUENGINE
 * \brief Core engine class, deals with handles and data transfer.
 *
 * The engine class is a wrapper around the cuBlas and cuSparse handles
 * and holds basic meta-data such as the associated GPU device ID.
 * The engine-type is used in template overloads to indicate the use
 * of GPU libraries and the engine provides basis I/O methods
 * for sending/receiving vectors, which is useful in making sure the
 * vectors are associated with the correct GPU device.
 */
struct gpu_engine{
private:
    /*!
     * \internal
     * \brief The id of the currently associated GPU device, inherited by all higher engines.
     *
     * \endinternal
     */
    int cgpu;

public:
    #ifdef HALA_ENABLE_CUDA
    //! \brief Default constructor, create handles on the device given by \b deviceid.
    gpu_engine(int deviceid = 0) : cgpu(deviceid),
        hcublas(make_cublas_handle(cgpu), cuda_deleter(ptr_ownership::own)),
        hcusparse(make_cusparse_handle(cgpu), cuda_deleter(ptr_ownership::own)),
        hcusolverdn(make_cusolverdn_handle(cgpu), cuda_deleter(ptr_ownership::own))
    {
        assert((cgpu >= 0) && (cgpu < gpu_device_count()));
    }
    //! \brief Take an alias of the handles without owning them.
    gpu_engine(int deviceid, cublasHandle_t extern_cublas, cusparseHandle_t extern_cusparse, cusolverDnHandle_t extern_cusolver) :
        cgpu(deviceid),
        hcublas(extern_cublas, cuda_deleter(ptr_ownership::not_own)),
        hcusparse(extern_cusparse, cuda_deleter(ptr_ownership::not_own)),
        hcusolverdn(extern_cusolver, cuda_deleter(ptr_ownership::not_own))
    {
        assert((cgpu >= 0) && (cgpu < gpu_device_count()));
    }
    //! \brief Create handles on the device given by \b deviceid and associates with the \b streamid.
    gpu_engine(cudaStream_t streamid, int deviceid) : gpu_engine(deviceid){ set_stream(streamid); }

    //! \brief Copy with non-owning references.
    gpu_engine(gpu_engine const &other) : gpu_engine(other.cgpu, other, other, other){}

    //! \brief Synchronizes with the device.
    void synchronize() const{ cudaDeviceSynchronize(); }

    //! \brief Set cuda stream for this engine.
    void set_stream(cudaStream_t streamid) const{
        check_cuda( cublasSetStream(hcublas.get(), streamid), "cublasSetStream()" );
        check_cuda( cusparseSetStream(hcusparse.get(), streamid), "cusparseSetStream()" );
        check_cuda( cusolverDnSetStream(hcusolverdn.get(), streamid), "cusolverDnSetStream()" );
    }

    //! \brief Set the current gpu device to the one used by the engine.
    void set_active_device() const{ cudaSetDevice(cgpu); }

    //! \brief Automatic return of the cuBlas handle.
    operator cublasHandle_t () const{ return hcublas.get(); }

    //! \brief Automatic return of the cuSparse handle.
    operator cusparseHandle_t () const{ return hcusparse.get(); }

    //! \brief Automatic return of the cuSolverDn handle.
    operator cusolverDnHandle_t () const{ return hcusolverdn.get(); }

    //! \brief Sets the cuBlas handle to use device pointers.
    void set_blas_device_pntr() const{ check_cuda( cublasSetPointerMode(hcublas.get(), CUBLAS_POINTER_MODE_DEVICE), "cublas set device pntr" ); }
    //! \brief Sets the cuBlas handle to use device pointers.
    void reset_blas_device_pntr() const{ check_cuda( cublasSetPointerMode(hcublas.get(), CUBLAS_POINTER_MODE_HOST), "cublas set host pntr" ); }
    //! \brief Sets the cuBlas handle to use device pointers.
    void set_cusparse_device_pntr() const{ check_cuda( cusparseSetPointerMode(hcusparse.get(), CUSPARSE_POINTER_MODE_DEVICE), "cusparse set device pntr" ); }
    //! \brief Sets the cuBlas handle to use device pointers.
    void reset_cusparse_device_pntr() const{ check_cuda( cusparseSetPointerMode(hcusparse.get(), CUSPARSE_POINTER_MODE_HOST), "cusparse set host pntr" ); }
    //! \brief Returns the current status of the cuBlas pointer mode.
    cublasPointerMode_t get_blas_pointer_mode() const{
        cublasPointerMode_t mode;
        check_cuda( cublasGetPointerMode(hcublas.get(), &mode), "cublasGetPointerMode()" );
        return mode;
    }

    /*!
     * \internal
     * \brief The handle used for all cuBlas calls.
     *
     * \endinternal
     */
    mutable std::unique_ptr<typename std::remove_pointer<cublasHandle_t>::type, cuda_deleter> hcublas;
    /*!
     * \internal
     * \brief The handle used for all cuSparse calls.
     *
     * \endinternal
     */
    mutable std::unique_ptr<typename std::remove_pointer<cusparseHandle_t>::type, cuda_deleter> hcusparse;
    /*!
     * \internal
     * \brief The handle used for all cuSolverDn calls.
     *
     * \endinternal
     */
    mutable std::unique_ptr<typename std::remove_pointer<cusolverDnHandle_t>::type, cuda_deleter> hcusolverdn;

    static cublasHandle_t make_cublas_handle(int gpu_device){
        cudaSetDevice(gpu_device);
        cublasHandle_t tcublas;
        check_cuda( cublasCreate(&tcublas), "cublasCreate()" );
        check_cuda( cublasSetPointerMode(tcublas, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode()" );
        return tcublas;
    }

    static cusparseHandle_t make_cusparse_handle(int gpu_device){
        cudaSetDevice(gpu_device);
        cusparseHandle_t tcusparse;
        check_cuda( cusparseCreate(&tcusparse), "cusparseCreate()" );
        check_cuda( cusparseSetPointerMode(tcusparse, CUSPARSE_POINTER_MODE_HOST), "cusparseSetPointerMode()" );
        return tcusparse;
    }

    static cusolverDnHandle_t make_cusolverdn_handle(int gpu_device){
        cudaSetDevice(gpu_device);
        cusolverDnHandle_t tcusolverdn;
        check_cuda( cusolverDnCreate(&tcusolverdn), "cusolverDnCreate()" );
        return tcusolverdn;
    }
    #endif

    #ifdef HALA_ENABLE_ROCM
    //! \brief ROCM overload for default engine constructor.
    gpu_engine(int deviceid = 0) : cgpu(deviceid),
        hrocblas(make_rocblas_handle(cgpu), rocm_deleter(ptr_ownership::own)),
        hrocsparse(make_rocsparse_handle(cgpu), rocm_deleter(ptr_ownership::own))
    {
        assert((cgpu >= 0) && (cgpu < gpu_device_count()));
    }

    //! \brief Take control of the handles without owning them.
    gpu_engine(int deviceid, rocblas_handle extern_rocblas, rocsparse_handle extern_rocsparse) :
        cgpu(deviceid),
        hrocblas(extern_rocblas, rocm_deleter(ptr_ownership::not_own)),
        hrocsparse(extern_rocsparse, rocm_deleter(ptr_ownership::not_own))
    {
        assert((cgpu >= 0) && (cgpu < gpu_device_count()));
    }

    //! \brief Copy with non-owning references.
    gpu_engine(gpu_engine const &other) : gpu_engine(other.cgpu, other, other){}

    //! \brief Automatic return of the rocBlas handle.
    operator rocblas_handle () const{ return hrocblas.get(); }
    //! \brief Automatic return of the rocSparse handle.
    operator rocsparse_handle () const{ return hrocsparse.get(); }

    void set_active_device() const{ check_rocm( hipSetDevice(cgpu), "hipSetDevice()"); }

    void set_blas_device_pntr() const{ check_rocm( rocblas_set_pointer_mode(hrocblas.get(), rocblas_pointer_mode_device), "rocblas_set_pointer_device()" ); }
    void reset_blas_device_pntr() const{ check_rocm( rocblas_set_pointer_mode(hrocblas.get(), rocblas_pointer_mode_host), "rocblas_set_pointer_device()" ); }
    void set_cusparse_device_pntr() const{ rocm_ignore(0); }
    void reset_cusparse_device_pntr() const{ rocm_ignore(0); }

    rocblas_pointer_mode get_blas_pointer_mode() const{
        rocblas_pointer_mode mode;
        check_rocm( rocblas_get_pointer_mode(hrocblas.get(), &mode), "rocblas_get_pointer_mode()" );
        return mode;
    }

    /*!
     * \internal
     * \brief The handle used for all rocBlas calls, inherited by all higher engines.
     *
     * \endinternal
     */
    mutable std::unique_ptr<typename std::remove_pointer<rocblas_handle>::type, rocm_deleter> hrocblas;
    /*!
     * \internal
     * \brief The handle used for all rocSparse calls, inherited by all higher engines.
     *
     * \endinternal
     */
    mutable std::unique_ptr<typename std::remove_pointer<rocsparse_handle>::type, rocm_deleter> hrocsparse;

    //! \brief Make a rocBlas handle.
    static rocblas_handle make_rocblas_handle(int device){
        check_rocm( hipSetDevice(device), "hipSetDevice()");
        rocblas_handle trocblas;
        check_rocm( rocblas_create_handle(&trocblas), "rocblas_create_handle()" );
        check_rocm( rocblas_set_pointer_mode(trocblas, rocblas_pointer_mode_host), "rocblas_set_pointer_mode()" );
        return trocblas;
    }

    //! \brief Make a rocSparse handle.
    static rocsparse_handle make_rocsparse_handle(int device){
        check_rocm( hipSetDevice(device), "hipSetDevice()");
        rocsparse_handle trocsparse;
        check_rocm( rocsparse_create_handle(&trocsparse), "rocsparse_create_handle()"  );
        check_rocm( rocsparse_set_pointer_mode(trocsparse, rocsparse_pointer_mode_host), "rocsparse_set_pointer_mode()" );
        return trocsparse;
    }
    #endif

    //! \brief Move initialize.
    gpu_engine(gpu_engine &&other) = default;

    //! \brief Move another class into this one.
    gpu_engine& operator =(gpu_engine &&other) = default;

    //! \brief Cannot copy into by creating non-owning aliases.
    gpu_engine& operator =(gpu_engine const &other){
        *this = gpu_engine(other);
        return *this;
    }

    //! \brief Return the device id associated with the engine.
    int device() const{ return cgpu; }

    //! \brief Return a hala::gpu_vector associated with the current device and loaded with the cpu data.
    template<class VectorLike>
    auto load(const VectorLike &cpu_data) const{ return make_gpu_vector(cpu_data, cgpu); }

    //! \brief Return a hala::gpu_vector associated with the current device and loaded with the first size entries of the cpu data.
    template<class VectorLike>
    auto load(const VectorLike &cpu_data, size_t size) const{
        assert( check_size(cpu_data, size) );
        return make_gpu_vector(wrap_array(get_data(cpu_data), size), cgpu);
    }

    //! \brief Return a \b std::vector with the data of the gpu vector.
    template<class VectorLike>
    auto unload(const VectorLike &gpu_data) const{
        cpu_engine e;
        using standard_type = get_standard_type<VectorLike>;
        auto result = new_vector(e, std::vector<standard_type>());
        force_size(get_size(gpu_data), result);
        gpu_copy_n<copy_direction::device2host>(get_data(gpu_data), get_size(gpu_data), get_data(result));
        return result;
    }

    //! \brief Return a hala::gpu_vector associated with the current device, with the specified number of entries, and filled with the value.
    template<typename T>
    auto vector(size_t num_entries, T value) const{
        gpu_vector<T> x(num_entries, cgpu);
        x.fill(value);
        return x;
    }

    //! \brief Return a default vector type with a copy of \b x (note that \b x may sit on another device if device IDs for the vectors are known).
    template<class vec> auto vcopy(vec const &x) const;

    //! \brief Wraps an array located on the gpu device associated with this engine; see hala::wrap_gpu_array()
    template<typename T>
    auto wrap_array(T *array, size_t num_entries) const{ return wrap_gpu_array(array, num_entries); }

    //! \brief Default vector is hala::gpu_vector
    template<typename T> using dfvector = typename hala::gpu_vector<T>;

    //! \brief Assert that \b v1.gpu() matches \b this.gpu(), terminates the variadric recursion.
    template<class one_gpu_vec>
    void check_gpu(one_gpu_vec const &x) const{
        int vector_device = get_device(x);
        if (vector_device > -1) // if the vector has no way of reporting the device, then ignore the check
            assert(cgpu == vector_device);
    }

    //! \brief Recursively asserts that the devices of all vectors match the device of this engine.
    template<class first_gpu_vec, class second_gpu_vec, class... other_gpu_vecs>
    void check_gpu(first_gpu_vec const &x, second_gpu_vec const &y, other_gpu_vecs const &... other) const{
        check_gpu(x);
        check_gpu(y, other...);
    }
};

/*!
 * \ingroup HALAGPUENGINE
 * \brief Specialization that assigns the device ID to the hala::gpu_vector.
 *
 * By default he hala::gpu_vector() uses device 0, this specialization copies the device from the gpu_engine.
 */
template<typename T> struct vector_constructor<gpu_vector<T>>{
    //! \brief Optionally, the vector could use information from the compute engine, e.g., device ID.
    static auto make_one(gpu_engine const &engine){ return hala::gpu_vector<T>(engine.device()); }
};

/*!
 * \ingroup HALAGPUENGINE
 * \brief Returns a default vector with the default constructor.
 *
 * Uses both hala::define_vdefault and hala::vector_constructor to make a default (probably) empty vector.
 */
template<class VectorLike>
auto new_vector(gpu_engine const &engine, VectorLike const &){
    return vector_constructor<get_vdefault<gpu_engine, VectorLike>>::make_one(engine);
}

/*!
 * \ingroup HALAGPUENGINE
 * \brief Overwrite the first \b num_entries of the vector with zeros.
 */
template<class VectorLike>
void set_zero(gpu_engine const &engine, size_t num_entries, VectorLike &&x){
    if (num_entries == 0) return;
    check_set_size(assume_output, x, num_entries);
    engine.set_active_device();

    using standard_type = get_standard_type<VectorLike>;
    auto data = get_standard_data(x);

    standard_type value = get_cast<standard_type>(0.0);
    gpu_copy_n<copy_direction::host2device>(&value, 1, data);

    size_t loaded = 1;
    while(loaded < num_entries){
        gpu_copy_n<copy_direction::device2device>(data, std::min(loaded, num_entries - loaded), data + loaded);
        loaded *= 2;
    }
}


/*!
 * \ingroup HALAGPUENGINE
 * \brief Creates an instance of hala::binded_gpu_vector using the device id of the engine.
 */
template<class VectorLike>
auto gpu_bind_vector(gpu_engine const &e, VectorLike &x){
    using standard_type = get_standard_type<VectorLike>;
    return binded_gpu_vector<standard_type, VectorLike>(e.device(), x);
}

/*!
 * \ingroup HALAGPUENGINE
 * \brief Creates an instance of hala::gpu_wrapped_array.
 */
template<typename ArrayType>
auto wrap_array(gpu_engine const &engine, ArrayType arr[], size_t num_entries){
    return engine.wrap_array(arr, num_entries);
}

/*!
 * \ingroup HALAGPUENGINE
 * \brief Allows expressively using scalar pointer on the GPU device, see hala::gpu_pntr.
 */
struct device_pntr{};

/*!
 * \ingroup HALAGPUENGINE
 * \brief Allows expressively using scalar pointer on the Host (CPU) device, see hala::gpu_pntr.
 */
struct host_pntr{};

/*!
 * \ingroup HALAGPUENGINE
 * Allows for RAII style of management for temporarily setting the device pointer mode.
 *
 * Instantiated with either hala::device_pntr or hala::host_pntr, the struct will set the corresponding
 * pointer type to a gpu engine and reset the original type when destroyed.
 *
 * The default pointer mode is "host pointer" which allows for scalars to be passed by value.
 * Suppose we are working with the default mode, but want to make one call to a different mode
 * One option is to use gpu_engine::set_cublas_device_pntr() and gpu_engine::reset_cublas_device_pntr()
 * But this requires two function calls and it is easy to forget to reset which will cause a segfault error in a future call.
 * Furthermore, sometimes the state of the engine is not known which would require even more code
 * to check and reset the correct status.
 * A temporary state can be set using a single expressive call.
 *
 * Example:
 * \code
 *  // Working in host or unknown pointer mode
 *  { // want to make one call using alpha and beta on the device
 *      hala::gpu_pntr<hala::device_pntr> hold(engine);
 *      hala::gemm(engine, ..., alpha, ..., beta, ...);
 *  } // at the bracket, the hold object will be destroyed and the pointer mode reset
 * \endcode
 */
template<typename pntr_mode>
struct gpu_pntr{
    //! \brief Constructor, sets the device pointer.
    explicit gpu_pntr(gpu_engine const &engine) : eng(engine), original_mode(engine.get_blas_pointer_mode()){
        static_assert(std::is_same<pntr_mode, device_pntr>::value || std::is_same<pntr_mode, host_pntr>::value,
                      "gpu_pntr can be used only with device_pntr and host_pntr pntr_mode types");
        if (std::is_same<pntr_mode, device_pntr>::value){
            eng.set_blas_device_pntr();
        }else{
            eng.reset_blas_device_pntr();
        }
    }
    //! \brief Destructor, reset the pointer.
    ~gpu_pntr(){
        #ifdef HALA_ENABLE_CUDA
        if (original_mode == CUBLAS_POINTER_MODE_DEVICE)
            eng.set_cusparse_device_pntr();
        else
            eng.reset_blas_device_pntr();
        #endif
        #ifdef HALA_ENABLE_ROCM
        if (original_mode == rocblas_pointer_mode_device)
            eng.set_cusparse_device_pntr();
        else
            eng.reset_blas_device_pntr();
        #endif
    }

private:
    gpu_engine const &eng;
    #ifdef HALA_ENABLE_CUDA
    cublasPointerMode_t original_mode;
    #endif
    #ifdef HALA_ENABLE_ROCM
    rocblas_pointer_mode original_mode;
    #endif
};

}

#endif
