# The Motivation behind HALA

The modern C++ programming language offers powerful techniques such generic template meta-programming, object oriented programming, and advanced functional programming with nice features such as overloads, default parameters, RAII resource management, and many others. The combination of these powerful features means that more and better code can be written and maintained faster and easier than ever before.

However, the field of high-performance scientific computing far predates such advances and ancient standards and inflexible interfaces written in older and even obsolete languages burden the field. Combined with quirks (and right-out flaws) in the C++ standard, this obstructive heritage hinders the progress and prevents developers from fully utilizing the advanced features of the C++ language.

The main objective behind HALA is to create a seamless and consistent interface between standards for accelerated linear algebra and C++ generic template programming.


### BLAS and LAPACK

The first linear algebra standards were written in Fortran77, and while at that time the language was far ahead of any competition (at least in terms of linear algebra), by modern standards Fortran77 is simply obsolete. Variables were always passed by reference and there was little (if any) type safety and const correctness. There were no method overloads and no default parameters either. From a C++ perspective the interface looks like:
```
float snrm2_(const int *N, const float *x, const int *incx);
double dnrm2_(const int *N, const double *x, const int *incx);
```
Using the same call for a C++ std::vector container entails:
```
std::vector<double> x (10, 1.0);
int xsize = (int) x.size();
int incx = 1;
double xnorm = dnrm_(&xsize, x.data(), &incx);
```
Note the need for the two extra variables and the redundancy in passing the size of a container as a separate parameter. Furthermore, the code will have to be refactored for std::vector if the type changes from double to float or std::complex.

The CBLAS interface offers some improvements, e.g.,
```
double xnorm = cblas_dnrm(x.size(), x.data(), 1);
```
however, there is still redundancy between the stride 1 and using the full vector size, and no overloads are available for different floating point types. In addition, the CBLAS interface uses either void pointers or custom complex type, while ABI compatible with std::complex the approach requires reinterpret-casts which is at best cumbersome and at worse vert type-unsafe. Both BLAS and CBLAS standard work with raw-arrays and do not provide basic sanity check on sizes, e.g., increasing the stride above will cause a segfault without a meaningful error message. And last, but not least, the CBLAS standard is considerably less ubiquitous than the Fortran BLAS API.

In addition to these issues, the LAPACK adds the problem of temporary workspaces. The Fortran77 language did not allow for dynamic memory allocation/deletion and all array sizes had to be decided at compile time; thus extra workspace had to be pre-determined with either a pen-and-paper calculation or a compile-and-test run ahead of the real work. For example, see [sgesvd_](http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html). Every modern programming language today provides dynamic memory management, but allocations and dealocations can be expensive and repeatedly assigning the same amount of memory for similar operations is undesirable in high-performance codes. Thus, the user is still often burdened with manual memory management and due to the lack of overloads that has to be done even in instances where the call is done only once.

HALA offers template wrappers that allow for calls of the form:
```
  std::vector<T> x(10, 1.0);
  auto xnorm = hala::norm2(x);
  ...
  hala::svd(columns, rows, matrix, singular_values);
```
In the example above, **T** can be real or complex with single (float) or double precision, the result **xnorm** will have the appropriate precision type and the size and incx variables will default to the total vector size.
* The hala::norm2() template will auto-detect all the needed information about the types, ensuring generality.
* The template will call the appropriate BLAS function, thus using the optimized implementation.
* The call will inline, ensuring no run-time overhead.
* The LAPACK method will handle all necessary workspace in the background, including appropriately resizing the output vector with the singular values.

In addition to the above:
* HALA works with a very broad and extensible list of containers and even raw-arrays.
* HALA can accept any complex number implementation that is ABI compatible with std::complex, regardless of the name attached to it.
* HALA generalizes the BLAS API to a near seamless inclusion of the GPU accelerated standards, e.g., cuBlas znd rocBlas.
See further down for details on those points.


### NVIDIA cuBlas and cuSparse

In recent years, the GPU accelerators have provided unprecedented performance boost to modern computing. The classical BLAS standards were ported, preserving both the functionality and the burdensome legacy, and the standards were extended to sparse linear algebra. However, the new capabilities come with new issues:
* The standards define new complex number variable types, ABI compatible but different in the eyes of the compiler.
* The new methods have C-style of naming conventions using prefixes such as `cublas` and `cusparse` and adding a mix of upper and lower case letter to indicate the algorithm and precision type.
* While the calling conventions mimic BLAS, all calls require special handle variables, e.g., `cublasHandle_t` which is an opaque pointer with neither const-correctness nor container allocate/free safety.
* The classic BLAS calls also deviate from the standard by allowing all scalar variables to be pointers allocated either on the GPU on CPU memory space.
* Memory space is segregated and the included thrust containers come with very heavy (and potentially code breaking) CUDA capabilities making them hard to include globally in mixed CPU-GPU code.
* The cuSparse standard requires redundant C-style of structs wrappers around different matrix types and have to be manually created/deleted.

For example, calling cuSparse matrix-vector multiply requires calls to allocate (later deallocate) `cusparseHandle_t`, compute the required buffer size with `cusparseSpMV_bufferSize`, allocate (later deallocate) the corresponding buffer, calling `cusparseSpMV` with a specific variable given on run-time that specifies the type. The corresponding HALA alternative encapsulates the handle in a class and uses the call:
```
  hala::gpu_engine engine;
  hala::sparse_gemv(engine, 1.0, pntr, indx, vals, x, 0.0, y);
```
The HALA answer to the above include:
* Templates that are fully consistent with the C++ standards and do not require any language extensions.
* Wrappers that encapsulate the handles and the C-style CUDA raw-arrays for RAII-style classes.
* Full interoperability between the CUDA complex numbers and std::complex templates.
* Overloads that allow for either on-CPU scalar variables or pointer on either the CPU or GPU (i.e., there is no loss of capability).
* Workspace allocation is handled automatically in the background without encumbering the user.

Furthermore, HALA provides overloads with a unified naming convention between cuBlas and BLAS, cuSparse and HALA sparse templates. The overloads distinguish between the backends either by the first variable, which is either a CPU or a GPU engine object, and by using vector containers bind to the backend hadnles. As a result, a single high level algorithm (e.g., a solver) can be implemented in terms of HALA templates and later instantiated with either backend by simply switching a single engine context.


### AMD RocBlas and RocSparse

AMD provides a similar standard to cuBlas and cuSparse, which comes with both the advantages and disadvantage outlined above. In addition, the Rocm framework is still relatively new with many methods lacking proper implementation or suffering from various bugs. Development is happening very rapidly and achieving compliance with the Rocm standards involves chasing a moving target, and while Rocm and CUDA are very similar, sometimes there are differences that can preclude writing a portable code. HALA templates are designed to smooth out the differences and provide a fully unified API.


### The Good, the Bad and the Ugly of the C++ Type System

The strong type system and type-safety is a very important design paradigm of C++ that allows for general and portable code through overloads and templates. In addition, the RAII style of resource management helps prevent many common bugs by encapsulating resources (e.g., memory) in objects that automatically free the resources whenever they leave scope. While all of these are great tools, they also come at a cost.

The C++ 2014 standard used by HALA offers two containers that wrap around contiguous arrays of variables and *know* the associated memory size (vector and valarray). In addition, there are numerous third-party containers, e.g., `thrust::host_vector`, that are used in different applications and often come with relevant advantages. The C++ 2020 standard is currently considering the addition of at least two new containers (span and mdarray). However,
* at the low level of implementation, these simply define a contiguous arrays of values
* at the higher level of linear algebra these are simply representations of matrices and vectors
* but the C++ language insists on treating them as separate entities

A similar problem exists with the numerous implementations of complex numbers, where practically all standards define a struct/class/union that is an encapsulation of two floats or doubles representing the real and complex components of the number (first real then complex). The different types are ABI-compatible and arrays of one type can be safely cast to another and even passed to BLAS calls, yet C++ insists that they are different. Furthermore, the C++ standard methods for handling complex operations lack the necessary overloads to allow for general code, for example consider:
'''
template<typename T> T dot(std::vector<T> const &x, std::vector<T> const &y){
    assert( x.size() == y.size() );
    T sum = 0.0;
    for(size_t i=0; i<x.size(); i++) sum += std::conj(x[i]) * y[i];
    return sum;
}
'''
The code above cannot be instantiated with a float or double since the std::conj() method always returns a complex number even if the input is real.

The HALA templates are generalized (or even over-generalized) to allow interoperability between different container types, e.g., a matrix stored in std::vector can be safely multiplied by a matrix stored in a std::valarray and the result can be written to a raw-array. The list of vector-like containers accepted by HALA is easily extensible by adding a few appropriate specializations in the user code, i.e., there is no need to intrude inside the HALA headers. The ability to work with raw-arrays also enables easy interfaces with other languages, e.g., C and Python, further extending the potential generality and portability.

HALA also provides overloaded methods such as hala::cconj() and hala::creal() that generalize to both real and complex numbers and can be applied to custom (ABI compatible) complex types, for example, in the code above we can use:
'''
    for(size_t i=0; i<x.size(); i++) sum += hala::cconj(x[i]) * y[i];
'''
The hala::cconj() will generalize to all real and complex numbers.


### Advanced Vector Extensions and Non-Linear Methods

Linear operations are usually at the core of math algorithms, but non-linear operations may also require significant computing resource. Modern CPUs have extended registers that can perform multiple floating point operations simultaneously which significantly improves performance; and C++ compilers can automatically take advantage of such register extensions, but the resulting optimization often falls short of fulfilling the true potential of the hardware. A good example of the problem is the complex number arithmetic where even a simple vector scaled by a constant can be an order of magnitude slower than a direct semi-manual vectorization:
```
    std::complex<float> alpha;
    std::vector<std::complex<float>> x;
    ... initialize alpha and x ...
    // standard code
    for(auto &v : x) v *= alpha;
    // hala alternative, assuming x.size() divides evenly into strides
    // otherwise the remainder has to be done manually
    auto mmalpha = hala::mmload(alpha);
    for(size_t i=0; i<x.size(); i += mmalpha.stride)
        hala::mmbind(&x[i]) *= mmalpha;
```
See the included benchmarks, in the vector scale case with complex numbers, even with the smallest sse registers, the manual vectorization is over 10x faster. The simple std::complex templates are too much of a challenge to the compiler optimization algorithms and more convoluted non-liner code often requires manual vectorization.

