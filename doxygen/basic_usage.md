# Basic Usage

HALA consists of C++ headers only and heavily uses templates. The recommended use case for the library is through CMake install or `add_subdirectory()`; however, the individual modules can be included separately.

### CMake Usage

The core headers containing the BLAS, LAPACK, and sparse templates (both CPU and GPU versions) are accessed though the main header:
```
#include "hala.hpp"
```
The `hala.h` header is also included as an alias to `hala.hpp`. In addition those modules, HALA offers several extras that reside in separate headers to limit the clutter:

**The solvers** module provides implementation of several sparse linear solvers, use the header
```
#include "hala_solvers.hpp"
```
The solvers are implemented using the core functionality and offer overloads for the appropriate GPU backend. Currently, the provided solvers are the Gauss-Sciedel (GS), Conjugate-Gradient (CG), and General Minimum RESsidual (GMRES) with Arnoidi iteration. The two Krylov solvers also use the incomplete lower-upper (ILU) preconditioner.

**The Cholmod** module provides wrappers around the direct sparse solvers of the Cholmod library
```
#include "hala_cholmod.hpp" // note that this requires HALA_ENABLE_CHOLMOD=ON in CMake
```
Unlike the rest of HALA, the Cholmod wrappers work only in double precision real values and do not come with strictly GPU variants (although Cholmod may use an Nvidia GPU in the background). Single precision real algorithm or the GPU implementation is simply not provided. Nevertheless, Cholmod offers very efficient solvers that can compete with GPU accelerated CG and GMRES.

**Advanced vectorization extensions** are specialized instruction available in most CPU chips that allow for multiple floating point operations to be performed with a single instruction. Those extensions are particularly powerful when working with complex numbers where each addition or multiplication translates to multiple operation. HALA provides a module that implements packed variables that consist of several real or complex numbers an can be manipulated with regular arithmetic operators:
```
#include "hala_vex.hpp"
```
The wrappers use the 128-bit SSE2 and SSE3 instruction sets, the 256-bit AVX, and the 512-bit AXV512 standards.

Including all of HALA would look like this:
```
#include "hala.hpp"
#include "hala_solvers.hpp"
#include "hala_cholmod.hpp"
#include "hala_vex.hpp"
```
The extra modules are kept in separate headers to reduce potential conflicts and minimize repeated processing of long headers when those are included in multiple project files.


### Writing the Code

All templates reside in the namespace `hala`. Including the entire space is feasible since the most obvious conflict names use double prefix, e.g., `hala::cconj` and `hala::vcopy`, but this is probably not desirable since many of the short BLAS and LAPACK names can easily create conflicts:

* Should probably work:
```
#include "hala.hpp"
using namespace hala;

vcopy(x, y);
axpy(1.0, x, y);
```
* Preferred safer way:
```
#include "hala.hpp"

hala::vcopy(x, y);
hala::axpy(-1.0, x, y);
```

See the included examples for algorithms implemented using the HALA templates.


### Using Individual Modules without CMake

HALA consists of only headers and templates and CMake is used only to find library dependencies and to simplify management of the compiler flags. If the user handles manually the necessary includes and flags, the individual modules can be added one at a time. The modules should also be added in the correct order:
```
// everything uses the common, vex and wax modules
// everything needs <hala-source>/common/, <hala-source>/vex/ and <hala-source>/wax/
#include "hala_blas.hpp"                // needs <hala-source>/blas/
#include "hala_lapack.hpp"              // needs <hala-source>/lapack/
#include "hala_sparse.hpp"              // needs <hala-source>/sparse/

#define HALA_ENABLE_CUDA or HALA_ENABLE_ROCM
#include "hala_gpu.hpp"                 // needs <hala-source>/gpu/

#include "hala_wrapped_extensions.hpp"  // needs <hala-source>/wax/

// additional capabilities
#include "hala_cholmod.hpp"  // needs <hala-source>/hex/cholmod/
#include "hala_solvers.hpp"  // needs <hala-source>/hex/solvers/
#include "hala_vex.hpp"      // needs <hala-source>/hex/vex/
```
* `hala_gpu.hpp` requires either `HALA_ENABLE_CUDA` or `HALA_ENABLE_ROCM` defined prior to including the file
* `hala_wrapped_extensions.hpp` is equivalent to `hala.hpp`
