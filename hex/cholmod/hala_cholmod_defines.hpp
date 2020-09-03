#ifndef __HALA_CHOLMOD_DEFINES_HPP
#define __HALA_CHOLMOD_DEFINES_HPP
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
 * \file hala_cholmod_defines.hpp
 * \brief HALA mockup header for Cholmod functions.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALACHOLMOD
 *
 * Definitions that avoid directly calling Cholmod and
 * including the long list of convoluted types.
 * \endinternal
 */

#include "hala_wrapped_extensions.hpp"

#ifndef __HALA_DOXYGEN_SKIP // tell Doxygen to skip this section
// Mockup header
extern "C"{
#ifndef cholmod_start
    void cholmod_start(char*);
#endif
#ifndef cholmod_allocate_sparse
    void* cholmod_allocate_sparse(int, int, int, int, int, int, int, char*);
#endif
#ifndef cholmod_analyze
    void* cholmod_analyze(void*, char*);
#endif
#ifndef cholmod_factorize
    void cholmod_factorize(void*, void*, char*);
#endif
#ifndef cholmod_allocate_dense
    void* cholmod_allocate_dense(int, int, int, int, char*);
#endif
#ifndef cholmod_solve
    void* cholmod_solve(int, void*, void*, char*);
#endif
#ifndef cholmod_sdmult
    void cholmod_sdmult(void*, int, void*, void*, void*, void*, char*);
#endif
#ifndef cholmod_free_dense
    void cholmod_free_dense(void**, char*);
#endif
#ifndef cholmod_free_sparse
    void cholmod_free_sparse(void**, char*);
#endif
#ifndef cholmod_free_factor
    void cholmod_free_factor(void**, char*);
#endif
#ifndef cholmod_finish
    void cholmod_finish(char*);
#endif
}

namespace hala{
namespace cheat_cholmod{
    // Mimics the cholmod sparse data-structure, allows read/write access.
    struct struct_sparse{
        size_t a, b, c;
        void *p, *i, *nnz, *x;
    };
    // Mimics the cholmod dense data-structure, allows read/write access.
    struct struct_dense{
        size_t a, b, c, d;
        void *x;
    };
    // Returns the internal array of the cholmod dense structure.
    template<typename T>
    T* get_dense_x(void *data){
        return reinterpret_cast<T*>(reinterpret_cast<struct_dense*>(data)->x);
    }
    // Return the internal array to the pntr of the sparse cholmod data structure.
    inline int* get_sparse_pntr(void *data){
        return reinterpret_cast<int*>(reinterpret_cast<struct_sparse*>(data)->p);
    }
    // Return the internal array to the indx of the sparse cholmod data structure.
    inline int* get_sparse_indx(void *data){
        return reinterpret_cast<int*>(reinterpret_cast<struct_sparse*>(data)->i);
    }
    // Return the internal array to the vals of the sparse cholmod data structure.
    template<typename T>
    T* get_sparse_vals(void *data){
        return reinterpret_cast<T*>(reinterpret_cast<struct_sparse*>(data)->x);
    }
}
}
#endif // __HALA_DOXYGEN_SKIP

#endif
