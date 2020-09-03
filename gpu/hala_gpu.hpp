#ifndef __HALA_GPU_WRAPPERS_HPP
#define __HALA_GPU_WRAPPERS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#include "hala_gpu_overloads.hpp"

/*!
 * \file hala_gpu.hpp
 * \brief HALA wrapper for GPU functions
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAGPU
 *
 * Includes all HALA-ROCM wrapper templates.
 */

/*!
 * \defgroup HALAGPU HALA CUDA and ROCM Wrapper Templates
 *
 * \par CUDA and ROCM
 * The two frameworks provided by cuBlas and rocblas provide functionality
 * that is very similar to each other and almost identical to BLAS.
 * The difference with BLAS comes from the need to keep an additional
 * handle (pointer to an opaque data-structure) and that the data of all
 * vector like classes resided in device memory.
 * HALA templates allow for the selection of one of the two and the
 * templates can be instantiated with the correct backend.
 * In addition, HALA provides the hala::gpu_wrapped_array, hala::gpu_vector,
 * and hala::gpu_engine classes for manipulating memory and the opaque handles.
 *
 */

#endif
