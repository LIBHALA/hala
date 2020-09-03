#ifndef __HALA_ROCM_WRAPPERS_HPP
#define __HALA_ROCM_WRAPPERS_HPP
/*
 * Code Author: Miroslav Stoyanov
 *
 * Copyright (C) 2018  Miroslav Stoyanov
 *
 * This file is part of
 * Hardware Accelerated Linear Algebra (HALA)
 *
 */

#ifdef HALA_ENABLE_CUDA
    #error "Cannot include both cuda and rocm!"
#endif

#define HALA_ENABLE_ROCM
#include "hala_gpu.hpp" // this is actually supposed to work

/*!
 * \file hala_rocm.hpp
 * \brief HALA wrapper for ROCM functions
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 * \ingroup HALAROCM
 *
 * Includes all HALA-ROCM wrapper templates.
 */

/*!
 * \defgroup HALAROCM HALA ROCM Wrapper Templates
 *
 * \par ROCM Wrapper Templates
 * Are designed to provide a unified API between AMD ROCM infrastructure and Nvidia CUDA.
 * Preprocessor macros are used to select between the CUDA and ROCM and to call the
 * corresponding backend function.
 *
 */

namespace hala{}

#endif
