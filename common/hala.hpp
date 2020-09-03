#ifndef __HALA_HPP__
#define __HALA_HPP__
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
 * \ingroup HALA
 * \file hala.hpp
 * \brief HALA main header.
 * \author Miroslav Stoyanov
 * \copyright BSD-3-Clause
 *
 * \par Main Header
 * The only file needed to import all configured HALA templates.
*/

/*!
 * \defgroup HALA HALA Main Headers
 *
 * All HALA templates are accessible through either the \b hala.hpp header
 * or the alternative \b hala.h
 * \code
 *  #include <hala.hpp> // use either this
 *  #include <hala.h>   // or this
 * \endcode
 *
 * Note that the modules can also be included individually without CMake;
 * however, the user will be responsible for linking with the corresponding
 * set of libraries. See the installation instructions.
 */

#include "hala_config.hpp"

#include "hala_wrapped_extensions.hpp"

#endif
