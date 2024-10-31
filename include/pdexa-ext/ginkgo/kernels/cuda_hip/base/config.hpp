// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#if defined(GKO_COMPILING_CUDA)
#include "../../cuda/base/config.hpp"
#elif defined(GKO_COMPILING_HIP)
#include "../../hip/base/config.hip.hpp"
#else
#error "Executor definition missing"
#endif
