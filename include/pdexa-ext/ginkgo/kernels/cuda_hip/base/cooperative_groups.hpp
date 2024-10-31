// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#if defined(GKO_COMPILING_CUDA)
#include "../../cuda/base/cooperative_groups.cuh"
#elif defined(GKO_COMPILING_HIP)
#include "../../hip/base/cooperative_groups.hip.hpp"
#else
#error "Executor definition missing"
#endif
