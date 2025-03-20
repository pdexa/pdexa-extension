// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_COOPERATIVE_GROUPS_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_COOPERATIVE_GROUPS_HPP_


#if defined(PDEXA_EXT_ENABLE_CUDA)
#include "pdexa-ext/ginkgo/backend/cuda/components/cooperative_groups.cuh"
#elif defined(PDEXA_EXT_ENABLE_HIP)
#include "pdexa-ext/ginkgo/backend/hip/components/cooperative_groups.hip.hpp"
#else
#error "Executor definition missing"
#endif


#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_COOPERATIVE_GROUPS_HPP_
