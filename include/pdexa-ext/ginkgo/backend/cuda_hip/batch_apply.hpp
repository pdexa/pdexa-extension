// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "pdexa-ext/ginkgo/backend/cuda_hip/batch_csr_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/cuda_hip/batch_identity_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/cuda_hip/batch_multi_vector_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/kernel_tags.hpp"
#include "pdexa-ext/ginkgo/core/matrix/batch_struct.hpp"

namespace gko::kernels::GKO_DEVICE_NAMESPACE::batch_template::batch_single_kernels {

struct simple_apply_fn {
  template<typename ValueType, typename IndexType>
  __device__ __forceinline__ void operator()(const batch::matrix::csr::batch_item<const ValueType, IndexType> a,
                                             const batch::multi_vector::batch_item<const ValueType> b,
                                             batch::multi_vector::batch_item<ValueType> c) const {
    simple_apply_impl(a, b, c);
  }

  template<typename ValueType>
  __device__ __forceinline__ void operator()(batch_preconditioner::Identity<ValueType> a,
                                             const batch::multi_vector::batch_item<const ValueType> b,
                                             batch::multi_vector::batch_item<ValueType> c) const {
    simple_apply_impl(a, b, c);
  }

  template<typename T, typename ValueType>
  __device__ __forceinline__ void operator()(const T a,
                                             const batch::multi_vector::batch_item<const ValueType> b,
                                             batch::multi_vector::batch_item<ValueType> c) const {
    simple_apply(a, b, c, device_kernel{});
  }
};

GKO_CPO_STORAGE constexpr simple_apply_fn simple_apply{};

template<typename ValueType, typename IndexType>
__device__ __forceinline__ void compute_residual(const batch::matrix::csr::batch_item<const ValueType, IndexType> a,
                                                 const batch::multi_vector::batch_item<const ValueType> x,
                                                 const batch::multi_vector::batch_item<const ValueType> b,
                                                 batch::multi_vector::batch_item<ValueType> r) {
  single_rhs_copy(b, r);
  advanced_apply_impl(-one<ValueType>(), a, x, one<ValueType>(), r);
}

template<typename T, typename ValueType>
__device__ __forceinline__ void compute_residual(const T a,
                                                 const batch::multi_vector::batch_item<const ValueType> x,
                                                 const batch::multi_vector::batch_item<const ValueType> b,
                                                 batch::multi_vector::batch_item<ValueType> r) {

  // r = A*x
  simple_apply(a, batch::to_const(x), r);
  // r *= -1
  auto neg_one_v = -one<ValueType>();
  scale(batch::multi_vector::batch_item<const ValueType>{&neg_one_v, 1, 1, 1}, r, [](auto...) {return 0;});
  // r = r + b
  auto one_v = one<ValueType>();
  add_scaled(batch::multi_vector::batch_item<const ValueType>{&one_v, 1, 1, 1}, b, r, [](auto...) {return 0;});
}


} // namespace gko::kernels::GKO_DEVICE_NAMESPACE::batch_template::batch_single_kernels
