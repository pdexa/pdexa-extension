// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"
#include "pdexa-ext/ginkgo/backend/cuda_hip/components/reduction.hpp"

#include <thrust/functional.h>

namespace gko::kernels::GKO_DEVICE_NAMESPACE::batch_template::batch_single_kernels {

template<typename ValueType, typename Mapping>
__device__ __forceinline__ void scale(const batch::multi_vector::batch_item<const ValueType>& alpha,
                                      const batch::multi_vector::batch_item<ValueType>& x,
                                      Mapping map) {
  const int max_li = x.num_rows * x.num_rhs;
  for (auto li = static_cast<int>(threadIdx.x); li < max_li; li += static_cast<int>(blockDim.x)) {
    const int row = li / x.num_rhs;
    const int col = li % x.num_rhs;

    x[row * x.stride + col] = alpha[map(row, col, alpha.stride)] * x[row * x.stride + col];
  }
}

template<typename ValueType, typename Mapping>
__device__ __forceinline__ void add_scaled(const batch::multi_vector::batch_item<const ValueType>& alpha,
                                           const batch::multi_vector::batch_item<const ValueType>& x,
                                           const batch::multi_vector::batch_item<ValueType>& y,
                                           Mapping map) {
  const int max_li = x.num_rows * x.num_rhs;
  for (auto li = static_cast<int>(threadIdx.x); li < max_li; li += static_cast<int>(blockDim.x)) {
    const int row = li / x.num_rhs;
    const int col = li % x.num_rhs;

    y[row * y.stride + col] += alpha[map(col)] * x[row * x.stride + col];
  }
}

template<typename Group, typename ValueType>
__device__ __forceinline__ void single_rhs_compute_conj_dot(Group subgroup,
                                                            const int num_rows,
                                                            const batch::multi_vector::batch_item<const ValueType> x,
                                                            const batch::multi_vector::batch_item<const ValueType> y,
                                                            ValueType& result)

{
  ValueType val = zero<ValueType>();
  for (int r = static_cast<int>(subgroup.thread_rank()); r < num_rows; r += static_cast<int>(subgroup.size())) { val += conj(x.values[r]) * y.values[r]; }

  // subgroup level reduction
  val = reduce(subgroup, val, thrust::plus<ValueType>{});

  if (subgroup.thread_rank() == 0) { result = val; }
}

template<typename Group, typename ValueType>
__device__ __forceinline__ void single_rhs_compute_norm2(Group subgroup,
                                                         const int num_rows,
                                                         const batch::multi_vector::batch_item<const ValueType> x,
                                                         remove_complex<ValueType>& result) {
  using real_type = remove_complex<ValueType>;
  real_type val = zero<real_type>();

  for (int r = static_cast<int>(subgroup.thread_rank()); r < num_rows; r += static_cast<int>(subgroup.size())) { val += squared_norm(x.values[r]); }

  // subgroup level reduction
  val = reduce(subgroup, val, thrust::plus<remove_complex<ValueType>>{});

  if (subgroup.thread_rank() == 0) { result = sqrt(val); }
}

template<typename ValueType>
__device__ __forceinline__ void single_rhs_copy(const int num_rows,
                                                const batch::multi_vector::batch_item<const ValueType> in,
                                                batch::multi_vector::batch_item<ValueType> out) {
  for (auto iz = static_cast<int>(threadIdx.x); iz < num_rows; iz += static_cast<int>(blockDim.x)) { out.values[iz] = in.values[iz]; }
}

} // namespace gko::kernels::GKO_DEVICE_NAMESPACE::batch_template::batch_single_kernels
