// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
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

    x.at(row, col) = alpha[map(row, col, alpha.stride)] * x.at(row, col);
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

    y.at(row, col) += alpha[map(col)] * x.at(row, col);
  }
}

template<typename Group, typename ValueType>
__device__ __forceinline__ void single_rhs_compute_conj_dot(Group subgroup,
                                                            const int num_rows,
                                                            const batch::multi_vector::batch_item<const ValueType> x,
                                                            const batch::multi_vector::batch_item<const ValueType> y,
                                                            xstd::type_identity_t<ValueType>& result)

{
  ValueType val = zero<ValueType>();
  for (int r = static_cast<int>(subgroup.thread_rank()); r < num_rows; r += static_cast<int>(subgroup.size())) { val += conj(x.values[r]) * y.values[r]; }

  // subgroup level reduction
  val = reduce(subgroup, val, thrust::plus<ValueType>{});

  if (subgroup.thread_rank() == 0) { result = val; }
}

template<int n_rhs, typename Group, typename ValueType>
__device__ __forceinline__ void compute_conj_dot(Group group,
                                                 batch::multi_vector::batch_item<const ValueType> x,
                                                 batch::multi_vector::batch_item<const ValueType> y,
                                                 xstd::type_identity_t<ValueType>* result) {
  auto subgroup = group::tiled_partition<n_rhs>(group);

  for (auto col = subgroup.thread_rank(); col < x.num_rhs; col += subgroup.num_threads()) {
    ValueType val = zero<ValueType>();

    for (auto row = subgroup.meta_group_rank(); row < x.num_rows; row += subgroup.meta_group_size()) {
      val += conj(x.at(row, col)) * y.at(row, col);
    }

    // accumulate between all subwarps in the warp
#pragma unroll
    for (unsigned i = subgroup.num_threads(); i < group.num_threads(); i *= 2) {
      // use XOR to reduce along pair-wise groups with distance i, e.g. for a subgroup size of 4 and a group size of 32:
      // i=4  [0..3] + [4..7], [8..11] + [12..15], ... + [28..31]
      // i=8  [0..3] + [8..11], [16..19] + [24..27]
      // i=16 [0..3] + [16..19]
      val = val + group.shfl_xor(val, i);
    }

    if (subgroup.meta_group_rank() == 0) { result[col] = val; }
  }
}

template<typename Group, typename ValueType>
__device__ __forceinline__ void single_rhs_compute_norm2(Group subgroup,
                                                         const int num_rows,
                                                         const batch::multi_vector::batch_item<const ValueType> x,
                                                         xstd::type_identity_t<remove_complex<ValueType>>& result) {
  using real_type = remove_complex<ValueType>;
  real_type val = zero<real_type>();

  for (int r = static_cast<int>(subgroup.thread_rank()); r < num_rows; r += static_cast<int>(subgroup.size())) { val += squared_norm(x.values[r]); }

  // subgroup level reduction
  val = reduce(subgroup, val, thrust::plus<remove_complex<ValueType>>{});

  if (subgroup.thread_rank() == 0) { result = sqrt(val); }
}

template<int n_rhs, typename Group, typename ValueType>
__device__ __forceinline__ void compute_norm2(Group group,
                                              batch::multi_vector::batch_item<const ValueType> x,
                                              xstd::type_identity_t<remove_complex<ValueType>>* result) {
  using real_type = remove_complex<ValueType>;

  auto subgroup = group::tiled_partition<n_rhs>(group);

  for (auto col = subgroup.thread_rank(); col < x.num_rhs; col += subgroup.num_threads()) {
    real_type val = zero<real_type>();

    for (auto row = subgroup.meta_group_rank(); row < x.num_rows; row += subgroup.meta_group_size()) {
      val += squared_norm(x.at(row, col));
    }

    // accumulate between all subwarps in the warp
#pragma unroll
    for (unsigned i = subgroup.num_threads(); i < group.num_threads(); i *= 2) {
      // use XOR to reduce along pair-wise groups of size i
      val = val + group.shfl_xor(val, i);
    }

    if (subgroup.meta_group_rank() == 0) { result[col] = sqrt(val); }
  }
}

template<typename ValueType>
__device__ __forceinline__ void single_rhs_copy(
                                                const batch::multi_vector::batch_item<const ValueType> in,
                                                batch::multi_vector::batch_item<ValueType> out) {
  assert(in.num_rows == out.num_rows);
  for (auto iz = static_cast<int>(threadIdx.x); iz < in.num_rows; iz += static_cast<int>(blockDim.x)) { out.values[iz] = in.values[iz]; }
}


template<typename ValueType>
__device__ __forceinline__ void copy(const batch::multi_vector::batch_item<const ValueType> in,
                                     batch::multi_vector::batch_item<ValueType> out) {
  assert(in.num_rows == out.num_rows && in.num_rhs == out.num_rhs);
  for (auto li = static_cast<int>(threadIdx.x); li < in.num_rows * in.num_rhs; li += static_cast<int>(blockDim.x)) {
    auto row = li / in.num_rhs;
    auto col = li % in.num_rhs;
    out.at(row, col) = in.at(row, col);
  }
}


} // namespace gko::kernels::GKO_DEVICE_NAMESPACE::batch_template::batch_single_kernels
