// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"
#include "pdexa-ext/ginkgo/core/matrix/batch_struct.hpp"

namespace gko::kernels::GKO_DEVICE_NAMESPACE::batch_template::batch_single_kernels {

template<typename ValueType, typename IndexType>
__device__ __forceinline__ void simple_apply_impl(const batch::matrix::csr::batch_item<const ValueType, IndexType>& mat,
                                                  const batch::multi_vector::batch_item<const ValueType> b,
                                                  batch::multi_vector::batch_item<ValueType> x) {
  const auto num_rows = mat.num_rows;
  const auto val = mat.values;
  const auto col = mat.col_idxs;
  for (auto row = static_cast<int>(threadIdx.x); row < num_rows; row += static_cast<int>(blockDim.x)) {
    auto temp = zero<ValueType>();
    for (auto nnz = mat.row_ptrs[row]; nnz < mat.row_ptrs[row + 1]; nnz++) {
      const auto col_idx = col[nnz];
      temp += val[nnz] * b.values[col_idx];
    }
    x.values[row] = temp;
  }
}

template<typename ValueType, typename IndexType>
__device__ __forceinline__ void
advanced_apply_impl(const ValueType alpha,
                    const batch::matrix::csr::batch_item<const ValueType, IndexType> mat,
                    const batch::multi_vector::batch_item<const ValueType> b,
                    const ValueType beta,
                    const batch::multi_vector::batch_item<ValueType> x) {
  const auto num_rows = mat.num_rows;
  const auto val = mat.values;
  const auto col = mat.col_idxs;
  for (auto row = static_cast<int>(threadIdx.x); row < num_rows; row += static_cast<int>(blockDim.x)) {
    auto temp = zero<ValueType>();
    for (auto nnz = mat.row_ptrs[row]; nnz < mat.row_ptrs[row + 1]; nnz++) {
      const auto col_idx = col[nnz];
      temp += alpha * val[nnz] * b.values[col_idx];
    }
    x.values[row] = temp + beta * x.values[row];
  }
}

} // namespace gko::kernels::GKO_DEVICE_NAMESPACE::batch_template::batch_single_kernels
