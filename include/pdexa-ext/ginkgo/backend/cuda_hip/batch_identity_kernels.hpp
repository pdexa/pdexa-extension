// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "pdexa-ext/ginkgo/core/preconditioner/batch_identity.hpp"

namespace gko::kernels::GKO_DEVICE_NAMESPACE::batch_template::batch_single_kernels {
template<typename ValueType>
__device__ __forceinline__ void simple_apply_impl(batch_preconditioner::Identity<ValueType> id,
                                                  batch::multi_vector::batch_item<const ValueType> r,
                                                  batch::multi_vector::batch_item<ValueType> z) {
  const int max_li = r.num_rows * r.num_rhs;
  for (auto li = static_cast<int>(threadIdx.x); li < max_li; li += static_cast<int>(blockDim.x)) {
    const int row = li / r.num_rhs;
    const int col = li % r.num_rhs;

    z.at(row, col) = r.at(row, col);
  }
}
} // namespace gko::kernels::GKO_DEVICE_NAMESPACE::batch_template::batch_single_kernels
