// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "pdexa-ext/ginkgo/core/preconditioner/batch_identity.hpp"

namespace gko::kernels::reference::batch_template::batch_single_kernels {
template<typename ValueType>
constexpr void simple_apply_impl(batch_preconditioner::Identity<ValueType> id,
                                                  batch::multi_vector::batch_item<const ValueType> r,
                                                  batch::multi_vector::batch_item<ValueType> z) {
  for (int i = 0; i < r.num_rows; ++i) { z.values[i * z.stride] = r.values[i * r.stride]; }
}
} // namespace gko::kernels::GKO_DEVICE_NAMESPACE::batch_template::batch_single_kernels
