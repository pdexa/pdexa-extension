// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#if PDEXA_EXT_ENABLE_CUDA && __CUDACC__
#include "../cuda_hip/batch_user_linop.hpp"
#endif

#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"


namespace gko::kernels::cuda::batch_template::batch_user {

template<typename ValueType, typename UserOpView>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const UserOpView mat,
           batch::multi_vector::uniform_batch<const ValueType> b,
           batch::multi_vector::uniform_batch<ValueType> x) {
#if PDEXA_EXT_ENABLE_CUDA && __CUDACC__
  auto num_rows = mat.num_rows;

  apply_kernel<<<mat.num_batch_items, get_num_threads_per_block(num_rows), 0, exec->get_stream()>>>(mat, b, x);
#else
  GKO_NOT_IMPLEMENTED;
#endif
}

} // namespace gko::kernels::cuda::batch_template::batch_user
