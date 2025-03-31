// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "batch_multi_vector_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/kernel_tags.hpp"
#include "pdexa-ext/ginkgo/backend/reference/batch_csr_kernels.hpp"
#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"
#include "pdexa-ext/ginkgo/core/matrix/batch_struct.hpp"

namespace gko::kernels::reference::batch_template::batch_single_kernels {

struct simple_apply_fn {
  template<typename ValueType, typename IndexType>
  void operator()(const batch::matrix::csr::batch_item<const ValueType, IndexType> a,
                  const batch::multi_vector::batch_item<const ValueType> b,
                  batch::multi_vector::batch_item<ValueType> c) const {
    simple_apply_impl(a, b, c);
  }

  template<typename T, typename ValueType>
  void operator()(const T a,
                  const batch::multi_vector::batch_item<const ValueType> b,
                  batch::multi_vector::batch_item<ValueType> c) const {
    simple_apply(a, b, c, reference_kernel{});
  }
};

inline constexpr simple_apply_fn simple_apply{};


} // namespace gko::kernels::reference::batch_template::batch_single_kernels
