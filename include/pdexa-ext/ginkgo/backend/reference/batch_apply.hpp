// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "batch_multi_vector_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/kernel_tags.hpp"
#include "pdexa-ext/ginkgo/backend/reference/batch_csr_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/reference/batch_identity_kernels.hpp"
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

  template<typename ValueType>
  void operator()(batch_preconditioner::Identity<ValueType> a,
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

template<typename ValueType, typename IndexType>
void compute_residual(const batch::matrix::csr::batch_item<const ValueType, IndexType> a,
                      const batch::multi_vector::batch_item<const ValueType> x,
                      const batch::multi_vector::batch_item<const ValueType> b,
                      batch::multi_vector::batch_item<ValueType> r) {
  copy_kernel(b, r);
  advanced_apply_impl(-one<ValueType>(), a, x, one<ValueType>(), r);
}

template<typename T, typename ValueType>
void compute_residual(const T a,
                      const batch::multi_vector::batch_item<const ValueType> x,
                      const batch::multi_vector::batch_item<const ValueType> b,
                      batch::multi_vector::batch_item<ValueType> r) {
  // r = A*x
  simple_apply(a, batch::to_const(x), r);
  // r *= -1
  auto neg_one_v = -one<ValueType>();
  scale_kernel(batch::multi_vector::batch_item<const ValueType>{&neg_one_v, 1, 1, 1}, r);
  // r = r + b
  auto one_v = one<ValueType>();
  add_scaled_kernel(batch::multi_vector::batch_item<const ValueType>{&one_v, 1, 1, 1}, b, r);
}

} // namespace gko::kernels::reference::batch_template::batch_single_kernels
