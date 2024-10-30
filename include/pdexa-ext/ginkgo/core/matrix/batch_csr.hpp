// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <pdexa-ext/ginkgo/kernels/base/batch_struct.hpp>

namespace gko {
namespace batch {
template<typename ValueType, typename IndexType>
constexpr matrix::csr::uniform_batch<ValueType, IndexType> create_view(matrix::Csr<ValueType, IndexType>* x) {
  return {
    x->get_const_values(),
    x->get_const_col_idxs(),
    x->get_const_row_ptrs(),
    x->get_num_batch_items(),
    static_cast<IndexType>(x->get_common_size()[0]),
    static_cast<IndexType>(x->get_common_size()[1]),
    static_cast<IndexType>(x->get_num_elements_per_item())
  };
}


template<typename ValueType, typename IndexType>
constexpr matrix::csr::uniform_batch<const ValueType, const IndexType> create_view(
  const matrix::Csr<ValueType, IndexType>* x) {
  return {
    x->get_const_values(),
    x->get_const_col_idxs(),
    x->get_const_row_ptrs(),
    x->get_num_batch_items(),
    static_cast<IndexType>(x->get_common_size()[0]),
    static_cast<IndexType>(x->get_common_size()[1]),
    static_cast<IndexType>(x->get_num_elements_per_item())
  };
}
} // namespace batch
} // namespace gko
