// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <pdexa-ext/ginkgo/kernels/base/batch_struct.hpp>

namespace gko {
namespace batch {
template<typename ValueType>
constexpr multi_vector::uniform_batch<ValueType> create_view(MultiVector<ValueType>* x) {
  using index_type = typename MultiVector<ValueType>::index_type;
  return {
    x->get_values(),
    x->get_num_batch_items(),
    static_cast<index_type>(x->get_common_size()[1]),
    static_cast<index_type>(x->get_common_size()[0]),
    static_cast<index_type>(x->get_common_size()[1])
  };
}


template<typename ValueType>
constexpr multi_vector::uniform_batch<const ValueType> create_view(const MultiVector<ValueType>* x) {
  using index_type = typename MultiVector<ValueType>::index_type;
  return {
    x->get_const_values(),
    x->get_num_batch_items(),
    static_cast<index_type>(x->get_common_size()[1]),
    static_cast<index_type>(x->get_common_size()[0]),
    static_cast<index_type>(x->get_common_size()[1])
  };
}
} // namespace batch
} // namespace gko
