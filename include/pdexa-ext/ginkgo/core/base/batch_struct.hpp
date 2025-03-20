// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_BATCH_STRUCT_HPP_
#define GKO_CORE_BASE_BATCH_STRUCT_HPP_

#include <ginkgo/core/base/types.hpp>

namespace gko {
namespace batch {
namespace multi_vector {

/**
 * Encapsulates one matrix from a batch of multi-vectors.
 */
template<typename ValueType>
struct batch_item {
  using value_type = ValueType;
  ValueType* __restrict__ values;
  int32 stride;
  int32 num_rows;
  int32 num_rhs;
};

/**
 * A 'simple' structure to store a global uniform batch of multi-vectors.
 */
template<typename ValueType>
struct uniform_batch {
  using value_type = ValueType;
  using entry_type = batch_item<ValueType>;

  ValueType* __restrict__ values;
  size_type num_batch_items;
  int32 stride;
  int32 num_rows;
  int32 num_rhs;

  [[nodiscard]] size_type get_single_item_num_nnz() const { return static_cast<size_type>(stride * num_rows); }
};

} // namespace multi_vector

template<typename ValueType>
GKO_ATTRIBUTES GKO_INLINE multi_vector::batch_item<const ValueType>
to_const(const multi_vector::batch_item<ValueType>& b) {
  return {b.values, b.stride, b.num_rows, b.num_rhs};
}

template<typename ValueType>
GKO_ATTRIBUTES GKO_INLINE multi_vector::uniform_batch<const ValueType>
to_const(const multi_vector::uniform_batch<ValueType>& ub) {
  return {ub.values, ub.num_batch_items, ub.stride, ub.num_rows, ub.num_rhs};
}

/**
 * Generates an immutable uniform batch struct from a batch of multi-vectors.
 */
template<typename ValueType>
GKO_ATTRIBUTES GKO_INLINE multi_vector::uniform_batch<const ValueType>
get_batch_struct(const MultiVector<ValueType>* const op) {
  return {as_device_type(op->get_const_values()), op->get_num_batch_items(),
          static_cast<int32>(op->get_common_size()[1]), static_cast<int32>(op->get_common_size()[0]),
          static_cast<int32>(op->get_common_size()[1])};
}

/**
 * Generates a uniform batch struct from a batch of multi-vectors.
 */
template<typename ValueType>
GKO_ATTRIBUTES GKO_INLINE multi_vector::uniform_batch<ValueType> get_batch_struct(MultiVector<ValueType>* const op) {
  return {as_device_type(op->get_values()), op->get_num_batch_items(), static_cast<int32>(op->get_common_size()[1]),
          static_cast<int32>(op->get_common_size()[0]), static_cast<int32>(op->get_common_size()[1])};
}

} // namespace batch
} // namespace gko

#endif // GKO_CORE_BASE_BATCH_STRUCT_HPP_
