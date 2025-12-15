// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_BATCH_STRUCT_HPP_
#define GKO_CORE_BASE_BATCH_STRUCT_HPP_

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>

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

  constexpr GKO_INLINE ValueType& operator[](int64 i) const {
    assert(i < num_rows * stride);
    return values[i];
  }

  constexpr GKO_INLINE ValueType& at(int64 i, int64 j) const {
    assert(i < num_rows && j < num_rhs);
    return values[i * stride + j];
  }
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

#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress 554
#endif
  operator uniform_batch<const ValueType>() const {
    return {values, num_batch_items, stride, num_rows, num_rhs};
  }
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_default 554
#endif

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
