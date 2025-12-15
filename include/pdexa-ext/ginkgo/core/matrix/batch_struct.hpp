// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_BATCH_STRUCT_HPP_
#define GKO_CORE_MATRIX_BATCH_STRUCT_HPP_

#include <ginkgo/core/base/types.hpp>
#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"
#include "pdexa-ext/ginkgo/core/preconditioner/batch_identity.hpp"

#if defined(PDEXA_EXT_ENABLE_CUDA) && __CUDACC__
#define GKO_CPO_STORAGE __device__
#else
#define GKO_CPO_STORAGE inline
#endif

namespace gko::batch {
namespace matrix {
namespace csr {

/**
 * Encapsulates one matrix from a batch of csr matrices.
 */
template<typename ValueType, typename IndexType = const int32>
struct batch_item {
  using value_type = ValueType;
  using index_type = IndexType;

  ValueType* __restrict__ values;
  index_type* __restrict__ col_idxs;
  index_type* __restrict__ row_ptrs;
  index_type num_rows;
  index_type num_cols;
  index_type num_nnz_per_item;

  [[nodiscard]] size_type get_single_item_num_nnz() const { return static_cast<size_type>(num_nnz_per_item); }
};

/**
 * A 'simple' structure to store a global uniform batch of csr matrices.
 */
template<typename ValueType, typename IndexType = const int32>
struct uniform_batch {
  using value_type = ValueType;
  using index_type = IndexType;
  using entry_type = batch_item<value_type, index_type>;

  ValueType* __restrict__ values;
  index_type* __restrict__ col_idxs;
  index_type* __restrict__ row_ptrs;
  size_type num_batch_items;
  index_type num_rows;
  index_type num_cols;
  index_type num_nnz_per_item;

  [[nodiscard]] size_type get_single_item_num_nnz() const { return static_cast<size_type>(num_nnz_per_item); }
};

} // namespace csr

namespace dense {

/**
 * Encapsulates one matrix from a batch of dense matrices.
 */
template<typename ValueType>
struct batch_item {
  using value_type = ValueType;
  value_type* values;
  int32 stride;
  int32 num_rows;
  int32 num_cols;

  [[nodiscard]] size_type get_single_item_num_nnz() const { return static_cast<size_type>(num_cols * num_rows); }

  constexpr value_type& operator()(int32 i, int32 j) const { return values[i * num_cols + j]; }
};

template<typename ValueType>
struct batch_item_colum_major {
  using value_type = ValueType;
  value_type* values;
  int32 stride;
  int32 num_rows;
  int32 num_cols;

  [[nodiscard]] size_type get_single_item_num_nnz() const { return static_cast<size_type>(num_cols * num_rows); }

  constexpr value_type& operator()(int32 i, int32 j) const { return values[i + j * num_rows]; }
};

/**
 * A 'simple' structure to store a global uniform batch of dense matrices.
 */
template<typename ValueType>
struct uniform_batch {
  using value_type = ValueType;
  using entry_type = batch_item<ValueType>;

  operator uniform_batch<const ValueType>() const {
    return {values, num_batch_items, stride, num_rows, num_cols};
  }

  ValueType* values;
  size_type num_batch_items;
  int32 stride;
  int32 num_rows;
  int32 num_cols;

  [[nodiscard]] size_type get_single_item_num_nnz() const { return static_cast<size_type>(stride * num_rows); }
};

} // namespace dense

namespace ell {

/**
 * Encapsulates one matrix from a batch of ell matrices.
 */
template<typename ValueType, typename IndexType = const int32>
struct batch_item {
  using value_type = ValueType;
  using index_type = IndexType;

  ValueType* values;
  index_type* col_idxs;
  index_type stride;
  index_type num_rows;
  index_type num_cols;
  index_type num_stored_elems_per_row;

  [[nodiscard]] size_type get_single_item_num_nnz() const {
    return static_cast<size_type>(stride * num_stored_elems_per_row);
  }
};

/**
 * A 'simple' structure to store a global uniform batch of ell matrices.
 */
template<typename ValueType, typename IndexType = const int32>
struct uniform_batch {
  using value_type = ValueType;
  using index_type = IndexType;
  using entry_type = batch_item<value_type, index_type>;

  ValueType* values;
  index_type* col_idxs;
  size_type num_batch_items;
  index_type stride;
  index_type num_rows;
  index_type num_cols;
  index_type num_stored_elems_per_row;

  [[nodiscard]] size_type get_single_item_num_nnz() const {
    return static_cast<size_type>(stride * num_stored_elems_per_row);
  }
};

} // namespace ell

template<typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE csr::batch_item<const ValueType, const IndexType>
to_const(const csr::batch_item<ValueType, IndexType>& b) {
  return {b.values, b.col_idxs, b.row_ptrs, b.num_rows, b.num_cols, b.num_nnz_per_item};
}

template<typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE csr::uniform_batch<const ValueType, const IndexType>
to_const(const csr::uniform_batch<ValueType, IndexType>& ub) {
  return {ub.values, ub.col_idxs, ub.row_ptrs, ub.num_batch_items, ub.num_rows, ub.num_cols, ub.num_nnz_per_item};
}

template<typename ValueType>
GKO_ATTRIBUTES GKO_INLINE dense::batch_item_colum_major<const ValueType> to_const(const dense::batch_item_colum_major<ValueType>& b) {
  return {b.values, b.stride, b.num_rows, b.num_cols};
}

template<typename ValueType>
GKO_ATTRIBUTES GKO_INLINE dense::batch_item<const ValueType> to_const(const dense::batch_item<ValueType>& b) {
  return {b.values, b.stride, b.num_rows, b.num_cols};
}

template<typename ValueType>
GKO_ATTRIBUTES GKO_INLINE dense::uniform_batch<const ValueType> to_const(const dense::uniform_batch<ValueType>& ub) {
  return {ub.values, ub.num_batch_items, ub.stride, ub.num_rows, ub.num_cols};
}

template<typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE ell::batch_item<const ValueType, const IndexType>
to_const(const ell::batch_item<ValueType, IndexType>& b) {
  return {b.values, b.col_idxs, b.stride, b.num_rows, b.num_cols, b.num_stored_elems_per_row};
}

template<typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE ell::uniform_batch<const ValueType, const IndexType>
to_const(const ell::uniform_batch<ValueType, IndexType>& ub) {
  return {ub.values, ub.col_idxs, ub.num_batch_items, ub.stride, ub.num_rows, ub.num_cols, ub.num_stored_elems_per_row};
}

} // namespace matrix

struct extract_batch_item_fn {
  template<typename ValueType, typename IndexType, typename Tag>
  constexpr matrix::csr::batch_item<ValueType, IndexType>
  operator()(const matrix::csr::uniform_batch<ValueType, IndexType>& batch, int64 batch_idx, Tag tag) const {
    return {batch.values + static_cast<IndexType>(batch_idx) * batch.num_nnz_per_item,
            batch.col_idxs,
            batch.row_ptrs,
            batch.num_rows,
            batch.num_cols,
            batch.num_nnz_per_item};
  }

  template<typename ValueType, typename Tag>
  constexpr matrix::dense::batch_item<ValueType> operator()(const matrix::dense::uniform_batch<ValueType>& batch,
                                                             int64 batch_idx, Tag tag) const {
    return {batch.values + batch_idx * batch.stride * batch.num_rows, batch.stride, batch.num_rows, batch.num_cols};
  }

  template<typename ValueType, typename IndexType, typename Tag>
  constexpr matrix::ell::batch_item<ValueType, IndexType>
  operator()(const matrix::ell::uniform_batch<ValueType, IndexType>& batch, const int64 batch_idx, Tag tag) const {
    return {batch.values + static_cast<IndexType>(batch_idx) * batch.num_stored_elems_per_row * batch.num_rows,
            batch.col_idxs,
            batch.stride,
            batch.num_rows,
            batch.num_cols,
            batch.num_stored_elems_per_row};
  }

  template<typename ValueType, typename Tag>
  constexpr multi_vector::batch_item<ValueType> operator()(const multi_vector::uniform_batch<ValueType>& batch,
                                                           int64 batch_idx, Tag tag) const {
    return {batch.values + batch_idx * batch.stride * batch.num_rows, batch.stride, batch.num_rows,
            batch.num_rhs};
  }

  template<typename T, typename Tag>
  constexpr auto operator()(const T& batch,  int64 batch_idx, Tag tag) const {
    return extract_batch_item(batch, batch_idx, tag);
  }
};

GKO_CPO_STORAGE constexpr extract_batch_item_fn extract_batch_item{};


struct generate_batch_item_fn {
  template<typename ValueType, typename Mat, typename Tag>
  constexpr batch_preconditioner::Identity<ValueType>
  operator()(batch_preconditioner::Identity<ValueType> id, Mat mat, char* buffer, int64 batch_idx, Tag tag) const {
    return id;
  }

  template<typename T, typename Mat, typename Tag>
  constexpr auto operator()(const T& batch, Mat mat, char* buffer, int64 batch_idx, Tag tag) const {
    return generate_batch_item(batch, mat, buffer, batch_idx, tag);
  }
};

GKO_CPO_STORAGE constexpr generate_batch_item_fn generate_batch_item{};


} // namespace gko::batch

#endif // GKO_CORE_MATRIX_BATCH_STRUCT_HPP_
