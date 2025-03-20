#pragma once

#include <ginkgo/core/matrix/batch_csr.hpp>

#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"
#include "pdexa-ext/ginkgo/core/matrix/batch_struct.hpp"

namespace gko::batch {
namespace detail {}

struct create_view_fn {
  template<typename ValueType>
  constexpr multi_vector::uniform_batch<const ValueType> operator()(const MultiVector<ValueType>* vec) const {
    return {vec->get_const_values(), vec->get_num_batch_items(), static_cast<int32>(vec->get_common_size()[1]),
            static_cast<int32>(vec->get_common_size()[0]), static_cast<int32>(vec->get_common_size()[1])};
  }

  template<typename ValueType>
  constexpr multi_vector::uniform_batch<ValueType> operator()(MultiVector<ValueType>* vec) const {
    return {vec->get_values(), vec->get_num_batch_items(), static_cast<int32>(vec->get_common_size()[1]),
            static_cast<int32>(vec->get_common_size()[0]), static_cast<int32>(vec->get_common_size()[1])};
  }

  template<typename ValueType, typename IndexType>
  constexpr matrix::csr::uniform_batch<const ValueType, const IndexType>
  operator()(const matrix::Csr<ValueType, IndexType>* mtx) const {
    return {mtx->get_const_values(),
            mtx->get_const_col_idxs(),
            mtx->get_const_row_ptrs(),
            mtx->get_num_batch_items(),
            static_cast<IndexType>(mtx->get_common_size()[0]),
            static_cast<IndexType>(mtx->get_common_size()[1]),
            static_cast<IndexType>(mtx->get_num_elements_per_item())};
  }

  template<typename ValueType, typename IndexType>
  constexpr matrix::csr::uniform_batch<ValueType, IndexType> operator()(matrix::Csr<ValueType, IndexType>* mtx) const {
    return {mtx->get_values(),
            mtx->get_col_idxs(),
            mtx->get_row_ptrs(),
            mtx->get_num_batch_items(),
            static_cast<IndexType>(mtx->get_common_size()[0]),
            static_cast<IndexType>(mtx->get_common_size()[1]),
            static_cast<IndexType>(mtx->get_num_elements_per_item())};
  }

  template<typename T>
  auto operator()(const T* obj) const {
    return create_view(obj);
  }
};

inline constexpr create_view_fn create_view{};

} // namespace gko::batch
