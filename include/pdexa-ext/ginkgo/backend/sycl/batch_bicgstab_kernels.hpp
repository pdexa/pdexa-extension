// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"
#include "pdexa-ext/ginkgo/core/solver/batch_bicgstab_settings.hpp"

#if PDEXA_EXT_ENABLE_SYCL

#include <ginkgo/core/log/batch_logger.hpp>

#include "../../batch_criteria.hpp"
#include "../../batch_identity.hpp"
#include "../../batch_logger.hpp"
#include "../../batch_multi_vector.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_template {
namespace batch_bicgstab {


template <typename ValueType, typename Op, typename Prec>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_bicgstab::settings<remove_complex<ValueType>>& options,
    const Op* mat, const Prec prec, multi_vector_view<const ValueType> b,
    multi_vector_view<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
    GKO_NOT_IMPLEMENTED;


}  // namespace batch_bicgstab
}  // namespace batch_template
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#else


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_template {
namespace batch_bicgstab {


template <typename ValueType, typename Op, typename Prec>
void apply(
    std::shared_ptr<const DefaultExecutor> ,
    const kernels::batch_bicgstab::settings<remove_complex<ValueType>>& ,
    const Op , const Prec, batch::multi_vector::uniform_batch<const ValueType> ,
    batch::multi_vector::uniform_batch<ValueType> ,
    batch::log::detail::log_data<remove_complex<ValueType>>&)
    GKO_NOT_IMPLEMENTED;


}  // namespace batch_bicgstab
}  // namespace batch_template
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif
