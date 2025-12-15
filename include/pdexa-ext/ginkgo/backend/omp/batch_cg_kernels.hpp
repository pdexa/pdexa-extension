// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"

#if PDEXA_EXT_ENABLE_OPENMP

#include <ginkgo/core/log/batch_logger.hpp>

#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"
#include "pdexa-ext/ginkgo/core/solver/batch_cg_settings.hpp"

#include "pdexa-ext/ginkgo/backend/reference/batch_apply.hpp"
#include "pdexa-ext/ginkgo/backend/reference/batch_multi_vector_kernels.hpp"
#include "pdexa-ext/ginkgo/core/stop/batch_criteria.hpp"
#include "pdexa-ext/ginkgo/core/log/batch_simple_logger.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace batch_template {
namespace batch_cg {


template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& options,
    const Op mat, batch::multi_vector::uniform_batch<const ValueType> b,
    batch::multi_vector::uniform_batch<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
    GKO_NOT_IMPLEMENTED;


}  // namespace batch_cg
}  // namespace batch_template
}  // namespace omp
}  // namespace kernels
}  // namespace gko


#else


namespace gko::kernels::omp::batch_template::batch_cg {

template<typename ValueType, typename Op>
void apply(std::shared_ptr<const DefaultExecutor> ,
           const kernels::batch_cg::settings<remove_complex<ValueType>>& ,
           const Op ,
           batch::multi_vector::uniform_batch<const ValueType> ,
           batch::multi_vector::uniform_batch<ValueType> ,
           batch::log::detail::log_data<remove_complex<ValueType>>& ) GKO_NOT_IMPLEMENTED;

}

#endif
