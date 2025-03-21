// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"
#include "pdexa-ext/ginkgo/core/solver/batch_bicgstab_settings.hpp"

#if PDEXA_EXT_ENABLE_HIP

#include <hip/hip_runtime.h>

#include <ginkgo/core/log/batch_logger.hpp>

#include "../cuda_hip/batch_bicgstab_kernels.hpp"
#include "pdexa-ext/ginkgo/core/log/batch_simple_logger.hpp"
#include "pdexa-ext/ginkgo/core/preconditioner/batch_identity.hpp"
#include "pdexa-ext/ginkgo/core/stop/batch_criteria.hpp"

namespace gko::kernels::hip::batch_template::batch_bicgstab {

inline int get_num_threads_per_block(std::shared_ptr<const DefaultExecutor> exec, const int num_rows) {
  int num_warps = std::max(num_rows / 4, 2);
  constexpr int warp_sz = static_cast<int>(config::warp_size);
  const int min_block_size = 2 * warp_sz;
  const int device_max_threads = ((std::max(num_rows, min_block_size)) / warp_sz) * warp_sz;
  // This value has been taken from ROCm docs. This is the number of registers
  // that maximizes the occupancy on an AMD GPU (MI200). HIP does not have an
  // API to query the number of registers a function uses.
  const int num_regs_used_per_thread = 64;
  int max_regs_blk = 0;
  GKO_ASSERT_NO_HIP_ERRORS(
    hipDeviceGetAttribute(&max_regs_blk, hipDeviceAttributeMaxRegistersPerBlock, exec->get_device_id()));
  int max_threads_regs = (max_regs_blk / num_regs_used_per_thread);
  max_threads_regs = (max_threads_regs / warp_sz) * warp_sz;
  int max_threads = std::min(max_threads_regs, device_max_threads);
  max_threads = max_threads <= 1024 ? max_threads : 1024;
  return std::max(std::min(num_warps * warp_sz, max_threads), min_block_size);
}

template<typename ValueType, typename BatchMatrixType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const kernels::batch_bicgstab::settings<remove_complex<ValueType>>& settings,
           const BatchMatrixType mat,
           batch::multi_vector::uniform_batch<const ValueType> b,
           batch::multi_vector::uniform_batch<ValueType> x,
           batch::log::detail::log_data<remove_complex<ValueType>>& logdata) {
  using PrecType = batch_preconditioner::Identity<ValueType>;
  using StopType = batch_stop::SimpleAbsResidual<ValueType>;

  using hip_value_type = hip_type<ValueType>;
  using real_type = remove_complex<hip_value_type>;

  const size_type num_batch_items = mat.num_batch_items;
  constexpr int align_multiple = 8;
  const int padded_num_rows = ceildiv(mat.num_rows, align_multiple) * align_multiple;
  int shmem_per_blk = 0;
  GKO_ASSERT_NO_HIP_ERRORS(
    hipDeviceGetAttribute(&shmem_per_blk, hipDeviceAttributeMaxSharedMemoryPerBlock, exec->get_device_id()));
  const int block_size = get_num_threads_per_block(exec, mat.num_rows);
  GKO_ASSERT(block_size >= static_cast<int>(2 * config::warp_size));
  GKO_ASSERT(block_size % static_cast<int>(config::warp_size) == 0);

  // Returns amount required in bytes
  const size_t prec_size = PrecType::dynamic_work_size(padded_num_rows, mat);
  const auto sconf = kernels::batch_bicgstab::compute_shared_storage<PrecType, hip_value_type>(
    shmem_per_blk, padded_num_rows, std::numeric_limits<int>::min(), b.num_rhs);
  const uint32 shared_size =
    sconf.n_shared * padded_num_rows * sizeof(hip_value_type) + (sconf.prec_shared ? prec_size : 0);
  auto workspace = array<hip_value_type>(exec, sconf.gmem_stride_bytes * num_batch_items / sizeof(hip_value_type));
  GKO_ASSERT(sconf.gmem_stride_bytes % sizeof(hip_value_type) == 0);

  hip_value_type* const workspace_data = workspace.get_data();

  auto prec = PrecType();
  auto logger = batch_log::SimpleFinalLogger<real_type>(logdata.res_norms.get_data(), logdata.iter_counts.get_data());

  batch_single_kernels::batch_bicgstab::apply_kernel<StopType, 0, false>
    <<<mat.num_batch_items, block_size, shared_size, exec->get_stream()>>>(
        sconf, settings.max_iterations,
        as_device_type(settings.residual_tol), logger, prec, mat, b,
        x, workspace_data);
}

} // namespace gko::kernels::hip::batch_bicgstab

#else

namespace gko::kernels::hip::batch_template::batch_bicgstab {

template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> ,
    const kernels::batch_bicgstab::settings<remove_complex<ValueType>>& ,
    const Op , batch::multi_vector::uniform_batch<const ValueType> ,
    batch::multi_vector::uniform_batch<ValueType> ,
    batch::log::detail::log_data<remove_complex<ValueType>>& )
    GKO_NOT_IMPLEMENTED;

}

#endif
