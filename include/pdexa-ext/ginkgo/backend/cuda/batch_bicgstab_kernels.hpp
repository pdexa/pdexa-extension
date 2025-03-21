// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"
#include "pdexa-ext/ginkgo/core/solver/batch_bicgstab_settings.hpp"

#if PDEXA_EXT_ENABLE_CUDA

#include <ginkgo/core/log/batch_logger.hpp>

#include "../cuda_hip/batch_bicgstab_kernels.hpp"
#include "pdexa-ext/ginkgo/core/log/batch_simple_logger.hpp"
#include "pdexa-ext/ginkgo/core/preconditioner/batch_identity.hpp"
#include "pdexa-ext/ginkgo/core/stop/batch_criteria.hpp"

namespace gko::kernels::cuda::batch_template::batch_bicgstab {

template<typename StopType, typename PrecType, typename LogType, typename BatchMatrixType, typename ValueType>
int get_num_threads_per_block(std::shared_ptr<const DefaultExecutor> exec, const int num_rows) {
  int num_warps = std::max(num_rows / 4, 2);
  constexpr int warp_sz = static_cast<int>(config::warp_size);
  const int min_block_size = 2 * warp_sz;
  const int device_max_threads = (std::max(num_rows, min_block_size) / warp_sz) * warp_sz;
  auto get_num_regs = [](const auto func) {
    cudaFuncAttributes funcattr;
    cudaFuncGetAttributes(&funcattr, func);
    return funcattr.numRegs;
  };
  const int num_regs_used =
    std::max(get_num_regs(batch_single_kernels::batch_bicgstab::apply_kernel<StopType, 9, true, PrecType, LogType,
                                                                             BatchMatrixType, ValueType>),
             get_num_regs(batch_single_kernels::batch_bicgstab::apply_kernel<StopType, 0, false, PrecType, LogType,
                                                                             BatchMatrixType, ValueType>));
  int max_regs_blk = 0;
  cudaDeviceGetAttribute(&max_regs_blk, cudaDevAttrMaxRegistersPerBlock, exec->get_device_id());
  const int max_threads_regs = ((max_regs_blk / static_cast<int>(num_regs_used)) / warp_sz) * warp_sz;
  int max_threads = std::min(max_threads_regs, device_max_threads);
  max_threads = max_threads <= max_bicgstab_threads ? max_threads : max_bicgstab_threads;
  return std::max(std::min(num_warps * warp_sz, max_threads), min_block_size);
}

template<typename StopType, typename PrecType, typename LogType, typename BatchMatrixType, typename ValueType>
int get_max_dynamic_shared_memory(std::shared_ptr<const DefaultExecutor> exec) {
  int shmem_per_sm = 0;
  cudaDeviceGetAttribute(&shmem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, exec->get_device_id());
  GKO_ASSERT_NO_CUDA_ERRORS(
    cudaFuncSetAttribute(batch_single_kernels::batch_bicgstab::apply_kernel<StopType, 9, true, PrecType, LogType,
                                                                            BatchMatrixType, ValueType>,
                         cudaFuncAttributePreferredSharedMemoryCarveout, 99 /*%*/));
  cudaFuncAttributes funcattr;
  cudaFuncGetAttributes(&funcattr,
                        batch_single_kernels::batch_bicgstab::apply_kernel<StopType, 9, true, PrecType, LogType,
                                                                           BatchMatrixType, ValueType>);
  return funcattr.maxDynamicSharedSizeBytes;
}

template<typename ValueType, typename BatchMatrixType, typename PrecType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const kernels::batch_bicgstab::settings<remove_complex<ValueType>>& settings,
           const BatchMatrixType mat,
           batch::multi_vector::uniform_batch<const ValueType> b,
           batch::multi_vector::uniform_batch<ValueType> x,
           batch::log::detail::log_data<remove_complex<ValueType>>& logdata) {
  using PrecType = batch_preconditioner::Identity<ValueType>;
  using StopType = batch_stop::SimpleAbsResidual<ValueType>;

  using cuda_value_type = cuda_type<ValueType>;
  using real_type = remove_complex<cuda_value_type>;

  const size_type num_batch_items = mat.num_batch_items;
  constexpr int align_multiple = 8;
  const int padded_num_rows = ceildiv(mat.num_rows, align_multiple) * align_multiple;
  int shmem_per_blk = 0;
  GKO_ASSERT_NO_HIP_ERRORS(
    cudaDeviceGetAttribute(&shmem_per_blk, cudaDeviceAttributeMaxSharedMemoryPerBlock, exec->get_device_id()));
  const int block_size = get_num_threads_per_block(exec, mat.num_rows);
  GKO_ASSERT(block_size >= static_cast<int>(2 * config::warp_size));
  GKO_ASSERT(block_size % static_cast<int>(config::warp_size) == 0);

  // Returns amount required in bytes
  const size_t prec_size = PrecType::dynamic_work_size(padded_num_rows, mat.get_single_item_num_nnz());
  const auto sconf = kernels::batch_bicgstab::compute_shared_storage<PrecType, cuda_value_type>(
    shmem_per_blk, padded_num_rows, mat.get_single_item_num_nnz(), b.num_rhs);
  const uint32 shared_size =
    sconf.n_shared * padded_num_rows * sizeof(cuda_value_type) + (sconf.prec_shared ? prec_size : 0);
  auto workspace = array<cuda_value_type>(exec, sconf.gmem_stride_bytes * num_batch_items / sizeof(cuda_value_type));
  GKO_ASSERT(sconf.gmem_stride_bytes % sizeof(cuda_value_type) == 0);

  cuda_value_type* const workspace_data = workspace.get_data();

  auto prec = PrecType();
  auto logger = batch_log::SimpleFinalLogger<real_type>(logdata.res_norms.get_data(), logdata.iter_counts.get_data());

  batch_single_kernels::batch_bicgstab::apply_kernel<StopType, 0, false>
    <<<mat.num_batch_items, block_size, shared_size, exec->get_stream()>>>(
      sconf, settings.max_iterations, as_device_type(settings.residual_tol), logger, prec, mat, b.values, x.values,
      workspace_data);
}

} // namespace gko::kernels::cuda::batch_template::batch_bicgstab

#else

namespace gko::kernels::cuda::batch_template::batch_bicgstab {

template<typename ValueType, typename Op>
void apply(std::shared_ptr<const DefaultExecutor>,
           const kernels::batch_bicgstab::settings<remove_complex<ValueType>>&,
           const Op,
           batch::multi_vector::uniform_batch<const ValueType>,
           batch::multi_vector::uniform_batch<ValueType>,
           batch::log::detail::log_data<remove_complex<ValueType>>&) GKO_NOT_IMPLEMENTED;

}

#endif
