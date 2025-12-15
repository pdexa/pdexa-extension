// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#include "pdexa-ext/ginkgo/backend/cuda_hip/base/config.hpp"
#include "pdexa-ext/ginkgo/backend/cuda_hip/batch_apply.hpp"
#include "pdexa-ext/ginkgo/backend/cuda_hip/batch_multi_vector_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/cuda_hip/components/cooperative_groups.hpp"
#include "pdexa-ext/ginkgo/backend/cuda_hip/components/uninitialized_array.hpp"
#include "pdexa-ext/ginkgo/core/solver/batch_bicgstab_settings.hpp"

namespace gko::kernels::GKO_DEVICE_NAMESPACE {

constexpr int max_bicgstab_threads = 1024;

namespace batch_template::batch_single_kernels::batch_bicgstab {

template<typename Group, typename BatchMatrixType_entry, typename ValueType, typename RealType, size_type n_rhs>
__device__ __forceinline__ void initialize(Group subgroup,
                                           const int num_rows,
                                           const BatchMatrixType_entry& mat_entry,
                                           batch::multi_vector::batch_item<const ValueType> b_global_entry,
                                           batch::multi_vector::batch_item<const ValueType> x_global_entry,
                                           uninitialized_array<ValueType, n_rhs>& rho_old,
                                           uninitialized_array<ValueType, n_rhs>& omega,
                                           uninitialized_array<ValueType, n_rhs>& alpha,
                                           batch::multi_vector::batch_item<ValueType> x_shared_entry,
                                           batch::multi_vector::batch_item<ValueType> r_shared_entry,
                                           batch::multi_vector::batch_item<ValueType> r_hat_shared_entry,
                                           batch::multi_vector::batch_item<ValueType> p_shared_entry,
                                           batch::multi_vector::batch_item<ValueType> p_hat_shared_entry,
                                           batch::multi_vector::batch_item<ValueType> v_shared_entry,
                                           uninitialized_array<RealType, n_rhs>& rhs_norm,
                                           uninitialized_array<RealType, n_rhs>& res_norm) {
  auto row_group = group::tiled_partition<n_rhs>(group::this_thread_block());
  auto col = row_group.thread_rank();

  if (row_group.meta_group_rank() == 0) {
    rho_old[col] = one<ValueType>();
    omega[col] = one<ValueType>();
    alpha[col] = one<ValueType>();
  }

  // copy x from global to shared memory
  // r = b
  for (int row = row_group.meta_group_rank(); row < num_rows; row += row_group.meta_group_size()) {
    x_shared_entry.at(row, col) = x_global_entry.at(row, col);
    r_shared_entry.at(row, col) = b_global_entry.at(row, col);
  }
  __syncthreads();

  compute_residual(mat_entry, batch::to_const(x_shared_entry), b_global_entry, r_shared_entry);
  __syncthreads();

  if (subgroup.meta_group_rank() == 0) { compute_norm2<n_rhs>(subgroup, batch::to_const(r_shared_entry), res_norm); }
  else if (subgroup.meta_group_rank() == 1) { compute_norm2<n_rhs>(subgroup, b_global_entry, rhs_norm); }
  __syncthreads();

  for (auto row = row_group.meta_group_rank(); row < num_rows; row += row_group.meta_group_size()) {
    r_hat_shared_entry.at(row, col) = r_shared_entry.at(row, col);
    p_shared_entry.at(row, col) = zero<ValueType>();
    p_hat_shared_entry.at(row, col) = zero<ValueType>();
    v_shared_entry.at(row, col) = zero<ValueType>();
  }
}

template<typename ValueType, size_type n_rhs>
__device__ __forceinline__ void update_p(const int num_rows,
                                         const uninitialized_array<ValueType, n_rhs>& rho_new,
                                         const uninitialized_array<ValueType, n_rhs>& rho_old,
                                         const uninitialized_array<ValueType, n_rhs>& alpha,
                                         const uninitialized_array<ValueType, n_rhs>& omega,
                                         batch::multi_vector::batch_item<const ValueType> r_shared_entry,
                                         batch::multi_vector::batch_item<const ValueType> v_shared_entry,
                                         batch::multi_vector::batch_item<ValueType> p_shared_entry) {
  auto group = group::tiled_partition<n_rhs>(group::this_thread_block());
  auto col = group.thread_rank();
  const ValueType beta = (rho_new[col] / rho_old[col]) * (alpha[col] / omega[col]);
  for (auto row = group.meta_group_rank(); row < num_rows; row += group.meta_group_size()) {
    p_shared_entry.at(row, col) =
      r_shared_entry.at(row, col) + beta * (p_shared_entry.at(row, col) - omega[col] * v_shared_entry.at(row, col));
  }
}

template<typename Group, typename ValueType, size_type n_rhs>
__device__ __forceinline__ void compute_alpha(Group subgroup,
                                              const uninitialized_array<ValueType, n_rhs>& rho_new,
                                              batch::multi_vector::batch_item<const ValueType> r_hat_shared_entry,
                                              batch::multi_vector::batch_item<const ValueType> v_shared_entry,
                                              uninitialized_array<ValueType, n_rhs>& alpha) {
  if (subgroup.meta_group_rank() == 0) { compute_conj_dot<n_rhs>(subgroup, r_hat_shared_entry, v_shared_entry, alpha); }
  __syncthreads();
  auto col = static_cast<int>(threadIdx.x);
  if (col < n_rhs) { alpha[col] = rho_new[col] / alpha[col]; }
}

template<typename ValueType, size_type n_rhs>
__device__ __forceinline__ void update_s(const int num_rows,
                                         batch::multi_vector::batch_item<const ValueType> r_shared_entry,
                                         const uninitialized_array<ValueType, n_rhs>& alpha,
                                         batch::multi_vector::batch_item<const ValueType> v_shared_entry,
                                         batch::multi_vector::batch_item<ValueType> s_shared_entry) {
  auto group = group::tiled_partition<n_rhs>(group::this_thread_block());
  auto col = group.thread_rank();
  for (auto row = group.meta_group_rank(); row < num_rows; row += group.meta_group_size()) {
    s_shared_entry.at(row, col) = r_shared_entry.at(row, col) - alpha[col] * v_shared_entry.at(row, col);
  }
}

template<typename Group, typename ValueType, size_type n_rhs>
__device__ __forceinline__ void compute_omega(Group subgroup,
                                              const int num_rows,
                                              batch::multi_vector::batch_item<const ValueType> t_shared_entry,
                                              batch::multi_vector::batch_item<const ValueType> s_shared_entry,
                                              uninitialized_array<ValueType, n_rhs>& temp,
                                              uninitialized_array<ValueType, n_rhs>& omega) {
  if (subgroup.meta_group_rank() == 0) { compute_conj_dot<n_rhs>(subgroup, t_shared_entry, s_shared_entry, omega); }
  else if (subgroup.meta_group_rank() == 1) { compute_conj_dot<n_rhs>(subgroup, t_shared_entry, t_shared_entry, temp); }

  __syncthreads();
  if (threadIdx.x < n_rhs) { omega[threadIdx.x] /= temp[threadIdx.x]; }
}

template<typename ValueType, size_type n_rhs>
__device__ __forceinline__ void update_x_and_r(const int num_rows,
                                               batch::multi_vector::batch_item<const ValueType> p_hat_shared_entry,
                                               batch::multi_vector::batch_item<const ValueType> s_hat_shared_entry,
                                               const uninitialized_array<ValueType, n_rhs>& alpha,
                                               const uninitialized_array<ValueType, n_rhs>& omega,
                                               batch::multi_vector::batch_item<const ValueType> s_shared_entry,
                                               batch::multi_vector::batch_item<const ValueType> t_shared_entry,
                                               batch::multi_vector::batch_item<ValueType> x_shared_entry,
                                               batch::multi_vector::batch_item<ValueType> r_shared_entry) {
  auto group = group::tiled_partition<n_rhs>(group::this_thread_block());
  auto col = group.thread_rank();
  for (auto row = group.meta_group_rank(); row < num_rows; row += group.meta_group_size()) {
    x_shared_entry.at(row, col) = x_shared_entry.at(row, col) + alpha[col] * p_hat_shared_entry.at(row, col) +
                                  omega[col] * s_hat_shared_entry.at(row, col);
    r_shared_entry.at(row, col) = s_shared_entry.at(row, col) - omega[col] * t_shared_entry.at(row, col);
  }
}

template<typename ValueType, size_type n_rhs>
__device__ __forceinline__ void update_x_middle(const uninitialized_array<ValueType, n_rhs>& alpha,
                                                batch::multi_vector::batch_item<const ValueType> p_hat_shared_entry,
                                                batch::multi_vector::batch_item<ValueType> x_shared_entry) {
  const int max_li = x_shared_entry.num_rows * x_shared_entry.num_rhs;
  for (auto li = static_cast<int>(threadIdx.x); li < max_li; li += static_cast<int>(blockDim.x)) {
    const int row = li / x_shared_entry.num_rhs;
    const int col = li % x_shared_entry.num_rhs;

    x_shared_entry.at(row, col) = x_shared_entry.at(row, col) + alpha[col] * p_hat_shared_entry.at(row, col);
  }
}

template<typename StopType,
         int n_shared,
         bool prec_shared_bool,
         typename PrecType,
         typename LogType,
         typename BatchMatrixType,
         size_type n_rhs,
         typename ValueType>
__global__ void __launch_bounds__(max_bicgstab_threads)
  apply_kernel(const kernels::batch_bicgstab::storage_config sconf,
               const int max_iter,
               const remove_complex<ValueType> tol,
               LogType logger,
               PrecType prec_shared,
               const BatchMatrixType mat,
               batch::multi_vector::uniform_batch<const ValueType> b,
               batch::multi_vector::uniform_batch<ValueType> x,
               ValueType* const __restrict__ workspace = nullptr) {
  using real_type = remove_complex<ValueType>;
  const auto num_batch_items = mat.num_batch_items;
  const auto num_rows = mat.num_rows;
  constexpr auto num_rhs = n_rhs;

  constexpr auto tile_size = config::warp_size;
  auto thread_block = group::this_thread_block();
  auto subgroup = group::tiled_partition<tile_size>(thread_block);

  for (auto batch_id = static_cast<int>(blockIdx.x); batch_id < num_batch_items;
       batch_id += static_cast<int>(gridDim.x)) {
    const int gmem_offset = batch_id * sconf.gmem_stride_bytes / sizeof(ValueType);
    extern __shared__ char local_mem_sh[];

    const batch::multi_vector::batch_item<ValueType> p_hat_sh{workspace + gmem_offset, num_rhs, num_rows, num_rhs};
    const batch::multi_vector::batch_item<ValueType> s_hat_sh{p_hat_sh.values + sconf.padded_vec_len, num_rhs, num_rows,
                                                              num_rhs};
    const batch::multi_vector::batch_item<ValueType> p_sh{s_hat_sh.values + sconf.padded_vec_len, num_rhs, num_rows,
                                                          num_rhs};
    const batch::multi_vector::batch_item<ValueType> s_sh{p_sh.values + sconf.padded_vec_len, num_rhs, num_rows,
                                                          num_rhs};
    const batch::multi_vector::batch_item<ValueType> r_sh{s_sh.values + sconf.padded_vec_len, num_rhs, num_rows,
                                                          num_rhs};
    const batch::multi_vector::batch_item<ValueType> r_hat_sh{r_sh.values + sconf.padded_vec_len, num_rhs, num_rows,
                                                              num_rhs};
    const batch::multi_vector::batch_item<ValueType> v_sh{r_hat_sh.values + sconf.padded_vec_len, num_rhs, num_rows,
                                                          num_rhs};
    const batch::multi_vector::batch_item<ValueType> t_sh{v_sh.values + sconf.padded_vec_len, num_rhs, num_rows,
                                                          num_rhs};
    const batch::multi_vector::batch_item<ValueType> x_sh{t_sh.values + sconf.padded_vec_len, num_rhs, num_rows,
                                                          num_rhs};
    const batch::multi_vector::batch_item<ValueType> prec_work_sh{x_sh.values + sconf.padded_vec_len, num_rhs, num_rows,
                                                                  num_rhs};

    __shared__ uninitialized_array<ValueType, n_rhs> rho_old_sh;
    __shared__ uninitialized_array<ValueType, n_rhs> rho_new_sh;
    __shared__ uninitialized_array<ValueType, n_rhs> omega_sh;
    __shared__ uninitialized_array<ValueType, n_rhs> alpha_sh;
    __shared__ uninitialized_array<ValueType, n_rhs> temp_sh;
    __shared__ uninitialized_array<real_type, n_rhs> norms_rhs_sh;
    __shared__ uninitialized_array<real_type, n_rhs> norms_res_sh;

    const auto mat_entry = batch::extract_batch_item(mat, batch_id, device_kernel{});
    const auto b_entry = batch::extract_batch_item(b, batch_id, device_kernel{});
    auto x_entry = batch::extract_batch_item(x, batch_id, device_kernel{});

    // generate preconditioner
    auto prec_entry =
      batch::generate_batch_item(prec_shared, mat_entry, reinterpret_cast<char*>(prec_work_sh.values), batch_id, device_kernel{});

    // initialization
    // rho_old_sh = 1, omega_sh = 1, alpha_sh = 1
    // compute b norms
    // copy x from global to shared memory
    // r = b - A*x
    // compute residual norms
    // r_hat = r
    // p = 0
    // p_hat = 0
    // v = 0
    initialize(subgroup, num_rows, mat_entry, batch::to_const(b_entry), batch::to_const(x_entry), rho_old_sh, omega_sh,
               alpha_sh, x_sh, r_sh, r_hat_sh, p_sh, p_hat_sh, v_sh, norms_rhs_sh, norms_res_sh);
    __syncthreads();

    // stopping criterion object
    StopType stop(tol, norms_rhs_sh);

    int iter = 0;
    for (; iter < max_iter; iter++) {
      if (stop.template check_converged<n_rhs>(norms_res_sh)) {
        logger.template log_iteration<n_rhs>(batch_id, iter, norms_res_sh);
        break;
      }

      // rho_new_sh =  < r_hat , r > = (r_hat)' * (r)
      if (subgroup.meta_group_rank() == 0) {
        compute_conj_dot<n_rhs>(subgroup, batch::to_const(r_hat_sh), batch::to_const(r_sh), rho_new_sh);
      }
      __syncthreads();

      // beta = (rho_new_sh / rho_old_sh)*(alpha_sh / omega_sh)
      // p = r + beta*(p - omega_sh * v)
      update_p(num_rows, rho_new_sh, rho_old_sh, alpha_sh, omega_sh, batch::to_const(r_sh), batch::to_const(v_sh),
               p_sh);
      __syncthreads();

      // p_hat = precond * p
      simple_apply(prec_entry, batch::to_const(p_sh), p_hat_sh);
      __syncthreads();

      // v = A * p_hat
      simple_apply(mat_entry, batch::to_const(p_hat_sh), v_sh);
      __syncthreads();

      // alpha_sh = rho_new_sh / < r_hat , v>
      compute_alpha<group::thread_block_tile<tile_size>, ValueType>(subgroup, rho_new_sh, batch::to_const(r_hat_sh),
                                                                    batch::to_const(v_sh), alpha_sh);
      __syncthreads();

      // s = r - alpha_sh*v
      update_s(num_rows, batch::to_const(r_sh), alpha_sh, batch::to_const(v_sh), s_sh);
      __syncthreads();

      // an estimate of residual norms
      if (subgroup.meta_group_rank() == 0) { compute_norm2<n_rhs>(subgroup, batch::to_const(s_sh), norms_res_sh); }
      __syncthreads();

      // if (norms_res_sh[col] / norms_rhs_sh[col] < tol) {
      if (stop.template check_converged<n_rhs>(norms_res_sh)) {
        update_x_middle(alpha_sh, batch::to_const(p_hat_sh), x_sh);
        logger.template log_iteration<n_rhs>(batch_id, iter, norms_res_sh);
        break;
      }

      // s_hat = precond * s
      simple_apply(prec_entry, batch::to_const(s_sh), s_hat_sh);
      __syncthreads();

      // t = A * s_hat
      simple_apply(mat_entry, batch::to_const(s_hat_sh), t_sh);
      __syncthreads();

      // omega_sh = <t,s> / <t,t>
      compute_omega(subgroup, num_rows, batch::to_const(t_sh), batch::to_const(s_sh), temp_sh, omega_sh);
      __syncthreads();

      // x = x + alpha_sh*p_hat + omega_sh *s_hat
      // r = s - omega_sh * t
      update_x_and_r(num_rows, batch::to_const(p_hat_sh), batch::to_const(s_hat_sh), alpha_sh, omega_sh,
                     batch::to_const(s_sh), batch::to_const(t_sh), x_sh, r_sh);
      __syncthreads();

      if (subgroup.meta_group_rank() == 0) { compute_norm2<n_rhs>(subgroup, batch::to_const(r_sh), norms_res_sh); }

      if (threadIdx.x == blockDim.x - 1) { rho_old_sh = rho_new_sh; }
      __syncthreads();
    }

    logger.template log_iteration<n_rhs>(batch_id, iter, norms_res_sh);

    // copy x back to global memory
    copy(batch::to_const(x_sh), x_entry);
    __syncthreads();
  }
}

} // namespace batch_template::batch_single_kernels::batch_bicgstab
} // namespace gko::kernels::GKO_DEVICE_NAMESPACE
