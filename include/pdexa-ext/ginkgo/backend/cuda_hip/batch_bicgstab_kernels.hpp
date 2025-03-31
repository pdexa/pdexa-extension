// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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

template<typename Group, typename BatchMatrixType_entry, typename ValueType>
__device__ __forceinline__ void initialize(Group subgroup,
                                           const int num_rows,
                                           const BatchMatrixType_entry& mat_entry,
                                           batch::multi_vector::batch_item<const ValueType> b_global_entry,
                                           batch::multi_vector::batch_item<const ValueType> x_global_entry,
                                           ValueType& rho_old,
                                           ValueType& omega,
                                           ValueType& alpha,
                                           batch::multi_vector::batch_item<ValueType> x_shared_entry,
                                           batch::multi_vector::batch_item<ValueType> r_shared_entry,
                                           batch::multi_vector::batch_item<ValueType> r_hat_shared_entry,
                                           batch::multi_vector::batch_item<ValueType> p_shared_entry,
                                           batch::multi_vector::batch_item<ValueType> p_hat_shared_entry,
                                           batch::multi_vector::batch_item<ValueType> v_shared_entry,
                                           remove_complex<ValueType>& rhs_norm,
                                           remove_complex<ValueType>& res_norm) {
  rho_old = one<ValueType>();
  omega = one<ValueType>();
  alpha = one<ValueType>();

  // copy x from global to shared memory
  // r = b
  for (int iz = static_cast<int>(threadIdx.x); iz < num_rows; iz += static_cast<int>(blockDim.x)) {
    x_shared_entry[iz] = x_global_entry[iz];
    r_shared_entry[iz] = b_global_entry[iz];
  }
  __syncthreads();

  compute_residual(mat_entry, batch::to_const(x_shared_entry), b_global_entry, r_shared_entry);
  __syncthreads();

  if (threadIdx.x / config::warp_size == 0) { single_rhs_compute_norm2(subgroup, num_rows, batch::to_const(r_shared_entry), res_norm); }
  else if (threadIdx.x / config::warp_size == 1) {
    // Compute norms of rhs
    single_rhs_compute_norm2(subgroup, num_rows, b_global_entry, rhs_norm);
  }
  __syncthreads();

  for (auto iz = static_cast<int>(threadIdx.x); iz < num_rows; iz += static_cast<int>(blockDim.x)) {
    r_hat_shared_entry[iz] = r_shared_entry[iz];
    p_shared_entry[iz] = zero<ValueType>();
    p_hat_shared_entry[iz] = zero<ValueType>();
    v_shared_entry[iz] = zero<ValueType>();
  }
}

template<typename ValueType>
__device__ __forceinline__ void update_p(const int num_rows,
                                         const ValueType& rho_new,
                                         const ValueType& rho_old,
                                         const ValueType& alpha,
                                         const ValueType& omega,
                                         batch::multi_vector::batch_item<const ValueType> r_shared_entry,
                                         batch::multi_vector::batch_item<const ValueType> v_shared_entry,
                                         batch::multi_vector::batch_item<ValueType> p_shared_entry) {
  const ValueType beta = (rho_new / rho_old) * (alpha / omega);
  for (auto r = static_cast<int>(threadIdx.x); r < num_rows; r += static_cast<int>(blockDim.x)) {
    p_shared_entry[r] = r_shared_entry[r] + beta * (p_shared_entry[r] - omega * v_shared_entry[r]);
  }
}

template<typename Group, typename ValueType>
__device__ __forceinline__ void compute_alpha(Group subgroup,
                                              const int num_rows,
                                              const ValueType& rho_new,
                                              batch::multi_vector::batch_item<const ValueType> r_hat_shared_entry,
                                              batch::multi_vector::batch_item<const ValueType> v_shared_entry,
                                              ValueType& alpha) {
  if (threadIdx.x / config::warp_size == 0) {
    single_rhs_compute_conj_dot(subgroup, num_rows, r_hat_shared_entry, v_shared_entry, alpha);
  }
  __syncthreads();
  if (threadIdx.x == 0) { alpha = rho_new / alpha; }
}

template<typename ValueType>
__device__ __forceinline__ void update_s(const int num_rows,
                                         batch::multi_vector::batch_item<const ValueType> r_shared_entry,
                                         const ValueType& alpha,
                                         batch::multi_vector::batch_item<const ValueType> v_shared_entry,
                                         batch::multi_vector::batch_item<ValueType> s_shared_entry) {
  for (auto r = static_cast<int>(threadIdx.x); r < num_rows; r += static_cast<int>(blockDim.x)) {
    s_shared_entry[r] = r_shared_entry[r] - alpha * v_shared_entry[r];
  }
}

template<typename Group, typename ValueType>
__device__ __forceinline__ void compute_omega(Group subgroup,
                                              const int num_rows,
                                              batch::multi_vector::batch_item<const ValueType> t_shared_entry,
                                              batch::multi_vector::batch_item<const ValueType> s_shared_entry,
                                              ValueType& temp,
                                              ValueType& omega) {
  if (threadIdx.x / config::warp_size == 0) {
    single_rhs_compute_conj_dot(subgroup, num_rows, t_shared_entry, s_shared_entry, omega);
  }
  else if (threadIdx.x / config::warp_size == 1) {
    single_rhs_compute_conj_dot(subgroup, num_rows, t_shared_entry, t_shared_entry, temp);
  }

  __syncthreads();
  if (threadIdx.x == 0) { omega /= temp; }
}

template<typename ValueType>
__device__ __forceinline__ void update_x_and_r(const int num_rows,
                                               batch::multi_vector::batch_item<const ValueType> p_hat_shared_entry,
                                               batch::multi_vector::batch_item<const ValueType> s_hat_shared_entry,
                                               const ValueType& alpha,
                                               const ValueType& omega,
                                               batch::multi_vector::batch_item<const ValueType> s_shared_entry,
                                               batch::multi_vector::batch_item<const ValueType> t_shared_entry,
                                               batch::multi_vector::batch_item<ValueType> x_shared_entry,
                                               batch::multi_vector::batch_item<ValueType> r_shared_entry) {
  for (auto r = static_cast<int>(threadIdx.x); r < num_rows; r += static_cast<int>(blockDim.x)) {
    x_shared_entry[r] = x_shared_entry[r] + alpha * p_hat_shared_entry[r] + omega * s_hat_shared_entry[r];
    r_shared_entry[r] = s_shared_entry[r] - omega * t_shared_entry[r];
  }
}

template<typename ValueType>
__device__ __forceinline__ void update_x_middle(const int num_rows,
                                                const ValueType& alpha,
                                                batch::multi_vector::batch_item<const ValueType> p_hat_shared_entry,
                                                batch::multi_vector::batch_item<ValueType> x_shared_entry) {
  for (auto r = static_cast<int>(threadIdx.x); r < num_rows; r += static_cast<int>(blockDim.x)) {
    x_shared_entry[r] = x_shared_entry[r] + alpha * p_hat_shared_entry[r];
  }
}

template<typename StopType,
         int n_shared,
         bool prec_shared_bool,
         typename PrecType,
         typename LogType,
         typename BatchMatrixType,
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
  const auto num_rhs = b.num_rhs;

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

    __shared__ uninitialized_array<ValueType, 1> rho_old_sh;
    __shared__ uninitialized_array<ValueType, 1> rho_new_sh;
    __shared__ uninitialized_array<ValueType, 1> omega_sh;
    __shared__ uninitialized_array<ValueType, 1> alpha_sh;
    __shared__ uninitialized_array<ValueType, 1> temp_sh;
    __shared__ real_type norms_rhs_sh[1];
    __shared__ real_type norms_res_sh[1];

    const auto mat_entry = gko::batch::extract_batch_item(mat, batch_id);
    const auto b_entry = batch::extract_batch_item(b, batch_id);
    auto x_entry = batch::extract_batch_item(x, batch_id);

    // generate preconditioner
    prec_shared.generate(batch_id, mat_entry, prec_work_sh.values);

    // initialization
    // rho_old = 1, omega = 1, alpha = 1
    // compute b norms
    // copy x from global to shared memory
    // r = b - A*x
    // compute residual norms
    // r_hat = r
    // p = 0
    // p_hat = 0
    // v = 0
    initialize(subgroup, num_rows, mat_entry, batch::to_const(b_entry), batch::to_const(x_entry), rho_old_sh[0],
               omega_sh[0], alpha_sh[0], x_sh, r_sh, r_hat_sh, p_sh, p_hat_sh, v_sh, norms_rhs_sh[0], norms_res_sh[0]);
    __syncthreads();

    // stopping criterion object
    StopType stop(tol, norms_rhs_sh);

    int iter = 0;
    for (; iter < max_iter; iter++) {
      if (stop.check_converged(norms_res_sh)) {
        logger.log_iteration(batch_id, iter, norms_res_sh[0]);
        break;
      }

      // rho_new =  < r_hat , r > = (r_hat)' * (r)
      if (threadIdx.x / config::warp_size == 0) {
        single_rhs_compute_conj_dot(subgroup, num_rows, batch::to_const(r_hat_sh), batch::to_const(r_sh),
                                    rho_new_sh[0]);
      }
      __syncthreads();

      // beta = (rho_new / rho_old)*(alpha / omega)
      // p = r + beta*(p - omega * v)
      update_p(num_rows, rho_new_sh[0], rho_old_sh[0], alpha_sh[0], omega_sh[0], batch::to_const(r_sh),
               batch::to_const(v_sh), p_sh);
      __syncthreads();

      // p_hat = precond * p
      prec_shared.apply(batch::to_const(p_sh), p_hat_sh);
      __syncthreads();

      // v = A * p_hat
      simple_apply(mat_entry, batch::to_const(p_hat_sh), v_sh);
      __syncthreads();

      // alpha = rho_new / < r_hat , v>
      compute_alpha(subgroup, num_rows, rho_new_sh[0], batch::to_const(r_hat_sh), batch::to_const(v_sh), alpha_sh[0]);
      __syncthreads();

      // s = r - alpha*v
      update_s(num_rows, batch::to_const(r_sh), alpha_sh[0], batch::to_const(v_sh), s_sh);
      __syncthreads();

      // an estimate of residual norms
      if (threadIdx.x / config::warp_size == 0) {
        single_rhs_compute_norm2(subgroup, num_rows, batch::to_const(s_sh), norms_res_sh[0]);
      }
      __syncthreads();

      // if (norms_res_sh[0] / norms_rhs_sh[0] < tol) {
      if (stop.check_converged(norms_res_sh)) {
        update_x_middle(num_rows, alpha_sh[0], batch::to_const(p_hat_sh), x_sh);
        logger.log_iteration(batch_id, iter, norms_res_sh[0]);
        break;
      }

      // s_hat = precond * s
      prec_shared.apply(batch::to_const(s_sh), s_hat_sh);
      __syncthreads();

      // t = A * s_hat
      simple_apply(mat_entry, batch::to_const(s_hat_sh), t_sh);
      __syncthreads();

      // omega = <t,s> / <t,t>
      compute_omega(subgroup, num_rows, batch::to_const(t_sh), batch::to_const(s_sh), temp_sh[0], omega_sh[0]);
      __syncthreads();

      // x = x + alpha*p_hat + omega *s_hat
      // r = s - omega * t
      update_x_and_r(num_rows, batch::to_const(p_hat_sh), batch::to_const(s_hat_sh), alpha_sh[0], omega_sh[0],
                     batch::to_const(s_sh), batch::to_const(t_sh), x_sh, r_sh);
      __syncthreads();

      if (threadIdx.x / config::warp_size == 0) {
        single_rhs_compute_norm2(subgroup, num_rows, batch::to_const(r_sh), norms_res_sh[0]);
      }
      //__syncthreads();

      if (threadIdx.x == blockDim.x - 1) { rho_old_sh[0] = rho_new_sh[0]; }
      __syncthreads();
    }

    logger.log_iteration(batch_id, iter, norms_res_sh[0]);

    // copy x back to global memory
    single_rhs_copy(batch::to_const(x_sh), x_entry);
    __syncthreads();
  }
}

} // namespace batch_template::batch_single_kernels::batch_bicgstab
} // namespace gko::kernels::GKO_DEVICE_NAMESPACE
