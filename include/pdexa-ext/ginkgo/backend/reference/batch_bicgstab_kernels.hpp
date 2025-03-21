// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#if PDEXA_EXT_ENABLE_REFERENCE

#include <ginkgo/core/log/batch_logger.hpp>

#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"
#include "pdexa-ext/ginkgo/core/matrix/batch_struct.hpp"
#include "pdexa-ext/ginkgo/core/solver/batch_bicgstab_settings.hpp"

#include "batch_apply.hpp"
#include "batch_multi_vector_kernels.hpp"
#include "pdexa-ext/ginkgo/core/log/batch_simple_logger.hpp"
#include "pdexa-ext/ginkgo/core/preconditioner/batch_identity.hpp"
#include "pdexa-ext/ginkgo/core/stop/batch_criteria.hpp"

namespace gko::kernels::reference::batch_template {
namespace batch_single_kernels {
constexpr int max_num_rhs = 1;

template<typename BatchMatrixType_entry, typename ValueType>
inline void initialize(const BatchMatrixType_entry& A_entry,
                       const batch::multi_vector::batch_item<const ValueType>& b_entry,
                       const batch::multi_vector::batch_item<const ValueType>& x_entry,
                       const batch::multi_vector::batch_item<ValueType>& rho_old_entry,
                       const batch::multi_vector::batch_item<ValueType>& omega_entry,
                       const batch::multi_vector::batch_item<ValueType>& alpha_entry,
                       const batch::multi_vector::batch_item<ValueType>& r_entry,
                       const batch::multi_vector::batch_item<ValueType>& r_hat_entry,
                       const batch::multi_vector::batch_item<ValueType>& p_entry,
                       const batch::multi_vector::batch_item<ValueType>& p_hat_entry,
                       const batch::multi_vector::batch_item<ValueType>& v_entry,
                       const batch::multi_vector::batch_item<remove_complex<ValueType>>& rhs_norms_entry,
                       const batch::multi_vector::batch_item<remove_complex<ValueType>>& res_norms_entry) {
  rho_old_entry.values[0] = one<ValueType>();
  omega_entry.values[0] = one<ValueType>();
  alpha_entry.values[0] = one<ValueType>();

  // Compute norms of rhs
  compute_norm2_kernel<ValueType>(b_entry, rhs_norms_entry);

  // r = b
  copy_kernel(b_entry, r_entry);

  // r = b - A*x
  advanced_apply(static_cast<ValueType>(-1.0), A_entry, batch::to_const(x_entry), static_cast<ValueType>(1.0), r_entry);
  compute_norm2_kernel<ValueType>(batch::to_const(r_entry), res_norms_entry);

  for (int r = 0; r < p_entry.num_rows; r++) {
    r_hat_entry.values[r * r_hat_entry.stride] = r_entry.values[r * r_entry.stride];
    p_entry.values[r * p_entry.stride] = zero<ValueType>();
    p_hat_entry.values[r * p_hat_entry.stride] = zero<ValueType>();
    v_entry.values[r * v_entry.stride] = zero<ValueType>();
  }
}

template<typename ValueType>
inline void update_p(const batch::multi_vector::batch_item<const ValueType>& rho_new_entry,
                     const batch::multi_vector::batch_item<const ValueType>& rho_old_entry,
                     const batch::multi_vector::batch_item<const ValueType>& alpha_entry,
                     const batch::multi_vector::batch_item<const ValueType>& omega_entry,
                     const batch::multi_vector::batch_item<const ValueType>& r_entry,
                     const batch::multi_vector::batch_item<const ValueType>& v_entry,
                     const batch::multi_vector::batch_item<ValueType>& p_entry) {
  const ValueType beta =
    (rho_new_entry.values[0] / rho_old_entry.values[0]) * (alpha_entry.values[0] / omega_entry.values[0]);
  for (int r = 0; r < p_entry.num_rows; r++) {
    p_entry.values[r * p_entry.stride] =
      r_entry.values[r * r_entry.stride] +
      beta * (p_entry.values[r * p_entry.stride] - omega_entry.values[0] * v_entry.values[r * v_entry.stride]);
  }
}

template<typename ValueType>
inline void compute_alpha(const batch::multi_vector::batch_item<const ValueType>& rho_new_entry,
                          const batch::multi_vector::batch_item<const ValueType>& r_hat_entry,
                          const batch::multi_vector::batch_item<const ValueType>& v_entry,
                          const batch::multi_vector::batch_item<ValueType>& alpha_entry) {
  compute_dot_product_kernel<ValueType>(r_hat_entry, v_entry, alpha_entry);
  alpha_entry.values[0] = rho_new_entry.values[0] / alpha_entry.values[0];
}

template<typename ValueType>
inline void update_s(const batch::multi_vector::batch_item<const ValueType>& r_entry,
                     const batch::multi_vector::batch_item<const ValueType>& alpha_entry,
                     const batch::multi_vector::batch_item<const ValueType>& v_entry,
                     const batch::multi_vector::batch_item<ValueType>& s_entry) {
  for (int r = 0; r < s_entry.num_rows; r++) {
    s_entry.values[r * s_entry.stride] =
      r_entry.values[r * r_entry.stride] - alpha_entry.values[0] * v_entry.values[r * v_entry.stride];
  }
}

template<typename ValueType>
inline void compute_omega(const batch::multi_vector::batch_item<const ValueType>& t_entry,
                          const batch::multi_vector::batch_item<const ValueType>& s_entry,
                          const batch::multi_vector::batch_item<ValueType>& temp_entry,
                          const batch::multi_vector::batch_item<ValueType>& omega_entry) {
  compute_dot_product_kernel<ValueType>(t_entry, s_entry, omega_entry);
  compute_dot_product_kernel<ValueType>(t_entry, t_entry, temp_entry);
  omega_entry.values[0] /= temp_entry.values[0];
}

template<typename ValueType>
inline void update_x_and_r(const batch::multi_vector::batch_item<const ValueType>& p_hat_entry,
                           const batch::multi_vector::batch_item<const ValueType>& s_hat_entry,
                           const batch::multi_vector::batch_item<const ValueType>& alpha_entry,
                           const batch::multi_vector::batch_item<const ValueType>& omega_entry,
                           const batch::multi_vector::batch_item<const ValueType>& s_entry,
                           const batch::multi_vector::batch_item<const ValueType>& t_entry,
                           const batch::multi_vector::batch_item<ValueType>& x_entry,
                           const batch::multi_vector::batch_item<ValueType>& r_entry) {
  const ValueType omega = omega_entry.values[0];
  for (int r = 0; r < x_entry.num_rows; r++) {
    x_entry.values[r * x_entry.stride] = x_entry.values[r * x_entry.stride] +
                                         alpha_entry.values[0] * p_hat_entry.values[r * p_hat_entry.stride] +
                                         omega * s_hat_entry.values[r * s_hat_entry.stride];

    r_entry.values[r * r_entry.stride] =
      s_entry.values[r * s_entry.stride] - omega * t_entry.values[r * t_entry.stride];
  }
}

template<typename ValueType>
inline void update_x_middle(const batch::multi_vector::batch_item<const ValueType>& alpha_entry,
                            const batch::multi_vector::batch_item<const ValueType>& p_hat_entry,
                            const batch::multi_vector::batch_item<ValueType>& x_entry) {
  for (int r = 0; r < x_entry.num_rows; r++) {
    x_entry.values[r * x_entry.stride] =
      x_entry.values[r * x_entry.stride] + alpha_entry.values[0] * p_hat_entry.values[r * p_hat_entry.stride];
  }
}

template<typename StopType, typename PrecType, typename LogType, typename BatchMatrixType, typename ValueType>
inline void batch_entry_bicgstab_impl(const kernels::batch_bicgstab::settings<remove_complex<ValueType>>& settings,
                                      LogType logger,
                                      PrecType prec,
                                      const BatchMatrixType& a,
                                      batch::multi_vector::batch_item<const ValueType> b_entry,
                                      batch::multi_vector::batch_item<ValueType> x_entry,
                                      const size_type batch_item_id,
                                      unsigned char* const local_space) {
  using real_type = remove_complex<ValueType>;
  const auto num_rows = a.num_rows;
  const auto num_rhs = b_entry.num_rhs;
  GKO_ASSERT(num_rhs <= max_num_rhs);

  unsigned char* const shared_space = local_space;
  ValueType* const r = reinterpret_cast<ValueType*>(shared_space);
  ValueType* const r_hat = r + num_rows * num_rhs;
  ValueType* const p = r_hat + num_rows * num_rhs;
  ValueType* const p_hat = p + num_rows * num_rhs;
  ValueType* const v = p_hat + num_rows * num_rhs;
  ValueType* const s = v + num_rows * num_rhs;
  ValueType* const s_hat = s + num_rows * num_rhs;
  ValueType* const t = s_hat + num_rows * num_rhs;
  ValueType* const prec_work = t + num_rows * num_rhs;
  ValueType rho_old[max_num_rhs];
  ValueType rho_new[max_num_rhs];
  ValueType omega[max_num_rhs];
  ValueType alpha[max_num_rhs];
  ValueType temp[max_num_rhs];
  real_type norms_rhs[max_num_rhs];
  real_type norms_res[max_num_rhs];

  const auto A_entry = batch::extract_batch_item(a, batch_item_id);

  const batch::multi_vector::batch_item<ValueType> r_entry{r, num_rhs, num_rows, num_rhs};
  const batch::multi_vector::batch_item<ValueType> r_hat_entry{r_hat, num_rhs, num_rows, num_rhs};
  const batch::multi_vector::batch_item<ValueType> p_entry{p, num_rhs, num_rows, num_rhs};
  const batch::multi_vector::batch_item<ValueType> p_hat_entry{p_hat, num_rhs, num_rows, num_rhs};
  const batch::multi_vector::batch_item<ValueType> v_entry{v, num_rhs, num_rows, num_rhs};
  const batch::multi_vector::batch_item<ValueType> s_entry{s, num_rhs, num_rows, num_rhs};
  const batch::multi_vector::batch_item<ValueType> s_hat_entry{s_hat, num_rhs, num_rows, num_rhs};
  const batch::multi_vector::batch_item<ValueType> t_entry{t, num_rhs, num_rows, num_rhs};
  const batch::multi_vector::batch_item<ValueType> rho_old_entry{rho_old, num_rhs, 1, num_rhs};
  const batch::multi_vector::batch_item<ValueType> rho_new_entry{rho_new, num_rhs, 1, num_rhs};
  const batch::multi_vector::batch_item<ValueType> omega_entry{omega, num_rhs, 1, num_rhs};
  const batch::multi_vector::batch_item<ValueType> alpha_entry{alpha, num_rhs, 1, num_rhs};
  const batch::multi_vector::batch_item<ValueType> temp_entry{temp, num_rhs, 1, num_rhs};
  const batch::multi_vector::batch_item<real_type> rhs_norms_entry{norms_rhs, num_rhs, 1, num_rhs};
  const batch::multi_vector::batch_item<real_type> res_norms_entry{norms_res, num_rhs, 1, num_rhs};

  // generate preconditioner
  prec.generate(batch_item_id, A_entry, prec_work);

  // initialization
  // rho_old = 1, omega = 1, alpha = 1
  // compute b norms
  // r = b - A*x
  // compute residual norms
  // r_hat = r
  // p = 0
  // p_hat = 0
  // v = 0
  initialize(A_entry, b_entry, batch::to_const(x_entry), rho_old_entry, omega_entry, alpha_entry, r_entry, r_hat_entry,
             p_entry, p_hat_entry, v_entry, rhs_norms_entry, res_norms_entry);

  // stopping criterion object
  StopType stop(settings.residual_tol, rhs_norms_entry.values);

  int iter{};

  for (iter = 0; iter < settings.max_iterations; iter++) {
    if (stop.check_converged(res_norms_entry.values)) {
      logger.log_iteration(batch_item_id, iter, res_norms_entry.values[0]);
      break;
    }

    // rho_new =  < r_hat , r > = (r_hat)' * (r)
    compute_dot_product_kernel<ValueType>(batch::to_const(r_hat_entry), batch::to_const(r_entry), rho_new_entry);

    // beta = (rho_new / rho_old)*(alpha / omega)
    // p = r + beta*(p - omega * v)
    update_p(batch::to_const(rho_new_entry), batch::to_const(rho_old_entry), batch::to_const(alpha_entry),
             batch::to_const(omega_entry), batch::to_const(r_entry), batch::to_const(v_entry), p_entry);

    // p_hat = precond * p
    prec.apply(batch::to_const(p_entry), p_hat_entry);

    // v = A * p_hat
    simple_apply(A_entry, batch::to_const(p_hat_entry), v_entry);

    // alpha = rho_new / < r_hat , v>
    compute_alpha(batch::to_const(rho_new_entry), batch::to_const(r_hat_entry), batch::to_const(v_entry), alpha_entry);

    // s = r - alpha*v
    update_s(batch::to_const(r_entry), batch::to_const(alpha_entry), batch::to_const(v_entry), s_entry);

    // an estimate of residual norms
    compute_norm2_kernel<ValueType>(batch::to_const(s_entry), res_norms_entry);

    if (stop.check_converged(res_norms_entry.values)) {
      // update x for the systems
      // x = x + alpha * p_hat
      update_x_middle(batch::to_const(alpha_entry), batch::to_const(p_hat_entry), x_entry);
      logger.log_iteration(batch_item_id, iter, res_norms_entry.values[0]);
      break;
    }

    // s_hat = precond * s
    prec.apply(batch::to_const(s_entry), s_hat_entry);

    // t = A * s_hat
    simple_apply(A_entry, batch::to_const(s_hat_entry), t_entry);
    // omega = <t,s> / <t,t>
    compute_omega(batch::to_const(t_entry), batch::to_const(s_entry), temp_entry, omega_entry);

    // x = x + alpha * p_hat + omega * s_hat
    // r = s - omega * t
    update_x_and_r(batch::to_const(p_hat_entry), batch::to_const(s_hat_entry), batch::to_const(alpha_entry),
                   batch::to_const(omega_entry), batch::to_const(s_entry), batch::to_const(t_entry), x_entry, r_entry);

    compute_norm2_kernel<ValueType>(batch::to_const(r_entry), res_norms_entry);

    // rho_old = rho_new
    copy_kernel(batch::to_const(rho_new_entry), rho_old_entry);
  }

  logger.log_iteration(batch_item_id, iter, res_norms_entry.values[0]);
}

} // namespace batch_bicgstab

namespace batch_bicgstab {

template<typename ValueType, typename Op>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const kernels::batch_bicgstab::settings<remove_complex<ValueType>>& options,
           const Op mat,
           batch::multi_vector::uniform_batch<const ValueType> b,
           batch::multi_vector::uniform_batch<ValueType> x,
           batch::log::detail::log_data<remove_complex<ValueType>>& logdata) {
  using real_type = remove_complex<ValueType>;
  const auto num_batch_items = static_cast<int64>(mat.num_batch_items);
  const auto num_rows = mat.num_rows;
  const auto num_rhs = b.num_rhs;
  if (num_rhs > 1) { GKO_NOT_IMPLEMENTED; }

  auto local_size_bytes = static_cast<size_type>(kernels::batch_bicgstab::local_memory_requirement<ValueType>(num_rows, num_rhs));
  array<unsigned char> local_space(exec, local_size_bytes);

  batch_log::SimpleFinalLogger<real_type> logger(logdata.res_norms.get_data(), logdata.iter_counts.get_data());

  auto prec = batch_preconditioner::Identity<ValueType>();

  for (int64 batch_id = 0; batch_id < num_batch_items; batch_id++) {
    batch_single_kernels::batch_entry_bicgstab_impl<batch_stop::SimpleRelResidual<ValueType>>(
      options, logger, prec, mat, batch::extract_batch_item(b, batch_id), batch::extract_batch_item(x, batch_id),
      batch_id, local_space.get_data());
  }
}

}
} // namespace gko::kernels::reference::batch_template

#endif
