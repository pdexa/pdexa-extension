// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/solver/batch_solver_base.hpp>

#include "pdexa-ext/ginkgo/core/base/view.hpp"

#include "pdexa-ext/ginkgo/core/solver/batch_bicgstab_settings.hpp"
#include "pdexa-ext/ginkgo/core/solver/batch_solver_base.hpp"

#include "pdexa-ext/ginkgo/backend/cuda/batch_bicgstab_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/hip/batch_bicgstab_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/omp/batch_bicgstab_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/reference/batch_bicgstab_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/sycl/batch_bicgstab_kernels.hpp"

namespace gko::batch_template::solver {
namespace bicgstab {

GKO_REGISTER_OPERATION(apply, batch_template::batch_bicgstab::apply);

}

template<typename Op, typename Prec>
class Bicgstab;

namespace bicgstab {

template<typename Prec>
class Factory;

template<typename Prec>
struct parameters_type {
  using factory = Factory<Prec>;

  parameters_type* self() noexcept { return this; }

  const parameters_type* self() const noexcept { return this; }

  /**
   * Default maximum number iterations allowed.
   *
   * Generated solvers are initialized with this value for their maximum
   * iterations.
   */
  int GKO_FACTORY_PARAMETER_SCALAR(max_iterations, 100);

  /**
   * Default residual tolerance.
   *
   * Generated solvers are initialized with this value for their residual
   * tolerance.
   */
  double GKO_FACTORY_PARAMETER_SCALAR(tolerance, 1e-11);

  /**
   * To specify which type of tolerance check is to be considered, absolute or
   * relative (to the rhs l2 norm)
   */
  ::gko::batch::stop::tolerance_type GKO_FACTORY_PARAMETER_SCALAR(tolerance_type,
                                                                  ::gko::batch::stop::tolerance_type::absolute);

  std::shared_ptr<const Prec> generated_preconditioner;

  /**
   * Already generated preconditioner. If one is provided, the factory
   * `preconditioner` will be ignored.
   */
  template<typename Prec2>
  [[nodiscard]] auto with_generated_preconditioner(std::shared_ptr<const Prec2> prec) -> parameters_type<Prec2> {
    return {max_iterations, tolerance, tolerance_type, prec};
  }

  [[nodiscard]] std::unique_ptr<factory> on(std::shared_ptr<const Executor> exec) const {
    auto copy = *this;
    auto result = std::unique_ptr<factory>(new factory(exec, copy));
    return result;
  }
};

template<typename Prec>
class Factory : public log::EnableLogging<Factory<Prec>> {
public:
  explicit Factory(std::shared_ptr<const Executor> exec) : exec_(std::move(exec)) {}

  explicit Factory(std::shared_ptr<const Executor> exec, const parameters_type<Prec>& parameters) :
      exec_(std::move(exec)), parameters_(parameters) {}

  template<typename Op>
  [[nodiscard]] std::unique_ptr<Bicgstab<Op, Prec>> generate(std::shared_ptr<Op> input) const {
    if (input->get_executor() != exec_) { input = gko::clone(exec_, input); }
    return std::unique_ptr<Bicgstab<Op, Prec>>(new Bicgstab<Op, Prec>(this, input));
  }

  const parameters_type<Prec>& get_parameters() const noexcept { return parameters_; };

  auto get_executor() const { return exec_; }

private:
  std::shared_ptr<const Executor> exec_;
  parameters_type<Prec> parameters_;
};

static auto build() -> parameters_type<batch_preconditioner::Identity<default_precision>> { return {}; }

} // namespace bicgstab

template<typename T, typename Prec>
class Bicgstab final : public BatchSolver<T>, public batch::EnableBatchLinOp<Bicgstab<T, Prec>> {
  friend class batch::EnableBatchLinOp<Bicgstab>;
  friend class EnablePolymorphicObject<Bicgstab, batch::BatchLinOp>;
  friend bicgstab::Factory<Prec>;

public:
  using value_type = typename T::value_type;
  using real_type = remove_complex<value_type>;

  void apply(ptr_param<const batch::MultiVector<value_type>> b, ptr_param<batch::MultiVector<value_type>> x) const {
    // this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    auto required_ws_size = b->get_num_batch_items() * b->get_common_size()[1] * 32;
    if (workspace_.get_size() < required_ws_size) { workspace_.resize_and_reset(required_ws_size); }
    auto view = workspace_.as_view();
    auto log_data_ = std::make_unique<batch::log::detail::log_data<real_type>>(
      exec, b->get_num_batch_items() * b->get_common_size()[1], view);
    this->solver_apply(make_temporary_clone(exec, b).get(), make_temporary_clone(exec, x).get(), log_data_.get());
    this->template log<log::Logger::batch_solver_completed>(log_data_->iter_counts, log_data_->res_norms);
  }

private:
  explicit Bicgstab(std::shared_ptr<const Executor> exec) : batch::EnableBatchLinOp<Bicgstab>(std::move(exec)) {}

  explicit Bicgstab(const bicgstab::Factory<Prec>* factory, std::shared_ptr<const T> system_matrix) :
      BatchSolver<T>(system_matrix,
                     factory->get_parameters().tolerance,
                     factory->get_parameters().max_iterations,
                     factory->get_parameters().tolerance_type),
      batch::EnableBatchLinOp<Bicgstab>(factory->get_executor(), transpose(system_matrix->get_size())),
      parameters_{factory->get_parameters()},
      workspace_(factory->get_executor(), system_matrix->get_num_batch_items() * 32) {
    if (!parameters_.generated_preconditioner) {
      if constexpr (std::is_same_v<Prec, batch_preconditioner::Identity<value_type>>) {
        parameters_.generated_preconditioner = std::make_shared<batch_preconditioner::Identity<value_type>>();
      }
      else { GKO_NOT_IMPLEMENTED; }
    }
  }

  void solver_apply(const batch::MultiVector<value_type>* b,
                    batch::MultiVector<value_type>* x,
                    batch::log::detail::log_data<real_type>* log_data) const {
    const kernels::batch_bicgstab::settings<remove_complex<value_type>> settings{
      this->max_iterations_, static_cast<real_type>(this->residual_tol_), parameters_.tolerance_type};
    auto exec = this->get_executor();
    exec->run(bicgstab::make_apply(settings, batch::create_view(this->system_matrix_.get()),
                                   batch::create_view(parameters_.generated_preconditioner.get()),
                                   batch::create_view(b), batch::create_view(x), *log_data));
  }

  bicgstab::parameters_type<Prec> parameters_;
  mutable array<unsigned char> workspace_;
};

} // namespace gko::batch_template::solver
