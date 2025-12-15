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

#include "pdexa-ext/ginkgo/core/base/view.hpp"

#include "pdexa-ext/ginkgo/core/solver/batch_cg_settings.hpp"
#include "pdexa-ext/ginkgo/core/solver/batch_solver_base.hpp"

#include "pdexa-ext/ginkgo/backend/cuda/batch_cg_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/hip/batch_cg_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/omp/batch_cg_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/reference/batch_cg_kernels.hpp"
#include "pdexa-ext/ginkgo/backend/sycl/batch_cg_kernels.hpp"

namespace gko::batch_template::solver {
namespace cg {

GKO_REGISTER_OPERATION(apply, batch_template::batch_cg::apply);

}

template<typename T>
class Cg final : public BatchSolver<T>, public batch::EnableBatchLinOp<Cg<T>> {
  friend class batch::EnableBatchLinOp<Cg>;
  friend class EnablePolymorphicObject<Cg, batch::BatchLinOp>;

public:
  using value_type = typename T::value_type;
  using real_type = remove_complex<value_type>;

  class Factory;

  struct parameters_type
      : batch::solver::enable_preconditioned_iterative_solver_factory_parameters<parameters_type, Factory> {};

  const parameters_type& get_parameters() const { return parameters_; }

  class Factory : public log::EnableLogging<Factory> {
  public:
    explicit Factory(std::shared_ptr<const Executor> exec) : exec_(std::move(exec)) {}

    explicit Factory(std::shared_ptr<const Executor> exec, const parameters_type& parameters) :
        exec_(std::move(exec)), parameters_(parameters) {}

    std::unique_ptr<Cg> generate(std::shared_ptr<T> input) const {
      if (input->get_executor() != exec_) { input = gko::clone(exec_, input); }
      return std::unique_ptr<Cg>(new Cg(this, input));
    }

    const parameters_type& get_parameters() const noexcept { return parameters_; };

    auto get_executor() const { return exec_; }

  private:
    std::shared_ptr<const Executor> exec_;
    parameters_type parameters_;
  };

  static auto build() -> parameters_type { return {}; }

  void apply(ptr_param<const batch::MultiVector<value_type>> b, ptr_param<batch::MultiVector<value_type>> x) const {
    // this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    auto view = workspace_.as_view();
    auto log_data_ = std::make_unique<batch::log::detail::log_data<real_type>>(exec, b->get_num_batch_items(), view);
    this->solver_apply(make_temporary_clone(exec, b).get(), make_temporary_clone(exec, x).get(), log_data_.get());
    this->template log<log::Logger::batch_solver_completed>(log_data_->iter_counts, log_data_->res_norms);
  }

private:
  explicit Cg(std::shared_ptr<const Executor> exec) : batch::EnableBatchLinOp<Cg>(std::move(exec)) {}

  explicit Cg(const Factory* factory, std::shared_ptr<const T> system_matrix) :
      BatchSolver<T>(system_matrix,
                     factory->get_parameters().tolerance,
                     factory->get_parameters().max_iterations,
                     factory->get_parameters().tolerance_type),
      batch::EnableBatchLinOp<Cg>(factory->get_executor(), transpose(system_matrix->get_size())),
      parameters_{factory->get_parameters()},
      workspace_(factory->get_executor(), system_matrix->get_num_batch_items() * 32) {}

  void solver_apply(const batch::MultiVector<value_type>* b,
                    batch::MultiVector<value_type>* x,
                    batch::log::detail::log_data<real_type>* log_data) const {
    const kernels::batch_cg::settings<remove_complex<value_type>> settings{
      this->max_iterations_, static_cast<real_type>(this->residual_tol_), parameters_.tolerance_type};
    auto exec = this->get_executor();
    exec->run(cg::make_apply(settings, batch::create_view(this->system_matrix_.get()), batch::create_view(b),
                             batch::create_view(x), *log_data));
  }

  parameters_type parameters_;
  mutable array<unsigned char> workspace_;
};

} // namespace gko::batch_template::solver
