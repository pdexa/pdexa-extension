// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/solver/batch_solver_base.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


#include <pdexa-ext/ginkgo/core/solver/batch_solver_base.hpp>
#include <pdexa-ext/ginkgo/core/solver/batch_cg_settings.hpp>

#include <pdexa-ext/ginkgo/core/base/batch_multi_vector.hpp>
#include <pdexa-ext/ginkgo/core/matrix/batch_csr.hpp>

#include <pdexa-ext/ginkgo/kernels/base/batch_struct.hpp>

#include <pdexa-ext/ginkgo/kernels/cuda/solver/batch_cg_kernels.hpp>
#include <pdexa-ext/ginkgo/kernels/hip/solver/batch_cg_kernels.hpp>
#include <pdexa-ext/ginkgo/kernels/omp/solver/batch_cg_kernels.hpp>
#include <pdexa-ext/ginkgo/kernels/reference/solver/batch_cg_kernels.hpp>
#include <pdexa-ext/ginkgo/kernels/sycl/solver/batch_cg_kernels.hpp>


namespace gko::batch_template::solver {
namespace cg {
GKO_REGISTER_OPERATION(apply, batch_template::batch_cg::apply);
}


template<typename T>
class Cg final : public EnableBatchSolver<Cg<T>, T> {
    friend class EnablePolymorphicObject<Cg, BatchSolver<typename T::value_type>>;

public:
    using op_type = T;
    using value_type = typename T::value_type;
    using real_type = remove_complex<value_type>;

    class Factory;

    struct parameters_type
        : batch::solver::
        enable_preconditioned_iterative_solver_factory_parameters<
            parameters_type, Factory> {};

    class Factory : public log::EnableLogging<Factory> {
    public:
        explicit Factory(std::shared_ptr<const Executor> exec)
            : exec_(std::move(exec)) {}

        explicit Factory(std::shared_ptr<const Executor> exec,
                         const parameters_type& parameters)
            : exec_(std::move(exec))
            , parameters_(parameters) {}

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

protected:
    explicit Cg(std::shared_ptr<const Executor> exec)
        : EnableBatchSolver<Cg, T>(std::move(exec)) {}

    explicit Cg(const Factory* factory, std::shared_ptr<const T> system_matrix)
        : EnableBatchSolver<Cg, T>(system_matrix,
                                   factory->get_parameters().tolerance,
                                   factory->get_parameters().max_iterations,
                                   factory->get_parameters().tolerance_type) {}

    void solver_apply(const batch::MultiVector<value_type>* b,
                      batch::MultiVector<value_type>* x,
                      batch::log::detail::log_data<real_type>* log_data) const override {
        const kernels::batch_cg::settings<remove_complex<value_type>> settings{
            this->get_max_iterations(),
            static_cast<real_type>(this->get_tolerance()),
            this->get_tolerance_type()
        };
        auto exec = this->get_executor();
        using batch::create_view;
        exec->run(
            cg::make_apply(settings,
                           create_view(this->get_system_matrix().get()),
                           create_view(b),
                           create_view(x),
                           *log_data));
    }
};
} // namespace gko::batch_template::solver
