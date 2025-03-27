// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/batch_lin_op.hpp>

#include "pdexa-ext/ginkgo/backend/cuda/batch_user_linop.hpp"
#include "pdexa-ext/ginkgo/backend/hip/batch_user_linop.hpp"
#include "pdexa-ext/ginkgo/backend/omp/batch_user_linop.hpp"
#include "pdexa-ext/ginkgo/backend/reference/batch_user_linop.hpp"
#include "pdexa-ext/ginkgo/backend/sycl/batch_user_linop.hpp"

namespace gko::batch_template {
namespace detail::user {

GKO_REGISTER_OPERATION(simple_apply, batch_template::batch_user::apply);

}

template <typename ValueType, typename Concrete>
class EnableBatchUserLinOp : public batch::EnableBatchLinOp<Concrete> {
public:
    using value_type = ValueType;

    void apply(ptr_param<const batch::MultiVector<value_type>> b,
               ptr_param<batch::MultiVector<value_type>> x) const
    {
        this->validate_application_parameters(b.get(), x.get());
        auto exec = self()->get_executor();
        exec->run(detail::user::make_simple_apply(
            self()->create_view(), b->create_view(), x->create_view()));
    }

protected:
    explicit EnableBatchUserLinOp(std::shared_ptr<const Executor> exec)
        : batch::EnableBatchLinOp<Concrete>(std::move(exec))
    {}

    explicit EnableBatchUserLinOp(std::shared_ptr<const Executor> exec,
                                  batch_dim<2> size)
        : batch::EnableBatchLinOp<Concrete>(std::move(exec), size)
    {}

private:
    GKO_ENABLE_SELF(Concrete);
};

}  // namespace gko::batch_template
