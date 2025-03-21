// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"


namespace gko::batch_preconditioner {


/**
 * Identity preconditioner for batch solvers. Enables unpreconditioned solves
 * by performing a copy of the preconditioned vector to the un-preconditioned
 * vector.
 */
template <typename ValueType>
class Identity final {
public:
    using value_type = ValueType;

    /**
     * The size of the work vector required in case of static allocation.
     */
    static constexpr int work_size = 0;

    /**
     * The size of the work vector required in case of dynamic allocation in
     * bytes.
     */
    template<typename OpType>
    static constexpr int dynamic_work_size(int, OpType) { return 0; }

    /**
     * Sets the input and generates the identity preconditioner.(Nothing needs
     * to be actually generated.)
     */
    template <typename batch_item_type>
    constexpr void generate(int64, const batch_item_type&, ValueType* const)
    {}

    /**
     * Applies the preconditioner to the vector. For the identity
     * preconditioner, this is equivalent to a copy.
     */
    constexpr void apply(batch::multi_vector::batch_item<const ValueType> r,
                         batch::multi_vector::batch_item<ValueType> z) const
    {
        for (int i = 0; i < r.num_rows; i++) {
            z.values[i * z.stride] = r.values[i * r.stride];
        }
    }
};


}  // namespace gko::batch_preconditioner
