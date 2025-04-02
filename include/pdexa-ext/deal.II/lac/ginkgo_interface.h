// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dealii_ginkgo_interface_h
#define dealii_ginkgo_interface_h

#include <deal.II/base/array_view.h>
#include <deal.II/base/config.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/extensions/kokkos.hpp>

namespace dealii {
namespace GinkgoInterface {

template<typename ValueType, typename MemorySpace>
std::unique_ptr<gko::matrix::Dense<ValueType>> create_vector(const std::shared_ptr<const gko::Executor>& exec,
                                                             ArrayView<ValueType, MemorySpace>& array) {
  std::shared_ptr<const gko::Executor> array_exec =
    gko::ext::kokkos::create_executor(typename MemorySpace::kokkos_space::execution_space{});

  gko::dim<2> size = {static_cast<gko::size_type>(array.size()), 1};
  return gko::matrix::Dense<ValueType>::create(exec, size, gko::make_array_view(array_exec, size[0], array.data()), 1);
}

template<typename ValueType, typename MemorySpace>
std::unique_ptr<gko::matrix::Dense<ValueType>> create_vector(ArrayView<ValueType, MemorySpace>& array) {
  std::shared_ptr<const gko::Executor> exec =
    gko::ext::kokkos::create_executor(typename MemorySpace::kokkos_space::execution_space{});
  return create_vector(std::move(exec), array);
}

template<typename ValueType, typename MemorySpace>
std::unique_ptr<const gko::matrix::Dense<std::decay_t<ValueType>>>
create_vector(const std::shared_ptr<const gko::Executor>& exec, const ArrayView<ValueType, MemorySpace>& array) {
  std::shared_ptr<const gko::Executor> array_exec =
    gko::ext::kokkos::create_executor(typename MemorySpace::kokkos_space::execution_space{});

  gko::dim<2> size = {static_cast<gko::size_type>(array.size()), 1};
  return gko::matrix::Dense<std::decay_t<ValueType>>::create_const(
    exec, size, gko::make_const_array_view(array_exec, size[0], array.data()), 1);
}

template<typename ValueType, typename MemorySpace>
std::unique_ptr<gko::matrix::Dense<ValueType>>
create_vector(const std::shared_ptr<const gko::Executor>& exec,
              LinearAlgebra::distributed::Vector<ValueType, MemorySpace>& v) {
  Assert(v.locally_owned_size() == v.size(),
         ExcMessage(" distributed vectors are only supported in the GinkgoInterface::MPI namespace "));
  return GinkgoInterface::create_vector(exec, ArrayView<ValueType, MemorySpace>{v.begin(), v.locally_owned_size()});
}

template<typename ValueType, typename MemorySpace>
std::unique_ptr<const gko::matrix::Dense<ValueType>>
create_vector(const std::shared_ptr<const gko::Executor>& exec,
              const LinearAlgebra::distributed::Vector<ValueType, MemorySpace>& v) {
  Assert(v.locally_owned_size() == v.size(),
         ExcMessage(" distributed vectors are only supported in the GinkgoInterface::MPI namespace "));
  return GinkgoInterface::create_vector(exec,
                                        ArrayView<const ValueType, MemorySpace>{v.begin(), v.locally_owned_size()});
}

namespace detail {
template<typename F>
void run_with_device_exec(const std::shared_ptr<const gko::Executor>& exec, F&& f) {
  if (auto p = std::dynamic_pointer_cast<const gko::CudaExecutor>(exec)) { f(std::move(p)); }
  else if (auto p = std::dynamic_pointer_cast<const gko::HipExecutor>(exec)) { f(std::move(p)); }
  else if (auto p = std::dynamic_pointer_cast<const gko::DpcppExecutor>(exec)) { f(std::move(p)); }
  else { Assert(false, ExcMessage(" encountered unknown device Executor type ")); }
}

template<typename Matrix, typename DealValueType, typename... ExtraArgs>
std::unique_ptr<Matrix> create_matrix(const std::shared_ptr<const gko::Executor>& exec,
                                      const SparseMatrix<DealValueType>& deal_csr,
                                      ExtraArgs&&... args) {
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;
  static_assert(std::is_base_of_v<gko::ReadableFromMatrixData<value_type, index_type>, Matrix>);
  gko::dim<2> size = {static_cast<gko::size_type>(deal_csr.m()), static_cast<gko::size_type>(deal_csr.n())};
  gko::matrix_data<value_type, index_type> md{size};
  md.nonzeros.reserve(deal_csr.n_nonzero_elements());
  for (const auto& e: deal_csr) {
    md.nonzeros.emplace_back(static_cast<index_type>(e.row()), static_cast<index_type>(e.column()),
                             static_cast<value_type>(e.value()));
  }
  auto device_md = gko::device_matrix_data<value_type, index_type>::create_from_host(exec, md);
  device_md.sort_row_major();
  auto gko_mtx = Matrix::create(exec, std::forward<ExtraArgs>(args)...);
  gko_mtx->read(std::move(device_md));
  return gko_mtx;
}
} // namespace detail

enum class csr_strategy { exec_based, classical, merge_path, sparselib, load_balance, automatical };

template<typename ValueType, typename IndexType, typename DealValueType>
std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>>
create_csr_matrix(const std::shared_ptr<const gko::Executor>& exec,
                  const SparseMatrix<DealValueType>& deal_csr,
                  csr_strategy strategy = csr_strategy::exec_based) {
  using CsrType = gko::matrix::Csr<ValueType, IndexType>;
  std::shared_ptr<typename CsrType::strategy_type> strategy_ptr;
  if (strategy == csr_strategy::exec_based) {
    strategy = std::dynamic_pointer_cast<const gko::ReferenceExecutor>(exec) ||
                   std::dynamic_pointer_cast<const gko::OmpExecutor>(exec)
                 ? csr_strategy::classical
                 : csr_strategy::automatical;
  }
  switch (strategy) {
    case csr_strategy::classical: strategy_ptr = std::make_shared<typename CsrType::classical>(); break;
    case csr_strategy::merge_path: strategy_ptr = std::make_shared<typename CsrType::merge_path>(); break;
    case csr_strategy::sparselib: strategy_ptr = std::make_shared<typename CsrType::sparselib>(); break;
    case csr_strategy::load_balance:
      detail::run_with_device_exec(exec, [&](auto concrete_exec) {
        strategy_ptr = std::make_shared<typename CsrType::load_balance>(concrete_exec);
      });
      break;
    case csr_strategy::automatical:
      detail::run_with_device_exec(exec, [&](auto concrete_exec) {
        strategy_ptr = std::make_shared<typename CsrType::automatical>(concrete_exec);
      });
      break;
    default: Assert(false, ExcMessage(" encountered unknown Csr strategy type "));
  }
  return detail::create_matrix<CsrType>(std::move(exec), deal_csr, std::move(strategy_ptr));
}

template<typename IndexType = gko::int32, typename DealValueType>
std::unique_ptr<gko::matrix::Csr<DealValueType, IndexType>>
create_csr_matrix(const std::shared_ptr<const gko::Executor>& exec,
                  const SparseMatrix<DealValueType>& deal_csr,
                  csr_strategy strategy = csr_strategy::exec_based) {
  return create_csr_matrix<DealValueType, IndexType>(std::move(exec), deal_csr, strategy);
}

namespace detail {
class GinkgoPayload : public internal::LinearOperatorImplementation::EmptyPayload {};
} // namespace detail

template<typename DomainValueType = double,
         typename RangeValueType = double,
         typename DomainMemorySpace = MemorySpace::Host,
         typename RangeMemorySpace = MemorySpace::Host>
LinearOperator<LinearAlgebra::distributed::Vector<DomainValueType, DomainMemorySpace>,
               LinearAlgebra::distributed::Vector<RangeValueType, RangeMemorySpace>,
               detail::GinkgoPayload>
linear_operator(const std::shared_ptr<const gko::Executor>& domain_exec,
                const std::shared_ptr<const gko::Executor>& range_exec,
                const std::shared_ptr<gko::LinOp>& gko_op) {
  using Domain = LinearAlgebra::distributed::Vector<DomainValueType, DomainMemorySpace>;
  using Range = LinearAlgebra::distributed::Vector<RangeValueType, RangeMemorySpace>;
  using domain_value_type = DomainValueType;
  using range_value_type = RangeValueType;
  using size_type = typename Domain::size_type;
  using value_type = std::common_type_t<domain_value_type, range_value_type>;
  using LinearOperator = LinearOperator<Domain, Range, detail::GinkgoPayload>;
  LinearOperator return_op{detail::GinkgoPayload{}};

  auto exec = gko_op->get_executor();

  auto one = gko::share(gko::initialize<gko::matrix::Dense<value_type>>({1.0}, exec));
  return_op.reinit_domain_vector = [gko_op](Domain& v, bool omit_zeroing_entries) {
    v.reinit(static_cast<size_type>(gko_op->get_size()[1]), omit_zeroing_entries);
  };
  return_op.reinit_range_vector = [gko_op](Range& v, bool omit_zeroing_entries) {
    v.reinit(static_cast<size_type>(gko_op->get_size()[0]), omit_zeroing_entries);
  };
  return_op.vmult = [gko_op, domain_exec, range_exec](Range& v, const Domain& u) {
    auto gko_u = create_vector(domain_exec, u);
    auto gko_v = create_vector(range_exec, v);
    gko_op->apply(gko_u, gko_v);
  };
  return_op.vmult_add = [gko_op, one, domain_exec, range_exec](Range& v, const Domain& u) {
    auto gko_u = create_vector(domain_exec, u);
    auto gko_v = create_vector(range_exec, v);
    gko_op->apply(one, gko_u, one, gko_v);
  };
  return_op.Tvmult = [gko_op, domain_exec, range_exec](Range& v, const Domain& u) {
    auto gko_u = create_vector(domain_exec, u);
    auto gko_v = create_vector(range_exec, v);

    // @todo: this creates a temporary transposed matrix
    gko::as<gko::Transposable>(gko_op)->transpose()->apply(gko_u, gko_v);
  };
  return_op.Tvmult_add = [gko_op, one, domain_exec, range_exec](Range& v, const Domain& u) {
    auto gko_u = create_vector(domain_exec, u);
    auto gko_v = create_vector(range_exec, v);

    // @todo: this creates a temporary transposed matrix
    gko::as<gko::Transposable>(gko_op)->transpose()->apply(one, gko_u, one, gko_v);
  };
  return return_op;
}

template<typename DomainValueType = double,
         typename RangeValueType = double,
         typename DomainMemorySpace = MemorySpace::Host,
         typename RangeMemorySpace = MemorySpace::Host>
LinearOperator<LinearAlgebra::distributed::Vector<DomainValueType, DomainMemorySpace>,
               LinearAlgebra::distributed::Vector<RangeValueType, RangeMemorySpace>,
               detail::GinkgoPayload>
linear_operator(const std::shared_ptr<gko::LinOp>& gko_op) {
  return linear_operator<DomainValueType, RangeValueType, DomainMemorySpace, RangeMemorySpace>(
    gko_op->get_executor(), gko_op->get_executor(), gko_op);
}

namespace MPI {

template<typename ValueType, typename MemorySpace>
std::unique_ptr<gko::experimental::distributed::Vector<ValueType>>
create_vector(const std::shared_ptr<const gko::Executor>& exec,
              LinearAlgebra::distributed::Vector<ValueType, MemorySpace>& v) {
  Assert(!v.has_ghost_elements(), ExcMessage(" Vectors with ghost elements are not supported "));
  return gko::experimental::distributed::Vector<ValueType>::create(
    exec, v.get_mpi_communicator(), gko::dim<2>{v.size(), 1},
    GinkgoInterface::create_vector(exec, ArrayView<ValueType, MemorySpace>{v.begin(), v.locally_owned_size()}));
}

template<typename ValueType, typename MemorySpace>
std::unique_ptr<const gko::experimental::distributed::Vector<ValueType>>
create_vector(const std::shared_ptr<const gko::Executor>& exec,
              const LinearAlgebra::distributed::Vector<ValueType, MemorySpace>& v) {
  Assert(!v.has_ghost_elements(), ExcMessage(" Vectors with ghost elements are not supported "));
  return gko::experimental::distributed::Vector<ValueType>::create_const(
    exec, v.get_mpi_communicator(), gko::dim<2>{v.size(), 1},
    GinkgoInterface::create_vector(exec, ArrayView<const ValueType, MemorySpace>{v.begin(), v.locally_owned_size()}));
}

template<typename DomainValueType = double,
         typename RangeValueType = double,
         typename DomainMemorySpace = MemorySpace::Host,
         typename RangeMemorySpace = MemorySpace::Host>
LinearOperator<LinearAlgebra::distributed::Vector<DomainValueType, DomainMemorySpace>,
               LinearAlgebra::distributed::Vector<RangeValueType, RangeMemorySpace>,
               detail::GinkgoPayload>
linear_operator(const std::shared_ptr<const gko::Executor>& domain_exec,
                const std::shared_ptr<const gko::Executor>& range_exec,
                const std::shared_ptr<gko::LinOp>& gko_op) {
  using Domain = LinearAlgebra::distributed::Vector<DomainValueType, DomainMemorySpace>;
  using Range = LinearAlgebra::distributed::Vector<RangeValueType, RangeMemorySpace>;
  using domain_value_type = DomainValueType;
  using range_value_type = RangeValueType;
  using size_type = typename Domain::size_type;
  using value_type = std::common_type_t<domain_value_type, range_value_type>;
  using LinearOperator = LinearOperator<Domain, Range, detail::GinkgoPayload>;
  LinearOperator return_op{detail::GinkgoPayload{}};

  auto exec = gko_op->get_executor();

  auto one = gko::share(gko::initialize<gko::matrix::Dense<value_type>>({1.0}, exec));
  return_op.reinit_domain_vector = [gko_op](Domain& v, bool omit_zeroing_entries) {
    v.reinit(static_cast<size_type>(gko_op->get_size()[1]), omit_zeroing_entries);
  };
  return_op.reinit_range_vector = [gko_op](Range& v, bool omit_zeroing_entries) {
    v.reinit(static_cast<size_type>(gko_op->get_size()[0]), omit_zeroing_entries);
  };
  return_op.vmult = [gko_op, domain_exec, range_exec](Range& v, const Domain& u) {
    auto gko_u = create_vector(domain_exec, u);
    auto gko_v = create_vector(range_exec, v);
    gko_op->apply(gko_u, gko_v);
  };
  return_op.vmult_add = [gko_op, one, domain_exec, range_exec](Range& v, const Domain& u) {
    auto gko_u = create_vector(domain_exec, u);
    auto gko_v = create_vector(range_exec, v);
    gko_op->apply(one, gko_u, one, gko_v);
  };
  return_op.Tvmult = [gko_op, domain_exec, range_exec](Range& v, const Domain& u) {
    auto gko_u = create_vector(domain_exec, u);
    auto gko_v = create_vector(range_exec, v);

    // @todo: this creates a temporary transposed matrix
    gko::as<gko::Transposable>(gko_op)->transpose()->apply(gko_u, gko_v);
  };
  return_op.Tvmult_add = [gko_op, one, domain_exec, range_exec](Range& v, const Domain& u) {
    auto gko_u = create_vector(domain_exec, u);
    auto gko_v = create_vector(range_exec, v);

    // @todo: this creates a temporary transposed matrix
    gko::as<gko::Transposable>(gko_op)->transpose()->apply(one, gko_u, one, gko_v);
  };
  return return_op;
}

template<typename DomainValueType = double,
         typename RangeValueType = double,
         typename DomainMemorySpace = MemorySpace::Host,
         typename RangeMemorySpace = MemorySpace::Host>
LinearOperator<LinearAlgebra::distributed::Vector<DomainValueType, DomainMemorySpace>,
               LinearAlgebra::distributed::Vector<RangeValueType, RangeMemorySpace>,
               detail::GinkgoPayload>
linear_operator(const std::shared_ptr<gko::LinOp>& gko_op) {
  return linear_operator<DomainValueType, RangeValueType, DomainMemorySpace, RangeMemorySpace>(
    gko_op->get_executor(), gko_op->get_executor(), gko_op);
}

} // namespace MPI

template<typename DomainValueType = double,
         typename RangeValueType = double,
         typename DomainMemorySpace = MemorySpace::Host,
         typename RangeMemorySpace = MemorySpace::Host>
LinearOperator<LinearAlgebra::distributed::Vector<DomainValueType, DomainMemorySpace>,
               LinearAlgebra::distributed::Vector<RangeValueType, RangeMemorySpace>,
               detail::GinkgoPayload>
inverse_operator(const std::shared_ptr<const gko::Executor>& domain_exec,
                 const std::shared_ptr<const gko::Executor>& range_exec,
                 const std::shared_ptr<gko::LinOp>& gko_op,
                 const std::shared_ptr<gko::LinOpFactory>& solver,
                 const std::shared_ptr<gko::log::Logger>& logger = nullptr) {
  auto solver_op = solver->generate(gko_op);
  if (logger) { solver_op->add_logger(logger); }
  if (std::dynamic_pointer_cast<gko::experimental::distributed::DistributedBase>(gko_op)) {
    return MPI::linear_operator<RangeValueType, DomainValueType, RangeMemorySpace, DomainMemorySpace>(
      std::move(solver_op));
  }
  else {
    return linear_operator<RangeValueType, DomainValueType, RangeMemorySpace, DomainMemorySpace>(std::move(solver_op));
  }
}

template<typename DomainValueType = double,
         typename RangeValueType = double,
         typename DomainMemorySpace = MemorySpace::Host,
         typename RangeMemorySpace = MemorySpace::Host>
LinearOperator<LinearAlgebra::distributed::Vector<DomainValueType, DomainMemorySpace>,
               LinearAlgebra::distributed::Vector<RangeValueType, RangeMemorySpace>,
               detail::GinkgoPayload>
inverse_operator(const std::shared_ptr<gko::LinOp>& gko_op,
                 const std::shared_ptr<gko::LinOpFactory>& solver,
                 const std::shared_ptr<gko::log::Logger>& logger = nullptr) {
  return inverse_operator<DomainValueType, RangeValueType, DomainMemorySpace, RangeMemorySpace>(
    gko_op->get_executor(), gko_op->get_executor(), gko_op, solver, logger);
}

} // namespace GinkgoInterface
} // namespace dealii

#endif // dealii_ginkgo_interface_h
