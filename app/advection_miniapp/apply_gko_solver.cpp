// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "apply_gko_solver.hpp"

#include "cellwise_operator.apply.hpp"
#include "cellwise_preconditioner.apply.hpp"
#include "pdexa-ext/ginkgo/core/solver/batch_bicgstab.hpp"

template<int dim, int fe_degree, typename Number, int vector_size>
void apply_gko_solver(
  std::shared_ptr<DGAdvection::GkoCellwiseOperator<dim, fe_degree, Number, vector_size>> gko_cellwise_op,
  std::shared_ptr<DGAdvection::CellwisePreconditionerFDM<dim, fe_degree, Number, vector_size>> gko_cellwise_precond,
  std::shared_ptr<gko::batch::MultiVector<Number>> gko_src,
  std::shared_ptr<gko::batch::MultiVector<Number>> gko_dst) {
  auto exec = gko_cellwise_op->get_executor();

  auto num_batches = gko_cellwise_op->get_num_batch_items();
  auto num_rows = gko_cellwise_op->get_common_size()[0];

  auto gko_solver =
    gko::batch_template::solver::bicgstab::build()
      .with_max_iterations(20u)
      .with_tolerance(1e-14)
      .with_generated_preconditioner(
        std::const_pointer_cast<const DGAdvection::CellwisePreconditionerFDM<dim, fe_degree, Number, vector_size>>(
          gko_cellwise_precond))
      .on(exec)
      ->generate(gko_cellwise_op);

  gko_solver->apply(gko_src, gko_dst);
}

#define DECLARE_APPLY_GKO_SOLVER(_dim, _vs)                                                                            \
  template void apply_gko_solver<_dim, DGAdvection::fe_degree, DGAdvection::Number, _vs>(                              \
    std::shared_ptr<DGAdvection::GkoCellwiseOperator<_dim, DGAdvection::fe_degree, DGAdvection::Number, _vs>>          \
      gko_cellwise_op,                                                                                                 \
    std::shared_ptr<DGAdvection::CellwisePreconditionerFDM<_dim, DGAdvection::fe_degree, DGAdvection::Number, _vs>>    \
      gko_cellwise_precond,                                                                                            \
    std::shared_ptr<gko::batch::MultiVector<DGAdvection::Number>> gko_src,                                       \
    std::shared_ptr<gko::batch::MultiVector<DGAdvection::Number>> gko_dst)

DECLARE_APPLY_GKO_SOLVER(2, 4);
DECLARE_APPLY_GKO_SOLVER(2, 2);
DECLARE_APPLY_GKO_SOLVER(2, 1);
DECLARE_APPLY_GKO_SOLVER(3, 4);
DECLARE_APPLY_GKO_SOLVER(3, 2);
DECLARE_APPLY_GKO_SOLVER(3, 1);
