// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include "cellwise_operator.hpp"
#include "cellwise_preconditioner.hpp"

template<int dim, int fe_degree, typename Number, int vector_size>
void apply_gko_solver(
  std::shared_ptr<DGAdvection::GkoCellwiseOperator<dim, fe_degree, Number, vector_size>> gko_cellwise_op,
  std::shared_ptr<DGAdvection::CellwisePreconditionerFDM<dim, fe_degree, Number, vector_size>> gko_cellwise_precond,
  std::shared_ptr<gko::batch::MultiVector<Number>> gko_src,
  std::shared_ptr<gko::batch::MultiVector<Number>> gko_dst);
