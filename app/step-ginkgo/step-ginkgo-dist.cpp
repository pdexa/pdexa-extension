/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2010 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Authors: Wolfgang Bangerth, Texas A&M University, 2009, 2010
 *          Timo Heister, University of Goettingen, 2009, 2010
 */

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <ginkgo/core/distributed/matrix.hpp>

namespace LA {
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) &&                                             \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
using namespace dealii::LinearAlgebraTrilinos;
#else
#error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/affine_constraints.templates.h> // required bc we mix deal.ii vectors and petsc matrices
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/lac/sparsity_tools.h>

#include <ginkgo/ginkgo.hpp>
#include <pdexa-ext/deal.II/lac/ginkgo_interface.h>

#include <fstream>
#include <iostream>

namespace Step40 {
using namespace dealii;

using Vector = LinearAlgebra::distributed::Vector<double, MemorySpace::Host>;

template<typename ValueType, typename IndexType = gko::int32>
std::unique_ptr<gko::experimental::distributed::Matrix<ValueType, IndexType, gko::int64>>
create_dist_matrix(ConditionalOStream& ostream,
                   std::shared_ptr<const gko::Executor> exec,
                   const LA::MPI::SparseMatrix& deal_csr,
                   GinkgoInterface::csr_strategy strategy = GinkgoInterface::csr_strategy::exec_based) {
  auto comm = deal_csr.get_mpi_communicator();

  auto local_idxs = deal_csr.locally_owned_domain_indices();
  Assert(local_idxs.is_contiguous(), ExcInternalError("Can handle only contiguous partitions at the moment"));
  local_idxs.print(ostream);

  const auto& interval = *local_idxs.begin_intervals();

  auto row_partition =
    gko::share(gko::experimental::distributed::build_partition_from_local_range<IndexType, gko::int64>(
      exec, comm, {*interval.begin(), interval.last() + 1}));

  // todo: need to create column partition

  gko::dim<2> size = {static_cast<gko::size_type>(deal_csr.m()), static_cast<gko::size_type>(deal_csr.n())};
  gko::matrix_data<ValueType, gko::int64> md{size};
  for (auto el_it = deal_csr.begin(*interval.begin()); el_it != deal_csr.end(interval.last()); ++el_it) {
    md.nonzeros.emplace_back(static_cast<gko::int64>(el_it->row()), static_cast<gko::int64>(el_it->column()),
                             static_cast<ValueType>(el_it->value()));
  }
  auto gko_mtx = gko::experimental::distributed::Matrix<ValueType, IndexType>::create(exec, comm);
  gko_mtx->read_distributed(std::move(md), row_partition);
  return gko_mtx;
}

template<int dim>
class LaplaceProblem {
public:
  LaplaceProblem();

  void run();

private:
  void setup_system();
  void assemble_system();
  void solve();
  void refine_grid();
  void output_results(const unsigned int cycle);

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;

  const FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  LA::MPI::SparseMatrix system_matrix;
  Vector locally_relevant_solution;
  Vector system_rhs;

  ConditionalOStream pcout;
  TimerOutput computing_timer;
};

template<int dim>
LaplaceProblem<dim>::LaplaceProblem() :
    mpi_communicator(MPI_COMM_WORLD),
    triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(Triangulation<dim>::smoothing_on_refinement |
                                                             Triangulation<dim>::smoothing_on_coarsening)),
    fe(2), dof_handler(triangulation), pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    computing_timer(mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times) {}

template<int dim>
void LaplaceProblem<dim>::setup_system() {
  TimerOutput::Scope t(computing_timer, "setup");

  dof_handler.distribute_dofs(fe);

  pcout << "   Number of active cells:       " << triangulation.n_global_active_cells() << std::endl
        << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(), mpi_communicator,
                                             locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
}

template<int dim>
void LaplaceProblem<dim>::assemble_system() {
  TimerOutput::Scope t(computing_timer, "assembly");

  const QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  dealii::Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto& cell: dof_handler.active_cell_iterators())
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);

      cell_matrix = 0.;
      cell_rhs = 0.;

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
        const double rhs_value = (fe_values.quadrature_point(q_point)[1] >
                                      0.5 + 0.25 * std::sin(4.0 * numbers::PI * fe_values.quadrature_point(q_point)[0])
                                    ? 1.
                                    : -1.);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) +=
              fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);

          cell_rhs(i) += rhs_value * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);
        }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

template<int dim>
void LaplaceProblem<dim>::solve() {
  TimerOutput::Scope t(computing_timer, "solve");

  Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

  auto logger = gko::share(gko::log::Convergence<double>::create());

  auto exec = gko::ReferenceExecutor::create();
  auto gko_mtx = create_dist_matrix<double>(pcout, exec, system_matrix);
  auto solver = GinkgoInterface::inverse_operator(
    std::move(gko_mtx),
    gko::solver::Cg<double>::build()
      .with_criteria(
        gko::stop::Iteration::build().with_max_iters(1000lu),
        gko::stop::ResidualNorm<double>::build().with_baseline(gko::stop::mode::rhs_norm).with_reduction_factor(1e-6))
      .on(exec),
    logger);
  solver.vmult(completely_distributed_solution, system_rhs);

  pcout << "   " << logger->get_num_iterations() << " CG iterations needed to obtain convergence." << std::endl;

  constraints.distribute(completely_distributed_solution);

  locally_relevant_solution = completely_distributed_solution;
}

template<int dim>
void LaplaceProblem<dim>::refine_grid() {
  TimerOutput::Scope t(computing_timer, "refine");

  dealii::Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(fe.degree + 1),
                                     std::map<types::boundary_id, const Function<dim>*>(), locally_relevant_solution,
                                     estimated_error_per_cell);
  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3,
                                                                         0.03);
  triangulation.execute_coarsening_and_refinement();
}

template<int dim>
void LaplaceProblem<dim>::output_results(const unsigned int cycle) {
  TimerOutput::Scope t(computing_timer, "output");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(locally_relevant_solution, "u");

  dealii::Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i) subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./", "solution", cycle, mpi_communicator, 2, 8);
}

template<int dim>
void LaplaceProblem<dim>::run() {
  pcout << "Running with "
#ifdef USE_PETSC_LA
        << "PETSc"
#else
        << "Trilinos"
#endif
        << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator) << " MPI rank(s)..." << std::endl;

  const unsigned int n_cycles = 1;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle) {
    pcout << "Cycle " << cycle << ':' << std::endl;

    if (cycle == 0) {
      GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(5);
    }
    else
      refine_grid();

    setup_system();
    assemble_system();
    solve();
    output_results(cycle);

    computing_timer.print_summary();
    computing_timer.reset();

    pcout << std::endl;
  }
}
} // namespace Step40

int main(int argc, char* argv[]) {
  try {
    using namespace dealii;
    using namespace Step40;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    LaplaceProblem<2> laplace_problem_2d;
    laplace_problem_2d.run();
  }
  catch (std::exception& exc) {
    std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }
  catch (...) {
    std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
