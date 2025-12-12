/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Marcel Koch, KIT, 2023 - 2025
 *         Martin Kronbichler, Ruhr University Bochum, 2024
 */

// @sect3{Include files}

// The first few (many?) include files have already been used in the previous
// example, so we will not explain their meaning here again.
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/affine_constraints.templates.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_creator.templates.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>
#include <deal.II/numerics/vector_tools_rhs.h>
#include <deal.II/numerics/vector_tools_rhs.templates.h>
#include <pdexa-ext/deal.II/lac/ginkgo_interface.h>

#include <ginkgo/ginkgo.hpp>

#include <fstream>
#include <iostream>

#include <deal.II/base/logstream.h>

using namespace dealii;

using memory_space = MemorySpace::Host;

template<int dim>
class StepGinkgo {
  using mtx = SparseMatrix<double>;
  using vec = LinearAlgebra::distributed::Vector<double, memory_space>;

public:
  StepGinkgo(std::shared_ptr<const gko::Executor> exec, const std::string& mtx_type = "csr");

  void run();

private:
  void make_grid();

  void setup_system();

  void assemble_system();

  void solve();

  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  SparsityPattern sparsity_pattern;
  mtx system_matrix;
  std::string mtx_type;

  vec solution;
  vec system_rhs;

  std::shared_ptr<const gko::Executor> solver_exec;
  std::shared_ptr<const gko::Executor> default_exec;
};

template<int dim>
StepGinkgo<dim>::StepGinkgo(std::shared_ptr<const gko::Executor> exec, const std::string& mtx_type) :
    fe(1), dof_handler(triangulation), solver_exec(exec), default_exec(gko::ReferenceExecutor::create()),
    mtx_type(mtx_type) {}

template<int dim>
void StepGinkgo<dim>::make_grid() {
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(4);

  std::cout << "   Number of active cells: " << triangulation.n_active_cells() << std::endl
            << "   Total number of cells: " << triangulation.n_cells() << std::endl;
}

template<int dim>
void StepGinkgo<dim>::setup_system() {
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

template<int dim>
void StepGinkgo<dim>::assemble_system() {
  QGauss<dim> quadrature_formula(fe.degree + 1);

  AffineConstraints<double> constraints;

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);

  VectorTools::interpolate_boundary_values(
    dof_handler, 0, FunctionFromFunctionObjects<dim>{{[](const auto& p) { return p.square(); }}}, constraints);
  constraints.close();

  MatrixCreator::create_laplace_matrix(dof_handler, quadrature_formula, system_matrix,
                                       static_cast<Function<dim>*>(nullptr), constraints);

  VectorTools::create_right_hand_side(dof_handler, quadrature_formula,
                                      FunctionFromFunctionObjects<dim>{{[](const auto& p) {
                                        double return_value = 0.0;
                                        for (unsigned int i = 0; i < dim; ++i)
                                          return_value += 4.0 * std::pow(p(i), 4.0);
                                        return return_value;
                                      }}},
                                      system_rhs, constraints);
}

template<int dim>
void StepGinkgo<dim>::solve() {
  solution = 0.0;

  auto logger = gko::share(gko::log::Convergence<double>::create());

  std::shared_ptr<gko::LinOp> gko_mtx = GinkgoInterface::create_csr_matrix(solver_exec, system_matrix);
  auto solver = GinkgoInterface::inverse_operator(
    default_exec, default_exec, gko_mtx,
    gko::solver::Cg<double>::build()
      .with_criteria(
        gko::stop::Iteration::build().with_max_iters(1000lu),
        gko::stop::ResidualNorm<double>::build().with_baseline(gko::stop::mode::rhs_norm).with_reduction_factor(1e-6))
      .on(solver_exec),
    logger);
  solver.vmult(solution, system_rhs);

  std::cout << "   " << logger->get_num_iterations() << " CG iterations needed to obtain convergence." << std::endl;
}

template<int dim>
void StepGinkgo<dim>::output_results() const {
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  data_out.build_patches();

  std::ofstream output(dim == 2 ? "solution-2d.vtk" : "solution-3d.vtk");
  data_out.write_vtk(output);
}

template<int dim>
void StepGinkgo<dim>::run() {
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;

  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}

int main(int argc, char** argv) {
  const auto executor_string = argc >= 2 ? argv[1] : "reference";

  const std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>> executor_factory{
    {"reference", []() { return gko::ReferenceExecutor::create(); }},
    {"omp", []() { return gko::OmpExecutor::create(); }},
    {"cuda", []() { return gko::CudaExecutor::create(0, gko::ReferenceExecutor::create()); }},
    {"hip", []() { return gko::HipExecutor::create(0, gko::ReferenceExecutor::create()); }},
    {"dpcpp", []() { return gko::DpcppExecutor::create(0, gko::ReferenceExecutor::create()); }}};

  auto exec = executor_factory.at(executor_string)();

  auto mtx_type = argc >= 3 ? argv[2] : "csr";
  {
    StepGinkgo<2> laplace_problem_2d{exec, mtx_type};
    laplace_problem_2d.run();
  }
  {
    StepGinkgo<3> laplace_problem_3d{exec, mtx_type};
    laplace_problem_3d.run();
  }

  return 0;
}
