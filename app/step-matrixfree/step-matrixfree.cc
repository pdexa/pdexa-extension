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
 * Author: Marcel Koch, KIT, 2023
 */


// @sect3{Include files}

// The first few (many?) include files have already been used in the previous
// example, so we will not explain their meaning here again.
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/affine_constraints.templates.h>
#include <pdexa-ext/deal.II/lac/ginkgo_solver.h>
#include <pdexa-ext/deal.II/lac/ginkgo_sparse_matrix.h>
#include <pdexa-ext/deal.II/lac/ginkgo_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_creator.templates.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>
#include <deal.II/numerics/vector_tools_rhs.h>
#include <deal.II/numerics/vector_tools_rhs.templates.h>
#include <pdexa-ext/deal.II/numerics/data_out_dof_data.templates.h>

#include <fstream>
#include <iostream>



template <int dim, typename Number>
class MatrixFreeOperator
  : public gko::EnableLinOp<MatrixFreeOperator<dim, Number>>
{
  using Base = gko::EnableLinOp<MatrixFreeOperator<dim, Number>>;

public:
  using VectorType = dealii::ArrayView<Number>;
  using gko::EnableLinOp<MatrixFreeOperator>::apply;

  MatrixFreeOperator(std::shared_ptr<const gko::Executor> exec)
    : Base(exec)
  {}

  MatrixFreeOperator(std::shared_ptr<const gko::Executor> exec,
                     const dealii::DoFHandler<dim>       &dof_handler)
    : Base(exec, gko::dim<2>(dof_handler.n_dofs(), dof_handler.n_dofs()))
    , mapping(
        std::make_shared<dealii::MappingQ<dim>>(dof_handler.get_fe().degree))
  {
    constraints = std::make_shared<dealii::AffineConstraints<Number>>();
    matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
    matrix_free->reinit(
      *mapping,
      dof_handler,
      *constraints,
      dealii::QGauss<1>(dof_handler.get_fe().degree + 1),
      typename dealii::MatrixFree<dim, Number>::AdditionalData());
  }

  void
  apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
  {
    auto b_dense =
      gko::as<gko::matrix::Dense<Number>>(const_cast<gko::LinOp *>(b));
    auto       x_dense = gko::as<gko::matrix::Dense<Number>>(x);
    VectorType b_deal(b_dense->get_values(),
                      b_dense->get_num_stored_elements());
    VectorType x_deal(x_dense->get_values(),
                      x_dense->get_num_stored_elements());
    vmult(x_deal, b_deal);
  }

  void
  apply_impl(const gko::LinOp *alpha,
             const gko::LinOp *b,
             const gko::LinOp *beta,
             gko::LinOp       *x) const override
  {
    auto b_dense =
      gko::as<gko::matrix::Dense<Number>>(const_cast<gko::LinOp *>(b));
    auto       x_dense = gko::as<gko::matrix::Dense<Number>>(x);
    VectorType b_deal(b_dense->get_values(),
                      b_dense->get_num_stored_elements());
    VectorType x_deal(x_dense->get_values(),
                      x_dense->get_num_stored_elements());
    dealii::Vector<Number> tmp(b_dense->get_num_stored_elements());
    VectorType tmp_deal(tmp);
    vmult(tmp_deal, b_deal);
    const Number alpha_num = gko::as<gko::matrix::Dense<Number>>(const_cast<gko::LinOp *>(alpha))->get_values()[0];
    const Number beta_num = gko::as<gko::matrix::Dense<Number>>(const_cast<gko::LinOp *>(beta))->get_values()[0];
    for (unsigned int i = 0; i < x_dense->get_num_stored_elements(); ++i)
      x_deal[i] = x_deal[i] * beta_num + tmp_deal[i] * alpha_num;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free->cell_loop(
      &MatrixFreeOperator::local_apply, this, dst, src, true);
  }

private:
  void
  local_apply(const dealii::MatrixFree<dim, Number>       &data,
              VectorType                                  &dst,
              const VectorType                            &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    dealii::FEEvaluation<dim, -1, 0, 1, Number> eval(data);
    const unsigned int                          n_q_points = eval.n_q_points;

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.read_dof_values(src);
        eval.evaluate(dealii::EvaluationFlags::values |
                      dealii::EvaluationFlags::gradients);
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            eval.submit_value(eval.get_value(q), q);
            eval.submit_gradient(eval.get_gradient(q), q);
          }
        eval.integrate(dealii::EvaluationFlags::values |
                       dealii::EvaluationFlags::gradients);
        eval.distribute_local_to_global(dst);
      }
  }

  std::shared_ptr<const dealii::Mapping<dim>>        mapping;
  std::shared_ptr<dealii::AffineConstraints<Number>> constraints;
  std::shared_ptr<dealii::MatrixFree<dim, Number>>   matrix_free;
};



template <int dim>
class StepGinkgo
{
  using vec = std::shared_ptr<gko::matrix::Dense<double>>;

public:
  StepGinkgo(std::shared_ptr<const gko::Executor> exec);

  void
  run();

private:
  void
  make_grid();

  void
  setup_system();

  void
  assemble_system();

  void
  solve();

  void
  output_results() const;

  dealii::Triangulation<dim> triangulation;
  dealii::FE_Q<dim>          fe;
  dealii::DoFHandler<dim>    dof_handler;

  std::shared_ptr<const gko::Executor> exec;

  std::string mtx_type;

  vec solution;
  vec system_rhs;
};


template <int dim>
StepGinkgo<dim>::StepGinkgo(std::shared_ptr<const gko::Executor> exec)
  : fe(1)
  , dof_handler(triangulation)
  , exec(exec)
{}

template <int dim>
void
StepGinkgo<dim>::make_grid()
{
  dealii::GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(4);

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}

template <int dim>
void
StepGinkgo<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  solution = gko::matrix::Dense<double>::create(exec, gko::dim<2>(dof_handler.n_dofs(), 1));
}

template <int dim>
void
StepGinkgo<dim>::assemble_system()
{
  dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

  dealii::AffineConstraints<double> constraints;

  // This will assemble the right-hand-side vector on the CPU and copy it to the
  // correct executor, which could be a GPU, afterward.
  dealii::Vector<double> vector(dof_handler.n_dofs());
  dealii::VectorTools::create_right_hand_side(
    dof_handler,
    quadrature_formula,
    dealii::FunctionFromFunctionObjects<dim>{{[](const auto &p) {
      double return_value = 0.0;
      for (unsigned int i = 0; i < dim; ++i)
        return_value += 4.0 * std::pow(p(i), 4.0);
      return return_value;
    }}},
    vector,
    constraints);
  auto vec = gko::matrix::Dense<double>::create(
    exec,
    gko::dim<2>(vector.size(), 1),
    gko::make_array_view(exec, vector.size(), vector.begin()),
    1);
  system_rhs = gko::clone(vec);
}

template <int dim>
void
StepGinkgo<dim>::solve()
{
  const auto matrix_free_operator =
    std::make_shared<MatrixFreeOperator<dim, double>>(exec, dof_handler);

  solution->fill(0.0);

  const double tolerance = 1e-12;
  using cg               = gko::solver::Cg<double>;
  auto solver_gen =
    cg::build()
      .with_criteria(gko::stop::Iteration::build().with_max_iters(1000u).on(
                       exec),
                     gko::stop::ResidualNorm<double>::build()
                       .with_baseline(gko::stop::mode::absolute)
                       .with_reduction_factor(tolerance)
                       .on(exec))
      .on(exec);

  auto solver = solver_gen->generate(matrix_free_operator);

  std::shared_ptr<const gko::log::Convergence<double>> logger =
    gko::log::Convergence<double>::create();
  solver->add_logger(logger);

  solver->apply(system_rhs, solution);
  auto res = gko::as<gko::matrix::Dense<double>>(logger->get_residual_norm());
  auto num_iteration = logger->get_num_iterations();
  std::cout << "Final residual norm sqrt(r^T r): " << res->at(0, 0)
            << " n_iter " << num_iteration << std::endl;
}

template <int dim>
void
StepGinkgo<dim>::output_results() const
{}

template <int dim>
void
StepGinkgo<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;

  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}

int
main(int argc, char **argv)
{
  const auto executor_string = argc >= 2 ? argv[1] : "reference";

  const std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
    executor_factory{
      {"reference", []() { return gko::ReferenceExecutor::create(); }},
      {"omp", []() { return gko::OmpExecutor::create(); }},
      {"cuda",
       []() {
         return gko::CudaExecutor::create(0, gko::ReferenceExecutor::create());
       }},
      {"hip",
       []() {
         return gko::HipExecutor::create(0, gko::ReferenceExecutor::create());
       }},
      {"dpcpp", []() {
         return gko::DpcppExecutor::create(0, gko::ReferenceExecutor::create());
       }}};

  auto exec = executor_factory.at(executor_string)();

  auto mtx_type = argc >= 3 ? argv[2] : "csr";

  {
    StepGinkgo<2> laplace_problem_2d{exec};
    laplace_problem_2d.run();
  }

  {
    StepGinkgo<3> laplace_problem_3d{exec};
    laplace_problem_3d.run();
  }

  return 0;
}
