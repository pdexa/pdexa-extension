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
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/affine_constraints.templates.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>
#include <deal.II/numerics/vector_tools_rhs.h>
#include <deal.II/numerics/vector_tools_rhs.templates.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_creator.templates.h>
#include <pdexa-ext/deal.II/lac/ginkgo_vector.h>
#include <pdexa-ext/deal.II/lac/ginkgo_sparse_matrix.h>
#include <pdexa-ext/deal.II/lac/ginkgo_solver.h>

#include <fstream>
#include <iostream>


namespace dealii::internal::DataOutImplementation {
namespace {
template<typename Number>
void copy_locally_owned_data_from(const GinkgoWrappers::Vector<Number> &src,
                                  LinearAlgebra::distributed::Vector<Number> &dst) {
  LinearAlgebra::ReadWriteVector<Number> temp;
  temp.reinit(src.locally_owned_elements());

  auto gko_obj = src.get_gko_object();
  GinkgoWrappers::Vector<Number> host_src(gko_obj->get_executor()->get_master());
  host_src = src;
  std::copy_n(host_src.begin(), host_src.size(), temp.begin());

  LinearAlgebra::ReadWriteVector<Number> temp2;
  temp2.reinit(temp, true);
  temp2 = temp;

  dst.import_elements(temp2, VectorOperation::insert);
}
}
}


#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/data_out_dof_data.templates.h>

#include <deal.II/base/logstream.h>


using namespace dealii;

template<int dim>
class StepGinkgo {
  using mtx = GinkgoWrappers::AbstractMatrix<double>;
  using vec = GinkgoWrappers::Vector<double>;

public:
  StepGinkgo(std::shared_ptr<const gko::Executor> exec,
             const std::string &mtx_type = "csr");

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

  std::shared_ptr<const gko::Executor> exec;

  std::unique_ptr<mtx> system_matrix;
  std::string mtx_type;

  vec solution;
  vec system_rhs;
};

template<typename Number, typename... Args>
std::unique_ptr<GinkgoWrappers::AbstractMatrix<Number>>
create_from_type(std::shared_ptr<const gko::Executor> exec,
                 const std::string &type,
                 Args &&...args) {
  if (type == "csr") {
    return std::make_unique<GinkgoWrappers::Csr<Number>>(
        std::move(exec), std::forward<Args>(args)...);
  }
  if (type == "coo") {
    return std::make_unique<GinkgoWrappers::Coo<Number>>(
        std::move(exec), std::forward<Args>(args)...);
  }
  if (type == "ell") {
    return std::make_unique<GinkgoWrappers::Ell<Number>>(
        std::move(exec), std::forward<Args>(args)...);
  }
  if (type == "hybrid") {
    return std::make_unique<GinkgoWrappers::Hybrid<Number>>(
        std::move(exec), std::forward<Args>(args)...);
  }
  if (type == "sellp") {
    return std::make_unique<GinkgoWrappers::Sellp<Number>>(
        std::move(exec), std::forward<Args>(args)...);
  }
}

template<int dim>
StepGinkgo<dim>::StepGinkgo(std::shared_ptr<const gko::Executor> exec,
                            const std::string &mtx_type)
    : fe(1), dof_handler(triangulation), exec(exec), system_matrix(), mtx_type(mtx_type), solution(exec->get_master()),
      system_rhs(exec->get_master()) {}

template<int dim>
void StepGinkgo<dim>::make_grid() {
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(4);

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}

template<int dim>
void StepGinkgo<dim>::setup_system() {
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  system_matrix = create_from_type<double>(exec,
                                           mtx_type,
                                           dof_handler.n_dofs(),
                                           dof_handler.n_dofs());

  solution =
      vec{solution.get_gko_object()->get_executor(), dof_handler.n_dofs()};
  system_rhs =
      vec{system_rhs.get_gko_object()->get_executor(), dof_handler.n_dofs()};
}

template<int dim>
void StepGinkgo<dim>::assemble_system() {
  QGauss<dim> quadrature_formula(fe.degree + 1);

  AffineConstraints<double> constraints;

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           FunctionFromFunctionObjects<dim>{
                                               {[](const auto &p) {
                                                 return p.square();
                                               }}},
                                           constraints);
  constraints.close();

  // This will assemble the matrix on the CPU and copy it to the correct
  // executor, which could be a GPU, afterward.
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       quadrature_formula,
                                       *system_matrix,
                                       static_cast<Function<dim>*>(nullptr),
                                       constraints);

  // This will assemble the right-hand-side vector on the CPU and copy it to the
  // correct executor, which could be a GPU, afterward.
  VectorTools::create_right_hand_side(dof_handler,
                                      quadrature_formula,
                                      FunctionFromFunctionObjects<dim>{
                                          {[](const auto &p) {
                                            double return_value = 0.0;
                                            for (unsigned int i = 0; i < dim; ++i)
                                              return_value +=
                                                  4.0 * std::pow(p(i), 4.0);
                                            return return_value;
                                          }}},
                                      system_rhs,
                                      constraints);
}

template<int dim>
void StepGinkgo<dim>::solve() {
  solution = 0.0;

  SolverControl solver_control(1000, 1e-12);
  GinkgoWrappers::SolverCG<double> solver(exec, solver_control);
  solver.solve(*system_matrix,
               solution,
               system_rhs,
               GinkgoWrappers::PreconditionIdentity<double>());

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
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
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;

  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}

int main(int argc, char** argv) {
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
    StepGinkgo<2> laplace_problem_2d{exec, mtx_type};
    laplace_problem_2d.run();
  }

  {
    StepGinkgo<3> laplace_problem_3d{exec, mtx_type};
    laplace_problem_3d.run();
  }

  return 0;
}
