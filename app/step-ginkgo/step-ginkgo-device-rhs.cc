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
#include <deal.II/numerics/data_out.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>
#include <deal.II/matrix_free/cuda_matrix_free.templates.h>
#include <deal.II/matrix_free/cuda_fe_evaluation.h>
#include <pdexa-ext/deal.II/numerics/data_out_dof_data.templates.h>
#include <pdexa-ext/deal.II/lac/ginkgo_vector.h>
#include <pdexa-ext/deal.II/lac/ginkgo_sparse_matrix.h>
#include <pdexa-ext/deal.II/lac/ginkgo_solver.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/logstream.h>

using namespace dealii;

std::shared_ptr<gko::Executor> create_default_host_executor() {
#ifdef KOKKOS_ENABLE_SERIAL
    if (std::is_same<Kokkos::DefaultHostExecutionSpace, Kokkos::Serial>::value) {
        return gko::ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
        if (std::is_same<Kokkos::DefaultHostExecutionSpace,
                     Kokkos::OpenMP>::value) {
        return gko::OmpExecutor::create();
    }
#endif
    GKO_NOT_IMPLEMENTED;
}

template<typename MemorySpace>
std::shared_ptr<gko::Executor> create_executor(MemorySpace) {
    using kokkos_space = typename MemorySpace::kokkos_space::execution_space;
#ifdef KOKKOS_ENABLE_SERIAL
    if (std::is_same<kokkos_space, Kokkos::Serial>::value) {
        return gko::ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if (std::is_same<kokkos_space, Kokkos::OpenMP>::value) {
        return gko::OmpExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same<kokkos_space, Kokkos::Cuda>::value) {
        return gko::CudaExecutor::create(Kokkos::device_id(),
                                    create_default_host_executor());
    }
#endif
#ifdef KOKKOS_ENABLE_HIP
    if (std::is_same<kokkos_space, Kokkos::HIP>::value) {
        return gko::HipExecutor::create(Kokkos::device_id(),
                                   create_default_host_executor());
    }
#endif
#ifdef KOKKOS_ENABLE_SYCL
    if (std::is_same<kokkos_space, Kokkos::Experimental::SYCL>::value) {
        return gko::DpcppExecutor::create(Kokkos::device_id(),
                                     create_default_host_executor());
    }
#endif
    throw std::runtime_error("No compatible kokkos memory space.");
}


template<int dim>
class StepGinkgo {
    using mtx = GinkgoWrappers::AbstractMatrix<double>;
    using vec = GinkgoWrappers::Vector<double>;
public:
    StepGinkgo(const std::string &mtx_type = "csr");

    void run();

private:
    void make_grid();

    void setup_system();

    void assemble_system();

    void assemble_rhs();

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

    dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Default> system_rhs_d;
};

template<typename Number, typename... Args>
std::unique_ptr<GinkgoWrappers::AbstractMatrix<Number>>
create_from_type(std::shared_ptr<const gko::Executor> exec, const std::string &type, Args &&...args) {
    if (type == "csr") {
        return std::make_unique<GinkgoWrappers::Csr<Number>>(std::move(exec), std::forward<Args>(args)
                ...);
    }
    if (type == "coo") {
        return std::make_unique<GinkgoWrappers::Coo<Number>>(std::move(exec), std::forward<Args>(args)
                ...);
    }
    if (type == "ell") {
        return std::make_unique<GinkgoWrappers::Ell<Number>>(std::move(exec), std::forward<Args>(args)
                ...);
    }
    if (type == "hybrid") {
        return std::make_unique<GinkgoWrappers::Hybrid<Number>>(std::move(exec), std::forward<Args>(args)
                ...);
    }
    if (type == "sellp") {
        return std::make_unique<GinkgoWrappers::Sellp<Number>>(std::move(exec), std::forward<Args>(args)
                ...);
    }
    throw std::runtime_error("Unsupported matrix stype: " + type);
}

template<int dim>
StepGinkgo<dim>::StepGinkgo(const std::string &mtx_type_)
        : fe(1), dof_handler(triangulation), exec(create_executor(dealii::MemorySpace::Default{})), system_matrix(),
          mtx_type(mtx_type_), solution(exec->get_master()), system_rhs(exec) {}

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
    system_matrix = create_from_type<double>(exec, mtx_type, dof_handler.n_dofs(), dof_handler.n_dofs());
    solution = vec{solution.get_gko_object()->get_executor(), dof_handler.n_dofs()};
    system_rhs_d.reinit(dof_handler.n_dofs());
    system_rhs = vec{
            gko::matrix::Dense<double>::create(exec, gko::dim<2>{static_cast<gko::size_type>(system_rhs_d.size()), 1},
                                               gko::make_array_view(exec, system_rhs_d.size(),
                                                                    system_rhs_d.get_values()), 1)};

}

template<int dim>
void StepGinkgo<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(fe.degree + 1);

    AffineConstraints<double> constraints;

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);
    VectorTools::interpolate_boundary_values(dof_handler, 0, FunctionFromFunctionObjects<dim>{{[](const auto &p) {
        return p.square();
    }}}, constraints);
    constraints.close();

    // This will assemble the matrix on the CPU and copy it to the correct
    // executor, which could be a GPU, afterward.
    MatrixCreator::create_laplace_matrix(dof_handler, quadrature_formula, *system_matrix,
                                         static_cast<Function<dim> *>(nullptr), constraints);
    assemble_rhs();
}

template<int dim, int fe_degree, typename Number>
struct functor {
    struct quad {
        DEAL_II_HOST_DEVICE void
        operator()(CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> *fe_eval,
                   const int q_point) const {
            const auto pos = gpu_data->get_quadrature_point(cell, static_cast<unsigned int>(q_point));

            Number f_val = 0.0;
            for (unsigned int i = 0; i < dim; ++i) {
                f_val += 4.0 * pow(pos(i), 4.0);
            }
            fe_eval->submit_value(f_val, q_point);
        }

        const typename CUDAWrappers::MatrixFree<dim, Number>::Data *gpu_data;
        unsigned int cell;
    };

    DEAL_II_HOST_DEVICE void
    operator()(const unsigned int cell, const typename CUDAWrappers::MatrixFree<dim, Number>::Data *gpu_data,
               CUDAWrappers::SharedData<dim, Number> *shared_data, const Number *, Number *dst) const {
        CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(gpu_data, shared_data);
        fe_eval.apply_for_each_quad_point(quad{gpu_data, cell});
        fe_eval.integrate(true, false);
        fe_eval.distribute_local_to_global(dst);
    }

    static constexpr unsigned int n_dofs_1d = fe_degree + 1;
    static constexpr unsigned int n_local_dofs = dealii::Utilities::pow(n_dofs_1d, dim);
    static constexpr unsigned int n_q_points = dealii::Utilities::pow(n_dofs_1d, dim);
};


template<int dim>
void StepGinkgo<dim>::assemble_rhs() {
    AffineConstraints<double> constraints;
    VectorTools::interpolate_boundary_values(dof_handler, 0, FunctionFromFunctionObjects<dim>{{[](const auto &p) {
        return p.square();
    }}}, constraints);
    constraints.close();
    const QGauss<1> quadrature_formula(fe.degree + 1);

    constexpr int fe_degree = 1;
    CUDAWrappers::MatrixFree<dim, double> mf;
    mf.reinit(dof_handler, constraints, quadrature_formula,
              {dealii::UpdateFlags::update_values | dealii::UpdateFlags::update_gradients | dealii::UpdateFlags::update_JxW_values |
               dealii::UpdateFlags::update_quadrature_points});

    // need to create dealii::LinearAlgebra::distributed::Vector
    // because even though `cell_loop` is templated, the internal
    // functions used are not...
    dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Default> src;
    mf.cell_loop(functor<dim, fe_degree, double>{}, src, system_rhs_d);

    // just set the constrained dofs, I have no idea if there is a better way to do that
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double> zeros(dofs_per_cell, 0);
    dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(system_rhs_d.size());
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (cell->is_locally_owned()) {
            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(zeros, local_dof_indices, rw_vector);
        }
    }
    dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Default> system_rhs_constraints(
            system_rhs_d.size());
    system_rhs_constraints.import_elements(rw_vector, dealii::VectorOperation::insert);
    mf.copy_constrained_values(system_rhs_constraints, system_rhs_d);
}


template<int dim>
void StepGinkgo<dim>::solve() {
    solution = 0.0;
    SolverControl solver_control(1000, 1e-12);
    GinkgoWrappers::SolverCG<double> solver(exec, solver_control);
    solver.solve(*system_matrix, solution, system_rhs, GinkgoWrappers::PreconditionIdentity<double>());
    std::cout << "   " << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
}

template<int dim>
void StepGinkgo<dim>::output_results() const {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();
    std::ofstream output(dim == 2 ? "solution-device-rhs-2d.vtk" : "solution-device-rhs-3d.vtk");
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


int main(int argc, char **argv) {
    auto mtx_type = argc >= 2 ? argv[1] : "csr";
    {
        StepGinkgo<2> laplace_problem_2d{mtx_type};
        laplace_problem_2d.run();
    }
    {
        StepGinkgo<3> laplace_problem_3d{mtx_type};
        laplace_problem_3d.run();
    }
    return 0;
}
