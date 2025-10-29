
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>


#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "consistent_splitting_solver.h"
#include "preconditioners.h"
#include "evaluators.h"


#include <fstream>

using namespace dealii;


const bool use_neumann_boundary                      = true;
const bool use_skew_symmetric_convective_formulation = true;
const bool use_divergence_formulation                = false;
const bool use_leray_projection                      = true;

const bool use_amg                  = false;
const bool use_hmg                  = true;
const bool use_pmg                  = true;
const bool use_cmg                  = true;
const bool use_pointjacobi_pressure = false;
const bool use_amg_as_coarse_grid_solver = false;

const bool use_velocity_point_jacobi = false;
const bool use_inverse_mass_velocity = false;
const bool use_mg_velocity           = true;
const bool use_cmg_vel               = true;
const bool use_pmg_vel               = true;
const bool use_hmg_vel               = true;
const bool use_amg_as_coarse_grid_solver_vel = false;

const double penalty_divergence = 1.0;
const double penalty_continuity = 1.0;

const double upwind_factor = 1.0;

const double viscosity = 1e-3;
const double u_x_max   = 1.5;



template <int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  AnalyticalSolutionVelocity(const double u_x_max, const double viscosity)
    : dealii::Function<dim>(dim, 0.0)
    , u_x_max(u_x_max)
    , viscosity(viscosity)
  {}

  double
  value(const dealii::Point<dim> &p, const unsigned int component = 0) const final
  {
    const double t      = this->get_time();
    const double pi     = dealii::numbers::PI;
    double       result = 0.0;
    if (component == 0 && std::abs(p[0]) < 1e-12)
      result =
        u_x_max * 4. * p[1] * (0.41 - p[1]) / (0.41 * 0.41) * std::sin(pi * t / 8.);

    return result;
  }

  dealii::Tensor<1, dim, double>
  gradient(const dealii::Point<dim> &, const unsigned int) const final
  {
    dealii::Tensor<1, dim, double> result{};
    result *= 0.;
    return result;
  }

private:
  const double u_x_max, viscosity;
};



template <int dim>
class AnalyticalSolutionPressure : public dealii::Function<dim>
{
public:
  AnalyticalSolutionPressure(const double u_x_max, const double viscosity)
    : dealii::Function<dim>(1 /*n_components*/, 0.0)
    , u_x_max(u_x_max)
    , viscosity(viscosity)
  {}

  double
  value(const dealii::Point<dim> &, const unsigned int /*component*/) const final
  {
    return 0.;
  }

private:
  const double u_x_max, viscosity;
};



template <int dim>
class AnalyticalRHS : public dealii::Function<dim>
{
public:
  AnalyticalRHS(const double u_x_max, const double viscosity)
    : dealii::Function<dim>(dim, 0.0)
    , u_x_max(u_x_max)
    , viscosity(viscosity)
  {}

  double
  value(const dealii::Point<dim> &p, const unsigned int component = 0) const final
  {
    (void)p;
    (void)component;
    double result = 0.0;
    return result;
  }

private:
  const double u_x_max, viscosity;
};


template <int dim, typename Number>
void
do_test(const unsigned int fe_degree,
        const unsigned int n_refinements,
        const unsigned int n_refinements_time)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  Timer                time_setup;
  FESystem<dim>        fe_u(FE_DGQ<dim>(fe_degree), dim);
  FE_DGQ<dim>          fe_p(fe_degree - 1);
  MappingQGeneric<dim> mapping(fe_degree);

  parallel::distributed::Triangulation<dim> tria(
    MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices);

  GridGenerator::channel_with_cylinder(tria);

  dealii::Point<dim> midpoint;
  midpoint[0] = 0.2;
  midpoint[1] = 0.2;

  for (auto cell : tria.cell_iterators())
    for (const auto &f : cell->face_indices())
      if (cell->face(f)->at_boundary())
        {
          if (std::abs(cell->face(f)->center()[0] - 2.2) < 1e-12)
            cell->face(f)->set_all_boundary_ids(1);
          else if (midpoint.distance(cell->face(f)->center()) <= 0.15)
            {
              cell->face(f)->set_all_boundary_ids(2);
            }
          else
            cell->face(f)->set_all_boundary_ids(0);
        }

  tria.refine_global(n_refinements);

  DoFHandler<dim> dof_handler_u(tria);
  dof_handler_u.distribute_dofs(fe_u);

  DoFHandler<dim> dof_handler_p(tria);
  dof_handler_p.distribute_dofs(fe_p);

  pcout << "number of active_cells: " << tria.n_global_active_cells() << std::endl;
  pcout << "Solving with " << fe_u.get_name() << " x " << fe_p.get_name() << " element"
        << std::endl;
  pcout << "number of degrees of freedom: " << dof_handler_u.n_dofs() << " + "
        << dof_handler_p.n_dofs() << std::endl;

  double h_min = std::numeric_limits<double>::max();
  for (const auto &cell : dof_handler_u.active_cell_iterators())
    h_min = std::min(h_min, cell->minimum_vertex_distance());
  double time_step_size = 1.;
  for (unsigned int i = 0; i < n_refinements_time; ++i)
    time_step_size *= 0.5;
  const double local_time_step =
    4.0 * time_step_size / std::pow(fe_degree, 1.5) * h_min / u_x_max;

  const Number time_step =
    1e-5 + 0. * dealii::Utilities::MPI::min(local_time_step,
                                            MPI_COMM_WORLD); // time_step_size
  pcout << "Time step size: " << time_step << std::endl;

  const unsigned int bdf_order   = 3;
  const unsigned int bdf_order_p = 2;

  MomentumOperator<dim, dim, Number> momentum_op;
  momentum_op.set_body_force_factory([=]() {
    return std::make_unique<AnalyticalRHS<dim>>(u_x_max, viscosity);
  });
  momentum_op.set_dirichletBC_pressure_factory ([=]() {
    return std::make_unique<AnalyticalSolutionPressure<dim>>(u_x_max, viscosity);
  });
  momentum_op.set_DirichletBC_velocity_factory ([=]() {
    return std::make_unique<AnalyticalSolutionVelocity<dim>>(u_x_max, viscosity);
  });
  // set up operator
  momentum_op.reinit(mapping, dof_handler_u, dof_handler_p, time_step, bdf_order, use_skew_symmetric_convective_formulation, use_divergence_formulation, upwind_factor);

  momentum_op.set_viscosity(viscosity);
  momentum_op.set_time(0.0);

  LinearAlgebra::distributed::Vector<Number> vec_u, vec_u_deriv, vec_u_rhs, vec_p,
    vec_u_norm, speed_extrapolated, vec_vorticity, vec_p_rhs, vec_p_rhs_n, vec_p_norm,
    vec_div_u, inverse_diagonal;
  momentum_op.initialize_dof_vector(vec_u, dof_no_v);
  momentum_op.initialize_dof_vector(vec_u_deriv, dof_no_v);
  momentum_op.initialize_dof_vector(vec_u_rhs, dof_no_v);
  momentum_op.initialize_dof_vector(vec_p, dof_no_p);
  momentum_op.initialize_dof_vector(vec_u_norm, dof_no_v);
  momentum_op.initialize_dof_vector(speed_extrapolated, dof_no_v);
  momentum_op.initialize_dof_vector(vec_vorticity, dof_no_v);
  momentum_op.initialize_dof_vector(vec_p_rhs, dof_no_p);
  momentum_op.initialize_dof_vector(vec_p_rhs_n, dof_no_p);
  momentum_op.initialize_dof_vector(vec_p_norm, dof_no_p);
  momentum_op.initialize_dof_vector(vec_div_u, dof_no_p);

  std::vector<LinearAlgebra::distributed::Vector<Number>> vec_u_old(bdf_order);
  for (auto &vec : vec_u_old)
    momentum_op.initialize_dof_vector(vec, dof_no_v);

  InverseMassPreconditioner<dim, Number> inverse_mass;
  inverse_mass.reinit(momentum_op.get_matrix_free(), time_step);

  MultigridPreconditionerVelocity<dim, Number, Number> preconditioner_velocity(
    momentum_op, mapping.get_degree(), viscosity, time_step, bdf_order, use_hmg_vel, use_cmg_vel, use_pmg_vel);

  DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>>
    preconditioner_velocity_pointjacobi;
  DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>>
    preconditioner_pressure_pointjacobi;


  PressureOperator<dim, Number> pressure_op;
  pressure_op.reinit(momentum_op.get_matrix_free(), bdf_order, time_step, use_leray_projection);
  pressure_op.set_body_force_factory([=]() {
    return std::make_unique<AnalyticalRHS<dim>>(u_x_max, viscosity);
  });
  pressure_op.set_time_step(time_step);
  pressure_op.set_viscosity(viscosity);
  pressure_op.set_dirichletBC_pressure_factory ([=]() {
    return std::make_unique<AnalyticalSolutionPressure<dim>>(u_x_max, viscosity);
  });
  pressure_op.set_DirichletBC_velocity_factory ([=]() {
    return std::make_unique<AnalyticalSolutionVelocity<dim>>(u_x_max, viscosity);
  });

  TrilinosWrappers::SparseMatrix pressure_system_matrix;
  if (use_amg)
    pressure_op.get_system_matrix(pressure_system_matrix);
  TrilinosWrappers::PreconditionAMG precondition_amg;

  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.smoother_sweeps = 1;
  amg_data.n_cycles        = 1;
  amg_data.smoother_type   = "ILU";

  if (use_amg)
    precondition_amg.initialize(pressure_system_matrix, amg_data);

  MultigridPreconditioner<dim, Number, Number> precondition_hmg(pressure_op,
                                                                mapping.get_degree(),
                                                                time_step,
                                                                bdf_order,
                                                                use_hmg,
                                                                use_cmg,
                                                                use_pmg,   
                                                                use_amg_as_coarse_grid_solver,
                                                                use_neumann_boundary);

  Number current_time = 0;

  AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
  AnalyticalSolutionPressure<dim> exact_pressure(u_x_max, viscosity);

  exact_velocity.set_time(current_time);
  VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u_old[0]);
  exact_pressure.set_time(current_time);
  VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p);

  const Number       end_time         = 8.0;
  const unsigned int output_interval  = 50;
  unsigned int       time_step_number = 0;

  Number drag_max = -10000000000.;
  Number lift_max = -10000000000.;
  Number drag_min = 100000000000.;
  Number lift_min = 100000000000.;

  const bool write_output = true;

  const double setup_time = time_setup.wall_time();
  pcout << "Setup time: " << setup_time << std::endl;
  Timer time_loop;
  while (current_time <= end_time)
    {
      Timer time_single_step;

      current_time += time_step;
      ++time_step_number;
      momentum_op.set_time(current_time);

      const unsigned int current_bdf_order =
        time_step_number < bdf_order ? time_step_number : bdf_order;
      BDFTimeIntegratorConstants bdf(current_bdf_order);
      momentum_op.set_bdf_order(current_bdf_order);
      pressure_op.set_bdf_order(current_bdf_order);
      BDFTimeIntegratorConstants bdf_p(time_step_number < bdf_order_p ? time_step_number :
                                                                        bdf_order_p);

      // Pressure step
      vec_p_rhs = 0.;
      if (use_leray_projection)
        for (unsigned int i = 0; i < bdf.get_order(); ++i)
          {
            pressure_op.set_time(current_time - (i + 1) * time_step);
            vec_div_u = 0.;
            pressure_op.compute_divergence(vec_div_u, vec_u_old[i]);
            vec_p_rhs.add(-bdf.get_alpha(i) / time_step, vec_div_u);
          }
      pressure_op.set_time(current_time);

      for (unsigned int i = 0; i < bdf_p.get_order(); ++i)
        {
          pressure_op.set_time(current_time - (i + 1) * time_step);
          vec_p_rhs_n = 0.;
          pressure_op.compute_convective_rhs(vec_p_rhs_n, vec_u_old[i]);
          vec_p_rhs.add(bdf_p.get_beta(i), vec_p_rhs_n);
        }

      speed_extrapolated = 0.;
      for (unsigned int i = 0; i < bdf_p.get_order(); ++i)
        speed_extrapolated.add(bdf_p.get_beta(i), vec_u_old[i]);


      pressure_op.set_time(current_time);
      vec_p_rhs_n   = 0.;
      vec_vorticity = 0.;
      momentum_op.evaluate_vorticity(vec_vorticity, speed_extrapolated);
      pressure_op.compute_rhs(vec_p_rhs_n, vec_vorticity);
      vec_p_rhs.add(1, vec_p_rhs_n);

      unsigned int iteration_count;
      if (!use_neumann_boundary)
        VectorTools::subtract_mean_value(vec_p_rhs);
      ReductionControl control(10000, 1e-12, 1e-6);
      SolverCG<LinearAlgebra::distributed::Vector<double>> solver(control);
      // vec_p = 0.;
      if (use_amg)
        {
          solver.solve(pressure_system_matrix, vec_p, vec_p_rhs, precondition_amg);
          iteration_count = control.last_step();
        }
      else if (use_cmg || use_hmg || use_pmg)
        {
          iteration_count = precondition_hmg.solve(pressure_op, vec_p, vec_p_rhs);
        }
      else if (use_pointjacobi_pressure)
        {
          pressure_op.compute_inverse_diagonal(
            preconditioner_pressure_pointjacobi.get_vector());

          solver.solve(pressure_op,
                       vec_p,
                       vec_p_rhs,
                       preconditioner_pressure_pointjacobi);
          iteration_count = control.last_step();
        }
      else
        {
          solver.solve(pressure_op, vec_p, vec_p_rhs, PreconditionIdentity());
          iteration_count = control.last_step();
        }
      if (!use_neumann_boundary)
        VectorTools::subtract_mean_value(vec_p);
      if (write_output && time_step_number % output_interval == 0)
        pcout << "Pressure solver: " << iteration_count << " iterations" << std::endl;

      // exact_pressure.set_time(current_time);
      // VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p);

      // Momentum step
      vec_u_deriv        = 0.;
      speed_extrapolated = 0.;

      for (unsigned int i = 0; i < bdf.get_order(); ++i)
        {
          vec_u_deriv.add(bdf.get_alpha(i) / time_step, vec_u_old[i]);
          speed_extrapolated.add(bdf.get_beta(i), vec_u_old[i]);
        }

      vec_u_rhs = 0.;
      momentum_op.rhs(vec_u_rhs, vec_u_deriv, speed_extrapolated, vec_p);
      if (use_mg_velocity)
        preconditioner_velocity.update(current_time, speed_extrapolated);

      ReductionControl control_mom(10000, 1e-12, 1e-6);
      SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver_mom(control_mom);
      vec_u.swap(speed_extrapolated); // = 0.;
      unsigned int n_iterations_vel;
      if (use_mg_velocity)
        {
          n_iterations_vel = preconditioner_velocity.solve(momentum_op, vec_u, vec_u_rhs, use_amg_as_coarse_grid_solver_vel);
        }
      else if (use_inverse_mass_velocity)
        {
          inverse_mass.set_scaling_factor(time_step / bdf.get_gamma0());
          solver_mom.solve(momentum_op, vec_u, vec_u_rhs, inverse_mass);
          n_iterations_vel = control_mom.last_step();
        }
      else if (use_velocity_point_jacobi)
        {
          momentum_op.compute_inverse_diagonal(
            preconditioner_velocity_pointjacobi.get_vector());
          solver_mom.solve(momentum_op,
                           vec_u,
                           vec_u_rhs,
                           preconditioner_velocity_pointjacobi);
          n_iterations_vel = control_mom.last_step();
        }
      else
        {
          solver_mom.solve(momentum_op, vec_u, vec_u_rhs, PreconditionIdentity());
          n_iterations_vel = control_mom.last_step();
        }

      if (write_output && time_step_number % output_interval == 0)
        pcout << "Momentum solver: " << n_iterations_vel << " iterations" << std::endl;

      // exact_velocity.set_time(current_time);
      // VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u);

      for (unsigned int i = bdf_order - 1; i != 0; --i)
        {
          std::swap(vec_u_old[i], vec_u_old[i - 1]);
        }

      vec_u_old[0].swap(vec_u);

      // Compute lift and drag
      {
        auto                   matrix_free = momentum_op.get_matrix_free();
        Tensor<1, dim, Number> Force;
        for (unsigned int d = 0; d < dim; ++d)
          Force[d] = 0.0;
        FEFaceEvaluation<dim, -1, 0, 1, Number>   integrator_pressure(matrix_free,
                                                                    true,
                                                                    1,
                                                                    1);
        FEFaceEvaluation<dim, -1, 0, dim, Number> integrator_velocity(matrix_free,
                                                                      true,
                                                                      0,
                                                                      1);

        for (unsigned int face = matrix_free.n_inner_face_batches();
             face <
             (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
             face++)
          {
            integrator_velocity.reinit(face);
            integrator_velocity.read_dof_values(vec_u_old[0]);
            integrator_velocity.evaluate(dealii::EvaluationFlags::gradients);

            integrator_pressure.reinit(face);
            integrator_pressure.read_dof_values(vec_p);
            integrator_pressure.evaluate(dealii::EvaluationFlags::values);

            dealii::types::boundary_id boundary_id = matrix_free.get_boundary_id(face);
            if (boundary_id == 2)
              {
                for (unsigned int q = 0; q < integrator_velocity.n_q_points; ++q)
                  {
                    dealii::VectorizedArray<Number> pressure =
                      integrator_pressure.get_value(q);

                    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> normal =
                      integrator_velocity.normal_vector(q);
                    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
                      velocity_gradient = integrator_velocity.get_gradient(q);

                    dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> tau =
                      pressure * normal -
                      viscosity * (velocity_gradient + transpose(velocity_gradient)) *
                        normal;

                    integrator_velocity.submit_value(tau, q);
                  }

                dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> Force_local =
                  integrator_velocity.integrate_value();

                // sum over all entries of dealii::VectorizedArray
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    for (unsigned int n = 0;
                         n < matrix_free.n_active_entries_per_face_batch(face);
                         ++n)
                      Force[d] += Force_local[d][n];
                  }
              }
          }
        Force                     = dealii::Utilities::MPI::sum(Force, MPI_COMM_WORLD);
        const Number current_drag = Force[0] / (4. / 9. * u_x_max * u_x_max * 0.5 * 0.1);
        const Number current_lift = Force[1] / (4. / 9. * u_x_max * u_x_max * 0.5 * 0.1);

        drag_max = std::max(drag_max, current_drag);
        drag_min = std::min(drag_min, current_drag);
        lift_max = std::max(lift_max, current_lift);
        lift_min = std::min(lift_min, current_lift);
      }

      if (write_output && time_step_number % output_interval == 0)
        {
          double single_step = time_single_step.wall_time();
          pcout << "Time single step: " << single_step << std::endl;

          Vector<double> error_per_cell;
          exact_velocity.set_time(current_time);
          exact_pressure.set_time(current_time);

          VectorTools::integrate_difference(mapping,
                                            dof_handler_u,
                                            vec_u_old[0],
                                            exact_velocity,
                                            error_per_cell,
                                            QGauss<dim>(fe_u.degree + 3),
                                            VectorTools::L2_norm);
          const double velocity_error =
            VectorTools::compute_global_error(tria, error_per_cell, VectorTools::L2_norm);

          VectorTools::integrate_difference(mapping,
                                            dof_handler_p,
                                            vec_p,
                                            exact_pressure,
                                            error_per_cell,
                                            QGauss<dim>(fe_p.degree + 3),
                                            VectorTools::L2_norm);
          const double pressure_error =
            VectorTools::compute_global_error(tria, error_per_cell, VectorTools::L2_norm);


          pcout << "L2 norm velocity/pressure: " << velocity_error << " "
                << pressure_error << std::endl;

          DataOut<dim> data_out;

          DataOutBase::VtkFlags flags;
          flags.write_higher_order_cells = true;
          data_out.set_flags(flags);

          data_out.add_data_vector(dof_handler_u, vec_u_old[0], "solution");
          data_out.add_data_vector(dof_handler_p, vec_p, "pressure");
          Vector<double> mpi_owner(tria.n_active_cells());
          mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
          data_out.add_data_vector(mpi_owner, "owner");
          data_out.build_patches(mapping, fe_u.degree, DataOut<dim>::curved_inner_cells);

          const std::string filename =
            "solution-L2-" + std::to_string(time_step_number) + ".vtu";
          // "solution-L2-" + std::to_string(n_refinements) + "_p_" +
          // std::to_string(degree) + ".vtu";
          data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);

          pcout << "Max/min drag/lift: " << drag_max << " " << drag_min << " " << lift_max
                << " " << lift_min << std::endl;
        }
    }

  Vector<double> error_per_cell;
  exact_velocity.set_time(current_time);
  exact_pressure.set_time(current_time);

  VectorTools::integrate_difference(mapping,
                                    dof_handler_u,
                                    vec_u_old[0],
                                    exact_velocity,
                                    error_per_cell,
                                    QGauss<dim>(fe_u.degree + 3),
                                    VectorTools::L2_norm); // H1_seminorm);
  const double velocity_error =
    VectorTools::compute_global_error(tria,
                                      error_per_cell,
                                      VectorTools::L2_norm); // H1_seminorm);

  VectorTools::integrate_difference(mapping,
                                    dof_handler_p,
                                    vec_p,
                                    exact_pressure,
                                    error_per_cell,
                                    QGauss<dim>(fe_p.degree + 3),
                                    VectorTools::L2_norm);
  const double pressure_error =
    VectorTools::compute_global_error(tria, error_per_cell, VectorTools::L2_norm);

  pcout << "L2 norm velocity/pressure: " << velocity_error << " " << pressure_error
        << std::endl;
  pcout << std::scientific << std::setprecision(20) << "Max/min drag/lift: " << drag_max
        << " " << drag_min << " " << lift_max << " " << lift_min << std::endl;

  const double loop_time = time_loop.wall_time();
  pcout << "Time loop time: " << loop_time << std::endl;
  pcout << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  // for (unsigned int i = 1; i < 7; ++i)
  //   do_test<2, double>(3, i, 14);

  // for (unsigned int i = 1; i < 7; ++i)
  // do_test<2, double>(5, i, 0);

  // for (unsigned int i = 1; i < 15; ++i)
  // do_test<2, double>(5, 4, i);
  do_test<2, double>(3, 2, 0);
}
