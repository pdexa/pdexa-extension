
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
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

#include <fstream>

#include "preconditioners.h"
#include "evaluators.h"
#include "consistent_splitting_solver.h"

using namespace dealii;


const bool use_neumann_boundary                      = false;
const bool use_skew_symmetric_convective_formulation = false;
const bool use_divergence_formulation                = true;
const bool use_leray_projection                      = true;

// Always use MG as preconditioner for the pressure
// const bool use_amg                       = false;
const bool use_hmg                       = true;
const bool use_pmg                       = false;
const bool use_cmg                       = false;
// const bool use_pointjacobi_pressure      = false;
const bool use_amg_as_coarse_grid_solver = false;

// Always uses inverse mass, no need to set these variables
// const bool use_velocity_point_jacobi         = false;
// const bool use_inverse_mass_velocity         = true;
// const bool use_mg_velocity                   = false;
// const bool use_cmg_vel                       = false;
// const bool use_pmg_vel                       = false;
// const bool use_hmg_vel                       = false;
// const bool use_amg_as_coarse_grid_solver_vel = false;

const double penalty_divergence = 1.0;
const double penalty_continuity = 1.0;


const double viscosity = 0.025;
const double u_x_max   = 1.;

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
    if (component == 0)
      result = -std::sin(2. * pi * p[1]);
    else if (component == 1)
      result = std::sin(2. * pi * p[0]);

    result *= std::exp(-4. * viscosity * pi * pi * t);
    return result;
  }

  dealii::Tensor<1, dim, double>
  gradient(const dealii::Point<dim> &p, const unsigned int component = 0) const final
  {
    const double                   t  = this->get_time();
    const double                   pi = dealii::numbers::PI;
    dealii::Tensor<1, dim, double> result;
    if (component == 0)
      {
        result[0] = 0.;
        result[1] = -2. * pi * std::cos(2. * pi * p[1]);
      }
    else if (component == 1)
      {
        result[0] = 2. * pi * std::cos(2. * pi * p[0]);
        result[1] = 0.;
      }
    result *= std::exp(-4. * viscosity * pi * pi * t);
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
  value(const dealii::Point<dim> &p, const unsigned int /*component*/) const final
  {
    const double t  = this->get_time();
    const double pi = dealii::numbers::PI;

    const double result = -std::cos(2. * pi * p[0]) * std::cos(2. * pi * p[1]) *
                          std::exp(-8. * viscosity * pi * pi * t);

    return result;
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

  FESystem<dim>  fe_u(FE_DGQ<dim>(fe_degree), dim);
  FE_DGQ<dim>    fe_p(fe_degree - 1);
  MappingQ1<dim> mapping;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  double L = 1.;
  GridGenerator::hyper_cube(tria, -L / 2., L / 2.);

  if (use_neumann_boundary)
    {
      tria.begin()->face(0)->set_all_boundary_ids(1);
      tria.begin()->face(1)->set_all_boundary_ids(1);
      tria.begin()->face(2)->set_all_boundary_ids(1);
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

  const Number time_step = std::pow(0.5, n_refinements_time);
  // std::min(5.0 * 1e-5, dealii::Utilities::MPI::min(local_time_step, MPI_COMM_WORLD));
  pcout << "Time step size: " << time_step << std::endl;

  unsigned int bdf_order   = 4;
  unsigned int bdf_order_p = 3;

  BDFTimeIntegratorConstants bdf(bdf_order);
  BDFTimeIntegratorConstants bdf_p(bdf_order_p);

  MomentumOperator<dim, dim, Number> momentum_op;
  // set up operator
  momentum_op.reinit(mapping, dof_handler_u, dof_handler_p, time_step, bdf_order, use_skew_symmetric_convective_formulation, use_divergence_formulation);

  momentum_op.set_viscosity(viscosity);
  momentum_op.set_time(0.0);
  momentum_op.set_body_force_factory([=]() {
    return std::make_unique<AnalyticalRHS<dim>>(u_x_max, viscosity);
  });
  momentum_op.set_dirichletBC_pressure_factory ([=]() {
    return std::make_unique<AnalyticalSolutionPressure<dim>>(u_x_max, viscosity);
  });
  momentum_op.set_DirichletBC_velocity_factory ([=]() {
    return std::make_unique<AnalyticalSolutionVelocity<dim>>(u_x_max, viscosity);
  });

  LinearAlgebra::distributed::Vector<Number> vec_u, vec_u_deriv, vec_u_rhs, vec_p,
    vec_u_norm, speed_extrapolated, vec_vorticity, vec_p_rhs, vec_p_rhs_n, vec_p_norm,
    vec_div_u;
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

  std::vector<LinearAlgebra::distributed::Vector<double>> vec_u_old(bdf_order);
  for (auto &vec : vec_u_old)
    momentum_op.initialize_dof_vector(vec, dof_no_v);

  InverseMassPreconditioner<dim, Number> inverse_mass;
  inverse_mass.reinit(momentum_op.get_matrix_free(), time_step);

  PressureOperator<dim, double> pressure_op;
  pressure_op.reinit(momentum_op.get_matrix_free(), bdf_order, time_step, use_leray_projection);
  pressure_op.set_body_force_factory([=]() {
    return std::make_unique<AnalyticalRHS<dim>>(u_x_max, viscosity);
  });
  pressure_op.set_viscosity(viscosity);
  pressure_op.set_dirichletBC_pressure_factory ([=]() {
    return std::make_unique<AnalyticalSolutionPressure<dim>>(u_x_max, viscosity);
  });
  pressure_op.set_DirichletBC_velocity_factory ([=]() {
    return std::make_unique<AnalyticalSolutionVelocity<dim>>(u_x_max, viscosity);
  });

  MultigridPreconditioner<dim, Number, Number> precondition_hmg(pressure_op,
                                                                mapping.get_degree(),
                                                                time_step,
                                                                bdf_order,
                                                               use_hmg,
                                                               use_cmg,
                                                               use_pmg,   
                                                               use_amg_as_coarse_grid_solver,
                                                               use_neumann_boundary);

  Number      current_time          = 0;
  std::size_t n_momentum_iterations = 0, n_pressure_iterations = 0;

  AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
  AnalyticalSolutionPressure<dim> exact_pressure(u_x_max, viscosity);

  for (unsigned int i = 0; i < bdf_order; ++i)
    {
      exact_velocity.set_time(current_time);
      VectorTools::interpolate(mapping,
                               dof_handler_u,
                               exact_velocity,
                               vec_u_old[bdf_order - 1 - i]);

      current_time += time_step;
    }

  current_time -= time_step;
  const Number end_time          = 1.0;
  unsigned int time_step_number  = bdf.get_order() - 1;
  unsigned int n_performed_steps = 0;

  const bool write_output = false;
  while (current_time <= end_time)
    {
      current_time += time_step;
      ++time_step_number;
      ++n_performed_steps;
      momentum_op.set_time(current_time);

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


      if (!use_neumann_boundary)
        VectorTools::subtract_mean_value(vec_p_rhs);

      const unsigned int iteration_count =
        precondition_hmg.solve(pressure_op, vec_p, vec_p_rhs);
      if (!use_neumann_boundary)
        VectorTools::subtract_mean_value(vec_p);
      if (write_output)
        pcout << "Pressure solver: " << iteration_count << " iterations" << std::endl;
      n_pressure_iterations += iteration_count;

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

      ReductionControl control_mom(10000, 1e-12, 1e-6);
      SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData gmres_data;
      gmres_data.max_basis_size        = 100;
      gmres_data.right_preconditioning = true;
      SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver_mom(control_mom,
                                                                         gmres_data);
      //SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver_mom(control_mom);
      inverse_mass.set_scaling_factor(time_step / bdf.get_gamma0());
      vec_u.swap(speed_extrapolated); // = 0.;
      solver_mom.solve(momentum_op, vec_u, vec_u_rhs, inverse_mass);

      if (write_output)
        pcout << "Momentum solver: " << control_mom.last_step() << " iterations"
              << std::endl;

      n_momentum_iterations += control_mom.last_step();

      // exact_velocity.set_time(current_time);
      // VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u);

      for (unsigned int i = bdf.get_order() - 1; i != 0; --i)
        {
          std::swap(vec_u_old[i], vec_u_old[i - 1]);
        }

      vec_u_old[0].swap(vec_u);

      if (write_output)
        {
          Vector<double> error_per_cell;
          Vector<double> norm_per_cell;
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

          vec_u_norm = 0.;
          VectorTools::integrate_difference(mapping,
                                            dof_handler_u,
                                            vec_u_norm,
                                            exact_velocity,
                                            norm_per_cell,
                                            QGauss<dim>(fe_u.degree + 3),
                                            VectorTools::L2_norm);
          const double velocity_norm =
            VectorTools::compute_global_error(tria, norm_per_cell, VectorTools::L2_norm);

          vec_p_norm = 0.;
          VectorTools::integrate_difference(mapping,
                                            dof_handler_p,
                                            vec_p_norm,
                                            exact_pressure,
                                            norm_per_cell,
                                            QGauss<dim>(fe_p.degree + 3),
                                            VectorTools::L2_norm);
          const double pressure_norm =
            VectorTools::compute_global_error(tria, norm_per_cell, VectorTools::L2_norm);


          pcout << "L2 error velocity/pressure: " << velocity_error / velocity_norm << " "
                << pressure_error / pressure_norm << std::endl;
          pcout << std::endl;

          DataOut<dim> data_out;

          DataOutBase::VtkFlags flags;
          flags.write_higher_order_cells = true;
          data_out.set_flags(flags);

          data_out.add_data_vector(dof_handler_u, vec_u_old[0], "solution");
          VectorTools::interpolate(mapping,
                                   dof_handler_u,
                                   exact_velocity,
                                   speed_extrapolated);
          data_out.add_data_vector(dof_handler_u, speed_extrapolated, "analytical");
          data_out.add_data_vector(dof_handler_p, vec_p, "pressure");
          VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p_rhs);
          data_out.add_data_vector(dof_handler_p, vec_p_rhs, "pressure_analytical");
          Vector<double> mpi_owner(tria.n_active_cells());
          mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
          data_out.add_data_vector(mpi_owner, "owner");
          data_out.build_patches(mapping, fe_u.degree, DataOut<dim>::curved_inner_cells);

          const std::string filename =
            "solution-L2-" + std::to_string(time_step_number) + ".vtu";
          // "solution-L2-" + std::to_string(n_refinements) + "_p_" +
          // std::to_string(degree) + ".vtu";
          data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
        }
    }

  Vector<double> error_per_cell;
  Vector<double> norm_per_cell;
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

  vec_u_norm = 0.;
  VectorTools::integrate_difference(mapping,
                                    dof_handler_u,
                                    vec_u_norm,
                                    exact_velocity,
                                    norm_per_cell,
                                    QGauss<dim>(fe_u.degree + 3),
                                    VectorTools::L2_norm); // H1_seminorm);
  const double velocity_norm =
    VectorTools::compute_global_error(tria,
                                      norm_per_cell,
                                      VectorTools::L2_norm); // H1_seminorm);

  vec_p_norm = 0.;
  VectorTools::integrate_difference(mapping,
                                    dof_handler_p,
                                    vec_p_norm,
                                    exact_pressure,
                                    norm_per_cell,
                                    QGauss<dim>(fe_p.degree + 3),
                                    VectorTools::L2_norm);
  const double pressure_norm =
    VectorTools::compute_global_error(tria, norm_per_cell, VectorTools::L2_norm);

  pcout << "L2 error velocity/pressure: " << velocity_error / velocity_norm << " "
        << pressure_error / pressure_norm << std::endl;
  pcout << "Average iteration count in " << n_performed_steps
        << " time steps for pressure / momentum: "
        << static_cast<double>(n_pressure_iterations) / n_performed_steps << " / "
        << static_cast<double>(n_momentum_iterations) / n_performed_steps << std::endl;
  pcout << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

   // for (unsigned int i = 1; i < 7; ++i)
   //  do_test<2, double>(3, i, 14);

  // for (unsigned int i = 1; i < 7; ++i)
  //   do_test<2, double>(5, i, 14);

  for (unsigned int i = 1; i < 15; ++i)
    do_test<2, double>(5, 4, i);
  // do_test<2, double>(5, 4, 14);
}