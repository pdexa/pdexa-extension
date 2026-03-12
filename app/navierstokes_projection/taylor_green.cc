
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
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "consistent_splitting_solver.h"
#include "evaluators.h"
#include "preconditioners.h"

using namespace dealii;


const bool use_neumann_boundary                      = false;
const bool use_skew_symmetric_convective_formulation = false;
const bool use_divergence_formulation                = true;
const bool use_leray_projection                      = true;

const bool use_amg                       = false;
const bool use_hmg                       = true;
const bool use_pmg                       = true;
const bool use_cmg                       = false;
const bool use_pointjacobi_pressure      = false;
const bool use_amg_as_coarse_grid_solver = false;

const bool use_velocity_point_jacobi         = false;
const bool use_velocity_block_jacobi_fdm     = true;
const bool use_inverse_mass_velocity         = false;
const bool use_mg_velocity                   = false;
const bool analyze_preconditioners           = false;
const bool use_cmg_vel                       = false;
const bool use_pmg_vel                       = false;
const bool use_hmg_vel                       = false;
const bool use_amg_as_coarse_grid_solver_vel = false;

const double penalty_divergence                 = 1.0;
const double penalty_continuity                 = 1.0;
const bool   do_penalty_terms_as_postprocessing = true;

const double upwind_factor = 1.0;

const double viscosity = 1. / 1600.0;
const double u_x_max   = 1.0;
const double L         = 1.0;

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
    double result = 0.0;
    if (component == 0)
      result = u_x_max * std::sin(p[0] / L) * std::cos(p[1] / L) * std::cos(p[2] / L);
    else if (component == 1)
      result = -u_x_max * std::cos(p[0] / L) * std::sin(p[1] / L) * std::cos(p[2] / L);

    return result;
  }

  dealii::Tensor<1, dim, double>
  gradient(const dealii::Point<dim> &, const unsigned int) const final
  {
    DEAL_II_NOT_IMPLEMENTED();
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
    return u_x_max * u_x_max / 16. * (std::cos(2. * p[0] / L) + std::cos(2. * p[1] / L)) *
           (std::cos(2 * p[2] / L) + 2.);
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
        const double       courant)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  Timer                time_setup;
  FESystem<dim>        fe_u(FE_DGQ<dim>(fe_degree), dim);
  FE_DGQ<dim>          fe_p(fe_degree - 1);
  MappingQGeneric<dim> mapping(fe_degree);

  parallel::distributed::Triangulation<dim> tria(
    MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices);

  pcout << "Running with fe_degree=" << fe_degree << ", n_refine=" << n_refinements
        << " and Courant=" << courant << std::endl;
  pcout << "Set up tria" << std::endl;
  GridGenerator::subdivided_hyper_cube(tria, 1, -L * numbers::PI, L * numbers::PI);
  pcout << "Set up boundary" << std::endl;

  const bool periodic_boundary = true;
  if (periodic_boundary)
    {
      if (use_neumann_boundary)
        DEAL_II_NOT_IMPLEMENTED();
      for (auto &cell : tria.cell_iterators())
        for (auto f : cell->face_indices())
          if (cell->face(f)->at_boundary())
            {
              if (std::abs(cell->face(f)->center()[0] + L * numbers::PI) < 1e-12)
                cell->face(f)->set_all_boundary_ids(0);
              else if (std::abs(cell->face(f)->center()[0] - L * numbers::PI) < 1e-12)
                cell->face(f)->set_all_boundary_ids(1);
              else if (std::abs(cell->face(f)->center()[1] + L * numbers::PI) < 1e-12)
                cell->face(f)->set_all_boundary_ids(2);
              else if (std::abs(cell->face(f)->center()[1] - L * numbers::PI) < 1e-12)
                cell->face(f)->set_all_boundary_ids(3);
              else if (std::abs(cell->face(f)->center()[2] + L * numbers::PI) < 1e-12)
                cell->face(f)->set_all_boundary_ids(4);
              else if (std::abs(cell->face(f)->center()[2] - L * numbers::PI) < 1e-12)
                cell->face(f)->set_all_boundary_ids(5);
              else
                DEAL_II_NOT_IMPLEMENTED();
            }
      //   tria.begin()->face(face)-->set_all_boundary_ids(face);

      std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
        periodic_faces;
      for (unsigned int d = 0; d < dim; ++d)
        GridTools::collect_periodic_faces(tria, 2 * d, 2 * d + 1, d, periodic_faces);
      tria.add_periodicity(periodic_faces);
    }
  pcout << "Refine tria" << std::endl;

  tria.refine_global(n_refinements);

  DoFHandler<dim> dof_handler_u(tria);
  dof_handler_u.distribute_dofs(fe_u);

  DoFHandler<dim> dof_handler_p(tria);
  dof_handler_p.distribute_dofs(fe_p);

  pcout << "Number of active_cells: " << tria.n_global_active_cells() << std::endl;
  pcout << "Solving with " << fe_u.get_name() << " x " << fe_p.get_name() << " element"
        << std::endl;
  pcout << "Number of degrees of freedom: " << dof_handler_u.n_dofs() << " + "
        << dof_handler_p.n_dofs() << std::endl;

  double h_min = std::numeric_limits<double>::max();
  for (const auto &cell : dof_handler_u.active_cell_iterators())
    h_min = std::min(h_min, cell->minimum_vertex_distance());
  h_min = Utilities::MPI::min(h_min, dof_handler_u.get_mpi_communicator());

  const double time_step = courant * h_min / u_x_max;
  pcout << "Time step size: " << time_step << std::endl;

  const unsigned int bdf_order   = 3;
  const unsigned int bdf_order_p = 2;

  MomentumOperator<dim, dim, Number> momentum_op;
  momentum_op.set_body_force_factory(
    [=]() { return std::make_unique<AnalyticalRHS<dim>>(u_x_max, viscosity); });
  momentum_op.set_dirichletBC_pressure_factory([=]() {
    return std::make_unique<AnalyticalSolutionPressure<dim>>(u_x_max, viscosity);
  });
  momentum_op.set_DirichletBC_velocity_factory([=]() {
    return std::make_unique<AnalyticalSolutionVelocity<dim>>(u_x_max, viscosity);
  });
  // set up operator
  momentum_op.reinit(mapping,
                     dof_handler_u,
                     dof_handler_p,
                     time_step,
                     bdf_order,
                     use_skew_symmetric_convective_formulation,
                     use_divergence_formulation,
                     upwind_factor);
  if (do_penalty_terms_as_postprocessing)
    {
      momentum_op.set_divergence_penalty(0.0);
      momentum_op.set_continuity_penalty(0.0);
    }
  PenaltyOperator<dim, dim, Number> penalty_operator;
  penalty_operator.reinit(momentum_op.get_matrix_free(),
                          penalty_divergence * time_step,
                          penalty_continuity * time_step);

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

  BlockJacobi::PreconditionerMomentum<dim, Number> preconditioner_block_jacobi(
    momentum_op.get_matrix_free(), 0, 1);

  MultigridPreconditionerVelocity<dim, Number, Number> preconditioner_velocity(
    momentum_op,
    mapping.get_degree(),
    viscosity,
    time_step,
    bdf_order,
    use_hmg_vel,
    use_cmg_vel,
    use_pmg_vel);

  DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>>
    preconditioner_velocity_pointjacobi;
  DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>>
    preconditioner_pressure_pointjacobi;


  PressureOperator<dim, Number> pressure_op;
  pressure_op.reinit(momentum_op.get_matrix_free(),
                     bdf_order,
                     time_step,
                     use_leray_projection);
  pressure_op.set_body_force_factory(
    [=]() { return std::make_unique<AnalyticalRHS<dim>>(u_x_max, viscosity); });
  pressure_op.set_time_step(time_step);
  pressure_op.set_viscosity(viscosity);
  pressure_op.set_dirichletBC_pressure_factory([=]() {
    return std::make_unique<AnalyticalSolutionPressure<dim>>(u_x_max, viscosity);
  });
  pressure_op.set_DirichletBC_velocity_factory([=]() {
    return std::make_unique<AnalyticalSolutionVelocity<dim>>(u_x_max, viscosity);
  });

  TrilinosWrappers::SparseMatrix pressure_system_matrix;
  if (use_amg)
    pressure_op.get_system_matrix(AffineConstraints<double>(), pressure_system_matrix);
  TrilinosWrappers::PreconditionAMG precondition_amg;

  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.smoother_sweeps = 1;
  amg_data.n_cycles        = 1;
  amg_data.smoother_type   = "ILU";

  if (use_amg)
    precondition_amg.initialize(pressure_system_matrix, amg_data);

  MultigridPreconditioner<dim, Number, Number> precondition_hmg(
    pressure_op,
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

  Number max_dissipation = 0.;
  {
    Number energy;
    Number enstrophy;
    Number dissipation;
    Number max_vorticity;

    momentum_op.calculate_energy(
      vec_u_old[0], energy, enstrophy, dissipation, max_vorticity);
    pcout << "Energy, enstrophy, dissipation, max vorticity at t=0: " << energy << " "
          << enstrophy << " " << dissipation << " " << max_vorticity << std::endl;
    max_dissipation = std::max(max_dissipation, dissipation);
  }

  const Number       end_time = 20.0;
  const unsigned int output_interval =
    std::max(static_cast<unsigned int>(1.0 / time_step), 1u);
  unsigned int time_step_number = 0;

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

      for (unsigned int i = 0; i < bdf.get_order(); ++i)
        {
          pressure_op.set_time(current_time - (i + 1) * time_step);
          vec_p_rhs_n = 0.;
          pressure_op.compute_convective_rhs(vec_p_rhs_n, vec_u_old[i]);
          vec_p_rhs.add(bdf.get_beta(i), vec_p_rhs_n);
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
      ReductionControl                                     control(10000, 1e-12, 1e-6);
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
      SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData gmres_data;
      gmres_data.max_basis_size        = 20;
      gmres_data.right_preconditioning = true;
      SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver_mom(control_mom,
                                                                         gmres_data);
      if (false)
        solver_mom.connect_eigenvalues_slot(
          [](const std::vector<std::complex<double>> &eigenvalues) {
            std::cout << "Eigenvalue estimate: ";
            for (const auto &a : eigenvalues)
              std::cout << ' ' << a;
            std::cout << std::endl;
          });
      vec_u = speed_extrapolated; // = 0.;
      unsigned int n_iterations_vel;
      if (use_mg_velocity)
        {
          n_iterations_vel = preconditioner_velocity.solve(
            momentum_op, vec_u, vec_u_rhs, use_amg_as_coarse_grid_solver_vel);
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
      else if (use_velocity_block_jacobi_fdm)
        {
          preconditioner_block_jacobi.reinit(speed_extrapolated,
                                             bdf.get_gamma0() / time_step);
          solver_mom.solve(momentum_op, vec_u, vec_u_rhs, preconditioner_block_jacobi);
          n_iterations_vel = control_mom.last_step();
        }
      else
        {
          solver_mom.solve(momentum_op, vec_u, vec_u_rhs, PreconditionIdentity());
          n_iterations_vel = control_mom.last_step();
        }

      if (write_output && time_step_number % output_interval == 0)
        pcout << "Momentum solver: " << n_iterations_vel << " iterations" << std::endl;

      if (do_penalty_terms_as_postprocessing)
        {
          penalty_operator.rhs(vec_u_rhs, vec_u);
          SolverControl control_penalty(500, 1e-8 * vec_u_rhs.l2_norm());
          SolverCG<LinearAlgebra::distributed::Vector<double>> solver_penalty(
            control_penalty);
          solver_penalty.solve(penalty_operator, vec_u, vec_u_rhs, inverse_mass);
          if (write_output && time_step_number % output_interval == 0)
            pcout << "Penalty solver: " << control_penalty.last_step() << " iterations"
                  << std::endl;
        }

      if (analyze_preconditioners)
        {
          momentum_op.set_divergence_penalty(0.0);
          momentum_op.set_continuity_penalty(0.0);
          vec_u_rhs = 0.;
          momentum_op.rhs(vec_u_rhs, vec_u_deriv, speed_extrapolated, vec_p);
          inverse_mass.set_scaling_factor(time_step / bdf.get_gamma0());
          Timer time;
          vec_u = 0;
          solver_mom.solve(momentum_op, vec_u, vec_u_rhs, inverse_mass);
          pcout << "Precondition inverse mass: " << control_mom.last_step()
                << " iterations in time " << time.wall_time() << std::endl;

          if (time_step_number % 10 == 1)
            momentum_op.compute_inverse_diagonal(
              preconditioner_velocity_pointjacobi.get_vector());
          vec_u = 0;
          time.restart();
          solver_mom.solve(momentum_op,
                           vec_u,
                           vec_u_rhs,
                           preconditioner_velocity_pointjacobi);
          pcout << "Precondition point Jacobi: " << control_mom.last_step()
                << " iterations in time " << time.wall_time() << std::endl;

          preconditioner_block_jacobi.reinit(speed_extrapolated,
                                             bdf.get_gamma0() / time_step);
          vec_u = 0;
          time.restart();
          solver_mom.solve(momentum_op, vec_u, vec_u_rhs, preconditioner_block_jacobi);
          pcout << "Precondition block Jacobi: " << control_mom.last_step()
                << " iterations in time " << time.wall_time() << std::endl;

          if (!do_penalty_terms_as_postprocessing)
            {
              momentum_op.set_divergence_penalty(penalty_divergence);
              momentum_op.set_continuity_penalty(penalty_continuity);
            }
        }

      // exact_velocity.set_time(current_time);
      // VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u);

      for (unsigned int i = bdf_order - 1; i != 0; --i)
        {
          std::swap(vec_u_old[i], vec_u_old[i - 1]);
        }

      vec_u_old[0].swap(vec_u);



      if (write_output && time_step_number % output_interval == 0)
        {
          double single_step = time_single_step.wall_time();
          pcout << "Time single step: " << single_step << std::endl;

          {
            Number energy;
            Number enstrophy;
            Number dissipation;
            Number max_vorticity;

            momentum_op.calculate_energy(
              vec_u_old[0], energy, enstrophy, dissipation, max_vorticity);
            pcout << "Energy, enstrophy, dissipation, max vorticity at t=" << current_time
                  << ": " << energy << " " << enstrophy << " " << dissipation << " "
                  << max_vorticity << std::endl;
            max_dissipation = std::max(max_dissipation, dissipation);
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

          if (true)
            {
              DataOut<dim> data_out;

              DataOutBase::VtkFlags flags;
              flags.write_higher_order_cells = true;
              data_out.set_flags(flags);

              std::vector<DataComponentInterpretation::DataComponentInterpretation>
                data_component_interpretation(
                  dim, DataComponentInterpretation::component_is_part_of_vector);
              data_out.add_data_vector(dof_handler_u,
                                       vec_u_old[0],
                                       std::vector<std::string>(dim, "velocity"),
                                       data_component_interpretation);
              data_out.add_data_vector(dof_handler_p, vec_p, "pressure");
              Vector<double> mpi_owner(tria.n_active_cells());
              mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
              data_out.add_data_vector(mpi_owner, "owner");
              data_out.build_patches(mapping,
                                     fe_u.degree,
                                     DataOut<dim>::curved_inner_cells);

              const std::string filename =
                "solution-taylor_green-" +
                std::to_string(time_step_number / output_interval) + ".vtu";
              // "solution-L2-" + std::to_string(n_refinements) + "_p_" +
              // std::to_string(degree) + ".vtu";
              data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
            }
          pcout << std::endl;
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

  {
    Number energy;
    Number enstrophy;
    Number dissipation;
    Number max_vorticity;

    momentum_op.calculate_energy(
      vec_u_old[0], energy, enstrophy, dissipation, max_vorticity);
    pcout << "Energy, enstrophy, dissipation, max vorticity at t=" << current_time << ": "
          << energy << " " << enstrophy << " " << dissipation << " " << max_vorticity
          << std::endl;
    max_dissipation = std::max(max_dissipation, dissipation);
  }

  pcout << "Max dissipation is: " << max_dissipation << std::endl;
  const double loop_time = time_loop.wall_time();
  pcout << "Time loop time: " << loop_time << std::endl;
  pcout << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  unsigned int dim      = 2;
  unsigned int n_refine = 2;
  unsigned int degree   = 4;
  double       courant  = 0.2;

  if (argc % 2 == 0)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Error, expected odd number of common line arguments!" << std::endl
                  << "Expected line of the form (or permutation of)" << std::endl
                  << "dim 3 n_refine 2 degree 4 courant 0.2" << std::endl;
      std::abort();
    }

  // parse from the command line
  for (int l = 1; l < argc; l += 2)
    {
      std::string option = argv[l];
      if (option == "dim")
        dim = std::atoll(argv[l + 1]);
      else if (option == "n_refine")
        n_refine = std::atoll(argv[l + 1]);
      else if (option == "degree")
        degree = std::atoll(argv[l + 1]);
      else if (option == "courant")
        courant = std::atof(argv[l + 1]);
      else
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Given command-line argument " << option << " not supported!"
                      << std::endl
                      << "Expected line of the form (or premutation of)" << std::endl
                      << "dim 3 n_refine 2 degree 4 courant 0.2" << std::endl;
          std::abort();
        }
    }


  if (dim == 2)
    do_test<2, double>(degree, n_refine, courant);
  else
    do_test<3, double>(degree, n_refine, courant);
}
