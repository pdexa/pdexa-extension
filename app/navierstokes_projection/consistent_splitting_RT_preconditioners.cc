
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "deformed_cube_manifold_exaDG.h"

#include <fstream>
#include <sstream>

#include <typeinfo>


#include "consistent_splitting_solver_RT.h"
#include "evaluators.h"
#include "preconditioners.h"

using namespace dealii;


const bool use_extrapolated_velocity                 = false;
const bool use_pressure_convective_upwind_flux       = false;
const bool use_neumann_boundary                      = false;
const bool use_pure_neummann_boundary                = false;
const bool use_periodic_boundary                     = false;
const bool use_analytical_curl                       = false;
const bool use_skew_symmetric_convective_formulation = false;
const bool use_leray_projection                      = true;
const bool use_stationary_stokes                     = false;
const bool use_Hdiv_projection                       = false;


const bool use_hmg                       = true;
const bool use_pmg                       = false;
const bool use_cmg                       = false;

const bool use_amg_as_coarse_grid_solver = false;

const bool use_inverse_mass_velocity = false;
const bool use_Jacobi_velocity       = false;

const bool apply_manifold = false;

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

template <int dim>
class DivergencePostprocessor : public DataPostprocessorScalar<dim>
{
public:
  DivergencePostprocessor()
    : DataPostprocessorScalar<dim>("divergence", update_gradients)
  {}

  virtual void evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &input_data,
    std::vector<Vector<double>> &               computed_results) const override
  {
    for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        double div = 0;
        for (unsigned int d = 0; d < dim; ++d)
          // The divergence is the trace of the gradient tensor
          div += input_data.solution_gradients[p][d][d];
          
        computed_results[p][0] = div;
      }
  }
};




template <int dim, typename Number>
void
do_test(const unsigned int fe_degree,
        const unsigned int n_refinements,
        const unsigned int n_refinements_time = 14,
        const double deformation_coefficient = 1.0)
{
  const Number time_step = std::pow(0.5, n_refinements_time);
  std::ostringstream filename;   
  /*filename << "space_converg_k"
	     <<fe_degree
	     <<"_refinement"
             << n_refinements
             <<"_stabilizedP.txt"   ;*/
  filename<<"timeConvergence_divU_timeStep_"
	  << std::scientific << std::setprecision(6)
	  <<time_step<<"_BDF2_stabilizedP.txt";
  std::ofstream fout(filename.str());
  ConditionalOStream pfout(fout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);


  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  std::cout<<std::setprecision(12);

  FESystem<dim>  fe_curl(FE_DGQ<dim>(fe_degree + 1), dim);
  //fe_u = std::make_shared<dealii::FE_RaviartThomasNodal<dim>>(param.degree_u - 1);
  FE_RaviartThomasNodal<dim>    fe_u(fe_degree);
  FE_DGQ<dim>    fe_p(fe_degree);
  MappingQ1<dim> mapping;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  double L = 1.;
  GridGenerator::hyper_cube(tria, -L / 2., L / 2.);
  const double Left = -L / 2.;
  const double Right = L / 2.;
  // Consider a mapped interior of the grid if desired.
  if(apply_manifold)
    {
      unsigned int const frequency   = 1;
      double const       deformation = (Right- Left) * deformation_coefficient;
      apply_deformed_cube_manifold(tria, Left, Right, deformation, frequency);
    }

  if (use_neumann_boundary)
  {
    tria.begin()->face(0)->set_all_boundary_ids(1);
    tria.begin()->face(1)->set_all_boundary_ids(1);
  }
  else if (use_pure_neummann_boundary)
  {
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
      ++face)
      tria.begin()->face(face)->set_all_boundary_ids(1);
  }
  else if (use_periodic_boundary)
  {
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
      ++face)
      tria.begin()->face(face)->set_all_boundary_ids(face);
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_faces;
    for (unsigned int d = 0; d < dim; ++d)
      GridTools::collect_periodic_faces(
        tria, 2 * d, 2 * d + 1, d, periodic_faces);
    tria.add_periodicity(periodic_faces);
  }
  
  tria.refine_global(n_refinements);
  
  if(apply_manifold)
  {
    pcout<<"Max deformation eigenvalue: "<<max_deformation_eigenvalue<<std::endl;
  }

  DoFHandler<dim> dof_handler_u(tria);
  dof_handler_u.distribute_dofs(fe_u);
  DoFHandler<dim> dof_handler_p(tria);
  dof_handler_p.distribute_dofs(fe_p);
  DoFHandler<dim> dof_handler_curl(tria);
  dof_handler_curl.distribute_dofs(fe_curl);

  /*std::ostringstream filename;
  filename << "DeformationRobust_deformation_"
             << std::scientific << std::setprecision(2)
             << deformation_coefficient
	          <<"_k"<<fe_degree
             << ".txt";
  std::ofstream fout(filename.str());
  ConditionalOStream pfout(fout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);*/

  pcout << "number of active_cells: " << tria.n_global_active_cells() << std::endl;
  pcout << "Solving with " << fe_u.get_name() << " x " << fe_p.get_name() << " element"
        << std::endl;
  pcout << "number of degrees of freedom: " << dof_handler_u.n_dofs() << " + "
        << dof_handler_p.n_dofs() << std::endl;
  
  pcout << "number of degrees of freedom (for curl): " << dof_handler_curl.n_dofs() << std::endl;

  double h_min = std::numeric_limits<double>::max();
  for (const auto &cell : dof_handler_u.active_cell_iterators())
    h_min = std::min(h_min, cell->minimum_vertex_distance());

  // std::min(5.0 * 1e-5, dealii::Utilities::MPI::min(local_time_step, MPI_COMM_WORLD));
  pcout << "Time step size: " << time_step << std::endl;

  unsigned int bdf_order   = 2;
  unsigned int bdf_order_p = 2;
  unsigned int bdf_order_c = 2;

  BDFTimeIntegratorConstants bdf(bdf_order);
  BDFTimeIntegratorConstants bdf_p(bdf_order_p);
  BDFTimeIntegratorConstants bdf_c(bdf_order_c);

  MomentumOperator<dim, dim, Number> momentum_op;

  Number divergence_norm = 0.0;
  // set up operator
  momentum_op.reinit(mapping, dof_handler_u, dof_handler_p, dof_handler_curl, time_step, bdf_order);

  momentum_op.set_viscosity(viscosity);
  momentum_op.set_time(0.0);
  momentum_op.set_body_force_factory(
    [=]() { return std::make_unique<AnalyticalRHS<dim>>(u_x_max, viscosity); });
  momentum_op.set_dirichletBC_pressure_factory([=]() {
    return std::make_unique<AnalyticalSolutionPressure<dim>>(u_x_max, viscosity);
  });
  momentum_op.set_DirichletBC_velocity_factory([=]() {
    return std::make_unique<AnalyticalSolutionVelocity<dim>>(u_x_max, viscosity);
  });

  LinearAlgebra::distributed::Vector<Number> vec_u, vec_u_deriv, vec_u_rhs, vec_p, vec_u_extrapolated,
    vec_p_extrapolated, vec_p_projected, vec_u_norm, speed_extrapolated, vec_vorticity, vec_vorticity_rhs, vec_p_rhs,
    vec_p_norm, vec_laplacian_u, vec_div_u, vec_p_rhs_n;
  momentum_op.initialize_dof_vector(vec_u, dof_no_v);
  momentum_op.initialize_dof_vector(vec_u_deriv, dof_no_v);
  momentum_op.initialize_dof_vector(vec_u_rhs, dof_no_v);
  momentum_op.initialize_dof_vector(vec_p, dof_no_p);
  momentum_op.initialize_dof_vector(vec_p_extrapolated, dof_no_p);
  momentum_op.initialize_dof_vector(vec_p_projected, dof_no_v);
  momentum_op.initialize_dof_vector(vec_u_extrapolated, dof_no_v);                
  momentum_op.initialize_dof_vector(vec_u_norm, dof_no_v);
  momentum_op.initialize_dof_vector(speed_extrapolated, dof_no_v);
  momentum_op.initialize_dof_vector(vec_vorticity, use_Hdiv_projection ? dof_no_v : dof_no_curl);
  momentum_op.initialize_dof_vector(vec_vorticity_rhs, use_Hdiv_projection ? dof_no_v : dof_no_curl);
  momentum_op.initialize_dof_vector(vec_laplacian_u, dof_no_curl);
  momentum_op.initialize_dof_vector(vec_p_rhs, dof_no_p);
  momentum_op.initialize_dof_vector(vec_p_rhs_n, dof_no_p);
  momentum_op.initialize_dof_vector(vec_p_norm, dof_no_p);

  momentum_op.initialize_dof_vector(vec_div_u, dof_no_p);

  InverseMassPreconditionerRT<dim, Number> inverse_mass;
  inverse_mass.reinit(momentum_op.get_matrix_free(), time_step);

  DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>>
    preconditioner_velocity_pointjacobi;


  std::vector<LinearAlgebra::distributed::Vector<double>> vec_u_old(bdf_order);
  for (auto &vec : vec_u_old)
    momentum_op.initialize_dof_vector(vec, dof_no_v);

  PressureOperator<dim, double> pressure_op;

  pressure_op.set_viscosity(viscosity);
  pressure_op.reinit(momentum_op.get_matrix_free(), bdf_order, time_step, use_leray_projection);
  //pressure_op.set_time_step(time_step);
  pressure_op.set_body_force_factory(
    [=]() { return std::make_unique<AnalyticalRHS<dim>>(u_x_max, viscosity); });
  pressure_op.set_dirichletBC_pressure_factory([=]() {
    return std::make_unique<AnalyticalSolutionPressure<dim>>(u_x_max, viscosity);
  });
  pressure_op.set_DirichletBC_velocity_factory([=]() {
    return std::make_unique<AnalyticalSolutionVelocity<dim>>(u_x_max, viscosity);
  });

  ProjectorOperator<dim> projector_op;
  projector_op.reinit(momentum_op.get_matrix_free());

  Number current_time = 0;

  AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
  AnalyticalSolutionPressure<dim> exact_pressure(u_x_max, viscosity);
  AnalyticalRHS<dim> exact_rhs(u_x_max, viscosity);

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
    
  std::size_t n_momentum_iterations = 0, n_pressure_iterations = 0;


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

  const bool write_its    = true;
  const bool write_output = true;
  const unsigned int vtk_output_interval = 4000;
  unsigned int iteration_count; 
  std::cout<<std::setprecision(20);

  while (current_time <= end_time)
    {
      if(time_step_number % vtk_output_interval == 0)
        pcout<<"Time step "<<time_step_number<<", time: "<<current_time<<"\n";
      current_time += time_step;
      ++time_step_number;
      ++n_performed_steps;

      pressure_op.set_time(current_time);
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

      for (unsigned int i = 0; i < bdf_c.get_order(); ++i)
        {
          pressure_op.set_time(current_time - (i + 1) * time_step);
          vec_p_rhs_n = 0.;
          pressure_op.compute_convective_rhs(vec_p_rhs_n, vec_u_old[i]);
          vec_p_rhs.add(bdf_c.get_beta(i), vec_p_rhs_n);
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


      if (!use_neumann_boundary && !use_pure_neummann_boundary)
        VectorTools::subtract_mean_value(vec_p_rhs);
      //pcout<<"vec p rhs norm before pressure solve: "<<vec_p_rhs.l2_norm()<<"\n";
      SolverControl control(100000, 1e-12 * vec_p_rhs.l2_norm());
      //ReductionControl control(100000, 1e-12, 1e-6);
      SolverCG<LinearAlgebra::distributed::Vector<double>> solver(control);
      if (use_cmg || use_hmg || use_pmg)
      {
        iteration_count = precondition_hmg.solve(pressure_op, vec_p, vec_p_rhs);
      }
      else
      {
        solver.solve(pressure_op, vec_p, vec_p_rhs, PreconditionIdentity());
        iteration_count = control.last_step();
      }
      if (!use_neumann_boundary && !use_pure_neummann_boundary)
        VectorTools::subtract_mean_value(vec_p);
      
      if (write_output && time_step_number % vtk_output_interval == 0)
        pcout << "Pressure solver: " << iteration_count << " iterations" << std::endl;

      // Momentum step
      vec_u_deriv        = 0.;
      speed_extrapolated = 0.;
      //vec_p_extrapolated = 0.;

      for (unsigned int i = 0; i < bdf.get_order(); ++i)
        {
          vec_u_deriv.add(bdf.get_alpha(i) / time_step, vec_u_old[i]);
          speed_extrapolated.add(bdf.get_beta(i), vec_u_old[i]);
          //vec_p_extrapolated.add(bdf.get_beta(i), vec_p_old[i]);
        }
      
      /*exact_pressure.set_time(current_time);
      VectorTools::interpolate(mapping,
                               dof_handler_p,
                               exact_pressure,
                               vec_p_extrapolated);*/
      vec_u_rhs = 0.;
      momentum_op.rhs(vec_u_rhs, vec_u_deriv, speed_extrapolated, vec_p);
      //if(write_output && time_step_number % vtk_output_interval == 0)
      //  pcout<<"vec u rhs norm before momentum solve: "<<vec_u_rhs.l2_norm()<<"\n";
      vec_u = 0.;
      //vec_u.update_ghost_values();
      
      
      SolverControl control_mom(100000, 1e-12 * vec_u_rhs.l2_norm());
      //ReductionControl control_mom(100000, 1e-12, 1e-6);
      SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver_mom(control_mom);
      unsigned int n_iterations_vel;
      if (use_inverse_mass_velocity)
        {
          inverse_mass.set_scaling_factor(time_step / bdf.get_gamma0());
          solver_mom.solve(momentum_op, vec_u, vec_u_rhs, inverse_mass);
          n_iterations_vel = control_mom.last_step();
        }
      else if (use_Jacobi_velocity)
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
          if(write_output && time_step_number % vtk_output_interval == 0)
            pcout<<"momentum solver without preconditioner"<<std::endl;
          solver_mom.solve(momentum_op, vec_u, vec_u_rhs, PreconditionIdentity());
          n_iterations_vel = control_mom.last_step();
        }
      if(use_periodic_boundary || use_pure_neummann_boundary)
        VectorTools::subtract_mean_value(vec_u);
      if (write_output && time_step_number % vtk_output_interval == 0)
        pcout << "Momentum solver: " << n_iterations_vel << " iterations"
              << std::endl;
      if(write_output && time_step_number % vtk_output_interval == 0)
        pcout<<"vec u norm after momentum solve: "<<vec_u.l2_norm()<<"\n";

      /*exact_velocity.set_time(current_time);
      VectorTools::interpolate(mapping,
                               dof_handler_u,
                               exact_velocity,
                               vec_u);*/
      for (unsigned int i = bdf.get_order() - 1; i != 0; --i)
        {
          std::swap(vec_u_old[i], vec_u_old[i - 1]);
        }

      vec_u_old[0].swap(vec_u);
      
      if (write_output && time_step_number % vtk_output_interval == 0)
        {    
          if(use_periodic_boundary)
          {
            momentum_op.get_constraints().distribute(vec_u_old[0]);
            //vec_u.update_ghost_values();
            //momentum_op.get_constraints().distribute(vec_vorticity);
            //vec_vorticity.update_ghost_values();
          }
          Vector<double> error_per_cell;
          Vector<double> norm_per_cell;
          exact_velocity.set_time(current_time);
          exact_pressure.set_time(current_time);
          exact_rhs.set_time(current_time);
          VectorTools::interpolate(mapping,
                                            dof_handler_p,
                                            exact_pressure,
                                            vec_p_extrapolated);
          
          VectorTools::interpolate(mapping,
                                            dof_handler_u,
                                            exact_rhs,
                                            vec_u_rhs);
          
          if(std::abs(vec_p_extrapolated.mean_value()) > 1e-18)
            vec_p.add(vec_p_extrapolated.mean_value());
          
          VectorTools::integrate_difference(mapping,
                                            dof_handler_p,
                                            vec_p,
                                            exact_pressure,
                                            error_per_cell,
                                            QGauss<dim>(fe_p.degree + 3),
                                            VectorTools::L2_norm);
          const double pressure_error =
            VectorTools::compute_global_error(tria, error_per_cell, VectorTools::L2_norm);
          vec_u_old[0].update_ghost_values();
          VectorTools::integrate_difference(mapping,
                                            dof_handler_u,
                                            vec_u_old[0],
                                            exact_velocity,
                                            error_per_cell,
                                            QGauss<dim>(fe_u.degree + 5),
                                            VectorTools::L2_norm);
          vec_u_old[0].zero_out_ghost_values();
          const double velocity_error =
            VectorTools::compute_global_error(tria, error_per_cell, VectorTools::L2_norm);

          vec_u_norm = 0.;
          vec_u_norm.update_ghost_values();
          VectorTools::integrate_difference(mapping,
                                            dof_handler_u,
                                            vec_u_norm,
                                            exact_velocity,
                                            norm_per_cell,
                                            QGauss<dim>(fe_u.degree + 3),
                                            VectorTools::L2_norm);
          vec_u_norm.zero_out_ghost_values();
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

          
          pcout << "velocity error norm: "<<velocity_error<<"\n";
          pcout << "L2 error velocity/pressure: " << velocity_error / velocity_norm << " "
                << pressure_error / pressure_norm << std::endl;
          momentum_op.calculate_divergence_norm(vec_u_old[0], divergence_norm);
          pcout<<"Divergence L2 norm: "<<divergence_norm<<"\n";
          pcout << std::endl;
          vec_u_old[0].zero_out_ghost_values();
          DataOut<dim> data_out;

          DataOutBase::VtkFlags flags;
          flags.write_higher_order_cells = true;
          data_out.set_flags(flags);

          

          data_out.add_data_vector(dof_handler_u, vec_u, "velocity");
          DivergencePostprocessor<dim> div_processor;
          data_out.add_data_vector(dof_handler_u, vec_u, div_processor);
          VectorTools::interpolate(mapping,
                                   dof_handler_u,
                                   exact_velocity,
                                   speed_extrapolated);
          data_out.add_data_vector(dof_handler_u, speed_extrapolated, "velocity_analytical");

          data_out.add_data_vector(dof_handler_p, vec_p, "pressure");
          VectorTools::interpolate(mapping,
                                   dof_handler_p,
                                   exact_pressure,
                                   vec_p_extrapolated);
          VectorTools::subtract_mean_value(vec_p_extrapolated);
          data_out.add_data_vector(dof_handler_p,
                                   vec_p_extrapolated,
                                   "pressure_analytical");
          Vector<double> mpi_owner(tria.n_active_cells());
          mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
          data_out.add_data_vector(mpi_owner, "owner");
          data_out.build_patches(mapping, 0, DataOut<dim>::curved_inner_cells);

          const std::string filename =
            "solution-L2-NS_Dirichlet-checkDiv-" + std::to_string(time_step_number) + ".vtu";
          data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
          if(use_periodic_boundary)
          {
            vec_u_old[0].zero_out_ghost_values();
            vec_vorticity.zero_out_ghost_values();
          }
          if(!use_neumann_boundary && !use_pure_neummann_boundary)
          {
            VectorTools::subtract_mean_value(vec_p);
          }
        }
    }

    
  //vec_u_old[0].update_ghost_values();
  if(use_periodic_boundary)
  {
    momentum_op.get_constraints().distribute(vec_u_old[0]);
  }
  
  Vector<double> error_per_cell;
  Vector<double> norm_per_cell;
  exact_velocity.set_time(current_time);
  exact_pressure.set_time(current_time);

  vec_u_old[0].update_ghost_values();
  VectorTools::integrate_difference(mapping,
                                    dof_handler_u,
                                    vec_u_old[0],
                                    exact_velocity,
                                    error_per_cell,
                                    QGauss<dim>(fe_u.degree + 3),
                                    VectorTools::L2_norm); // H1_seminorm);
  momentum_op.calculate_divergence_norm(vec_u_old[0], divergence_norm);
          pcout<<"Divergence L2 norm: "<<divergence_norm<<"\n";
 vec_u_old[0].zero_out_ghost_values();
  const double velocity_error =
    VectorTools::compute_global_error(tria,
                                      error_per_cell,
                                      VectorTools::L2_norm); // H1_seminorm);
  
  VectorTools::interpolate(mapping,
                            dof_handler_p,
                            exact_pressure,
                            vec_p_extrapolated);
          
    pcout<<std::setprecision(12);
    pcout<<"viscosity: "<<viscosity<<"\n";
  if(std::abs(vec_p_extrapolated.mean_value() - vec_p.mean_value()) > 1e-16 && 
     std::abs(vec_p_extrapolated.mean_value()) > 1e-18)
    vec_p.add(vec_p_extrapolated.mean_value());
   
   /*if(fe_degree == 1)
    vec_p.update_ghost_values(); */  
  VectorTools::integrate_difference(mapping,
                                      dof_handler_p,
                                      vec_p,
                                      exact_pressure,
                                      error_per_cell,
                                      QGauss<dim>(fe_p.degree + 3),
                                      VectorTools::L2_norm);
  /*if (fe_degree == 1)
    vec_p.zero_out_ghost_values(); */

  const double pressure_error =
    VectorTools::compute_global_error(tria, error_per_cell, VectorTools::L2_norm);

  vec_u_norm = 0.;
  vec_u_norm.update_ghost_values();
  VectorTools::integrate_difference(mapping,
                                    dof_handler_u,
                                    vec_u_norm,
                                    exact_velocity,
                                    norm_per_cell,
                                    QGauss<dim>(fe_u.degree + 3),
                                    VectorTools::L2_norm); // H1_seminorm);
  vec_u_norm.zero_out_ghost_values();
  const double velocity_norm =
    VectorTools::compute_global_error(tria,
                                      norm_per_cell,
                                      VectorTools::L2_norm); // H1_seminorm);

  vec_p_norm = 0.;
  /*if (fe_degree == 1)
    vec_p_norm.update_ghost_values();*/
  VectorTools::integrate_difference(mapping,
                                    dof_handler_p,
                                    vec_p_norm,
                                    exact_pressure,
                                    norm_per_cell,
                                    QGauss<dim>(fe_p.degree + 3),
                                    VectorTools::L2_norm);
  /*if (fe_degree == 1)
    vec_p_norm.zero_out_ghost_values();*/
  const double pressure_norm =
    VectorTools::compute_global_error(tria, norm_per_cell, VectorTools::L2_norm);

  pcout << "L2 error velocity/pressure: " << velocity_error / velocity_norm << " "
        << pressure_error / pressure_norm << std::endl;
  pcout << std::endl;

  pfout<<std::setprecision(12);
  pfout<<"degree/time step: \t"<<fe_degree<<"/"<<time_step<<std::endl;

  pfout << "L2 error velocity/pressure/divu: " << velocity_error / velocity_norm << " "
        << pressure_error / pressure_norm << " " << divergence_norm << std::endl;
  fout.close();

  DataOut<dim> data_out;

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);

  

  data_out.add_data_vector(dof_handler_u, vec_u_old[0], "velocity");
  DivergencePostprocessor<dim> div_processor;
  data_out.add_data_vector(dof_handler_u, vec_u_old[0], div_processor);
  VectorTools::interpolate(mapping,
                            dof_handler_u,
                            exact_velocity,
                            speed_extrapolated);
  data_out.add_data_vector(dof_handler_u, speed_extrapolated, "velocity_analytical");

  data_out.add_data_vector(dof_handler_p, vec_p, "pressure");
  VectorTools::interpolate(mapping,
                            dof_handler_p,
                            exact_pressure,
                            vec_p_extrapolated);
  VectorTools::subtract_mean_value(vec_p_extrapolated);
  data_out.add_data_vector(dof_handler_p,
                            vec_p_extrapolated,
                            "pressure_analytical");
  Vector<double> mpi_owner(tria.n_active_cells());
  mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  data_out.add_data_vector(mpi_owner, "owner");
  data_out.build_patches(mapping, 0, DataOut<dim>::curved_inner_cells);

  const std::string filenameVisual =
    "solution-L2-NS_Dirichlet-DivU-" + std::to_string(time_step) + ".vtu";
  data_out.write_vtu_in_parallel(filenameVisual, MPI_COMM_WORLD);

}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  do_test<2, double>(5, 5);
}