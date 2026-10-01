
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
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/reference_cell.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "binary_mesh_reader.h"
#include "consistent_splitting_solver.h"
#include "evaluators.h"
#include "preconditioners.h"

using namespace dealii;

const bool use_neumann_boundary                      = true;
const bool use_skew_symmetric_convective_formulation = false;
const bool use_divergence_formulation                = false;
const bool use_leray_projection                      = true;

const bool use_mixed_mesh      = false;
const bool use_hyper_cube_mesh = false;
const bool use_simplex_mesh    = false;
const bool use_grid_in         = true;

const bool use_preconditioning_pressure = true;
// const bool use_amg                       = false;
const bool use_hmg = false;
const bool use_pmg = true;
const bool use_cmg = true;
// const bool use_pointjacobi_pressure      = false;
const bool use_amg_as_coarse_grid_solver = false;

const bool use_preconditioning_velocity = true;
// const bool use_velocity_point_jacobi         = false;
// const bool use_inverse_mass_velocity         = true;
// const bool use_mg_velocity                   = false;
// const bool use_cmg_vel                       = false;
// const bool use_pmg_vel                       = false;
// const bool use_hmg_vel                       = false;
// const bool use_amg_as_coarse_grid_solver_vel = false;

const double penalty_divergence = 1.0;
const double penalty_continuity = 1.0;

const double viscosity = 0.25 / (1.69 * 1e5);
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
  value(const dealii::Point<dim> &p,
        const unsigned int        component = 0) const final
  {
    const double t      = this->get_time();
    double       result = 0.0;
    if (component == 0)
      if (std::abs(p[0] + 8) < 1e-12 || std::abs(p[0] - 12) < 1e-12 ||
          std::abs(p[1] + 0.05) < 1e-8 || std::abs(p[1]) < 1e-8 ||
          std::abs(p[2] + 0.025) < 1e-12 || std::abs(p[2] - 2.975) < 1e-12)
        result = std::max(1.0, t) * 1.0;

    return result;
  }

  dealii::Tensor<1, dim, double>
  gradient(const dealii::Point<dim> &, const unsigned int) const final
  {
    return dealii::Tensor<1, dim, double>();
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
  value(const dealii::Point<dim> &,
        const unsigned int /*component*/) const final
  {
    return 0.0;
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
  value(const dealii::Point<dim> &p,
        const unsigned int        component = 0) const final
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
        const unsigned int)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  const double L        = 1.;
  unsigned int mg_level = 0;

  const auto serial_grid_generator =
    [&n_refinements, L, &mg_level](
      dealii::Triangulation<dim, dim> &tria_serial) {
      // set up triangulation
      if (use_mixed_mesh)
        {
          std::vector<Point<dim>>    vertices;
          std::vector<CellData<dim>> cells;
          vertices.emplace_back(0.0, 0.0, 0.0);  // 0
          vertices.emplace_back(1.0, 0.0, 0.0);  // 1
          vertices.emplace_back(0.0, 1.0, 0.0);  // 2
          vertices.emplace_back(1.0, 1.0, 0.0);  // 3
          vertices.emplace_back(0.0, 0.0, 1.0);  // 4
          vertices.emplace_back(1.0, 0.0, 1.0);  // 5
          vertices.emplace_back(0.0, 1.0, 1.0);  // 6
          vertices.emplace_back(1.0, 1.0, 1.0);  // 7
          vertices.emplace_back(2, 0.5, 0.25);   // 8
          vertices.emplace_back(2, 0.5, 0.75);   // 9
          vertices.emplace_back(-1.0, 0.5, 0.5); // 10
          vertices.emplace_back(-1.0, 0.5, 1);   // 11
          vertices.emplace_back(-1.0, 0.5, 0);   // 12
          vertices.emplace_back(-1.0, 0.0, 1.0); // 13
          vertices.emplace_back(-1.0, 1.0, 1.0); // 14
          vertices.emplace_back(-1.0, 0.0, 0.0); // 15
          vertices.emplace_back(-1.0, 1.0, 0.0); // 16
          vertices.emplace_back(-1.0, 1.0, 0.5); // 17
          vertices.emplace_back(-1.0, 0.0, 0.5); // 18
          vertices.emplace_back(2, 0.5, 0.0);    // 19
          vertices.emplace_back(2, 0.5, 1.0);    // 20
          vertices.emplace_back(2, 0.0, 0.0);    // 21
          vertices.emplace_back(2, 0.0, 1.0);    // 22
          vertices.emplace_back(2, 1.0, 0.0);    // 23
          vertices.emplace_back(2, 1.0, 1.0);    // 24
          for (auto &p : vertices)
            {
              const double x = p[0];
              p[0]           = (x + 1.0) / 3.0;
            }
          {
            CellData<dim> hex;
            hex.vertices = {0, 1, 2, 3, 4, 5, 6, 7};
            cells.push_back(hex);
          }
          {
            CellData<dim> wedge;
            wedge.vertices = {1, 8, 3, 5, 9, 7};
            cells.push_back(wedge);
          }
          {
            CellData<dim> pyramid;
            pyramid.vertices = {0, 4, 2, 6, 10};
            cells.push_back(pyramid);
          }
          {
            CellData<dim> tet;
            tet.vertices = {4, 6, 10, 11};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {0, 2, 12, 10};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 11, 13, 4};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 11, 6, 14};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 12, 0, 15};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 12, 16, 2};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {17, 10, 6, 14};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 17, 6, 2};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 17, 2, 16};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 18, 15, 0};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 18, 0, 4};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {10, 18, 4, 13};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {5, 9, 7, 20};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {1, 3, 8, 19};
            cells.push_back(tet);
          }
          {
            CellData<dim> wedge;
            wedge.vertices = {8, 1, 21, 9, 5, 22};
            cells.push_back(wedge);
          }
          {
            CellData<dim> tet;
            tet.vertices = {5, 20, 22, 9};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {1, 21, 19, 8};
            cells.push_back(tet);
          }
          {
            CellData<dim> wedge;
            wedge.vertices = {3, 8, 23, 7, 9, 24};
            cells.push_back(wedge);
          }
          {
            CellData<dim> tet;
            tet.vertices = {20, 9, 7, 24};
            cells.push_back(tet);
          }
          {
            CellData<dim> tet;
            tet.vertices = {19, 3, 8, 23};
            cells.push_back(tet);
          }
          tria_serial.create_triangulation(vertices, cells, SubCellData());
        }
      else if (use_hyper_cube_mesh)
        {
          GridGenerator::hyper_cube(tria_serial, -L / 2., L / 2.);
        }
      else if (use_simplex_mesh)
        {
          GridGenerator::subdivided_hyper_cube_with_simplices(tria_serial,
                                                              1,
                                                              -L / 2.,
                                                              L / 2.);
        }
      else if (use_grid_in)
        {
          create_mesh_from_file(tria_serial, "reordered_mesh.bin", true);
        }
      else
        DEAL_II_NOT_IMPLEMENTED();

      if (use_neumann_boundary && use_grid_in)
        {
          for (auto &cell : tria_serial.active_cell_iterators())
            for (const auto f : cell->face_indices())
              if (cell->face(f)->at_boundary())
                if (std::abs(cell->face(f)->center()[0] - 12) < 1e-12)
                  cell->face(f)->set_boundary_id(1);
        }

      if (n_refinements - mg_level > 0)
        tria_serial.refine_global(n_refinements - mg_level);
    };
  const auto serial_grid_partitioner =
    [&](dealii::Triangulation<dim, dim> &tria_serial,
        const MPI_Comm                   comm,
        const unsigned int) {
      std::vector<unsigned int> weights(tria_serial.n_active_cells(), 1);
      for (const auto &cell : tria_serial.active_cell_iterators())
        if (cell->reference_cell().is_hyper_cube())
          weights[cell->active_cell_index()] = 1;
        else if (cell->reference_cell().is_simplex())
          weights[cell->active_cell_index()] = 10;
        else if (cell->reference_cell() == ReferenceCells::Pyramid)
          weights[cell->active_cell_index()] = 12;
        else if (cell->reference_cell() == ReferenceCells::Wedge)
          weights[cell->active_cell_index()] = 8;

      dealii::GridTools::partition_triangulation(
        dealii::Utilities::MPI::n_mpi_processes(comm), weights, tria_serial);
    };

  const unsigned int group_size = 32;

  parallel::fullydistributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  typename dealii::TriangulationDescription::Settings
    triangulation_description_setting =
      dealii::TriangulationDescription::default_setting;
  const auto description = dealii::TriangulationDescription::Utilities::
    create_description_from_triangulation_in_groups<dim, dim>(
      serial_grid_generator,
      serial_grid_partitioner,
      tria.get_mpi_communicator(),
      group_size,
      dealii::Triangulation<dim>::none,
      triangulation_description_setting);

  tria.create_triangulation(description);

  std::vector<std::shared_ptr<const Triangulation<dim>>>
    coarse_grid_triangulations(use_hmg ? n_refinements + 1 : 1);
  coarse_grid_triangulations.back().reset(&tria, [](auto *) {});
  if (use_hmg)
    for (unsigned int l = 0; l < n_refinements; ++l)
      {
        mg_level = n_refinements - l;

        const auto description_level = dealii::TriangulationDescription::
          Utilities::create_description_from_triangulation_in_groups<dim, dim>(
            serial_grid_generator,
            serial_grid_partitioner,
            tria.get_mpi_communicator(),
            group_size,
            dealii::Triangulation<dim>::none,
            triangulation_description_setting);

        const auto level_triangulation =
          std::make_shared<parallel::fullydistributed::Triangulation<dim>>(
            tria.get_mpi_communicator());
        level_triangulation->create_triangulation(description_level);
        coarse_grid_triangulations[l] = level_triangulation;
      }

  hp::MappingCollection<dim> mapping;
  hp::FECollection<dim>      fe_u;
  hp::FECollection<dim>      fe_p;
  hp::QCollection<dim>       quad_error;
  // create collections
  {
    FE_PyramidP<dim> mapping_fe_pyramid(1, true);
    FE_WedgeP<dim>   mapping_fe_wedge(1, true);
    MappingFE<dim>   mapping_pyramid(mapping_fe_pyramid);
    MappingFE<dim>   mapping_wedge(mapping_fe_wedge);
    MappingP1<dim>   mapping_simplex;
    MappingQ1<dim>   mapping_hypercube;

    FE_PyramidDGP<dim> fe_pyramidp_u(fe_degree, false);
    FE_WedgeDGP<dim>   fe_wedgep_u(fe_degree, false);
    FE_SimplexDGP<dim> fe_simplexp_u(fe_degree, false);
    FE_DGQ<dim>        fe_q_u(fe_degree);

    FE_PyramidDGP<dim> fe_pyramidp_p(fe_degree - 1, false);
    FE_WedgeDGP<dim>   fe_wedgep_p(fe_degree - 1, false);
    FE_SimplexDGP<dim> fe_simplexp_p(fe_degree - 1, false);
    FE_DGQ<dim>        fe_q_p(fe_degree - 1);

    for (const auto &ref_cell : tria.get_reference_cells())
      {
        if (ref_cell.is_hyper_cube())
          {
            fe_p.push_back(fe_q_p);
            fe_u.push_back(FESystem<dim>(fe_q_u, dim));
            mapping.push_back(mapping_hypercube);
            quad_error.push_back(QGauss<dim>(fe_degree + 3));
          }
        else if (ref_cell.is_simplex())
          {
            fe_p.push_back(fe_simplexp_p);
            fe_u.push_back(FESystem<dim>(fe_simplexp_u, dim));
            mapping.push_back(mapping_simplex);
            quad_error.push_back(QGaussSimplex<dim>(fe_degree + 3));
          }
        else if (ref_cell == ReferenceCells::Pyramid)
          {
            fe_p.push_back(fe_pyramidp_p);
            fe_u.push_back(FESystem<dim>(fe_pyramidp_u, dim));
            mapping.push_back(mapping_pyramid);
            quad_error.push_back(QGaussPyramid<dim>(fe_degree + 3));
          }
        else if (ref_cell == ReferenceCells::Wedge)
          {
            fe_p.push_back(fe_wedgep_p);
            fe_u.push_back(FESystem<dim>(fe_wedgep_u, dim));
            mapping.push_back(mapping_wedge);
            quad_error.push_back(QGaussWedge<dim>(fe_degree + 3));
          }
        else
          DEAL_II_NOT_IMPLEMENTED();
      }
  }

  DoFHandler<dim> dof_handler_u(tria);
  for (auto &cell : dof_handler_u.active_cell_iterators())
    for (unsigned int i = 0; i < tria.get_reference_cells().size(); ++i)
      if (cell->reference_cell() == tria.get_reference_cells()[i])
        cell->set_active_fe_index(i);
  dof_handler_u.distribute_dofs(fe_u);

  DoFHandler<dim> dof_handler_p(tria);
  for (auto &cell : dof_handler_p.active_cell_iterators())
    for (unsigned int i = 0; i < tria.get_reference_cells().size(); ++i)
      if (cell->reference_cell() == tria.get_reference_cells()[i])
        cell->set_active_fe_index(i);

  dof_handler_p.distribute_dofs(fe_p);
  pcout << "number of active_cells: " << tria.n_global_active_cells()
        << std::endl;
  pcout << "Solving with ";
  for (unsigned int i = 0; i < fe_u.size(); ++i)
    pcout << fe_u[i].get_name() << " x " << fe_p[i].get_name() << ", ";
  pcout << " elements" << std::endl;
  pcout << "number of degrees of freedom: " << dof_handler_u.n_dofs() << " + "
        << dof_handler_p.n_dofs() << std::endl;

  double h_min = std::numeric_limits<double>::max();
  for (const auto &cell : dof_handler_u.active_cell_iterators())
    h_min = std::min(h_min, cell->minimum_vertex_distance());

  const Number time_step = 5.0 * 1e-5; // std::pow(0.5, n_refinements_time);
  // std::min(5.0 * 1e-5, dealii::Utilities::MPI::min(local_time_step,
  // MPI_COMM_WORLD));
  pcout << "Time step size: " << time_step << std::endl;

  const unsigned int bdf_order   = 3;
  const unsigned int bdf_order_p = 2;

  MomentumOperator<dim, dim, Number> momentum_op;
  // set up operator
  momentum_op.reinit(mapping,
                     dof_handler_u,
                     dof_handler_p,
                     time_step,
                     bdf_order,
                     use_skew_symmetric_convective_formulation,
                     use_divergence_formulation);

  momentum_op.set_viscosity(viscosity);
  momentum_op.set_time(0.0);
  momentum_op.set_body_force_factory(
    [=]() { return std::make_unique<AnalyticalRHS<dim>>(u_x_max, viscosity); });
  momentum_op.set_dirichletBC_pressure_factory([=]() {
    return std::make_unique<AnalyticalSolutionPressure<dim>>(u_x_max,
                                                             viscosity);
  });
  momentum_op.set_DirichletBC_velocity_factory([=]() {
    return std::make_unique<AnalyticalSolutionVelocity<dim>>(u_x_max,
                                                             viscosity);
  });

  LinearAlgebra::distributed::Vector<Number> vec_u, vec_u_deriv, vec_u_rhs,
    vec_p, vec_u_norm, speed_extrapolated, vec_vorticity, vec_p_rhs,
    vec_p_rhs_n, vec_p_norm, vec_div_u;
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
  std::vector<LinearAlgebra::distributed::Vector<double>> vec_p_old(bdf_order);
  for (auto &vec : vec_u_old)
    momentum_op.initialize_dof_vector(vec, dof_no_v);

  for (auto &vec : vec_p_old)
    momentum_op.initialize_dof_vector(vec, dof_no_p);


  InverseMassOperator<dim, Number> inverse_mass;
  inverse_mass.reinit(momentum_op.get_matrix_free(), time_step, 0, 1, false);


  PressureOperator<dim, double> pressure_op;
  pressure_op.reinit(momentum_op.get_matrix_free(),
                     bdf_order,
                     time_step,
                     use_leray_projection,
                     false);
  pressure_op.set_body_force_factory(
    [=]() { return std::make_unique<AnalyticalRHS<dim>>(u_x_max, viscosity); });
  pressure_op.set_viscosity(viscosity);
  pressure_op.set_dirichletBC_pressure_factory([=]() {
    return std::make_unique<AnalyticalSolutionPressure<dim>>(u_x_max,
                                                             viscosity);
  });
  pressure_op.set_DirichletBC_velocity_factory([=]() {
    return std::make_unique<AnalyticalSolutionVelocity<dim>>(u_x_max,
                                                             viscosity);
  });

  std::unique_ptr<MultigridPreconditioner<dim, Number, Number>>
    precondition_hmg;
  if (use_preconditioning_pressure)
    precondition_hmg =
      std::make_unique<MultigridPreconditioner<dim, Number, Number>>(
        pressure_op,
        1, // mapping.get_degree(),
        time_step,
        bdf_order,
        use_hmg,
        use_cmg,
        use_pmg,
        use_amg_as_coarse_grid_solver,
        use_neumann_boundary,
        1,
        coarse_grid_triangulations);


  Number current_time = 0;

  AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
  AnalyticalSolutionPressure<dim> exact_pressure(u_x_max, viscosity);

  exact_velocity.set_time(current_time);
  VectorTools::interpolate(mapping,
                           dof_handler_u,
                           exact_velocity,
                           vec_u_old[0]);
  exact_pressure.set_time(current_time);
  VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p);

  const Number       end_time          = 8.0;
  const unsigned int output_interval   = 1;
  unsigned int       time_step_number  = 0;
  unsigned int       n_performed_steps = 0;

  const bool write_output = true;
  const bool write_vtk    = false;

  while (current_time <= end_time)
    {
      current_time += time_step;
      ++time_step_number;
      ++n_performed_steps;
      momentum_op.set_time(current_time);

      const unsigned int current_bdf_order =
        time_step_number < bdf_order ? time_step_number : bdf_order;
      BDFTimeIntegratorConstants bdf(current_bdf_order);
      momentum_op.set_bdf_order(current_bdf_order);
      pressure_op.set_bdf_order(current_bdf_order);
      BDFTimeIntegratorConstants bdf_p(
        time_step_number < bdf_order_p ? time_step_number : bdf_order_p);

      // Pressure step
      vec_p     = 0.;
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
          vec_p.add(bdf.get_beta(i), vec_p_old[i]);
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

      unsigned int iteration_count;
      if (use_preconditioning_pressure)
        {
          iteration_count =
            precondition_hmg->solve(pressure_op, vec_p, vec_p_rhs);
        }
      else
        {
          ReductionControl control(10000, 1e-12, 1e-6);
          SolverCG<LinearAlgebra::distributed::Vector<double>> solver(control);
          solver.solve(pressure_op, vec_p, vec_p_rhs, PreconditionIdentity());
          iteration_count = control.last_step();
        }
      if (!use_neumann_boundary)
        VectorTools::subtract_mean_value(vec_p);
      if (write_output && (n_performed_steps - 1) % output_interval == 0)
        pcout << "Pressure solver: " << iteration_count << " iterations"
              << std::endl;

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
      SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData
        gmres_data;
      gmres_data.max_basis_size        = 100;
      gmres_data.right_preconditioning = true;
      SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver_mom(
        control_mom, gmres_data);
      // SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver_mom(
      // control_mom);
      vec_u.swap(speed_extrapolated); // = 0.;
      if (use_preconditioning_velocity)
        {
          inverse_mass.set_scaling_factor(time_step / bdf.get_gamma0());
          solver_mom.solve(momentum_op, vec_u, vec_u_rhs, inverse_mass);
        }
      else
        solver_mom.solve(momentum_op, vec_u, vec_u_rhs, PreconditionIdentity());

      if (write_output && (n_performed_steps - 1) % output_interval == 0)
        pcout << "Momentum solver: " << control_mom.last_step() << " iterations"
              << std::endl;

      // exact_velocity.set_time(current_time);
      // VectorTools::interpolate(mapping, dof_handler_u, exact_velocity,
      // vec_u);

      for (unsigned int i = bdf.get_order() - 1; i != 0; --i)
        {
          std::swap(vec_u_old[i], vec_u_old[i - 1]);
          std::swap(vec_p_old[i], vec_p_old[i - 1]);
        }

      vec_u_old[0].swap(vec_u);
      vec_p_old[0].swap(vec_p);

      if (write_output && (n_performed_steps - 1) % output_interval == 0)
        {
          Vector<double> error_per_cell;
          exact_velocity.set_time(current_time);
          exact_pressure.set_time(current_time);

          VectorTools::integrate_difference(mapping,
                                            dof_handler_u,
                                            vec_u_old[0],
                                            exact_velocity,
                                            error_per_cell,
                                            quad_error,
                                            VectorTools::L2_norm);
          const double velocity_error =
            VectorTools::compute_global_error(tria,
                                              error_per_cell,
                                              VectorTools::L2_norm);

          VectorTools::integrate_difference(mapping,
                                            dof_handler_p,
                                            vec_p,
                                            exact_pressure,
                                            error_per_cell,
                                            quad_error,
                                            VectorTools::L2_norm);
          const double pressure_error =
            VectorTools::compute_global_error(tria,
                                              error_per_cell,
                                              VectorTools::L2_norm);


          pcout << "L2 norm velocity/pressure: " << velocity_error << " "
                << pressure_error << std::endl;

          if (write_vtk)
            {
              DataOut<dim> data_out;

              DataOutBase::VtkFlags flags;
              flags.write_higher_order_cells = true;
              data_out.set_flags(flags);

              data_out.add_data_vector(dof_handler_u, vec_u_old[0], "solution");
              data_out.add_data_vector(dof_handler_p, vec_p, "pressure");
              Vector<double> mpi_owner(tria.n_active_cells());
              mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
              data_out.add_data_vector(mpi_owner, "owner");
              data_out.build_patches(mapping, 1, DataOut<dim>::no_curved_cells);

              const std::string filename =
                "solution-L2-" + std::to_string(time_step_number) + ".vtu";
              // "solution-L2-" + std::to_string(n_refinements) + "_p_" +
              // std::to_string(degree) + ".vtu";
              data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
            }
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
                                    quad_error,
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
                                    quad_error,
                                    VectorTools::L2_norm);
  const double pressure_error =
    VectorTools::compute_global_error(tria,
                                      error_per_cell,
                                      VectorTools::L2_norm);

  pcout << "L2 norm velocity/pressure: " << velocity_error << " "
        << pressure_error << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  // for (unsigned int i = 1; i < 7; ++i)
  //  do_test<2, double>(3, i, 14);

  // for (unsigned int i = 1; i < 7; ++i)
  //   do_test<2, double>(5, i, 14);

  // for (unsigned int i = 1; i < 15; ++i)
  //   do_test<2, double>(5, 4, 14 - i);
  do_test<3, double>(4, 0, 0);
}
