
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/reference_cell.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "binary_mesh_reader.h"
#include "consistent_splitting_solver.h"
#include "preconditioners.h"


using namespace dealii;

const bool use_mixed_mesh      = false;
const bool use_hyper_cube_mesh = false;
const bool use_simplex_mesh    = false;
const bool use_grid_in         = true;

const bool use_preconditioning_pressure  = true;
const bool use_hmg                       = true;
const bool use_pmg                       = true;
const bool use_cmg                       = true;
const bool use_amg_as_coarse_grid_solver = false;

const double FREQUENCY = 3.0 * dealii::numbers::PI;

template <int dim>
class AnalyticalSolution : public dealii::Function<dim>
{
public:
  AnalyticalSolution()
    : dealii::Function<dim>(1 /*n_components*/, 0.0)
  {}

  double
  value(const dealii::Point<dim> &p,
        const unsigned int /*component*/) const final
  {
    double result = 1.0;
    for (unsigned int d = 0; d < dim; ++d)
      result *= std::sin(FREQUENCY * p[d]);

    return result;
  }
};

template <int dim>
class AnalyticalRHS : public dealii::Function<dim>
{
public:
  AnalyticalRHS()
    : dealii::Function<dim>(1, 0.0)
  {}

  double
  value(const dealii::Point<dim> &p,
        const unsigned int /* component */) const final
  {
    double result = FREQUENCY * FREQUENCY * dim;
    for (unsigned int d = 0; d < dim; ++d)
      result *= std::sin(FREQUENCY * p[d]);

    return result;
  }
};


struct convergencedata
{
  unsigned int cycle;
  unsigned int cells;
  unsigned int dofs;
  double       L2_error;
  double       L2_error_relativ;
};



template <int dim, typename Number>
convergencedata
do_test(const unsigned int fe_degree, const unsigned int n_refinements)
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

          // std::cout << "write vtk";
          // std::ofstream out("reordered_mesh.vtk");
          // GridOut       grid_out;
          // grid_out.write_vtk(tria_serial, out);
          // std::cout << " ... done" << std::endl;
        }
      else
        DEAL_II_NOT_IMPLEMENTED();

      for (auto &c : tria_serial.active_cell_iterators())
        for (const auto &f : c->face_indices())
          if (c->face(f)->at_boundary())
            c->face(f)->set_boundary_id(1);

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

    FE_PyramidDGP<dim> fe_pyramidp_u(1, false);
    FE_WedgeDGP<dim>   fe_wedgep_u(1, false);
    FE_SimplexDGP<dim> fe_simplexp_u(1, false);
    FE_DGQ<dim>        fe_q_u(1);

    FE_PyramidDGP<dim> fe_pyramidp_p(fe_degree, false);
    FE_WedgeDGP<dim>   fe_wedgep_p(fe_degree, false);
    FE_SimplexDGP<dim> fe_simplexp_p(fe_degree, false);
    FE_DGQ<dim>        fe_q_p(fe_degree);

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
    pcout << fe_p[i].get_name() << ", ";
  pcout << " elements" << std::endl;
  pcout << "number of degrees of freedom: " << dof_handler_p.n_dofs()
        << std::endl;

  MomentumOperator<dim, dim, Number> momentum_op;
  // set up operator
  momentum_op.reinit(
    mapping, dof_handler_u, dof_handler_p, 1.0, 1, false, false);

  LinearAlgebra::distributed::Vector<Number> vec_p, vec_p_rhs, vec_p_norm;
  momentum_op.initialize_dof_vector(vec_p, dof_no_p);
  momentum_op.initialize_dof_vector(vec_p_rhs, dof_no_p);
  momentum_op.initialize_dof_vector(vec_p_norm, dof_no_p);

  PressureOperator<dim, double> pressure_op;
  pressure_op.reinit(momentum_op.get_matrix_free(), 1, 1.0, true, false);
  pressure_op.set_dirichletBC_pressure_factory(
    [=]() { return std::make_unique<AnalyticalSolution<dim>>(); });
  pressure_op.set_analytical_rhs_pressure_factory(
    [=]() { return std::make_unique<AnalyticalRHS<dim>>(); });

  std::unique_ptr<MultigridPreconditioner<dim, Number, Number>>
    precondition_hmg;
  if (use_preconditioning_pressure)
    precondition_hmg =
      std::make_unique<MultigridPreconditioner<dim, Number, Number>>(
        pressure_op,
        1, // mapping.get_degree(),
        1.0,
        1,
        use_hmg,
        use_cmg,
        use_pmg,
        use_amg_as_coarse_grid_solver,
        true,
        1,
        coarse_grid_triangulations);

  vec_p_rhs = 0.;
  pressure_op.compute_analytical_rhs(vec_p_rhs);
  vec_p = 0.;

  unsigned int iteration_count;
  if (use_preconditioning_pressure)
    {
      iteration_count = precondition_hmg->solve(pressure_op, vec_p, vec_p_rhs);
    }
  else
    {
      ReductionControl control(10000, 1e-12, 1e-6);
      SolverCG<LinearAlgebra::distributed::Vector<double>> solver(control);
      solver.solve(pressure_op, vec_p, vec_p_rhs, PreconditionIdentity());
      iteration_count = control.last_step();
    }

  pcout << "Pressure solver: " << iteration_count << " iterations" << std::endl;

  AnalyticalSolution<dim> exact_pressure;

  Vector<double> error_per_cell;
  Vector<double> norm_per_cell;

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

  vec_p_norm = 0.;
  VectorTools::integrate_difference(mapping,
                                    dof_handler_p,
                                    vec_p_norm,
                                    exact_pressure,
                                    norm_per_cell,
                                    quad_error,
                                    VectorTools::L2_norm);
  const double pressure_norm =
    VectorTools::compute_global_error(tria,
                                      norm_per_cell,
                                      VectorTools::L2_norm);

  pcout << "L2 error pressure: " << pressure_error << " ("
        << pressure_error / pressure_norm << ")" << std::endl;
  pcout << std::endl;

  convergencedata data;
  data.cells            = tria.n_global_active_cells();
  data.dofs             = dof_handler_p.n_dofs();
  data.cycle            = n_refinements;
  data.L2_error         = pressure_error;
  data.L2_error_relativ = pressure_error / pressure_norm;

  return data;
}


void
get_convergence_tables(const unsigned int fe_degree)
{
  ConvergenceTable convergence_table;
  for (unsigned int n_refinements = 0; n_refinements < 4; ++n_refinements)
    {
      const auto data = do_test<3, double>(fe_degree, n_refinements);

      convergence_table.add_value("cycle", data.cycle);
      convergence_table.add_value("cells", data.cells);
      convergence_table.add_value("dofs", data.dofs);
      convergence_table.add_value("L2", data.L2_error);
      convergence_table.add_value("L2_relative", data.L2_error_relativ);
    }

  convergence_table.set_precision("L2", 3);
  convergence_table.set_scientific("L2", true);

  convergence_table.set_tex_caption("cells", "\\# cells");
  convergence_table.set_tex_caption("dofs", "\\# dofs");
  convergence_table.set_tex_caption("L2", "$L^2$-error");
  convergence_table.set_tex_caption("L2_relative", "$L^2_{relative}$-error");

  convergence_table.set_tex_format("cells", "r");
  convergence_table.set_tex_format("dofs", "r");

  convergence_table.evaluate_convergence_rates(
    "L2", ConvergenceTable::reduction_rate);
  convergence_table.evaluate_convergence_rates(
    "L2", ConvergenceTable::reduction_rate_log2);
  convergence_table.evaluate_convergence_rates(
    "L2_relative", ConvergenceTable::reduction_rate);
  convergence_table.evaluate_convergence_rates(
    "L2_relative", ConvergenceTable::reduction_rate_log2);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      convergence_table.write_text(std::cout);

      std::string error_filename = "mixed_poisson_mg_p";
      error_filename += std::to_string(fe_degree);
      error_filename += ".tex";
      std::ofstream error_table_file(error_filename);

      convergence_table.write_tex(error_table_file);
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  for (unsigned int degree = 1; degree < 4; ++degree)
    get_convergence_tables(degree);
}