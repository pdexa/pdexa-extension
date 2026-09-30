
#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <memory>
#include <vector>

// #include "consistent_splitting_solver.h"

using namespace dealii;

template <int dim1, int n_components, typename Number>
class MomentumOperator;

template <int dim, typename Number>
class PressureOperator;

template <typename Number>
void
make_zero_mean(const std::vector<unsigned int> &constrained_dofs,
               dealii::LinearAlgebra::distributed::Vector<Number> &vec)
{
  // set constrained entries to zero
  for (const unsigned int index : constrained_dofs)
    vec.local_element(index) = 0.;

  // rescale mean value computed among all vector entries to the vector size
  // without constraints
  const unsigned int n_unconstrained_dofs =
    vec.locally_owned_size() - constrained_dofs.size();
  vec.add(-vec.mean_value() * vec.size() /
          dealii::Utilities::MPI::sum(n_unconstrained_dofs,
                                      vec.get_mpi_communicator()));

  // set constrained entries to zero again, this should now have zero mean
  for (const unsigned int index : constrained_dofs)
    vec.local_element(index) = 0.;
}

template <typename VectorType>
class MGCoarseAMG : public MGCoarseGridBase<VectorType>
{
private:
public:
  MGCoarseAMG(const std::vector<unsigned int>         &constrained_dofs,
              const TrilinosWrappers::PreconditionAMG &amg,
              const bool                               is_singular_in)
  {
    this->constrained_dofs = constrained_dofs;
    amg_preconditioner     = &amg;
    is_singular            = is_singular_in;
  }

  void
  operator()(const unsigned int /*level*/,
             VectorType       &dst,
             const VectorType &src) const final
  {
    if constexpr (std::is_same_v<
                    VectorType,
                    LinearAlgebra::distributed::Vector<TrilinosScalar>>)
      {
        if (is_singular)
          {
            VectorType r(src);
            make_zero_mean(constrained_dofs, r);
            amg_preconditioner->vmult(dst, r);
            make_zero_mean(constrained_dofs, dst);
          }
        else
          amg_preconditioner->vmult(dst, src);
      }
    else
      {
        LinearAlgebra::distributed::Vector<TrilinosScalar> src_;
        LinearAlgebra::distributed::Vector<TrilinosScalar> dst_;

        src_ = src;
        dst_ = dst;

        if (is_singular)
          dealii::VectorTools::subtract_mean_value(src_);
        amg_preconditioner->vmult(dst_, src_);

        dst = dst_;
      }
  }

private:
  const TrilinosWrappers::PreconditionAMG *amg_preconditioner;
  bool                                     is_singular;
  std::vector<unsigned int>                constrained_dofs;
};

template <int dim, typename number_system, typename number = number_system>
class MultigridPreconditionerVelocity
{
  using VectorType       = LinearAlgebra::distributed::Vector<number>;
  using VectorTypeSystem = LinearAlgebra::distributed::Vector<number_system>;
  using SystemMatrixType = MomentumOperator<dim, dim, number_system>;
  using LevelMatrixType  = MomentumOperator<dim, dim, number>;

  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                                           VectorType,
                                                           SmootherPreconditionerType>;
  using PreconditionerType =
    PreconditionMG<dim,
                   VectorType,
                   MGTransferGlobalCoarsening<dim, VectorType>>;

public:
  MultigridPreconditionerVelocity(SystemMatrixType  &momentum_operator,
                                  const unsigned int mapping_degree,
                                  const number       viscosity,
                                  const number       time_step,
                                  const unsigned int bdf_order,
                                  const bool         use_hmg_vel,
                                  const bool         use_cmg_vel,
                                  const bool         use_pmg_vel,
                                  const unsigned int dof_no_v = 0,
                                  const unsigned int dof_no_p = 1)
  {
    const auto  mf          = momentum_operator.get_matrix_free();
    const auto &dof_handler = mf.get_dof_handler(dof_no_v);
    this->dof_no_v          = dof_no_v;
    this->dof_no_p          = dof_no_p;
    this->viscosity         = viscosity;

    if (use_hmg_vel)
      coarse_grid_triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          dof_handler.get_triangulation());
    else
      coarse_grid_triangulations.emplace_back(
        &(dof_handler.get_triangulation()), [](auto *) {});
    const unsigned int n_h_levels = coarse_grid_triangulations.size() - 1;

    const std::vector<unsigned int> level_degrees =
      use_pmg_vel ?
        MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
          mf.get_dof_handler(dof_no_p).get_fe().degree,
          MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
            bisect) :
        std::vector<unsigned int>{mf.get_dof_handler(dof_no_p).get_fe().degree};
    const unsigned int n_p_levels = level_degrees.size();

    const unsigned int minlevel = 0;
    const unsigned int maxlevel =
      use_cmg_vel ? n_h_levels + n_p_levels : n_h_levels + n_p_levels - 1;

    dof_handlers_p.resize(minlevel, maxlevel);
    dof_handlers_u.resize(minlevel, maxlevel);
    mg_matrices.resize(minlevel, maxlevel);
    transfers.resize(minlevel, maxlevel);

    // h-MG with linear elements
    for (unsigned int l = 0; l < n_h_levels; ++l)
      {
        auto &dof_handler_p = dof_handlers_p[l];
        auto &dof_handler_u = dof_handlers_u[l];

        if (use_cmg_vel)
          {
            const FE_Q<dim>     fe_p(level_degrees[0]);
            const FESystem<dim> fe_u(FE_Q<dim>(level_degrees[0] + 1), dim);

            dof_handler_p.reinit(*coarse_grid_triangulations[l]);
            dof_handler_p.distribute_dofs(fe_p);
            dof_handler_u.reinit(*coarse_grid_triangulations[l]);
            dof_handler_u.distribute_dofs(fe_u);
          }
        else
          {
            const FE_DGQ<dim>   fe_p(level_degrees[0]);
            const FESystem<dim> fe_u(FE_DGQ<dim>(level_degrees[0] + 1), dim);

            dof_handler_p.reinit(*coarse_grid_triangulations[l]);
            dof_handler_p.distribute_dofs(fe_p);
            dof_handler_u.reinit(*coarse_grid_triangulations[l]);
            dof_handler_u.distribute_dofs(fe_u);
          }
      }
    // p-MG
    const unsigned int max_loop_it = use_cmg_vel ? maxlevel : maxlevel + 1;
    for (unsigned int i = 0, l = n_h_levels; l < max_loop_it; ++l, ++i)
      {
        auto &dof_handler_p = dof_handlers_p[l];
        auto &dof_handler_u = dof_handlers_u[l];

        if (use_cmg_vel)
          {
            const FE_Q<dim>     fe_p(level_degrees[i]);
            const FESystem<dim> fe_u(FE_Q<dim>(level_degrees[i] + 1), dim);

            dof_handler_p.reinit(*coarse_grid_triangulations[n_h_levels]);
            dof_handler_p.distribute_dofs(fe_p);
            dof_handler_u.reinit(*coarse_grid_triangulations[n_h_levels]);
            dof_handler_u.distribute_dofs(fe_u);
          }
        else
          {
            const FE_DGQ<dim>   fe_p(level_degrees[i]);
            const FESystem<dim> fe_u(FE_DGQ<dim>(level_degrees[i] + 1), dim);

            dof_handler_p.reinit(*coarse_grid_triangulations[n_h_levels]);
            dof_handler_p.distribute_dofs(fe_p);
            dof_handler_u.reinit(*coarse_grid_triangulations[n_h_levels]);
            dof_handler_u.distribute_dofs(fe_u);
          }
      }
    // c-MG
    if (use_cmg_vel)
      {
        const unsigned int l             = maxlevel;
        auto              &dof_handler_p = dof_handlers_p[l];
        auto              &dof_handler_u = dof_handlers_u[l];

        const FE_DGQ<dim>   fe_p(level_degrees[level_degrees.size() - 1]);
        const FESystem<dim> fe_u(
          FE_DGQ<dim>(level_degrees[level_degrees.size() - 1] + 1), dim);

        dof_handler_p.reinit(*coarse_grid_triangulations[n_h_levels]);
        dof_handler_p.distribute_dofs(fe_p);
        dof_handler_u.reinit(*coarse_grid_triangulations[n_h_levels]);
        dof_handler_u.distribute_dofs(fe_u);
      }

    // init levels
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        mg_matrices[level].reinit(level < n_h_levels ?
                                    MappingQGeneric<dim>(1) :
                                    MappingQGeneric<dim>(mapping_degree),
                                  dof_handlers_u[level],
                                  dof_handlers_p[level],
                                  time_step,
                                  bdf_order);
        mg_matrices[level].set_viscosity(viscosity);
        mg_matrices[level].set_time(momentum_operator.get_time());
        mg_matrices[level].set_body_force_factory(
          momentum_operator.get_body_force_factory());
        mg_matrices[level].set_DirichletBC_velocity_factory(
          momentum_operator.get_DirichletBC_velocity_factory());
        mg_matrices[level].set_dirichletBC_pressure_factory(
          momentum_operator.get_dirichletBC_pressure_factory());
      }

    // init transfer
    for (unsigned int level = minlevel; level < maxlevel; ++level)
      transfers[level + 1].reinit(dof_handlers_u[level + 1],
                                  dof_handlers_u[level]);

    transfer = MGTransferGlobalCoarsening<dim, VectorType>(
      transfers, [&](const auto l, auto &vec) {
        mg_matrices[l].get_matrix_free().initialize_dof_vector(vec, dof_no_v);
      });

    // Setup smoother for every level
    smoother_data.resize(minlevel, maxlevel);

    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        if (level > 0)
          {
            smoother_data[level].smoothing_range     = 15.;
            smoother_data[level].degree              = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
          }
        else
          {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree          = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
          }

        smoother_data[level].preconditioner =
          std::make_shared<SmootherPreconditionerType>();
        mg_matrices[level].compute_inverse_diagonal(
          smoother_data[level].preconditioner->get_vector());
      }

    mg_smoother.initialize(mg_matrices, smoother_data);
  }

  void
  update(number time, const VectorTypeSystem speed_extrapolated)
  {
    const unsigned int min_level = mg_matrices.min_level();
    const unsigned int max_level = mg_matrices.max_level();

    MGLevelObject<VectorType> speed_on_levels(min_level, max_level);
    transfer.interpolate_to_mg(speed_on_levels, speed_extrapolated);
    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        mg_matrices[level].set_time(time);

        VectorType dummy, dummy2, dummy_p;
        mg_matrices[level].initialize_dof_vector(dummy, dof_no_v);
        mg_matrices[level].initialize_dof_vector(dummy2, dof_no_v);
        mg_matrices[level].initialize_dof_vector(dummy_p, dof_no_p);
        dummy   = 0.;
        dummy2  = 0.;
        dummy_p = 0.;

        mg_matrices[level].set_viscosity(viscosity);

        mg_matrices[level].rhs(dummy, dummy2, speed_on_levels[level], dummy_p);
      }

    // Update smoother for every level
    smoother_data.resize(min_level, max_level);

    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        if (level > 0)
          {
            smoother_data[level].smoothing_range     = 15.;
            smoother_data[level].degree              = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
          }
        else
          {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree          = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
          }

        smoother_data[level].preconditioner =
          std::make_shared<SmootherPreconditionerType>();
        mg_matrices[level].compute_inverse_diagonal(
          smoother_data[level].preconditioner->get_vector());
      }

    mg_smoother.initialize(mg_matrices, smoother_data);
  }

  unsigned int
  solve(SystemMatrixType       &momentum_operator,
        VectorTypeSystem       &vec_u,
        const VectorTypeSystem &vec_u_rhs,
        const bool              use_amg_as_coarse_grid_solver_vel)
  {
    const unsigned int min_level = mg_matrices.min_level();
    std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

    const auto precond_point_jacobi = *smoother_data[min_level].preconditioner;
    ReductionControl coarse_grid_solver_control(
      10000, 1e-12, 1e-4, false, false);
    SolverGMRES<VectorType> coarse_grid_solver(coarse_grid_solver_control);

    // Coarse grid solver
    if (use_amg_as_coarse_grid_solver_vel)
      {
        mg_matrices[min_level].get_system_matrix(coarse_system_matrix);

        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.elliptic = false;
        amg_data.n_cycles = 2;

        precondition_amg.initialize(coarse_system_matrix, amg_data);
        mg_coarse =
          std::make_unique<MGCoarseAMG<VectorType>>(std::vector<unsigned int>(),
                                                    precondition_amg,
                                                    false);
      }
    else
      {
        mg_coarse = std::make_unique<
          MGCoarseGridIterativeSolver<VectorType,
                                      SolverGMRES<VectorType>,
                                      LevelMatrixType,
                                      decltype(precond_point_jacobi)>>(
          coarse_grid_solver, mg_matrices[min_level], precond_point_jacobi);
      }

    // Set up levels and transfers
    mg::Matrix<VectorType> mg_matrix(mg_matrices);
    Multigrid<VectorType>  mg(
      mg_matrix, *mg_coarse, transfer, mg_smoother, mg_smoother);

    PreconditionerType preconditioner(
      momentum_operator.get_matrix_free().get_dof_handler(dof_no_v),
      mg,
      transfer);

    ReductionControl              control(10000, 1e-12, 1e-6);
    SolverGMRES<VectorTypeSystem> solver_gmres(control);

    solver_gmres.solve(momentum_operator, vec_u, vec_u_rhs, preconditioner);
    return control.last_step();
  }

private:
  MGLevelObject<LevelMatrixType>                                    mg_matrices;
  MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;

  MGLevelObject<DoFHandler<dim>>                     dof_handlers_p;
  MGLevelObject<DoFHandler<dim>>                     dof_handlers_u;
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
  MGTransferGlobalCoarsening<dim, VectorType>        transfer;

  std::vector<std::shared_ptr<const Triangulation<dim>>>
    coarse_grid_triangulations;

  TrilinosWrappers::SparseMatrix    coarse_system_matrix;
  TrilinosWrappers::PreconditionAMG precondition_amg;

  unsigned int dof_no_v;
  unsigned int dof_no_p;
  number       viscosity;
};

template <typename VectorType, typename OperatorType>
class MGCoarseCG : public MGCoarseGridBase<VectorType>
{
private:
public:
  MGCoarseCG(const OperatorType                                &operator_in,
             const std::shared_ptr<DiagonalMatrix<VectorType>> &preconditioner,
             const bool                                         is_singular_in)
    : is_singular(is_singular_in)
    , preconditioner(preconditioner)
    , op(operator_in)
  {}

  void
  operator()(const unsigned int /*level*/,
             VectorType       &dst,
             const VectorType &src) const final
  {
    ReductionControl coarse_grid_solver_control(
      10000, 1e-12, 1e-3, false, false);
    SolverCG<VectorType> coarse_grid_solver(coarse_grid_solver_control);

    if (is_singular)
      {
        VectorType r(src);
        dealii::VectorTools::subtract_mean_value(r);
        coarse_grid_solver.solve(op, dst, r, *preconditioner);
      }
    else
      coarse_grid_solver.solve(op, dst, src, *preconditioner);
  }

private:
  const bool                                        is_singular;
  const std::shared_ptr<DiagonalMatrix<VectorType>> preconditioner;
  const OperatorType                               &op;
};

template <int dim, typename number_operator, typename number = number_operator>
class MultigridPreconditioner
{
  using VectorType       = LinearAlgebra::distributed::Vector<number>;
  using VectorTypeSystem = LinearAlgebra::distributed::Vector<number_operator>;
  using SystemMatrixType = PressureOperator<dim, number_operator>;
  using LevelMatrixType  = PressureOperator<dim, number>;

  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                                           VectorType,
                                                           SmootherPreconditionerType>;
  using PreconditionerType =
    PreconditionMG<dim,
                   VectorType,
                   MGTransferGlobalCoarsening<dim, VectorType>>;

public:
  MultigridPreconditioner(
    SystemMatrixType  &pressure_operator,
    const unsigned int mapping_degree,
    const number       time_step,
    const unsigned int bdf_order,
    const bool         use_hmg,
    const bool         use_cmg,
    const bool         use_pmg,
    const bool         use_amg_as_coarse_grid_solver,
    const bool         use_Neumann_boundary,
    const unsigned int dof_no_p,
    const std::vector<std::shared_ptr<const Triangulation<dim>>>
      &coarse_grid_triangulations_in)
  {
    const auto &dof_handler =
      pressure_operator.get_matrix_free().get_dof_handler(dof_no_p);

    this->dof_no_p                      = dof_no_p;
    this->use_neumann_boundary          = use_Neumann_boundary;
    this->use_amg_as_coarse_grid_solver = use_amg_as_coarse_grid_solver;

    if (use_hmg && coarse_grid_triangulations_in.empty())
      coarse_grid_triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          dof_handler.get_triangulation());
    else if (use_hmg && !coarse_grid_triangulations_in.empty())
      coarse_grid_triangulations = coarse_grid_triangulations_in;
    else
      coarse_grid_triangulations.emplace_back(
        &(dof_handler.get_triangulation()), [](auto *) {});
    const unsigned int n_h_levels = coarse_grid_triangulations.size() - 1;

    const std::vector<unsigned int> level_degrees =
      use_pmg ?
        MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
          dof_handler.get_fe().degree,
          MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
            bisect) :
        std::vector<unsigned int>{dof_handler.get_fe().degree};
    const unsigned int n_p_levels = level_degrees.size();

    const unsigned int minlevel = 0;
    const unsigned int maxlevel =
      use_cmg ? n_h_levels + n_p_levels : n_h_levels + n_p_levels - 1;

    dof_handlers.resize(minlevel, maxlevel);
    mg_matrices.resize(minlevel, maxlevel);
    mg_matrices_mf.resize(minlevel, maxlevel);
    transfers.resize(minlevel, maxlevel);

    // h-MG with linear elements
    for (unsigned int l = 0; l < n_h_levels; ++l)
      {
        auto &dof_handler = dof_handlers[l];

        if (use_cmg)
          {
            hp::FECollection<dim> fe;
            for (const auto &ref_cell :
                 coarse_grid_triangulations[l]->get_reference_cells())
              {
                if (ref_cell.is_hyper_cube())
                  {
                    fe.push_back(FE_Q<dim>(level_degrees[0]));
                  }
                else if (ref_cell.is_simplex())
                  {
                    fe.push_back(FE_SimplexP<dim>(level_degrees[0], false));
                  }
                else if (ref_cell == ReferenceCells::Pyramid)
                  {
                    fe.push_back(FE_PyramidP<dim>(level_degrees[0], false));
                  }
                else if (ref_cell == ReferenceCells::Wedge)
                  {
                    fe.push_back(FE_WedgeP<dim>(level_degrees[0], false));
                  }
                else
                  DEAL_II_NOT_IMPLEMENTED();
              }

            dof_handler.reinit(*coarse_grid_triangulations[l]);
            for (auto &cell : dof_handler.active_cell_iterators())
              for (unsigned int i = 0;
                   i <
                   dof_handler.get_triangulation().get_reference_cells().size();
                   ++i)
                if (cell->reference_cell() ==
                    dof_handler.get_triangulation().get_reference_cells()[i])
                  cell->set_active_fe_index(i);
            dof_handler.distribute_dofs(fe);
          }
        else
          {
            hp::FECollection<dim> fe;
            for (const auto &ref_cell :
                 coarse_grid_triangulations[l]->get_reference_cells())
              {
                if (ref_cell.is_hyper_cube())
                  {
                    fe.push_back(FE_DGQ<dim>(level_degrees[0]));
                  }
                else if (ref_cell.is_simplex())
                  {
                    fe.push_back(FE_SimplexDGP<dim>(level_degrees[0], false));
                  }
                else if (ref_cell == ReferenceCells::Pyramid)
                  {
                    fe.push_back(FE_PyramidDGP<dim>(level_degrees[0], false));
                  }
                else if (ref_cell == ReferenceCells::Wedge)
                  {
                    fe.push_back(FE_WedgeDGP<dim>(level_degrees[0], false));
                  }
                else
                  DEAL_II_NOT_IMPLEMENTED();
              }

            dof_handler.reinit(*coarse_grid_triangulations[l]);
            for (auto &cell : dof_handler.active_cell_iterators())
              for (unsigned int i = 0;
                   i <
                   dof_handler.get_triangulation().get_reference_cells().size();
                   ++i)
                if (cell->reference_cell() ==
                    dof_handler.get_triangulation().get_reference_cells()[i])
                  cell->set_active_fe_index(i);
            dof_handler.distribute_dofs(fe);
          }
      }
    // p-MG
    const unsigned int max_loop_it = use_cmg ? maxlevel : maxlevel + 1;
    for (unsigned int i = 0, l = n_h_levels; l < max_loop_it; ++l, ++i)
      {
        auto &dof_handler = dof_handlers[l];

        if (use_cmg)
          {
            hp::FECollection<dim> fe;
            for (const auto &ref_cell :
                 coarse_grid_triangulations[n_h_levels]->get_reference_cells())
              {
                if (ref_cell.is_hyper_cube())
                  {
                    fe.push_back(FE_Q<dim>(level_degrees[i]));
                  }
                else if (ref_cell.is_simplex())
                  {
                    fe.push_back(FE_SimplexP<dim>(level_degrees[i], false));
                  }
                else if (ref_cell == ReferenceCells::Pyramid)
                  {
                    fe.push_back(FE_PyramidP<dim>(level_degrees[i], false));
                  }
                else if (ref_cell == ReferenceCells::Wedge)
                  {
                    fe.push_back(FE_WedgeP<dim>(level_degrees[i], false));
                  }
                else
                  DEAL_II_NOT_IMPLEMENTED();
              }

            dof_handler.reinit(*coarse_grid_triangulations[n_h_levels]);
            for (auto &cell : dof_handler.active_cell_iterators())
              for (unsigned int i = 0;
                   i <
                   dof_handler.get_triangulation().get_reference_cells().size();
                   ++i)
                if (cell->reference_cell() ==
                    dof_handler.get_triangulation().get_reference_cells()[i])
                  cell->set_active_fe_index(i);
            dof_handler.distribute_dofs(fe);
          }
        else
          {
            hp::FECollection<dim> fe;
            for (const auto &ref_cell :
                 coarse_grid_triangulations[n_h_levels]->get_reference_cells())
              {
                if (ref_cell.is_hyper_cube())
                  {
                    fe.push_back(FE_DGQ<dim>(level_degrees[i]));
                  }
                else if (ref_cell.is_simplex())
                  {
                    fe.push_back(FE_SimplexDGP<dim>(level_degrees[i], false));
                  }
                else if (ref_cell == ReferenceCells::Pyramid)
                  {
                    fe.push_back(FE_PyramidDGP<dim>(level_degrees[i], false));
                  }
                else if (ref_cell == ReferenceCells::Wedge)
                  {
                    fe.push_back(FE_WedgeDGP<dim>(level_degrees[i], false));
                  }
                else
                  DEAL_II_NOT_IMPLEMENTED();
              }

            dof_handler.reinit(*coarse_grid_triangulations[n_h_levels]);
            for (auto &cell : dof_handler.active_cell_iterators())
              for (unsigned int i = 0;
                   i <
                   dof_handler.get_triangulation().get_reference_cells().size();
                   ++i)
                if (cell->reference_cell() ==
                    dof_handler.get_triangulation().get_reference_cells()[i])
                  cell->set_active_fe_index(i);
            dof_handler.distribute_dofs(fe);
          }
      }
    // c-MG
    if (use_cmg)
      {
        const unsigned int l           = maxlevel;
        auto              &dof_handler = dof_handlers[l];

        hp::FECollection<dim> fe;
        for (const auto &ref_cell :
             coarse_grid_triangulations[n_h_levels]->get_reference_cells())
          {
            if (ref_cell.is_hyper_cube())
              {
                fe.push_back(
                  FE_DGQ<dim>(level_degrees[level_degrees.size() - 1]));
              }
            else if (ref_cell.is_simplex())
              {
                fe.push_back(
                  FE_SimplexDGP<dim>(level_degrees[level_degrees.size() - 1],
                                     false));
              }
            else if (ref_cell == ReferenceCells::Pyramid)
              {
                fe.push_back(
                  FE_PyramidDGP<dim>(level_degrees[level_degrees.size() - 1],
                                     false));
              }
            else if (ref_cell == ReferenceCells::Wedge)
              {
                fe.push_back(
                  FE_WedgeDGP<dim>(level_degrees[level_degrees.size() - 1],
                                   false));
              }
            else
              DEAL_II_NOT_IMPLEMENTED();
          }

        dof_handler.reinit(*coarse_grid_triangulations[n_h_levels]);
        for (auto &cell : dof_handler.active_cell_iterators())
          for (unsigned int i = 0;
               i < dof_handler.get_triangulation().get_reference_cells().size();
               ++i)
            if (cell->reference_cell() ==
                dof_handler.get_triangulation().get_reference_cells()[i])
              cell->set_active_fe_index(i);
        dof_handler.distribute_dofs(fe);
      }

    std::vector<AffineConstraints<number>> level_constraints(maxlevel + 1);
    // init levels
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        const unsigned int fe_degree_p = dof_handlers[level].get_fe().degree;
        Quadrature<1>      quadrature_dummy = QGauss<1>(1);
        Quadrature<1>      quadrature_p     = QGauss<1>(fe_degree_p + 1);

        FE_PyramidP<dim>           mapping_fe_pyramid(1, true);
        FE_WedgeP<dim>             mapping_fe_wedge(1, true);
        MappingFE<dim>             mapping_pyramid(mapping_fe_pyramid);
        MappingFE<dim>             mapping_wedge(mapping_fe_wedge);
        MappingP1<dim>             mapping_simplex;
        MappingQ1<dim>             mapping_hypercube;
        hp::MappingCollection<dim> mapping;
        hp::QCollection<dim>       quadrature_collection_dummy;
        hp::QCollection<dim>       quadrature_collection_p;
        for (const auto &ref_cell :
             dof_handlers[level].get_triangulation().get_reference_cells())
          {
            if (ref_cell.is_hyper_cube())
              {
                mapping.push_back(mapping_hypercube);
                quadrature_collection_p.push_back(QGauss<dim>(fe_degree_p + 1));
                quadrature_collection_dummy.push_back(QGauss<dim>(1));
              }
            else if (ref_cell.is_simplex())
              {
                quadrature_collection_p.push_back(
                  QGaussSimplex<dim>(fe_degree_p + 1));
                quadrature_collection_dummy.push_back(QGaussSimplex<dim>(1));
                mapping.push_back(mapping_simplex);
              }
            else if (ref_cell == ReferenceCells::Pyramid)
              {
                quadrature_collection_p.push_back(
                  QGaussPyramid<dim>(fe_degree_p + 1));
                quadrature_collection_dummy.push_back(QGaussPyramid<dim>(1));
                mapping.push_back(mapping_pyramid);
              }
            else if (ref_cell == ReferenceCells::Wedge)
              {
                quadrature_collection_p.push_back(
                  QGaussWedge<dim>(fe_degree_p + 1));
                quadrature_collection_dummy.push_back(QGaussWedge<dim>(1));
                mapping.push_back(mapping_wedge);
              }
            else
              DEAL_II_NOT_IMPLEMENTED();
          }

        typename MatrixFree<dim, number>::AdditionalData data;
        data.mapping_update_flags = (update_gradients | update_JxW_values |
                                     update_quadrature_points | update_values);
        if (!use_cmg || level == maxlevel)
          {
            data.mapping_update_flags_inner_faces =
              (update_gradients | update_JxW_values | update_normal_vectors |
               update_quadrature_points);
            data.mapping_update_flags_boundary_faces =
              (update_gradients | update_JxW_values | update_normal_vectors |
               update_quadrature_points);
          }

        // periodicity constraints
        AffineConstraints<number> dummy;
        level_constraints[level].reinit(
          dof_handlers[level].locally_owned_dofs(),
          DoFTools::extract_locally_relevant_dofs(dof_handlers[level]));
        dealii::ndarray<unsigned int, dim, 2> periodic_ids;
        for (unsigned int d = 0; d < dim; ++d)
          for (unsigned int e = 0; e < 2; ++e)
            periodic_ids[d][e] = numbers::invalid_unsigned_int;

        if (dof_handlers[level]
              .get_triangulation()
              .all_reference_cells_are_hyper_cube())
          {
            for (const auto &cell :
                 dof_handlers[level].cell_iterators_on_level(0))
              for (unsigned int d = 0; d < dim; ++d)
                if (cell->at_boundary(2 * d) &&
                    cell->has_periodic_neighbor(2 * d))
                  {
                    periodic_ids[d][0] = cell->face(2 * d)->boundary_id();
                    periodic_ids[d][1] =
                      cell->periodic_neighbor(2 * d)
                        ->face(cell->periodic_neighbor_face_no(2 * d))
                        ->boundary_id();
                  }
            for (unsigned int d = 0; d < dim; ++d)
              if (periodic_ids[d][0] != numbers::invalid_unsigned_int)
                dealii::DoFTools::make_periodicity_constraints(
                  dof_handlers[level],
                  periodic_ids[d][0],
                  periodic_ids[d][1],
                  d,
                  level_constraints[level]);
          }

        if (dof_handlers[level]
              .get_triangulation()
              .all_reference_cells_are_hyper_cube())
          VectorTools::interpolate_boundary_values(
            dof_handlers[level],
            /* neumann_boundary_id */ 1,
            Functions::ZeroFunction<dim, number>(1),
            level_constraints[level]);
        else
          VectorTools::interpolate_boundary_values(
            mapping,
            dof_handlers[level],
            /* neumann_boundary_id */ 1,
            Functions::ZeroFunction<dim, number>(1),
            level_constraints[level]);

        level_constraints[level].close();

        if (dof_handlers[level]
              .get_triangulation()
              .all_reference_cells_are_hyper_cube())
          mg_matrices_mf[level].reinit(
            level < n_h_levels ? MappingQGeneric<dim>(1) :
                                 MappingQGeneric<dim>(mapping_degree),
            std::vector<const DoFHandler<dim> *>{&dof_handlers[level],
                                                 &dof_handlers[level]},
            std::vector<const AffineConstraints<number> *>{
              &dummy, &level_constraints[level]},
            std::vector<Quadrature<1>>{
              {quadrature_dummy, quadrature_dummy, quadrature_p}},
            data);
        else
          {
            std::vector<hp::QCollection<dim>> quad_vector;
            quad_vector.push_back(quadrature_collection_dummy);
            quad_vector.push_back(quadrature_collection_dummy);
            quad_vector.push_back(quadrature_collection_p);

            mg_matrices_mf[level].reinit(
              mapping,
              std::vector<const DoFHandler<dim> *>{&dof_handlers[level],
                                                   &dof_handlers[level]},
              std::vector<const AffineConstraints<number> *>{
                &dummy, &level_constraints[level]},
              quad_vector,
              data);
          }

        mg_matrices[level].reinit(
          mg_matrices_mf[level],
          bdf_order,
          time_step,
          pressure_operator.get_use_leray_projection(),
          pressure_operator.get_use_traction_boundary_condition());
      }

    // init transfer
    for (unsigned int level = minlevel; level < maxlevel; ++level)
      transfers[level + 1].reinit(dof_handlers[level + 1],
                                  dof_handlers[level],
                                  level_constraints[level + 1],
                                  level_constraints[level]);

    transfer = MGTransferGlobalCoarsening<dim, VectorType>(
      transfers, [&](const auto l, auto &vec) {
        mg_matrices[l].get_matrix_free().initialize_dof_vector(vec, dof_no_p);
      });

    // Set up smoother for every level
    smoother_data.resize(minlevel, maxlevel);

    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        smoother_data[level].preconditioner =
          std::make_shared<SmootherPreconditionerType>();
        mg_matrices[level].compute_inverse_diagonal(
          smoother_data[level].preconditioner->get_vector());

        // manually compute the eigenvalue estimate for Chebyshev because we
        // need to be careful with the constrained indices
        dealii::IterationNumberControl control(12, 1e-6, false, false);

        dealii::SolverCG<VectorType>        solver(control);
        dealii::internal::EigenvalueTracker eigenvalue_tracker;
        solver.connect_eigenvalues_slot(
          [&eigenvalue_tracker](const std::vector<double> &eigenvalues) {
            eigenvalue_tracker.slot(eigenvalues);
          });

        VectorType sol, tmp, rhs;
        mg_matrices[level].get_matrix_free().initialize_dof_vector(sol,
                                                                   dof_no_p);
        mg_matrices[level].get_matrix_free().initialize_dof_vector(tmp,
                                                                   dof_no_p);
        mg_matrices[level].get_matrix_free().initialize_dof_vector(rhs,
                                                                   dof_no_p);

        const auto &constrained_dofs =
          mg_matrices[level].get_matrix_free().get_constrained_dofs(dof_no_p);
        const auto partioner =
          mg_matrices[level].get_matrix_free().get_vector_partitioner(dof_no_p);

        std::set<types::global_dof_index> global_dofs;
        for (const auto &d : constrained_dofs)
          global_dofs.insert(partioner->local_to_global(d));
        global_dofs =
          Utilities::MPI::compute_set_union(global_dofs,
                                            partioner->get_mpi_communicator());

        if (mg_matrices[level]
              .get_matrix_free()
              .get_dof_handler(dof_no_p)
              .n_dofs() != global_dofs.size())
          {
            dealii::internal::set_initial_guess(rhs);
            make_zero_mean(
              mg_matrices[level].get_matrix_free().get_constrained_dofs(
                dof_no_p),
              rhs);
            solver.solve(mg_matrices[level],
                         tmp,
                         rhs,
                         *smoother_data[level].preconditioner);
          }
        smoother_data[level].eig_cg_n_iterations = 0;
        if (eigenvalue_tracker.values.empty())
          smoother_data[level].max_eigenvalue = 1.0;
        else
          smoother_data[level].max_eigenvalue =
            eigenvalue_tracker.values.back();
        smoother_data[level].smoothing_range = 20.;
        smoother_data[level].degree          = 5;
      }

    mg_smoother.initialize(mg_matrices, smoother_data);

    // Setup coarse grid AMG
    mg_matrices[minlevel].get_system_matrix(level_constraints[minlevel],
                                            coarse_system_matrix);
    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;

    if (!use_neumann_boundary)
      {
        amg_data.constant_modes =
          DoFTools::extract_constant_modes(dof_handlers[minlevel],
                                           ComponentMask());
      }

    if (use_amg_as_coarse_grid_solver)
      {
        amg_data.smoother_sweeps = 1;
        amg_data.n_cycles        = 2;
        amg_data.smoother_type   = "ILU";
        precondition_amg.initialize(coarse_system_matrix, amg_data);
        mg_coarse = std::make_unique<MGCoarseAMG<VectorType>>(
          mg_matrices[0].get_matrix_free().get_constrained_dofs(dof_no_p),
          precondition_amg,
          !use_neumann_boundary);
      }
    else
      {
        // amg_data.elliptic              = true;
        // amg_data.higher_order_elements = false;
        // amg_data.w_cycle               = false;
        // amg_data.aggregation_threshold = 0.2;
        // amg_data.smoother_sweeps       = 5;
        // amg_data.n_cycles              = 1;
        // amg_data.smoother_type         = "Chebyshev";
        // precondition_amg.initialize(coarse_system_matrix, amg_data);

        mg_coarse = std::make_unique<MGCoarseCG<VectorType, LevelMatrixType>>(
          mg_matrices[minlevel],
          smoother_data[minlevel].preconditioner,
          !use_neumann_boundary);
      }
  }

  unsigned int
  solve(SystemMatrixType       &pressure_operator,
        VectorTypeSystem       &vec_p,
        const VectorTypeSystem &vec_p_rhs)
  {
    // Set up levels and transfers
    mg::Matrix<VectorType> mg_matrix(mg_matrices);
    Multigrid<VectorType>  mg(
      mg_matrix, *mg_coarse, transfer, mg_smoother, mg_smoother);

    PreconditionerType preconditioner(
      pressure_operator.get_matrix_free().get_dof_handler(dof_no_p),
      mg,
      transfer);

    ReductionControl           control(10000, 1e-12, 1e-6);
    SolverCG<VectorTypeSystem> solver_cg(control);

    solver_cg.solve(pressure_operator, vec_p, vec_p_rhs, preconditioner);
    return control.last_step();
  }

private:
  MGLevelObject<LevelMatrixType>         mg_matrices;
  MGLevelObject<MatrixFree<dim, number>> mg_matrices_mf;
  MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;

  MGLevelObject<DoFHandler<dim>>                     dof_handlers;
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
  MGTransferGlobalCoarsening<dim, VectorType>        transfer;
  std::unique_ptr<MGCoarseGridBase<VectorType>>      mg_coarse;

  std::vector<std::shared_ptr<const Triangulation<dim>>>
    coarse_grid_triangulations;

  TrilinosWrappers::SparseMatrix    coarse_system_matrix;
  TrilinosWrappers::PreconditionAMG precondition_amg;

  bool         use_amg_as_coarse_grid_solver;
  bool         use_neumann_boundary;
  unsigned int dof_no_p;
};
