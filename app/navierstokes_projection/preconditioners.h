
#include <deal.II/lac/dynamic_sparsity_pattern.h>
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


//#include "consistent_splitting_solver.h"

using namespace dealii;

template <int dim1, int n_components, typename Number>
class MomentumOperator;

template <int dim, typename Number>
class PressureOperator;


template <int dim, typename number>
class InverseMassPreconditioner
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;
  typedef InverseMassPreconditioner<dim, number> This;


  InverseMassPreconditioner() = default;

  void
  reinit(const MatrixFree<dim, number> &matrix_free, number scaling_factor_in, const unsigned int dof_no_v = 0, 
            const unsigned int quad_no_v = 0, const unsigned int quad_no_v_mass = 1)
  {
    scaling_factor    = scaling_factor_in;
    this->matrix_free = &matrix_free;
    this->dof_no_v    = dof_no_v;
    this->quad_no_v   = quad_no_v;
    this->quad_no_v_mass = quad_no_v_mass;
  }

  void
  set_scaling_factor(number scaling_factor_in)
  {
    scaling_factor = scaling_factor_in;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    dst.zero_out_ghost_values();

    matrix_free->cell_loop(
      &This::cell_loop_matrix_free_operator,
      this,
      dst,
      src,
      [&](const unsigned int start_range, const unsigned int end_range) {
        for (unsigned int i = start_range; i < end_range; ++i)
          dst.local_element(i) *= 0.;
      },
      [&](const unsigned int start_range, const unsigned int end_range) {
        for (unsigned int i = start_range; i < end_range; ++i)
          dst.local_element(i) *= scaling_factor;
      },
      dof_no_v);
  }

private:
  void
  cell_loop_matrix_free_operator(
    const dealii::MatrixFree<dim, number> &,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, dim, number> integrator(*matrix_free,
                                                     dof_no_v,
                                                     quad_no_v_mass);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim, number> inverse_mass(
      integrator);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        integrator.reinit(cell);
        integrator.read_dof_values(src, 0);

        inverse_mass.apply(integrator.begin_dof_values(), integrator.begin_dof_values());

        integrator.set_dof_values(dst, 0);
      }
  }

  const MatrixFree<dim, number> *matrix_free;
  number                         scaling_factor;
  unsigned int                  dof_no_v;
  unsigned int                 quad_no_v;
  unsigned int                 quad_no_v_mass;
};



template <typename VectorType>
class MGCoarseAMG : public MGCoarseGridBase<VectorType>
{
private:
public:
  MGCoarseAMG(const TrilinosWrappers::PreconditionAMG &amg, const bool is_singular_in)
  {
    amg_preconditioner = &amg;
    is_singular        = is_singular_in;
  }

  void
  operator()(const unsigned int /*level*/,
             VectorType       &dst,
             const VectorType &src) const final
  {
    if constexpr (std::is_same_v<VectorType,
                                 LinearAlgebra::distributed::Vector<TrilinosScalar>>)
      {
        if (is_singular)
          {
            VectorType r(src);
            dealii::VectorTools::subtract_mean_value(r);
            amg_preconditioner->vmult(dst, r);
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
};

template <int dim, typename number_system, typename number = number_system>
class MultigridPreconditionerVelocity
{
  using VectorType       = LinearAlgebra::distributed::Vector<number>;
  using VectorTypeSystem = LinearAlgebra::distributed::Vector<number_system>;
  using SystemMatrixType = MomentumOperator<dim, dim, number_system>;
  using LevelMatrixType  = MomentumOperator<dim, dim, number>;

  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType =
    PreconditionChebyshev<LevelMatrixType, VectorType, SmootherPreconditionerType>;
  using PreconditionerType =
    PreconditionMG<dim, VectorType, MGTransferGlobalCoarsening<dim, VectorType>>;

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
    this->dof_no_v      = dof_no_v;
    this->dof_no_p      = dof_no_p;
    this->viscosity    = viscosity;

    if (use_hmg_vel)
      coarse_grid_triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          dof_handler.get_triangulation());
    else
      coarse_grid_triangulations.emplace_back(&(dof_handler.get_triangulation()),
                                              [](auto *) {});
    const unsigned int n_h_levels = coarse_grid_triangulations.size() - 1;

    const std::vector<unsigned int> level_degrees =
      use_pmg_vel ?
        MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
          mf.get_dof_handler(dof_no_p).get_fe().degree,
          MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::bisect) :
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
        const FESystem<dim> fe_u(FE_DGQ<dim>(level_degrees[level_degrees.size() - 1] + 1),
                                 dim);

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
        mg_matrices[level].set_body_force_factory(momentum_operator.get_body_force_factory());
        mg_matrices[level].set_DirichletBC_velocity_factory(momentum_operator.get_DirichletBC_velocity_factory());
        mg_matrices[level].set_dirichletBC_pressure_factory(momentum_operator.get_dirichletBC_pressure_factory());
      }

    // init transfer
    for (unsigned int level = minlevel; level < maxlevel; ++level)
      transfers[level + 1].reinit(dof_handlers_u[level + 1], dof_handlers_u[level]);

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
            smoother_data[0].smoothing_range     = 1e-3;
            smoother_data[0].degree              = numbers::invalid_unsigned_int;
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
            smoother_data[0].smoothing_range     = 1e-3;
            smoother_data[0].degree              = numbers::invalid_unsigned_int;
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
    const unsigned int                            min_level = mg_matrices.min_level();
    std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

    const auto       precond_point_jacobi = *smoother_data[min_level].preconditioner;
    ReductionControl coarse_grid_solver_control(10000, 1e-12, 1e-4, false, false);
    SolverGMRES<VectorType> coarse_grid_solver(coarse_grid_solver_control);

    // Coarse grid solver
    if (use_amg_as_coarse_grid_solver_vel)
      {
        mg_matrices[min_level].get_system_matrix(coarse_system_matrix);

        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.elliptic = false;
        amg_data.n_cycles = 2;

        precondition_amg.initialize(coarse_system_matrix, amg_data);
        mg_coarse = std::make_unique<MGCoarseAMG<VectorType>>(precondition_amg, false);
      }
    else
      {
        mg_coarse =
          std::make_unique<MGCoarseGridIterativeSolver<VectorType,
                                                       SolverGMRES<VectorType>,
                                                       LevelMatrixType,
                                                       decltype(precond_point_jacobi)>>(
            coarse_grid_solver, mg_matrices[min_level], precond_point_jacobi);
      }

    // Set up levels and transfers
    mg::Matrix<VectorType> mg_matrix(mg_matrices);
    Multigrid<VectorType>  mg(mg_matrix, *mg_coarse, transfer, mg_smoother, mg_smoother);

    PreconditionerType preconditioner(
      momentum_operator.get_matrix_free().get_dof_handler(dof_no_v), mg, transfer);

    ReductionControl control(10000, 1e-12, 1e-6);
    SolverGMRES<VectorTypeSystem> solver_gmres(control);

    solver_gmres.solve(momentum_operator, vec_u, vec_u_rhs, preconditioner);
    return control.last_step();
  }

private:
  MGLevelObject<LevelMatrixType>                                    mg_matrices;
  MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData>              smoother_data;

  MGLevelObject<DoFHandler<dim>>                     dof_handlers_p;
  MGLevelObject<DoFHandler<dim>>                     dof_handlers_u;
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
  MGTransferGlobalCoarsening<dim, VectorType>        transfer;

  std::vector<std::shared_ptr<const Triangulation<dim>>> coarse_grid_triangulations;

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
  MGCoarseCG(const OperatorType                      &operator_in,
             const TrilinosWrappers::PreconditionAMG &amg,
             const bool                               is_singular_in)
  {
    is_singular        = is_singular_in;
    amg_preconditioner = &amg;
    op                 = &operator_in;
  }

  void
  operator()(const unsigned int /*level*/,
             VectorType       &dst,
             const VectorType &src) const final
  {
    ReductionControl coarse_grid_solver_control(10000, 1e-12, 1e-3, false, false);
    SolverCG<LinearAlgebra::distributed::Vector<TrilinosScalar>> coarse_grid_solver(
      coarse_grid_solver_control);

    if constexpr (std::is_same_v<VectorType,
                                 LinearAlgebra::distributed::Vector<TrilinosScalar>>)
      {
        if (is_singular)
          {
            VectorType r(src);
            dealii::VectorTools::subtract_mean_value(r);
            coarse_grid_solver.solve(*op, dst, r, *amg_preconditioner);
          }
        else
          coarse_grid_solver.solve(*op, dst, src, *amg_preconditioner);
      }
    else
      {
        LinearAlgebra::distributed::Vector<TrilinosScalar> src_;
        LinearAlgebra::distributed::Vector<TrilinosScalar> dst_;

        src_ = src;
        dst_ = dst;

        if (is_singular)
          dealii::VectorTools::subtract_mean_value(src_);
        coarse_grid_solver.solve(*op, dst_, src_, *amg_preconditioner);

        dst = dst_;
      }
  }

private:
  bool                                     is_singular;
  const TrilinosWrappers::PreconditionAMG *amg_preconditioner;
  const OperatorType                      *op;
};


template <int dim, typename number_operator, typename number = number_operator>
class MultigridPreconditioner
{
  using VectorType       = LinearAlgebra::distributed::Vector<number>;
  using VectorTypeSystem = LinearAlgebra::distributed::Vector<number_operator>;
  using SystemMatrixType = PressureOperator<dim, number_operator>;
  using LevelMatrixType  = PressureOperator<dim, number>;

  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType =
    PreconditionChebyshev<LevelMatrixType, VectorType, SmootherPreconditionerType>;
  using PreconditionerType =
    PreconditionMG<dim, VectorType, MGTransferGlobalCoarsening<dim, VectorType>>;

public:
  MultigridPreconditioner(SystemMatrixType  &pressure_operator,
                          const unsigned int mapping_degree,
                          const number       time_step,
                          const unsigned int bdf_order,
                          const bool         use_hmg,
                          const bool         use_cmg,
                          const bool         use_pmg,   
                          const bool         use_amg_as_coarse_grid_solver,
                          const bool         use_Neumann_boundary,  
                          const unsigned int dof_no_p = 1)
  {
    const auto &dof_handler =
      pressure_operator.get_matrix_free().get_dof_handler(dof_no_p);
    
    this->dof_no_p = dof_no_p;
    this->use_neumann_boundary = use_Neumann_boundary;
    this->use_amg_as_coarse_grid_solver = use_amg_as_coarse_grid_solver;

    if (use_hmg)
      coarse_grid_triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          dof_handler.get_triangulation());
    else
      coarse_grid_triangulations.emplace_back(&(dof_handler.get_triangulation()),
                                              [](auto *) {});
    const unsigned int n_h_levels = coarse_grid_triangulations.size() - 1;

    const std::vector<unsigned int> level_degrees =
      use_pmg ?
        MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
          dof_handler.get_fe().degree,
          MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::bisect) :
        std::vector<unsigned int>{dof_handler.get_fe().degree};
    const unsigned int n_p_levels = level_degrees.size();

    const unsigned int minlevel = 0;
    const unsigned int maxlevel =
      use_cmg ? n_h_levels + n_p_levels : n_h_levels + n_p_levels - 1;

    dof_handlers.resize(minlevel, maxlevel);
    dof_handlers_u.resize(minlevel, maxlevel);
    mg_matrices.resize(minlevel, maxlevel);
    mg_matrices_mf.resize(minlevel, maxlevel);
    transfers.resize(minlevel, maxlevel);

    // h-MG with linear elements
    for (unsigned int l = 0; l < n_h_levels; ++l)
      {
        auto &dof_handler   = dof_handlers[l];
        auto &dof_handler_u = dof_handlers_u[l];

        if (use_cmg)
          {
            const FE_Q<dim>     fe(level_degrees[0]);
            const FESystem<dim> fe_u(FE_Q<dim>(level_degrees[0] + 1), dim);

            dof_handler.reinit(*coarse_grid_triangulations[l]);
            dof_handler.distribute_dofs(fe);
            dof_handler_u.reinit(*coarse_grid_triangulations[l]);
            dof_handler_u.distribute_dofs(fe_u);
          }
        else
          {
            const FE_DGQ<dim>   fe(level_degrees[0]);
            const FESystem<dim> fe_u(FE_DGQ<dim>(level_degrees[0] + 1), dim);

            dof_handler.reinit(*coarse_grid_triangulations[l]);
            dof_handler.distribute_dofs(fe);
            dof_handler_u.reinit(*coarse_grid_triangulations[l]);
            dof_handler_u.distribute_dofs(fe_u);
          }
      }
    // p-MG
    const unsigned int max_loop_it = use_cmg ? maxlevel : maxlevel + 1;
    for (unsigned int i = 0, l = n_h_levels; l < max_loop_it; ++l, ++i)
      {
        auto &dof_handler   = dof_handlers[l];
        auto &dof_handler_u = dof_handlers_u[l];

        if (use_cmg)
          {
            const FE_Q<dim>     fe(level_degrees[i]);
            const FESystem<dim> fe_u(FE_Q<dim>(level_degrees[i] + 1), dim);

            dof_handler.reinit(*coarse_grid_triangulations[n_h_levels]);
            dof_handler.distribute_dofs(fe);
            dof_handler_u.reinit(*coarse_grid_triangulations[n_h_levels]);
            dof_handler_u.distribute_dofs(fe_u);
          }
        else
          {
            const FE_DGQ<dim>   fe(level_degrees[i]);
            const FESystem<dim> fe_u(FE_DGQ<dim>(level_degrees[i] + 1), dim);

            dof_handler.reinit(*coarse_grid_triangulations[n_h_levels]);
            dof_handler.distribute_dofs(fe);
            dof_handler_u.reinit(*coarse_grid_triangulations[n_h_levels]);
            dof_handler_u.distribute_dofs(fe_u);
          }
      }
    // c-MG
    if (use_cmg)
      {
        const unsigned int l             = maxlevel;
        auto              &dof_handler   = dof_handlers[l];
        auto              &dof_handler_u = dof_handlers_u[l];

        const FE_DGQ<dim>   fe(level_degrees[level_degrees.size() - 1]);
        const FESystem<dim> fe_u(FE_DGQ<dim>(level_degrees[level_degrees.size() - 1] + 1),
                                 dim);


        dof_handler.reinit(*coarse_grid_triangulations[n_h_levels]);
        dof_handler.distribute_dofs(fe);
        dof_handler_u.reinit(*coarse_grid_triangulations[n_h_levels]);
        dof_handler_u.distribute_dofs(fe_u);
      }

    // init levels
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        const unsigned int fe_degree_u     = dof_handlers_u[level].get_fe().degree;
        const unsigned int fe_degree_p     = dof_handlers[level].get_fe().degree;
        Quadrature<1>      quadrature      = QGauss<1>(fe_degree_u + 2);
        Quadrature<1>      quadrature_mass = QGauss<1>(fe_degree_u + 1);
        Quadrature<1>      quadrature_p    = QGauss<1>(fe_degree_p + 1);


        typename MatrixFree<dim, number>::AdditionalData data;
        data.mapping_update_flags = (update_gradients | update_JxW_values |
                                     update_quadrature_points | update_values);
        data.mapping_update_flags_inner_faces =
          (update_gradients | update_JxW_values | update_normal_vectors |
           update_quadrature_points);
        data.mapping_update_flags_boundary_faces =
          (update_gradients | update_JxW_values | update_normal_vectors |
           update_quadrature_points);
        // data.mg_level = level;
        AffineConstraints<double> dummy;
        dummy.close();

        mg_matrices_mf[level].reinit(
          level < n_h_levels ? MappingQGeneric<dim>(1) :
                               MappingQGeneric<dim>(mapping_degree),
          std::vector<const DoFHandler<dim> *>{&dof_handlers_u[level],
                                               &dof_handlers[level]},
          std::vector<const AffineConstraints<double> *>{&dummy, &dummy},
          std::vector<Quadrature<1>>{{quadrature, quadrature_mass, quadrature_p}},
          data);

        mg_matrices[level].reinit(mg_matrices_mf[level], bdf_order, time_step, pressure_operator.get_use_leray_projection());
      }

    // init transfer
    for (unsigned int level = minlevel; level < maxlevel; ++level)
      transfers[level + 1].reinit(dof_handlers[level + 1], dof_handlers[level]);

    transfer = MGTransferGlobalCoarsening<dim, VectorType>(
      transfers, [&](const auto l, auto &vec) {
        mg_matrices[l].get_matrix_free().initialize_dof_vector(vec, dof_no_p);
      });

    // Setup smoother for every level
    smoother_data.resize(minlevel, maxlevel);

    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        if (level > 0)
          {
            smoother_data[level].smoothing_range     = 20.;
            smoother_data[level].degree              = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
          }
        else
          {
            smoother_data[0].smoothing_range     = 1e-3;
            smoother_data[0].degree              = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
          }
        smoother_data[level].preconditioner =
          std::make_shared<SmootherPreconditionerType>();
        mg_matrices[level].compute_inverse_diagonal(
          smoother_data[level].preconditioner->get_vector());
      }

    mg_smoother.initialize(mg_matrices, smoother_data);

    // Setup corase grid AMG
    mg_matrices[minlevel].get_system_matrix(coarse_system_matrix);
    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;

    if (!use_neumann_boundary)
      {
        amg_data.constant_modes =
          DoFTools::extract_constant_modes(dof_handlers[minlevel], ComponentMask());
      }

    if (use_amg_as_coarse_grid_solver)
      {
        amg_data.smoother_sweeps = 1;
        amg_data.n_cycles        = 2;
        amg_data.smoother_type   = "ILU";
      }
    else
      {
        amg_data.elliptic              = true;
        amg_data.higher_order_elements = false;
        amg_data.w_cycle               = false;
        amg_data.aggregation_threshold = 0.2;
        amg_data.smoother_sweeps       = 5;
        amg_data.n_cycles              = 1;
        amg_data.smoother_type         = "Chebyshev";
      }


    precondition_amg.initialize(coarse_system_matrix, amg_data);
  }

  unsigned int
  solve(SystemMatrixType       &pressure_operator,
        VectorTypeSystem       &vec_p,
        const VectorTypeSystem &vec_p_rhs)
  {
    // Coarse grid solver
    std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;
    if (use_amg_as_coarse_grid_solver)
      mg_coarse = std::make_unique<MGCoarseAMG<VectorType>>(precondition_amg,
                                                            !use_neumann_boundary);
    else
      mg_coarse =
        std::make_unique<MGCoarseCG<VectorType, TrilinosWrappers::SparseMatrix>>(
          coarse_system_matrix, precondition_amg, !use_neumann_boundary);


    // Setup levels and transfers
    mg::Matrix<VectorType> mg_matrix(mg_matrices);
    Multigrid<VectorType>  mg(mg_matrix, *mg_coarse, transfer, mg_smoother, mg_smoother);

    PreconditionerType preconditioner(
      pressure_operator.get_matrix_free().get_dof_handler(dof_no_p), mg, transfer);

    ReductionControl control(10000, 1e-12, 1e-6);
    SolverCG<VectorTypeSystem> solver_cg(control);

    solver_cg.solve(pressure_operator, vec_p, vec_p_rhs, preconditioner);
    return control.last_step();
  }

private:
  MGLevelObject<LevelMatrixType>                                    mg_matrices;
  MGLevelObject<MatrixFree<dim, number>>                            mg_matrices_mf;
  MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData>              smoother_data;

  MGLevelObject<DoFHandler<dim>>                     dof_handlers;
  MGLevelObject<DoFHandler<dim>>                     dof_handlers_u;
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
  MGTransferGlobalCoarsening<dim, VectorType>        transfer;

  std::vector<std::shared_ptr<const Triangulation<dim>>> coarse_grid_triangulations;

  TrilinosWrappers::SparseMatrix    coarse_system_matrix;
  TrilinosWrappers::PreconditionAMG precondition_amg;

  bool use_amg_as_coarse_grid_solver;
  bool use_neumann_boundary;
  unsigned int dof_no_p;
};