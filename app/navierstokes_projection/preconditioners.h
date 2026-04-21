
#pragma once

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

#include "consistent_splitting_solver.h"


using namespace dealii;


template <int dim, typename number>
class InverseMassPreconditioner
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;
  typedef InverseMassPreconditioner<dim, number> This;


  InverseMassPreconditioner() = default;

  void
  reinit(const MatrixFree<dim, number> &matrix_free,
         number                         scaling_factor_in,
         const unsigned int             dof_no_v       = 0,
         const unsigned int             quad_no_v      = 0,
         const unsigned int             quad_no_v_mass = 1)
  {
    scaling_factor       = scaling_factor_in;
    this->matrix_free    = &matrix_free;
    this->dof_no_v       = dof_no_v;
    this->quad_no_v      = quad_no_v;
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

    matrix_free->cell_loop(&This::cell_loop_matrix_free_operator, this, dst, src);
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
        for (unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] *= scaling_factor;

        integrator.set_dof_values(dst, 0);
      }
  }

  const MatrixFree<dim, number> *matrix_free;
  number                         scaling_factor;
  unsigned int                   dof_no_v;
  unsigned int                   quad_no_v;
  unsigned int                   quad_no_v_mass;
};

template <int dim, typename number>
class MassOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;
  typedef MassOperator<dim, number> This;

  MassOperator() = default;

  void
  reinit(const MatrixFree<dim, number> &matrix_free,
         const unsigned int             dof_no_v       = 0,
         const unsigned int             quad_no_v      = 0,
         const unsigned int             quad_no_v_mass = 1)
  {
    this->matrix_free    = &matrix_free;
    this->dof_no_v       = dof_no_v;
    this->quad_no_v      = quad_no_v;
    this->quad_no_v_mass = quad_no_v_mass;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    dst = 0;
    dst.zero_out_ghost_values();
    // std::cout<<"Mass operator vmult\n";
    // std::cout<<"src norm: "<<src.l2_norm()<<"\n";
    matrix_free->cell_loop(&This::do_cell_integral_range, this, dst, src, true);
  }

  void
  compute_inverse_diagonal(VectorType &diagonal_vector) const
  {
    this->matrix_free->initialize_dof_vector(diagonal_vector, dof_no_v);

    MatrixFreeTools::compute_diagonal<dim, -1, 0, dim, number, VectorizedArray<number>>(
      *matrix_free,
      diagonal_vector,
      [&](auto &phi) { do_cell_integral_local(phi); },
      dof_no_v,
      quad_no_v_mass);

    for (unsigned int i = 0; i < diagonal_vector.locally_owned_size(); ++i)
      {
        if (std::abs(diagonal_vector.local_element(i)) > 1.0e-10)
          diagonal_vector.local_element(i) = 1.0 / diagonal_vector.local_element(i);
        else
          diagonal_vector.local_element(i) = 1.0;
      }
  }

private:
  const MatrixFree<dim, number> *matrix_free;

  unsigned int dof_no_v;
  unsigned int quad_no_v;
  unsigned int quad_no_v_mass;

  void
  do_cell_integral_range(const MatrixFree<dim, number>               &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim, -1, 0, dim, number> integrator(matrix_free,
                                                     dof_no_v,
                                                     quad_no_v_mass);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);
        integrator.read_dof_values(src);
        do_cell_integral_local(integrator);
        integrator.distribute_local_to_global(dst);
      }
  }

  void
  do_cell_integral_local(FEEvaluation<dim, -1, 0, dim, number> &integrator) const
  {
    integrator.evaluate(EvaluationFlags::values);
    // loop over quadrature points and compute the local volume flux
    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_value(integrator.get_value(q), q);

    // multiply by nabla v^h(x) and sum
    integrator.integrate(EvaluationFlags::values);
  }
};



template <typename Number>
void
make_zero_mean(const std::vector<unsigned int>                    &constrained_dofs,
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
          dealii::Utilities::MPI::sum(n_unconstrained_dofs, vec.get_mpi_communicator()));

  // set constrained entries to zero again, this should now have zero mean
  for (const unsigned int index : constrained_dofs)
    vec.local_element(index) = 0.;
}



template <int dim, typename number>
class InverseMassPreconditionerRT
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;
  typedef InverseMassPreconditionerRT<dim, number> This;


  InverseMassPreconditionerRT() = default;

  void
  reinit(const MatrixFree<dim, number> &matrix_free,
         number                         scaling_factor_in,
         const unsigned int             dof_no_v       = 0,
         const unsigned int             quad_no_v      = 0,
         const unsigned int             quad_no_v_mass = 1)
  {
    scaling_factor       = scaling_factor_in;
    this->matrix_free    = &matrix_free;
    this->dof_no_v       = dof_no_v;
    this->quad_no_v      = quad_no_v;
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
    // std::cout<<"scaling factor: "<<scaling_factor<<"\n";
    dst = 0;
    dst.zero_out_ghost_values();
    MassOperator<dim, number> mass_operator;
    mass_operator.reinit(*matrix_free, dof_no_v, quad_no_v, quad_no_v_mass);
    /*std::cout<<"src norm: \n";
    std::cout<<src.l2_norm()<<"\n\n";*/
    AssertThrow(std::isfinite(src.l2_norm()), ExcMessage("src contains NaN or Inf"));
    // ReductionControl control_massInv(100000, 1e-12, 1e-9, false, false);
    DiagonalMatrix<LinearAlgebra::distributed::Vector<number>> preconditioner_mass;
    mass_operator.compute_inverse_diagonal(preconditioner_mass.get_vector());
    SolverControl control_massInv(100000, 1e-12 * src.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> solver_massInv(control_massInv);
    solver_massInv.solve(mass_operator, dst, src, preconditioner_mass);
    // std::cout<<"Preconditioner solver iterations: "<<control_massInv.last_step()<<"\n";
    dst *= scaling_factor;
  }

private:
  const MatrixFree<dim, number> *matrix_free;
  number                         scaling_factor;
  unsigned int                   dof_no_v;
  unsigned int                   quad_no_v;
  unsigned int                   quad_no_v_mass;
};



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
    if constexpr (std::is_same_v<VectorType,
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
    this->dof_no_v          = dof_no_v;
    this->dof_no_p          = dof_no_p;
    this->viscosity         = viscosity;

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
        mg_matrices[level].set_body_force_factory(
          momentum_operator.get_body_force_factory());
        mg_matrices[level].set_DirichletBC_velocity_factory(
          momentum_operator.get_DirichletBC_velocity_factory());
        mg_matrices[level].set_dirichletBC_pressure_factory(
          momentum_operator.get_dirichletBC_pressure_factory());
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
        mg_coarse = std::make_unique<MGCoarseAMG<VectorType>>(std::vector<unsigned int>(),
                                                              precondition_amg,
                                                              false);
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

    ReductionControl              control(10000, 1e-12, 1e-6);
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
    ReductionControl     coarse_grid_solver_control(10000, 1e-12, 1e-3, false, false);
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

    this->dof_no_p                      = dof_no_p;
    this->use_neumann_boundary          = use_Neumann_boundary;
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
    mg_matrices.resize(minlevel, maxlevel);
    mg_matrices_mf.resize(minlevel, maxlevel);
    transfers.resize(minlevel, maxlevel);

    // h-MG with linear elements
    for (unsigned int l = 0; l < n_h_levels; ++l)
      {
        auto &dof_handler = dof_handlers[l];

        if (use_cmg)
          {
            const FE_Q<dim> fe(level_degrees[0]);

            dof_handler.reinit(*coarse_grid_triangulations[l]);
            dof_handler.distribute_dofs(fe);
          }
        else
          {
            const FE_DGQ<dim> fe(level_degrees[0]);

            dof_handler.reinit(*coarse_grid_triangulations[l]);
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
            const FE_Q<dim> fe(level_degrees[i]);

            dof_handler.reinit(*coarse_grid_triangulations[n_h_levels]);
            dof_handler.distribute_dofs(fe);
          }
        else
          {
            const FE_DGQ<dim> fe(level_degrees[i]);

            dof_handler.reinit(*coarse_grid_triangulations[n_h_levels]);
            dof_handler.distribute_dofs(fe);
          }
      }
    // c-MG
    if (use_cmg)
      {
        const unsigned int l           = maxlevel;
        auto              &dof_handler = dof_handlers[l];

        const FE_DGQ<dim> fe(level_degrees[level_degrees.size() - 1]);

        dof_handler.reinit(*coarse_grid_triangulations[n_h_levels]);
        dof_handler.distribute_dofs(fe);
      }

    std::vector<AffineConstraints<double>> level_constraints(maxlevel + 1);
    // init levels
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        const unsigned int fe_degree_p      = dof_handlers[level].get_fe().degree;
        Quadrature<1>      quadrature_dummy = QGauss<1>(1);
        Quadrature<1>      quadrature_p     = QGauss<1>(fe_degree_p + 1);


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
        AffineConstraints<double> dummy;
        level_constraints[level].reinit(dof_handlers[level].locally_owned_dofs(),
                                        DoFTools::extract_locally_relevant_dofs(
                                          dof_handlers[level]));
        dealii::ndarray<unsigned int, dim, 2> periodic_ids;
        for (unsigned int d = 0; d < dim; ++d)
          for (unsigned int e = 0; e < 2; ++e)
            periodic_ids[d][e] = numbers::invalid_unsigned_int;
        {
          for (const auto &cell : dof_handlers[level].cell_iterators_on_level(0))
            for (unsigned int d = 0; d < dim; ++d)
              if (cell->at_boundary(2 * d) && cell->has_periodic_neighbor(2 * d))
                {
                  periodic_ids[d][0] = cell->face(2 * d)->boundary_id();
                  periodic_ids[d][1] = cell->periodic_neighbor(2 * d)
                                         ->face(cell->periodic_neighbor_face_no(2 * d))
                                         ->boundary_id();
                }
          for (unsigned int d = 0; d < dim; ++d)
            if (periodic_ids[d][0] != numbers::invalid_unsigned_int)
              dealii::DoFTools::make_periodicity_constraints(dof_handlers[level],
                                                             periodic_ids[d][0],
                                                             periodic_ids[d][1],
                                                             d,
                                                             level_constraints[level]);
        }

        VectorTools::interpolate_boundary_values(dof_handlers[level],
                                                 /* neumann_boundary_id */ 1,
                                                 Functions::ZeroFunction<dim>(1),
                                                 level_constraints[level]);

        level_constraints[level].close();

        mg_matrices_mf[level].reinit(
          level < n_h_levels ? MappingQGeneric<dim>(1) :
                               MappingQGeneric<dim>(mapping_degree),
          std::vector<const DoFHandler<dim> *>{&dof_handlers[level],
                                               &dof_handlers[level]},
          std::vector<const AffineConstraints<double> *>{&dummy,
                                                         &level_constraints[level]},
          std::vector<Quadrature<1>>{{quadrature_dummy, quadrature_dummy, quadrature_p}},
          data);

        mg_matrices[level].reinit(mg_matrices_mf[level],
                                  bdf_order,
                                  time_step,
                                  pressure_operator.get_use_leray_projection());
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
        mg_matrices[level].get_matrix_free().initialize_dof_vector(sol, dof_no_p);
        mg_matrices[level].get_matrix_free().initialize_dof_vector(tmp, dof_no_p);
        mg_matrices[level].get_matrix_free().initialize_dof_vector(rhs, dof_no_p);

        dealii::internal::set_initial_guess(rhs);
        make_zero_mean(
          mg_matrices[level].get_matrix_free().get_constrained_dofs(dof_no_p), rhs);
        solver.solve(mg_matrices[level], tmp, rhs, *smoother_data[level].preconditioner);

        smoother_data[level].eig_cg_n_iterations = 0;
        if (eigenvalue_tracker.values.empty())
          smoother_data[level].max_eigenvalue = 1.0;
        else
          smoother_data[level].max_eigenvalue = eigenvalue_tracker.values.back();
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
          DoFTools::extract_constant_modes(dof_handlers[minlevel], ComponentMask());
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
    Multigrid<VectorType>  mg(mg_matrix, *mg_coarse, transfer, mg_smoother, mg_smoother);

    PreconditionerType preconditioner(
      pressure_operator.get_matrix_free().get_dof_handler(dof_no_p), mg, transfer);

    SolverControl              control(10000, 1e-9 * vec_p_rhs.l2_norm());
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
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
  MGTransferGlobalCoarsening<dim, VectorType>        transfer;
  std::unique_ptr<MGCoarseGridBase<VectorType>>      mg_coarse;

  std::vector<std::shared_ptr<const Triangulation<dim>>> coarse_grid_triangulations;

  TrilinosWrappers::SparseMatrix    coarse_system_matrix;
  TrilinosWrappers::PreconditionAMG precondition_amg;

  bool         use_amg_as_coarse_grid_solver;
  bool         use_neumann_boundary;
  unsigned int dof_no_p;
};



namespace BlockJacobi
{
  unsigned int
  extract_real_eigenvalues(LAPACKFullMatrix<double> &A,
                           std::vector<double>      &eigenvalues,
                           FullMatrix<double>       &eigenvectors)
  {
    A.compute_eigenvalues(true, false);
    FullMatrix<std::complex<double>> eig_vectors = A.get_right_eigenvectors();

    eigenvalues.resize(A.n());
    eigenvectors.reinit(A.n(), A.n());
    std::vector<unsigned int> real_eigenvalue_indices;
    unsigned int              j = 0;
    for (unsigned int i = 0; i < A.n();)
      if (i + 1 < A.n() && std::abs(A.eigenvalue(i).imag()) > 1e-12)
        {
          AssertThrow(
            std::abs(A.eigenvalue(i).imag() + A.eigenvalue(i + 1).imag()) < 1e-12 &&
              std::abs(A.eigenvalue(i).real() - A.eigenvalue(i + 1).real()) < 1e-12,
            ExcInternalError("Eigenvalues do not come in complex-conjugate pairs"));
          eigenvalues[j]     = A.eigenvalue(i).real();
          eigenvalues[j + 1] = A.eigenvalue(i).imag();
          for (unsigned int k = 0; k < A.n(); ++k)
            {
              eigenvectors(k, j)     = eig_vectors(k, i).real();
              eigenvectors(k, j + 1) = eig_vectors(k, i).imag();
            }
          j += 2;
          i += 2;
        }
      else
        {
          AssertThrow(std::abs(A.eigenvalue(i).imag()) <= 1e-12, ExcInternalError());
          real_eigenvalue_indices.push_back(i);
          ++i;
        }
    for (unsigned int i : real_eigenvalue_indices)
      {
        eigenvalues[j] = A.eigenvalue(i).real();
        for (unsigned int k = 0; k < A.n(); ++k)
          eigenvectors(k, j) = eig_vectors(k, i).real();
        ++j;
      }
    AssertThrow(j == A.n(), ExcDimensionMismatch(j, A.n()));

    // complex eigenvalues come in complex-conjugate pairs
    return (A.n() - real_eigenvalue_indices.size()) / 2;
  }



  template <int n_components, int dim, int fe_degree, typename Number = double>
  class CellwisePreconditionerFDM
  {
  public:
    static constexpr int n = fe_degree + 1;
    using vcomplex         = std::complex<VectorizedArray<Number>>;

    CellwisePreconditionerFDM()
    {
      for (unsigned int i2 = 0; i2 < 2; ++i2)
        for (unsigned int i1 = 0; i1 < 2; ++i1)
          for (unsigned int i0 = 0; i0 < 2; ++i0)
            for (unsigned int j2 = 0, j = 0; j2 < (dim > 2 ? 2 : 1); ++j2)
              for (unsigned int j1 = 0; j1 < (dim > 1 ? 2 : 1); ++j1)
                for (unsigned int j0 = 0; j0 < 2; ++j0, ++j)
                  offsets[i2][i1][i0][j] = (i2 * j2 * n + i1 * j1) * n + i0 * j0;

      for (unsigned int d = 0; d < dim; ++d)
        previous_blend_factor[d] = -1.0;

      n_complex_eigenvalues = numbers::invalid_unsigned_int;
    }

    void
    reinit(const std::array<FullMatrix<double>, 2>       &eigenvectors,
           const std::array<FullMatrix<double>, 2>       &inverse_eigenvectors,
           const std::array<std::vector<double>, 2>      &eigenvalues,
           const int                                      n_complex_eigenvalues,
           const VectorizedArray<Number>                  inv_jacobian_determinant,
           const Tensor<1, dim, VectorizedArray<Number>> &average_velocity,
           const double                                   inv_dt)
    {
      Tensor<1, dim, VectorizedArray<Number>> blend_factor_eig;
      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
          if (average_velocity[d][v] < 0.0)
            blend_factor_eig[d][v] = 1.0;
          else
            blend_factor_eig[d][v] = 0.0;

      dealii::ndarray<vcomplex, dim, n> tmp_eig;
      for (int i0 = 0; i0 < n_complex_eigenvalues; ++i0)
        {
          const vcomplex eig0(eigenvalues[0][2 * i0], eigenvalues[0][2 * i0 + 1]);
          const vcomplex eig1(eigenvalues[1][2 * i0], eigenvalues[1][2 * i0 + 1]);
          for (unsigned int d = 0; d < dim; ++d)
            {
              tmp_eig[d][i0] = average_velocity[d] * ((1.0 - blend_factor_eig[d]) * eig0 +
                                                      blend_factor_eig[d] * eig1);
            }
        }

      const int n_eigenvalues = n - n_complex_eigenvalues;
      for (int i0 = n_complex_eigenvalues, i = 2 * i0; i < n; ++i0, ++i)
        for (unsigned int d = 0; d < dim; ++d)
          tmp_eig[d][i0] =
            average_velocity[d] * ((1.0 - blend_factor_eig[d]) * eigenvalues[0][i] +
                                   blend_factor_eig[d] * eigenvalues[1][i]);

      for (int i2 = 0, c = 0; i2 < (dim > 2 ? n_eigenvalues : 1); ++i2)
        for (int i1 = 0; i1 < n_eigenvalues; ++i1)
          {
            std::array<vcomplex, 4> diagonal_element_yz;
            if constexpr (dim == 2)
              {
                diagonal_element_yz[0] = tmp_eig[1][i1];
                diagonal_element_yz[1] = conj(tmp_eig[1][i1]);
              }
            else if constexpr (dim == 3)
              {
                diagonal_element_yz[0] = tmp_eig[2][i2] + tmp_eig[1][i1];
                diagonal_element_yz[1] = tmp_eig[2][i2] + conj(tmp_eig[1][i1]);
                diagonal_element_yz[2] = conj(diagonal_element_yz[1]);
                diagonal_element_yz[3] = conj(diagonal_element_yz[0]);
              }
            for (int i0 = 0; i0 < n_eigenvalues; ++i0, ++c)
              {
                const vcomplex val0 =
                  tmp_eig[0][i0] + make_vectorized_array<Number>(inv_dt);
                const vcomplex val1 = conj(val0);
                for (unsigned int d = 0; d < Utilities::pow(2, dim - 1); ++d)
                  {
                    inverse_eigenvalues_for_cell[c][2 * d] =
                      Utilities::fixed_power<dim>(0.5) * inv_jacobian_determinant /
                      (val0 + diagonal_element_yz[d]);
                    inverse_eigenvalues_for_cell[c][2 * d + 1] =
                      Utilities::fixed_power<dim>(0.5) * inv_jacobian_determinant /
                      (val1 + diagonal_element_yz[d]);
                  }
              }
          }
      // std::cout << "eigvals cell: " << std::endl;
      // for (unsigned int i = 0; i < 4; ++i)
      //   std::cout << inverse_eigenvalues_for_cell[0][i] << std::endl;
      // std::cout << "dst" << std::endl;

      if ((blend_factor_eig - previous_blend_factor).norm_square().sum() > 0)
        for (unsigned int d = 0; d < dim; ++d)
          {
            for (unsigned int i = 0; i < n; ++i)
              for (unsigned int j = 0; j < n; ++j)
                {
                  this->eigenvectors[d][j * n + i] =
                    (1.0 - blend_factor_eig[d]) * eigenvectors[0](j, i) +
                    blend_factor_eig[d] * eigenvectors[1](j, i);
                  this->inverse_eigenvectors[d][j * n + i] =
                    (1.0 - blend_factor_eig[d]) * inverse_eigenvectors[0](j, i) +
                    blend_factor_eig[d] * inverse_eigenvectors[1](j, i);
                }
          }
      previous_blend_factor       = blend_factor_eig;
      this->n_complex_eigenvalues = n_complex_eigenvalues;
    }

    void
    vmult(Vector<Number> &dst, const Vector<Number> &src) const
    {
      constexpr unsigned int n_lanes = VectorizedArray<Number>::size();
      AssertDimension(n_lanes * data_array.size(), dst.size() / n_components);
      AssertDimension(n_lanes * data_array.size(), src.size() / n_components);
      apply(reinterpret_cast<const VectorizedArray<Number> *>(src.begin()),
            reinterpret_cast<VectorizedArray<Number> *>(dst.begin()));
    }

    void
    apply(const VectorizedArray<Number> *src, VectorizedArray<Number> *dst) const
    {
      constexpr unsigned int n_dofs = Utilities::pow(n, dim);
      AssertDimension(n_dofs, data_array.size());

      const int n_eigenvalues = n - n_complex_eigenvalues;
      for (unsigned int comp = 0; comp < n_components; ++comp)
        {
          using Eval = internal::EvaluatorTensorProduct<internal::evaluate_general,
                                                        dim,
                                                        n,
                                                        n,
                                                        VectorizedArray<Number>,
                                                        VectorizedArray<Number>>;
          // apply V^{-1} M^{-1}
          Eval::template apply<0, false, false>(inverse_eigenvectors[0].data(),
                                                src + n_dofs * comp,
                                                data_array.data());
          if constexpr (dim > 1)
            Eval::template apply<1, false, false>(inverse_eigenvectors[1].data(),
                                                  data_array.data(),
                                                  data_array.data());
          if constexpr (dim > 2)
            Eval::template apply<2, false, false>(inverse_eigenvectors[2].data(),
                                                  data_array.data(),
                                                  data_array.data());

          constexpr int n_pairs = Utilities::pow(2, dim);

          for (int i2 = 0, j2 = 0, c = 0; i2 < (dim > 2 ? n_eigenvalues : 1);
               j2 += (dim > 2 && i2 < n_complex_eigenvalues ? 2 : 1), ++i2)
            for (int i1 = 0, j1 = 0; i1 < n_eigenvalues;
                 j1 += (i1 < n_complex_eigenvalues ? 2 : 1), ++i1)
              {
                const auto &offsets_xy =
                  offsets[(dim == 2 || i2 >= n_complex_eigenvalues ? 0 : 1)]
                         [i1 >= n_complex_eigenvalues ? 0 : 1];
                const int i_xy = (j2 * n + j1) * n;
                for (int i0 = 0, j0 = 0; i0 < n_eigenvalues;
                     j0 += (i0 < n_complex_eigenvalues ? 2 : 1), ++i0, ++c)
                  {
                    const unsigned int                       i = i_xy + j0;
                    const std::array<unsigned int, n_pairs> &my_offsets =
                      offsets_xy[i0 >= n_complex_eigenvalues ? 0 : 1];
                    std::array<vcomplex, n_pairs> data_i;
                    for (unsigned int d = 0; d < n_pairs; ++d)
                      {
                        const unsigned int j = my_offsets[d] + i;
                        AssertIndexRange(j, n_dofs);
                        data_i[d] = vcomplex(data_array[j], -data_array[j]);
                      }
                    apply_complex_inverse(data_i, inverse_eigenvalues_for_cell[c]);
                    for (unsigned int d = 0; d < n_pairs; ++d)
                      data_array[my_offsets[d] + i] = data_i[d].real();
                  }
              }

          // apply V
          if constexpr (dim > 1)
            Eval::template apply<1, false, false>(eigenvectors[1].data(),
                                                  data_array.data(),
                                                  data_array.data());
          if constexpr (dim > 2)
            Eval::template apply<2, false, false>(eigenvectors[2].data(),
                                                  data_array.data(),
                                                  data_array.data());
          Eval::template apply<0, false, false>(eigenvectors[0].data(),
                                                data_array.data(),
                                                dst + comp * n_dofs);
        }
    }

  private:
    void
    apply_complex_inverse(
      std::array<vcomplex, Utilities::pow(2, dim)>      &data_i,
      const std::array<vcomplex, Utilities::pow(2, dim)> inverse_eigenvalues_i) const
    {
      for (unsigned int d = 0; d < Utilities::pow(2, dim - 1); ++d)
        {
          const vcomplex tmp0 = data_i[d * 2];
          const vcomplex tmp1 = data_i[d * 2 + 1];
          data_i[d * 2] = vcomplex(tmp0.real() + tmp1.imag(), tmp0.imag() - tmp1.real());
          data_i[d * 2 + 1] =
            vcomplex(tmp0.real() - tmp1.imag(), tmp0.imag() + tmp1.real());
        }
      for (unsigned int d = 0; d < (dim == 3 ? 2 : 1); ++d)
        for (unsigned int e = 0; e < 2; ++e)
          {
            const vcomplex tmp0 = data_i[d * 4 + e];
            const vcomplex tmp1 = data_i[d * 4 + 2 + e];
            data_i[d * 4 + e] =
              vcomplex(tmp0.real() + tmp1.imag(), tmp0.imag() - tmp1.real());
            data_i[d * 4 + 2 + e] =
              vcomplex(tmp0.real() - tmp1.imag(), tmp0.imag() + tmp1.real());
          }
      if constexpr (dim == 3)
        for (unsigned int d = 0; d < Utilities::pow(2, dim - 1); ++d)
          {
            const vcomplex tmp0 = data_i[d];
            const vcomplex tmp1 = data_i[d + 4];
            data_i[d] = vcomplex(tmp0.real() + tmp1.imag(), tmp0.imag() - tmp1.real());
            data_i[d + 4] =
              vcomplex(tmp0.real() - tmp1.imag(), tmp0.imag() + tmp1.real());
          }

      for (unsigned int d = 0; d < Utilities::pow(2, dim); ++d)
        data_i[d] *= inverse_eigenvalues_i[d];

      for (unsigned int d = 0; d < Utilities::pow(2, dim - 1); ++d)
        {
          const vcomplex tmp0 = data_i[d * 2];
          const vcomplex tmp1 = data_i[d * 2 + 1];
          data_i[d * 2] = vcomplex(tmp0.real() + tmp1.real(), tmp0.imag() + tmp1.imag());
          data_i[d * 2 + 1] =
            vcomplex(tmp1.imag() - tmp0.imag(), tmp0.real() - tmp1.real());
        }
      for (unsigned int d = 0; d < (dim == 3 ? 2 : 1); ++d)
        for (unsigned int e = 0; e < 2; ++e)
          {
            const vcomplex tmp0 = data_i[d * 4 + e];
            const vcomplex tmp1 = data_i[d * 4 + 2 + e];
            data_i[d * 4 + e] =
              vcomplex(tmp0.real() + tmp1.real(), tmp0.imag() + tmp1.imag());
            data_i[d * 4 + 2 + e] =
              vcomplex(tmp1.imag() - tmp0.imag(), tmp0.real() - tmp1.real());
          }
      if constexpr (dim == 3)
        for (unsigned int d = 0; d < Utilities::pow(2, dim - 1); ++d)
          {
            const vcomplex tmp0 = data_i[d];
            const vcomplex tmp1 = data_i[d + 4];
            data_i[d] = vcomplex(tmp0.real() + tmp1.real(), tmp0.imag() + tmp1.imag());
            data_i[d + 4] =
              vcomplex(tmp1.imag() - tmp0.imag(), tmp0.real() - tmp1.real());
          }
    }

    dealii::ndarray<VectorizedArray<Number>, dim, n * n> eigenvectors;
    dealii::ndarray<VectorizedArray<Number>, dim, n * n> inverse_eigenvectors;
    dealii::ndarray<vcomplex, Utilities::pow(n, dim), Utilities::pow(2, dim)>
      inverse_eigenvalues_for_cell;
    mutable dealii::ndarray<VectorizedArray<Number>, Utilities::pow(n, dim)> data_array;

    dealii::ndarray<unsigned int, 2, 2, 2, Utilities::pow(2, dim)> offsets;

    Tensor<1, dim, VectorizedArray<Number>> previous_blend_factor;
    int                                     n_complex_eigenvalues;
  };



  template <typename VectorType>
  class MyVectorMemory : public VectorMemory<VectorType>
  {
  public:
    MyVectorMemory()
      : first_unused(vectors.end())
    {}

    virtual VectorType *
    alloc() override
    {
      if (first_unused == vectors.end())
        {
          vectors.push_back(VectorType());
          return &vectors.back();
        }
      else
        {
          VectorType *return_value = &(*first_unused);
          ++first_unused;
          return return_value;
        }
    }

    virtual void
    free(const VectorType *const vector) override
    {
      typename std::list<VectorType>::iterator it = vectors.begin();
      while (&*it != vector)
        ++it;

      Assert(it != first_unused && vector == &*it, ExcInternalError());
      vectors.splice(first_unused, vectors, it);
      --first_unused;
    }

  private:
    std::list<VectorType>                    vectors;
    typename std::list<VectorType>::iterator first_unused;
  };


  template <int dim, int degree, typename Number>
  class CellwiseOperatorMomentum
  {
  public:
    CellwiseOperatorMomentum(const MomentumOperator<dim, dim, Number> &momentum_op,
                             const unsigned int                        cell_batch_index)
      : momentum_op(momentum_op)
      , cell_batch_index(cell_batch_index)
    {}

    void
    vmult(Vector<Number> &dst, const Vector<Number> &src) const
    {
      constexpr unsigned int n_q_points = degree + (degree + 2) / 2;
      momentum_op.template apply_cellwise_operator<n_q_points>(
        cell_batch_index,
        (const VectorizedArray<Number> *)src.begin(),
        (VectorizedArray<Number> *)dst.begin());
    }

  private:
    const MomentumOperator<dim, dim, Number> &momentum_op;
    const unsigned int                        cell_batch_index;
  };



  template <int dim, typename Number>
  class PreconditionerMomentum
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    PreconditionerMomentum(const MomentumOperator<dim, dim, Number> &momentum_op,
                           const unsigned int velocity_dof_handler_in_mf,
                           const unsigned int batched_solver_iterations,
                           const double       diffusivity)
      : matrix_free(momentum_op.get_matrix_free())
      , momentum_op(momentum_op)
      , dof_index_velocity(velocity_dof_handler_in_mf)
      , batched_solver_iterations(batched_solver_iterations)
    {
      const double flux_alpha = 0.5;
      scaled_cell_velocity.resize_fast(matrix_free.n_cell_batches());
      const FiniteElement<dim> &fe =
        matrix_free.get_dof_handler(dof_index_velocity).get_fe();
      const unsigned int n = fe.degree + 1;

      QGauss<1>               gauss_quad(fe.degree + 1);
      QGaussLobatto<1>        lobatto_quad(gauss_quad.size());
      FE_DGQArbitraryNodes<1> fe_1d(/*do_batched_solver ?
                                      static_cast<Quadrature<1> &>(gauss_quad) :*/
                                    static_cast<Quadrature<1> &>(lobatto_quad));
      for (unsigned int c = 0; c < 2; ++c)
        {
          LAPACKFullMatrix<double> deriv_matrix(n, n);
          LAPACKFullMatrix<double> mass_matrix(n, n);
          const double             sign_advection = (c == 0) ? 1.0 : -1.0;

          for (unsigned int q = 0; q < n; ++q)
            {
              for (unsigned int i = 0; i < n; ++i)
                for (unsigned int j = 0; j < n; ++j)
                  deriv_matrix(i, j) += (sign_advection * diffusivity *
                                           fe_1d.shape_grad(i, gauss_quad.point(q)) *
                                           fe_1d.shape_grad(j, gauss_quad.point(q)) -
                                         fe_1d.shape_grad(i, gauss_quad.point(q))[0] *
                                           fe_1d.shape_value(j, gauss_quad.point(q))) *
                                        gauss_quad.weight(q);
              for (unsigned int i = 0; i < n; ++i)
                for (unsigned int j = 0; j < n; ++j)
                  mass_matrix(i, j) += fe_1d.shape_value(i, gauss_quad.point(q)) *
                                       fe_1d.shape_value(j, gauss_quad.point(q)) *
                                       gauss_quad.weight(q);
            }
          const double sigma = (fe.degree + 1) * (fe.degree + 1);
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              deriv_matrix(i, j) +=
                (-fe_1d.shape_value(i, Point<1>()) * fe_1d.shape_value(j, Point<1>()) *
                   (0.5 - flux_alpha * 0.5 * sign_advection) +
                 (0.5 * fe_1d.shape_value(i, Point<1>()) *
                    fe_1d.shape_grad(j, Point<1>())[0] +
                  0.5 * fe_1d.shape_value(j, Point<1>()) *
                    fe_1d.shape_grad(i, Point<1>())[0] +
                  fe_1d.shape_value(i, Point<1>()) * fe_1d.shape_value(j, Point<1>()) *
                    sigma) *
                   diffusivity * sign_advection) +
                (fe_1d.shape_value(i, Point<1>(1.0)) *
                   fe_1d.shape_value(j, Point<1>(1.0)) *
                   (0.5 + flux_alpha * 0.5 * sign_advection) -
                 (0.5 * fe_1d.shape_value(i, Point<1>(1.0)) *
                    fe_1d.shape_grad(j, Point<1>(1.0))[0] +
                  0.5 * fe_1d.shape_value(j, Point<1>(1.0)) *
                    fe_1d.shape_grad(i, Point<1>(1.0))[0] -
                  fe_1d.shape_value(i, Point<1>(1.0)) *
                    fe_1d.shape_value(j, Point<1>(1.0)) * sigma) *
                   diffusivity * sign_advection);

          mass_matrix.set_property(LAPACKSupport::symmetric);
          mass_matrix.compute_cholesky_factorization();
          mass_matrix.solve(deriv_matrix);

          n_complex_eigenvalues =
            extract_real_eigenvalues(deriv_matrix, eigenvalues[c], eigenvectors[c]);

          auto tmp = eigenvectors[c];
          tmp.gauss_jordan();
          mass_matrix.invert();
          inverse_eigenvectors[c].reinit(n, n);
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              {
                double sum = 0;
                for (unsigned int k = 0; k < n; ++k)
                  sum += mass_matrix(k, j) * tmp(i, k);
                inverse_eigenvectors[c](i, j) = sum;
              }
        }
    }

    void
    reinit(const VectorType &velocity, const Number inverse_dt)
    {
      FEEvaluation<dim, -1, 0, dim, Number> evaluator(matrix_free, dof_index_velocity);
      for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        {
          evaluator.reinit(cell);
          evaluator.gather_evaluate(velocity, EvaluationFlags::values);
          Tensor<1, dim, VectorizedArray<Number>> integrated_velocity;
          VectorizedArray<Number>                 volume = 0;
          for (const unsigned int q : evaluator.quadrature_point_indices())
            {
              volume += evaluator.JxW(q);
              integrated_velocity += transpose(evaluator.inverse_jacobian(q)) *
                                     evaluator.get_value(q) * evaluator.JxW(q);
            }
          scaled_cell_velocity[cell] = integrated_velocity / volume;
        }
      this->inverse_dt = inverse_dt;
    }

    void
    vmult(LinearAlgebra::distributed::Vector<Number>       &dst,
          const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      const unsigned int degree =
        matrix_free.get_dof_handler(dof_index_velocity).get_fe().degree;
      if (degree == 1)
        do_vmult<1>(dst, src);
      else if (degree == 2)
        do_vmult<2>(dst, src);
      else if (degree == 3)
        do_vmult<3>(dst, src);
      else if (degree == 4)
        do_vmult<4>(dst, src);
      else if (degree == 5)
        do_vmult<5>(dst, src);
      else if (degree == 6)
        do_vmult<6>(dst, src);
      else if (degree == 7)
        do_vmult<7>(dst, src);
      else if (degree == 8)
        do_vmult<8>(dst, src);
      else if (degree == 9)
        do_vmult<9>(dst, src);
      else
        AssertThrow(false,
                    ExcNotImplemented("Degree " + std::to_string(degree) +
                                      " not instantiated"));
    }

    template <int degree>
    void
    do_vmult(LinearAlgebra::distributed::Vector<Number>       &dst,
             const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      FEEvaluation<dim, degree, degree + 1, dim, Number> eval(matrix_free);
      MyVectorMemory<Vector<Number>>                     memory;
      Vector<Number> local_src(eval.dofs_per_cell * VectorizedArray<Number>::size());
      Vector<Number> local_dst(local_src.size());

      IterationNumberControl control(batched_solver_iterations, 1e-18, false, false);
      typename SolverGMRES<Vector<Number>>::AdditionalData gmres_data;
      gmres_data.right_preconditioning = true;
      gmres_data.orthogonalization_strategy =
        LinearAlgebra::OrthogonalizationStrategy::classical_gram_schmidt;
      gmres_data.max_basis_size = std::min(1u, batched_solver_iterations);
      gmres_data.batched_mode   = true;
      SolverGMRES<Vector<Number>> gmres(control, memory, gmres_data);
      CellwisePreconditionerFDM<dim, dim, degree, Number> cell_fdm;

      for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        {
          eval.reinit(cell);
          eval.read_dof_values(src);
          cell_fdm.reinit(eigenvectors,
                          inverse_eigenvectors,
                          eigenvalues,
                          n_complex_eigenvalues,
                          determinant(eval.inverse_jacobian(0)),
                          scaled_cell_velocity[cell],
                          inverse_dt);

          if (batched_solver_iterations == 0)
            {
              cell_fdm.apply(eval.begin_dof_values(), eval.begin_dof_values());
            }
          else
            {
              CellwiseOperatorMomentum<dim, degree, Number> local_operator(momentum_op,
                                                                           cell);
              for (unsigned int i = 0; i < eval.dofs_per_cell; ++i)
                {
                  ((VectorizedArray<Number> *)local_dst.data())[i] =
                    VectorizedArray<Number>();
                  ((VectorizedArray<Number> *)local_src.data())[i] =
                    eval.begin_dof_values()[i];
                }
              gmres.solve(local_operator, local_dst, local_src, cell_fdm);
              for (unsigned int i = 0; i < eval.dofs_per_cell; ++i)
                eval.begin_dof_values()[i] =
                  ((const VectorizedArray<Number> *)local_dst.data())[i];
            }

          eval.set_dof_values(dst);
        }
    }

  private:
    const MatrixFree<dim, Number>            &matrix_free;
    const MomentumOperator<dim, dim, Number> &momentum_op;
    const unsigned int                        dof_index_velocity;
    const unsigned int                        batched_solver_iterations;
    Number                                    inverse_dt;
    std::array<FullMatrix<double>, 2>         eigenvectors, inverse_eigenvectors;
    std::array<std::vector<double>, 2>        eigenvalues;
    unsigned int                              n_complex_eigenvalues;
    AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> scaled_cell_velocity;
  };

} // namespace BlockJacobi
