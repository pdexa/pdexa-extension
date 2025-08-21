
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
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

using namespace dealii;


const bool use_extrapolated_velocity                 = false;
const bool use_pressure_convective_upwind_flux       = false;
const bool use_neumann_boundary                      = false;
const bool use_analytical_curl                       = false;
const bool use_skew_symmetric_convective_formulation = true;
const bool use_leray_projection                      = true;

const bool use_amg                       = false;
const bool use_hmg                       = true;
const bool use_pmg                       = true;
const bool use_cmg                       = true;
const bool use_pointjacobi_pressure      = false;
const bool use_amg_as_coarse_grid_solver = true;


const bool use_velocity_point_jacobi         = false;
const bool use_inverse_mass_velocity         = false;
const bool use_mg_velocity                   = true;
const bool use_cmg_vel                       = true;
const bool use_pmg_vel                       = true;
const bool use_hmg_vel                       = true;
const bool use_amg_as_coarse_grid_solver_vel = true;

const double penalty_divergence = 1.0;
const double penalty_continuity = 1.0;


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


template <int dim, typename Number, int n_components = dim>
Tensor<1, n_components, VectorizedArray<Number>>
evaluate_function(const Function<dim>                       &function,
                  const Point<dim, VectorizedArray<Number>> &p_vectorized)
{
  AssertDimension(function.n_components, n_components);
  Tensor<1, n_components, VectorizedArray<Number>> result;
  for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = p_vectorized[d][v];
      for (unsigned int d = 0; d < n_components; ++d)
        result[d][v] = function.value(p, d);
    }
  return result;
}



template <int dim, typename Number>
VectorizedArray<Number>
evaluate_scalar_function(const Function<dim>                       &function,
                         const Point<dim, VectorizedArray<Number>> &p_vectorized)
{
  AssertDimension(function.n_components, 1);
  VectorizedArray<Number> result;
  for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = p_vectorized[d][v];
      result[v] = function.value(p);
    }
  return result;
}


template <int dim, typename number, int n_components = dim>
Tensor<2, n_components, VectorizedArray<number>>
evaluate_tensor_function(const Function<dim>                       &function,
                         const Point<dim, VectorizedArray<number>> &p_vectorized)
{
  Tensor<2, n_components, VectorizedArray<number>> result;
  for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = p_vectorized[d][v];
      for (unsigned int d = 0; d < n_components; ++d)
        {
          auto func_eval = function.gradient(p, d);
          for (unsigned int e = 0; e < dim; ++e)
            result[d][e][v] = func_eval[e];
        }
    }
  return result;
}



template <int dim, typename FEEval>
struct CurlCompute
{
  static typename FEEval::value_type
  compute(const FEEval &fe_eval, const unsigned int q_point)
  {
    return fe_eval.get_curl(q_point);
  }
};

template <typename FEEval>
struct CurlCompute<2, FEEval>
{
  static typename FEEval::value_type
  compute(const FEEval &fe_eval, const unsigned int q_point)
  {
    typename FEEval::gradient_type temp = fe_eval.get_gradient(q_point);
    typename FEEval::value_type    curl;
    curl[0] = temp[0][1];  //   d(phi)/dx2
    curl[1] = -temp[0][0]; // - d(phi)/dx1
    return curl;
  }
};


class BDFTimeIntegratorConstants
{
public:
  BDFTimeIntegratorConstants(const unsigned int current_order)
  {
    alpha.resize(current_order);
    beta.resize(current_order);
    switch (current_order)
      {
        case 1:
          {
            gamma0   = 1.0;
            alpha[0] = 1.0;
            beta[0]  = 1.0;
            break;
          }
        case 2:
          {
            gamma0   = 3.0 / 2.0;
            alpha[0] = 2.0;
            alpha[1] = -0.5;
            beta[0]  = 2.0;
            beta[1]  = -1.0;
            break;
          }
        case 3:
          {
            gamma0   = 11. / 6.;
            alpha[0] = 3.;
            alpha[1] = -1.5;
            alpha[2] = 1. / 3.;
            beta[0]  = 3.0;
            beta[1]  = -3.0;
            beta[2]  = 1.0;
            break;
          }
        case 4:
          {
            gamma0   = 25. / 12.;
            alpha[0] = 4.;
            alpha[1] = -3.;
            alpha[2] = 4. / 3.;
            alpha[3] = -1. / 4.;
            beta[0]  = 4.;
            beta[1]  = -6.;
            beta[2]  = 4.;
            beta[3]  = -1.;
            break;
          }
        default:
          {
            AssertThrow(false,
                        dealii::ExcMessage(
                          "Specified order of BDF scheme not implemented."));
          }
      }
  }

  double
  get_gamma0()
  {
    return gamma0;
  }
  double
  get_alpha(const unsigned int i)
  {
    AssertIndexRange(i, alpha.size());
    return alpha[i];
  }
  double
  get_beta(const unsigned int i)
  {
    AssertIndexRange(i, beta.size());
    return beta[i];
  }
  unsigned int
  get_order()
  {
    return alpha.size();
  }

  std::vector<double> alpha;
  std::vector<double> beta;
  double              gamma0;
};



const unsigned int dof_no_v = 0;
const unsigned int dof_no_p = 1;

const unsigned int quad_no_v      = 0;
const unsigned int quad_no_v_mass = 1;
const unsigned int quad_no_p      = 2;

template <int dim_, int n_components = dim_, typename Number = double>
class MomentumOperator : public Subscriptor
{
public:
  typedef MomentumOperator<dim_, n_components, Number> This;
  using value_type = Number;
  using number     = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  static const int dim = dim_;

  void
  reinit(const Mapping<dim>    &mapping,
         const DoFHandler<dim> &dof_handler_u,
         const DoFHandler<dim> &dof_handler_p,
         const number           time_step_in,
         const unsigned int     bdf_order_in)
  {
    bdf_order = bdf_order_in;
    time_step = time_step_in;
    is_dg     = dof_handler_u.get_fe().n_dofs_per_vertex() == 0;

    fe_degree_u                        = dof_handler_u.get_fe().degree;
    const unsigned int fe_degree_p     = dof_handler_p.get_fe().degree;
    Quadrature<1>      quadrature      = QGauss<1>(fe_degree_u + 2);
    Quadrature<1>      quadrature_mass = QGauss<1>(fe_degree_u + 1);
    Quadrature<1>      quadrature_p    = QGauss<1>(fe_degree_p + 1);


    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points | update_values);
    data.mapping_update_flags_inner_faces =
      (update_gradients | update_JxW_values | update_normal_vectors |
       update_quadrature_points);
    data.mapping_update_flags_boundary_faces =
      (update_gradients | update_JxW_values | update_normal_vectors |
       update_quadrature_points);

    AffineConstraints<double> dummy;
    dummy.close();


    matrix_free.reinit(
      mapping,
      std::vector<const DoFHandler<dim> *>{&dof_handler_u, &dof_handler_p},
      std::vector<const AffineConstraints<double> *>{&dummy, &dummy},
      std::vector<Quadrature<1>>{{quadrature, quadrature_mass, quadrature_p}},
      data);

    FEEvaluation<dim, -1, 0, dim, number> eval_cell(matrix_free, 0, 0);
    speeds_cells.reinit(matrix_free.n_cell_batches(), eval_cell.n_q_points);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_face(matrix_free, true, 0, 0);
    speeds_faces.reinit(matrix_free.n_inner_face_batches() +
                          matrix_free.n_boundary_face_batches(),
                        eval_face.n_q_points);


    penalty_factor =
      1.0 * (dof_handler_u.get_fe().degree + 1) * (dof_handler_u.get_fe().degree);
    {
      unsigned int n_cells =
        matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
      array_penalty_parameter.resize(n_cells);
      penalty_factor_divergence.resize(n_cells);
      penalty_factor_continuity.resize(n_cells);


      const dealii::FiniteElement<dim> &fe = dof_handler_u.get_fe();
      const auto reference_cells           = dof_handler_u.get_fe().reference_cell();

      const auto quadrature =
        reference_cells.template get_gauss_type_quadrature<dim>(fe_degree_u + 1);
      dealii::FEValues<dim> fe_values(mapping, fe, quadrature, dealii::update_JxW_values);

      const auto face_quadrature =
        reference_cells.face_reference_cell(0)
          .template get_gauss_type_quadrature<dim - 1>(fe_degree_u + 1);
      dealii::FEFaceValues<dim> fe_face_values(mapping,
                                               fe,
                                               face_quadrature,
                                               dealii::update_JxW_values);

      for (unsigned int i = 0; i < n_cells; ++i)
        {
          for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(i);
               ++v)
            {
              typename dealii::DoFHandler<dim>::cell_iterator cell =
                matrix_free.get_cell_iterator(i, v);
              fe_values.reinit(cell);

              // calculate cell volume
              number volume = 0;
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                {
                  volume += fe_values.JxW(q);
                }

              // calculate surface area
              number surface_area = 0;
              for (const unsigned int f : cell->face_indices())
                {
                  fe_face_values.reinit(cell, f);
                  const number factor =
                    (cell->at_boundary(f) and not(cell->has_periodic_neighbor(f))) ? 1. :
                                                                                     0.5;
                  for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                    {
                      surface_area += fe_face_values.JxW(q) * factor;
                    }
                }

              array_penalty_parameter[i][v] = surface_area / volume;
            }
        }
    }
  }


  virtual void
  initialize_dof_vector(VectorType &vec, const unsigned int dof_index) const
  {
    matrix_free.initialize_dof_vector(vec, dof_index);
  }


  virtual void
  set_viscosity(number viscosity_in)
  {
    viscosity = viscosity_in;
  }

  virtual void
  set_bdf_order(const unsigned int bdf_order_in)
  {
    bdf_order = bdf_order_in;
  }

  virtual void
  set_time(number time_in)
  {
    time = time_in;
  }

  virtual void
  vmult(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.loop(&MomentumOperator::do_cell_integral_range,
                           &MomentumOperator::do_face_integral_range,
                           &MomentumOperator::do_boundary_integral_range,
                           this,
                           dst,
                           src,
                           true,
                           MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                           MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

  virtual void
  rhs(VectorType       &dst,
      const VectorType &velocity_derivative,
      const VectorType &speed,
      const VectorType &pressure) const
  {
    this->matrix_free.loop(&MomentumOperator::do_rhs_cell_integral_range,
                           &MomentumOperator::do_rhs_face_integral_range,
                           &MomentumOperator::do_rhs_boundary_range,
                           this,
                           dst,
                           std::vector<const VectorType *>{&velocity_derivative,
                                                           &speed,
                                                           &pressure},
                           true,
                           MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                           MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }



  void
  evaluate_vorticity(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &MomentumOperator::local_vorticity_domain, this, dst, src, true);

    FEEvaluation<dim, -1, 0, dim, number> eval_u(matrix_free, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim, number> mass_inv(eval_u);
    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        eval_u.reinit(cell);
        eval_u.read_dof_values(dst);
        mass_inv.apply(eval_u.begin_dof_values(), eval_u.begin_dof_values());
        eval_u.set_dof_values(dst);
      }
  }


  const MatrixFree<dim, number> &
  get_matrix_free() const
  {
    return matrix_free;
  }

  void
  get_system_matrix(TrilinosWrappers::SparseMatrix &system_matrix)
  {
    const auto &dof_handler = matrix_free.get_dof_handler(dof_no_v);

    TrilinosWrappers::SparsityPattern dsp(
      dof_handler.locally_owned_dofs(),
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(dof_handler),
      dof_handler.get_triangulation().get_communicator());

    if (is_dg)
      DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, AffineConstraints<number>());
    else
      DoFTools::make_sparsity_pattern(dof_handler, dsp, AffineConstraints<number>());

    dsp.compress();
    system_matrix.reinit(dsp);
    system_matrix = 0.;

    if (is_dg)
      MatrixFreeTools::compute_matrix<dim, -1, 0, dim, number, VectorizedArray<number>>(
        matrix_free,
        AffineConstraints<number>(),
        system_matrix,
        [&](auto &phi) { do_cell_integral_local(phi); },
        [&](auto &phi_m, auto &phi_p) { do_face_integral_local(phi_m, phi_p); },
        [&](auto &phi) { do_boundary_integral_local(phi); },
        dof_no_v,
        quad_no_v,
        0);
    else
      MatrixFreeTools::compute_matrix<dim, -1, 0, dim, number, VectorizedArray<number>>(
        matrix_free,
        AffineConstraints<number>(),
        system_matrix,
        [&](auto &phi) { do_cell_integral_local(phi); },
        {},
        [&](auto &phi) { do_boundary_integral_local(phi); },
        dof_no_v,
        quad_no_v,
        0);
  }

  void
  compute_inverse_diagonal(VectorType &diagonal_vector) const
  {
    initialize_dof_vector(diagonal_vector, dof_no_v);

    if (is_dg)
      MatrixFreeTools::compute_diagonal<dim, -1, 0, dim, number, VectorizedArray<number>>(
        matrix_free,
        diagonal_vector,
        [&](auto &phi) { do_cell_integral_local(phi); },
        [&](auto &phi_m, auto &phi_p) { do_face_integral_local(phi_m, phi_p); },
        [&](auto &phi) { do_boundary_integral_local(phi); },
        dof_no_v,
        quad_no_v);
    else
      MatrixFreeTools::compute_diagonal<dim, -1, 0, dim, number, VectorizedArray<number>>(
        matrix_free,
        diagonal_vector,
        [&](auto &phi) { do_cell_integral_local(phi); },
        {},
        [&](auto &phi) { do_boundary_integral_local(phi); },
        dof_no_v,
        quad_no_v);

    for (unsigned int i = 0; i < diagonal_vector.locally_owned_size(); ++i)
      {
        if (std::abs(diagonal_vector.local_element(i)) > 1.0e-10)
          diagonal_vector.local_element(i) = 1.0 / diagonal_vector.local_element(i);
        else
          diagonal_vector.local_element(i) = 1.0;
      }
  }


  types::global_dof_index
  m() const
  {
    if (matrix_free.get_mg_level() == numbers::invalid_unsigned_int)
      return matrix_free.get_dof_handler(dof_no_v).n_dofs();
    else
      return matrix_free.get_dof_handler(dof_no_v).n_dofs(matrix_free.get_mg_level());
  }

  void
  Tvmult(VectorType &, const VectorType &) const
  {
    DEAL_II_NOT_IMPLEMENTED();
  }

  number
  el(unsigned int, unsigned int) const
  {
    DEAL_II_NOT_IMPLEMENTED();
    return 0;
  }

  void
  calculate_energy(VectorType &velocity,
                   Number     &energy,
                   Number     &enstrophy,
                   Number     &dissipation,
                   Number     &max_vorticity)
  {
    std::vector<Number> dst(5, 0.0);
    this->matrix_free.cell_loop(&MomentumOperator::do_energy_cell_integral_range,
                                this,
                                dst,
                                velocity);

    // sum over all MPI processes
    Number volume;
    volume      = dealii::Utilities::MPI::sum(dst.at(0), MPI_COMM_WORLD);
    energy      = dealii::Utilities::MPI::sum(dst.at(1), MPI_COMM_WORLD);
    enstrophy   = dealii::Utilities::MPI::sum(dst.at(2), MPI_COMM_WORLD);
    dissipation = dealii::Utilities::MPI::sum(dst.at(3), MPI_COMM_WORLD);

    energy /= volume;
    enstrophy /= volume;
    dissipation /= volume;

    max_vorticity = dealii::Utilities::MPI::max(dst.at(4), MPI_COMM_WORLD);
  };

private:
  number
  get_penalty_factor() const
  {
    return penalty_factor;
  }


  void
  do_cell_integral_range(const MatrixFree<dim, number>               &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim, -1, 0, n_components, Number> integrator(matrix_free,
                                                              dof_no_v,
                                                              quad_no_v);

    BDFTimeIntegratorConstants integration_constants(bdf_order);
    double                     gamma0 = integration_constants.get_gamma0();

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);
        integrator.gather_evaluate(src,
                                   EvaluationFlags::values | EvaluationFlags::gradients);

        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            const auto u          = integrator.get_value(q);
            const auto time_deriv = make_vectorized_array<number>(gamma0 / time_step) * u;

            const auto grad_u = integrator.get_gradient(q);
            const auto speed  = speeds_cells(cell, q);

            const auto divergence_penalty =
              penalty_factor_divergence[cell] * integrator.get_divergence(q);
            Tensor<2, dim, VectorizedArray<number>> div_penalty;
            for (unsigned int d = 0; d < dim; ++d)
              for (unsigned int e = 0; e < dim; ++e)
                div_penalty[d][e] = 0.;

            for (unsigned int d = 0; d < dim; ++d)
              div_penalty[d][d] = divergence_penalty;

            if (!use_skew_symmetric_convective_formulation)
              {
                const auto convective_flux = grad_u * speed;

                integrator.submit_value(time_deriv + convective_flux, q);
                integrator.submit_gradient(
                  div_penalty + make_vectorized_array<number>(viscosity) * grad_u, q);
              }
            else
              {
                const auto convective_value_flux    = 0.5 * grad_u * speed;
                const auto convective_gradient_flux = -0.5 * outer_product(speed, u);

                integrator.submit_value(time_deriv + convective_value_flux, q);
                integrator.submit_gradient(div_penalty +
                                             make_vectorized_array<number>(viscosity) *
                                               grad_u +
                                             convective_gradient_flux,
                                           q);
              }
          }
        integrator.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                     dst);
      }
  }


  void
  do_cell_integral_local(FEEvaluation<dim, -1, 0, n_components, Number> &integrator) const
  {
    BDFTimeIntegratorConstants integration_constants(bdf_order);
    double                     gamma0 = integration_constants.get_gamma0();

    integrator.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        const auto u          = integrator.get_value(q);
        const auto time_deriv = make_vectorized_array<number>(gamma0 / time_step) * u;

        const auto grad_u = integrator.get_gradient(q);
        const auto speed  = speeds_cells(integrator.get_current_cell_index(), q);

        const auto divergence_penalty =
          penalty_factor_divergence[integrator.get_current_cell_index()] *
          integrator.get_divergence(q);
        Tensor<2, dim, VectorizedArray<number>> div_penalty;
        for (unsigned int d = 0; d < dim; ++d)
          for (unsigned int e = 0; e < dim; ++e)
            div_penalty[d][e] = 0.;

        for (unsigned int d = 0; d < dim; ++d)
          div_penalty[d][d] = divergence_penalty;

        if (!use_skew_symmetric_convective_formulation)
          {
            const auto convective_flux = grad_u * speed;

            integrator.submit_value(time_deriv + convective_flux, q);
            integrator.submit_gradient(
              div_penalty + make_vectorized_array<number>(viscosity) * grad_u, q);
          }
        else
          {
            const auto convective_value_flux    = 0.5 * grad_u * speed;
            const auto convective_gradient_flux = -0.5 * outer_product(speed, u);

            integrator.submit_value(time_deriv + convective_value_flux, q);
            integrator.submit_gradient(div_penalty +
                                         make_vectorized_array<number>(viscosity) *
                                           grad_u +
                                         convective_gradient_flux,
                                       q);
          }
      }
    integrator.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }


  void
  do_face_integral_range(const MatrixFree<dim, number>               &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &range) const
  {
    if (!is_dg)
      return;
    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_inner(matrix_free,
                                                                        true,
                                                                        dof_no_v,
                                                                        quad_no_v);
    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_outer(matrix_free,
                                                                        false,
                                                                        dof_no_v,
                                                                        quad_no_v);


    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_inner.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);
        integrator_outer.reinit(face);
        integrator_outer.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);

        const VectorizedArray<number> sigma =
          std::max(integrator_inner.read_cell_data(array_penalty_parameter),
                   integrator_outer.read_cell_data(array_penalty_parameter)) *
          get_penalty_factor();

        const VectorizedArray<number> cont_pen =
          0.5 * (integrator_inner.read_cell_data(penalty_factor_continuity) +
                 integrator_outer.read_cell_data(penalty_factor_continuity));

        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
          {
            const auto normal = integrator_inner.normal_vector(q);

            const Tensor<1, dim, VectorizedArray<number>> solution_jump =
              (integrator_inner.get_value(q) - integrator_outer.get_value(q));

            const Tensor<1, dim, VectorizedArray<number>> solution_average =
              make_vectorized_array<number>(0.5) *
              (integrator_inner.get_value(q) + integrator_outer.get_value(q));


            const Tensor<1, dim, VectorizedArray<number>> averaged_normal_derivative =
              make_vectorized_array<number>(viscosity) *
              (integrator_inner.get_normal_derivative(q) +
               integrator_outer.get_normal_derivative(q)) *
              number(0.5);

            const Tensor<1, dim, VectorizedArray<number>> test_by_value =
              make_vectorized_array<number>(viscosity) * solution_jump * sigma -
              averaged_normal_derivative;

            const auto speed        = speeds_faces(face, q);
            const auto speed_normal = speed * normal;
            const auto convective_flux =
              speed_normal * solution_average +
              make_vectorized_array<number>(0.5) * std::abs(speed_normal) * solution_jump;

            const auto convective_flux_inner =
              convective_flux - speed_normal * integrator_inner.get_value(q);
            const auto convective_flux_outer =
              -convective_flux + speed_normal * integrator_outer.get_value(q);

            const auto continuity_penalty_value =
              cont_pen * (solution_jump * normal) * normal;

            if (!use_skew_symmetric_convective_formulation)
              {
                integrator_inner.submit_value(continuity_penalty_value + test_by_value +
                                                convective_flux_inner,
                                              q);
                integrator_outer.submit_value(-continuity_penalty_value - test_by_value +
                                                convective_flux_outer,
                                              q);
              }
            else
              {
                const auto convective_value_flux =
                  speed * (solution_average * normal) +
                  0.5 * (std::abs(speed_normal) * solution_jump);

                integrator_inner.submit_value(continuity_penalty_value + test_by_value +
                                                0.5 * convective_flux_inner +
                                                0.5 * convective_value_flux,
                                              q);
                integrator_outer.submit_value(-continuity_penalty_value - test_by_value +
                                                0.5 * convective_flux_outer -
                                                0.5 * convective_value_flux,
                                              q);
              }

            integrator_inner.submit_normal_derivative(
              -solution_jump * make_vectorized_array<number>(viscosity) * number(0.5), q);
            integrator_outer.submit_normal_derivative(
              -solution_jump * make_vectorized_array<number>(viscosity) * number(0.5), q);
          }

        integrator_inner.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
        integrator_outer.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
      }
  }


  void
  do_face_integral_local(
    FEFaceEvaluation<dim, -1, 0, n_components, Number> &integrator_inner,
    FEFaceEvaluation<dim, -1, 0, n_components, Number> &integrator_outer) const
  {
    integrator_inner.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    integrator_outer.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    const VectorizedArray<number> sigma =
      std::max(integrator_inner.read_cell_data(array_penalty_parameter),
               integrator_outer.read_cell_data(array_penalty_parameter)) *
      get_penalty_factor();

    const VectorizedArray<number> cont_pen =
      0.5 * (integrator_inner.read_cell_data(penalty_factor_continuity) +
             integrator_outer.read_cell_data(penalty_factor_continuity));

    for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
      {
        const auto normal = integrator_inner.normal_vector(q);

        const Tensor<1, dim, VectorizedArray<number>> solution_jump =
          (integrator_inner.get_value(q) - integrator_outer.get_value(q));

        const Tensor<1, dim, VectorizedArray<number>> solution_average =
          make_vectorized_array<number>(0.5) *
          (integrator_inner.get_value(q) + integrator_outer.get_value(q));


        const Tensor<1, dim, VectorizedArray<number>> averaged_normal_derivative =
          make_vectorized_array<number>(viscosity) *
          (integrator_inner.get_normal_derivative(q) +
           integrator_outer.get_normal_derivative(q)) *
          number(0.5);

        const Tensor<1, dim, VectorizedArray<number>> test_by_value =
          make_vectorized_array<number>(viscosity) * solution_jump * sigma -
          averaged_normal_derivative;

        const auto speed = speeds_faces(integrator_inner.get_cell_or_face_batch_id(), q);
        const auto speed_normal = speed * normal;
        const auto convective_flux =
          speed_normal * solution_average +
          make_vectorized_array<number>(0.5) * std::abs(speed_normal) * solution_jump;

        const auto convective_flux_inner =
          convective_flux - speed_normal * integrator_inner.get_value(q);
        const auto convective_flux_outer =
          -convective_flux + speed_normal * integrator_outer.get_value(q);

        const auto continuity_penalty_value =
          cont_pen * (solution_jump * normal) * normal;

        if (!use_skew_symmetric_convective_formulation)
          {
            integrator_inner.submit_value(continuity_penalty_value + test_by_value +
                                            convective_flux_inner,
                                          q);
            integrator_outer.submit_value(-continuity_penalty_value - test_by_value +
                                            convective_flux_outer,
                                          q);
          }
        else
          {
            const auto convective_value_flux =
              speed * (solution_average * normal) +
              0.5 * (std::abs(speed_normal) * solution_jump);

            integrator_inner.submit_value(continuity_penalty_value + test_by_value +
                                            0.5 * convective_flux_inner +
                                            0.5 * convective_value_flux,
                                          q);
            integrator_outer.submit_value(-continuity_penalty_value - test_by_value +
                                            0.5 * convective_flux_outer -
                                            0.5 * convective_value_flux,
                                          q);
          }

        integrator_inner.submit_normal_derivative(
          -solution_jump * make_vectorized_array<number>(viscosity) * number(0.5), q);
        integrator_outer.submit_normal_derivative(
          -solution_jump * make_vectorized_array<number>(viscosity) * number(0.5), q);
      }

    integrator_inner.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    integrator_outer.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }


  void
  do_boundary_integral_range(const MatrixFree<dim, number>               &matrix_free,
                             VectorType                                  &dst,
                             const VectorType                            &src,
                             const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_inner(matrix_free,
                                                                        true,
                                                                        dof_no_v,
                                                                        quad_no_v);
    AnalyticalSolutionVelocity<dim>                    exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(time);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_inner.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);

        const VectorizedArray<number> sigma =
          integrator_inner.read_cell_data(array_penalty_parameter) * get_penalty_factor();

        const VectorizedArray<number> cont_pen =
          integrator_inner.read_cell_data(penalty_factor_continuity);

        // Dirichlet boundary
        if (matrix_free.get_boundary_id(face) == 0 ||
            matrix_free.get_boundary_id(face) == 2)
          {
            for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
              {
                const auto normal = integrator_inner.normal_vector(q);

                const Tensor<1, dim, VectorizedArray<number>> u_inner =
                  make_vectorized_array<number>(viscosity) *
                  integrator_inner.get_value(q);

                const Tensor<1, dim, VectorizedArray<number>> normal_derivative_inner =
                  make_vectorized_array<number>(viscosity) *
                  integrator_inner.get_normal_derivative(q);

                const Tensor<1, dim, VectorizedArray<number>> test_by_value =
                  number(2.0) * u_inner * sigma - normal_derivative_inner;

                const auto speed        = speeds_faces(face, q);
                const auto speed_normal = speed * normal;
                const auto convective_flux =
                  (std::abs(speed_normal) - speed_normal) * integrator_inner.get_value(q);


                const auto g =
                  evaluate_function(exact_velocity, integrator_inner.quadrature_point(q));
                const auto continuity_penalty_value =
                  2. * cont_pen * ((integrator_inner.get_value(q) - g) * normal) * normal;

                if (!use_skew_symmetric_convective_formulation)
                  {
                    integrator_inner.submit_value(continuity_penalty_value +
                                                    test_by_value + convective_flux,
                                                  q);
                  }
                else
                  {
                    const auto convective_value_flux =
                      (std::abs(speed_normal)) * integrator_inner.get_value(q);
                    integrator_inner.submit_value(continuity_penalty_value +
                                                    test_by_value +
                                                    0.5 * convective_flux +
                                                    0.5 * convective_value_flux,
                                                  q);
                  }

                integrator_inner.submit_normal_derivative(-u_inner, q);
              }
          }
        else if (matrix_free.get_boundary_id(face) == 1)
          {
            // Nothing to do
            // Convective term cancels and viscous term is only inhomogenious
            for (const unsigned int q : integrator_inner.quadrature_point_indices())
              {
                integrator_inner.submit_normal_derivative(
                  Tensor<1, dim, VectorizedArray<number>>(), q);
                if (!use_skew_symmetric_convective_formulation)
                  integrator_inner.submit_value(Tensor<1, dim, VectorizedArray<number>>(),
                                                q);
                else
                  {
                    const auto speed = speeds_faces(face, q);
                    const auto convective_flux =
                      speed *
                      (integrator_inner.get_value(q) * integrator_inner.normal_vector(q));
                    integrator_inner.submit_value(0.5 * convective_flux, q);
                  }
              }
          }
        else
          AssertThrow(false,
                      ExcNotImplemented(
                        "Boundary id " +
                        std::to_string(int(matrix_free.get_boundary_id(face))) +
                        " not known"));

        integrator_inner.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
      }
  }


  void
  do_boundary_integral_local(
    FEFaceEvaluation<dim, -1, 0, n_components, Number> &integrator_inner) const
  {
    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(time);

    integrator_inner.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    const VectorizedArray<number> sigma =
      integrator_inner.read_cell_data(array_penalty_parameter) * get_penalty_factor();

    const VectorizedArray<number> cont_pen =
      integrator_inner.read_cell_data(penalty_factor_continuity);

    // Dirichlet boundary
    if (integrator_inner.boundary_id() == 0 || integrator_inner.boundary_id() == 2)
      {
        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
          {
            const auto normal = integrator_inner.normal_vector(q);

            const Tensor<1, dim, VectorizedArray<number>> u_inner =
              make_vectorized_array<number>(viscosity) * integrator_inner.get_value(q);

            const Tensor<1, dim, VectorizedArray<number>> normal_derivative_inner =
              make_vectorized_array<number>(viscosity) *
              integrator_inner.get_normal_derivative(q);

            const Tensor<1, dim, VectorizedArray<number>> test_by_value =
              number(2.0) * u_inner * sigma - normal_derivative_inner;

            const auto speed =
              speeds_faces(integrator_inner.get_cell_or_face_batch_id(), q);
            const auto speed_normal = speed * normal;
            const auto convective_flux =
              (std::abs(speed_normal) - speed_normal) * integrator_inner.get_value(q);


            const auto g =
              evaluate_function(exact_velocity, integrator_inner.quadrature_point(q));
            const auto continuity_penalty_value =
              2. * cont_pen * ((integrator_inner.get_value(q) - g) * normal) * normal;

            if (!use_skew_symmetric_convective_formulation)
              {
                integrator_inner.submit_value(continuity_penalty_value + test_by_value +
                                                convective_flux,
                                              q);
              }
            else
              {
                const auto convective_value_flux =
                  (std::abs(speed_normal)) * integrator_inner.get_value(q);
                integrator_inner.submit_value(continuity_penalty_value + test_by_value +
                                                0.5 * convective_flux +
                                                0.5 * convective_value_flux,
                                              q);
              }

            integrator_inner.submit_normal_derivative(-u_inner, q);
          }
      }
    else if (integrator_inner.boundary_id() == 1)
      {
        // Nothing to do
        // Convective term cancels and viscous term is only inhomogenious
        for (const unsigned int q : integrator_inner.quadrature_point_indices())
          {
            integrator_inner.submit_normal_derivative(
              Tensor<1, dim, VectorizedArray<number>>(), q);
            if (!use_skew_symmetric_convective_formulation)
              integrator_inner.submit_value(Tensor<1, dim, VectorizedArray<number>>(), q);
            else
              {
                const auto speed =
                  speeds_faces(integrator_inner.get_cell_or_face_batch_id(), q);
                const auto convective_flux = speed * (integrator_inner.get_value(q) *
                                                      integrator_inner.normal_vector(q));
                integrator_inner.submit_value(0.5 * convective_flux, q);
              }
          }
      }
    else
      AssertThrow(false,
                  ExcNotImplemented("Boundary id " +
                                    std::to_string(int(integrator_inner.boundary_id())) +
                                    " not known"));

    integrator_inner.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }


  void
  do_rhs_cell_integral_range(const MatrixFree<dim, number>               &matrix_free,
                             VectorType                                  &dst,
                             const std::vector<const VectorType *>       &src,
                             const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim, -1, 0, n_components, Number> integrator(matrix_free,
                                                              dof_no_v,
                                                              quad_no_v);
    FEEvaluation<dim, -1, 0, n_components, Number> integrator_speed(matrix_free,
                                                                    dof_no_v,
                                                                    quad_no_v);
    FEEvaluation<dim, -1, 0, 1, Number> integrator_p(matrix_free, dof_no_p, quad_no_v);

    AnalyticalRHS<dim> rhs(u_x_max, viscosity);
    rhs.set_time(time);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);
        integrator_speed.reinit(cell);
        integrator_p.reinit(cell);

        integrator.gather_evaluate(*src[0], EvaluationFlags::values);
        integrator_speed.gather_evaluate(*src[1], EvaluationFlags::values);
        // integrator_p.gather_evaluate(*src[2], EvaluationFlags::values);
        integrator_p.gather_evaluate(*src[2], EvaluationFlags::gradients);

        VectorizedArray<number> avg_veclocity(0.);
        VectorizedArray<number> volume(0.);
        for (unsigned int q = 0; q < integrator_speed.n_q_points; ++q)
          {
            volume += integrator_speed.JxW(q);
            avg_veclocity +=
              integrator_speed.JxW(q) * integrator_speed.get_value(q).norm();
          }
        avg_veclocity /= volume;
        penalty_factor_divergence[cell] = penalty_divergence * avg_veclocity *
                                          std::exp(std::log(volume) / (number)dim) /
                                          ((number)fe_degree_u + 1);
        penalty_factor_continuity[cell] = penalty_continuity * avg_veclocity;


        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            speeds_cells(cell, q) = integrator_speed.get_value(q);

            const auto f = evaluate_function(rhs, integrator.quadrature_point(q));

            // const auto p = integrator_p.get_value(q);
            const auto grad_p = integrator_p.get_gradient(q);

            const auto alpha_u = integrator.get_value(q);

            // integrator.submit_value(f + alpha_u, q);
            // integrator.submit_divergence(p, q);
            integrator.submit_value(f + alpha_u - grad_p, q);
          }
        //integrator.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        integrator.integrate_scatter(EvaluationFlags::values, dst);
      }
  }


  void
  do_rhs_face_integral_range(const MatrixFree<dim, number>               &matrix_free,
                             VectorType                                  &dst,
                             const std::vector<const VectorType *>       &src,
                             const std::pair<unsigned int, unsigned int> &range) const
  {
    if (!is_dg)
      return;

    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_inner(matrix_free,
                                                                        true,
                                                                        dof_no_v,
                                                                        quad_no_v);
    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_outer(matrix_free,
                                                                        false,
                                                                        dof_no_v,
                                                                        quad_no_v);
    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_speed_inner(matrix_free,
                                                                              true,
                                                                              dof_no_v,
                                                                              quad_no_v);
    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_speed_outer(matrix_free,
                                                                              false,
                                                                              dof_no_v,
                                                                              quad_no_v);
    FEFaceEvaluation<dim, -1, 0, 1, Number>            integrator_inner_p(matrix_free,
                                                               true,
                                                               dof_no_p,
                                                               quad_no_v);
    FEFaceEvaluation<dim, -1, 0, 1, Number>            integrator_outer_p(matrix_free,
                                                               false,
                                                               dof_no_p,
                                                               quad_no_v);


    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_speed_inner.reinit(face);
        integrator_inner_p.reinit(face);

        integrator_outer.reinit(face);
        integrator_speed_outer.reinit(face);
        integrator_outer_p.reinit(face);


        integrator_speed_inner.gather_evaluate(*src[1], EvaluationFlags::values);
        integrator_inner_p.gather_evaluate(*src[2], EvaluationFlags::values);

        integrator_speed_outer.gather_evaluate(*src[1], EvaluationFlags::values);
        integrator_outer_p.gather_evaluate(*src[2], EvaluationFlags::values);


        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
          {
            const auto normal = integrator_inner.normal_vector(q);

            speeds_faces(face, q) =
              make_vectorized_array<number>(0.5) *
              (integrator_speed_inner.get_value(q) + integrator_speed_outer.get_value(q));


            // const Tensor<1, dim, VectorizedArray<number>> p_avg =
            // number(0.5)*(integrator_inner_p.get_value(q) +
            // integrator_outer_p.get_value(q)) * integrator_inner.normal_vector(q);
            const Tensor<1, dim, VectorizedArray<number>> p_jump =
              number(0.5) *
              (integrator_inner_p.get_value(q) - integrator_outer_p.get_value(q)) *
              normal;


            // integrator_inner.submit_value(-p_avg, q);
            // integrator_outer.submit_value(p_avg, q);
            integrator_inner.submit_value(p_jump, q);
            integrator_outer.submit_value(p_jump, q);
          }

        integrator_inner.integrate_scatter(EvaluationFlags::values, dst);
        integrator_outer.integrate_scatter(EvaluationFlags::values, dst);
      }
  }


  void
  do_rhs_boundary_range(const MatrixFree<dim, number>               &matrix_free,
                        VectorType                                  &dst,
                        const std::vector<const VectorType *>       &src,
                        const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_inner(matrix_free,
                                                                        true,
                                                                        dof_no_v,
                                                                        quad_no_v);
    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_speed_inner(matrix_free,
                                                                              true,
                                                                              dof_no_v,
                                                                              quad_no_v);
    FEFaceEvaluation<dim, -1, 0, 1, Number>            integrator_inner_p(matrix_free,
                                                               true,
                                                               dof_no_p,
                                                               quad_no_v);

    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(time);
    AnalyticalSolutionPressure<dim> exact_pressure(u_x_max, viscosity);
    exact_pressure.set_time(time);

    AnalyticalSolutionVelocity<dim> exact_velocity_m(u_x_max, viscosity);
    exact_velocity_m.set_time(time - time_step);

    AnalyticalSolutionVelocity<dim> exact_velocity_m2(u_x_max, viscosity);
    exact_velocity_m2.set_time(time - 2.0 * time_step);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_speed_inner.reinit(face);
        integrator_inner_p.reinit(face);

        integrator_speed_inner.gather_evaluate(*src[1], EvaluationFlags::values);
        integrator_inner_p.gather_evaluate(*src[2], EvaluationFlags::values);

        const VectorizedArray<number> sigma =
          integrator_inner.read_cell_data(array_penalty_parameter) * get_penalty_factor();

        if (matrix_free.get_boundary_id(face) == 0 ||
            matrix_free.get_boundary_id(face) == 2)
          {
            for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
              {
                const auto normal = integrator_inner.normal_vector(q);

                const auto g =
                  evaluate_function(exact_velocity, integrator_inner.quadrature_point(q));

                Tensor<1, dim, VectorizedArray<number>> speed;

                if (use_extrapolated_velocity)
                  {
                    const auto u_plus_m =
                      evaluate_function(exact_velocity_m,
                                        integrator_inner.quadrature_point(q));
                    const auto u_plus_m2 =
                      evaluate_function(exact_velocity_m2,
                                        integrator_inner.quadrature_point(q));
                    auto extrapolated_velocity = 2.0 * u_plus_m - u_plus_m2;
                    speed =
                      0.5 * (integrator_speed_inner.get_value(q) + extrapolated_velocity);
                  }
                else
                  {
                    speed = make_vectorized_array<number>(0.5) *
                            (integrator_speed_inner.get_value(q) + g);
                  }

                speeds_faces(face, q)   = speed;
                const auto speed_normal = speed * normal;

                const auto convective_flux = (std::abs(speed_normal) - speed_normal) * g;

                const auto value_flux =
                  make_vectorized_array<number>(2.0 * viscosity) * sigma * g;
                const auto gradient_flux = make_vectorized_array<number>(viscosity) * g;

                const Tensor<1, dim, VectorizedArray<number>> p =
                  0. * integrator_inner.normal_vector(q);
                // integrator_inner_p.get_value(q) * integrator_inner.normal_vector(q);

                integrator_inner.submit_normal_derivative(-gradient_flux, q);

                if (!use_skew_symmetric_convective_formulation)
                  {
                    integrator_inner.submit_value(value_flux - p + convective_flux, q);
                  }
                else
                  {
                    const auto convective_value_flux =
                      -speed * (g * normal) + std::abs(speed_normal) * g;
                    integrator_inner.submit_value(value_flux - p + 0.5 * convective_flux +
                                                    0.5 * convective_value_flux,
                                                  q);
                  }
              }
          }
        else if (matrix_free.get_boundary_id(face) == 1)
          {
            for (const unsigned int q : integrator_inner.quadrature_point_indices())
              {
                speeds_faces(face, q) = integrator_speed_inner.get_value(q);
                const auto normal     = integrator_inner.normal_vector(q);

                const auto grad_g =
                  evaluate_tensor_function(exact_velocity,
                                           integrator_inner.quadrature_point(q));
                const auto h_u = make_vectorized_array(viscosity) * grad_g * normal;

                const auto p_plus =
                  evaluate_scalar_function(exact_pressure,
                                           integrator_inner.quadrature_point(q));
                const auto p_minus  = integrator_inner_p.get_value(q);
                const auto pressure = (p_minus - p_plus) * normal;

                integrator_inner.submit_normal_derivative(
                  Tensor<1, dim, VectorizedArray<number>>(), q);
                integrator_inner.submit_value(h_u + pressure, q);
              }
          }

        integrator_inner.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
      }
  }


  void
  local_vorticity_domain(const MatrixFree<dim, number>               &data,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, dim, number> eval_u(data, 0);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_u.reinit(cell);
        eval_u.gather_evaluate(src, EvaluationFlags::gradients);

        for (unsigned int q = 0; q < eval_u.n_q_points; ++q)
          {
            if constexpr (dim == 2)
              {
                const auto omega = eval_u.get_curl(q);
                dealii::Tensor<1, dim, dealii::VectorizedArray<number>> omega_vector;
                for (unsigned int d = 0; d < dim; ++d)
                  omega_vector[d] = 0.;
                omega_vector[0] = omega[0];
                eval_u.submit_value(omega_vector, q);
              }
            else if constexpr (dim == 3)
              {
                eval_u.submit_value(eval_u.get_curl(q), q);
              }
          }

        eval_u.integrate_scatter(dealii::EvaluationFlags::values, dst);
      }
  }


  void
  do_energy_cell_integral_range(
    const MatrixFree<dim, number>               &matrix_free,
    std::vector<Number>                         &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, n_components, Number> fe_eval(matrix_free,
                                                           dof_no_v,
                                                           quad_no_v);

    Number volume        = 0.;
    Number energy        = 0.;
    Number enstrophy     = 0.;
    Number dissipation   = 0.;
    Number max_vorticity = 0.;

    // Loop over all elements
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(dealii::EvaluationFlags::values |
                         dealii::EvaluationFlags::gradients);

        VectorizedArray<Number> volume_vec    = dealii::make_vectorized_array<Number>(0.);
        VectorizedArray<Number> energy_vec    = dealii::make_vectorized_array<Number>(0.);
        VectorizedArray<Number> enstrophy_vec = dealii::make_vectorized_array<Number>(0.);
        VectorizedArray<Number> dissipation_vec =
          dealii::make_vectorized_array<Number>(0.);
        VectorizedArray<Number> max_vorticity_vec =
          dealii::make_vectorized_array<Number>(0.);

        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          {
            volume_vec += fe_eval.JxW(q);

            auto velocity = fe_eval.get_value(q);
            energy_vec += fe_eval.JxW(q) * dealii::make_vectorized_array<Number>(0.5) *
                          velocity * velocity;

            auto velocity_gradient = fe_eval.get_gradient(q);
            dissipation_vec += fe_eval.JxW(q) *
                               dealii::make_vectorized_array<Number>(viscosity) *
                               scalar_product(velocity_gradient, velocity_gradient);

            dealii::Tensor<1, dim == 3 ? dim : 1, VectorizedArray<Number>> omega =
              fe_eval.get_curl(q);

            VectorizedArray<Number> norm_omega = omega * omega;

            enstrophy_vec +=
              fe_eval.JxW(q) * dealii::make_vectorized_array<Number>(0.5) * norm_omega;

            max_vorticity_vec = std::max(max_vorticity_vec, std::sqrt(norm_omega));
          }

        // sum over entries of dealii::VectorizedArray, but only over those
        // that are "active"
        for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell);
             ++v)
          {
            volume += volume_vec[v];
            energy += energy_vec[v];
            enstrophy += enstrophy_vec[v];
            dissipation += dissipation_vec[v];

            max_vorticity = std::max(max_vorticity, max_vorticity_vec[v]);
          }
      }

    dst.at(0) += volume;
    dst.at(1) += energy;
    dst.at(2) += enstrophy;
    dst.at(3) += dissipation;
    dst.at(4) = std::max(dst.at(4), max_vorticity);
  }

  MatrixFree<dim, number> matrix_free;

  number       penalty_factor;
  number       viscosity;
  number       time_step;
  number       time;
  unsigned int bdf_order;
  unsigned int fe_degree_u;
  bool         is_dg;

  dealii::AlignedVector<dealii::VectorizedArray<Number>>    array_penalty_parameter;
  mutable Table<2, Tensor<1, dim, VectorizedArray<number>>> speeds_cells;
  mutable Table<2, Tensor<1, dim, VectorizedArray<number>>> speeds_faces;

  mutable dealii::AlignedVector<dealii::VectorizedArray<Number>>
    penalty_factor_divergence;
  mutable dealii::AlignedVector<dealii::VectorizedArray<Number>>
    penalty_factor_continuity;
};


template <int dim, typename number>
class InverseMassPreconditioner
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;
  typedef InverseMassPreconditioner<dim, number> This;


  InverseMassPreconditioner() = default;

  void
  reinit(const MatrixFree<dim, number> &matrix_free, number scaling_factor_in)
  {
    scaling_factor    = scaling_factor_in;
    this->matrix_free = &matrix_free;
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
                                  const number       time_step,
                                  const unsigned int bdf_order)
  {
    const auto  mf          = momentum_operator.get_matrix_free();
    const auto &dof_handler = mf.get_dof_handler(dof_no_v);

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
        const VectorTypeSystem &vec_u_rhs)
  {
    const unsigned int                            min_level = mg_matrices.min_level();
    std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

    const auto       precond_point_jacobi = *smoother_data[min_level].preconditioner;
    ReductionControl coarse_grid_solver_control(10000, 1e-20, 1e-4, false, false);
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

    // Setup levels and transfers
    mg::Matrix<VectorType> mg_matrix(mg_matrices);
    Multigrid<VectorType>  mg(mg_matrix, *mg_coarse, transfer, mg_smoother, mg_smoother);

    PreconditionerType preconditioner(
      momentum_operator.get_matrix_free().get_dof_handler(dof_no_v), mg, transfer);

    SolverControl                 control(100000, 1e-12 * vec_u_rhs.l2_norm());
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
};



template <int dim, typename number>
class PressureOperator : public Subscriptor
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;
  using value_type = number;
  using size_type  = types::global_dof_index;

  PressureOperator() = default;

  void
  reinit(const MatrixFree<dim, number> &matrix_free_in,
         const unsigned int             bdf_order_in,
         const number                   time_step_in)
  {
    bdf_order         = bdf_order_in;
    time_step         = time_step_in;
    this->matrix_free = &matrix_free_in;
    is_dg = matrix_free->get_dof_handler(dof_no_p).get_fe().n_dofs_per_vertex() == 0;

    const unsigned int fe_degree = matrix_free->get_dof_handler(dof_no_p).get_fe().degree;
    const double       penalty_factor = 1.0 * (fe_degree + 1) * (fe_degree);
    {
      unsigned int n_cells =
        matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();
      array_penalty_parameter.resize(n_cells);

      const dealii::FiniteElement<dim> &fe =
        matrix_free->get_dof_handler(dof_no_p).get_fe();
      const auto reference_cells =
        matrix_free->get_dof_handler(dof_no_p).get_fe().reference_cell();
      const auto mapping = matrix_free->get_mapping_info().mapping;

      const auto quadrature =
        reference_cells.template get_gauss_type_quadrature<dim>(fe_degree + 1);
      dealii::FEValues<dim> fe_values(*mapping,
                                      fe,
                                      quadrature,
                                      dealii::update_JxW_values);

      const auto face_quadrature =
        reference_cells.face_reference_cell(0)
          .template get_gauss_type_quadrature<dim - 1>(fe_degree + 1);
      dealii::FEFaceValues<dim> fe_face_values(*mapping,
                                               fe,
                                               face_quadrature,
                                               dealii::update_JxW_values);

      for (unsigned int i = 0; i < n_cells; ++i)
        {
          for (unsigned int v = 0; v < matrix_free->n_active_entries_per_cell_batch(i);
               ++v)
            {
              typename dealii::DoFHandler<dim>::cell_iterator cell =
                matrix_free->get_cell_iterator(i, v, dof_no_p);
              fe_values.reinit(cell);

              // calculate cell volume
              number volume = 0;
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                {
                  volume += fe_values.JxW(q);
                }

              // calculate surface area
              number surface_area = 0;
              for (const unsigned int f : cell->face_indices())
                {
                  fe_face_values.reinit(cell, f);
                  const number factor =
                    (cell->at_boundary(f) and not(cell->has_periodic_neighbor(f))) ? 1. :
                                                                                     0.5;
                  for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                    {
                      surface_area += fe_face_values.JxW(q) * factor;
                    }
                }

              array_penalty_parameter[i][v] = surface_area / volume * penalty_factor;
            }
        }
    }
  }

  void
  set_time(const double time)
  {
    this->time = time;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free->loop(&PressureOperator::local_apply_domain,
                      &PressureOperator::local_apply_inner_face,
                      &PressureOperator::local_apply_boundary_face,
                      this,
                      dst,
                      src,
                      true,
                      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

  void
  vmult_add(VectorType &dst, const VectorType &src) const
  {
    matrix_free->loop(&PressureOperator::local_apply_domain,
                      &PressureOperator::local_apply_inner_face,
                      &PressureOperator::local_apply_boundary_face,
                      this,
                      dst,
                      src,
                      false,
                      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  void
  Tvmult_add(VectorType &dst, const VectorType &src) const
  {
    vmult_add(dst, src);
  }

  void
  compute_rhs(VectorType &dst, const VectorType &vorticity)
  {
    matrix_free->loop(&PressureOperator::local_rhs_domain,
                      &PressureOperator::local_rhs_inner_face,
                      &PressureOperator::local_rhs_boundary_face,
                      this,
                      dst,
                      vorticity,
                      true,
                      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

  void
  compute_convective_rhs(VectorType &dst, const VectorType &velocity)
  {
    matrix_free->loop(&PressureOperator::local_convective_domain,
                      &PressureOperator::local_convective_inner_face,
                      &PressureOperator::local_convective_boundary_face,
                      this,
                      dst,
                      velocity,
                      true,
                      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

  void
  compute_divergence(VectorType &dst, const VectorType &velocity)
  {
    matrix_free->loop(&PressureOperator::local_divergence_domain,
                      &PressureOperator::local_divergence_inner_face,
                      &PressureOperator::local_divergence_boundary_face,
                      this,
                      dst,
                      velocity,
                      true,
                      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

  void
  set_time_step(const double t)
  {
    time_step = t;
  }

  void
  set_bdf_order(const unsigned int bdf_order_in)
  {
    bdf_order = bdf_order_in;
  }

  void
  get_system_matrix(TrilinosWrappers::SparseMatrix &system_matrix)
  {
    const auto &dof_handler = this->matrix_free->get_dof_handler(dof_no_p);

    TrilinosWrappers::SparsityPattern dsp(
      dof_handler.locally_owned_dofs(),
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(dof_handler),
      dof_handler.get_triangulation().get_communicator());

    if (is_dg)
      DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, AffineConstraints<number>());
    else
      DoFTools::make_sparsity_pattern(dof_handler, dsp, AffineConstraints<number>());

    dsp.compress();
    system_matrix.reinit(dsp);
    system_matrix = 0.;

    if (is_dg)
      MatrixFreeTools::compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
        *matrix_free,
        AffineConstraints<number>(),
        system_matrix,
        [&](auto &phi) { local_apply_domain_matrix_based(phi); },
        [&](auto &phi_m, auto &phi_p) {
          local_apply_inner_face_matrix_based(phi_m, phi_p);
        },
        [&](auto &phi) { local_apply_boundary_face_matrix_based(phi); },
        dof_no_p,
        quad_no_p,
        0);
    else
      MatrixFreeTools::compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
        *matrix_free,
        AffineConstraints<number>(),
        system_matrix,
        [&](auto &phi) { local_apply_domain_matrix_based(phi); },
        {},
        [&](auto &phi) { local_apply_boundary_face_matrix_based(phi); },
        dof_no_p,
        quad_no_p,
        0);
  }

  void
  compute_inverse_diagonal(VectorType &diagonal_vector) const
  {
    this->matrix_free->initialize_dof_vector(diagonal_vector, dof_no_p);

    if (is_dg)
      MatrixFreeTools::compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
        *matrix_free,
        diagonal_vector,
        [&](auto &phi) { local_apply_domain_matrix_based(phi); },
        [&](auto &phi_m, auto &phi_p) {
          local_apply_inner_face_matrix_based(phi_m, phi_p);
        },
        [&](auto &phi) { local_apply_boundary_face_matrix_based(phi); },
        dof_no_p,
        quad_no_p,
        0);
    else
      MatrixFreeTools::compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
        *matrix_free,
        diagonal_vector,
        [&](auto &phi) { local_apply_domain_matrix_based(phi); },
        {},
        [&](auto &phi) { local_apply_boundary_face_matrix_based(phi); },
        dof_no_p,
        quad_no_p,
        0);

    for (unsigned int i = 0; i < diagonal_vector.locally_owned_size(); ++i)
      {
        if (std::abs(diagonal_vector.local_element(i)) > 1.0e-10)
          diagonal_vector.local_element(i) = 1.0 / diagonal_vector.local_element(i);
        else
          diagonal_vector.local_element(i) = 1.0;
      }
  }


  number
  el(unsigned int, unsigned int) const
  {
    DEAL_II_NOT_IMPLEMENTED();
    return 0;
  }

  types::global_dof_index
  m() const
  {
    return matrix_free->get_dof_handler(dof_no_p).n_dofs();
  }

  const MatrixFree<dim, number> &
  get_matrix_free() const
  {
    return *matrix_free;
  }


private:
  const MatrixFree<dim, number>                         *matrix_free;
  dealii::AlignedVector<dealii::VectorizedArray<number>> array_penalty_parameter;
  number                                                 time;
  double                                                 time_step;
  unsigned int                                           bdf_order;
  bool                                                   is_dg;

  void
  local_apply_domain(const MatrixFree<dim, number>               &data,
                     VectorType                                  &dst,
                     const VectorType                            &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, 1, number> eval(data, 1, 2);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        // compute u^h(x) from src
        eval.gather_evaluate(src, EvaluationFlags::gradients);

        // loop over quadrature points and compute the local volume flux
        for (const unsigned int q : eval.quadrature_point_indices())
          eval.submit_gradient(eval.get_gradient(q), q);

        // multiply by nabla v^h(x) and sum
        eval.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  void
  local_apply_inner_face(const MatrixFree<dim, number>               &data,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const
  {
    if (!is_dg)
      return;

    FEFaceEvaluation<dim, -1, 0, 1, number> eval_minus(data, true, 1, 2);
    FEFaceEvaluation<dim, -1, 0, 1, number> eval_plus(data, false, 1, 2);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);
        eval_minus.gather_evaluate(src,
                                   EvaluationFlags::values | EvaluationFlags::gradients);
        eval_plus.gather_evaluate(src,
                                  EvaluationFlags::values | EvaluationFlags::gradients);

        const VectorizedArray<number> penalty_factor =
          std::max(eval_minus.read_cell_data(array_penalty_parameter),
                   eval_plus.read_cell_data(array_penalty_parameter));

        for (const unsigned int q : eval_minus.quadrature_point_indices())
          {
            const auto u_minus = eval_minus.get_value(q);
            const auto u_plus  = eval_plus.get_value(q);

            const auto viscous_value_flux =
              make_vectorized_array<number>(0.5) * (eval_minus.get_normal_derivative(q) +
                                                    eval_plus.get_normal_derivative(q)) -
              penalty_factor * (u_minus - u_plus);
            const auto viscous_gradient_flux =
              make_vectorized_array<number>(0.5) * (u_plus - u_minus);

            eval_minus.submit_normal_derivative(viscous_gradient_flux, q);
            eval_plus.submit_normal_derivative(viscous_gradient_flux, q);

            eval_minus.submit_value(-viscous_value_flux, q);
            eval_plus.submit_value(viscous_value_flux, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                     dst);
        eval_plus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                    dst);
      }
  }

  void
  local_apply_boundary_face(const MatrixFree<dim, number>               &data,
                            VectorType                                  &dst,
                            const VectorType                            &src,
                            const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, number> eval_minus(data, true, 1, 2);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src,
                                   EvaluationFlags::values | EvaluationFlags::gradients);

        const VectorizedArray<number> penalty_factor =
          eval_minus.read_cell_data(array_penalty_parameter);

        if (data.get_boundary_id(face) == 0 || data.get_boundary_id(face) == 2)
          {
            // Do nothing
            for (const unsigned int q : eval_minus.quadrature_point_indices())
              {
                eval_minus.submit_normal_derivative({}, q);
                eval_minus.submit_value({}, q);
              }
          }
        else if (data.get_boundary_id(face) == 1)
          {
            for (const unsigned int q : eval_minus.quadrature_point_indices())
              {
                const auto u_minus = eval_minus.get_value(q);

                const auto viscous_value_flux =
                  2.0 * penalty_factor * u_minus - eval_minus.get_normal_derivative(q);
                const auto viscous_gradient_flux = -u_minus;

                eval_minus.submit_normal_derivative(viscous_gradient_flux, q);
                eval_minus.submit_value(viscous_value_flux, q);
              }
          }
        else
          AssertThrow(false,
                      ExcNotImplemented("Boundary id " +
                                        std::to_string(int(data.get_boundary_id(face))) +
                                        " not known"));

        eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                     dst);
      }
  }

  void
  local_apply_domain_matrix_based(FEEvaluation<dim, -1, 0, 1, number> &eval) const
  {
    eval.evaluate(EvaluationFlags::gradients);

    for (const unsigned int q : eval.quadrature_point_indices())
      eval.submit_gradient(eval.get_gradient(q), q);

    eval.integrate(EvaluationFlags::gradients);
  }


  void
  local_apply_inner_face_matrix_based(
    FEFaceEvaluation<dim, -1, 0, 1, number> &eval_minus,
    FEFaceEvaluation<dim, -1, 0, 1, number> &eval_plus) const
  {
    eval_minus.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    eval_plus.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    const VectorizedArray<number> penalty_factor =
      std::max(eval_minus.read_cell_data(array_penalty_parameter),
               eval_plus.read_cell_data(array_penalty_parameter));

    for (const unsigned int q : eval_minus.quadrature_point_indices())
      {
        const auto u_minus = eval_minus.get_value(q);
        const auto u_plus  = eval_plus.get_value(q);

        const auto viscous_value_flux =
          make_vectorized_array<number>(0.5) *
            (eval_minus.get_normal_derivative(q) + eval_plus.get_normal_derivative(q)) -
          penalty_factor * (u_minus - u_plus);
        const auto viscous_gradient_flux =
          make_vectorized_array<number>(0.5) * (u_plus - u_minus);

        eval_minus.submit_normal_derivative(viscous_gradient_flux, q);
        eval_plus.submit_normal_derivative(viscous_gradient_flux, q);

        eval_minus.submit_value(-viscous_value_flux, q);
        eval_plus.submit_value(viscous_value_flux, q);
      }

    eval_minus.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    eval_plus.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }


  void
  local_apply_boundary_face_matrix_based(
    FEFaceEvaluation<dim, -1, 0, 1, number> &eval_minus) const
  {
    eval_minus.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    const VectorizedArray<number> penalty_factor =
      eval_minus.read_cell_data(array_penalty_parameter);

    if (eval_minus.boundary_id() == 0 || eval_minus.boundary_id() == 2)
      {
        // Do nothing
        for (const unsigned int q : eval_minus.quadrature_point_indices())
          {
            eval_minus.submit_normal_derivative({}, q);
            eval_minus.submit_value({}, q);
          }
      }
    else if (eval_minus.boundary_id() == 1)
      {
        for (const unsigned int q : eval_minus.quadrature_point_indices())
          {
            const auto u_minus = eval_minus.get_value(q);

            const auto viscous_value_flux =
              2.0 * penalty_factor * u_minus - eval_minus.get_normal_derivative(q);
            const auto viscous_gradient_flux = -u_minus;

            eval_minus.submit_normal_derivative(viscous_gradient_flux, q);
            eval_minus.submit_value(viscous_value_flux, q);
          }
      }
    else
      AssertThrow(false,
                  ExcNotImplemented("Boundary id " +
                                    std::to_string(int(eval_minus.boundary_id())) +
                                    " not known"));

    eval_minus.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  void
  local_rhs_domain(const MatrixFree<dim, number> &data,
                   VectorType                    &dst,
                   const VectorType &,
                   const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, 1, number> eval_p(data, 1, 1);

    AnalyticalRHS<dim> rhs(u_x_max, viscosity);
    rhs.set_time(time);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_p.reinit(cell);

        // loop over quadrature points and compute the local volume flux
        for (const unsigned int q : eval_p.quadrature_point_indices())
          {
            const auto f = evaluate_function(rhs, eval_p.quadrature_point(q));
            eval_p.submit_gradient(f, q);
          }

        // multiply by nabla v^h(x) and sum
        eval_p.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  void
  local_rhs_inner_face(const MatrixFree<dim, number> &data,
                       VectorType                    &dst,
                       const VectorType &,
                       const std::pair<unsigned int, unsigned int> &face_range) const
  {
    if (!is_dg)
      return;

    FEFaceEvaluation<dim, -1, 0, 1, number> eval_p_minus(data, true, 1, 1);
    FEFaceEvaluation<dim, -1, 0, 1, number> eval_p_plus(data, false, 1, 1);

    AnalyticalRHS<dim> rhs(u_x_max, viscosity);
    rhs.set_time(time);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_p_minus.reinit(face);
        eval_p_plus.reinit(face);

        for (const unsigned int q : eval_p_minus.quadrature_point_indices())
          {
            const auto f      = evaluate_function(rhs, eval_p_minus.quadrature_point(q));
            const auto normal = eval_p_minus.normal_vector(q);
            const auto flux   = f * normal;

            eval_p_minus.submit_value(-flux, q);
            eval_p_plus.submit_value(flux, q);
          }

        eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
        eval_p_plus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  void
  local_rhs_boundary_face(const MatrixFree<dim, number>               &data,
                          VectorType                                  &dst,
                          const VectorType                            &src,
                          const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_minus(data, true, 1, 1);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_vorticity(data, true, 0, 1);

    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(time);
    AnalyticalSolutionPressure<dim> exact_pressure(u_x_max, viscosity);
    exact_pressure.set_time(time);
    AnalyticalRHS<dim> rhs(u_x_max, viscosity);
    rhs.set_time(time);

    BDFTimeIntegratorConstants integration_constants(bdf_order);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        if (data.get_boundary_id(face) == 0 || data.get_boundary_id(face) == 2)
          {
            eval_p_minus.reinit(face);
            eval_vorticity.reinit(face);

            eval_vorticity.gather_evaluate(src, EvaluationFlags::gradients);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
              {
                const auto normal = eval_p_minus.normal_vector(q);

                exact_velocity.set_time(time);
                const auto g =
                  evaluate_function(exact_velocity, eval_p_minus.quadrature_point(q));
                auto u_plus =
                  make_vectorized_array(integration_constants.get_gamma0() / time_step) *
                  g;

                for (unsigned int i = 0; i < integration_constants.get_order(); ++i)
                  {
                    exact_velocity.set_time(time - (i + 1) * time_step);
                    u_plus -=
                      make_vectorized_array(integration_constants.get_alpha(i) /
                                            time_step) *
                      evaluate_function(exact_velocity, eval_p_minus.quadrature_point(q));
                  }
                const auto flux = (-u_plus) * normal;

                Tensor<1, dim, VectorizedArray<number>> curl_omega =
                  CurlCompute<dim, FEFaceEvaluation<dim, -1, 0, dim, number>>::compute(
                    eval_vorticity, q);

                const auto curl_flux = (-viscosity) * normal * curl_omega;

                eval_p_minus.submit_value(flux + curl_flux, q);
              }

            eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
          }
        else
          {
            eval_p_minus.reinit(face);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
              {
                const auto f = evaluate_function(rhs, eval_p_minus.quadrature_point(q));

                const auto normal = eval_p_minus.normal_vector(q);

                const auto flux = -f * normal;

                const auto g_p =
                  evaluate_scalar_function(exact_pressure,
                                           eval_p_minus.quadrature_point(q));

                const VectorizedArray<number> penalty_factor =
                  eval_p_minus.read_cell_data(array_penalty_parameter);
                const auto penalty = penalty_factor * 2. * g_p;

                eval_p_minus.submit_value(flux + penalty, q);
                eval_p_minus.submit_normal_derivative(-g_p, q);
              }

            eval_p_minus.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
          }
      }
  }



  void
  local_convective_domain(const MatrixFree<dim, number>               &data,
                          VectorType                                  &dst,
                          const VectorType                            &src,
                          const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, 1, number>   eval_p(data, 1, 1);
    FEEvaluation<dim, -1, 0, dim, number> eval_u(data, 0, 1);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_p.reinit(cell);
        eval_u.reinit(cell);

        eval_u.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        // loop over quadrature points and compute the local volume flux
        for (const unsigned int q : eval_p.quadrature_point_indices())
          {
            const auto convective_flux = -eval_u.get_gradient(q) * eval_u.get_value(q);
            eval_p.submit_gradient(convective_flux, q);
          }

        // multiply by nabla v^h(x) and sum
        eval_p.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  void
  local_convective_inner_face(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    if (!is_dg)
      return;
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_minus(data, true, 1, 1);
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_plus(data, false, 1, 1);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_minus(data, true, 0, 1);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_plus(data, false, 0, 1);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_p_minus.reinit(face);
        eval_p_plus.reinit(face);
        eval_u_minus.reinit(face);
        eval_u_plus.reinit(face);

        eval_u_minus.gather_evaluate(src,
                                     EvaluationFlags::values |
                                       EvaluationFlags::gradients);
        eval_u_plus.gather_evaluate(src,
                                    EvaluationFlags::values | EvaluationFlags::gradients);

        for (const unsigned int q : eval_p_minus.quadrature_point_indices())
          {
            const auto normal = eval_p_minus.normal_vector(q);

            if (use_pressure_convective_upwind_flux)
              {
                const auto u_minus      = eval_u_minus.get_value(q);
                const auto u_plus       = eval_u_plus.get_value(q);
                const auto u_minus_grad = eval_u_minus.get_gradient(q);
                const auto u_plus_grad  = eval_u_plus.get_gradient(q);
                const auto value_flux =
                  0.5 * (u_minus_grad + u_plus_grad) * 0.5 * (u_minus + u_plus);

                eval_p_minus.submit_value(value_flux * normal, q);
                eval_p_plus.submit_value(-value_flux * normal, q);


                const auto grad_flux = 0.5 * (u_minus - u_plus);
                eval_p_minus.submit_gradient(grad_flux * (u_minus * normal), q);
                eval_p_plus.submit_gradient(grad_flux * (u_plus * normal), q);
              }
            else
              {
                const auto gradu_u_minus =
                  eval_u_minus.get_gradient(q) * eval_u_minus.get_value(q);
                const auto gradu_u_plus =
                  eval_u_plus.get_gradient(q) * eval_u_plus.get_value(q);
                const auto convective_flux =
                  number(0.5) * (gradu_u_minus + gradu_u_plus) * normal;

                eval_p_minus.submit_value(convective_flux, q);
                eval_p_plus.submit_value(-convective_flux, q);
                eval_p_minus.submit_gradient({}, q);
                eval_p_plus.submit_gradient({}, q);
              }
          }

        eval_p_minus.integrate_scatter(EvaluationFlags::values |
                                         EvaluationFlags::gradients,
                                       dst);
        eval_p_plus.integrate_scatter(EvaluationFlags::values |
                                        EvaluationFlags::gradients,
                                      dst);
      }
  }

  void
  local_convective_boundary_face(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_minus(data, true, 1, 1);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_minus(data, true, 0, 1);

    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(time);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        if (data.get_boundary_id(face) == 0 || data.get_boundary_id(face) == 2)
          {
            eval_p_minus.reinit(face);
            eval_u_minus.reinit(face);

            eval_u_minus.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
              {
                const auto normal = eval_p_minus.normal_vector(q);

                exact_velocity.set_time(time);
                const auto g =
                  evaluate_function(exact_velocity, eval_p_minus.quadrature_point(q));

                if (use_pressure_convective_upwind_flux)
                  {
                    const auto u_minus      = eval_u_minus.get_value(q);
                    const auto u_minus_grad = eval_u_minus.get_gradient(q);

                    const auto grad_flux  = (u_minus - g) * (u_minus * normal);
                    const auto value_flux = -u_minus_grad * (u_minus - g);

                    eval_p_minus.submit_value(value_flux * normal, q);
                    eval_p_minus.submit_gradient(grad_flux, q);
                  }
                else
                  {
                    const auto u      = eval_u_minus.get_value(q);
                    const auto grad_u = eval_u_minus.get_gradient(q);

                    const auto convective_value_flux = (grad_u * (g - u)) * normal;

                    eval_p_minus.submit_value(convective_value_flux, q);
                    eval_p_minus.submit_gradient({}, q);
                  }
              }

            eval_p_minus.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
          }
        else
          {
            eval_p_minus.reinit(face);
            eval_u_minus.reinit(face);

            eval_u_minus.gather_evaluate(src, EvaluationFlags::values);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
              {
                const auto normal = eval_p_minus.normal_vector(q);

                exact_velocity.set_time(time);
                const auto grad_u =
                  evaluate_tensor_function(exact_velocity,
                                           eval_p_minus.quadrature_point(q));
                const auto u_plus          = eval_u_minus.get_value(q);
                const auto convective_flux = (grad_u * u_plus) * normal;

                eval_p_minus.submit_value(convective_flux, q);
              }

            eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
          }
      }
  }

  void
  local_divergence_domain(const MatrixFree<dim, number>               &data,
                          VectorType                                  &dst,
                          const VectorType                            &src,
                          const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, 1, number>   eval_p(data, 1, 1);
    FEEvaluation<dim, -1, 0, dim, number> eval_u(data, 0, 1);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_p.reinit(cell);
        eval_u.reinit(cell);

        eval_u.gather_evaluate(src, EvaluationFlags::gradients);

        // loop over quadrature points and compute the local volume flux
        for (const unsigned int q : eval_p.quadrature_point_indices())
          {
            const auto div_u = eval_u.get_divergence(q);
            eval_p.submit_value(div_u, q);
          }

        // multiply by nabla v^h(x) and sum
        eval_p.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  void
  local_divergence_inner_face(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    if (!is_dg)
      return;
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_minus(data, true, 1, 1);
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_plus(data, false, 1, 1);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_minus(data, true, 0, 1);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_plus(data, false, 0, 1);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_p_minus.reinit(face);
        eval_p_plus.reinit(face);
        eval_u_minus.reinit(face);
        eval_u_plus.reinit(face);

        eval_u_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_u_plus.gather_evaluate(src, EvaluationFlags::values);

        for (const unsigned int q : eval_p_minus.quadrature_point_indices())
          {
            const auto normal = eval_p_minus.normal_vector(q);
            const auto div_factor =
              -0.5 * (eval_u_minus.get_value(q) - eval_u_plus.get_value(q)) * normal;

            eval_p_minus.submit_value(div_factor, q);
            eval_p_plus.submit_value(div_factor, q);
          }

        eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
        eval_p_plus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  void
  local_divergence_boundary_face(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_minus(data, true, 1, 1);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_minus(data, true, 0, 1);

    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(time);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        if (data.get_boundary_id(face) == 0 || data.get_boundary_id(face) == 2)
          {
            eval_p_minus.reinit(face);
            eval_u_minus.reinit(face);

            eval_u_minus.gather_evaluate(src, EvaluationFlags::values);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
              {
                const auto normal = eval_p_minus.normal_vector(q);

                exact_velocity.set_time(time);
                const auto g =
                  evaluate_function(exact_velocity, eval_p_minus.quadrature_point(q));
                const auto u     = eval_u_minus.get_value(q);
                const auto div_u = -(u - g) * normal;

                eval_p_minus.submit_value(div_u, q);
              }

            eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
          }
        else
          {
            eval_p_minus.reinit(face);
            eval_u_minus.reinit(face);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
              {
                eval_p_minus.submit_value({}, q);
              }

            eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
          }
      }
  }
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
    ReductionControl coarse_grid_solver_control(10000, 1e-20, 1e-4, false, false);
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
  MultigridPreconditioner(SystemMatrixType     &pressure_operator,
                          const unsigned int    mapping_degree,
                          const number_operator time_step,
                          const unsigned int    bdf_order)
  {
    const auto &dof_handler =
      pressure_operator.get_matrix_free().get_dof_handler(dof_no_p);

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

        mg_matrices[level].reinit(mg_matrices_mf[level], bdf_order, time_step);
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
        std::vector<std::vector<bool>> constant_modes;
        DoFTools::extract_constant_modes(dof_handlers[minlevel],
                                         ComponentMask(),
                                         constant_modes);
        amg_data.constant_modes = constant_modes;
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

    SolverControl              control(100000, 1e-12 * vec_p_rhs.l2_norm());
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

  pcout << "Setup tria" << std::endl;
  GridGenerator::subdivided_hyper_cube(tria, 4, -L * numbers::PI, L * numbers::PI);
  pcout << "Setup boundary" << std::endl;

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
    dealii::Utilities::MPI::min(local_time_step, MPI_COMM_WORLD); // time_step_size
  pcout << "Time step size: " << time_step << std::endl;

  const unsigned int bdf_order   = 2;
  const unsigned int bdf_order_p = 2;

  MomentumOperator<dim, dim, Number> momentum_op;
  // set up operator
  momentum_op.reinit(mapping, dof_handler_u, dof_handler_p, time_step, bdf_order);

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
    momentum_op, mapping.get_degree(), time_step, bdf_order);

  DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>>
    preconditioner_velocity_pointjacobi;
  DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>>
    preconditioner_pressure_pointjacobi;


  PressureOperator<dim, Number> pressure_op;
  pressure_op.reinit(momentum_op.get_matrix_free(), bdf_order, time_step);
  pressure_op.set_time_step(time_step);

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
                                                                bdf_order);

  Number current_time = 0;

  AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
  AnalyticalSolutionPressure<dim> exact_pressure(u_x_max, viscosity);

  exact_velocity.set_time(current_time);
  VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u_old[0]);
  exact_pressure.set_time(current_time);
  VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p);

  {
    Number energy;
    Number enstrophy;
    Number dissipation;
    Number max_vorticity;

    momentum_op.calculate_energy(
      vec_u_old[0], energy, enstrophy, dissipation, max_vorticity);
    pcout << "Energy, enstrophy, dissipation, max vorticity at t=0: " << energy << " "
          << enstrophy << " " << dissipation << " " << max_vorticity << std::endl;
  }

  const Number       end_time         = 20.0;
  const unsigned int output_interval  = 1;
  unsigned int       time_step_number = 0;

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
      SolverControl control(100000, 1e-12 * vec_p_rhs.l2_norm());
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

      SolverControl control_mom(10000, 1e-12 * vec_u_rhs.l2_norm());
      SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver_mom(control_mom);
      vec_u.swap(speed_extrapolated); // = 0.;
      unsigned int n_iterations_vel;
      if (use_mg_velocity)
        {
          n_iterations_vel = preconditioner_velocity.solve(momentum_op, vec_u, vec_u_rhs);
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

          if (false)
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
              data_out.build_patches(mapping,
                                     fe_u.degree,
                                     DataOut<dim>::curved_inner_cells);

              const std::string filename =
                "solution-taylor_green-" + std::to_string(time_step_number) + ".vtu";
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
  }

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

  for (unsigned int i = 3; i < 7; ++i)
    do_test<3, double>(8, 3, i);
  // do_test<3, double>(5, 5, 3);
}
