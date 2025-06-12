
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

#include <fstream>

using namespace dealii;


const bool use_extrapolated_velocity           = false;
const bool use_pressure_convective_upwind_flux = false;

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
    order = current_order;
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
    return alpha[i];
  }
  double
  get_beta(const unsigned int i)
  {
    return beta[i];
  }
  unsigned int
  get_order()
  {
    return order;
  }

  std::vector<double> alpha;
  std::vector<double> beta;
  double              gamma0;
  unsigned int        order;
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

    const unsigned int fe_degree_u     = dof_handler_u.get_fe().degree;
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
        eval_u.read_dof_values(
          dst); // TODO: check if we have to copy the vector here first
        mass_inv.apply(eval_u.begin_dof_values(), eval_u.begin_dof_values());
        eval_u.set_dof_values(dst);
      }
  }


  const MatrixFree<dim, number> &
  get_matrix_free() const
  {
    return matrix_free;
  }


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
            const auto time_deriv =
              make_vectorized_array<number>(gamma0 / time_step) * integrator.get_value(q);

            const auto grad_u = integrator.get_gradient(q);
            const auto speed  = speeds_cells(cell, q);

            const auto convective_flux = grad_u * speed;

            integrator.submit_value(time_deriv + convective_flux, q);
            integrator.submit_gradient(make_vectorized_array<number>(viscosity) * grad_u,
                                       q);
          }
        integrator.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                     dst);
      }
  }


  void
  do_face_integral_range(const MatrixFree<dim, number>               &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &range) const
  {
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


        for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
          {
            const Tensor<1, dim, VectorizedArray<number>> solution_jump =
              (integrator_inner.get_value(q) - integrator_outer.get_value(q));

            const Tensor<1, dim, VectorizedArray<number>> solution_average =
              make_vectorized_array<number>(0.5) *
              (integrator_inner.get_value(q) + integrator_outer.get_value(q));

            const auto speed_normal = speeds_faces(face, q);
            const auto convective_flux =
              speed_normal * solution_average +
              make_vectorized_array<number>(0.5) * std::abs(speed_normal) * solution_jump;

            const auto convective_flux_inner =
              convective_flux - speed_normal * integrator_inner.get_value(q);
            const auto convective_flux_outer =
              -convective_flux + speed_normal * integrator_outer.get_value(q);

            const Tensor<1, dim, VectorizedArray<number>> averaged_normal_derivative =
              make_vectorized_array<number>(viscosity) *
              (integrator_inner.get_normal_derivative(q) +
               integrator_outer.get_normal_derivative(q)) *
              number(0.5);

            const Tensor<1, dim, VectorizedArray<number>> test_by_value =
              make_vectorized_array<number>(viscosity) * solution_jump * sigma -
              averaged_normal_derivative;


            integrator_inner.submit_value(test_by_value + convective_flux_inner, q);
            integrator_outer.submit_value(-test_by_value + convective_flux_outer, q);

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
  do_boundary_integral_range(const MatrixFree<dim, number>               &matrix_free,
                             VectorType                                  &dst,
                             const VectorType                            &src,
                             const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_inner(matrix_free,
                                                                        true,
                                                                        dof_no_v,
                                                                        quad_no_v);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_inner.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);


        const VectorizedArray<number> sigma =
          integrator_inner.read_cell_data(array_penalty_parameter) * get_penalty_factor();

        // Dirichlet boundary
        if (matrix_free.get_boundary_id(face) == 0)
          {
            for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
              {
                const Tensor<1, dim, VectorizedArray<number>> u_inner =
                  make_vectorized_array<number>(viscosity) *
                  integrator_inner.get_value(q);

                const Tensor<1, dim, VectorizedArray<number>> normal_derivative_inner =
                  make_vectorized_array<number>(viscosity) *
                  integrator_inner.get_normal_derivative(q);

                const Tensor<1, dim, VectorizedArray<number>> test_by_value =
                  number(2.0) * u_inner * sigma - normal_derivative_inner;

                const auto speed = speeds_faces(face, q);
                const auto convective_flux =
                  (std::abs(speed) - speed) * integrator_inner.get_value(q);

                integrator_inner.submit_normal_derivative(-u_inner, q);
                integrator_inner.submit_value(test_by_value + convective_flux, q);
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
                integrator_inner.submit_value(Tensor<1, dim, VectorizedArray<number>>(),
                                              q);
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

            speeds_faces(face, q) = make_vectorized_array<number>(0.5) *
                                    (integrator_speed_inner.get_value(q) +
                                     integrator_speed_outer.get_value(q)) *
                                    normal;


            // const Tensor<1, dim, VectorizedArray<number>> p_avg =
            // number(0.5)*(integrator_inner_p.get_value(q) +
            // integrator_outer_p.get_value(q)) * integrator_inner.normal_vector(q);
            const Tensor<1, dim, VectorizedArray<number>> p_jump =
              number(0.5) *
              (integrator_inner_p.get_value(q) - integrator_outer_p.get_value(q)) *
              integrator_inner.normal_vector(q);


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

        if (matrix_free.get_boundary_id(face) == 0)
          {
            for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
              {
                const auto normal = integrator_inner.normal_vector(q);

                const auto g =
                  evaluate_function(exact_velocity, integrator_inner.quadrature_point(q));

                VectorizedArray<number> speed_normal;

                if (use_extrapolated_velocity)
                  {
                    const auto u_plus_m =
                      evaluate_function(exact_velocity_m,
                                        integrator_inner.quadrature_point(q));
                    const auto u_plus_m2 =
                      evaluate_function(exact_velocity_m2,
                                        integrator_inner.quadrature_point(q));
                    auto extrapolated_velocity = 2.0 * u_plus_m - u_plus_m2;
                    speed_normal = 0.5 * (normal * (integrator_speed_inner.get_value(q) +
                                                    extrapolated_velocity));
                  }
                else
                  {
                    speed_normal = make_vectorized_array<number>(0.5) *
                                   (integrator_speed_inner.get_value(q) + g) * normal;
                  }

                speeds_faces(face, q) = speed_normal;

                const auto convective_flux = (std::abs(speed_normal) - speed_normal) * g;

                const auto value_flux =
                  make_vectorized_array<number>(2.0 * viscosity) * sigma * g;
                const auto gradient_flux = make_vectorized_array<number>(viscosity) * g;

                const Tensor<1, dim, VectorizedArray<number>> p =
                  0. * integrator_inner.normal_vector(q);
                // integrator_inner_p.get_value(q) * integrator_inner.normal_vector(q);

                integrator_inner.submit_normal_derivative(-gradient_flux, q);
                integrator_inner.submit_value(value_flux - p + convective_flux, q);
              }
          }
        else if (matrix_free.get_boundary_id(face) == 1)
          {
            for (const unsigned int q : integrator_inner.quadrature_point_indices())
              {
                const auto normal = integrator_inner.get_normal_vector(q);

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

  MatrixFree<dim, number> matrix_free;

  Number penalty_factor;

  number viscosity;

  number time_step;

  number time;

  unsigned int bdf_order;

  dealii::AlignedVector<dealii::VectorizedArray<Number>>    array_penalty_parameter;
  mutable Table<2, Tensor<1, dim, VectorizedArray<number>>> speeds_cells;
  mutable Table<2, VectorizedArray<number>>                 speeds_faces;
};

template <int dim, typename number>
class PressureOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;

  PressureOperator() = default;

  void
  reinit(const MatrixFree<dim, number> &matrix_free, const unsigned int bdf_order_in)
  {
    bdf_order                    = bdf_order_in;
    this->matrix_free            = &matrix_free;
    const unsigned int fe_degree = matrix_free.get_dof_handler(dof_no_p).get_fe().degree;
    const double       penalty_factor = 1.0 * (fe_degree + 1) * (fe_degree);
    {
      unsigned int n_cells =
        matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
      array_penalty_parameter.resize(n_cells);

      const dealii::FiniteElement<dim> &fe =
        matrix_free.get_dof_handler(dof_no_p).get_fe();
      const auto reference_cells =
        matrix_free.get_dof_handler(dof_no_p).get_fe().reference_cell();
      MappingQ1<dim> mapping;

      const auto quadrature =
        reference_cells.template get_gauss_type_quadrature<dim>(fe_degree + 1);
      dealii::FEValues<dim> fe_values(mapping, fe, quadrature, dealii::update_JxW_values);

      const auto face_quadrature =
        reference_cells.face_reference_cell(0)
          .template get_gauss_type_quadrature<dim - 1>(fe_degree + 1);
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
                matrix_free.get_cell_iterator(i, v, dof_no_p);
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
  compute_rhs(VectorType &dst, const VectorType &velocity, const VectorType &vorticity)
  {
    matrix_free->loop(&PressureOperator::local_rhs_domain,
                      &PressureOperator::local_rhs_inner_face,
                      &PressureOperator::local_rhs_boundary_face,
                      this,
                      dst,
                      std::vector<const VectorType *>{&velocity, &vorticity},
                      true,
                      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

  void
  set_time_step(const double t)
  {
    time_step = t;
  }

private:
  const MatrixFree<dim, number>                         *matrix_free;
  dealii::AlignedVector<dealii::VectorizedArray<number>> array_penalty_parameter;
  number                                                 time;
  double                                                 time_step;
  unsigned int                                           bdf_order;

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

        if (data.get_boundary_id(face) == 0)
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
  local_rhs_domain(const MatrixFree<dim, number>               &data,
                   VectorType                                  &dst,
                   const std::vector<const VectorType *>       &src,
                   const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, 1, number>   eval_p(data, 1, 1);
    FEEvaluation<dim, -1, 0, dim, number> eval_u(data, 0, 1);

    AnalyticalRHS<dim> rhs(u_x_max, viscosity);
    rhs.set_time(time);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_p.reinit(cell);
        eval_u.reinit(cell);

        eval_u.gather_evaluate(*src[0],
                               EvaluationFlags::values | EvaluationFlags::gradients);

        // loop over quadrature points and compute the local volume flux
        for (const unsigned int q : eval_p.quadrature_point_indices())
          {
            const auto f = evaluate_function(rhs, eval_p.quadrature_point(q));

            const auto convective_flux = eval_u.get_gradient(q) * eval_u.get_value(q);

            eval_p.submit_gradient(f - convective_flux, q);
          }

        // multiply by nabla v^h(x) and sum
        eval_p.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  void
  local_rhs_inner_face(const MatrixFree<dim, number>               &data,
                       VectorType                                  &dst,
                       const std::vector<const VectorType *>       &src,
                       const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_minus(data, true, 1, 1);
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_plus(data, false, 1, 1);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_minus(data, true, 0, 1);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_plus(data, false, 0, 1);

    AnalyticalRHS<dim> rhs(u_x_max, viscosity);
    rhs.set_time(time);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_p_minus.reinit(face);
        eval_p_plus.reinit(face);
        eval_u_minus.reinit(face);
        eval_u_plus.reinit(face);

        eval_u_minus.gather_evaluate(*src[0],
                                     EvaluationFlags::values |
                                       EvaluationFlags::gradients);
        eval_u_plus.gather_evaluate(*src[0],
                                    EvaluationFlags::values | EvaluationFlags::gradients);

        for (const unsigned int q : eval_p_minus.quadrature_point_indices())
          {
            const auto f = evaluate_function(rhs, eval_p_minus.quadrature_point(q));

            const auto normal = eval_p_minus.normal_vector(q);

            const auto flux = f * normal;

            if (use_pressure_convective_upwind_flux)
              {
                const auto u_minus      = eval_u_minus.get_value(q);
                const auto u_plus       = eval_u_plus.get_value(q);
                const auto u_minus_grad = eval_u_minus.get_gradient(q);
                const auto u_plus_grad  = eval_u_plus.get_gradient(q);
                const auto value_flux =
                  0.5 * (u_minus_grad + u_plus_grad) * 0.5 * (u_minus + u_plus);

                eval_p_minus.submit_value(value_flux * normal - f * normal, q);
                eval_p_plus.submit_value(-value_flux * normal + f * normal, q);


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

                eval_p_minus.submit_value(convective_flux - flux, q);
                eval_p_plus.submit_value(flux - convective_flux, q);
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
  local_rhs_boundary_face(const MatrixFree<dim, number>               &data,
                          VectorType                                  &dst,
                          const std::vector<const VectorType *>       &src,
                          const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_minus(data, true, 1, 1);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_vorticity(data, true, 0, 1);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_minus(data, true, 0, 1);

    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(time);
    AnalyticalSolutionPressure<dim> exact_pressure(u_x_max, viscosity);
    exact_pressure.set_time(time);
    AnalyticalRHS<dim> rhs(u_x_max, viscosity);
    rhs.set_time(time);

    BDFTimeIntegratorConstants integration_constants(bdf_order);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        if (data.get_boundary_id(face) == 0)
          {
            eval_p_minus.reinit(face);
            eval_vorticity.reinit(face);
            eval_u_minus.reinit(face);

            eval_vorticity.gather_evaluate(*src[1], EvaluationFlags::gradients);
            eval_u_minus.gather_evaluate(*src[0],
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);

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

                if (false)
                  {
                    curl_omega =
                      make_vectorized_array<number>(4.0 * numbers::PI * numbers::PI) * g;
                  }

                const auto curl_flux = (-viscosity) * normal * curl_omega;

                if (use_pressure_convective_upwind_flux)
                  {
                    const auto u_minus      = eval_u_minus.get_value(q);
                    const auto u_minus_grad = eval_u_minus.get_gradient(q);

                    const auto grad_flux = (u_minus - g) * (u_minus * normal);
                    const auto value_flux =
                      -u_plus - u_minus_grad * (u_minus - g) - viscosity * curl_omega;

                    eval_p_minus.submit_value(value_flux * normal, q);
                    eval_p_minus.submit_gradient(grad_flux, q);
                  }
                else
                  {
                    const auto u      = eval_u_minus.get_value(q);
                    const auto grad_u = eval_u_minus.get_gradient(q);

                    const auto convective_value_flux = (grad_u * (g - u)) * normal;

                    eval_p_minus.submit_value(flux + curl_flux + convective_value_flux,
                                              q);
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

            eval_u_minus.gather_evaluate(*src[0], EvaluationFlags::values);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
              {
                const auto f = evaluate_function(rhs, eval_p_minus.quadrature_point(q));

                const auto normal = eval_p_minus.normal_vector(q);

                const auto flux = -f * normal;

                exact_velocity.set_time(time);
                const auto grad_u =
                  evaluate_tensor_function(exact_velocity,
                                           eval_p_minus.quadrature_point(q));
                const auto u_plus          = eval_u_minus.get_value(q);
                const auto convective_flux = (grad_u * u_plus) * normal;


                const auto g_p =
                  evaluate_scalar_function(exact_pressure,
                                           eval_p_minus.quadrature_point(q));

                const VectorizedArray<number> penalty_factor =
                  eval_p_minus.read_cell_data(array_penalty_parameter);
                const auto penalty = penalty_factor * 2. * g_p;

                eval_p_minus.submit_value(flux + convective_flux + penalty, q);
                eval_p_minus.submit_normal_derivative(-g_p, q);
              }

            eval_p_minus.integrate_scatter(EvaluationFlags::values |
                                             EvaluationFlags::gradients,
                                           dst);
          }
      }
  }
};

template <int dim, typename Number>
void
do_test(const unsigned int fe_degree, const unsigned int n_refinements)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  FESystem<dim>  fe_u(FE_DGQ<dim>(fe_degree), dim);
  FE_DGQ<dim>    fe_p(fe_degree - 1);
  MappingQ1<dim> mapping;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  double L = 1.;
  GridGenerator::hyper_cube(tria, -L / 2., L / 2.);

  const bool use_neumann_boundary = false;
  if (use_neumann_boundary)
    tria.begin()->face(0)->set_all_boundary_ids(1);
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
  const double local_time_step = 0.1 / std::pow(fe_degree, 1.5) * h_min / u_x_max;
  const Number time_step =
    std::min(5.0 * 1e-5, dealii::Utilities::MPI::min(local_time_step, MPI_COMM_WORLD));
  pcout << "Time step size: " << time_step << std::endl;

  unsigned int               bdf_order = 2;
  BDFTimeIntegratorConstants bdf(bdf_order);

  MomentumOperator<dim, dim, Number> momentum_op;
  // set up operator
  momentum_op.reinit(mapping, dof_handler_u, dof_handler_p, time_step, bdf_order);

  momentum_op.set_viscosity(viscosity);
  momentum_op.set_time(0.0);

  LinearAlgebra::distributed::Vector<Number> vec_u, vec_u_deriv, vec_u_rhs, vec_p,
    vec_p_extrapolated, vec_u_norm, speed_extrapolated, vec_vorticity, vec_p_rhs,
    vec_p_norm;
  momentum_op.initialize_dof_vector(vec_u, dof_no_v);
  momentum_op.initialize_dof_vector(vec_u_deriv, dof_no_v);
  momentum_op.initialize_dof_vector(vec_u_rhs, dof_no_v);
  momentum_op.initialize_dof_vector(vec_p, dof_no_p);
  momentum_op.initialize_dof_vector(vec_p_extrapolated, dof_no_p);
  momentum_op.initialize_dof_vector(vec_u_norm, dof_no_v);
  momentum_op.initialize_dof_vector(speed_extrapolated, dof_no_v);
  momentum_op.initialize_dof_vector(vec_vorticity, dof_no_v);
  momentum_op.initialize_dof_vector(vec_p_rhs, dof_no_p);
  momentum_op.initialize_dof_vector(vec_p_norm, dof_no_p);

  std::vector<LinearAlgebra::distributed::Vector<double>> vec_u_old(bdf_order);
  std::vector<LinearAlgebra::distributed::Vector<double>> vec_p_old(bdf_order);
  for (auto &vec : vec_u_old)
    momentum_op.initialize_dof_vector(vec, dof_no_v);
  for (auto &vec : vec_p_old)
    momentum_op.initialize_dof_vector(vec, dof_no_p);

  PressureOperator<dim, double> pressure_op;
  pressure_op.reinit(momentum_op.get_matrix_free(), bdf_order);
  pressure_op.set_time_step(time_step);

  Number current_time = 0;

  AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
  AnalyticalSolutionPressure<dim> exact_pressure(u_x_max, viscosity);

  for (unsigned int i = 0; i < bdf_order; ++i)
    {
      exact_velocity.set_time(current_time);
      VectorTools::interpolate(mapping,
                               dof_handler_u,
                               exact_velocity,
                               vec_u_old[bdf_order - 1 - i]);

      exact_pressure.set_time(current_time);
      VectorTools::interpolate(mapping,
                               dof_handler_p,
                               exact_pressure,
                               vec_p_old[bdf_order - 1 - i]);

      current_time += time_step;
    }

  current_time -= time_step;
  const Number end_time         = 1.0;
  unsigned int time_step_number = bdf.get_order() - 1;

  const bool write_output = false;
  while (current_time <= end_time)
    {
      current_time += time_step;
      ++time_step_number;

      pressure_op.set_time(current_time);
      momentum_op.set_time(current_time);

      // Momentum step
      vec_u_deriv        = 0.;
      speed_extrapolated = 0.;
      vec_p_extrapolated = 0.;

      for (unsigned int i = 0; i < bdf.get_order(); ++i)
        {
          vec_u_deriv.add(bdf.get_alpha(i) / time_step, vec_u_old[i]);
          speed_extrapolated.add(bdf.get_beta(i), vec_u_old[i]);
          vec_p_extrapolated.add(bdf.get_beta(i), vec_p_old[i]);
        }

      vec_u_rhs = 0.;
      momentum_op.rhs(vec_u_rhs, vec_u_deriv, speed_extrapolated, vec_p_extrapolated);

      SolverControl control_mom(10000, 1e-12 * vec_u_rhs.l2_norm());
      SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver_mom(control_mom);
      vec_u.swap(speed_extrapolated); // = 0.;
      solver_mom.solve(momentum_op, vec_u, vec_u_rhs, PreconditionIdentity());
      if (write_output)
        pcout << "Momentum solver: " << control_mom.last_step() << " iterations"
              << std::endl;

      // exact_velocity.set_time(current_time);
      // VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u);

      // Pressure step
      vec_vorticity = 0.;
      momentum_op.evaluate_vorticity(vec_vorticity, vec_u);
      pressure_op.compute_rhs(vec_p_rhs, vec_u, vec_vorticity);
      if (!use_neumann_boundary)
        VectorTools::subtract_mean_value(vec_p_rhs);
      SolverControl control(10000, 1e-12 * vec_p_rhs.l2_norm());
      SolverCG<LinearAlgebra::distributed::Vector<double>> solver(control);
      vec_p.swap(vec_p_extrapolated); // = 0.;
      solver.solve(pressure_op, vec_p, vec_p_rhs, PreconditionIdentity());
      if (!use_neumann_boundary)
        VectorTools::subtract_mean_value(vec_p);
      if (write_output)
        pcout << "Pressure solver: " << control.last_step() << " iterations" << std::endl;

      // exact_pressure.set_time(current_time);
      // VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p);

      for (unsigned int i = bdf.get_order() - 1; i != 0; --i)
        {
          std::swap(vec_u_old[i], vec_u_old[i - 1]);
          std::swap(vec_p_old[i], vec_p_old[i - 1]);
        }

      vec_u_old[0].swap(vec_u);
      vec_p_old[0].swap(vec_p);

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
                                            vec_p_old[0],
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
          data_out.add_data_vector(dof_handler_p, vec_p_old[0], "pressure");
          VectorTools::interpolate(mapping,
                                   dof_handler_p,
                                   exact_pressure,
                                   vec_p_extrapolated);
          data_out.add_data_vector(dof_handler_p,
                                   vec_p_extrapolated,
                                   "pressure_analytical");
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
                                    vec_p_old[0],
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
  pcout << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  for (unsigned int i = 2; i < 7; ++i)
    do_test<2, double>(3, i);

  for (unsigned int i = 1; i < 7; ++i)
    do_test<2, double>(5, i);
}