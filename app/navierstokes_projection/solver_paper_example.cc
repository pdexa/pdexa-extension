
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>


namespace NavierStokes
{
  const double viscosity         = 1.;
  const double u_x_max           = 1.;
  const double factor_convective = 1.;


  using namespace dealii;
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
        const double t  = this->get_time();
        const double pi = dealii::numbers::PI;
  
        double result = 0.0;
        if (component == 0)
          result = pi * std::sin(2 * pi * p[1]) * std::sin(pi * p[0]) * std::sin(pi * p[0]) * std::sin(t);
        else if (component == 1)
          result = -pi * std::sin(2 * pi * p[0]) * std::sin(pi * p[1]) * std::sin(pi * p[1]) * std::sin(t);
        
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
  
        const double result = std::cos(pi * p[0]) * std::sin(pi * p[1]) * std::sin(t);
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
      const double t  = this->get_time();
      const double pi = dealii::numbers::PI;

      const double x = p[0];
      const double y = p[1];

      double result = 0.0;
      if (component == 0)
        result = pi*(4.0*pi*pi*std::sin(t)*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*y)*std::cos(pi*x) + 16.0*pi*pi*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::cos(pi*y) - 1.0*std::sin(t)*std::sin(pi*x) - 4.0*pi*pi*std::sin(t)*std::cos(pi*y) + 2.0*std::sin(pi*x)*std::sin(pi*x)*std::cos(t)*std::cos(pi*y))*std::sin(pi*y);

      else if (component == 1)
        result = pi*(4.0*pi*pi*std::sin(t)*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::sin(pi*y)*std::cos(pi*y) - 16.0*pi*pi*std::sin(t)*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::cos(pi*x) + 4.0*pi*pi*std::sin(t)*std::sin(pi*x)*std::cos(pi*x) + 1.0*std::sin(t)*std::cos(pi*x)*std::cos(pi*y) - 2.0*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::cos(t)*std::cos(pi*x));

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



  template <int dim, typename Number>
  class MomentumOperator
  {
  public:
    using VectorType   = LinearAlgebra::distributed::Vector<Number>;
    MomentumOperator() = default;

    void
    reinit(const DoFHandler<dim> &dof_handler_u, const DoFHandler<dim> &dof_handler_p);

    void
    initialize_dof_vector(VectorType &vec)
    {
      data.initialize_dof_vector(vec);
    }

    const MatrixFree<dim, Number> &
    get_matrix_free() const
    {
      return data;
    }

    void
    set_time_factor(const double time_factor)
    {
      this->time_factor = time_factor;
    }

    void
    set_time(const double time)
    {
      this->time = time;
    }

    void
    set_time_step(const double t)
    {
      time_step = t;
    }

    void
    compute_rhs(VectorType       &dst,
                const VectorType &u_rhs_time_derivative,
                const VectorType &u_extrapolated,
                const VectorType &pressure)
    {
      data.loop(&MomentumOperator::local_rhs_domain,
                &MomentumOperator::local_rhs_inner_face,
                &MomentumOperator::local_rhs_boundary_face,
                this,
                dst,
                std::vector<const VectorType *>{&u_rhs_time_derivative,
                                                &u_extrapolated,
                                                &pressure},
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::gradients,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      data.loop(&MomentumOperator::local_apply_domain,
                &MomentumOperator::local_apply_inner_face,
                &MomentumOperator::local_apply_boundary_face,
                this,
                dst,
                src,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::gradients,
                MatrixFree<dim, Number>::DataAccessOnFaces::gradients);
    }

    void
    evaluate_divergence(VectorType &dst, const VectorType &src) const
    {
      data.loop(&MomentumOperator::local_divergence_domain,
                &MomentumOperator::local_divergence_inner_face,
                &MomentumOperator::local_divergence_boundary_face,
                this,
                dst,
                src,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    void
    precondition_block_jacobi(VectorType &dst, const VectorType &src) const;

    void
    project_initial(VectorType &dst) const;

  private:
    MatrixFree<dim, Number> data;
    Number                  time;
    Number                  time_factor;
    Number time_step;

    Table<2, Tensor<1, dim, VectorizedArray<Number>>> speeds_cells;
    Table<2, VectorizedArray<Number>>                 speeds_faces;
    AlignedVector<VectorizedArray<Number>>            penalty_factors;

    void
    local_apply_domain(const MatrixFree<dim, Number>               &data,
                       VectorType                                  &dst,
                       const VectorType                            &src,
                       const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    local_apply_inner_face(const MatrixFree<dim, Number>               &data,
                           VectorType                                  &dst,
                           const VectorType                            &src,
                           const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    local_apply_boundary_face(
      const MatrixFree<dim, Number>               &data,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    local_rhs_domain(const MatrixFree<dim, Number>               &data,
                     VectorType                                  &dst,
                     const std::vector<const VectorType *>       &src,
                     const std::pair<unsigned int, unsigned int> &cell_range);

    void
    local_rhs_inner_face(const MatrixFree<dim, Number>               &data,
                         VectorType                                  &dst,
                         const std::vector<const VectorType *>       &src,
                         const std::pair<unsigned int, unsigned int> &cell_range);
    void
    local_rhs_boundary_face(const MatrixFree<dim, Number>               &data,
                            VectorType                                  &dst,
                            const std::vector<const VectorType *>       &src,
                            const std::pair<unsigned int, unsigned int> &cell_range);

    void
    local_divergence_domain(
      const MatrixFree<dim, Number>               &data,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    local_divergence_inner_face(
      const MatrixFree<dim, Number>               &data,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    local_divergence_boundary_face(
      const MatrixFree<dim, Number>               &data,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;
  };



  template <int dim, typename Number>
  void
  MomentumOperator<dim, Number>::reinit(const DoFHandler<dim> &dof_handler_u,
                                        const DoFHandler<dim> &dof_handler_p)
  {
    const unsigned int   fe_degree = dof_handler_u.get_fe().degree;
    MappingQGeneric<dim> mapping(fe_degree);
    Quadrature<1>        quadrature      = QGauss<1>(fe_degree + 2);
    Quadrature<1>        quadrature_mass = QGauss<1>(fe_degree + 1);
    Quadrature<1>        quadrature_p    = QGauss<1>(dof_handler_p.get_fe().degree + 1);
    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.overlap_communication_computation = false;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points | update_values);
    additional_data.mapping_update_flags_inner_faces =
      (update_JxW_values | update_normal_vectors | update_quadrature_points |
       update_values);
    additional_data.mapping_update_flags_boundary_faces =
      (update_JxW_values | update_normal_vectors | update_quadrature_points |
       update_values);

    AffineConstraints<double> dummy;
    dummy.close();
    data.reinit(mapping,
                std::vector<const DoFHandler<dim> *>{&dof_handler_u, &dof_handler_p},
                std::vector<const AffineConstraints<double> *>{&dummy, &dummy},
                std::vector<Quadrature<1>>{{quadrature, quadrature_mass, quadrature_p}},
                additional_data);

    {
      FEEvaluation<dim, -1, 0, dim, Number> eval_cell(data);
      speeds_cells.reinit(data.n_cell_batches(), eval_cell.n_q_points);
      FEFaceEvaluation<dim, -1, 0, dim, Number> eval_face(data, true);
      FEFaceEvaluation<dim, -1, 0, dim, Number> eval_face_out(data, false);
      speeds_faces.reinit(data.n_inner_face_batches() + data.n_boundary_face_batches(),
                          eval_face.n_q_points);
      penalty_factors.resize(data.n_inner_face_batches() +
                             data.n_boundary_face_batches());
      for (unsigned int face = 0; face < penalty_factors.size(); ++face)
        {
          eval_face.reinit(face);
          if (face < data.n_inner_face_batches())
            {
              eval_face_out.reinit(face);
              penalty_factors[face] =
                (std::abs((eval_face.normal_vector(0) *
                           eval_face.inverse_jacobian(0))[dim - 1]) +
                 std::abs((eval_face_out.normal_vector(0) *
                           eval_face_out.inverse_jacobian(0))[dim - 1])) *
                (Number)(std::max(fe_degree, 1u) * (fe_degree + 1.0));
            }
          else
            penalty_factors[face] = 2.0 *
                                    std::abs((eval_face.normal_vector(0) *
                                              eval_face.inverse_jacobian(0))[dim - 1]) *
                                    (Number)(std::max(fe_degree, 1u) * (fe_degree + 1.0));
        }
    }

    /*
    QGauss<1>               gauss_quad(dof_handler_u.get_fe().degree + 1);
    FE_DGQArbitraryNodes<1> fe_1d(gauss_quad);
    constexpr unsigned int  n = fe_degree + 1;
    for (unsigned int c = 0; c < 2; ++c)
      {
        LAPACKFullMatrix<double> deriv_matrix(n, n);
        for (unsigned int q = 0; q < n; ++q)
          {
            for (unsigned int i = 0; i < n; ++i)
              for (unsigned int j = 0; j < n; ++j)
                deriv_matrix(i, j) -= fe_1d.shape_grad(i, gauss_quad.point(q))[0] *
                                      fe_1d.shape_value(j, gauss_quad.point(q)) *
                                      gauss_quad.weight(q);
          }
        const double sign_advection = (c == 0) ? 1.0 : -1.0;
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            deriv_matrix(i, j) +=
              -fe_1d.shape_value(i, Point<1>()) * fe_1d.shape_value(j, Point<1>()) *
                (0.5 - 0.5 * sign_advection) +
              fe_1d.shape_value(i, Point<1>(1.0)) * fe_1d.shape_value(j, Point<1>(1.0)) *
                (0.5 + 0.5 * sign_advection);

        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            deriv_matrix(i, j) *= (1. / gauss_quad.weight(i));
        deriv_matrix.compute_eigenvalues(true, false);

        eigenvalues[c].resize(n);
        for (unsigned int i = 0; i < n; ++i)
          eigenvalues[c][i] = deriv_matrix.eigenvalue(i);

        eigenvectors[c]         = deriv_matrix.get_right_eigenvectors();
        inverse_eigenvectors[c] = eigenvectors[c];
        inverse_eigenvectors[c].gauss_jordan();
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            inverse_eigenvectors[c](i, j) *= (1. / gauss_quad.weight(j));
      }
    */
  }



  template <int dim, typename Number>
  void
  MomentumOperator<dim, Number>::local_apply_domain(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, dim, Number> eval(data);
    const Number                          factor = time_factor;

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        // compute u^h(x) from src
        eval.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        // loop over quadrature points and compute the local volume flux
        for (const unsigned int q : eval.quadrature_point_indices())
          {
            const auto                                    u     = eval.get_value(q);
            const auto                                    gradu = eval.get_gradient(q);
            const auto                                    speed = speeds_cells(cell, q);
            const Tensor<2, dim, VectorizedArray<Number>> gradient_flux =
              outer_product(u, -0.5 * speed) + viscosity * gradu;
            eval.submit_gradient(gradient_flux, q);
            const Tensor<1, dim, VectorizedArray<Number>> value_flux =
              factor * u + 0.5 * gradu * speed;
            eval.submit_value(value_flux, q);
          }

        // multiply by nabla v^h(x) and sum
        eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }



  template <int dim, typename Number>
  void
  MomentumOperator<dim, Number>::local_apply_inner_face(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, dim, Number> eval_minus(data, true);
    FEFaceEvaluation<dim, -1, 0, dim, Number> eval_plus(data, false);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);
        eval_minus.gather_evaluate(src,
                                   EvaluationFlags::values | EvaluationFlags::gradients);
        eval_plus.gather_evaluate(src,
                                  EvaluationFlags::values | EvaluationFlags::gradients);

        for (const unsigned int q : eval_minus.quadrature_point_indices())
          {
            const auto speed_normal = speeds_faces(face, q);
            const auto u_minus      = eval_minus.get_value(q);
            const auto u_plus       = eval_plus.get_value(q);

            const auto convective_flux =
              (0.5 * speed_normal) * (u_minus + u_plus) +
              std::abs(0.5 * speed_normal) * (u_minus - u_plus);
            const auto convective_flux_m =
              convective_flux - (0.5 * speed_normal) * u_minus;
            const auto convective_flux_p =
              (0.5 * speed_normal) * u_plus - convective_flux;

            const auto viscous_value_flux =
              make_vectorized_array<Number>(0.5 * viscosity) *
                (eval_minus.get_normal_derivative(q) +
                 eval_plus.get_normal_derivative(q)) -
              viscosity * penalty_factors[face] * (u_minus - u_plus);
            const auto viscous_gradient_flux =
              make_vectorized_array<Number>(0.5 * viscosity) * (u_plus - u_minus);

            eval_minus.submit_normal_derivative(viscous_gradient_flux, q);
            eval_plus.submit_normal_derivative(viscous_gradient_flux, q);

            eval_minus.submit_value(convective_flux_m - viscous_value_flux, q);
            eval_plus.submit_value(convective_flux_p + viscous_value_flux, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                     dst);
        eval_plus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                    dst);
      }
  }



  template <int dim, typename Number>
  void
  MomentumOperator<dim, Number>::local_apply_boundary_face(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, dim, Number> eval_minus(data, true);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src,
                                   EvaluationFlags::values | EvaluationFlags::gradients);

        // Dirichlet boundary
        if (data.get_boundary_id(face) == 0)
          for (const unsigned int q : eval_minus.quadrature_point_indices())
            {
              const auto speed_normal = speeds_faces(face, q);
              const auto u_minus      = eval_minus.get_value(q);

              const auto convective_flux =
                /*0.5 * speed_normal * u_minus +*/
                0.5 * std::abs(speed_normal) * u_minus
                /*- 0.5 * speed_normal * u_minus */;

              //const auto convective_flux =
              //  std::abs(speed_normal) * (u_minus) - (0.5 * speed_normal) * u_minus; //TODO: choose flux

              const auto viscous_value_flux =
                make_vectorized_array<Number>(viscosity) *
                  eval_minus.get_normal_derivative(q) -
                2.0 * viscosity * penalty_factors[face] * u_minus;
              const auto viscous_gradient_flux =
                make_vectorized_array<Number>(-viscosity) * (u_minus);

              eval_minus.submit_normal_derivative(viscous_gradient_flux, q);

              eval_minus.submit_value(convective_flux - viscous_value_flux, q);
            }
        else if (data.get_boundary_id(face) == 1)
          for (const unsigned int q : eval_minus.quadrature_point_indices())
            {
              const auto speed_normal = speeds_faces(face, q);
              const auto u_minus      = eval_minus.get_value(q);

              const auto convective_flux = speed_normal * (u_minus);
              const auto convective_flux_m =
                convective_flux - 0.5 * speed_normal * u_minus;

              eval_minus.submit_normal_derivative(
                Tensor<1, dim, VectorizedArray<Number>>()
                /*viscous_gradient_flux*/,
                q);

              eval_minus.submit_value(convective_flux_m /* + viscous_value_flux*/, q);
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



  template <int dim, typename Number>
  void
  MomentumOperator<dim, Number>::local_rhs_domain(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const std::vector<const VectorType *>       &src,
    const std::pair<unsigned int, unsigned int> &cell_range)
  {
    FEEvaluation<dim, -1, 0, dim, Number> eval_u(data, 0);
    FEEvaluation<dim, -1, 0, dim, Number> eval_u_extrap(data, 0);
    FEEvaluation<dim, -1, 0, 1, Number>   eval_p(data, 1);

    AnalyticalRHS<dim> forcing_term(u_x_max, viscosity);
    forcing_term.set_time(time);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_u.reinit(cell);
        eval_u_extrap.reinit(cell);
        eval_p.reinit(cell);

        eval_u.gather_evaluate(*src[0], EvaluationFlags::values);
        eval_u_extrap.gather_evaluate(*src[1], EvaluationFlags::values);
        eval_p.gather_evaluate(*src[2], EvaluationFlags::gradients);

        // loop over quadrature points and compute the local volume flux
        for (const unsigned int q : eval_u.quadrature_point_indices())
          {
            speeds_cells(cell, q) = factor_convective * eval_u_extrap.get_value(q);
            const auto f =
              evaluate_function(forcing_term, eval_u.quadrature_point(q));
            const auto u          = eval_u.get_value(q);
            const auto gradp      = eval_p.get_gradient(q);
            eval_u.submit_value(u - gradp + f, q);
          }

        eval_u.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, typename Number>
  void
  MomentumOperator<dim, Number>::local_rhs_inner_face(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const std::vector<const VectorType *>       &src,
    const std::pair<unsigned int, unsigned int> &face_range)
  {
    FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_minus(data, true, 0);
    FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_plus(data, false, 0);
    FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_minus(data, true, 1);
    FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_plus(data, false, 1);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_u_minus.reinit(face);
        eval_u_plus.reinit(face);
        eval_p_minus.reinit(face);
        eval_p_plus.reinit(face);

        eval_u_minus.gather_evaluate(*src[1], EvaluationFlags::values);
        eval_u_plus.gather_evaluate(*src[1], EvaluationFlags::values);
        eval_p_minus.gather_evaluate(*src[2], EvaluationFlags::values);
        eval_p_plus.gather_evaluate(*src[2], EvaluationFlags::values);

        for (const unsigned int q : eval_u_minus.quadrature_point_indices())
          {
            const auto normal = eval_u_minus.get_normal_vector(q);
            speeds_faces(face, q) =
              0.5 * factor_convective *
              (normal * (eval_u_minus.get_value(q) + eval_u_plus.get_value(q)));
            const auto p_minus       = eval_p_minus.get_value(q);
            const auto p_plus        = eval_p_plus.get_value(q);
            const auto p_jump_normal = 0.5 * normal * (p_minus - p_plus);
            eval_u_minus.submit_value(p_jump_normal, q);
            eval_u_plus.submit_value(p_jump_normal, q);
          }

        eval_u_minus.integrate_scatter(EvaluationFlags::values, dst);
        eval_u_plus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, typename Number>
  void
  MomentumOperator<dim, Number>::local_rhs_boundary_face(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const std::vector<const VectorType *>       &src,
    const std::pair<unsigned int, unsigned int> &face_range)
  {
    FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_minus(data, true, 0);
    FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_minus(data, true, 1);

    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(time);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_u_minus.reinit(face);
        eval_p_minus.reinit(face);

        eval_u_minus.gather_evaluate(*src[1], EvaluationFlags::values);
        eval_p_minus.gather_evaluate(*src[2], EvaluationFlags::values);

        // Dirichlet boundary
        for (const unsigned int q : eval_u_minus.quadrature_point_indices())
          {
            const auto normal = eval_u_minus.get_normal_vector(q);
            const auto u_plus =
              evaluate_function(exact_velocity, eval_u_minus.quadrature_point(q));

            const auto speed_normal =
              0.5 * factor_convective * (normal * (eval_u_minus.get_value(q) + u_plus));
            speeds_faces(face, q) = speed_normal;
            const auto convective_flux =
              0.5 * (-speed_normal + std::abs(speed_normal)) * u_plus; //TODO: choose flux
              //(-speed_normal + std::abs(speed_normal)) * u_plus;
            const auto viscous_value_flux =
              -2.0 * viscosity * penalty_factors[face] * u_plus;
            const auto viscous_gradient_flux =
              make_vectorized_array<Number>(-viscosity) * (u_plus);

            const auto p_minus       = eval_p_minus.get_value(q);
            const auto p_jump_normal = normal * (p_minus - p_minus);

            eval_u_minus.submit_normal_derivative(viscous_gradient_flux, q);

            eval_u_minus.submit_value(convective_flux - viscous_value_flux +
                                        p_jump_normal,
                                      q);
          }

        eval_u_minus.integrate_scatter(EvaluationFlags::values |
                                         EvaluationFlags::gradients,
                                       dst);
      }
  }



  template <int dim, typename Number>
  void
  MomentumOperator<dim, Number>::local_divergence_domain(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, dim, Number> eval_u(data, 0, 1);
    FEEvaluation<dim, -1, 0, 1, Number>   eval_p(data, 1, 1);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_p.reinit(cell);
        eval_u.reinit(cell);

        // compute u^h(x) from src
        eval_u.gather_evaluate(src, EvaluationFlags::values);

        // loop over quadrature points and compute the local volume flux
        for (const unsigned int q : eval_u.quadrature_point_indices())
          {
            const auto u = eval_u.get_value(q);
            eval_p.submit_gradient(u, q);
          }

        // multiply by nabla v^h(x) and sum
        eval_p.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }



  template <int dim, typename Number>
  void
  MomentumOperator<dim, Number>::local_divergence_inner_face(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_minus(data, true, 0, 1);
    FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_plus(data, false, 0, 1);
    FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_minus(data, true, 1, 1);
    FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_plus(data, false, 1, 1);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_p_minus.reinit(face);
        eval_p_plus.reinit(face);
        eval_u_minus.reinit(face);
        eval_u_plus.reinit(face);
        eval_u_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_u_plus.gather_evaluate(src, EvaluationFlags::values);

        for (const unsigned int q : eval_u_minus.quadrature_point_indices())
          {
            const auto u_minus = eval_u_minus.get_value(q);
            const auto u_plus  = eval_u_plus.get_value(q);
            const auto normal  = eval_u_minus.normal_vector(q);

            const auto flux = 0.5 * normal * (u_minus + u_plus);

            eval_p_minus.submit_value(-flux, q);
            eval_p_plus.submit_value(flux, q);
          }

        eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
        eval_p_plus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, typename Number>
  void
  MomentumOperator<dim, Number>::local_divergence_boundary_face(
    const MatrixFree<dim, Number> &data,
    VectorType                    &dst,
    const VectorType &,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, Number> eval_p_minus(data, true, 1, 1);

    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(time);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_p_minus.reinit(face);

        for (const unsigned int q : eval_p_minus.quadrature_point_indices())
          {
            const auto u_plus =
              evaluate_function(exact_velocity, eval_p_minus.quadrature_point(q));
            const auto normal = eval_p_minus.normal_vector(q);

            const auto flux = normal * u_plus;

            eval_p_minus.submit_value(-flux, q);
          }

        eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, typename Number>
  class PressureOperator
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    PressureOperator() = default;

    void
    reinit(const MatrixFree<dim, Number> &matrix_free)
    {
      this->matrix_free            = &matrix_free;
      const unsigned int fe_degree = matrix_free.get_dof_handler(1).get_fe().degree;
      FEFaceEvaluation<dim, -1, 0, dim, Number> eval_face(matrix_free, true);
      FEFaceEvaluation<dim, -1, 0, dim, Number> eval_face_out(matrix_free, false);
      penalty_factors.resize(matrix_free.n_inner_face_batches() +
                             matrix_free.n_boundary_face_batches());
      for (unsigned int face = 0; face < penalty_factors.size(); ++face)
        {
          eval_face.reinit(face);
          if (face < matrix_free.n_inner_face_batches())
            {
              eval_face_out.reinit(face);
              penalty_factors[face] =
                (std::abs((eval_face.normal_vector(0) *
                           eval_face.inverse_jacobian(0))[dim - 1]) +
                 std::abs((eval_face_out.normal_vector(0) *
                           eval_face_out.inverse_jacobian(0))[dim - 1])) *
                (Number)(std::max(fe_degree, 1u) * (fe_degree + 1.0));
            }
          else
            penalty_factors[face] = 2.0 *
                                    std::abs((eval_face.normal_vector(0) *
                                              eval_face.inverse_jacobian(0))[dim - 1]) *
                                    (Number)(std::max(fe_degree, 1u) * (fe_degree + 1.0));
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
                        MatrixFree<dim, Number>::DataAccessOnFaces::gradients,
                        MatrixFree<dim, Number>::DataAccessOnFaces::gradients);
    }

    void
    compute_rhs(VectorType &dst, const VectorType &src)
    {
      matrix_free->loop(&PressureOperator::local_rhs_domain,
                        &PressureOperator::local_rhs_inner_face,
                        &PressureOperator::local_rhs_boundary_face,
                        this,
                        dst,
                        src,
                        true,
                        MatrixFree<dim, Number>::DataAccessOnFaces::gradients,
                        MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    void
    update_pressure(VectorType &pressure, const VectorType &velocity)
    {
      FEEvaluation<dim, -1, 0, 1, Number> eval_p(*matrix_free, 1, 2);
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 1, Number> mass_inv(eval_p);
      for (unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
        {
          eval_p.reinit(cell);
          eval_p.read_dof_values(velocity);
          mass_inv.apply(eval_p.begin_dof_values(), eval_p.begin_dof_values());
          eval_p.set_dof_values(pressure);
        }
    }

    

    void
    set_time_step(const double t)
    {
      time_step = t;
    }

  private:
    const MatrixFree<dim, Number>         *matrix_free;
    AlignedVector<VectorizedArray<Number>> penalty_factors;
    Number                                 time;
    double                                 time_step;

    void
    local_apply_domain(const MatrixFree<dim, Number>               &data,
                       VectorType                                  &dst,
                       const VectorType                            &src,
                       const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, 1, Number> eval(data, 1, 2);

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
    local_apply_inner_face(const MatrixFree<dim, Number>               &data,
                           VectorType                                  &dst,
                           const VectorType                            &src,
                           const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, 1, Number> eval_minus(data, true, 1, 2);
      FEFaceEvaluation<dim, -1, 0, 1, Number> eval_plus(data, false, 1, 2);

      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          eval_minus.reinit(face);
          eval_plus.reinit(face);
          eval_minus.gather_evaluate(src,
                                     EvaluationFlags::values |
                                       EvaluationFlags::gradients);
          eval_plus.gather_evaluate(src,
                                    EvaluationFlags::values | EvaluationFlags::gradients);

          for (const unsigned int q : eval_minus.quadrature_point_indices())
            {
              const auto u_minus = eval_minus.get_value(q);
              const auto u_plus  = eval_plus.get_value(q);

              const auto viscous_value_flux = make_vectorized_array<Number>(0.5) *
                                                (eval_minus.get_normal_derivative(q) +
                                                 eval_plus.get_normal_derivative(q)) -
                                              penalty_factors[face] * (u_minus - u_plus);
              const auto viscous_gradient_flux =
                make_vectorized_array<Number>(0.5) * (u_plus - u_minus);

              eval_minus.submit_normal_derivative(viscous_gradient_flux, q);
              eval_plus.submit_normal_derivative(viscous_gradient_flux, q);

              eval_minus.submit_value(-viscous_value_flux, q);
              eval_plus.submit_value(viscous_value_flux, q);
            }

          eval_minus.integrate_scatter(EvaluationFlags::values |
                                         EvaluationFlags::gradients,
                                       dst);
          eval_plus.integrate_scatter(EvaluationFlags::values |
                                        EvaluationFlags::gradients,
                                      dst);
        }
    }

    void
    local_apply_boundary_face(
      const MatrixFree<dim, Number>               &data,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, 1, Number> eval_minus(data, true, 1, 2);

      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          eval_minus.reinit(face);
          eval_minus.gather_evaluate(src,
                                     EvaluationFlags::values |
                                       EvaluationFlags::gradients);

          // Dirichlet boundary
          if (data.get_boundary_id(face) == 1)
            for (const unsigned int q : eval_minus.quadrature_point_indices())
              {
                const auto u_minus = eval_minus.get_value(q);

                const auto viscous_value_flux = 2.0 * penalty_factors[face] * u_minus -
                                                eval_minus.get_normal_derivative(q);
                const auto viscous_gradient_flux = -u_minus;

                eval_minus.submit_normal_derivative(viscous_gradient_flux, q);

                eval_minus.submit_value(viscous_value_flux, q);
              }
          else if (data.get_boundary_id(face) == 0)
            for (const unsigned int q : eval_minus.quadrature_point_indices())
              {
                // const auto u_minus = eval_minus.get_value(q);

                eval_minus.submit_normal_derivative({} /*viscous_gradient_flux*/, q);

                eval_minus.submit_value({} /*viscous_value_flux*/, q);
              }
          else
            AssertThrow(false,
                        ExcNotImplemented(
                          "Boundary id " +
                          std::to_string(int(data.get_boundary_id(face))) +
                          " not known"));

          eval_minus.integrate_scatter(EvaluationFlags::values |
                                         EvaluationFlags::gradients,
                                       dst);
        }
    }

    void
    local_rhs_domain(const MatrixFree<dim, Number> & data,
                     VectorType & dst,
                     const VectorType & src,
                     const std::pair<unsigned int, unsigned int> & cell_range) const
    {
        (void) src;

        FEEvaluation<dim, -1, 0, 1, Number>   eval_p(data, 1, 1);

        AnalyticalRHS<dim> rhs(u_x_max, viscosity);
        rhs.set_time(time);
        AnalyticalRHS<dim> rhs_m(u_x_max, viscosity);
        rhs_m.set_time(time - time_step);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
            eval_p.reinit(cell);

            // loop over quadrature points and compute the local volume flux
            for (const unsigned int q : eval_p.quadrature_point_indices())
            {
                const auto f =
                    evaluate_function(rhs, eval_p.quadrature_point(q));
                const auto f_m = 
                    evaluate_function(rhs_m, eval_p.quadrature_point(q));

                
                eval_p.submit_gradient(f - f_m, q);
            }

            // multiply by nabla v^h(x) and sum
            eval_p.integrate_scatter(EvaluationFlags::gradients, dst);
        }
    }

    void
    local_rhs_inner_face(const MatrixFree<dim, Number> &data,
                         VectorType &dst,
                         const VectorType &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const
    {
        (void) src;
        FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_minus(data, true, 1, 1);
        FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_plus(data, false, 1, 1);

        AnalyticalRHS<dim> rhs(u_x_max, viscosity);
        rhs.set_time(time);
        AnalyticalRHS<dim> rhs_m(u_x_max, viscosity);
        rhs_m.set_time(time - time_step);

        for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
            eval_p_minus.reinit(face);
            eval_p_plus.reinit(face);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
            {
                const auto f =
                    evaluate_function(rhs, eval_p_minus.quadrature_point(q));
                const auto f_m = 
                    evaluate_function(rhs_m, eval_p_minus.quadrature_point(q));

                const auto normal  = eval_p_minus.normal_vector(q);

                const auto flux = (f - f_m) * normal; //TODO: check this

                eval_p_minus.submit_value(-flux, q);
                eval_p_plus.submit_value(flux, q);
                }

            eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
            eval_p_plus.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    void
    local_rhs_boundary_face(const MatrixFree<dim, Number>               &data,
                            VectorType                                  &dst,
                            const VectorType                            &src,
                            const std::pair<unsigned int, unsigned int> &face_range) const
    {
        (void) src;
        FEFaceEvaluation<dim, -1, 0, 1, Number> eval_p_minus(data, true, 1, 1);
    
        AnalyticalRHS<dim> rhs(u_x_max, viscosity);
        rhs.set_time(time);
        AnalyticalRHS<dim> rhs_m(u_x_max, viscosity);
        rhs_m.set_time(time - time_step);
    
        for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
            eval_p_minus.reinit(face);
    
            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
            {
                const auto f =
                    evaluate_function(rhs, eval_p_minus.quadrature_point(q));
                const auto f_m = 
                    evaluate_function(rhs_m, eval_p_minus.quadrature_point(q));
    
                const auto normal = eval_p_minus.normal_vector(q);
    
                const auto flux = (f - f_m) * normal; //TODO: check this
    
                eval_p_minus.submit_value(-flux, q);
            }
    
            eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
        }
    }
  };



  template <int dim>
  void
  test(const unsigned int degree)
  {
    FESystem<dim> fe_u(FE_DGQ<dim>(degree), dim);
    // FE_RaviartThomasNodal<dim> fe_u(degree - 1);
    FE_DGQ<dim> fe_p(degree - 1);

    Triangulation<dim> tria;
    GridGenerator::hyper_cube(tria, 0., 1.);
    tria.refine_global(5);

    DoFHandler<dim> dof_handler_u(tria);
    dof_handler_u.distribute_dofs(fe_u);
    DoFHandler<dim> dof_handler_p(tria);
    dof_handler_p.distribute_dofs(fe_p);
    std::cout << "Number of active_cells: " << tria.n_global_active_cells() << std::endl;
    std::cout << "Solving with " << fe_u.get_name() << " x " << fe_p.get_name()
              << " element" << std::endl;
    std::cout << "Number of degrees of freedom: " << dof_handler_u.n_dofs() << " + "
              << dof_handler_p.n_dofs() << std::endl;

    const double time_step = 0.01 * tria.begin_active()->minimum_vertex_distance();
    std::cout << "Time step size: " << time_step << std::endl;

    MomentumOperator<dim, double> momentum_op;
    momentum_op.reinit(dof_handler_u, dof_handler_p);
    momentum_op.set_time(0.);
    momentum_op.set_time_factor(1.5 / time_step);
    momentum_op.set_time_step(time_step);

    LinearAlgebra::distributed::Vector<double> vec_u, vec_u_m, vec_u_m2,
      vec_u_extrapolated, vec_u_old, vec_u_rhs, vec_p, vec_p_update, vec_p_rhs, vec_p_tmp;
    std::vector<LinearAlgebra::distributed::Vector<double>> old_divergences(3);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_m, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_m2, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_rhs, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_extrapolated, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_old, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_update, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_rhs, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_tmp, 1);
    for (auto &vec : old_divergences)
      momentum_op.get_matrix_free().initialize_dof_vector(vec, 1);

    PressureOperator<dim, double> pressure_op;
    pressure_op.reinit(momentum_op.get_matrix_free());

    MappingQ1<dim>                  mapping;
    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    AnalyticalSolutionPressure<dim> exact_pressure(u_x_max, viscosity);

    exact_pressure.set_time(0.);
    VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p_tmp);
    exact_pressure.set_time(time_step);
    VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p);

    exact_velocity.set_time(0.);
    VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u_m);
    momentum_op.set_time(0.);
    momentum_op.evaluate_divergence(old_divergences[0], vec_u_m);
    exact_velocity.set_time(time_step);
    VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u);
    momentum_op.set_time(time_step);

    double       time     = time_step;
    const double end_time = 0.5;

    unsigned int time_step_number = 1;
    while (time <= end_time)
      {
        for (unsigned int i = old_divergences.size() - 1; i != 0; --i)
          std::swap(old_divergences[i], old_divergences[i - 1]);
        momentum_op.evaluate_divergence(old_divergences[0], vec_u);
        std::cout << "divergence norm: " << old_divergences[0].l2_norm() << std::endl;

        ++time_step_number;
        time += time_step;
        pressure_op.set_time(time);
        momentum_op.set_time(time);

        if (time_step_number > 2)
          {
            vec_p_rhs = old_divergences[2];
            vec_p_rhs.sadd(0.5 / time_step, -2.5 / time_step, old_divergences[1]);
            vec_p_rhs.add(3.5 / time_step, old_divergences[0]);
            pressure_op.compute_rhs(vec_p_update, vec_p_rhs);
            vec_p_rhs += vec_p_update; 

            VectorTools::subtract_mean_value(vec_p_rhs);
            SolverControl control(2000, 1e-10 * vec_p_rhs.l2_norm());
            SolverCG<LinearAlgebra::distributed::Vector<double>> solver(control);
            vec_p_update = 0;
            solver.solve(pressure_op, vec_p_update, vec_p_rhs, PreconditionIdentity());
            VectorTools::subtract_mean_value(vec_p_update);
            std::cout << "Pressure solver: " << control.last_step() << " iterations"
                      << std::endl;

            vec_p += vec_p_update;
            pressure_op.update_pressure(vec_p_tmp, old_divergences[0]);
            vec_p.add(viscosity, vec_p_tmp);
          }
        else
          {
            vec_p.sadd(2.0, -1.0, vec_p_tmp);
          }

        std::swap(vec_u_m2, vec_u_m);
        std::swap(vec_u_m, vec_u);

        // exact_velocity.set_time(time - 2.0 * time_step);
        // VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u_m2);

        // exact_velocity.set_time(time - 1.0 * time_step);
        // VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u_m);

        vec_u_extrapolated = vec_u_m2;
        vec_u_extrapolated.sadd(-1., 2., vec_u_m);
        vec_u_old = vec_u_m2;
        vec_u_old.sadd(-0.5 / time_step, 2. / time_step, vec_u_m);

        momentum_op.compute_rhs(vec_u_rhs, vec_u_old, vec_u_extrapolated, vec_p);

        SolverControl control(1000, 1e-10 * vec_u_rhs.l2_norm());
        SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver(control);
        vec_u = 0;
        solver.solve(momentum_op, vec_u, vec_u_rhs, PreconditionIdentity());

        // exact_velocity.set_time(time);
        // VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u);

        std::cout << "Momentum solver: " << control.last_step() << " iterations"
                  << std::endl;

        Vector<double> error_per_cell;

        exact_pressure.set_time(time);
        VectorTools::integrate_difference(mapping,
                                          dof_handler_p,
                                          vec_p,
                                          exact_pressure,
                                          error_per_cell,
                                          QGauss<dim>(fe_p.degree + 2),
                                          VectorTools::L2_norm);
        const double pressure_error =
          VectorTools::compute_global_error(tria, error_per_cell, VectorTools::L2_norm);

        exact_velocity.set_time(time);
        VectorTools::integrate_difference(mapping,
                                          dof_handler_u,
                                          vec_u,
                                          exact_velocity,
                                          error_per_cell,
                                          QGauss<dim>(fe_u.degree + 1),
                                          VectorTools::L2_norm);
        const double velocity_error =
          VectorTools::compute_global_error(tria, error_per_cell, VectorTools::L2_norm);

        std::cout << "L2 errors velocity / pressure: " << velocity_error << " / "
                  << pressure_error << std::endl;

        DataOut<dim> data_out;

        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;
        data_out.set_flags(flags);

        data_out.add_data_vector(dof_handler_u, vec_u, "solution");
        VectorTools::interpolate(mapping,
                                 dof_handler_u,
                                 exact_velocity,
                                 vec_u_extrapolated);
        data_out.add_data_vector(dof_handler_u, vec_u_extrapolated, "analytical");
        data_out.add_data_vector(dof_handler_p, vec_p, "pressure");
        data_out.add_data_vector(dof_handler_p, vec_p_update, "pressure_update");
        data_out.add_data_vector(dof_handler_p, vec_p_tmp, "pressure_from_div");
        VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p_update);
        data_out.add_data_vector(dof_handler_p, vec_p_update, "pressure_analytical");
        Vector<double> mpi_owner(tria.n_active_cells());
        mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        data_out.add_data_vector(mpi_owner, "owner");
        data_out.build_patches(mapping, fe_u.degree, DataOut<dim>::curved_inner_cells);

        const std::string filename =
          "solution-L2-" + std::to_string(time_step_number) + ".vtu";
        data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
      }
  }
} // namespace NavierStokes



int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  NavierStokes::test<2>(2);
  // NavierStokes::test<2>(3);
}