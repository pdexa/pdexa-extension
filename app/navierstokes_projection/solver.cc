
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
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
  const double viscosity = 1.0;
  const double u_x_max   = 1.;

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

      const double result = std::cos(pi * p[0]) * std::sin(pi * p[1])* std::sin(t);
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
      
     //This RHS is true for Stokes problem
      /*if (component == 0)
        result = -viscosity * 2 * pi * pi * pi * std::sin(2 * pi * y) * (2 * cos(2 * pi * x) - 1) - pi * std::sin(pi * x) * std::sin(pi * y);
      else if (component == 1)
        result = viscosity * 2 * pi * pi * pi * std::sin(2 * pi * x) * (2 * cos(2 * pi * y) - 1) + pi * std::cos(pi * x) * std::cos(pi * y);
      return result;*/
      /*if (component == 0)
        result = - pi * std::sin(pi * x) * std::sin(pi * y) + 4 * pi * pi * pi * std::sin(pi * p[0]) * std::sin(pi * p[0]) * std::sin(pi * p[0])
        * std::cos(pi * p[0]) * std::sin(pi * p[1]) * std::sin(pi * p[1]);
      else if (component == 1)
        result = pi * std::cos(pi * x) * std::cos(pi * y) + 4 * pi * pi * pi * std::sin(pi * p[1]) * std::sin(pi * p[1]) * std::sin(pi * p[1])
        * std::cos(pi * p[1]) * std::sin(pi * p[0]) * std::sin(pi * p[0]);*/
      return result;
    }

  private:
    const double u_x_max, viscosity;
  };

  template <int dim>
  class AnalyticalRHSDivergence : public dealii::Function<dim>
  {
  public:
  AnalyticalRHSDivergence(const double u_x_max, const double viscosity)
      : dealii::Function<dim>(dim, 0.0)
      , u_x_max(u_x_max)
      , viscosity(viscosity)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int ) const final
    {
      const double t  = this->get_time();
      const double pi = dealii::numbers::PI;
      const double x = p[0];
      const double y = p[1];

      //double result = pi*pi*(-16.0*pi*pi*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*y) - 16.0*pi*pi*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::sin(pi*y) + 24.0*pi*pi*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*y) - 2.0*std::cos(pi*x))*std::sin(t)*std::sin(pi*y);

      double result = -2 * pi * pi * std::cos(pi * x) * std::sin(pi * y);
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
  evaluate_function_scalar(const Function<dim>                       &function,
                    const Point<dim, VectorizedArray<Number>> &p_vectorized)
  {
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
      std::cout << "Set factor " << time_factor << std::endl;
    }

    void
    set_time(const double time)
    {
      this->time = time;
    }

    void
    compute_rhs(VectorType       &dst,
                const VectorType &u_rhs_time_derivative,
                const VectorType &u_extrapolated,
                const VectorType &pressure_extrapolated)
    {
      data.loop(&MomentumOperator::local_rhs_domain,
                &MomentumOperator::local_rhs_inner_face,
                &MomentumOperator::local_rhs_boundary_face,
                this,
                dst,
                std::vector<const VectorType *>{&u_rhs_time_derivative,
                                                &u_extrapolated,
                                                &pressure_extrapolated},
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
    precondition_block_jacobi(VectorType &dst, const VectorType &src) const;

    void
    project_initial(VectorType &dst) const;

  private:
    MatrixFree<dim, Number> data;
    Number                  time;
    Number                  time_factor;

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
            penalty_factors[face] =
              2.0 * std::abs((eval_face.normal_vector(0) *
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
            const Tensor<2, dim, VectorizedArray<Number>> gradient_flux = viscosity * gradu;
            eval.submit_gradient(gradient_flux, q);
            const Tensor<1, dim, VectorizedArray<Number>> value_flux =
              factor * u + gradu * speed;
            eval.submit_value(value_flux, q);
          }

        // multiply by nabla v^h(x) and sum
        eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        //eval.integrate_scatter(EvaluationFlags::gradients, dst);
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
              convective_flux - speed_normal * u_minus;
            const auto convective_flux_p =
              speed_normal * u_plus - convective_flux;

            const auto viscous_value_flux =
              make_vectorized_array<Number>(0.5 * viscosity) *
                (eval_minus.get_normal_derivative(q) +
                 eval_plus.get_normal_derivative(q)) -
              viscosity * penalty_factors[face] * (u_minus - u_plus);
            const auto viscous_gradient_flux =
              make_vectorized_array<Number>(0.5 * viscosity) * (u_plus - u_minus);

            eval_minus.submit_normal_derivative(viscous_gradient_flux, q);
            eval_plus.submit_normal_derivative(viscous_gradient_flux, q);

            //was before
            eval_minus.submit_value(convective_flux_m - viscous_value_flux, q);
            eval_plus.submit_value(convective_flux_p + viscous_value_flux, q);
            //eval_minus.submit_value(- viscous_value_flux, q);
            //eval_plus.submit_value(viscous_value_flux, q);
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
    
  }



  template <int dim, typename Number>
  void
  MomentumOperator<dim, Number>::local_rhs_domain(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const std::vector<const VectorType *>       &src,
    const std::pair<unsigned int, unsigned int> &cell_range)
  {
    FEEvaluation<dim, -1, 0, dim, Number> eval_u_rhs_timeDer(data, 0);
    FEEvaluation<dim, -1, 0, dim, Number> eval_u_extrap(data, 0);
    FEEvaluation<dim, -1, 0, 1, Number>   eval_p_extrap(data, 1);

    AnalyticalRHS<dim> forcing_term(u_x_max, viscosity);
    forcing_term.set_time(time);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_u_rhs_timeDer.reinit(cell);
        eval_u_extrap.reinit(cell);
        eval_p_extrap.reinit(cell);

        eval_u_rhs_timeDer.gather_evaluate(*src[0], EvaluationFlags::values);
        eval_u_extrap.gather_evaluate(*src[1], EvaluationFlags::values);
        eval_p_extrap.gather_evaluate(*src[2], EvaluationFlags::gradients);

        // loop over quadrature points and compute the local volume flux
        for (const unsigned int q : eval_u_rhs_timeDer.quadrature_point_indices())
          {
            speeds_cells(cell, q) = eval_u_extrap.get_value(q);

            const auto f =
              evaluate_function(forcing_term, eval_u_rhs_timeDer.quadrature_point(q));


            const auto u          = eval_u_rhs_timeDer.get_value(q);
            const auto gradp      = eval_p_extrap.get_gradient(q);
          //was before
            eval_u_rhs_timeDer.submit_value(
              u - gradp + f, q);
            //eval_u_rhs_timeDer.submit_value(- gradp + f, q);
          }

          eval_u_rhs_timeDer.integrate_scatter(EvaluationFlags::values, dst);
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
    FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_extrap_minus(data, true, 1);
    FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_extrap_plus(data, false, 1);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_u_minus.reinit(face);
        eval_u_plus.reinit(face);
        eval_p_extrap_minus.reinit(face);
        eval_p_extrap_plus.reinit(face);

        eval_u_minus.gather_evaluate(*src[1], EvaluationFlags::values);
        eval_u_plus.gather_evaluate(*src[1], EvaluationFlags::values);
        eval_p_extrap_minus.gather_evaluate(*src[2], EvaluationFlags::values);
        eval_p_extrap_plus.gather_evaluate(*src[2], EvaluationFlags::values);

        for (const unsigned int q : eval_u_minus.quadrature_point_indices())
          {
            const auto normal = eval_u_minus.get_normal_vector(q);
            speeds_faces(face, q) =
              0.5 * (normal * (eval_u_minus.get_value(q) + eval_u_plus.get_value(q)));
            const auto p_extrap_minus       = eval_p_extrap_minus.get_value(q);
            const auto p_extrap_plus        = eval_p_extrap_plus.get_value(q);
            const auto p_jump_normal = 0.5 * normal * (p_extrap_minus - p_extrap_plus);
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
          /*if (face < matrix_free.n_inner_face_batches())
            {*/
              eval_face_out.reinit(face);
              penalty_factors[face] =
                (std::abs((eval_face.normal_vector(0) *
                           eval_face.inverse_jacobian(0))[dim - 1]) +
                 std::abs((eval_face_out.normal_vector(0) *
                           eval_face_out.inverse_jacobian(0))[dim - 1])) *
                (Number)(std::max(fe_degree, 1u) * (fe_degree + 1.0));
            //}
          /*else
            penalty_factors[face] =
            2.0 *
            std::abs((eval_face.normal_vector(0) *
                      eval_face.inverse_jacobian(0))[dim - 1]) *
            (Number)(std::max(fe_degree, 1u) * (fe_degree + 1.0));*/
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
    compute_rhs(VectorType &dst, const VectorType &u_rhs,
      const VectorType &u_extrapolated)
    {
      matrix_free->loop(&PressureOperator::local_rhs_domain,
                        &PressureOperator::local_rhs_inner_face,
                        &PressureOperator::local_rhs_boundary_face,
                        this,
                        dst,
                        std::vector<const VectorType *>{&u_rhs,
                          &u_extrapolated},
                        true,
                        MatrixFree<dim, Number>::DataAccessOnFaces::gradients,
                        MatrixFree<dim, Number>::DataAccessOnFaces::gradients);
    }

    double time_step;

    void set_time_step(const double t)
    {
      time_step = t;
    }

  private:
    const MatrixFree<dim, Number>         *matrix_free;
    AlignedVector<VectorizedArray<Number>> penalty_factors;
    Number                                 time;

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
                                    EvaluationFlags::values | 
                                      EvaluationFlags::gradients);

          for (const unsigned int q : eval_minus.quadrature_point_indices())
            {
              const auto p_minus = eval_minus.get_value(q);
              const auto p_plus  = eval_plus.get_value(q);

              const auto viscous_value_flux = - make_vectorized_array<Number>(0.5) *
                                                (eval_minus.get_normal_derivative(q) +
                                                 eval_plus.get_normal_derivative(q)) +
                                                penalty_factors[face] * (p_minus - p_plus);
              const auto viscous_gradient_flux =
                make_vectorized_array<Number>(0.5) * (p_plus - p_minus);

              eval_minus.submit_normal_derivative(viscous_gradient_flux, q);
              eval_plus.submit_normal_derivative(viscous_gradient_flux, q);

              eval_minus.submit_value(viscous_value_flux, q);
              eval_plus.submit_value(-viscous_value_flux, q);
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
      
    }

    void
    local_rhs_domain(const MatrixFree<dim, Number>               &data,
                     VectorType                                  &dst,
                     const std::vector<const VectorType *>       &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, dim, Number> eval_u(data, 0, 1);
      FEEvaluation<dim, -1, 0, dim, Number> eval_u_extrap(data, 0, 1);
      FEEvaluation<dim, -1, 0, 1, Number>   eval_p(data, 1, 1);

      //AnalyticalRHSDivergence<dim> forcing_term_diverg(u_x_max, viscosity);
      AnalyticalRHS<dim> forcing_term(u_x_max, viscosity);
      forcing_term.set_time(time);
      //forcing_term_diverg.set_time(time);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          eval_u.reinit(cell);
          eval_u_extrap.reinit(cell);
          eval_p.reinit(cell);

          eval_u.gather_evaluate(*src[0], EvaluationFlags::gradients);
          eval_u_extrap.gather_evaluate(*src[1], EvaluationFlags::values);


          // loop over quadrature points and compute the local volume flux
          for (const unsigned int q : eval_u.quadrature_point_indices())
            {
              const auto f =
                evaluate_function(forcing_term, eval_u.quadrature_point(q));

              //eval_p.submit_value(-f, q);
              
              eval_p.submit_gradient(f - eval_u.get_gradient(q) * eval_u_extrap.get_value(q), q); //that was before with all terms
              //eval_p.submit_gradient(f, q);
            }

          eval_p.integrate_scatter(EvaluationFlags::gradients, dst);
        }
    }

    void
    local_rhs_inner_face(const MatrixFree<dim, Number>               &data,
                         VectorType                                  &dst,
                         const std::vector<const VectorType *>       &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_minus(data, true, 0, 1);
      FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_plus(data, false, 0, 1);
      FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_extrap_minus(data, true, 0, 1);
      FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_extrap_plus(data, false, 0, 1);
      FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_minus(data, true, 1, 1);
      FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_plus(data, false, 1, 1);

      AnalyticalRHS<dim> forcing_term(u_x_max, viscosity);
      forcing_term.set_time(time);

      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          eval_p_minus.reinit(face);
          eval_p_plus.reinit(face);
          eval_u_minus.reinit(face);
          eval_u_plus.reinit(face);
          eval_u_extrap_minus.reinit(face);
          eval_u_extrap_plus.reinit(face);
          eval_u_minus.gather_evaluate(*src[0], EvaluationFlags::values | EvaluationFlags::gradients);
          eval_u_plus.gather_evaluate(*src[0], EvaluationFlags::values | EvaluationFlags::gradients);
          eval_u_extrap_minus.gather_evaluate(*src[1], EvaluationFlags::values);
          eval_u_extrap_plus.gather_evaluate(*src[1], EvaluationFlags::values);


          for (const unsigned int q : eval_u_minus.quadrature_point_indices())
            {
              const auto u_minus = eval_u_minus.get_value(q);
              const auto u_plus  = eval_u_plus.get_value(q);
              const auto u_extrap_minus = eval_u_extrap_minus.get_value(q);
              const auto u_extrap_plus = eval_u_extrap_plus.get_value(q);
              const auto u_minus_grad = eval_u_minus.get_gradient(q);
              const auto u_plus_grad  = eval_u_plus.get_gradient(q);
              const auto normal  = eval_u_minus.normal_vector(q);
              const auto f =
              evaluate_function(forcing_term, eval_u_minus.quadrature_point(q));
              /*
              const auto func_value =
                2.0 * (time_factor + 4. * dim * numbers::PI * numbers::PI * viscosity) *
                (evaluate_function(exact_velocity_n1, eval_u_minus.quadrature_point(q)) -
                 evaluate_function(exact_velocity_n, eval_u_minus.quadrature_point(q)));
              */
              //This flux might need changes
              const auto value_flux = 0.5 * (u_minus_grad + u_plus_grad) * 0.5 * (u_extrap_minus + u_extrap_plus);

              //const auto value_flux = 0.5 * (u_minus_grad * u_extrap_minus + u_plus_grad * u_extrap_plus);

              //was before
              eval_p_minus.submit_value(value_flux * normal - f * normal, q);
              eval_p_plus.submit_value(-value_flux * normal + f * normal, q);

              /*eval_p_minus.submit_value(- f * normal, q);
              eval_p_plus.submit_value( f * normal, q);*/


              //This flux might need changes
              const auto grad_flux = 0.5 * (u_minus - u_plus); 
              const auto grad_flux_minus = 0.5 * (u_minus + u_plus) * (u_extrap_minus * normal) 
                                          + 0.5 * std::abs((u_extrap_minus * normal)) * 0.5 * (u_minus - u_plus);
              const auto grad_flux_plus =  -0.5 * (u_minus + u_plus) * (u_extrap_plus * normal) 
                                          - 0.5 * std::abs((u_extrap_plus * normal)) * 0.5 * (u_minus - u_plus);
              eval_p_minus.submit_gradient(grad_flux * (u_extrap_minus * normal), q);
              eval_p_plus.submit_gradient(grad_flux * (u_extrap_plus * normal), q);
              //was  before
              /*eval_p_minus.submit_gradient(grad_flux_minus, q);
              eval_p_plus.submit_gradient(grad_flux_plus, q);*/
            }
            //was before
          eval_p_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
          eval_p_plus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
          /*eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
          eval_p_plus.integrate_scatter(EvaluationFlags::values, dst);*/
        }
    }

    //needs to be fixed! It is the same as it was for velocity-correction projection methods
    void
    local_rhs_boundary_face(const MatrixFree<dim, Number> &data,
                            VectorType &dst,
                            const std::vector<const VectorType *> &src,
                            const std::pair<unsigned int, unsigned int> &face_range) const
    {
     
    }

    void
    local_update_domain(const MatrixFree<dim, Number>               &data,
                        VectorType                                  &dst,
                        const VectorType                            &src,
                        const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, dim, Number> eval_u(data, 0, 1);
      FEEvaluation<dim, -1, 0, 1, Number>   eval_p(data, 1, 1);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          eval_u.reinit(cell);
          eval_p.reinit(cell);

          eval_u.gather_evaluate(src, EvaluationFlags::values);

          // loop over quadrature points and compute the local volume flux
          for (const unsigned int q : eval_u.quadrature_point_indices())
            {
              eval_p.submit_gradient(eval_u.get_value(q), q);
            }

          eval_p.integrate_scatter(EvaluationFlags::gradients, dst);
        }
    }

    void
    local_update_inner_face(const MatrixFree<dim, Number>               &data,
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
              const auto flux    = 0.5 * (normal * (u_minus + u_plus));

              eval_p_minus.submit_value(-flux, q);
              eval_p_plus.submit_value(flux, q);
            }

          eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
          eval_p_plus.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    void
    local_update_boundary_face(
      const MatrixFree<dim, Number>               &data,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_minus(data, true, 0, 1);
      FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_minus(data, true, 1, 1);

      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          eval_p_minus.reinit(face);
          eval_u_minus.reinit(face);
          eval_u_minus.gather_evaluate(src, EvaluationFlags::values);

          AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
          exact_velocity.set_time(time - time_step);
          
          if (false)
          for (const unsigned int q : eval_u_minus.quadrature_point_indices())
            {
              const auto u_minus = eval_u_minus.get_value(q);
              const auto normal  = eval_u_minus.normal_vector(q);
              const auto flux    = normal * u_minus;

              eval_p_minus.submit_value(-flux, q);
            }
          if (true)
          for (const unsigned int q : eval_u_minus.quadrature_point_indices())
          {
            const auto g =
              evaluate_function(exact_velocity, eval_u_minus.quadrature_point(q));
            const auto normal  = eval_u_minus.normal_vector(q);
            const auto flux    = normal * g;

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
    FE_DGQ<dim>   fe_p(degree - 1);

    Triangulation<dim> tria;
    GridGenerator::hyper_cube(tria, -1, 1.);
    
    // set boundary ids on boundaries to the number of the face
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
      ++face)
    tria.begin()->face(face)->set_all_boundary_ids(face);

    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_faces;
    for (unsigned int d = 0; d < dim; ++d)
      GridTools::collect_periodic_faces(
        tria, 2 * d, 2 * d + 1, d, periodic_faces);
    tria.add_periodicity(periodic_faces);

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

    const double time_step = 0.02 * tria.begin_active()->minimum_vertex_distance();

    MomentumOperator<dim, double> momentum_op;
    momentum_op.reinit(dof_handler_u, dof_handler_p);
    momentum_op.set_time(0.);
    momentum_op.set_time_factor(1.5 / time_step);

    LinearAlgebra::distributed::Vector<double> vec_u, vec_u_m, vec_u_analytical,
      vec_u_extrapolated, vec_u_old, vec_u_rhs, vec_u_difference, vec_p,  vec_p_m, vec_p_rhs, vec_p_extrapolated, vec_p_analytical, vec_p_difference;
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_m, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_rhs, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_analytical, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_extrapolated, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_old, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_difference, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_m, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_rhs, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_analytical, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_extrapolated, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_difference, 1);

    PressureOperator<dim, double> pressure_op;
    pressure_op.reinit(momentum_op.get_matrix_free());

    MappingQ1<dim>                  mapping;
    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(0.);
    VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u_m);
    exact_velocity.set_time(time_step);
    VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u);

    

    AnalyticalSolutionPressure<dim> exact_pressure(u_x_max, viscosity);
    exact_pressure.set_time(0.);
    VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p_m);
    exact_pressure.set_time(time_step);
    VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p);
   

    double                          time     = time_step;
    const double                    end_time = 1000. * time_step;

    unsigned int time_step_number = 1;
    std::cout<<"Time step: "<<time_step<<"\nEnd time: "<<end_time<<"\n";
    while (time <= end_time)
    {
      ++time_step_number;
      time += time_step;
      std::cout<<"Time step #:"<<time_step_number<<std::endl;
      
      momentum_op.set_time(time);
      vec_u_extrapolated = vec_u_m;
      vec_u_extrapolated.sadd(-1., 2., vec_u);
      vec_u_old = vec_u_m;
      vec_u_old.sadd(-0.5 / time_step, 2. / time_step, vec_u);

      std::swap(vec_u_m, vec_u);

      //Extrapolation of pressure
      vec_p_extrapolated = vec_p_m;
      vec_p_extrapolated.sadd(-1., 2., vec_p);
      std::swap(vec_p_m, vec_p);

      //Calculation of RHS for Momentum eq.
      momentum_op.compute_rhs(vec_u_rhs, vec_u_old, vec_u_extrapolated, vec_p_extrapolated);

      SolverControl control(1000, 1e-8 * vec_u_rhs.l2_norm());
      SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver(control);
      vec_u = 0;
      solver.solve(momentum_op, vec_u, vec_u_rhs, PreconditionIdentity());
      std::cout << "Momentum solver: " << control.last_step() << " iterations"
                  << std::endl;


      pressure_op.set_time(time);
      pressure_op.compute_rhs(vec_p_rhs, vec_u, vec_u_extrapolated);

      VectorTools::subtract_mean_value(vec_p_rhs);
      SolverControl control_p(1000, 1e-10 * vec_p_rhs.l2_norm());
      SolverCG<LinearAlgebra::distributed::Vector<double>> solver_p(control_p);
      vec_p = 0;
      solver_p.solve(pressure_op, vec_p, vec_p_rhs, PreconditionIdentity());
      //VectorTools::subtract_mean_value(vec_p);

  
      std::cout << "Pressure solver: " << control_p.last_step() << " iterations"
                << std::endl;
      
      Vector<double> error_per_cell;
      exact_velocity.set_time(time);
      VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u_analytical);

      vec_u_difference = 0.0;
      vec_u_difference.sadd(1.0, -1.0, vec_u);
      vec_u_difference.sadd(1.0, 1.0, vec_u_analytical);
      VectorTools::integrate_difference(mapping,
                                        dof_handler_u,
                                        vec_u,
                                        exact_velocity,
                                        error_per_cell,
                                        QGauss<dim>(fe_u.degree + 1),
                                        VectorTools::L2_norm);
      std::cout << "L2 errors: "
                << VectorTools::compute_global_error(tria,
                                                     error_per_cell,
                                                     VectorTools::L2_norm)
                << std::endl;
                

      exact_pressure.set_time(time);
      VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p_analytical);

      
      VectorTools::integrate_difference(mapping,
                                        dof_handler_p,
                                        vec_p,
                                        exact_pressure,
                                        error_per_cell,
                                        QGauss<dim>(fe_p.degree + 2),
                                        VectorTools::L2_norm);
      const double pressure_error =
      VectorTools::compute_global_error(tria, error_per_cell, VectorTools::L2_norm);

      vec_p_difference = 0.0;
      vec_p_difference.sadd(1.0, -1.0, vec_p);
      vec_p_difference.sadd(1.0, 1.0, vec_p_analytical);
      std::cout << "L2 errors pressure: " <<  pressure_error<< std::endl;

      std::cout<<"pressure relative L2 norm: "<<vec_p_difference.l2_norm() / vec_p_analytical.l2_norm() << std::endl;
      std::cout<<"pressure velocity L2 norm: "<<vec_u_difference.l2_norm() / vec_u_analytical.l2_norm() << std::endl;
      
      if(time_step_number % 5 == 0)
      {
        DataOut<dim> data_out;

        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;
        data_out.set_flags(flags);

        data_out.add_data_vector(dof_handler_u, vec_u, "velocity");
        data_out.add_data_vector(dof_handler_u, vec_u_analytical, "velocity_analytical");  
        data_out.add_data_vector(dof_handler_u, vec_u_difference, "velocity_difference"); 

        
        data_out.add_data_vector(dof_handler_p, vec_p, "pressure");
        data_out.add_data_vector(dof_handler_p, vec_p_analytical, "pressure_analytical");  
        data_out.add_data_vector(dof_handler_p, vec_p_difference, "pressure_difference");  

        Vector<double> mpi_owner(tria.n_active_cells());
        mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        data_out.add_data_vector(mpi_owner, "owner");
        data_out.build_patches(mapping, fe_u.degree, DataOut<dim>::curved_inner_cells);

        const std::string filename =
          "solution-L2-" + std::to_string(time_step_number) + ".vtu";
        data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);

      }

    }
        
    
  }
  

} // namespace NavierStokes



int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  NavierStokes::test<2>(2);
  //NavierStokes::test<2>(3);
}
