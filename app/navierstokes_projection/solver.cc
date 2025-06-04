//Version with all terms
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

#include<fstream>


namespace NavierStokes
{
  const double viscosity = 1;
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
  class AnalyticalCurl : public dealii::Function<dim>
  {
  public:
    AnalyticalCurl(const double u_x_max, const double viscosity)
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
        result = - 2 * pi * pi * (std::cos(2 * pi * p[0]) * std::sin(pi * p[1]) * std::sin(pi * p[1]) + 
        std::cos(2 * pi * p[1]) * std::sin(pi * p[0]) * std::sin(pi * p[0])) * std::sin(t);
      else if (component == 1)
        result = 0;
      
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
        /*result = pi * std::sin(pi * x) * std::sin(pi * x) * (2 * pi * pi * std::sin(t) * std::sin(t) * std::sin(2 * pi * x) * std::sin(pi * y) * std::sin(pi * y)
        + std::cos(t) * std::sin(2 * pi * y)) - pi * std::sin(t) * std::sin(pi * x) * std::sin(pi * y)
        - 2 * viscosity * pi * pi * pi * (2 * std::cos(2 * pi * x) - 1) * std::sin(2 * pi * y) * sin (t);*/
      else if (component == 1)
        result = pi*(4.0*pi*pi*std::sin(t)*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::sin(pi*y)*std::cos(pi*y) - 16.0*pi*pi*std::sin(t)*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::cos(pi*x) + 4.0*pi*pi*std::sin(t)*std::sin(pi*x)*std::cos(pi*x) + 1.0*std::sin(t)*std::cos(pi*x)*std::cos(pi*y) - 2.0*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::cos(t)*std::cos(pi*x));
        /*result = pi * std::sin(pi * y) * std::sin(pi * y) * (2 * pi * pi * std::sin(t) * std::sin(t) * std::sin(2 * pi * y) * std::sin(pi * x) * std::sin(pi * x)
        - std::cos(t) * std::sin(2 * pi * x)) + pi * std::sin(t) * std::cos(pi * x) * std::cos(pi * y)
        + 2 * viscosity * pi * pi * pi * (2 * std::cos(2 * pi * y) - 1) * std::sin(2 * pi * x) * sin (t);*/
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

  class BDFTimeIntegratorConstants
  {
    public:
      BDFTimeIntegratorConstants(unsigned int const current_order)
      {
          order = current_order;
          alpha.resize(current_order);
          beta.resize(current_order);
          switch(current_order)
          {
              case 1:
              {
              gamma0   = 1.0;
              alpha[0] = 1.0;
              beta[0] = 1.0;
              break;
              }
              case 2:
              {
              gamma0   = 3.0 / 2.0;
              alpha[0] = 2.0;
              alpha[1] = -0.5;
              beta[0] = 2.0;
              beta[1] = -1.0;
              break;
              }
              case 3:
              {
              gamma0   = 11. / 6.;
              alpha[0] = 3.;
              alpha[1] = -1.5;
              alpha[2] = 1. / 3.;
              beta[0] = 3.0;
              beta[1] = -3.0;
              beta[2] = 1.0;
              break;
              }
              case 4:
              {
              gamma0   = 25. / 12.;
              alpha[0] = 4.;
              alpha[1] = -3.;
              alpha[2] = 4. / 3.;
              alpha[3] = -1. / 4.;
              beta[0] = 4.;
              beta[1] = -6.;
              beta[2] = 4.;
              beta[3] = -1.;
              break;
              }
              default:
              {
                  AssertThrow(false, dealii::ExcMessage("Specified order of BDF scheme not implemented."));
              }
          }            
      }

      double get_gamma0()
      {
          return gamma0;
      }
      double get_alpha(const unsigned int i)
      {
          return alpha[i];
      }
      double get_beta(const unsigned int i)
      {
          return beta[i];
      }
      unsigned int get_order()
      {
        return order;
      }

    std::vector<double> alpha;
    std::vector<double> beta;
    double gamma0;
    unsigned int order;
  };

  template<int dim, typename FEEval>
  struct CurlCompute
  {
    static typename FEEval::value_type
    compute(FEEval const & fe_eval, unsigned int const q_point)
    {
      return fe_eval.get_curl(q_point);
    }
  };

  template<typename FEEval>
  struct CurlCompute<2, FEEval>
  {
    static typename FEEval::value_type
    compute(FEEval const & fe_eval, unsigned int const q_point)
    {
      typename FEEval::gradient_type temp = fe_eval.get_gradient(q_point);
      typename FEEval::value_type curl;
      curl[0] = temp[0][1];  //   d(phi)/dx2
      curl[1] = -temp[0][0]; // - d(phi)/dx1
      return curl;
    }
  };


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
    project_curlVelocity(const VectorType &velocity, VectorType &dst)
    {
      data.cell_loop(
        &MomentumOperator::local_vorticity_domain,
        this,
        dst,
        velocity,
        true);
      FEEvaluation<dim, -1, 0, dim, Number> eval_u(data, 0, 1);
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim, Number> mass_inv(eval_u);
        
      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        eval_u.reinit(cell);
        eval_u.read_dof_values(dst);
        mass_inv.apply(eval_u.begin_dof_values(), eval_u.begin_dof_values());
        eval_u.set_dof_values(dst);
      }
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
                            void
    local_vorticity_domain(
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
    FEFaceEvaluation<dim, -1, 0, dim, Number> eval_minus(data, true);
    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(time);

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
              const auto u_plus = evaluate_function(exact_velocity, eval_minus.quadrature_point(q));

              const auto convective_flux =
                std::abs(speed_normal) * u_minus - speed_normal * u_minus;

              const auto viscous_value_flux =
                make_vectorized_array<Number>(viscosity) *
                eval_minus.get_normal_derivative(q) -
                2.0 * make_vectorized_array<Number>(viscosity) * 
                penalty_factors[face] * u_minus;
              const auto viscous_gradient_flux =
                make_vectorized_array<Number>(-viscosity) * (u_minus);

              eval_minus.submit_normal_derivative(viscous_gradient_flux, q);

              eval_minus.submit_value(convective_flux - viscous_value_flux, q);
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
    FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_minus(data, true, 0);
    FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_minus(data, true, 1);

    double time_step = (1.0 / this->time_factor) * (3.0 / 2.0); 

    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(time);

    AnalyticalSolutionVelocity<dim> exact_velocity_m(u_x_max, viscosity);
    exact_velocity_m.set_time(time - time_step);

    AnalyticalSolutionVelocity<dim> exact_velocity_m2(u_x_max, viscosity);
    exact_velocity_m2.set_time(time - 2.0 * time_step);


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
            const auto u_plus_m =
              evaluate_function(exact_velocity_m, eval_u_minus.quadrature_point(q));
              const auto u_plus_m2 =
              evaluate_function(exact_velocity_m2, eval_u_minus.quadrature_point(q));

            auto extrapolated_velocity = 2.0 * u_plus_m - u_plus_m2;
            //extrapolated_velocity = u_plus;
            const auto speed_normal =
              0.5 * (normal * (eval_u_minus.get_value(q) + extrapolated_velocity));
            speeds_faces(face, q) = speed_normal;
            const auto convective_flux =
              (-speed_normal + std::abs(speed_normal)) * u_plus;
            const auto viscous_value_flux =
              2.0 * make_vectorized_array<Number>(viscosity) *
              penalty_factors[face] * u_plus;
            const auto viscous_gradient_flux =
              make_vectorized_array<Number>(-viscosity) * (u_plus);
            
            const auto p_minus       = eval_p_minus.get_value(q);
            const auto p_jump_normal = normal * (p_minus - p_minus);

            eval_u_minus.submit_normal_derivative(viscous_gradient_flux, q);

            eval_u_minus.submit_value(convective_flux + viscous_value_flux +
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
  MomentumOperator<dim, Number>::local_vorticity_domain(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, dim, Number> eval_u(data, 0);
      AnalyticalCurl<dim> exact_curl(u_x_max, viscosity);
      exact_curl.set_time(time);
    
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_u.reinit(cell);
        eval_u.gather_evaluate(src, EvaluationFlags::gradients);
    
        for(unsigned int q = 0; q < eval_u.n_q_points; ++q)
        {
          const auto curl =
              evaluate_function(exact_curl, eval_u.quadrature_point(q));
          if constexpr(dim == 2)
          {
            const auto omega = eval_u.get_curl(q);
            dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> omega_vector;
            for (unsigned int d = 0; d < dim; ++d)
              omega_vector[d] = 0.;
            omega_vector[0] = omega[0];
            eval_u.submit_value(omega_vector, q);   
            //if(omega_vector[0][0]/omega[0][0] < 31.5 || omega_vector[0][0]/omega[0][0] > 32.1)
              //std::cout<<omega_vector[0]/omega[0]<<"\n";
          }
          else if constexpr(dim == 3)
          {
            eval_u.submit_value(eval_u.get_curl(q), q);
          }
        }
    
        eval_u.integrate_scatter(EvaluationFlags::values, dst);
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
            penalty_factors[face] =
            2.0 *
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
    compute_rhs(VectorType &dst, const VectorType &u_rhs,
      const VectorType &u_extrapolated, const VectorType &curl_u)
    {
      matrix_free->loop(&PressureOperator::local_rhs_domain,
                        &PressureOperator::local_rhs_inner_face,
                        &PressureOperator::local_rhs_boundary_face,
                        this,
                        dst,
                        std::vector<const VectorType *>{&u_rhs,
                          &u_extrapolated, &curl_u},
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
      FEFaceEvaluation<dim, -1, 0, 1, Number> eval_minus(data, true, 1, 2);

      for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        if (data.get_boundary_id(face) == 0)
            for (const unsigned int q : eval_minus.quadrature_point_indices())
            {
              eval_minus.submit_normal_derivative({} , q);

              eval_minus.submit_value({} , q);
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

    void
    local_rhs_boundary_face(const MatrixFree<dim, Number> &data,
                            VectorType &dst,
                            const std::vector<const VectorType *> &src,
                            const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_minus(data, true, 0, 1);
      FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_extrap_minus(data, true, 0, 1);
      FEFaceEvaluation<dim, -1, 0, dim, Number> eval_curlu_minus(data, true, 0, 1);
      FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_minus(data, true, 1, 1);

      AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
      exact_velocity.set_time(time);
      AnalyticalSolutionVelocity<dim> exact_velocity_m(u_x_max, viscosity);
      exact_velocity_m.set_time(time - time_step);
      AnalyticalSolutionVelocity<dim> exact_velocity_m2(u_x_max, viscosity);
      exact_velocity_m2.set_time(time - 2.0 *  time_step);

      BDFTimeIntegratorConstants integration_constants(2);


      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          eval_p_minus.reinit(face);
          eval_u_minus.reinit(face);
          eval_u_extrap_minus.reinit(face);
          eval_curlu_minus.reinit(face);

          eval_u_minus.gather_evaluate(*src[0], EvaluationFlags::values | EvaluationFlags::gradients);
          eval_u_extrap_minus.gather_evaluate(*src[1], EvaluationFlags::values);
          eval_curlu_minus.gather_evaluate(*src[2], EvaluationFlags::gradients);

          /*if (false)
          for (const unsigned int q : eval_u_minus.quadrature_point_indices())
            {
              const auto u_minus = eval_u_minus.get_value(q);
              const auto u_plus  = u_minus;
              const auto normal  = eval_u_minus.normal_vector(q);

              const auto flux = 0.5 * normal * (u_minus + u_plus);

              eval_p_minus.submit_value(-flux, q);
            }*/

          
            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
            {

              exact_velocity.set_time(time);
              auto g = evaluate_function(exact_velocity, eval_p_minus.quadrature_point(q));
              auto g_time_der =  make_vectorized_array(integration_constants.get_gamma0() / time_step) * g;
                  
              
              for (unsigned int i = 0; i < integration_constants.get_order(); ++i)
              {
                exact_velocity.set_time(time - (i + 1) * time_step);
                g_time_der -= make_vectorized_array(integration_constants.get_alpha(i) / time_step) *
                  evaluate_function(exact_velocity, eval_p_minus.quadrature_point(q));
              }
              /*const auto g =
                evaluate_function(exact_velocity, eval_p_minus.quadrature_point(q));*/
              const auto g_m =
                evaluate_function(exact_velocity_m, eval_p_minus.quadrature_point(q));
                const auto g_m2 =
                evaluate_function(exact_velocity_m2, eval_p_minus.quadrature_point(q));
              
              const auto velocity_extrapolated = (2.0 * g_m - g_m2);
              const auto normal  = eval_u_minus.normal_vector(q);
              const auto u_minus = eval_u_minus.get_value(q);
              const auto u_minus_grad = eval_u_minus.get_gradient(q);
              const auto u_extrap_minus = eval_u_extrap_minus.get_value(q);
              //const auto u_extrap_minus = eval_u_minus.get_value(q);

              Tensor<1, dim, VectorizedArray<Number>> u_minus_curlCurl 
              = CurlCompute<dim, FEFaceEvaluation<dim, -1, 0, dim, Number>>::compute(eval_curlu_minus, q);
              //const auto flux    = normal * velocity_extrapolated;

              

              const auto grad_flux = (u_minus - g) * (u_extrap_minus * normal);
              //const auto g_time_der = make_vectorized_array(3./(2*time_step)) * g - make_vectorized_array(2./time_step) * g_m
              //                        + make_vectorized_array(1./(2*time_step)) * g_m2;
              const auto value_flux = -g_time_der - u_minus_grad * (u_minus - g) 
              - viscosity * u_minus_curlCurl;
              
              eval_p_minus.submit_value(value_flux * normal, q);
              eval_p_minus.submit_gradient(grad_flux, q);
            }

          eval_p_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        }
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
  std::vector<double>
  test(const unsigned int degree, const unsigned int refinement = 5, const double endTime = 1.0, double time_step = -1, std::ofstream *fout = nullptr)
  {
    FESystem<dim> fe_u(FE_DGQ<dim>(degree), dim);
    FE_DGQ<dim>   fe_p(degree - 1);

    std::vector<double> relativeErrorNorms(3);

    Triangulation<dim> tria;
    GridGenerator::hyper_cube(tria, 0 , 1);
    tria.refine_global(refinement);

    DoFHandler<dim> dof_handler_u(tria);
    dof_handler_u.distribute_dofs(fe_u);
    DoFHandler<dim> dof_handler_p(tria);
    dof_handler_p.distribute_dofs(fe_p);
    std::cout << "Number of active_cells: " << tria.n_global_active_cells() << std::endl;
    std::cout << "Solving with " << fe_u.get_name() << " x " << fe_p.get_name()
              << " element" << std::endl;
    std::cout << "Number of degrees of freedom: " << dof_handler_u.n_dofs() << " + "
              << dof_handler_p.n_dofs() << std::endl;
    if(time_step < 0)
      time_step = 0.02 * tria.begin_active()->minimum_vertex_distance();

    MomentumOperator<dim, double> momentum_op;
    momentum_op.reinit(dof_handler_u, dof_handler_p);
    momentum_op.set_time(0.);
    momentum_op.set_time_factor(1.5 / time_step);

    LinearAlgebra::distributed::Vector<double> vec_u, vec_u_m, vec_u_analytical,
      vec_curl_exact, vec_curl_diff,
      vec_u_extrapolated, vec_u_old, vec_u_rhs, vec_u_difference, vec_u_curl, vec_p,  vec_p_m, vec_p_rhs, vec_p_extrapolated, vec_p_analytical, vec_p_difference;
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_m, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_rhs, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_analytical, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_extrapolated, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_old, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_difference, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_curl, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_m, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_rhs, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_analytical, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_extrapolated, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_difference, 1);


    momentum_op.get_matrix_free().initialize_dof_vector(vec_curl_exact, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_curl_diff, 0);

    PressureOperator<dim, double> pressure_op;
    pressure_op.reinit(momentum_op.get_matrix_free());
    pressure_op.set_time_step(time_step);

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

    AnalyticalCurl<dim> exact_curl(u_x_max, viscosity);
    exact_curl.set_time(0.);
   

    double                          time     = time_step;
    //const double                    end_time = 100. * time_step;

    unsigned int time_step_number = 1;
    unsigned int numberTimeSteps = int(endTime/time_step);

    std::cout<<"Time step: "<<time_step<<"\nEnd time: "<<endTime<<"\n";
    std::cout<<"number of time steps: "<<numberTimeSteps<<"\n";

    while (time <= endTime)
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

      SolverControl control(5000, 1e-8 * vec_u_rhs.l2_norm());
      SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver(control);
      vec_u = 0;
      solver.solve(momentum_op, vec_u, vec_u_rhs, PreconditionIdentity());
      std::cout << "Momentum solver: " << control.last_step() << " iterations"
                  << std::endl;
      
      momentum_op.project_curlVelocity(vec_u, vec_u_curl);

      pressure_op.set_time(time);
      pressure_op.compute_rhs(vec_p_rhs, vec_u, vec_u_extrapolated, vec_u_curl);

      VectorTools::subtract_mean_value(vec_p_rhs);
      SolverControl control_p(5000, 1e-10 * vec_p_rhs.l2_norm());
      SolverCG<LinearAlgebra::distributed::Vector<double>> solver_p(control_p);
      vec_p = 0;
      solver_p.solve(pressure_op, vec_p, vec_p_rhs, PreconditionIdentity());
      VectorTools::subtract_mean_value(vec_p);

  
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
      std::cout<<"velocity relative L2 norm: "<<vec_u_difference.l2_norm() / vec_u_analytical.l2_norm() << std::endl;
      
      relativeErrorNorms[0] = tria.begin_active()->minimum_vertex_distance();
      relativeErrorNorms[1] = vec_p_difference.l2_norm() / vec_p_analytical.l2_norm();
      relativeErrorNorms[2] = vec_u_difference.l2_norm() / vec_u_analytical.l2_norm();

      exact_curl.set_time(time);
      VectorTools::interpolate(mapping, dof_handler_u, exact_curl, vec_curl_exact);
      vec_curl_diff = 0.0;
      vec_curl_diff.sadd(1.0, -1.0, vec_u_curl);
      vec_curl_diff.sadd(1.0, 1.0, vec_curl_exact);

      if(fout != nullptr)
      {
        std::cout<<"printing in file\n";
        (*fout)<< time << "\t" << vec_p_difference.l2_norm() / vec_p_analytical.l2_norm() << "\t"
          << vec_u_difference.l2_norm() / vec_u_analytical.l2_norm() <<"\n";
      }
      
      if(time_step_number % 100 == 0)
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

        data_out.add_data_vector(dof_handler_u, vec_u_curl, "curl");
        data_out.add_data_vector(dof_handler_u, vec_curl_exact, "curl_analytical");  
        data_out.add_data_vector(dof_handler_u, vec_curl_diff, "curl_difference");  

        Vector<double> mpi_owner(tria.n_active_cells());
        mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        data_out.add_data_vector(mpi_owner, "owner");
        data_out.build_patches(mapping, fe_u.degree, DataOut<dim>::curved_inner_cells);

        const std::string filename =
          "solutionJacobian-L2-"+ std::to_string(tria.begin_active()->minimum_vertex_distance())+ "-" + std::to_string(time_step_number) + ".vtu";
        data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);

      }

    }
    return relativeErrorNorms;
    
  }

  void  convergence_space(std::ofstream & fout)
  {
    unsigned int max_refinement =5, min_refinement = 1;
    std::vector<double> relative_error_norms;
    fout<<"delta x,Error p,Error u\n";
    for(unsigned int ref = min_refinement; ref <= max_refinement; ++ref)
    {
      relative_error_norms = test<2>(3, ref, 1.5, -1);
      fout<<relative_error_norms[0]<<","<<relative_error_norms[1]
        <<","<<relative_error_norms[2]<<"\n";
    }
  } 
  

} // namespace NavierStokes



int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  /*std::ofstream fout_test("relative_errors_over_time.txt");
  NavierStokes::test<2>(3, 5, 1, -1, &fout_test);
  fout_test.close();*/

  std::ofstream fout;
  fout.open("convergece_space1-5timeStepCFL.csv");
  NavierStokes::convergence_space(fout);
  fout.close();
  
  //NavierStokes::test<2>(3);
  //NavierStokes::test<2>(3);
}
