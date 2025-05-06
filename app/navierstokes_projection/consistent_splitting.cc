
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>

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
  const double viscosity         = 1.0;
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
        result = pi*(16.0*pi*pi*viscosity*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::cos(pi*y) - 4.0*pi*pi*viscosity*std::sin(t)*std::cos(pi*y) + 4.0*pi*pi*std::sin(t)*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*y)*std::cos(pi*x) - 1.0*std::sin(t)*std::sin(pi*x) + 2.0*std::sin(pi*x)*std::sin(pi*x)*std::cos(t)*std::cos(pi*y))*std::sin(pi*y);
        //pi*(4.0*pi*pi*std::sin(t)*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*y)*std::cos(pi*x) + 16.0*pi*pi*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::cos(pi*y) - 1.0*std::sin(t)*std::sin(pi*x) - 4.0*pi*pi*std::sin(t)*std::cos(pi*y) + 2.0*std::sin(pi*x)*std::sin(pi*x)*std::cos(t)*std::cos(pi*y))*std::sin(pi*y);

      else if (component == 1)
        result = pi*(-16.0*pi*pi*viscosity*std::sin(t)*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::cos(pi*x) + 4.0*pi*pi*viscosity*std::sin(t)*std::sin(pi*x)*std::cos(pi*x) + 4.0*pi*pi*std::sin(t)*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::sin(pi*y)*std::cos(pi*y) + 1.0*std::sin(t)*std::cos(pi*x)*std::cos(pi*y) - 2.0*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::cos(t)*std::cos(pi*x));
        //pi*(4.0*pi*pi*std::sin(t)*std::sin(t)*std::sin(pi*x)*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::sin(pi*y)*std::cos(pi*y) - 16.0*pi*pi*std::sin(t)*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::cos(pi*x) + 4.0*pi*pi*std::sin(t)*std::sin(pi*x)*std::cos(pi*x) + 1.0*std::sin(t)*std::cos(pi*x)*std::cos(pi*y) - 2.0*std::sin(pi*x)*std::sin(pi*y)*std::sin(pi*y)*std::cos(t)*std::cos(pi*x));

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
    reinit(const DoFHandler<dim> &dof_handler_u, const DoFHandler<dim> &dof_handler_p, const unsigned int time_integrator_order);

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
                const VectorType &u,
                const VectorType &u_extrapolated,
                const VectorType &pressure_extrapolated)
    {
      data.loop(&MomentumOperator::local_rhs_domain,
                &MomentumOperator::local_rhs_inner_face,
                &MomentumOperator::local_rhs_boundary_face,
                this,
                dst,
                std::vector<const VectorType *>{&u,
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
    evaluate_vorticity(VectorType &dst, const VectorType &src) const
    {
      data.cell_loop(
                &MomentumOperator::local_vorticity_domain,
                this,
                dst,
                src,
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

  private:
    MatrixFree<dim, Number> data;
    Number                  time;
    Number                  time_factor;
    Number time_step;
    unsigned int bdf_order;

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
                                        const DoFHandler<dim> &dof_handler_p,
                                        const unsigned int     time_integrator_order)
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

    bdf_order = time_integrator_order;
    
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
    BDFTimeIntegratorConstants integration_constants(bdf_order);
    double gamma0 = integration_constants.get_gamma0();

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
              viscosity * gradu;
            eval.submit_gradient(gradient_flux, q);
            
            const Tensor<1, dim, VectorizedArray<Number>> value_flux =
              gamma0 / time_step * u + gradu * speed;
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
              convective_flux - speed_normal * u_minus;
            const auto convective_flux_p =
              speed_normal * u_plus - convective_flux;

            const auto viscous_value_flux =
              make_vectorized_array<Number>(0.5 * viscosity) *
                (eval_minus.get_normal_derivative(q) + //n- -> -n+
                 eval_plus.get_normal_derivative(q)) - //n- -> -n+
              viscosity * penalty_factors[face] * (u_minus - u_plus);
            const auto viscous_gradient_flux =
              make_vectorized_array<Number>(0.5 * viscosity) * (u_plus - u_minus);

            eval_minus.submit_normal_derivative(viscous_gradient_flux, q); // n-
            eval_plus.submit_normal_derivative(viscous_gradient_flux, q); // n-

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

              //const auto convective_flux =
                /*0.5 * speed_normal * u_minus +*/
                //0.5 * std::abs(speed_normal) * u_minus
                /*- 0.5 * speed_normal * u_minus */;

              const auto convective_flux =
                std::abs(speed_normal) * (u_minus) - speed_normal * u_minus; //TODO: choose flux

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
            const auto b =
              evaluate_function(forcing_term, eval_u.quadrature_point(q));
            const auto u          = eval_u.get_value(q);
            const auto gradp      = eval_p.get_gradient(q);
            eval_u.submit_value(u - gradp + b, q);
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
            eval_u_plus.submit_value(p_jump_normal, q); //TODO: why no -???
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

    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    exact_velocity.set_time(time);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_u_minus.reinit(face);

        eval_u_minus.gather_evaluate(*src[1], EvaluationFlags::values);

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
              //0.5 * (-speed_normal + std::abs(speed_normal)) * u_plus; //TODO: choose flux
              (-speed_normal + std::abs(speed_normal)) * u_plus;
            const auto viscous_value_flux =
              -2.0 * viscosity * penalty_factors[face] * u_plus;
            const auto viscous_gradient_flux =
              make_vectorized_array<Number>(-viscosity) * (u_plus);


            eval_u_minus.submit_normal_derivative(viscous_gradient_flux, q);

            eval_u_minus.submit_value(convective_flux - viscous_value_flux,
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
    
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_u.reinit(cell);
        eval_u.gather_evaluate(src, EvaluationFlags::gradients);
    
        for(unsigned int q = 0; q < eval_u.n_q_points; ++q)
        {
          if constexpr(dim == 2)
          {
            const auto omega = eval_u.get_curl(q);
            dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> omega_vector;
            for (unsigned int d = 0; d < dim; ++d)
              omega_vector[d] = 0.;
            omega_vector[0] = omega[0];
            eval_u.submit_value(omega_vector, q);   
          }
          else if constexpr(dim == 3)
          {
            eval_u.submit_value(eval_u.get_curl(q), q);
          }
        }
    
        eval_u.integrate_scatter(dealii::EvaluationFlags::values, dst);
      }
    }




  template <int dim, typename Number>
  class PressureOperator
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    PressureOperator() = default;

    void
    reinit(const MatrixFree<dim, Number> &matrix_free, const unsigned int time_order)
    {
      bdf_order = time_order;
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
    compute_rhs(VectorType &dst, const VectorType &velocity, const VectorType &vorticity)
    {
      matrix_free->loop(&PressureOperator::local_rhs_domain,
                        &PressureOperator::local_rhs_inner_face,
                        &PressureOperator::local_rhs_boundary_face,
                        this,
                        dst,
                        std::vector<const VectorType *>{&velocity, &vorticity},
                        true,
                        MatrixFree<dim, Number>::DataAccessOnFaces::gradients,
                        MatrixFree<dim, Number>::DataAccessOnFaces::values);
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
    unsigned int bdf_order;

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
          if (data.get_boundary_id(face) == 0)
          {
            // Do nothing
          }  
          else
          {
            eval_minus.reinit(face);
            eval_minus.gather_evaluate(src,
                                        EvaluationFlags::values |
                                        EvaluationFlags::gradients);

            // Dirichlet boundary
            if (data.get_boundary_id(face) == 1)
                for (const unsigned int q : eval_minus.quadrature_point_indices())
                {
                    DEAL_II_NOT_IMPLEMENTED();
                    const auto u_minus = eval_minus.get_value(q);

                    const auto viscous_value_flux = 2.0 * penalty_factors[face] * u_minus -
                                                    eval_minus.get_normal_derivative(q);
                    const auto viscous_gradient_flux = -u_minus;

                    eval_minus.submit_normal_derivative(viscous_gradient_flux, q);

                    eval_minus.submit_value(viscous_value_flux, q);
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
    }

    void
    local_rhs_domain(const MatrixFree<dim, Number> & data,
                     VectorType & dst,
                     const std::vector<const VectorType *> & src,
                     const std::pair<unsigned int, unsigned int> & cell_range) const
    {
        FEEvaluation<dim, -1, 0, 1, Number>   eval_p(data, 1, 1);
        FEEvaluation<dim, -1, 0, dim, Number> eval_u(data, 0, 1);

        AnalyticalRHS<dim> rhs(u_x_max, viscosity);
        rhs.set_time(time);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
            eval_p.reinit(cell);
            eval_u.reinit(cell);

            eval_u.gather_evaluate(*src[0], EvaluationFlags::values | EvaluationFlags::gradients); // change this to extrapolated velocity

            // loop over quadrature points and compute the local volume flux
            for (const unsigned int q : eval_p.quadrature_point_indices())
            {
                const auto f =
                    evaluate_function(rhs, eval_p.quadrature_point(q));
                
                const auto convective_flux = eval_u.get_gradient(q) * eval_u.get_value(q); //eval_u_extrapolated.get_value(q);
                eval_p.submit_gradient(f - convective_flux, q);
            }

            // multiply by nabla v^h(x) and sum
            eval_p.integrate_scatter(EvaluationFlags::gradients, dst);
        }
    }

    void
    local_rhs_inner_face(const MatrixFree<dim, Number> &data,
                         VectorType &dst,
                         const std::vector<const VectorType *> &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const
    {
        FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_minus(data, true, 1, 1);
        FEFaceEvaluation<dim, -1, 0, 1, Number>   eval_p_plus(data, false, 1, 1);
        FEFaceEvaluation<dim, -1, 0, dim, Number>   eval_u_minus(data, true, 0, 1);
        FEFaceEvaluation<dim, -1, 0, dim, Number>   eval_u_plus(data, false, 0, 1);

        AnalyticalRHS<dim> rhs(u_x_max, viscosity);
        rhs.set_time(time);


        for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
            eval_p_minus.reinit(face);
            eval_p_plus.reinit(face);
            eval_u_minus.reinit(face);
            eval_u_plus.reinit(face);

            eval_u_minus.gather_evaluate(*src[0], EvaluationFlags::values | EvaluationFlags::gradients);
            eval_u_plus.gather_evaluate(*src[0], EvaluationFlags::values | EvaluationFlags::gradients);


            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
              {
                const auto f =
                    evaluate_function(rhs, eval_p_minus.quadrature_point(q));
               
                const auto normal  = eval_p_minus.normal_vector(q);

                const auto flux = f * normal;

                //const auto u = Number(0.5) * (eval_u_minus.get_value(q) + eval_u_plus.get_value(q)); //TODO: adjust this flux
                //const auto u_grad = Number(0.5) * (eval_u_minus.get_gradient(q) + eval_u_plus.get_gradient(q));
                //const auto convective_flux = (u_grad * u) * normal;
                const auto gradu_u_minus = eval_u_minus.get_gradient(q) * eval_u_minus.get_value(q);
                const auto gradu_u_plus = eval_u_plus.get_gradient(q) * eval_u_plus.get_value(q);
                const auto convective_flux = Number(0.5)*(gradu_u_minus + gradu_u_plus) * normal;

                eval_p_minus.submit_value(convective_flux - flux, q);
                eval_p_plus.submit_value(flux - convective_flux, q);
              }

            eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
            eval_p_plus.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    void
    local_rhs_boundary_face(const MatrixFree<dim, Number>               &data,
                            VectorType                                  &dst,
                            const std::vector<const VectorType *>       &src,
                            const std::pair<unsigned int, unsigned int> &face_range) const
    {
        FEFaceEvaluation<dim, -1, 0, 1, Number> eval_p_minus(data, true, 1, 1);
        FEFaceEvaluation<dim, -1, 0, dim, Number> eval_vorticity(data, true, 0, 1);
        FEFaceEvaluation<dim, -1, 0, dim, Number> eval_u_minus(data, true, 0, 1);

        AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
        exact_velocity.set_time(time);

        BDFTimeIntegratorConstants integration_constants(bdf_order);
        
        for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
            eval_p_minus.reinit(face);
            eval_vorticity.reinit(face);
            eval_u_minus.reinit(face);

            eval_vorticity.gather_evaluate(*src[1], EvaluationFlags::gradients);
            eval_u_minus.gather_evaluate(*src[0], EvaluationFlags::values | EvaluationFlags::gradients);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
            { 
              const auto normal = eval_p_minus.normal_vector(q);

              auto g = evaluate_function(exact_velocity, eval_p_minus.quadrature_point(q));
              auto u_plus =  make_vectorized_array(integration_constants.get_gamma0() / time_step) * g;
                  
              
              for (unsigned int i = 0; i < integration_constants.get_order(); ++i)
              {
                exact_velocity.set_time(time - (i + 1) * time_step);
                u_plus -= make_vectorized_array(integration_constants.get_alpha(i) / time_step) *
                  evaluate_function(exact_velocity, eval_p_minus.quadrature_point(q));
              }
              
              
              const auto flux = (-u_plus) * normal;              

              Tensor<1, dim, VectorizedArray<Number>> curl_omega = CurlCompute<dim, FEFaceEvaluation<dim, -1, 0, dim, Number>>::compute(eval_vorticity, q);
              /*
              for (unsigned int i = 0; i < 4; ++i)
              {
                const auto p = eval_p_minus.quadrature_point(q);
                Number x = p[0][i];
                Number y = p[1][i];
                double pi = numbers::PI;
                Number curl_1 = -2.*pi*pi*pi*(-1.+2.*std::cos(2*pi*x))*std::sin(2*pi*y)*std::sin(time);
                Number curl_2 = 2.*pi*pi*pi*(-1.+2.*std::cos(2*pi*y))*std::sin(2*pi*x)*std::sin(time);
                //if (std::abs(curl_omega[0][i]- curl_1) > 1e-2)
                  //std::cout << std::abs(curl_omega[0][i]- curl_1) << std::endl;
                //if (std::abs(curl_omega[1][i]- curl_2) > 1e-2)
                  //std::cout << std::abs(curl_omega[1][i]- curl_2) << std::endl;
                
                curl_omega[0][i] = curl_1;
                curl_omega[1][i] = curl_2;

              }
              */
              const auto curl_flux = (-viscosity) * normal * curl_omega;

              const auto u =  eval_u_minus.get_value(q);
              const auto grad_u = eval_u_minus.get_gradient(q);

              const auto convective_value_flux = (grad_u * (g - u)) * normal;

              eval_p_minus.submit_value(flux + curl_flux + convective_value_flux, q);
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
    GridGenerator::hyper_cube(tria, -1., 1.);
    const bool periodic_boundary = false;
    if (periodic_boundary)
    {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
            ++face)
                tria.begin()->face(face)->set_all_boundary_ids(face);
  
      std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
        periodic_faces;
      for (unsigned int d = 0; d < dim; ++d)
        GridTools::collect_periodic_faces(
          tria, 2 * d, 2 * d + 1, d, periodic_faces);
      tria.add_periodicity(periodic_faces);
    }
    tria.refine_global(5);

    const unsigned int bdf_order = 2;

    BDFTimeIntegratorConstants bdf(bdf_order);


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
    momentum_op.reinit(dof_handler_u, dof_handler_p, bdf_order);
    momentum_op.set_time(0.);
    momentum_op.set_time_factor(1.5 / time_step);
    momentum_op.set_time_step(time_step);

    LinearAlgebra::distributed::Vector<double> vec_u, vec_u_extrapolated, speed_extrapolated, vec_u_rhs, vec_p, vec_p_extrapolated, vec_p_rhs, vec_vorticity;
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_extrapolated, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(speed_extrapolated, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_u_rhs, 0);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_rhs, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_p_extrapolated, 1);
    momentum_op.get_matrix_free().initialize_dof_vector(vec_vorticity, 0);

    std::vector<LinearAlgebra::distributed::Vector<double>> vec_u_old(bdf_order);
    std::vector<LinearAlgebra::distributed::Vector<double>> vec_p_old(bdf_order);
    
    for (auto &vec : vec_u_old)
      momentum_op.get_matrix_free().initialize_dof_vector(vec, 0);
    for (auto &vec : vec_p_old)
      momentum_op.get_matrix_free().initialize_dof_vector(vec, 1);

    PressureOperator<dim, double> pressure_op;
    pressure_op.reinit(momentum_op.get_matrix_free(), bdf_order);
    pressure_op.set_time_step(time_step);

    MappingQ1<dim>                  mapping;
    AnalyticalSolutionVelocity<dim> exact_velocity(u_x_max, viscosity);
    AnalyticalSolutionPressure<dim> exact_pressure(u_x_max, viscosity);

    double time = 0;
    unsigned int time_step_number = 0;
    for (unsigned int i = 0; i < bdf_order; ++i)
    {
      // Fill inital data
      exact_pressure.set_time(time);
      VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p_old[bdf_order - 1 - i]);

      exact_velocity.set_time(time);
      VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u_old[bdf_order - 1 - i]);

      time += time_step;
      ++time_step_number;
    }

    time -= time_step;
    time_step_number -= 1;
    const double end_time = 0.5;

    exact_velocity.set_time(time);
    VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u);

    while (time <= end_time)
      {
        ++time_step_number;
        time += time_step;
        pressure_op.set_time(time);
        momentum_op.set_time(time);

        // Momentum step
        vec_u_extrapolated = 0.;
        speed_extrapolated = 0.;
        vec_p_extrapolated = 0.;
        for(unsigned int i = 0; i < bdf.get_order(); ++i)
        {
          vec_u_extrapolated.add(bdf.get_alpha(i) / time_step, vec_u_old[i]);
          speed_extrapolated.add(bdf.get_beta(i), vec_u_old[i]);
          vec_p_extrapolated.add(bdf.get_beta(i), vec_p_old[i]);
        }

        //speed_extrapolated = vec_u;
        momentum_op.compute_rhs(vec_u_rhs, vec_u_extrapolated, speed_extrapolated, vec_p_extrapolated);

        SolverControl control_mom(1000, 1e-8 * vec_u_rhs.l2_norm());
        SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver_mom(control_mom);
        vec_u = 0.;
        solver_mom.solve(momentum_op, vec_u, vec_u_rhs, PreconditionIdentity());
        std::cout << "Momentum solver: " << control_mom.last_step() << " iterations"
                  << std::endl;

        //exact_velocity.set_time(time);
        //VectorTools::interpolate(mapping, dof_handler_u, exact_velocity, vec_u);
        // Pressure step
        momentum_op.evaluate_vorticity(vec_vorticity, vec_u);
        pressure_op.compute_rhs(vec_p_rhs, vec_u, vec_vorticity);
 
        VectorTools::subtract_mean_value(vec_p_rhs);
        SolverControl control(2000, 1e-8 * vec_p_rhs.l2_norm());
        SolverCG<LinearAlgebra::distributed::Vector<double>> solver(control);
        vec_p = 0.;
        solver.solve(pressure_op, vec_p, vec_p_rhs, PreconditionIdentity());
        VectorTools::subtract_mean_value(vec_p);
        std::cout << "Pressure solver: " << control.last_step() << " iterations"
                  << std::endl;

        //exact_pressure.set_time(time);
        //VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p);

        for (unsigned int i = vec_u_old.size() - 1; i != 0; --i)
          std::swap(vec_u_old[i], vec_u_old[i - 1]);
        for (unsigned int i = vec_p_old.size() - 1; i != 0; --i)
          std::swap(vec_p_old[i], vec_p_old[i - 1]);
        vec_p_old[0] = vec_p;
        vec_u_old[0] = vec_u;

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
        VectorTools::interpolate(mapping, dof_handler_p, exact_pressure, vec_p_extrapolated);
        data_out.add_data_vector(dof_handler_p, vec_p_extrapolated, "pressure_analytical");
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

  NavierStokes::test<2>(3);
  //NavierStokes::test<2>(3);
}
