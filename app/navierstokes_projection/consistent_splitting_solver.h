
#pragma once


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include "evaluators.h"

using namespace dealii;



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

  boost::container::small_vector<double, 6> alpha;
  boost::container::small_vector<double, 6> beta;
  double                                    gamma0;
};



const unsigned int dof_no_v = 0;
const unsigned int dof_no_p = 1;

const unsigned int quad_no_v      = 0;
const unsigned int quad_no_v_mass = 1;
const unsigned int quad_no_p      = 2;

template <int dim_, int n_components = dim_, typename Number = double>
class MomentumOperator : public EnableObserverPointer
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
         const unsigned int     bdf_order_in,
         const bool             use_skew_symmetric_convective_formulation = true,
         const bool             use_divergence_formulation                = false,
         const double           upwind_factor                             = 1.0,
         const number           penalty_divergence_in                     = 1.0,
         const number           penalty_continuity_in                     = 1.0,
         const number           penalty_factor_const                      = 1.0)
  {
    bdf_order = bdf_order_in;
    time_step = time_step_in;
    is_dg     = dof_handler_u.get_fe().n_dofs_per_vertex() == 0;
    this->use_skew_symmetric_convective_formulation =
      use_skew_symmetric_convective_formulation;
    this->use_divergence_formulation = use_divergence_formulation;
    this->penalty_divergence         = penalty_divergence_in;
    this->penalty_continuity         = penalty_continuity_in;
    this->upwind_factor              = upwind_factor;

    fe_degree_u                        = dof_handler_u.get_fe().degree;
    const unsigned int fe_degree_p     = dof_handler_p.get_fe().degree;
    Quadrature<1>      quadrature      = QGauss<1>(fe_degree_u + (fe_degree_u + 2) / 2);
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

    FEEvaluation<dim, -1, 0, dim, number> eval_cell(matrix_free, 0, quad_no_v);
    speeds_cells.reinit(matrix_free.n_cell_batches(), eval_cell.n_q_points);
    speeds_cells_mass.reinit(matrix_free.n_cell_batches(),
                             Utilities::pow(quadrature_mass.size(), dim));
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_face(matrix_free, true, 0, 0);
    speeds_faces.reinit(matrix_free.n_inner_face_batches() +
                          matrix_free.n_boundary_face_batches(),
                        eval_face.n_q_points);
    speeds_outer_faces.reinit(matrix_free.n_inner_face_batches() +
                                matrix_free.n_boundary_face_batches(),
                              eval_face.n_q_points);


    penalty_factor = penalty_factor_const * (dof_handler_u.get_fe().degree + 1) *
                     (dof_handler_u.get_fe().degree);
    {
      unsigned int n_cells =
        matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
      array_penalty_parameter.resize(n_cells);
      penalty_factor_divergence.resize(n_cells);
      penalty_factor_continuity.resize(n_cells);


      const dealii::FiniteElement<dim> &fe = dof_handler_u.get_fe();
      const auto reference_cells           = dof_handler_u.get_fe().reference_cell();

      const auto quadrature = reference_cells.get_gauss_type_quadrature(fe_degree_u + 1);
      dealii::FEValues<dim> fe_values(mapping, fe, quadrature, dealii::update_JxW_values);

      const auto face_quadrature =
        reference_cells.face_reference_cell(0).get_gauss_type_quadrature(fe_degree_u + 1);
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

    const auto poly_space =
      Polynomials::generate_complete_Lagrange_basis(quadrature_mass.get_points());
    interpolate_to_face.resize(poly_space.size());
    for (unsigned int q = 0; q < poly_space.size(); ++q)
      {
        std::array<double, 2> val_and_der;
        poly_space[q].value(0.0, 1, val_and_der.data());
        interpolate_to_face[q][0] = val_and_der[0];
        interpolate_to_face[q][1] = val_and_der[1];
        poly_space[q].value(1.0, 1, val_and_der.data());
        interpolate_to_face[q][2] = val_and_der[0];
        interpolate_to_face[q][3] = val_and_der[1];
      }

    Quadrature<dim - 1> quad_face(quadrature_mass);
    face_quadrature_weights.clear();
    face_quadrature_weights.reserve(quad_face.size());
    for (unsigned int q = 0; q < quad_face.size(); ++q)
      face_quadrature_weights.push_back(quad_face.weight(q));
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

  void
  set_body_force_factory(std::function<std::unique_ptr<Function<dim>>()> &&body_force_in)
  {
    body_force_factory = std::move(body_force_in);
  }

  std::function<std::unique_ptr<Function<dim>>()>
  get_body_force_factory()
  {
    return body_force_factory;
  }

  void
  set_DirichletBC_velocity_factory(
    std::function<std::unique_ptr<Function<dim>>()> &&dirichlet_bc_velocity_in)
  {
    dirichletBC_velocity_factory = std::move(dirichlet_bc_velocity_in);
  }

  std::function<std::unique_ptr<Function<dim>>()>
  get_DirichletBC_velocity_factory()
  {
    return dirichletBC_velocity_factory;
  }

  void
  set_dirichletBC_pressure_factory(
    std::function<std::unique_ptr<Function<dim>>()> &&dirichlet_bc_pressure_in)
  {
    dirichletBC_pressure_factory = std::move(dirichlet_bc_pressure_in);
  }

  std::function<std::unique_ptr<Function<dim>>()>
  get_dirichletBC_pressure_factory()
  {
    return dirichletBC_pressure_factory;
  }


  virtual void
  set_divergence_penalty(number penalty_divergence_in)
  {
    penalty_divergence = penalty_divergence_in;
  }

  virtual void
  set_continuity_penalty(number penalty_continuity_in)
  {
    penalty_continuity = penalty_continuity_in;
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

  virtual number
  get_time()
  {
    return time;
  }

  virtual number
  get_viscosity()
  {
    return viscosity;
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

    TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_dofs(),
                                          dof_handler.locally_owned_dofs(),
                                          DoFTools::extract_locally_relevant_dofs(
                                            dof_handler),
                                          dof_handler.get_mpi_communicator());

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
    const Number volume = dealii::Utilities::MPI::sum(dst.at(0), MPI_COMM_WORLD);
    energy              = dealii::Utilities::MPI::sum(dst.at(1), MPI_COMM_WORLD);
    enstrophy           = dealii::Utilities::MPI::sum(dst.at(2), MPI_COMM_WORLD);
    dissipation         = dealii::Utilities::MPI::sum(dst.at(3), MPI_COMM_WORLD);

    energy /= volume;
    enstrophy /= volume;
    dissipation /= volume;

    max_vorticity = dealii::Utilities::MPI::max(dst.at(4), MPI_COMM_WORLD);
  };

  void
  cacheFunctions()
  {
    if (body_force_factory)
      {
        rhs_cached = body_force_factory();
        rhs_cached->set_time(time);
      }
    else
      AssertThrow(false, dealii::ExcMessage("RHS factory empty at call site!"));
    if (dirichletBC_velocity_factory)
      {
        velocity_bc_cached = dirichletBC_velocity_factory();
        velocity_bc_cached->set_time(time);
      }
    else
      AssertThrow(false, dealii::ExcMessage("Velocity BC factory empty at call site!"));
    if (dirichletBC_pressure_factory)
      {
        pressure_bc_cached = dirichletBC_pressure_factory();
        pressure_bc_cached->set_time(time);
      }
    else
      AssertThrow(false, dealii::ExcMessage("Pressure BC factory empty at call site!"));
  }

  template <int n_q_points_1d>
  void
  apply_cellwise_operator_in_collocation_basis(const unsigned int             cell_batch,
                                               const VectorizedArray<Number> *src,
                                               VectorizedArray<Number>       *dst) const
  {
    const auto &shape_info_data =
      matrix_free.get_shape_info(dof_no_v, quad_no_v_mass).data[0];
    AssertThrow(n_q_points_1d == shape_info_data.n_q_points_1d,
                ExcDimensionMismatch(n_q_points_1d, shape_info_data.n_q_points_1d));
    AssertThrow(src != dst,
                ExcMessage("Source and destination may not be the same pointer"));

    apply_cellwise_operator_on_cell<n_q_points_1d>(cell_batch, src, dst);
    apply_add_cellwise_operator_on_faces<n_q_points_1d>(cell_batch, src, dst);
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

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);
        integrator.read_dof_values(src);
        do_cell_integral_local(integrator);
        integrator.distribute_local_to_global(dst);
      }
  }


  void
  do_cell_integral_local(FEEvaluation<dim, -1, 0, n_components, Number> &integrator) const
  {
    BDFTimeIntegratorConstants integration_constants(bdf_order);
    double                     gamma0 = integration_constants.get_gamma0();

    integrator.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (const unsigned int q : integrator.quadrature_point_indices())
      {
        const auto u          = integrator.get_value(q);
        const auto time_deriv = make_vectorized_array<number>(gamma0 / time_step) * u;

        const auto grad_u = integrator.get_gradient(q);
        const auto speed  = speeds_cells(integrator.get_current_cell_index(), q);

        const auto divergence_penalty =
          penalty_factor_divergence[integrator.get_current_cell_index()] * trace(grad_u);
        Tensor<2, dim, VectorizedArray<number>> div_penalty;
        for (unsigned int d = 0; d < dim; ++d)
          div_penalty[d][d] = divergence_penalty;

        if (use_skew_symmetric_convective_formulation)
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
        else if (use_divergence_formulation)
          {
            const auto convective_gradient_flux = outer_product(u, speed);

            integrator.submit_value(time_deriv, q);
            integrator.submit_gradient(div_penalty +
                                         make_vectorized_array<number>(viscosity) *
                                           grad_u -
                                         convective_gradient_flux,
                                       q);
          }
        else
          {
            const auto convective_flux = grad_u * speed;

            integrator.submit_value(time_deriv + convective_flux, q);
            integrator.submit_gradient(
              div_penalty + make_vectorized_array<number>(viscosity) * grad_u, q);
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
        integrator_inner.read_dof_values(src);
        integrator_outer.reinit(face);
        integrator_outer.read_dof_values(src);

        do_face_integral_local(integrator_inner, integrator_outer);

        integrator_inner.distribute_local_to_global(dst);
        integrator_outer.distribute_local_to_global(dst);
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
          make_vectorized_array<number>(viscosity) * sigma * solution_jump -
          averaged_normal_derivative;

        const auto speed = speeds_faces(integrator_inner.get_cell_or_face_batch_id(), q);
        const auto speed_outer =
          speeds_outer_faces(integrator_inner.get_cell_or_face_batch_id(), q);
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

        if (use_skew_symmetric_convective_formulation)
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
        else if (use_divergence_formulation)
          {
            const auto uM = integrator_inner.get_value(q);
            const auto uP = integrator_outer.get_value(q);

            const auto wM = speed;
            const auto wP = speed_outer;

            const auto wM_n = wM * normal;
            const auto wP_n = wP * normal;

            const VectorizedArray<number> Lambda =
              upwind_factor * std::max(std::abs(wM_n), std::abs(wP_n));

            const auto average_normal_flux =
              dealii::make_vectorized_array<Number>(0.5) * (uM * wM_n + uP * wP_n);

            const auto jump_value = uM - uP;

            const auto convective_div_flux =
              average_normal_flux + 0.5 * Lambda * jump_value;

            integrator_inner.submit_value(continuity_penalty_value + test_by_value +
                                            convective_div_flux,
                                          q);
            integrator_outer.submit_value(-continuity_penalty_value - test_by_value -
                                            convective_div_flux,
                                          q);
          }
        else
          {
            integrator_inner.submit_value(continuity_penalty_value + test_by_value +
                                            convective_flux_inner,
                                          q);
            integrator_outer.submit_value(-continuity_penalty_value - test_by_value +
                                            convective_flux_outer,
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
    auto velocity_bc = dirichletBC_velocity_factory();
    velocity_bc->set_time(time);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_inner.read_dof_values(src);
        do_boundary_integral_local(integrator_inner);
        integrator_inner.distribute_local_to_global(dst);
      }
  }


  void
  do_boundary_integral_local(
    FEFaceEvaluation<dim, -1, 0, n_components, Number> &integrator_inner) const
  {
    auto velocity_bc = dirichletBC_velocity_factory();
    velocity_bc->set_time(time);

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
              evaluate_function((*velocity_bc), integrator_inner.quadrature_point(q));
            const auto continuity_penalty_value =
              2. * cont_pen * ((integrator_inner.get_value(q) - g) * normal) * normal;

            if (use_skew_symmetric_convective_formulation)
              {
                const auto convective_value_flux =
                  (std::abs(speed_normal)) * integrator_inner.get_value(q);
                integrator_inner.submit_value(continuity_penalty_value + test_by_value +
                                                0.5 * convective_flux +
                                                0.5 * convective_value_flux,
                                              q);
              }
            else if (use_divergence_formulation)
              {
                const auto uM   = integrator_inner.get_value(q);
                const auto wM   = speed;
                const auto wM_n = wM * normal;

                const VectorizedArray<number> Lambda = upwind_factor * std::abs(wM_n);

                const auto convective_div_flux = Lambda * uM;

                integrator_inner.submit_value(continuity_penalty_value + test_by_value +
                                                convective_div_flux,
                                              q);
              }
            else
              {
                integrator_inner.submit_value(continuity_penalty_value + test_by_value +
                                                convective_flux,
                                              q);
              }
            integrator_inner.submit_normal_derivative(-u_inner, q);
          }
      }
    else if (integrator_inner.boundary_id() == 1)
      {
        // Nothing to do
        // Convective term cancels and viscous term is only inhomogeneous
        for (const unsigned int q : integrator_inner.quadrature_point_indices())
          {
            integrator_inner.submit_normal_derivative(
              Tensor<1, dim, VectorizedArray<number>>(), q);
            if (use_skew_symmetric_convective_formulation)
              {
                const auto speed =
                  speeds_faces(integrator_inner.get_cell_or_face_batch_id(), q);
                const auto convective_flux = speed * (integrator_inner.get_value(q) *
                                                      integrator_inner.normal_vector(q));
                integrator_inner.submit_value(0.5 * convective_flux, q);
              }
            else if (use_divergence_formulation)
              {
                const auto speed =
                  speeds_faces(integrator_inner.get_cell_or_face_batch_id(), q);
                const auto boundary_flux = (speed * integrator_inner.normal_vector(q)) *
                                           integrator_inner.get_value(q);
                integrator_inner.submit_value(boundary_flux, q);
              }
            else
              integrator_inner.submit_value(Tensor<1, dim, VectorizedArray<number>>(), q);
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
    FEEvaluation<dim, -1, 0, n_components, Number> integrator_speed_mass(matrix_free,
                                                                         dof_no_v,
                                                                         quad_no_v_mass);
    FEEvaluation<dim, -1, 0, 1, Number> integrator_p(matrix_free, dof_no_p, quad_no_v);

    auto rhs = body_force_factory();
    rhs->set_time(time);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);
        integrator_speed.reinit(cell);
        integrator_speed_mass.reinit(cell);
        integrator_p.reinit(cell);

        integrator.gather_evaluate(*src[0], EvaluationFlags::values);
        integrator_speed.gather_evaluate(*src[1], EvaluationFlags::values);
        integrator_speed_mass.gather_evaluate(*src[1], EvaluationFlags::values);
        // integrator_p.gather_evaluate(*src[2], EvaluationFlags::values);
        integrator_p.gather_evaluate(*src[2], EvaluationFlags::gradients);

        VectorizedArray<number> avg_velocity(0.);
        VectorizedArray<number> volume(0.);
        for (unsigned int q = 0; q < integrator_speed.n_q_points; ++q)
          {
            volume += integrator_speed.JxW(q);
            avg_velocity +=
              integrator_speed.JxW(q) * integrator_speed.get_value(q).norm();
          }
        avg_velocity /= volume;
        penalty_factor_divergence[cell] = penalty_divergence * avg_velocity *
                                          std::exp(std::log(volume) / (number)dim) /
                                          ((number)fe_degree_u + 1);
        penalty_factor_continuity[cell] = penalty_continuity * avg_velocity;


        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            speeds_cells(cell, q) = integrator_speed.get_value(q);

            const auto f = evaluate_function((*rhs), integrator.quadrature_point(q));

            // const auto p = integrator_p.get_value(q);
            const auto grad_p = integrator_p.get_gradient(q);

            const auto alpha_u = integrator.get_value(q);

            // integrator.submit_value(f + alpha_u, q);
            // integrator.submit_divergence(p, q);
            integrator.submit_value(f + alpha_u - grad_p, q);
          }

        for (const unsigned int q : integrator_speed_mass.quadrature_point_indices())
          speeds_cells_mass(cell, q) = integrator_speed_mass.get_value(q);

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

            if (use_divergence_formulation)
              {
                speeds_faces(face, q)       = integrator_speed_inner.get_value(q);
                speeds_outer_faces(face, q) = integrator_speed_outer.get_value(q);
              }
            else
              {
                const auto speed = make_vectorized_array<number>(0.5) *
                                   (integrator_speed_inner.get_value(q) +
                                    integrator_speed_outer.get_value(q));

                speeds_outer_faces(face, q) = speed;
                speeds_faces(face, q)       = speed;
              }

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

    auto velocity_bc = dirichletBC_velocity_factory();
    velocity_bc->set_time(time);
    auto pressure_bc = dirichletBC_pressure_factory();
    pressure_bc->set_time(time);

    auto velocity_bc_m = dirichletBC_velocity_factory();
    velocity_bc_m->set_time(time - time_step);

    auto velocity_bc_m2 = dirichletBC_velocity_factory();
    velocity_bc_m2->set_time(time - 2.0 * time_step);

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
                  evaluate_function((*velocity_bc), integrator_inner.quadrature_point(q));

                Tensor<1, dim, VectorizedArray<number>> speed;

                speed = make_vectorized_array<number>(0.5) *
                        (integrator_speed_inner.get_value(q) + g);

                speeds_faces(face, q)       = speed;
                speeds_outer_faces(face, q) = speed;
                const auto speed_normal     = speed * normal;

                const auto convective_flux = (std::abs(speed_normal) - speed_normal) * g;

                const auto value_flux =
                  make_vectorized_array<number>(2.0 * viscosity) * sigma * g;
                const auto gradient_flux = make_vectorized_array<number>(viscosity) * g;

                const Tensor<1, dim, VectorizedArray<number>> p =
                  0. * integrator_inner.normal_vector(q);
                // integrator_inner_p.get_value(q) * integrator_inner.normal_vector(q);

                integrator_inner.submit_normal_derivative(-gradient_flux, q);

                if (use_skew_symmetric_convective_formulation)
                  {
                    const auto convective_value_flux =
                      -speed * (g * normal) + std::abs(speed_normal) * g;
                    integrator_inner.submit_value(value_flux - p + 0.5 * convective_flux +
                                                    0.5 * convective_value_flux,
                                                  q);
                  }
                else if (use_divergence_formulation)
                  {
                    const auto convective_div_flux =
                      -g * (g * normal) + std::abs(g * normal) * g;
                    integrator_inner.submit_value(value_flux - p + convective_div_flux,
                                                  q);
                  }
                else
                  {
                    integrator_inner.submit_value(value_flux - p + convective_flux, q);
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
                  evaluate_tensor_function((*velocity_bc),
                                           integrator_inner.quadrature_point(q));
                const auto h_u = make_vectorized_array(viscosity) * grad_g * normal;

                const auto p_plus =
                  evaluate_scalar_function((*pressure_bc),
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
                omega_vector[0] = omega;
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

            const auto omega = fe_eval.get_curl(q);

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

  template <int n_q_points_1d>
  void
  apply_cellwise_operator_on_cell(const unsigned int             cell_batch,
                                  const VectorizedArray<Number> *src,
                                  VectorizedArray<Number>       *dst) const
  {
    const auto &shape_info_data =
      matrix_free.get_shape_info(dof_no_v, quad_no_v_mass).data[0];
    const auto  &mapping_data = matrix_free.get_mapping_info().cell_data[quad_no_v_mass];
    const auto   cell_type    = matrix_free.get_mapping_info().cell_type[cell_batch];
    double       gamma0       = BDFTimeIntegratorConstants(bdf_order).get_gamma0();
    const Number factor_mass  = gamma0 / time_step;
    const Tensor<1, dim, VectorizedArray<Number>> *cell_speed =
      &speeds_cells_mass(cell_batch, 0);
    constexpr unsigned int  n_points_2d = n_q_points_1d * n_q_points_1d;
    constexpr unsigned int  n_points    = Utilities::pow(n_q_points_1d, dim);
    VectorizedArray<Number> grad_z[dim * n_points + 1];
    VectorizedArray<Number> grad_y[dim * Utilities::pow(n_q_points_1d, 2) + 1];
    VectorizedArray<Number> grad_x[dim * n_q_points_1d];

    // Assume collocation here, which allows us to directly start with the
    // cell integrals that write into the result, without overwriting the
    // 'values at quadrature points' slot that is needed later for the faces
    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     3,
                                     n_q_points_1d,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval_grad_z({}, shape_info_data.shape_gradients_collocation_eo, {});
    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     2,
                                     n_q_points_1d,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval_grad_y({}, shape_info_data.shape_gradients_collocation_eo, {});
    internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                     1,
                                     n_q_points_1d,
                                     n_q_points_1d,
                                     VectorizedArray<Number>,
                                     Number>
      eval_grad_x({}, shape_info_data.shape_gradients_collocation_eo, {});
    for (unsigned int comp = 0; comp < dim; ++comp)
      if (dim > 2)
        eval_grad_z.template gradients<2, true, false, dim>(src + comp * n_points,
                                                            grad_z + comp);
    for (unsigned int iz = 0, q = 0; iz < (dim == 2 ? 1 : n_q_points_1d); ++iz)
      {
        for (unsigned int comp = 0; comp < dim; ++comp)
          eval_grad_y.template gradients<1, true, false, dim>(src + iz * n_points_2d +
                                                                comp * n_points,
                                                              grad_y + comp);
        for (unsigned int iy = 0; iy < n_q_points_1d; ++iy)
          {
            for (unsigned int comp = 0; comp < dim; ++comp)
              eval_grad_x.template gradients<0, true, false, dim>(
                src + iy * n_q_points_1d + iz * n_points_2d + comp * n_points,
                grad_x + comp);

            for (unsigned int ix = 0; ix < n_q_points_1d; ++ix, ++q)
              {
                Tensor<1, dim, VectorizedArray<Number>> u;
                for (unsigned int d = 0; d < dim; ++d)
                  u[d] = src[q + d * n_points];

                const unsigned int index_offset =
                  mapping_data.data_index_offsets[cell_batch];
                const Tensor<2, dim, VectorizedArray<Number>> jac =
                  cell_type <= internal::MatrixFreeFunctions::affine ?
                    mapping_data.jacobians[0][index_offset] :
                    mapping_data.jacobians[0][index_offset + q];
                const VectorizedArray<Number> JxW =
                  cell_type <= internal::MatrixFreeFunctions::affine ?
                    mapping_data.JxW_values[index_offset] *
                      mapping_data.descriptor[0].quadrature_weights[q] :
                    mapping_data.JxW_values[index_offset + q];
                Tensor<2, dim, VectorizedArray<Number>> grad_u;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    for (unsigned int e = 0; e < dim; ++e)
                      {
                        if constexpr (dim == 3)
                          grad_u[d][e] =
                            grad_x[dim * ix + d] * jac[e][0] +
                            grad_y[dim * (iy * n_q_points_1d + ix) + d] * jac[e][1] +
                            grad_z[dim * q + d] * jac[e][2];
                        else
                          grad_u[d][e] = grad_x[dim * ix + d] * jac[e][0] +
                                         grad_y[dim * q + d] * jac[e][1];
                      }
                  }

                const auto divergence_penalty =
                  penalty_factor_divergence[cell_batch] * trace(grad_u);
                const auto speed = cell_speed[q];

                // collect terms for
                // div(v) * divergence_penalty * div(u) + grad(v) * viscosity * grad(u)
                Tensor<2, dim, VectorizedArray<number>> test_grad_u;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    test_grad_u[d][d] =
                      divergence_penalty +
                      make_vectorized_array<number>(viscosity) * grad_u[d][d];
                    for (unsigned int e = d + 1; e < dim; ++e)
                      {
                        test_grad_u[d][e] =
                          make_vectorized_array<number>(viscosity) * grad_u[d][e];
                        test_grad_u[e][d] =
                          make_vectorized_array<number>(viscosity) * grad_u[e][d];
                      }
                  }

                Tensor<1, dim, VectorizedArray<Number>> test_u = factor_mass * u;
                if (use_skew_symmetric_convective_formulation)
                  {
                    test_u += make_vectorized_array<Number>(0.5) * (grad_u * speed);
                    test_grad_u -=
                      make_vectorized_array<Number>(0.5) * outer_product(speed, u);
                  }
                else if (use_divergence_formulation)
                  {
                    test_grad_u -= outer_product(u, speed);
                  }
                else
                  test_u += grad_u * speed;

                for (unsigned int d = 0; d < dim; ++d)
                  dst[q + d * n_points] = test_u[d] * JxW;

                test_grad_u = JxW * test_grad_u * jac;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    grad_x[dim * ix + d]                        = test_grad_u[d][0];
                    grad_y[dim * (iy * n_q_points_1d + ix) + d] = test_grad_u[d][1];
                    if (dim > 2)
                      grad_z[dim * q + d] = test_grad_u[d][2];
                  }
              }
            for (unsigned int comp = 0; comp < dim; ++comp)
              eval_grad_x.template gradients<0, false, true, dim>(
                grad_x + comp,
                dst + iy * n_q_points_1d + iz * n_points_2d + comp * n_points);
          }
        for (unsigned int comp = 0; comp < dim; ++comp)
          eval_grad_y.template gradients<1, false, true, dim>(grad_y + comp,
                                                              dst + iz * n_points_2d +
                                                                comp * n_points);
      }
    if (dim > 2)
      for (unsigned int comp = 0; comp < dim; ++comp)
        eval_grad_z.template gradients<2, false, true, dim>(grad_z + comp,
                                                            dst + comp * n_points);
  }

  template <int n_q_points_1d>
  void
  apply_add_cellwise_operator_on_faces(const unsigned int             cell_batch,
                                       const VectorizedArray<Number> *src,
                                       VectorizedArray<Number>       *dst) const
  {
    const auto &shape_info_data =
      matrix_free.get_shape_info(dof_no_v, quad_no_v_mass).data[0];
    const Tensor<1, dim, VectorizedArray<Number>> *cell_speed =
      &speeds_cells_mass(cell_batch, 0);
    AssertThrow(matrix_free.get_mapping_info().cell_type[cell_batch] ==
                  internal::MatrixFreeFunctions::cartesian,
                ExcNotImplemented());
    const auto &mapping_data = matrix_free.get_mapping_info().cell_data[quad_no_v_mass];
    const unsigned int index_offset = mapping_data.data_index_offsets[cell_batch];
    const Tensor<2, dim, VectorizedArray<Number>> inverse_jacobian =
      mapping_data.jacobians[0][index_offset];

    constexpr unsigned int n_q_points_face = Utilities::pow(n_q_points_1d, dim - 1);
    constexpr unsigned int n_q_points      = Utilities::pow(n_q_points_1d, dim);
    for (unsigned int face_direction = 0; face_direction < dim; ++face_direction)
      {
        VectorizedArray<Number> values_face[2][dim][n_q_points_face + 1];
        VectorizedArray<Number> normal_derivative_face[2][dim][n_q_points_face + 1];
        VectorizedArray<Number> tang_derivatives_face[dim][dim - 1][n_q_points_face + 1];

        // interpolate evaluated solution to faces, compute gradients
        internal::EvaluatorTensorProduct<internal::evaluate_evenodd,
                                         dim - 1,
                                         n_q_points_1d,
                                         n_q_points_1d,
                                         VectorizedArray<Number>,
                                         Number>
          eval({}, shape_info_data.shape_gradients_collocation_eo.data(), {});
        const unsigned int stride = Utilities::pow(n_q_points_1d, face_direction);
        for (unsigned int i1 = 0; i1 < (dim > 2 ? n_q_points_1d : 1); ++i1)
          for (unsigned int i0 = 0; i0 < n_q_points_1d; ++i0)
            {
              unsigned int idx = 0;
              if (face_direction == 0)
                idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
              else if (face_direction == 1)
                idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
              else
                idx = i0 + i1 * n_q_points_1d;

              const VectorizedArray<Number>                   *my_vals = src + idx;
              dealii::ndarray<VectorizedArray<Number>, 4, dim> val_on_face;
              for (unsigned int comp = 0; comp < dim; ++comp)
                {
                  val_on_face[0][comp] =
                    interpolate_to_face[0][0] * my_vals[comp * n_q_points];
                  val_on_face[1][comp] =
                    interpolate_to_face[0][1] * my_vals[comp * n_q_points];
                  val_on_face[2][comp] =
                    interpolate_to_face[0][2] * my_vals[comp * n_q_points];
                  val_on_face[3][comp] =
                    interpolate_to_face[0][3] * my_vals[comp * n_q_points];
                }
              for (unsigned int i = 1; i < n_q_points_1d; ++i)
                for (unsigned int comp = 0; comp < dim; ++comp)
                  {
                    val_on_face[0][comp] +=
                      interpolate_to_face[i][0] * my_vals[comp * n_q_points + i * stride];
                    val_on_face[1][comp] +=
                      interpolate_to_face[i][1] * my_vals[comp * n_q_points + i * stride];
                    val_on_face[2][comp] +=
                      interpolate_to_face[i][2] * my_vals[comp * n_q_points + i * stride];
                    val_on_face[3][comp] +=
                      interpolate_to_face[i][3] * my_vals[comp * n_q_points + i * stride];
                  }
              const unsigned int out_idx = i1 * n_q_points_1d + i0;
              for (unsigned int side = 0; side < 2; ++side)
                for (unsigned int comp = 0; comp < dim; ++comp)
                  {
                    values_face[side][comp][out_idx] = val_on_face[side * 2][comp];
                    normal_derivative_face[side][comp][out_idx] =
                      val_on_face[side * 2 + 1][comp];
                  }
            }
        const auto face_jac_det = inverse_jacobian[face_direction][face_direction] /
                                  determinant(inverse_jacobian);

        const VectorizedArray<number> sigma =
          array_penalty_parameter[cell_batch] * get_penalty_factor();

        const VectorizedArray<number> cont_pen = penalty_factor_continuity[cell_batch];

        for (unsigned int side = 0; side < 2; ++side)
          {
            Tensor<1, dim, VectorizedArray<Number>> normal;
            normal[face_direction]  = (side == 0 ? -1. : 1.);
            const auto normal_x_jac = normal * inverse_jacobian;
            for (unsigned int comp = 0; comp < dim; ++comp)
              {
                eval.template gradients<0, true, false>(values_face[side][comp],
                                                        tang_derivatives_face[comp][0]);
                if constexpr (dim > 2)
                  eval.template gradients<1, true, false>(values_face[side][comp],
                                                          tang_derivatives_face[comp][1]);
              }
            const unsigned int first  = (face_direction == 0 ? 1 : 0);
            const unsigned int second = (face_direction == 2 ? 1 : 2);
            for (unsigned int i1 = 0, q = 0; i1 < (dim > 2 ? n_q_points_1d : 1); ++i1)
              for (unsigned int i0 = 0; i0 < n_q_points_1d; ++i0, ++q)
                {
                  unsigned int idx = 0;
                  if (face_direction == 0)
                    idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
                  else if (face_direction == 1)
                    idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
                  else
                    idx = i0 + i1 * n_q_points_1d;
                  Tensor<1, dim, VectorizedArray<Number>> velocity =
                    interpolate_to_face[0][2 * side] * cell_speed[idx];
                  for (unsigned int i = 1; i < n_q_points_1d; ++i)
                    velocity +=
                      interpolate_to_face[i][2 * side] * cell_speed[idx + stride * i];
                  const auto speed_normal = velocity * normal;

                  // operation at quadrature points
                  Tensor<1, dim, VectorizedArray<Number>> sol_val;
                  for (unsigned int comp = 0; comp < dim; ++comp)
                    sol_val[comp] = values_face[side][comp][q];

                  Tensor<1, dim, VectorizedArray<Number>> normal_derivative;
                  for (unsigned int comp = 0; comp < dim; ++comp)
                    {
                      normal_derivative[comp] = normal_x_jac[face_direction] *
                                                normal_derivative_face[side][comp][q];
                      normal_derivative[comp] +=
                        normal_x_jac[first] * tang_derivatives_face[comp][0][q];
                      if constexpr (dim == 3)
                        normal_derivative[comp] +=
                          normal_x_jac[second] * tang_derivatives_face[comp][1][q];
                    }
                  const Tensor<1, dim, VectorizedArray<number>> test_by_value =
                    make_vectorized_array<number>(viscosity) * sigma * sol_val -
                    0.5 * viscosity * normal_derivative +
                    cont_pen * (sol_val * normal) * normal +
                    (std::abs(speed_normal) - speed_normal) * 0.5 * sol_val;

                  const Tensor<1, dim, VectorizedArray<number>> test_by_gradient =
                    -0.5 * viscosity * sol_val;
                  const VectorizedArray<number> JxW =
                    (face_jac_det * face_quadrature_weights[q]);
                  for (unsigned int comp = 0; comp < dim; ++comp)
                    {
                      values_face[side][comp][q] = test_by_value[comp] * JxW;
                      normal_derivative_face[side][comp][q] =
                        test_by_gradient[comp] * JxW * normal_x_jac[face_direction];
                      tang_derivatives_face[comp][0][q] =
                        test_by_gradient[comp] * JxW * normal_x_jac[first];
                      if constexpr (dim > 2)
                        tang_derivatives_face[comp][1][q] =
                          test_by_gradient[comp] * JxW * normal_x_jac[second];
                    }
                }
            for (unsigned int comp = 0; comp < dim; ++comp)
              {
                eval.template gradients<0, false, true>(tang_derivatives_face[comp][0],
                                                        values_face[side][comp]);
                if constexpr (dim > 2)
                  eval.template gradients<1, false, true>(tang_derivatives_face[comp][1],
                                                          values_face[side][comp]);
              }
          }
        for (unsigned int i1 = 0, q = 0; i1 < (dim > 2 ? n_q_points_1d : 1); ++i1)
          for (unsigned int i0 = 0; i0 < n_q_points_1d; ++i0, ++q)
            {
              unsigned int idx = 0;
              if (face_direction == 0)
                idx = n_q_points_1d * (n_q_points_1d * i1 + i0);
              else if (face_direction == 1)
                idx = i0 + i1 * n_q_points_1d * n_q_points_1d;
              else
                idx = i0 + i1 * n_q_points_1d;

              VectorizedArray<Number> *my_vals = dst + idx;
              for (unsigned int i = 0; i < n_q_points_1d; ++i)
                {
                  const std::array<Number, 4> interpolate = interpolate_to_face[i];
                  for (unsigned int comp = 0; comp < dim; ++comp)
                    my_vals[comp * n_q_points + i * stride] +=
                      values_face[0][comp][q] * interpolate[0] +
                      values_face[1][comp][q] * interpolate[2] +
                      normal_derivative_face[0][comp][q] * interpolate[1] +
                      normal_derivative_face[1][comp][q] * interpolate[3];
                }
            }
      }
  }

  MatrixFree<dim, number> matrix_free;

  number penalty_factor;
  number penalty_divergence;
  number penalty_continuity;
  number viscosity;
  number time_step;
  number time;
  number upwind_factor;


  unsigned int bdf_order;
  unsigned int fe_degree_u;
  bool         is_dg;
  bool         use_skew_symmetric_convective_formulation;
  bool         use_divergence_formulation;
  dealii::AlignedVector<dealii::VectorizedArray<Number>>    array_penalty_parameter;
  mutable Table<2, Tensor<1, dim, VectorizedArray<number>>> speeds_cells;
  mutable Table<2, Tensor<1, dim, VectorizedArray<number>>> speeds_cells_mass;
  mutable Table<2, Tensor<1, dim, VectorizedArray<number>>> speeds_faces;
  mutable Table<2, Tensor<1, dim, VectorizedArray<number>>> speeds_outer_faces;

  std::vector<std::array<Number, 4>> interpolate_to_face;
  std::vector<Number>                face_quadrature_weights;

  mutable dealii::AlignedVector<dealii::VectorizedArray<Number>>
    penalty_factor_divergence;
  mutable dealii::AlignedVector<dealii::VectorizedArray<Number>>
    penalty_factor_continuity;

  std::function<std::unique_ptr<Function<dim>>()> dirichletBC_velocity_factory;
  std::function<std::unique_ptr<Function<dim>>()> dirichletBC_pressure_factory;
  std::function<std::unique_ptr<Function<dim>>()> body_force_factory;

  std::unique_ptr<Function<dim>> velocity_bc_cached;
  std::unique_ptr<Function<dim>> pressure_bc_cached;
  std::unique_ptr<Function<dim>> rhs_cached;
};



template <int dim, typename number>
class PressureOperator : public EnableObserverPointer
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;
  using value_type = number;
  using size_type  = types::global_dof_index;

  PressureOperator() = default;

  void
  reinit(const MatrixFree<dim, number> &matrix_free_in,
         const unsigned int             bdf_order_in,
         const number                   time_step_in,
         const bool                     use_leray_projection_in)
  {
    bdf_order         = bdf_order_in;
    time_step         = time_step_in;
    this->matrix_free = &matrix_free_in;
    is_dg = matrix_free->get_dof_handler(dof_no_p).get_fe().n_dofs_per_vertex() == 0;
    use_leray_projection = use_leray_projection_in;

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

      const auto quadrature = reference_cells.get_gauss_type_quadrature(fe_degree + 1);
      dealii::FEValues<dim> fe_values(*mapping,
                                      fe,
                                      quadrature,
                                      dealii::update_JxW_values);

      const auto face_quadrature =
        reference_cells.face_reference_cell(0).get_gauss_type_quadrature(fe_degree + 1);
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

  bool
  get_use_leray_projection()
  {
    return use_leray_projection;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    if (is_dg)
      matrix_free->loop(&PressureOperator::local_apply_domain,
                        &PressureOperator::local_apply_inner_face,
                        &PressureOperator::local_apply_boundary_face,
                        this,
                        dst,
                        src,
                        true,
                        MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                        MatrixFree<dim, number>::DataAccessOnFaces::gradients);
    else
      {
        matrix_free->cell_loop(
          &PressureOperator::local_apply_domain, this, dst, src, true);
        for (const unsigned int i : matrix_free->get_constrained_dofs(dof_no_p))
          dst.local_element(i) = src.local_element(i);
      }
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
  get_system_matrix(const AffineConstraints<number> &constraints,
                    TrilinosWrappers::SparseMatrix  &system_matrix)
  {
    const auto &dof_handler = this->matrix_free->get_dof_handler(dof_no_p);

    TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_dofs(),
                                          dof_handler.locally_owned_dofs(),
                                          DoFTools::extract_locally_relevant_dofs(
                                            dof_handler),
                                          dof_handler.get_mpi_communicator());

    if (is_dg)
      DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, constraints);
    else
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);

    dsp.compress();
    system_matrix.reinit(dsp);
    system_matrix = 0.;

    if (is_dg)
      MatrixFreeTools::compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
        *matrix_free,
        constraints,
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
        constraints,
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

  void
  set_body_force_factory(std::function<std::unique_ptr<Function<dim>>()> &&body_force_in)
  {
    body_force_factory = std::move(body_force_in);
  }

  void
  set_DirichletBC_velocity_factory(
    std::function<std::unique_ptr<Function<dim>>()> &&dirichlet_bc_velocity_in)
  {
    dirichletBC_velocity_factory = std::move(dirichlet_bc_velocity_in);
  }

  void
  set_dirichletBC_pressure_factory(
    std::function<std::unique_ptr<Function<dim>>()> &&dirichlet_bc_pressure_in)
  {
    dirichletBC_pressure_factory = std::move(dirichlet_bc_pressure_in);
  }

  void
  set_viscosity(const number viscosity_in)
  {
    viscosity = viscosity_in;
  }


private:
  const MatrixFree<dim, number>                         *matrix_free;
  dealii::AlignedVector<dealii::VectorizedArray<number>> array_penalty_parameter;
  number                                                 time;
  number                                                 viscosity;
  double                                                 time_step;
  unsigned int                                           bdf_order;
  bool                                                   is_dg;
  bool                                                   use_leray_projection;

  std::function<std::unique_ptr<Function<dim>>()> dirichletBC_velocity_factory;
  std::function<std::unique_ptr<Function<dim>>()> dirichletBC_pressure_factory;
  std::function<std::unique_ptr<Function<dim>>()> body_force_factory;



  void
  local_apply_domain(const MatrixFree<dim, number>               &data,
                     VectorType                                  &dst,
                     const VectorType                            &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, 1, number> eval(data, dof_no_p, quad_no_p);

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

    FEFaceEvaluation<dim, -1, 0, 1, number> eval_minus(data, true, dof_no_p, quad_no_p);
    FEFaceEvaluation<dim, -1, 0, 1, number> eval_plus(data, false, dof_no_p, quad_no_p);

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
    FEFaceEvaluation<dim, -1, 0, 1, number> eval_minus(data, true, dof_no_p, quad_no_p);

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

    // AnalyticalRHS<dim> rhs(u_x_max, viscosity);
    auto rhs = body_force_factory();
    rhs->set_time(time);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_p.reinit(cell);

        // loop over quadrature points and compute the local volume flux
        for (const unsigned int q : eval_p.quadrature_point_indices())
          {
            const auto f = evaluate_function((*rhs), eval_p.quadrature_point(q));
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

    auto rhs = body_force_factory();
    rhs->set_time(time);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_p_minus.reinit(face);
        eval_p_plus.reinit(face);

        for (const unsigned int q : eval_p_minus.quadrature_point_indices())
          {
            const auto f = evaluate_function((*rhs), eval_p_minus.quadrature_point(q));
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

    auto velocity_bc = dirichletBC_velocity_factory();
    velocity_bc->set_time(time);
    auto pressure_bc = dirichletBC_pressure_factory();
    pressure_bc->set_time(time);
    auto rhs = body_force_factory();
    rhs->set_time(time);

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

                velocity_bc->set_time(time);
                const auto g =
                  evaluate_function((*velocity_bc), eval_p_minus.quadrature_point(q));
                auto u_plus =
                  make_vectorized_array(integration_constants.get_gamma0() / time_step) *
                  g;

                if (!use_leray_projection)
                  for (unsigned int i = 0; i < integration_constants.get_order(); ++i)
                    {
                      velocity_bc->set_time(time - (i + 1) * time_step);
                      u_plus -= make_vectorized_array(integration_constants.get_alpha(i) /
                                                      time_step) *
                                evaluate_function((*velocity_bc),
                                                  eval_p_minus.quadrature_point(q));
                    }

                const auto flux = (-u_plus) * normal;

                Tensor<1, dim, VectorizedArray<number>> curl_omega;
                if constexpr (dim == 3)
                  curl_omega = eval_vorticity.get_curl(q);
                else
                  {
                    // curl is a scalar in 2d, but evaluator still returns a vector where
                    // only the first component is non-zero
                    curl_omega[0] =
                      eval_vorticity.get_gradient(q)[0][1]; //   d(omega)/dx2
                    curl_omega[1] =
                      -eval_vorticity.get_gradient(q)[0][0]; // - d(omega)/dx1
                  }

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
                const auto f =
                  evaluate_function((*rhs), eval_p_minus.quadrature_point(q));

                const auto normal = eval_p_minus.normal_vector(q);

                const auto flux = -f * normal;

                const auto g_p =
                  evaluate_scalar_function((*pressure_bc),
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
    FEEvaluation<dim, -1, 0, 1, number>   eval_p(data, dof_no_p, quad_no_v);
    FEEvaluation<dim, -1, 0, dim, number> eval_u(data, dof_no_v, quad_no_v);

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
    FEFaceEvaluation<dim, -1, 0, 1, number> eval_p_minus(data, true, dof_no_p, quad_no_v);
    FEFaceEvaluation<dim, -1, 0, 1, number> eval_p_plus(data, false, dof_no_p, quad_no_v);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_minus(data,
                                                           true,
                                                           dof_no_v,
                                                           quad_no_v);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_plus(data,
                                                          false,
                                                          dof_no_v,
                                                          quad_no_v);

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
            const auto gradu_u_minus =
              eval_u_minus.get_gradient(q) * eval_u_minus.get_value(q);
            const auto gradu_u_plus =
              eval_u_plus.get_gradient(q) * eval_u_plus.get_value(q);
            const auto convective_flux =
              number(0.5) * (gradu_u_minus + gradu_u_plus) * normal;

            eval_p_minus.submit_value(convective_flux, q);
            eval_p_plus.submit_value(-convective_flux, q);
          }

        eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
        eval_p_plus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  void
  local_convective_boundary_face(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, number> eval_p_minus(data, true, dof_no_p, quad_no_v);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_minus(data,
                                                           true,
                                                           dof_no_v,
                                                           quad_no_v);

    auto dirichlet_bc_velocity = dirichletBC_velocity_factory();
    dirichlet_bc_velocity->set_time(time);
    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        if (data.get_boundary_id(face) == 0 || data.get_boundary_id(face) == 2)
          {
            eval_p_minus.reinit(face);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
              {
                eval_p_minus.submit_value({}, q);
              }

            eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
          }
        else
          {
            eval_p_minus.reinit(face);
            eval_u_minus.reinit(face);

            eval_u_minus.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           EvaluationFlags::gradients);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
              {
                const auto normal = eval_p_minus.normal_vector(q);

                dirichlet_bc_velocity->set_time(time);
                const auto grad_u = eval_u_minus.get_gradient(q);
                // evaluate_tensor_function((*dirichlet_bc_velocity),
                //                         eval_p_minus.quadrature_point(q));
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
    FEEvaluation<dim, -1, 0, 1, number>   eval_p(data, dof_no_p, quad_no_v_mass);
    FEEvaluation<dim, -1, 0, dim, number> eval_u(data, dof_no_v, quad_no_v_mass);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval_p.reinit(cell);
        eval_u.reinit(cell);

        eval_u.gather_evaluate(src, EvaluationFlags::values);

        // loop over quadrature points and compute the local volume flux
        for (const unsigned int q : eval_p.quadrature_point_indices())
          {
            const auto u = -eval_u.get_value(q);
            eval_p.submit_gradient(u, q);
          }

        // multiply by nabla v^h(x) and sum
        eval_p.integrate_scatter(EvaluationFlags::gradients, dst);
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
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_minus(data,
                                                         true,
                                                         dof_no_p,
                                                         quad_no_v_mass);
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_plus(data,
                                                        false,
                                                        dof_no_p,
                                                        quad_no_v_mass);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_minus(data,
                                                           true,
                                                           dof_no_v,
                                                           quad_no_v_mass);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_plus(data,
                                                          false,
                                                          dof_no_v,
                                                          quad_no_v_mass);

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
              0.5 * (eval_u_minus.get_value(q) + eval_u_plus.get_value(q)) * normal;

            eval_p_minus.submit_value(div_factor, q);
            eval_p_plus.submit_value(-div_factor, q);
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
    FEFaceEvaluation<dim, -1, 0, 1, number>   eval_p_minus(data,
                                                         true,
                                                         dof_no_p,
                                                         quad_no_v_mass);
    FEFaceEvaluation<dim, -1, 0, dim, number> eval_u_minus(data,
                                                           true,
                                                           dof_no_v,
                                                           quad_no_v_mass);

    auto velocity_bc = dirichletBC_velocity_factory();
    velocity_bc->set_time(time);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        if (data.get_boundary_id(face) == 0 || data.get_boundary_id(face) == 2)
          {
            eval_p_minus.reinit(face);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
              {
                eval_p_minus.submit_value({}, q);
              }

            eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
          }
        else
          {
            eval_p_minus.reinit(face);
            eval_u_minus.reinit(face);

            eval_u_minus.gather_evaluate(src, EvaluationFlags::values);

            for (const unsigned int q : eval_p_minus.quadrature_point_indices())
              {
                const auto value_flux =
                  eval_u_minus.get_value(q) * eval_u_minus.normal_vector(q);
                eval_p_minus.submit_value(value_flux, q);
              }

            eval_p_minus.integrate_scatter(EvaluationFlags::values, dst);
          }
      }
  }
};



template <int dim_, int n_components = dim_, typename Number = double>
class PenaltyOperator : public EnableObserverPointer
{
public:
  using value_type = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  static const int dim = dim_;

  void
  reinit(const MatrixFree<dim, Number> &matrix_free,
         const Number                   penalty_divergence_in = 1.0,
         const Number                   penalty_continuity_in = 1.0)
  {
    this->matrix_free        = &matrix_free;
    this->penalty_divergence = penalty_divergence_in;
    this->penalty_continuity = penalty_continuity_in;

    fe_degree_u = matrix_free.get_dof_handler(dof_no_v).get_fe().degree;

    unsigned int n_cells =
      matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    penalty_factor_divergence.resize(n_cells);
    penalty_factor_continuity.resize(n_cells);
  }

  virtual void
  set_divergence_penalty(Number penalty_divergence_in)
  {
    penalty_divergence = penalty_divergence_in;
  }

  virtual void
  set_continuity_penalty(Number penalty_continuity_in)
  {
    penalty_continuity = penalty_continuity_in;
  }

  virtual void
  vmult(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free->loop(&PenaltyOperator::do_cell_integral_range,
                            &PenaltyOperator::do_face_integral_range,
                            &PenaltyOperator::do_boundary_integral_range,
                            this,
                            dst,
                            src,
                            true,
                            MatrixFree<dim, Number>::DataAccessOnFaces::values,
                            MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  virtual void
  rhs(VectorType &dst, const VectorType &speed) const
  {
    this->matrix_free->cell_loop(
      &PenaltyOperator::do_rhs_cell_integral_range, this, dst, speed, true);
  }


private:
  void
  do_cell_integral_range(const MatrixFree<dim, Number>               &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim, -1, 0, n_components, Number> integrator(matrix_free,
                                                              dof_no_v,
                                                              quad_no_v_mass);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);
        integrator.gather_evaluate(src,
                                   EvaluationFlags::values | EvaluationFlags::gradients);

        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            const auto u     = integrator.get_value(q);
            const auto div_u = integrator.get_divergence(q);

            const auto divergence_penalty = penalty_factor_divergence[cell] * div_u;

            integrator.submit_value(u, q);
            integrator.submit_divergence(divergence_penalty, q);
          }
        integrator.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                     dst);
      }
  }

  void
  do_face_integral_range(const MatrixFree<dim, Number>               &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_inner(matrix_free,
                                                                        true,
                                                                        dof_no_v,
                                                                        quad_no_v_mass);
    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_outer(matrix_free,
                                                                        false,
                                                                        dof_no_v,
                                                                        quad_no_v_mass);


    for (unsigned int face = range.first; face < range.second; ++face)
      {
        integrator_inner.reinit(face);
        integrator_inner.gather_evaluate(src, EvaluationFlags::values);
        integrator_outer.reinit(face);
        integrator_outer.gather_evaluate(src, EvaluationFlags::values);

        const VectorizedArray<Number> cont_pen =
          0.5 * (integrator_inner.read_cell_data(penalty_factor_continuity) +
                 integrator_outer.read_cell_data(penalty_factor_continuity));

        for (const unsigned int q : integrator_inner.quadrature_point_indices())
          {
            const auto normal = integrator_inner.normal_vector(q);

            const Tensor<1, dim, VectorizedArray<Number>> solution_jump =
              (integrator_inner.get_value(q) - integrator_outer.get_value(q));

            const auto continuity_penalty_value =
              cont_pen * (solution_jump * normal) * normal;

            integrator_inner.submit_value(continuity_penalty_value, q);
            integrator_outer.submit_value(-continuity_penalty_value, q);
          }

        integrator_inner.integrate_scatter(EvaluationFlags::values, dst);
        integrator_outer.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  void
  do_boundary_integral_range(const MatrixFree<dim, Number>               &matrix_free,
                             VectorType                                  &dst,
                             const VectorType                            &src,
                             const std::pair<unsigned int, unsigned int> &range) const
  {
    FEFaceEvaluation<dim, -1, 0, n_components, Number> integrator_inner(matrix_free,
                                                                        true,
                                                                        dof_no_v,
                                                                        quad_no_v_mass);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        // Dirichlet boundary
        if (matrix_free.get_boundary_id(face) == 0 ||
            matrix_free.get_boundary_id(face) == 2)
          {
            integrator_inner.reinit(face);
            integrator_inner.gather_evaluate(src, EvaluationFlags::values);

            const VectorizedArray<Number> cont_pen =
              integrator_inner.read_cell_data(penalty_factor_continuity);
            for (unsigned int q = 0; q < integrator_inner.n_q_points; ++q)
              {
                const auto normal = integrator_inner.normal_vector(q);

                const auto continuity_penalty_value =
                  2. * cont_pen * ((integrator_inner.get_value(q)) * normal) * normal;

                integrator_inner.submit_value(continuity_penalty_value, q);
              }
            integrator_inner.integrate_scatter(EvaluationFlags::values, dst);
          }
        else if (matrix_free.get_boundary_id(face) == 1)
          {
            // Nothing to do
            // Convective term cancels and viscous term is only inhomogeneous
          }
        else
          AssertThrow(false,
                      ExcNotImplemented(
                        "Boundary id " +
                        std::to_string(int(matrix_free.get_boundary_id(face))) +
                        " not known"));
      }
  }

  void
  do_rhs_cell_integral_range(const MatrixFree<dim, Number>               &matrix_free,
                             VectorType                                  &dst,
                             const VectorType                            &src,
                             const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim, -1, 0, n_components, Number> integrator_speed(matrix_free,
                                                                    dof_no_v,
                                                                    quad_no_v_mass);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator_speed.reinit(cell);
        integrator_speed.gather_evaluate(src, EvaluationFlags::values);

        Tensor<1, dim, VectorizedArray<Number>> avg_velocity;
        VectorizedArray<Number>                 volume(0.);
        for (unsigned int q = 0; q < integrator_speed.n_q_points; ++q)
          {
            const auto u = integrator_speed.get_value(q);
            volume += integrator_speed.JxW(q);
            avg_velocity += u * integrator_speed.JxW(q);
            integrator_speed.submit_value(u, q);
          }
        const VectorizedArray<Number> avg_velocity_norm = avg_velocity.norm() / volume;
        penalty_factor_divergence[cell] = penalty_divergence * avg_velocity_norm *
                                          std::exp(std::log(volume) / (Number)dim) /
                                          ((Number)fe_degree_u + 1);
        penalty_factor_continuity[cell] = penalty_continuity * avg_velocity_norm;

        integrator_speed.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  const MatrixFree<dim, Number> *matrix_free;

  unsigned int fe_degree_u;

  Number penalty_divergence;
  Number penalty_continuity;

  mutable dealii::AlignedVector<dealii::VectorizedArray<Number>>
    penalty_factor_divergence;
  mutable dealii::AlignedVector<dealii::VectorizedArray<Number>>
    penalty_factor_continuity;
};
