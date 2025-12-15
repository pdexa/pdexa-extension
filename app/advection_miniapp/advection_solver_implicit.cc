// SPDX-FileCopyrightText: 2014-2022 Martin Kronbichler, Technical University of Munich
// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

// Program for time integration of the advection problem, realizing an
// implicit backward Euler integration with local solvers
//
// This program shares many similarities with the step-67 tutorial program of
// deal.II, see https://dealii.org/developer/doxygen/deal.II/step_67.html ,
// but it implements a simpler equation and is therefore ideal for learning
// about matrix-free evaluators.
//
// Compared to the main program advection_solver.cc, this program implements
// a variant with variable transport speed derived from an analytical
// expression. This file intentionally duplicates most of the other
// program to keep all implementation in a single file; for sustainable
// software design it would be advisable to share common code between the two
// cases.

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_idr.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/point_value_history.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include <cxxopts.hpp>

#include "pdexa-ext/ginkgo/core/solver/batch_bicgstab.hpp"
#include "pdexa-ext/ginkgo/core/matrix/batch_user_linop.hpp"

#include "apply_gko_solver.hpp"
#include "cellwise_operator.hpp"
#include "cellwise_preconditioner.hpp"
#include "definitions.hpp"


cxxopts::ParseResult args;


namespace DGAdvection {
using namespace dealii;


template<std::size_t align>
struct GkoAlignedAllocator : public gko::CpuAllocatorBase {
  void* allocate(gko::size_type num_bytes) override {
    return ::operator new(num_bytes, static_cast<std::align_val_t>(align));
  }
  void deallocate(void* ptr) override { ::operator delete(ptr, static_cast<std::align_val_t>(align)); }
};

// Analytical solution of the problem
template<int dim>
class ExactSolution : public Function<dim> {
  public:
  ExactSolution(const double time = 0.) : Function<dim>(1, time) {}

  virtual double value(const Point<dim>& p, const unsigned int /*component*/ = 0) const override {
    return value<double>(p);
  }

  template<typename Number>
  Number value(const Point<dim, Number>& p) const {
    return std::exp(-400. * ((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.75) * (p[1] - 0.75)));
  }
};


template<int dim>
class TransportSpeed {
  public:
  TransportSpeed(const double time) : time(time) {}

  template<typename Number>
  Tensor<1, dim, Number> value(const Point<dim, Number>& p) const {
    const double factor = std::cos(numbers::PI * time / FINAL_TIME) * 2.;
    Tensor<1, dim, Number> result;

    result[0] = factor * std::sin(2 * numbers::PI * p[1]) * std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[0]);
    result[1] =
        -factor * std::sin(2 * numbers::PI * p[0]) * std::sin(numbers::PI * p[1]) * std::sin(numbers::PI * p[1]);
    return result;
  }

  private:
  const double time;
};


// Description of curved mesh
template<int dim>
class DeformedCubeManifold : public ChartManifold<dim, dim, dim> {
  public:
  DeformedCubeManifold(const double left,
                       const double right,
                       const double deformation,
                       const unsigned int frequency = 1) :
      left(left), right(right), deformation(deformation), frequency(frequency) {}

  Point<dim> push_forward(const Point<dim>& chart_point) const override {
    double sinval = deformation;
    for (unsigned int d = 0; d < dim; ++d)
      sinval *= std::sin(frequency * numbers::PI * (chart_point(d) - left) / (right - left));
    Point<dim> space_point;
    for (unsigned int d = 0; d < dim; ++d) space_point(d) = chart_point(d) + sinval;
    return space_point;
  }

  Point<dim> pull_back(const Point<dim>& space_point) const override {
    Point<dim> x = space_point;
    Point<dim> one;
    for (unsigned int d = 0; d < dim; ++d) one(d) = 1.;

    // Newton iteration to solve the nonlinear equation given by the point
    Tensor<1, dim> sinvals;
    for (unsigned int d = 0; d < dim; ++d)
      sinvals[d] = std::sin(frequency * numbers::PI * (x(d) - left) / (right - left));

    double sinval = deformation;
    for (unsigned int d = 0; d < dim; ++d) sinval *= sinvals[d];
    Tensor<1, dim> residual = space_point - x - sinval * one;
    unsigned int its = 0;
    while (residual.norm() > 1e-12 && its < 100) {
      Tensor<2, dim> jacobian;
      for (unsigned int d = 0; d < dim; ++d) jacobian[d][d] = 1.;
      for (unsigned int d = 0; d < dim; ++d) {
        double sinval_der = deformation * frequency / (right - left) * numbers::PI *
                            std::cos(frequency * numbers::PI * (x(d) - left) / (right - left));
        for (unsigned int e = 0; e < dim; ++e)
          if (e != d) sinval_der *= sinvals[e];
        for (unsigned int e = 0; e < dim; ++e) jacobian[e][d] += sinval_der;
      }

      x += invert(jacobian) * residual;

      for (unsigned int d = 0; d < dim; ++d)
        sinvals[d] = std::sin(frequency * numbers::PI * (x(d) - left) / (right - left));

      sinval = deformation;
      for (unsigned int d = 0; d < dim; ++d) sinval *= sinvals[d];
      residual = space_point - x - sinval * one;
      ++its;
    }
    AssertThrow(residual.norm() < 1e-12, ExcMessage("Newton for point did not converge."));
    return x;
  }

  std::unique_ptr<Manifold<dim>> clone() const override {
    return std::make_unique<DeformedCubeManifold<dim>>(left, right, deformation, frequency);
  }

  private:
  const double left;
  const double right;
  const double deformation;
  const unsigned int frequency;
};

std::shared_ptr<const Utilities::MPI::Partitioner>
create_partitioner_multiple(const std::shared_ptr<const Utilities::MPI::Partitioner>& scalar_partitioner,
                            const unsigned int multiplicity) {
  IndexSet owned(multiplicity * scalar_partitioner->size());
  owned.add_range(multiplicity * scalar_partitioner->local_range().first,
                  multiplicity * scalar_partitioner->local_range().second);
  IndexSet ghosted(owned.size());
  for (auto it = scalar_partitioner->ghost_indices().begin_intervals();
       it != scalar_partitioner->ghost_indices().end_intervals(); ++it)
    ghosted.add_range(multiplicity * (*it->begin()), multiplicity * (it->last() + 1));
  return std::make_shared<Utilities::MPI::Partitioner>(owned, ghosted, scalar_partitioner->get_mpi_communicator());
}

template<int dim, int fe_degree, typename Number, typename VectorizedNumber>
void transform_to_collocation(const internal::MatrixFreeFunctions::UnivariateShapeData<Number>& shape,
                              const VectorizedNumber* src_ptr,
                              Number* dst) {
  internal::EvaluatorTensorProduct<internal::evaluate_general, dim, fe_degree + 1, fe_degree + 1, VectorizedNumber,
                                   Number>
      evaluator(nullptr, nullptr, nullptr);
  VectorizedNumber* dst_ptr = reinterpret_cast<VectorizedNumber*>(dst);

  const VectorizedNumber* in = src_ptr;
  VectorizedNumber* out = dst_ptr;
  // Need to select 'apply' method with hessian slot because values
  // assume symmetries that do not exist in the inverse shapes
  evaluator.template apply<0, true, false>(shape.inverse_shape_values.data(), in, out);
  if (dim > 1) evaluator.template apply<1, true, false>(shape.inverse_shape_values.data(), out, out);
  if (dim > 2) evaluator.template apply<2, true, false>(shape.inverse_shape_values.data(), out, out);
}

template<int dim, int fe_degree, typename Number, typename VectorizedNumber>
void transform_from_collocation(const internal::MatrixFreeFunctions::UnivariateShapeData<Number>& shape,
                                const Number* src,
                                VectorizedNumber* dst_ptr) {
  internal::EvaluatorTensorProduct<internal::evaluate_general, dim, fe_degree + 1, fe_degree + 1, VectorizedNumber,
                                   Number>
      evaluator(nullptr, nullptr, nullptr);
  const VectorizedNumber* src_ptr = reinterpret_cast<const VectorizedNumber*>(src);

  const VectorizedNumber* in = src_ptr;
  VectorizedNumber* out = dst_ptr;
  // Need to select 'apply' method with hessian slot because values
  // assume symmetries that do not exist in the inverse shapes
  evaluator.template apply<0, false, false>(shape.inverse_shape_values.data(), in, out);
  if (dim > 1) evaluator.template apply<1, false, false>(shape.inverse_shape_values.data(), out, out);
  if (dim > 2) evaluator.template apply<2, false, false>(shape.inverse_shape_values.data(), out, out);
}

// Implementation of the advection operation
template<int dim, int fe_degree, int vector_size>
class AdvectionOperation {
  public:
  using Number = double;
    using VectorizedNumber = VectorizedArray<Number, vector_size>;

  AdvectionOperation() : computing_times(5) {}

  void reinit(const DoFHandler<dim>& dof_handler);

  void initialize_dof_vector(LinearAlgebra::distributed::Vector<Number>& vec) { data.initialize_dof_vector(vec); }

  const MatrixFree<dim, Number, VectorizedNumber>& get_matrix_free() const { return data; }

  ~AdvectionOperation() {
    if (computing_times[2] > 0) {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Advection operator evaluated " << (std::size_t) computing_times[2] << " times." << std::endl
                  << "Time vmult (min / avg / max): ";
      Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(computing_times[0], MPI_COMM_WORLD);
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << data.min << " (proc_" << data.min_index << ") / " << data.avg << " / " << data.max << " (proc_"
                  << data.max_index << ")" << std::endl;
      data = Utilities::MPI::min_max_avg(computing_times[1], MPI_COMM_WORLD);
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Time rhs compute (min / avg / max): " << data.min << " (proc_" << data.min_index << ") / "
                  << data.avg << " / " << data.max << " (proc_" << data.max_index << ")" << std::endl;
      data = Utilities::MPI::min_max_avg(computing_times[3], MPI_COMM_WORLD);
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Time block-Jacobi prec (min / avg / max): " << data.min << " (proc_" << data.min_index << ") / "
                  << data.avg << " / " << data.max << " (proc_" << data.max_index << ")" << std::endl;
      data = Utilities::MPI::min_max_avg(computing_times[4], MPI_COMM_WORLD);
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Time block-Jacobi prec solver only (min / avg / max): " << data.min << " (proc_" << data.min_index
                  << ") / " << data.avg << " / " << data.max << " (proc_" << data.max_index << ")" << std::endl;
    }
  }

  void set_time(const double current_time, const double time_step) {
    this->time = current_time;
    this->time_step = time_step;
  }

  void compute_rhs(LinearAlgebra::distributed::Vector<Number>& dst,
                   const LinearAlgebra::distributed::Vector<Number>& src) const {
    Timer time;
    data.loop(&AdvectionOperation<dim, fe_degree, vector_size>::local_rhs_domain,
              &AdvectionOperation<dim, fe_degree, vector_size>::local_rhs_inner_face,
              &AdvectionOperation<dim, fe_degree, vector_size>::local_rhs_boundary_face, this, dst, src, true,
              MatrixFree<dim, Number, VectorizedNumber>::DataAccessOnFaces::values,
              MatrixFree<dim, Number, VectorizedNumber>::DataAccessOnFaces::values);
    dst *= -1. / time_step;
    computing_times[1] += time.wall_time();
  }

  void update_solution(LinearAlgebra::distributed::Vector<Number>& solution,
                       const LinearAlgebra::distributed::Vector<Number>& stage_solution) const {
    Timer time;
    DEAL_II_OPENMP_SIMD_PRAGMA
    for (unsigned int i = 0; i < stage_solution.locally_owned_size(); ++i) {
      Number value = stage_solution.local_element(i);
      solution.local_element(i) += time_step * value;
    }
    computing_times[1] += time.wall_time();
  }

  void vmult(LinearAlgebra::distributed::Vector<Number>& dst,
             const LinearAlgebra::distributed::Vector<Number>& src) const {
    Timer time;
    data.loop(&AdvectionOperation<dim, fe_degree, vector_size>::local_apply_domain,
              &AdvectionOperation<dim, fe_degree, vector_size>::local_apply_inner_face,
              &AdvectionOperation<dim, fe_degree, vector_size>::local_apply_boundary_face, this, dst, src, true,
              MatrixFree<dim, Number, VectorizedNumber>::DataAccessOnFaces::values,
              MatrixFree<dim, Number, VectorizedNumber>::DataAccessOnFaces::values);
    computing_times[0] += time.wall_time();
    ++computing_times[2];
  }

  void precondition_block_jacobi(LinearAlgebra::distributed::Vector<Number>& dst,
                                 const LinearAlgebra::distributed::Vector<Number>& src) const;

  void project_initial(LinearAlgebra::distributed::Vector<Number>& dst) const;

  Tensor<1, 3> compute_mass_and_energy(const LinearAlgebra::distributed::Vector<Number>& vec) const;

  private:
  MatrixFree<dim, Number, VectorizedNumber> data;
  double time;
  double time_step;

  mutable std::vector<double> computing_times;

  Table<2, Tensor<1, dim, VectorizedNumber>> speeds_cells;
  Table<2, Tensor<1, dim, VectorizedNumber>> speeds_faces;
  std::vector<Table<2, VectorizedNumber>> normal_speeds_faces;

  std::array<FullMatrix<std::complex<double>>, 2> eigenvectors, inverse_eigenvectors;
  std::array<std::vector<std::complex<double>>, 2> eigenvalues;
  AlignedVector<Tensor<1, dim, VectorizedNumber>> scaled_cell_velocity;

  std::shared_ptr<gko::Executor> exec =
      args["exec"].as<std::string>() == "cuda"
          ? std::shared_ptr<gko::Executor>(gko::CudaExecutor::create(
          0, gko::OmpExecutor::create(std::make_shared<GkoAlignedAllocator<vector_size * sizeof(Number)>>())))
      : std::shared_ptr<gko::Executor>(
          gko::ReferenceExecutor::create(std::make_shared<GkoAlignedAllocator<vector_size * sizeof(Number)>>()));

  std::shared_ptr<GkoCellwiseOperator<dim, fe_degree, Number, vector_size>> gko_cellwise_op;
  std::shared_ptr<CellwisePreconditionerFDM<dim, fe_degree, Number, vector_size>> gko_cellwise_precond;
  mutable std::shared_ptr<gko::batch::MultiVector<Number>> gko_src;
  mutable std::shared_ptr<gko::batch::MultiVector<Number>> gko_dst;

  void local_apply_domain(const MatrixFree<dim, Number, VectorizedNumber>& data,
                          LinearAlgebra::distributed::Vector<Number>& dst,
                          const LinearAlgebra::distributed::Vector<Number>& src,
                          const std::pair<unsigned int, unsigned int>& cell_range) const;

  void local_apply_inner_face(const MatrixFree<dim, Number, VectorizedNumber>& data,
                              LinearAlgebra::distributed::Vector<Number>& dst,
                              const LinearAlgebra::distributed::Vector<Number>& src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const;
  void local_apply_boundary_face(const MatrixFree<dim, Number, VectorizedNumber>& data,
                                 LinearAlgebra::distributed::Vector<Number>& dst,
                                 const LinearAlgebra::distributed::Vector<Number>& src,
                                 const std::pair<unsigned int, unsigned int>& cell_range) const;

  void local_rhs_domain(const MatrixFree<dim, Number, VectorizedNumber>& data,
                        LinearAlgebra::distributed::Vector<Number>& dst,
                        const LinearAlgebra::distributed::Vector<Number>& src,
                        const std::pair<unsigned int, unsigned int>& cell_range) const;

  void local_rhs_inner_face(const MatrixFree<dim, Number, VectorizedNumber>& data,
                            LinearAlgebra::distributed::Vector<Number>& dst,
                            const LinearAlgebra::distributed::Vector<Number>& src,
                            const std::pair<unsigned int, unsigned int>& cell_range) const;
  void local_rhs_boundary_face(const MatrixFree<dim, Number, VectorizedNumber>& data,
                               LinearAlgebra::distributed::Vector<Number>& dst,
                               const LinearAlgebra::distributed::Vector<Number>& src,
                               const std::pair<unsigned int, unsigned int>& cell_range) const;
};


template<int dim, int fe_degree, int vector_size>
void AdvectionOperation<dim, fe_degree, vector_size>::reinit(const DoFHandler<dim>& dof_handler) {
  MappingQGeneric<dim> mapping(fe_degree);
  Quadrature<1> quadrature = QGauss<1>(fe_degree + 1);
  if (use_gl_quad) quadrature = QGaussLobatto<1>(fe_degree + 1);
  Quadrature<1> quadrature_mass = QGauss<1>(fe_degree + 1);
  if (use_gl_quad_mass || use_gl_quad) quadrature_mass = QGaussLobatto<1>(fe_degree + 1);
  typename MatrixFree<dim, Number, VectorizedNumber>::AdditionalData additional_data;
  additional_data.overlap_communication_computation = false;
  additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points | update_values);
  additional_data.mapping_update_flags_inner_faces =
      (update_JxW_values | update_normal_vectors | update_quadrature_points | update_values);
  additional_data.mapping_update_flags_boundary_faces =
      (update_JxW_values | update_normal_vectors | update_quadrature_points | update_values);

  AffineConstraints<double> dummy;
  dummy.close();
  data.reinit(mapping, std::vector<const DoFHandler<dim>*>{&dof_handler},
              std::vector<const AffineConstraints<double>*>{&dummy},
              std::vector<Quadrature<1>>{{quadrature, quadrature_mass}}, additional_data);

  // precompute spatial part of variable advection speed, where we utilize
  // that the scaling in time is 1 for time t=0
  TransportSpeed<dim> transport_speed(0);
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval(data);
    speeds_cells.reinit(data.n_cell_batches(), eval.n_q_points);
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
      eval.reinit(cell);
      for (unsigned int q = 0; q < eval.n_q_points; ++q)
        speeds_cells(cell, q) = transport_speed.value(eval.quadrature_point(q));
    }
  }
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval(data, true);
    speeds_faces.reinit(data.n_inner_face_batches() + data.n_boundary_face_batches(), eval.n_q_points);
    for (unsigned int face = 0; face < data.n_inner_face_batches() + data.n_boundary_face_batches(); ++face) {
      eval.reinit(face);
      for (unsigned int q = 0; q < eval.n_q_points; ++q)
        speeds_faces(face, q) = transport_speed.value(eval.quadrature_point(q));
    }

    FEFaceValues<dim> fe_face_values(mapping, dof_handler.get_fe(), QGauss<dim - 1>(fe_degree + 1),
                                     update_normal_vectors | update_quadrature_points);
    normal_speeds_faces.resize(data.n_cell_batches());
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
      normal_speeds_faces[cell].reinit(2 * dim, eval.n_q_points);
      for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
        for (unsigned int f = 0; f < 2 * dim; ++f) {
          fe_face_values.reinit(data.get_cell_iterator(cell, v), f);
          for (unsigned int q = 0; q < eval.n_q_points; ++q)
            normal_speeds_faces[cell][f][q][v] =
                transport_speed.value(fe_face_values.quadrature_point(q)) * fe_face_values.normal_vector(q);
        }
    }
  }

  QGauss<1> gauss_quad(dof_handler.get_fe().degree + 1);
  FE_DGQArbitraryNodes<1> fe_1d(gauss_quad);
  constexpr unsigned int n = fe_degree + 1;
  for (unsigned int c = 0; c < 2; ++c) {
    LAPACKFullMatrix<double> deriv_matrix(n, n);
    for (unsigned int q = 0; q < n; ++q) {
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          deriv_matrix(i, j) -= fe_1d.shape_grad(i, gauss_quad.point(q))[0] *
                                fe_1d.shape_value(j, gauss_quad.point(q)) * gauss_quad.weight(q);
    }
    const double sign_advection = (c == 0) ? 1.0 : -1.0;
    for (unsigned int i = 0; i < n; ++i)
      for (unsigned int j = 0; j < n; ++j)
        deriv_matrix(i, j) += -fe_1d.shape_value(i, Point<1>()) * fe_1d.shape_value(j, Point<1>()) *
                                  (0.5 - flux_alpha * 0.5 * sign_advection) +
                              fe_1d.shape_value(i, Point<1>(1.0)) * fe_1d.shape_value(j, Point<1>(1.0)) *
                                  (0.5 + flux_alpha * 0.5 * sign_advection);

    for (unsigned int i = 0; i < n; ++i)
      for (unsigned int j = 0; j < n; ++j) deriv_matrix(i, j) *= (1. / gauss_quad.weight(i));
    deriv_matrix.compute_eigenvalues(true, false);

    eigenvalues[c].resize(n);
    for (unsigned int i = 0; i < n; ++i) eigenvalues[c][i] = deriv_matrix.eigenvalue(i);

    eigenvectors[c] = deriv_matrix.get_right_eigenvectors();
    inverse_eigenvectors[c] = eigenvectors[c];
    inverse_eigenvectors[c].gauss_jordan();
    for (unsigned int i = 0; i < n; ++i)
      for (unsigned int j = 0; j < n; ++j) inverse_eigenvectors[c](i, j) *= (1. / gauss_quad.weight(j));
  }
  scaled_cell_velocity.resize(data.n_cell_batches());
  std::vector<FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber>> eval(data.n_cell_batches(),
                                                                                             data);
  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
    eval[cell].reinit(cell);
    Tensor<1, dim, VectorizedNumber> average_velocity;
    VectorizedNumber cell_volume = {};
    for (unsigned int q = 0; q < eval[cell].n_q_points; ++q) {
      average_velocity += eval[cell].inverse_jacobian(q) * speeds_cells(cell, q) * eval[cell].JxW(q);
      cell_volume += eval[cell].JxW(q);
    }
    scaled_cell_velocity[cell] = average_velocity / cell_volume;
  }

  auto num_rows = static_cast<gko::size_type>(eval.front().dofs_per_cell);
  auto num_batches = static_cast<gko::size_type>(data.n_cell_batches());
  auto size = gko::batch_dim<2>{num_batches, gko::dim<2>{num_rows, num_rows}};
  auto vector_dim = gko::batch_dim<2>{num_batches, gko::dim<2>{num_rows, vector_size}};
  gko_cellwise_op = std::make_shared<GkoCellwiseOperator<dim, fe_degree, Number, vector_size>>(
    exec, size, data, eval, speeds_cells, normal_speeds_faces);
  gko_cellwise_precond = std::make_shared<CellwisePreconditionerFDM<dim, fe_degree, Number, vector_size>>(
    exec, eigenvectors.data(), inverse_eigenvectors.data(), eigenvalues, eval, scaled_cell_velocity);
  gko_src = gko::batch::MultiVector<Number>::create(exec, vector_dim);
  gko_dst = gko::batch::MultiVector<Number>::create(exec, vector_dim);
}

template<int dim, int fe_degree, int vector_size>
void AdvectionOperation<dim, fe_degree, vector_size>::local_apply_domain(
  const MatrixFree<dim, Number, VectorizedNumber>& data,
  LinearAlgebra::distributed::Vector<Number>& dst,
  const LinearAlgebra::distributed::Vector<Number>& src,
    const std::pair<unsigned int, unsigned int>& cell_range) const {
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval(data);
  const double inv_dt = 1. / time_step;

  const Number factor_time = std::cos(numbers::PI * (time + time_step) / FINAL_TIME);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
    eval.reinit(cell);

    // compute u^h(x) from src
    eval.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

    // loop over quadrature points and compute the local volume flux
    for (unsigned int q = 0; q < eval.n_q_points; ++q) {
      const auto speed = speeds_cells(cell, q);
      const auto u = eval.get_value(q);
      const auto gradu = eval.get_gradient(q);
      Tensor<1, dim, VectorizedNumber> volume_flux = ((-1.0 + factor_skew) * speed) * (factor_time * u);
      eval.submit_gradient(volume_flux, q);
      VectorizedNumber volume_val = ((factor_skew * speed) * gradu) * factor_time + inv_dt * u;
      eval.submit_value(volume_val, q);
    }

    // multiply by nabla v^h(x) and sum
    eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
  }
}

template<int dim, int fe_degree, int vector_size>
void AdvectionOperation<dim, fe_degree, vector_size>::local_apply_inner_face(
  const MatrixFree<dim, Number, VectorizedNumber>& data,
  LinearAlgebra::distributed::Vector<Number>& dst,
  const LinearAlgebra::distributed::Vector<Number>& src,
    const std::pair<unsigned int, unsigned int>& face_range) const {
  // On interior faces, we have two evaluators, one for the solution
  // 'u_minus' and one for the solution 'u_plus'. Note that the decision
  // about what is minus and plus is arbitrary at this point, so we must
  // assume that this can be arbitrarily oriented and we must only operate
  // with the generic quantities such as the normal vector.
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval_minus(data, true);
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval_plus(data, false);

  const Number factor_time = std::cos(numbers::PI * (time + time_step) / FINAL_TIME);

  for (unsigned int face = face_range.first; face < face_range.second; face++) {
    eval_minus.reinit(face);
    eval_plus.reinit(face);
    eval_minus.gather_evaluate(src, EvaluationFlags::values);
    eval_plus.gather_evaluate(src, EvaluationFlags::values);

    for (unsigned int q = 0; q < eval_minus.n_q_points; ++q) {
      const auto speed = speeds_faces(face, q);
      const auto u_minus = eval_minus.get_value(q);
      const auto u_plus = eval_plus.get_value(q);
      const auto normal_vector_minus = eval_minus.get_normal_vector(q);

      VectorizedNumber flux_minus;
      VectorizedNumber flux_plus;
      const auto normal_times_speed = (speed * normal_vector_minus) * factor_time;
      const auto flux_times_normal_of_u_minus = 0.5 * ((u_minus + u_plus) * normal_times_speed +
                                                       flux_alpha * std::abs(normal_times_speed) * (u_minus - u_plus));
      flux_minus = flux_times_normal_of_u_minus - factor_skew * normal_times_speed * u_minus;
      flux_plus = -flux_times_normal_of_u_minus + factor_skew * normal_times_speed * u_plus;

      eval_minus.submit_value(flux_minus, q);
      eval_plus.submit_value(flux_plus, q);
    }

    eval_minus.integrate_scatter(EvaluationFlags::values, dst);
    eval_plus.integrate_scatter(EvaluationFlags::values, dst);
  }
}

template<int dim, int fe_degree, int vector_size>
void AdvectionOperation<dim, fe_degree, vector_size>::local_apply_boundary_face(
  const MatrixFree<dim, Number, VectorizedNumber>& data,
  LinearAlgebra::distributed::Vector<Number>& dst,
  const LinearAlgebra::distributed::Vector<Number>& src,
    const std::pair<unsigned int, unsigned int>& face_range) const {
  AssertThrow(false, ExcNotImplemented());
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval_minus(data, true);
  ExactSolution<dim> solution(time + time_step);
  const Number factor_time = std::cos(numbers::PI * (time * time_step) / FINAL_TIME);

  for (unsigned int face = face_range.first; face < face_range.second; face++) {
    eval_minus.reinit(face);
    eval_minus.gather_evaluate(src, EvaluationFlags::values);

    for (unsigned int q = 0; q < eval_minus.n_q_points; ++q) {
      const auto speed = speeds_faces(face, q);
      // Dirichlet boundary
      const auto u_minus = eval_minus.get_value(q);
      const auto normal_vector = eval_minus.get_normal_vector(q);

      // Compute the outer solution value
      VectorizedNumber flux;
      const auto u_plus = solution.value(eval_minus.quadrature_point(q));

      // compute the flux
      const auto normal_times_speed = (normal_vector * speed) * factor_time;
      const auto flux_times_normal = 0.5 * ((u_minus + u_plus) * normal_times_speed +
                                            flux_alpha * std::abs(normal_times_speed) * (u_minus - u_plus));

      flux = flux_times_normal - factor_skew * normal_times_speed * u_minus;

      eval_minus.submit_value(flux, q);
    }

    eval_minus.integrate_scatter(EvaluationFlags::values, dst);
  }
}

template<int dim, int fe_degree, int vector_size>
void AdvectionOperation<dim, fe_degree, vector_size>::local_rhs_domain(
  const MatrixFree<dim, Number, VectorizedNumber>& data,
  LinearAlgebra::distributed::Vector<Number>& dst,
  const LinearAlgebra::distributed::Vector<Number>& src,
    const std::pair<unsigned int, unsigned int>& cell_range) const {
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval(data);
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval_src(data);

  Number factor_time = std::cos(numbers::PI * (time + time_step) / FINAL_TIME);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
    eval.reinit(cell);
    eval_src.reinit(cell);

    // compute u^h(x) from src
    eval_src.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

    // loop over quadrature points and compute the local volume flux
    for (unsigned int q = 0; q < eval.n_q_points; ++q) {
      const auto speed = speeds_cells(cell, q);
      const auto u = eval_src.get_value(q);
      const auto gradu = eval_src.get_gradient(q);
      Tensor<1, dim, VectorizedNumber> volume_flux = ((-1.0 + factor_skew) * speed * u) * factor_time;
      eval.submit_gradient(volume_flux, q);
      const VectorizedNumber volume_val = (factor_skew * (speed * gradu)) * factor_time;
      eval.submit_value(volume_val, q);
    }

    // multiply by nabla v^h(x) and sum
    eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
  }
}

template<int dim, int fe_degree, int vector_size>
void AdvectionOperation<dim, fe_degree, vector_size>::local_rhs_inner_face(
  const MatrixFree<dim, Number, VectorizedNumber>& data,
  LinearAlgebra::distributed::Vector<Number>& dst,
  const LinearAlgebra::distributed::Vector<Number>& src,
  const std::pair<unsigned int, unsigned int>& face_range) const {
  // On interior faces, we have two evaluators, one for the solution
  // 'u_minus' and one for the solution 'u_plus'. Note that the decision
  // about what is minus and plus is arbitrary at this point, so we must
  // assume that this can be arbitrarily oriented and we must only operate
  // with the generic quantities such as the normal vector.
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval_minus(data, true);
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval_plus(data, false);
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval_src_minus(data, true);
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval_src_plus(data, false);

  const Number factor_time = std::cos(numbers::PI * (time + time_step) / FINAL_TIME);

  for (unsigned int face = face_range.first; face < face_range.second; face++) {
    eval_minus.reinit(face);
    eval_plus.reinit(face);
    eval_src_minus.reinit(face);
    eval_src_plus.reinit(face);
    eval_src_minus.gather_evaluate(src, EvaluationFlags::values);
    eval_src_plus.gather_evaluate(src, EvaluationFlags::values);

    for (unsigned int q = 0; q < eval_minus.n_q_points; ++q) {
      const auto speed = speeds_faces(face, q);
      const auto u_minus = eval_src_minus.get_value(q);
      const auto u_plus = eval_src_plus.get_value(q);
      const auto normal_vector_minus = eval_minus.get_normal_vector(q);

      VectorizedNumber flux_minus;
      VectorizedNumber flux_plus;
      const auto normal_times_speed = (speed * normal_vector_minus) * factor_time;
      const auto flux_times_normal_of_u_minus = 0.5 * ((u_minus + u_plus) * normal_times_speed +
                                                       flux_alpha * std::abs(normal_times_speed) * (u_minus - u_plus));
      flux_minus = flux_times_normal_of_u_minus - factor_skew * normal_times_speed * u_minus;
      flux_plus = -flux_times_normal_of_u_minus + factor_skew * normal_times_speed * u_plus;

      eval_minus.submit_value(flux_minus, q);
      eval_plus.submit_value(flux_plus, q);
    }

    eval_minus.integrate_scatter(EvaluationFlags::values, dst);
    eval_plus.integrate_scatter(EvaluationFlags::values, dst);
  }
}

template<int dim, int fe_degree, int vector_size>
void AdvectionOperation<dim, fe_degree, vector_size>::local_rhs_boundary_face(
  const MatrixFree<dim, Number, VectorizedNumber>& data,
  LinearAlgebra::distributed::Vector<Number>& dst,
  const LinearAlgebra::distributed::Vector<Number>& src,
  const std::pair<unsigned int, unsigned int>& face_range) const {
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval_minus(data, true);
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval_src_minus(data, true);

  ExactSolution<dim> solution(time + time_step);
  const Number factor_time = std::cos(numbers::PI * (time + time_step) / FINAL_TIME);

  for (unsigned int face = face_range.first; face < face_range.second; face++) {
    eval_minus.reinit(face);
    eval_src_minus.reinit(face);
    eval_src_minus.gather_evaluate(src, EvaluationFlags::values);

    for (unsigned int q = 0; q < eval_minus.n_q_points; ++q) {
      const auto speed = speeds_faces(face, q);
      // Dirichlet boundary
      const auto u_minus = eval_src_minus.get_value(q);
      const auto normal_vector = eval_minus.get_normal_vector(q);

      // Compute the outer solution value
      VectorizedNumber flux;
      const auto u_plus = solution.value(eval_minus.quadrature_point(q));

      // compute the flux
      const auto normal_times_speed = (normal_vector * speed) * factor_time;
      const auto flux_times_normal = 0.5 * ((u_minus + u_plus) * normal_times_speed +
                                            flux_alpha * std::abs(normal_times_speed) * (u_minus - u_plus));

      flux = flux_times_normal - factor_skew * normal_times_speed * u_minus;

      eval_minus.submit_value(flux, q);
    }

    eval_minus.integrate_scatter(EvaluationFlags::values, dst);
  }
}

template<int dim, int fe_degree, int vector_size>
void AdvectionOperation<dim, fe_degree, vector_size>::project_initial(
  LinearAlgebra::distributed::Vector<Number>& dst) const {
  ExactSolution<dim> solution(0.);
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> phi(data);
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, Number, VectorizedNumber> inverse(phi);
#if DEAL_II_VERSION_GTE(9, 3, 0)
  dst.zero_out_ghost_values();
#else
  dst.zero_out_ghosts();
#endif
  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
    phi.reinit(cell);
    for (unsigned int q = 0; q < phi.n_q_points; ++q) phi.submit_dof_value(solution.value(phi.quadrature_point(q)), q);
    inverse.transform_from_q_points_to_basis(1, phi.begin_dof_values(), phi.begin_dof_values());
    phi.set_dof_values(dst);
  }
}

template<int dim, int fe_degree, int vector_size>
Tensor<1, 3> AdvectionOperation<dim, fe_degree, vector_size>::compute_mass_and_energy(
  const LinearAlgebra::distributed::Vector<Number>& vec) const {
  Tensor<1, 3> mass_energy = {};
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> phi(data);
  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
    phi.reinit(cell);
    phi.gather_evaluate(vec, EvaluationFlags::values | EvaluationFlags::gradients);
    VectorizedNumber mass = {};
    VectorizedNumber energy = {};
    VectorizedNumber H1semi = {};
    for (unsigned int q = 0; q < phi.n_q_points; ++q) {
      mass += phi.get_value(q) * phi.JxW(q);
      energy += phi.get_value(q) * phi.get_value(q) * phi.JxW(q);
      H1semi += (phi.get_gradient(q) * phi.get_gradient(q)) * phi.JxW(q);
    }
    for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v) {
      mass_energy[0] += mass[v];
      mass_energy[1] += energy[v];
      mass_energy[2] += H1semi[v];
    }
  }
  return Utilities::MPI::sum(mass_energy, vec.get_mpi_communicator());
}

template<typename VectorType>
class MyVectorMemory : public VectorMemory<VectorType> {
  public:
  MyVectorMemory() : first_unused(vectors.end()) {}

  virtual VectorType* alloc() override {
    if (first_unused == vectors.end()) {
      vectors.push_back(VectorType());
      return &vectors.back();
    } else {
      VectorType* return_value = &(*first_unused);
      ++first_unused;
      return return_value;
    }
  }

  virtual void free(const VectorType* const vector) override {
    typename std::list<VectorType>::iterator it = vectors.begin();
    while (&*it != vector) ++it;

    Assert(it != first_unused && vector == &*it, ExcInternalError());
    vectors.splice(first_unused, vectors, it);
    --first_unused;
  }

  private:
  std::list<VectorType> vectors;
  typename std::list<VectorType>::iterator first_unused;
};

template<int dim, int fe_degree, int vector_size>
void AdvectionOperation<dim, fe_degree, vector_size>::precondition_block_jacobi(
  LinearAlgebra::distributed::Vector<Number>& dst, const LinearAlgebra::distributed::Vector<Number>& src) const {
  Timer timer;
  const Number factor_time = std::cos(numbers::PI * (time + time_step) / FINAL_TIME);

  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedNumber> eval(data);

  auto num_rows = static_cast<gko::size_type>(eval.dofs_per_cell);
  auto num_batches = static_cast<gko::size_type>(data.n_cell_batches());

  AlignedVector<double> local_src;
  AlignedVector<double> local_dst;
  local_src.resize_fast(num_batches * num_rows * VectorizedNumber::size());
  local_dst.resize_fast(num_batches * num_rows * VectorizedNumber::size());

  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
    eval.reinit(cell);
    eval.read_dof_values(src);
    transform_to_collocation<dim, fe_degree>(data.get_shape_info().data[0], eval.begin_dof_values(),
                                             local_src.data() + num_rows * cell * VectorizedNumber::size());
  }

  gko_cellwise_op->inv_dt = 1. / time_step;
  gko_cellwise_op->time_factor = factor_time;

  gko_cellwise_precond->inv_dt = 1. / time_step;
  gko_cellwise_precond->time_factor = factor_time;

  auto gko_src_view = gko::batch::MultiVector<Number>::create_const(
    exec->get_master(), gko::batch_dim<2>{num_batches, gko::dim<2>{num_rows, vector_size}},
    gko::make_const_array_view(exec->get_master(), local_src.size(), local_src.data()));
  auto gko_dst_view = gko::batch::MultiVector<Number>::create(
    exec->get_master(), gko::batch_dim<2>{num_batches, gko::dim<2>{num_rows, vector_size}},
    gko::make_array_view(exec->get_master(), local_dst.size(), local_dst.data()));

  gko_src->copy_from(gko_src_view);
  gko_dst->fill(Number{});

  // extract to use different compiler
  Timer timer_solver;
  apply_gko_solver(gko_cellwise_op, gko_cellwise_precond, gko_src, gko_dst);
  computing_times[4] += timer_solver.wall_time();

  gko_dst_view->copy_from(gko_dst);

  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
    eval.reinit(cell);
    transform_from_collocation<dim, fe_degree>(data.get_shape_info().data[0], local_dst.data() + num_rows * cell * VectorizedNumber::size(),
                                               eval.begin_dof_values());
    eval.set_dof_values(dst);
  }
  computing_times[3] += timer.wall_time();
}

template<int dim, int vector_size>
class AdvectionProblem {
  public:
  using Number = typename AdvectionOperation<dim, fe_degree, vector_size>::Number;
    AdvectionProblem();
  void run();

  private:
    void make_grid();
    void setup_dofs();
    void output_results(const unsigned int timestep_number, const Tensor<1, 3> mass_and_energy);

  LinearAlgebra::distributed::Vector<Number> solution;

  std::shared_ptr<Triangulation<dim>> triangulation;
  MappingQGeneric<dim> mapping;
  FE_DGQArbitraryNodes<dim> fe;
  DoFHandler<dim> dof_handler;

  IndexSet locally_relevant_dofs;

  double time, time_step;

  ConditionalOStream pcout;
};

template<int dim, int vector_size>
AdvectionProblem<dim, vector_size>::AdvectionProblem() :
    mapping(fe_degree), fe(QGaussLobatto<1>(fe_degree + 1)), time(0), time_step(0),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
#ifdef DEAL_II_WITH_P4EST
  if (dim > 1)
    triangulation = std::make_shared<parallel::distributed::Triangulation<dim>>(MPI_COMM_WORLD);
  else
#endif
    triangulation = std::make_shared<Triangulation<dim>>();
}

template<int dim, int vector_size>
void AdvectionProblem<dim, vector_size>::make_grid() {
  time = 0;
  time_step = 0;
  triangulation->clear();
  Point<dim> p1;
  Point<dim> p2;
  for (unsigned int d = 0; d < dim; ++d) p2[d] = 1;
  std::vector<unsigned int> subdivisions(dim, 1);

  GridGenerator::subdivided_hyper_rectangle(*triangulation, subdivisions, p1, p2);

  if (periodic) {
    for (const auto& cell: triangulation->cell_iterators())
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->at_boundary(f)) cell->face(f)->set_all_boundary_ids(f);
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> periodic_faces;
    for (unsigned int d = 0; d < dim; ++d)
      GridTools::collect_periodic_faces(*triangulation, 2 * d, 2 * d + 1, d, periodic_faces);
    triangulation->add_periodicity(periodic_faces);
  }

  if (mesh_type == MeshType::deformed_cartesian) {
    DeformedCubeManifold<dim> manifold(0.0, 1.0, 0.12, 2);
    triangulation->set_all_manifold_ids(1);
    triangulation->set_manifold(1, manifold);

    std::vector<bool> vertex_touched(triangulation->n_vertices(), false);

    for (auto cell: triangulation->active_cell_iterators()) {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        if (vertex_touched[cell->vertex_index(v)] == false) {
          Point<dim>& vertex = cell->vertex(v);
          Point<dim> new_point = manifold.push_forward(vertex);
          vertex = new_point;
          vertex_touched[cell->vertex_index(v)] = true;
        }
      }
    }
  }

  triangulation->refine_global(args["refinement"].as<int>());

  pcout << "   Number of elements:            " << triangulation->n_global_active_cells() << std::endl;
}

template<int dim, int vector_size>
void AdvectionProblem<dim, vector_size>::setup_dofs() {
#if DEAL_II_VERSION_GTE(9, 3, 0)
  dof_handler.reinit(*triangulation);
  dof_handler.distribute_dofs(fe);
#else
  dof_handler.initialize(*triangulation, fe);
#endif

  if (time == 0.) {
    pcout << "   Polynomial degree:             " << dof_handler.get_fe().degree << std::endl;
    pcout << "   Number of degrees of freedom:  " << dof_handler.n_dofs() << std::endl;
  }

  double min_vertex_distance = std::numeric_limits<double>::max();
  for (const auto& cell: triangulation->active_cell_iterators())
    min_vertex_distance = std::min(min_vertex_distance, cell->minimum_vertex_distance());
  const double glob_min_vertex_distance = Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

  // Use hard-coded value for the maximal velocity of 2
  time_step = courant_number * glob_min_vertex_distance / 2.;

  time_step = FINAL_TIME / std::ceil(FINAL_TIME / time_step);

  if (time == 0)
    pcout << "   Time step size: " << time_step << ", minimum vertex distance: " << glob_min_vertex_distance
          << std::endl
          << std::endl;
}

template<int dim, int vector_size>
void AdvectionProblem<dim, vector_size>::output_results(const unsigned int output_number,
                                                        const Tensor<1, 3> mass_energy) {
  pcout << "   Time " << std::left << std::setw(6) << std::setprecision(3) << time << "  mass " << std::setprecision(10)
        << std::setw(16) << mass_energy[0] << "  energy " << std::setprecision(10) << std::setw(16) << mass_energy[1]
        << "  H1-semi " << std::setprecision(4) << std::setw(6) << mass_energy[2];

  if (!args["print-vtu"].as<bool>()) {
    return;
  }

  // Write output to a vtu file
  DataOut<dim> data_out;
  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches(mapping, fe_degree, DataOut<dim>::curved_inner_cells);

  const std::string filename = "solution_" + Utilities::int_to_string(output_number, 3) + ".vtu";
  data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
}

template<typename OperatorType>
class BlockJacobi {
  public:
  BlockJacobi(const OperatorType& operator_exemplar) : operator_exemplar(operator_exemplar) {}

  void vmult(LinearAlgebra::distributed::Vector<double>& dst,
             const LinearAlgebra::distributed::Vector<double>& src) const {
    operator_exemplar.precondition_block_jacobi(dst, src);
  }

  private:
  const OperatorType& operator_exemplar;
};

template<typename OperatorType, typename VectorType>
std::array<double, 5>
compute_least_squares_fit_neq(const OperatorType& op, const std::vector<VectorType>& vectors, const VectorType& rhs) {
  using Number = typename VectorType::value_type;
  std::vector<VectorType> tmp(vectors.size());
  dealii::FullMatrix<double> matrix(vectors.size(), vectors.size());
  AssertThrow(vectors.size() == 5, ExcNotImplemented());
  std::array<Number, 5> small_vector = {};
  unsigned int i = 0;
  for (; i < vectors.size(); ++i) {
    tmp[i].reinit(vectors[0], true);
    op.vmult(tmp[i], vectors[i]);
    for (unsigned int j = 0; j <= i; ++j) matrix(i, j) = tmp[i] * tmp[j];

    // compute row and column of Cholesky factorization
    for (unsigned int j = 0; j < i; ++j) {
      double inv_entry = matrix(i, j) / matrix(j, j);
      for (unsigned int k = j + 1; k <= i; ++k) matrix(i, k) -= matrix(k, j) * inv_entry;
    }
    if (matrix(i, i) < 1e-12 * matrix(0, 0) or matrix(0, 0) < 1e-30) break;
    small_vector[i] = tmp[i] * rhs;
    for (unsigned int j = 0; j < i; ++j) small_vector[i] -= matrix(i, j) / matrix(j, j) * small_vector[j];
  }
  // if (i > 0)
  // std::cout << std::setprecision(8) << matrix(i - 1, i - 1) << "  ";
  for (unsigned int s = i; s < small_vector.size(); ++s) small_vector[s] = 0.;
  for (int s = i - 1; s >= 0; --s) {
    double sum = small_vector[s];
    for (unsigned int j = s + 1; j < i; ++j) sum -= small_vector[j] * matrix(j, s);
    small_vector[s] = sum / matrix(s, s);
  }
  return small_vector;
}


template<int dim, int vector_size>
void AdvectionProblem<dim, vector_size>::run() {
  make_grid();
  setup_dofs();

  // Initialize the advection operator and the time integrator that will
  // perform all interesting steps
  AdvectionOperation<dim, fe_degree, vector_size> advection_operator;
  advection_operator.reinit(dof_handler);
  advection_operator.initialize_dof_vector(solution);
  /*
  const auto multiple_part =
    create_partitioner_multiple(solution.get_partitioner(), 4);
  LinearAlgebra::distributed::Vector<Number> sol2(multiple_part);
  pcout << "Vector sizes: " << solution.size() << " " << sol2.size() << " "
        << sol2.get_partitioner()->n_ghost_indices() << std::endl;
  */
  advection_operator.project_initial(solution);

  LinearAlgebra::distributed::Vector<Number> solution_copy = solution;
  LinearAlgebra::distributed::Vector<Number> rhs;
  rhs.reinit(solution);
  std::vector<LinearAlgebra::distributed::Vector<Number>> stage_sol(5, rhs);
  std::vector<LinearAlgebra::distributed::Vector<Number>> stage_mv(5, rhs);

  BlockJacobi<AdvectionOperation<dim, fe_degree, vector_size>> precondition(advection_operator);

  unsigned int n_output = 0;
  output_results(n_output++, advection_operator.compute_mass_and_energy(solution));
  pcout << std::endl;

  Timer timer;
  double prep_time = 0;
  double sol_time = 0;
  double output_time = 0;
  unsigned int timestep_number = 1;

  // This is the main time loop, asking the time integrator class to perform
  // the time step and update the content in the solution vector.
  while (time < FINAL_TIME - 1e-12) {
    timer.restart();

    advection_operator.set_time(time, time_step);

    advection_operator.compute_rhs(rhs, solution);

    // Compute upper triangular matrix with orthogonal factors of the
    // current matrix applied to old solutions of the linear system,
    // orthogonalized by the modified Gram-Schmidt process
    const std::array<double, 5> project_sol = compute_least_squares_fit_neq(advection_operator, stage_sol, rhs);

    // extrapolate solution from old values
    const unsigned int local_size = stage_sol[0].locally_owned_size();
    std::array<Number*, 5> vec_ptrs;
    for (unsigned int i = 0; i < vec_ptrs.size(); ++i) vec_ptrs[i] = stage_sol[i].begin();
    DEAL_II_OPENMP_SIMD_PRAGMA
    for (unsigned int i = 0; i < local_size; ++i) {
      const double sol_0 = vec_ptrs[0][i];
      const double sol_1 = vec_ptrs[1][i];
      const double sol_2 = vec_ptrs[2][i];
      const double sol_3 = vec_ptrs[3][i];
      const double sol_4 = vec_ptrs[4][i];
      vec_ptrs[4][i] = project_sol[0] * sol_0 + project_sol[1] * sol_1 + project_sol[2] * sol_2 +
                       project_sol[3] * sol_3 + project_sol[4] * sol_4;
    }
    for (unsigned int i = 4; i > 0; --i) std::swap(stage_sol[i], stage_sol[i - 1]);

    prep_time += timer.wall_time();
    timer.restart();

    const double rhs_norm = rhs.l2_norm();
    SolverControl control_fast(200, 1e-8 * rhs_norm);

    MyVectorMemory<LinearAlgebra::distributed::Vector<double>> memory;
    using SolverType = SolverFGMRES<LinearAlgebra::distributed::Vector<Number>>;
    typename SolverType::AdditionalData data;
    // data.exact_residual = false;
    SolverType solver(control_fast, memory, data);

    solver.solve(advection_operator, stage_sol[0], rhs, precondition);

    advection_operator.update_solution(solution, stage_sol[0]);

    time += time_step;
    timestep_number++;

    sol_time += timer.wall_time();

    timer.restart();

    if (static_cast<int>(time / output_tick) != static_cast<int>((time - time_step) / output_tick) ||
        time >= FINAL_TIME - 1e-12) {
      output_results(n_output++, advection_operator.compute_mass_and_energy(solution));
      pcout << " n iter " << control_fast.last_step() << " " << rhs_norm << " " << control_fast.initial_value() << " "
            << control_fast.last_value();
      for (const double s: project_sol) pcout << " " << s;
      pcout << std::endl;
    }
    output_time += timer.wall_time();
  }

  solution_copy -= solution;
  pcout << std::endl
        << "   Distance |final solution - initial_condition|: " << solution_copy.linfty_norm() << std::endl;

  pcout << std::endl << "   Performed " << timestep_number << " time steps." << std::endl;

  pcout << "   Average wall clock time per time step: " << (prep_time + sol_time) / timestep_number
        << "s, time per element: " << (prep_time + sol_time) / timestep_number / triangulation->n_global_active_cells()
        << "s" << std::endl;

  pcout << "   Spent " << output_time << "s on output, " << prep_time << "s on projection and " << sol_time
        << "s on solving." << std::endl;

  pcout << std::endl;

  // As 'advection_operator' goes out of scope, it will call its constructor
  // that prints the accumulated computing times over all time steps to
  // screen
}
} // namespace DGAdvection


int main(int argc, char** argv) {
  using namespace DGAdvection;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  cxxopts::Options options("DGAdvection", "DG advection problem");
  options.add_options()("h,help", "Print help")(
      "r,refinement", "Number of grid refinements, the number of elements is given by 2^(dim * n_global_refinements)",
      cxxopts::value<int>()->default_value("2"))("e,exec", "Ginkgo Executor",
                                                 cxxopts::value<std::string>()->default_value("reference"))(
    "p,print-vtu", "Print output to vtu file", cxxopts::value<bool>()->default_value("false"))(
    "v,vectorization", "Use vectorized kernels", cxxopts::value<bool>()->default_value("false"))(
    "d,dimension", "The dimension of the problem (2, 3)", cxxopts::value<int>()->default_value("2"));

  args = options.parse(argc, argv);

  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  try {
    deallog.depth_console(0);

    // The actual dimension is selected by inserting the global constant
    // 'dimension' as the actual template argument here, rather than the
    // placeholder 'dim' used as *template* in the class definitions above.
    auto dispatch_vectorization = []<int dim>(std::integral_constant<int, dim>) {
      if (args["vectorization"].as<bool>()) {
        AdvectionProblem<dim, VectorizedArray<Number>::size()> advect_problem;
        advect_problem.run();
      }
      else {
        AdvectionProblem<dim, 1> advect_problem;
        advect_problem.run();
      }
    };
    auto dispatch_dimension = [&] {
      auto dimension = args["dimension"].as<int>();
      if (dimension == 2) {
        dispatch_vectorization(std::integral_constant<int, 2>{});
      }
      if (dimension == 3) {
        dispatch_vectorization(std::integral_constant<int, 3>{});
      }
    };
    dispatch_dimension();
  }
  catch (std::exception& exc) {
    std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  } catch (...) {
    std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
