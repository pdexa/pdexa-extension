// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "cellwise_preconditioner.hpp"

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/matrix_free/fe_evaluation.h>

namespace DGAdvection {

template<int dim, int fe_degree, typename Number, int vector_size>
CellwisePreconditionerFDM<dim, fe_degree, Number, vector_size>::CellwisePreconditionerFDM(
  std::shared_ptr<const gko::Executor> exec,
  const dealii::FullMatrix<std::complex<double>>* eigenvectors_,
  const dealii::FullMatrix<std::complex<double>>* inverse_eigenvectors_,
  const std::array<std::vector<std::complex<double>>, 2>& eigenvalues_,
  const std::vector<
    dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, dealii::VectorizedArray<Number, vector_size>>>& eval,
  const dealii::AlignedVector<dealii::Tensor<1, dim, dealii::VectorizedArray<Number, vector_size>>>&
    average_velocity_) :
    gko::batch_template::EnableBatchUserLinOp<Number, CellwisePreconditionerFDM>(
      exec,
      gko::batch_dim<2>{average_velocity_.size(), gko::dim<2>{dealii::Utilities::pow(fe_degree + 1, dim),
                                                              dealii::Utilities::pow(fe_degree + 1, dim)}}),
    eigenvectors(gko::batch::matrix::Dense<std::complex<Number>>::create(
      exec, gko::batch_dim<2>{1, gko::dim<2>{2 * (fe_degree + 1), fe_degree + 1}})),
    inverse_eigenvectors(gko::batch::matrix::Dense<std::complex<Number>>::create(
      exec, gko::batch_dim<2>{1, gko::dim<2>{2 * (fe_degree + 1), fe_degree + 1}})),
    determinants(exec->get_master(), average_velocity_.size() * vector_size),
    eigenvalues(gko::batch::matrix::Dense<std::complex<Number>>::create(
      exec, gko::batch_dim<2>{1, gko::dim<2>{2, fe_degree + 1}})),
    average_velocity(gko::batch::MultiVector<Number>::create(
      exec, gko::batch_dim<2>{average_velocity_.size(), gko::dim<2>{dim, vector_size}})),
    data_array{
      gko::batch::MultiVector<std::complex<Number>>::create(
                 exec,
                 gko::batch_dim<2>{average_velocity_.size(),
                                   gko::dim<2>{dealii::Utilities::pow(fe_degree + 1, dim), vector_size}}),
               gko::batch::MultiVector<std::complex<Number>>::create(
                 exec,
                 gko::batch_dim<2>{average_velocity_.size(),
                                   gko::dim<2>{dealii::Utilities::pow(fe_degree + 1, dim), vector_size}})} {
  constexpr int n = fe_degree + 1;

  dealii::Tensor<1, dim, Number> blend_factor_eig;

  auto eigenvectors_batch = gko::matrix::Dense<std::complex<Number>>::create(exec->get_master(), gko::dim<2>{2 * n, n});
  auto inverse_eigenvectors_batch =
    gko::matrix::Dense<std::complex<Number>>::create(exec->get_master(), gko::dim<2>{2 * n, n});
  auto average_velocity_batch = gko::matrix::Dense<Number>::create(exec->get_master(), gko::dim<2>{dim, vector_size});

  auto eigenvalue_batch = gko::matrix::Dense<std::complex<Number>>::create(exec->get_master(), gko::dim<2>{2, n});
  for (int i = 0; i < n; ++i) {
    eigenvalue_batch->at(0, i) = eigenvalues_[0][i];
    eigenvalue_batch->at(1, i) = eigenvalues_[1][i];
  }
  eigenvalues->create_view_for_item(0)->copy_from(eigenvalue_batch);

  for (int d: {0, 1}) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        eigenvectors_batch->at(i + d * n, j) = eigenvectors_[d](i, j);
        inverse_eigenvectors_batch->at(i + d * n, j) = inverse_eigenvectors_[d](i, j);
      }
    }
  }
  eigenvectors->create_view_for_item(0)->copy_from(eigenvectors_batch);
  inverse_eigenvectors->create_view_for_item(0)->copy_from(inverse_eigenvectors_batch);

  for (int cell = 0; cell < this->get_num_batch_items(); ++cell) {
    for (int d = 0; d < dim; ++d) {
      for (int k = 0; k < vector_size; ++k) { average_velocity_batch->at(d, k) = average_velocity_[cell][d][k];
      }
    }
    average_velocity->create_view_for_item(cell)->copy_from(average_velocity_batch);

    auto determinant = dealii::determinant(eval[cell].inverse_jacobian(0));
    for (int k = 0; k < vector_size; ++k) { determinants.get_data()[k + cell * vector_size] = determinant[k]; }
  }

  determinants.set_executor(exec);
}

#define DECLARE_CELLWISEPRECONDITIONERFDM(_dim, _vs)\
template class CellwisePreconditionerFDM<_dim, fe_degree, Number, _vs>

DECLARE_CELLWISEPRECONDITIONERFDM(2, dealii::VectorizedArray<Number>::size());
DECLARE_CELLWISEPRECONDITIONERFDM(2, 1);
DECLARE_CELLWISEPRECONDITIONERFDM(3, dealii::VectorizedArray<Number>::size());
DECLARE_CELLWISEPRECONDITIONERFDM(3, 1);

} // namespace DGAdvection
