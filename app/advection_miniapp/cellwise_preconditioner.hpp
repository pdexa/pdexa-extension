// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include "complex.hpp"
#include "definitions.hpp"
#include "pdexa-ext/ginkgo/core/base/batch_struct.hpp"
#include "pdexa-ext/ginkgo/core/matrix/batch_struct.hpp"
#include "pdexa-ext/ginkgo/core/matrix/batch_user_linop.hpp"
#include "vectorization.hpp"

namespace dealii {

template<typename>
class FullMatrix;

template<int, int, int, int, typename, typename>
class FEEvaluation;

template<int, int, typename>
class Tensor;

template<typename>
class AlignedVector;

template<typename, std::size_t>
class VectorizedArray;
} // namespace dealii

namespace DGAdvection {

template<int dim, int fe_degree, typename Number, int vector_size>
class CellwisePreconditionerFDM
    : public gko::batch_template::EnableBatchUserLinOp<Number, CellwisePreconditionerFDM<dim, fe_degree, Number, vector_size>> {
public:
  explicit CellwisePreconditionerFDM(std::shared_ptr<const gko::Executor> exec) :
      gko::batch_template::EnableBatchUserLinOp<Number, CellwisePreconditionerFDM>(exec) {}

  CellwisePreconditionerFDM(
    std::shared_ptr<const gko::Executor> exec,
    const dealii::FullMatrix<std::complex<double>>* eigenvectors,
    const dealii::FullMatrix<std::complex<double>>* inverse_eigenvectors,
    const std::array<std::vector<std::complex<double>>, 2>& eigenvalues,
    const std::vector<dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, dealii::VectorizedArray<Number, vector_size>>>& eval,
    const dealii::AlignedVector<dealii::Tensor<1, dim, dealii::VectorizedArray<Number, vector_size>>>& average_velocity);

  std::shared_ptr<gko::batch::matrix::Dense<std::complex<Number>>> eigenvectors;
  std::shared_ptr<gko::batch::matrix::Dense<std::complex<Number>>> inverse_eigenvectors;
  gko::array<Number> determinants;
  std::shared_ptr<gko::batch::matrix::Dense<std::complex<Number>>> eigenvalues;
  std::shared_ptr<gko::batch::MultiVector<Number>> average_velocity;
  std::array<std::shared_ptr<gko::batch::MultiVector<std::complex<Number>>>, 2> data_array;
  Number inv_dt;
  Number time_factor;
};

template<int dim, int fe_degree, typename Number, int vector_size>
struct CellwisePreconditionerView {
  gko::int32 num_batch_items;
  gko::int32 num_rows;
  gko::batch::matrix::dense::uniform_batch<const std::complex<Number>> eigenvectors;
  gko::batch::matrix::dense::uniform_batch<const std::complex<Number>> inverse_eigenvectors;
  const Number* determinants;
  gko::batch::matrix::dense::uniform_batch<const std::complex<Number>> eigenvalues;
  gko::batch::multi_vector::uniform_batch<const Number> average_velocity;
  std::array<gko::batch::multi_vector::uniform_batch<std::complex<double>>, 2> data_array;
  Number inv_dt;
  Number time_factor;
};

template<int dim, int fe_degree, typename Number = double, typename VectorizedNumber = double>
struct CellwisePreconditionerItem {
  gko::int32 num_batch_items;
  gko::int32 num_rows;
  gko::batch::matrix::dense::batch_item<const std::complex<Number>> eigenvectors;
  gko::batch::matrix::dense::batch_item<const std::complex<Number>> inverse_eigenvectors;
  VectorizedNumber determinant;
  gko::batch::matrix::dense::batch_item<const std::complex<Number>> eigenvalues;
  gko::batch::multi_vector::batch_item<const VectorizedNumber> average_velocity;
  std::array<gko::batch::multi_vector::batch_item<std::complex<VectorizedNumber>>, 2> data_array;
  Number inv_dt;
  Number time_factor;
};

template<int dim, int fe_degree, typename Number, int vector_size>
[[nodiscard]] CellwisePreconditionerView<dim, fe_degree, Number, vector_size>
create_view(const CellwisePreconditionerFDM<dim, fe_degree, Number, vector_size>* op) {
  auto view = CellwisePreconditionerView<dim, fe_degree, Number, vector_size>{
    static_cast<gko::int32>(op->get_num_batch_items()),
    static_cast<gko::int32>(op->get_common_size()[0]),
    gko::batch::create_view(op->eigenvectors.get()),
    gko::batch::create_view(op->inverse_eigenvectors.get()),
    op->determinants.get_const_data(),
    gko::batch::create_view(op->eigenvalues.get()),
    gko::batch::create_view(op->average_velocity.get()),
    {gko::batch::create_view(op->data_array[0].get()), gko::batch::create_view(op->data_array[1].get())},
    op->inv_dt,
    op->time_factor};

  return view;
}

template<int dim, int fe_degree, typename Number, int vector_size, typename Mat>
constexpr CellwisePreconditionerItem<dim, fe_degree, Number, Number>
generate_batch_item(CellwisePreconditionerView<dim, fe_degree, Number, vector_size> view,
                    Mat mat,
                    char* buffer,
                    gko::int64 batch_id,
                    gko::cpu_kernel tag) {
  auto item = CellwisePreconditionerItem<dim, fe_degree, Number, Number>{
    view.num_batch_items,
    view.num_rows,
    gko::batch::extract_batch_item(view.eigenvectors, 0, tag),
    gko::batch::extract_batch_item(view.inverse_eigenvectors, 0, tag),
    view.determinants[batch_id],
    gko::batch::extract_batch_item(view.eigenvalues, 0, tag),
    gko::batch::extract_batch_item(view.average_velocity, batch_id, tag),
    {gko::batch::extract_batch_item(view.data_array[0], batch_id, tag),
     gko::batch::extract_batch_item(view.data_array[1], batch_id, tag)},
    view.inv_dt,
    view.time_factor};

  return item;
}

template<int dim, int fe_degree, typename Number, int vector_size, typename Mat>
constexpr CellwisePreconditionerItem<dim, fe_degree, Number, VectorizedNumber_t<vector_size>>
generate_batch_item(CellwisePreconditionerView<dim, fe_degree, Number, vector_size> view,
                    Mat mat,
                    char* buffer,
                    gko::int64 batch_id,
                    gko::cuda_kernel tag) {
  using VectorizedNumber_t = VectorizedNumber_t<vector_size>;
  std::array data_array = {
    gko::batch::multi_vector::uniform_batch < std::complex<VectorizedNumber_t>>{
      reinterpret_cast<std::complex<VectorizedNumber_t>*>(view.data_array[0].values),
      view.data_array[0].num_batch_items,
      view.data_array[0].stride / vector_size,
      view.data_array[0].num_rows,
      view.data_array[0].num_rhs / vector_size
    },
    gko::batch::multi_vector::uniform_batch < std::complex<VectorizedNumber_t>>{
      reinterpret_cast<std::complex<VectorizedNumber_t>*>(view.data_array[1].values),
      view.data_array[1].num_batch_items,
      view.data_array[1].stride / vector_size,
      view.data_array[1].num_rows,
      view.data_array[1].num_rhs / vector_size
    }
  };
  auto average_velocity = gko::batch::multi_vector::uniform_batch<const VectorizedNumber_t>{
    reinterpret_cast<const VectorizedNumber_t*>(view.average_velocity.values),
    view.average_velocity.num_batch_items,
    view.average_velocity.stride / vector_size,
    view.average_velocity.num_rows,
    view.average_velocity.num_rhs / vector_size
  };
  auto item = CellwisePreconditionerItem<dim, fe_degree, Number, VectorizedNumber_t>{
    view.num_batch_items,
    view.num_rows,
    gko::batch::extract_batch_item(view.eigenvectors, 0, tag),
    gko::batch::extract_batch_item(view.inverse_eigenvectors, 0, tag),
    reinterpret_cast<const VectorizedNumber_t*>(view.determinants)[batch_id],
    gko::batch::extract_batch_item(view.eigenvalues, 0, tag),
    gko::batch::extract_batch_item(average_velocity, batch_id, tag),
    {gko::batch::extract_batch_item(data_array[0], batch_id, tag),
     gko::batch::extract_batch_item(data_array[1], batch_id, tag)},
    view.inv_dt,
    view.time_factor};

  return item;
}

} // namespace DGAdvection
