// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include <ginkgo/core/matrix/batch_dense.hpp>
#include "pdexa-ext/ginkgo/core/base/view.hpp"
#include "pdexa-ext/ginkgo/core/matrix/batch_user_linop.hpp"

#include "definitions.hpp"
#include "vectorization.hpp"

// forward declaration to not pull in deal's vectorization config
namespace dealii {
template<int, typename, typename>
class MatrixFree;

template<int, int, int, int, typename, typename>
class FEEvaluation;

template<int, typename>
class Table;

template<int, int, typename>
class Tensor;

template<typename, std::size_t>
class VectorizedArray;
} // namespace dealii

namespace DGAdvection {
template<int dim, int fe_degree, typename Number, int vector_size>
class GkoCellwiseOperator
    : public gko::batch_template::EnableBatchUserLinOp<Number,
                                                       GkoCellwiseOperator<dim, fe_degree, Number, vector_size>> {
public:
  explicit GkoCellwiseOperator(std::shared_ptr<const gko::Executor> exec) :
      gko::batch_template::EnableBatchUserLinOp<Number, GkoCellwiseOperator>(std::move(exec)),
      tmp_storage_(this->get_executor()) {}

  explicit GkoCellwiseOperator(
      std::shared_ptr<const gko::Executor> exec,
    gko::batch_dim<2> size,
    const dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number, vector_size>>& data,
    const std::vector<
      dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, dealii::VectorizedArray<Number, vector_size>>>&
      eval,
    const dealii::Table<2, dealii::Tensor<1, dim, dealii::VectorizedArray<Number, vector_size>>>& deal_speed_cells,
    const std::vector<dealii::Table<2, dealii::VectorizedArray<Number, vector_size>>>& deal_normal_speed_faces) ;

  mutable gko::array<Number> tmp_storage_;

  std::shared_ptr<gko::batch::matrix::Dense<Number>> jac;
  std::array<gko::array<Number>, 2> quadrature_data_on_face;
  gko::array<Number> shape_values;
  gko::array<Number> shape_gradients_collocation;
  std::shared_ptr<gko::batch::matrix::Dense<Number>> speed_cells;
  std::shared_ptr<gko::batch::matrix::Dense<Number>> normal_speed_faces;
  gko::array<Number> cell_weights;
  gko::array<Number> face_weights;
  Number inv_dt;
  Number time_factor;
};

template<int dim, int fe_degree, typename Number, int vector_size>
struct GkoCellwiseOperatorView {
  gko::int32 num_batch_items;
  gko::int32 num_rows;

  mutable Number* tmp;

  gko::batch::matrix::dense::uniform_batch<const Number> jac;
  std::array<const Number*, 2> quadrature_data_on_face;
  const Number* shape_values;
  gko::batch::matrix::dense::batch_item<const Number> shape_gradients_collocation;
  gko::batch::matrix::dense::uniform_batch<const Number> speed_cells;
  gko::batch::matrix::dense::uniform_batch<const Number> normal_speed_faces;
  const Number* cell_weights;
  const Number* face_weights;
  Number inv_dt;
  Number time_factor;
};

template<int dim, int fe_degree, typename Number, typename VectorizedNumber>
struct GkoCellwiseOperatorItem {
  gko::int32 num_batch_items;
  gko::int32 num_rows;

  mutable VectorizedNumber* tmp;

  gko::batch::matrix::dense::batch_item<const VectorizedNumber> jac;
  std::array<const Number*, 2> quadrature_data_on_face;
  const Number* shape_values;
  gko::batch::matrix::dense::batch_item<const Number> shape_gradients_collocation;
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> speed_cells;
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> normal_speed_faces;
  const Number* cell_weights;
  const Number* face_weights;
  Number inv_dt;
  Number time_factor;
};

template<int dim, int fe_degree, typename Number, int vector_size>
[[nodiscard]] GkoCellwiseOperatorView<dim, fe_degree, Number, vector_size>
create_view(const GkoCellwiseOperator<dim, fe_degree, Number, vector_size>* op) {
  return GkoCellwiseOperatorView<dim, fe_degree, Number, vector_size>(
      static_cast<gko::int32>(op->get_num_batch_items()), static_cast<gko::int32>(op->get_common_size()[0]),
      op->tmp_storage_.get_data(), gko::batch::create_view(op->jac.get()),
      {{op->quadrature_data_on_face[0].get_const_data(), op->quadrature_data_on_face[1].get_const_data()}},
      op->shape_values.get_const_data(),
      {op->shape_gradients_collocation.get_const_data(), fe_degree + 1, fe_degree + 1, fe_degree + 1},
      gko::batch::create_view(op->speed_cells.get()), gko::batch::create_view(op->normal_speed_faces.get()),
      op->cell_weights.get_const_data(), op->face_weights.get_const_data(), op->inv_dt, op->time_factor);
}

template<int dim, int fe_degree, typename Number, int vector_size>
constexpr GkoCellwiseOperatorItem<dim, fe_degree, Number, Number> extract_batch_item(
  GkoCellwiseOperatorView<dim, fe_degree, Number, vector_size> op, gko::int64 batch_id, gko::cpu_kernel tag) {
  return GkoCellwiseOperatorItem<dim, fe_degree, Number, Number>(
    op.num_batch_items, op.num_rows, op.tmp + batch_id * op.num_rows * 3 * vector_size,
    gko::batch::extract_batch_item(op.jac, batch_id, tag), op.quadrature_data_on_face, op.shape_values,
    op.shape_gradients_collocation, gko::batch::extract_batch_item(op.speed_cells, batch_id, tag),
    gko::batch::extract_batch_item(op.normal_speed_faces, batch_id, tag), op.cell_weights, op.face_weights, op.inv_dt,
    op.time_factor);
}

template<int dim, int fe_degree, typename Number, int vector_size>
constexpr GkoCellwiseOperatorItem<dim, fe_degree, Number, VectorizedNumber_t<vector_size>> extract_batch_item(
  GkoCellwiseOperatorView<dim, fe_degree, Number, vector_size> op, gko::int64 batch_id, gko::cuda_kernel tag) {
  using VectorizedNumber_t = VectorizedNumber_t<vector_size>;
  auto jac = gko::batch::matrix::dense::uniform_batch(reinterpret_cast<const VectorizedNumber_t*>(op.jac.values),
                                                      op.jac.num_batch_items, op.jac.stride / vector_size,
                                                      op.jac.num_rows, op.jac.num_cols / vector_size);
  auto speed_cells = gko::batch::matrix::dense::uniform_batch(
    reinterpret_cast<const VectorizedNumber_t*>(op.speed_cells.values), op.speed_cells.num_batch_items,
    op.speed_cells.stride / vector_size, op.speed_cells.num_rows, op.speed_cells.num_cols / vector_size);
  auto normal_speed_faces = gko::batch::matrix::dense::uniform_batch(
    reinterpret_cast<const VectorizedNumber_t*>(op.normal_speed_faces.values), op.normal_speed_faces.num_batch_items,
    op.normal_speed_faces.stride / vector_size, op.normal_speed_faces.num_rows,
    op.normal_speed_faces.num_cols / vector_size);

  return GkoCellwiseOperatorItem<dim, fe_degree, Number, VectorizedNumber_t>(
    op.num_batch_items, op.num_rows, reinterpret_cast<VectorizedNumber_t*>(op.tmp) + batch_id * op.num_rows * 3,
    gko::batch::extract_batch_item(jac, batch_id, tag), op.quadrature_data_on_face, op.shape_values,
    op.shape_gradients_collocation, gko::batch::extract_batch_item(speed_cells, batch_id, tag),
    gko::batch::extract_batch_item(normal_speed_faces, batch_id, tag), op.cell_weights, op.face_weights, op.inv_dt,
    op.time_factor);
}

} // namespace DGAdvection
