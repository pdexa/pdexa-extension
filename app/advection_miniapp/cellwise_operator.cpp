// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "cellwise_operator.hpp"

#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "definitions.hpp"

namespace DGAdvection {
using namespace dealii;

template<int dim, int fe_degree, typename Number, int vector_size>
GkoCellwiseOperator<dim, fe_degree, Number, vector_size>::GkoCellwiseOperator(
  std::shared_ptr<const gko::Executor> exec,
  gko::batch_dim<2> size,
  const MatrixFree<dim, Number, dealii::VectorizedArray<Number, vector_size>>& data,
  const std::vector<FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number, VectorizedArray<Number, vector_size>>>& eval,
  const Table<2, Tensor<1, dim, VectorizedArray<Number, vector_size>>>& deal_speed_cells,
  const std::vector<Table<2, VectorizedArray<Number, vector_size>>>& deal_normal_speed_faces) :
    gko::batch_template::EnableBatchUserLinOp<Number, GkoCellwiseOperator>(exec, size),
    tmp_storage_(exec, size.get_num_batch_items() * size.get_common_size()[0] * 3 * vector_size),
    jac(gko::batch::matrix::Dense<Number>::create(
      exec, gko::batch_dim<2>(size.get_num_batch_items(), gko::dim<2>(dim, dim * vector_size)))),
    speed_cells(gko::batch::matrix::Dense<Number>::create(
      exec, gko::batch_dim<2>(size.get_num_batch_items(), gko::dim<2>(deal_speed_cells[0].size(), dim * vector_size)))),
    normal_speed_faces(gko::batch::matrix::Dense<Number>::create(
      exec,
      gko::batch_dim<2>(size.get_num_batch_items(),
                        gko::dim<2>(deal_normal_speed_faces.front().n_rows(),
                                    deal_normal_speed_faces.front().n_cols() * vector_size)))) {
  auto jac_values = gko::matrix::Dense<Number>::create(exec->get_master(), gko::dim<2>(dim, dim * vector_size));
  auto speed_cells_values = gko::matrix::Dense<Number>::create(exec->get_master(), speed_cells->get_common_size());
  auto normal_speed_faces_values =
    gko::matrix::Dense<Number>::create(exec->get_master(), normal_speed_faces->get_common_size());
  for (int batch_id = 0; batch_id < size.get_num_batch_items(); ++batch_id) {
    auto jac_batch = eval[batch_id].inverse_jacobian(0);
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        for (int k = 0; k < vector_size; ++k) { jac_values->at(i, k + j * vector_size) = jac_batch[i][j][k]; }
      }
    }
    jac->create_view_for_item(batch_id)->copy_from(jac_values);

    auto common_size = normal_speed_faces->get_common_size();
    for (int i = 0; i < common_size[0]; ++i) {
      for (int j = 0; j < common_size[1] / vector_size; ++j) {
        for (int k = 0; k < vector_size; ++k) {
          normal_speed_faces_values->at(i, k + j * vector_size) = deal_normal_speed_faces[batch_id][i][j][k];
        }
      }
    }
    normal_speed_faces->create_view_for_item(batch_id)->copy_from(normal_speed_faces_values);

    for (int i = 0; i < deal_speed_cells[0].size(); ++i) {
      for (int j = 0; j < dim; ++j) {
        for (int k = 0; k < vector_size; ++k) {
          speed_cells_values->at(i, k + j * vector_size) = deal_speed_cells[batch_id][i][j][k];
        }
      }
    }
    speed_cells->create_view_for_item(batch_id)->copy_from(speed_cells_values);
  }

  auto cell_quadrature = data.get_mapping_info().cell_data[0].descriptor[0].quadrature;
  cell_weights = gko::array<Number>(exec, cell_quadrature.get_weights().begin(), cell_quadrature.get_weights().end());

  auto face_quadrature = data.get_mapping_info().face_data[0].descriptor[0].quadrature;
  face_weights = gko::array<Number>(exec, face_quadrature.get_weights().begin(), face_quadrature.get_weights().end());

  auto deal_quadrature_data_on_face = data.get_shape_info().data[0].quadrature_data_on_face;
  quadrature_data_on_face = {
    {gko::array<Number>(exec, deal_quadrature_data_on_face[0].begin(), deal_quadrature_data_on_face[0].end()),
     gko::array<Number>(exec, deal_quadrature_data_on_face[1].begin(), deal_quadrature_data_on_face[1].end())}};

  auto deal_shape_values = data.get_shape_info().data[0].shape_values;
  shape_values = gko::array<Number>(exec, deal_shape_values.begin(), deal_shape_values.end());

  auto deal_shape_gradients_collocation = data.get_shape_info().data[0].shape_gradients_collocation;
  shape_gradients_collocation =
    gko::array<Number>(exec, deal_shape_gradients_collocation.begin(), deal_shape_gradients_collocation.end());
}

#define DECLARE_GKOCELLWISEOPERATOR(_dim, _vs)\
template class GkoCellwiseOperator<_dim, fe_degree, Number, _vs>

DECLARE_GKOCELLWISEOPERATOR(2, VectorizedArray<Number>::size());
DECLARE_GKOCELLWISEOPERATOR(2, 1);
DECLARE_GKOCELLWISEOPERATOR(3, VectorizedArray<Number>::size());
DECLARE_GKOCELLWISEOPERATOR(3, 1);

} // namespace DGAdvection
