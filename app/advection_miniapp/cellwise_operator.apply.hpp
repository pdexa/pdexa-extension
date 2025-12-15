// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <cub/cub.cuh>

#include <deal.II/base/utilities.h>
#include "cellwise_operator.hpp"
#include "definitions.hpp"
#include "tensor_helpers.hpp"
#include "tensor_kernel.cpu.hpp"
#include "tensor_kernel.gpu.hpp"

namespace DGAdvection {
namespace cpu {
template<int fe_degree, int dim, typename Number>
void face_kernel(gko::batch::matrix::dense::batch_item<const Number> jac,
                 gko::batch::matrix::dense::batch_item<const Number> normal_speed_faces,
                 const Number* shape_values,
                 const Number* weights,
                 Number time_factor,
                 gko::batch::multi_vector::batch_item<const Number> src,
                 gko::batch::multi_vector::batch_item<Number> dst) {
  const Number* src_ptr = src.values;
  Number* dst_ptr = dst.values;
  const unsigned int dofs_per_component = dealii::Utilities::pow(fe_degree + 1, dim);

  constexpr unsigned int stride = fe_degree + 1;
  constexpr unsigned int stride2 = stride * stride;
  auto constant = [](auto...) { return 0; };
  auto linear = [](auto i, auto...) { return i; };
  auto row_major = [](auto i, auto j, auto...) { return i * stride + j; };
  auto col_major = [](auto i, auto j, auto...) { return j * stride + i; };
  auto kji_major = [](auto i, auto j, auto k) { return k * stride2 + j * stride + i; };
  auto kij_major = [](auto i, auto j, auto k) { return k * stride2 + i * stride + j; };
  auto ikj_major = [](auto i, auto j, auto k) { return i * stride2 + k * stride + j; };

  for (unsigned int i = 0; i < dofs_per_component; ++i) { dst_ptr[i] = Number{}; }

  auto eval_shape = [&](auto d, auto idx, auto q_idx) {
    Number surface_JxW = 1.;
    for (unsigned int e = 0; e < dim; ++e) {
      if (d != e) { surface_JxW *= jac(e, e); }
    }
    surface_JxW = 1.0 / surface_JxW;

    for (unsigned int k = 0; k < (dim > 2 ? fe_degree + 1 : 1); ++k) {
      for (unsigned int j = 0; j < (dim > 1 ? fe_degree + 1 : 1); ++j) {
        Number sum_l = {}, sum_r = {};
        for (unsigned int i = 0; i < fe_degree + 1; ++i) {
          const Number value = src_ptr[idx(i, j, k)];
          sum_l += shape_values[i] * value;
          sum_r += shape_values[fe_degree - i] * value;
        }

        const unsigned int q = q_idx(j, k);
        {
          const auto speed = normal_speed_faces(2 * d, q) * time_factor;
          const auto coefficient = 0.5 * (speed + flux_alpha * std::abs(speed)) - factor_skew * speed;
          sum_l = sum_l * coefficient * (weights[q] * surface_JxW);
        }
        {
          const auto speed = normal_speed_faces(2 * d + 1, q) * time_factor;
          const auto coefficient = 0.5 * (speed + flux_alpha * std::abs(speed)) - factor_skew * speed;
          sum_r = sum_r * coefficient * (weights[q] * surface_JxW);
        }

        for (unsigned int i = 0; i < fe_degree + 1; ++i) {
          dst_ptr[idx(i, j, k)] += shape_values[i] * sum_l + shape_values[fe_degree - i] * sum_r;
        }
      }
    }
  };

  if constexpr (dim == 1) { eval_shape(0, linear, constant); }
  else if constexpr (dim == 2) {
    eval_shape(0, col_major, linear);
    eval_shape(1, row_major, linear);
  }
  else if constexpr (dim == 3) {
    eval_shape(0, kji_major, col_major);
    eval_shape(1, kij_major, row_major);
    eval_shape(2, ikj_major, col_major);
  }
}

template<int fe_degree, int dim, typename Number>
void cell_apply(gko::batch::matrix::dense::batch_item<const Number> jac,
                gko::batch::matrix::dense::batch_item<const Number> shape_gradients,
                gko::batch::matrix::dense::batch_item<const Number> speed_cells,
                const Number* cell_weights,
                Number time_factor,
                Number inv_dt,
                gko::batch::multi_vector::batch_item<const Number> in,
                gko::batch::multi_vector::batch_item<Number> out) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  constexpr auto dofs_per_component = dealii::Utilities::pow(fe_degree + 1, dim);
  std::array<std::array<Number, dofs_per_component>, dim> gradient_storage;
  std::array<tensor3d<Number>, dim> gradients{};
  for (uint32_t i = 0; i < dim; ++i) { gradients[i] = tensor3d(gradient_storage[i].data(), tensor_size); }

  auto in_tensor = tensor3d{in.values, tensor_size};
  auto out_tensor = tensor3d{out.values, tensor_size};

  cpu_tensor_apply<fe_degree, dim, 0, true, false>(shape_gradients, in_tensor, gradients[0]);
  if constexpr (dim > 1) { cpu_tensor_apply<fe_degree, dim, 1, true, false>(shape_gradients, in_tensor, gradients[1]); }
  if constexpr (dim > 2) { cpu_tensor_apply<fe_degree, dim, 2, true, false>(shape_gradients, in_tensor, gradients[2]); }

  const Number JxW = 1. / determinant<dim>(jac);
  for (unsigned int q = 0; q < dofs_per_component; ++q) {
    Number u = in[q];
    Number speed_gradu{};
    for (unsigned int d = 0; d < dim; ++d) { speed_gradu += speed_cells(q, d) * gradients[d].data[q] * jac(d, d); }
    Number flux = factor_skew * time_factor * speed_gradu;
    const Number result = (-1.0 + factor_skew) * time_factor * u * (JxW * cell_weights[q]);
    for (unsigned int d = 0; d < dim; ++d) { gradients[d].data[q] = result * speed_cells(q, d) * jac(d, d); }

    // mass matrix part
    u *= Number(inv_dt);
    flux += u;
    out[q] += flux * (JxW * cell_weights[q]);
  }

  cpu_tensor_apply<fe_degree, dim, 0, false, true>(shape_gradients, gradients[0].to_const(), out_tensor);
  if constexpr (dim > 1) {
    cpu_tensor_apply<fe_degree, dim, 1, false, true>(shape_gradients, gradients[1].to_const(), out_tensor);
  }
  if constexpr (dim > 2) {
    cpu_tensor_apply<fe_degree, dim, 2, false, true>(shape_gradients, gradients[2].to_const(), out_tensor);
  }
}
} // namespace cpu

template<int dim, int fe_degree, typename Number = double>
void simple_apply(GkoCellwiseOperatorItem<dim, fe_degree, Number, Number> a,
                  gko::batch::multi_vector::batch_item<const Number> b,
                  gko::batch::multi_vector::batch_item<Number> x,
                  gko::cpu_kernel) {
  cpu::face_kernel<fe_degree, dim>(a.jac, a.normal_speed_faces, a.quadrature_data_on_face[0], a.face_weights,
                                   a.time_factor, b, x);

  cpu::cell_apply<fe_degree, dim>(a.jac, a.shape_gradients_collocation, a.speed_cells, a.cell_weights, a.time_factor,
                                  a.inv_dt, b, x);
}

namespace gpu {
template<int fe_degree, int dim, typename Number1, typename Number2>
__device__ void face_kernel(gko::batch::matrix::dense::batch_item<const Number2> jac,
                            gko::batch::matrix::dense::batch_item<const Number2> normal_speed_faces,
                            const Number1* shape_values,
                            const Number1* weights,
                            Number1 time_factor,
                            gko::batch::multi_vector::batch_item<const Number2> src,
                            gko::batch::multi_vector::batch_item<Number2> dst) {
  constexpr unsigned int dofs_per_component = dealii::Utilities::pow(fe_degree + 1, dim);

  auto sid = threadIdx.x / (fe_degree + 1);
  constexpr auto num_subwarps = dofs_per_component / (fe_degree + 1);

  for (uint32_t tid = threadIdx.x; tid < dofs_per_component; tid += blockDim.x) {
    auto i = tid % (fe_degree + 1);
    auto j = (tid / (fe_degree + 1)) % (fe_degree + 1);
    auto k = tid / ((fe_degree + 1) * (fe_degree + 1));

    if (dim < 3 && k > 0) { continue; }
    if (dim < 2 && j > 0) { continue; }

    constexpr unsigned int stride = fe_degree + 1;
    constexpr unsigned int stride2 = stride * stride;
    auto constant = [](auto...) { return 0; };
    auto linear = [](auto ii, auto...) { return ii; };
    auto row_major = [](auto ii, auto jj, auto...) { return ii * stride + jj; };
    auto col_major = [](auto ii, auto jj, auto...) { return jj * stride + ii; };
    auto ijk = [](auto ii, auto jj, auto kk) { return kk * stride2 + jj * stride + ii; };
    auto jik = [](auto ii, auto jj, auto kk) { return kk * stride2 + ii * stride + jj; };
    auto jki = [](auto ii, auto jj, auto kk) { return ii * stride2 + kk * stride + jj; };

    dst[tid] = gko::zero<Number2>();

    auto eval_shape = [&](auto d, auto idx, auto q_idx) {
      auto surface_JxW = gko::one<Number2>();
      for (unsigned int e = 0; e < dim; ++e) {
        if (d != e) { surface_JxW = surface_JxW * jac(e, e); }
      }
      surface_JxW = 1.0 / surface_JxW;

      const auto value = src[idx(i, j, k)];

      using WarpReduce = cub::WarpReduce<Number2, fe_degree + 1>;
      using WarpScan = cub::WarpScan<Number2, fe_degree + 1>;
      union Storage {
        typename WarpReduce::TempStorage reduce[num_subwarps];
        typename WarpScan::TempStorage scan[num_subwarps];
      };
      __shared__ Storage storage;

      auto r_l = WarpReduce(storage.reduce[sid]).Sum(shape_values[i] * value);
      __syncthreads();
      r_l = WarpScan(storage.scan[sid]).Broadcast(r_l, 0);
      __syncthreads();

      auto r_r = WarpReduce(storage.reduce[sid]).Sum(shape_values[fe_degree - i] * value);
      __syncthreads();
      r_r = WarpScan(storage.scan[sid]).Broadcast(r_r, 0);

      const unsigned int q = q_idx(j, k);
      using std::abs; // enable ADL
      {
        const auto speed = normal_speed_faces(2 * d, q) * time_factor;
        const auto coefficient = 0.5 * (speed + flux_alpha * abs(speed)) - factor_skew * speed;
        r_l = r_l * coefficient * (weights[q] * surface_JxW);
      }
      {
        const auto speed = normal_speed_faces(2 * d + 1, q) * time_factor;
        const auto coefficient = 0.5 * (speed + flux_alpha * abs(speed)) - factor_skew * speed;
        r_r = r_r * coefficient * (weights[q] * surface_JxW);
      }

      dst[idx(i, j, k)] = dst[idx(i, j, k)] + shape_values[i] * r_l + shape_values[fe_degree - i] * r_r;
    };

    if constexpr (dim == 1) { eval_shape(0, linear, constant); }
    else if constexpr (dim == 2) {
      eval_shape(0, col_major, linear);
      __syncthreads();
      eval_shape(1, row_major, linear);
    }
    else if constexpr (dim == 3) {
      eval_shape(0, ijk, col_major);
      __syncthreads();
      eval_shape(1, jik, row_major);
      __syncthreads();
      eval_shape(2, jki, col_major);
    }
  }
}

namespace simple {
template<int fe_degree, int dim, bool force_global_storage = false, typename Number1, typename Number2>
__device__ void cell_apply(gko::batch::matrix::dense::batch_item<const Number2> jac,
                           gko::batch::matrix::dense::batch_item<const Number1> shape_gradients,
                           gko::batch::matrix::dense::batch_item<const Number2> speed_cells,
                           const Number1* cell_weights,
                           Number1 time_factor,
                           Number1 inv_dt,
                           gko::batch::multi_vector::batch_item<const Number2> in,
                           gko::batch::multi_vector::batch_item<Number2> out,
                           Number2* grad_buffer = nullptr) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  constexpr int dofs_per_component = dealii::Utilities::pow(fe_degree + 1, dim);
  std::array<tensor3d<Number2>, dim> gradients{};
  if constexpr (sizeof(Number2) > 8 || force_global_storage) {
    for (uint32_t i = 0; i < dim; ++i) { gradients[i] = tensor3d(grad_buffer + i * dofs_per_component, tensor_size); }
  }
  else {
    __shared__ Number2 gradient_storage[dim][dofs_per_component];
    for (uint32_t i = 0; i < dim; ++i) { gradients[i] = tensor3d(gradient_storage[i], tensor_size); }
  }

  auto in_tensor = tensor3d{in.values, tensor_size};
  auto out_tensor = tensor3d{out.values, tensor_size};

  simple_tensor_apply<fe_degree, dim, 0, true, false>(shape_gradients, in_tensor, gradients[0]);
  if constexpr (dim > 1) {
    simple_tensor_apply<fe_degree, dim, 1, true, false>(shape_gradients, in_tensor, gradients[1]);
  }
  if constexpr (dim > 2) {
    simple_tensor_apply<fe_degree, dim, 2, true, false>(shape_gradients, in_tensor, gradients[2]);
  }
  __syncthreads();

  const auto JxW = 1. / determinant<dim>(jac);
  for (unsigned int q = threadIdx.x; q < dofs_per_component; q += blockDim.x) {
    auto u = in[q];
    auto speed_gradu = gko::zero<Number2>();
    for (unsigned int d = 0; d < dim; ++d) {
      speed_gradu = speed_gradu + speed_cells(q, d) * gradients[d].data[q] * jac(d, d);
    }
    auto flux = factor_skew * time_factor * speed_gradu;
    const auto result = (-1.0 + factor_skew) * time_factor * u * (JxW * cell_weights[q]);
    for (unsigned int d = 0; d < dim; ++d) { gradients[d].data[q] = result * speed_cells(q, d) * jac(d, d); }

    // mass matrix part
    u = u * inv_dt;
    flux = flux + u;
    out[q] = out[q] + flux * (JxW * cell_weights[q]);
  }
  __syncthreads();

  simple_tensor_apply<fe_degree, dim, 0, false, true>(shape_gradients, gradients[0].to_const(), out_tensor);
  if constexpr (dim > 1) {
    __syncthreads();
    simple_tensor_apply<fe_degree, dim, 1, false, true>(shape_gradients, gradients[1].to_const(), out_tensor);
  }
  if constexpr (dim > 2) {
    __syncthreads();
    simple_tensor_apply<fe_degree, dim, 2, false, true>(shape_gradients, gradients[2].to_const(), out_tensor);
  }
}

template<int fe_degree, int dim, typename Number1, typename Number2>
__device__ void cell_apply_single_gradient(gko::batch::matrix::dense::batch_item<const Number2> jac,
                                           gko::batch::matrix::dense::batch_item<const Number1> shape_gradients,
                                           gko::batch::matrix::dense::batch_item<const Number2> speed_cells,
                                           const Number1* cell_weights,
                                           Number1 time_factor,
                                           Number1 inv_dt,
                                           gko::batch::multi_vector::batch_item<const Number2> in,
                                           gko::batch::multi_vector::batch_item<Number2> out) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  constexpr int dofs_per_component = dealii::Utilities::pow(fe_degree + 1, dim);
  __shared__ Number2 gradient_storage[dofs_per_component];
  tensor3d<Number2> gradient = tensor3d(gradient_storage, tensor_size);

  auto in_tensor = tensor3d{in.values, tensor_size};
  auto out_tensor = tensor3d{out.values, tensor_size};

  const auto JxW = 1. / determinant<dim>(jac);

  auto apply_mass_matrix = [&](auto d) {
    for (unsigned int q = threadIdx.x; q < dofs_per_component; q += blockDim.x) {
      auto speed_gradu = speed_cells(q, d) * gradient.data[q] * jac(d, d);
      auto flux = factor_skew * time_factor * speed_gradu;
      out[q] = out[q] + flux * JxW * cell_weights[q];

      auto u = in[q];
      const auto result = (-1.0 + factor_skew) * time_factor * u * (JxW * cell_weights[q]);
      gradient.data[q] = result * speed_cells(q, d) * jac(d, d);
    }
  };

  simple_tensor_apply<fe_degree, dim, 0, true, false>(shape_gradients, in_tensor, gradient);
  __syncthreads();
  apply_mass_matrix(0);
  __syncthreads();
  simple_tensor_apply<fe_degree, dim, 0, false, true>(shape_gradients, gradient.to_const(), out_tensor);

  if constexpr (dim > 1) {
    __syncthreads();
    simple_tensor_apply<fe_degree, dim, 1, true, false>(shape_gradients, in_tensor, gradient);
    __syncthreads();
    apply_mass_matrix(1);
    __syncthreads();
    simple_tensor_apply<fe_degree, dim, 1, false, true>(shape_gradients, gradient.to_const(), out_tensor);
  }

  if constexpr (dim > 2) {
    __syncthreads();
    simple_tensor_apply<fe_degree, dim, 2, true, false>(shape_gradients, in_tensor, gradient);
    __syncthreads();
    apply_mass_matrix(2);
    __syncthreads();
    simple_tensor_apply<fe_degree, dim, 2, false, true>(shape_gradients, gradient.to_const(), out_tensor);
  }

  __syncthreads();
  for (unsigned int q = threadIdx.x; q < dofs_per_component; q += blockDim.x) {
    auto u = in[q];
    out[q] = out[q] + u * inv_dt * JxW * cell_weights[q];
  }
}

} // namespace simple
namespace tensor {
template<int fe_degree, int dim, typename Number>
__device__ void cell_apply(row_major_matrix<const Number> jac,
                           row_major_matrix<const Number> shape_gradients,
                           row_major_matrix<const Number> speed_cells,
                           const Number* cell_weights,
                           Number time_factor,
                           Number inv_dt,
                           vector_item<const Number> in,
                           vector_item<Number> out) {
  using gko::batch::matrix::to_const;
  constexpr std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                                dim >= 3 ? fe_degree + 1 : 1};

  constexpr int dofs_per_component = tensor_size[0] * tensor_size[1] * tensor_size[2];
  __shared__ Number gradient_storage[dim][dofs_per_component];
  std::array<col_major_matrix<Number>, dim> gradients{};
  for (uint32_t i = 0; i < dim; ++i) {
    gradients[i] =
      col_major_matrix(&gradient_storage[i][0], tensor_size[0], tensor_size[0], tensor_size[1] * tensor_size[2]);
  }
  __shared__ Number buffer[2][dofs_per_component];
  std::array<col_major_matrix<Number>, 2> buffer_mat{
    col_major_matrix(buffer[0], tensor_size[0], tensor_size[0], tensor_size[1] * tensor_size[2]),
    col_major_matrix(buffer[1], tensor_size[0], tensor_size[0], tensor_size[1] * tensor_size[2])};

  auto in_mat = col_major_matrix(in.values, tensor_size[0], tensor_size[0], tensor_size[1] * tensor_size[2]);
  auto out_mat = col_major_matrix(out.values, tensor_size[0], tensor_size[0], tensor_size[1] * tensor_size[2]);

  auto shape_gradients_transposed = col_major_matrix(shape_gradients.values, shape_gradients.num_cols,
                                                     shape_gradients.num_cols, shape_gradients.num_rows);

  tensor_apply<1, fill_mode::overwrite>(shape_gradients_transposed, in_mat, gradients[0], buffer_mat);
  if constexpr (dim > 1) {
    __syncthreads();
    tensor_apply<2, fill_mode::overwrite>(shape_gradients_transposed, in_mat, gradients[1], buffer_mat);
  }
  if constexpr (dim > 2) {
    __syncthreads();
    tensor_apply<3, fill_mode::overwrite>(shape_gradients_transposed, in_mat, gradients[2], buffer_mat);
  }
  __syncthreads();

  const Number JxW = 1. / determinant<dim>(jac);
  for (unsigned int q = threadIdx.x; q < dofs_per_component; q += blockDim.x) {
    Number u = in[q];
    Number speed_gradu{};
    for (unsigned int d = 0; d < dim; ++d) { speed_gradu += speed_cells(q, d) * gradients[d].values[q] * jac(d, d); }
    const Number result = (-1.0 + factor_skew) * time_factor * u * (JxW * cell_weights[q]);
    for (unsigned int d = 0; d < dim; ++d) { gradients[d].values[q] = result * speed_cells(q, d) * jac(d, d); }
    // mass matrix part
    Number flux = factor_skew * time_factor * speed_gradu + u * inv_dt;
    out[q] += flux * (JxW * cell_weights[q]);
  }
  __syncthreads();

  tensor_apply<1, fill_mode::add>(shape_gradients, to_const(gradients[0]), out_mat, buffer_mat);
  if constexpr (dim > 1) {
    __syncthreads();
    tensor_apply<2, fill_mode::add>(shape_gradients, to_const(gradients[1]), out_mat, buffer_mat);
  }
  if constexpr (dim > 2) {
    __syncthreads();
    tensor_apply<3, fill_mode::add>(shape_gradients, to_const(gradients[2]), out_mat, buffer_mat);
  }
}
} // namespace tensor
} // namespace gpu

template<int dim, int fe_degree, typename Number, typename VectorizedNumber>
__device__ void simple_apply(GkoCellwiseOperatorItem<dim, fe_degree, Number, VectorizedNumber> a,
                             gko::batch::multi_vector::batch_item<const Number> b,
                             gko::batch::multi_vector::batch_item<Number> x,
                             gko::cuda_hip_kernel) {
  constexpr auto vector_size = static_cast<gko::int32>(sizeof(VectorizedNumber) / sizeof(Number));
  auto b_vectorized = gko::batch::multi_vector::batch_item<const VectorizedNumber>{
    reinterpret_cast<const VectorizedNumber*>(b.values), b.stride / vector_size, b.num_rows, b.num_rhs / vector_size};
  auto x_vectorized = gko::batch::multi_vector::batch_item<VectorizedNumber>{
    reinterpret_cast<VectorizedNumber*>(x.values), x.stride / vector_size, x.num_rows, x.num_rhs / vector_size};

  gpu::face_kernel<fe_degree, dim>(a.jac, a.normal_speed_faces, a.quadrature_data_on_face[0], a.face_weights,
                                   a.time_factor, b_vectorized, x_vectorized);

  gpu::simple::cell_apply<fe_degree, dim>(a.jac, a.shape_gradients_collocation, a.speed_cells, a.cell_weights,
                                          a.time_factor, a.inv_dt, b_vectorized, x_vectorized, a.tmp);
}

} // namespace DGAdvection
