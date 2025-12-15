// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <cstdlib>
#include <iostream>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <benchmark/benchmark.h>
#include <cxxopts.hpp>

#include <deal.II/base/utilities.h>

#include "cellwise_operator.apply.hpp"
#include "cellwise_operator.hpp"
#include "definitions.hpp"

using Number = DGAdvection::Number;
using dealii::Utilities::pow;

constexpr int max_vector_size = 4;

using vector_t = VectorizedNumber_t<max_vector_size>;

#ifndef CHECK_DEV_ERROR
#define CHECK_DEV_ERROR(expression)                                                                                    \
  if (auto status = (expression); status != cudaSuccess) {                                                             \
    fprintf(stderr, "cuda error: '%s'(%d) at %s:%d\n", cudaGetErrorString(status), status, __FILE__, __LINE__);        \
    exit(EXIT_FAILURE);                                                                                                \
  }
#endif

#ifndef CHECK_DEVBLAS_ERROR
#define CHECK_DEVBLAS_ERROR(expression)                                                                                \
  if (auto status = (expression); status != CUBLAS_STATUS_SUCCESS) {                                                   \
    fprintf(stderr, "cublas error: '%s'(%d) at %s:%d\n", cublasGetStatusName(status), status, __FILE__, __LINE__);     \
    exit(EXIT_FAILURE);                                                                                                \
  }
#endif

#define GENERATE_LAUNCH(_func, _vs)                                                                                    \
  template<int dim, int fe_degree, typename Number>                                                                    \
  double launch_##_func(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, _vs> a_view,                      \
                        gko::batch::multi_vector::uniform_batch<const Number> b_view,                                  \
                        gko::batch::multi_vector::uniform_batch<Number> x_view) {                                      \
    assert(b_view.num_batch_items == x_view.num_batch_items);                                                          \
    auto num_batches = b_view.num_batch_items;                                                                         \
    auto blockDim = dim3(b_view.num_rows, 1, 1);                                                                       \
    auto gridDim = dim3(std::min(1024ul, num_batches), 1, 1);                                                          \
    cudaEvent_t startEvent, stopEvent;                                                                                 \
    CHECK_DEV_ERROR(cudaEventCreate(&startEvent));                                                                     \
    CHECK_DEV_ERROR(cudaEventCreate(&stopEvent));                                                                      \
    CHECK_DEV_ERROR(cudaEventRecord(startEvent));                                                                      \
    _func<<<gridDim, blockDim>>>(a_view, b_view, x_view);                                                              \
    CHECK_DEV_ERROR(cudaEventRecord(stopEvent));                                                                       \
    auto elapsedTimeMs = 0.0f;                                                                                         \
    CHECK_DEV_ERROR(cudaEventSynchronize(stopEvent));                                                                  \
    CHECK_DEV_ERROR(cudaEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));                                      \
    CHECK_DEV_ERROR(cudaEventDestroy(startEvent));                                                                     \
    CHECK_DEV_ERROR(cudaEventDestroy(stopEvent));                                                                      \
    return static_cast<double>(elapsedTimeMs) / 1000.0;                                                                \
  }                                                                                                                    \
  static_assert(true)

namespace kernel {

template<int dim, int fe_degree>
__global__ void full_apply_simple(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                                  gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                  gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    DGAdvection::gpu::face_kernel<fe_degree, dim>(a.jac, a.normal_speed_faces, a.quadrature_data_on_face[0],
                                                  a.face_weights, a.time_factor, b, x);

    DGAdvection::gpu::simple::cell_apply<fe_degree, dim>(a.jac, a.shape_gradients_collocation, a.speed_cells,
                                                         a.cell_weights, a.time_factor, a.inv_dt, b, x);
  }
}
GENERATE_LAUNCH(full_apply_simple, 1);

template<int dim, int fe_degree>
__global__ void
full_apply_force_global_storage_simple(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                                       gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                       gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    DGAdvection::gpu::face_kernel<fe_degree, dim>(a.jac, a.normal_speed_faces, a.quadrature_data_on_face[0],
                                                  a.face_weights, a.time_factor, b, x);

    DGAdvection::gpu::simple::cell_apply<fe_degree, dim, true>(a.jac, a.shape_gradients_collocation, a.speed_cells,
                                                               a.cell_weights, a.time_factor, a.inv_dt, b, x,
                                                               a.tmp + bid * 3 * a.num_rows);
  }
}
GENERATE_LAUNCH(full_apply_force_global_storage_simple, 1);

template<int dim, int fe_degree>
__global__ void full_apply_simple_reduced_shared(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                                                 gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                                 gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    DGAdvection::gpu::face_kernel<fe_degree, dim>(a.jac, a.normal_speed_faces, a.quadrature_data_on_face[0],
                                                  a.face_weights, a.time_factor, b, x);

    DGAdvection::gpu::simple::cell_apply_single_gradient<fe_degree, dim>(
      a.jac, a.shape_gradients_collocation, a.speed_cells, a.cell_weights, a.time_factor, a.inv_dt, b, x);
  }
}
GENERATE_LAUNCH(full_apply_simple_reduced_shared, 1);

template<int dim, int fe_degree>
__global__ void full_apply_vectorized_simple(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, max_vector_size> a_view,
                                             gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                             gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    auto vectorized_jac = gko::batch::matrix::dense::batch_item(reinterpret_cast<const vector_t*>(a.jac.values),
                                                                a.jac.stride / max_vector_size, a.jac.num_rows, a.jac.num_cols / max_vector_size);
    auto vectorized_normal_speed_faces = gko::batch::matrix::dense::batch_item(
      reinterpret_cast<const vector_t*>(a.normal_speed_faces.values), a.normal_speed_faces.stride / max_vector_size,
      a.normal_speed_faces.num_rows, a.normal_speed_faces.num_cols / max_vector_size);
    auto vectorized_speed_cells = gko::batch::matrix::dense::batch_item(
      reinterpret_cast<const vector_t*>(a.speed_cells.values), a.speed_cells.stride / max_vector_size, a.speed_cells.num_rows,
      a.speed_cells.num_cols / max_vector_size);
    auto vectorized_buffer = reinterpret_cast<vector_t*>(a_view.tmp) + bid * 3 * a.num_rows;

    auto vectorized_b =
      gko::batch::multi_vector::batch_item(reinterpret_cast<const vector_t*>(b.values), 1, b.num_rows, 1);
    auto vectorized_x = gko::batch::multi_vector::batch_item(reinterpret_cast<vector_t*>(x.values), 1, b.num_rows, 1);

    DGAdvection::gpu::face_kernel<fe_degree, dim>(vectorized_jac, vectorized_normal_speed_faces,
                                                  a.quadrature_data_on_face[0], a.face_weights, a.time_factor,
                                                  vectorized_b, vectorized_x);

    DGAdvection::gpu::simple::cell_apply<fe_degree, dim>(vectorized_jac, a.shape_gradients_collocation,
                                                         vectorized_speed_cells, a.cell_weights, a.time_factor,
                                                         a.inv_dt, vectorized_b, vectorized_x, vectorized_buffer);
  }
}
GENERATE_LAUNCH(full_apply_vectorized_simple, max_vector_size);


template<int dim, int fe_degree>
__global__ void
full_apply_vectorized_simple_reduced_shared(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, max_vector_size> a_view,
                                            gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                            gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    auto vectorized_jac = gko::batch::matrix::dense::batch_item(reinterpret_cast<const vector_t*>(a.jac.values),
                                                                a.jac.stride / max_vector_size, a.jac.num_rows, a.jac.num_cols / max_vector_size);
    auto vectorized_normal_speed_faces = gko::batch::matrix::dense::batch_item(
      reinterpret_cast<const vector_t*>(a.normal_speed_faces.values), a.normal_speed_faces.stride / max_vector_size,
      a.normal_speed_faces.num_rows, a.normal_speed_faces.num_cols / max_vector_size);
    auto vectorized_speed_cells = gko::batch::matrix::dense::batch_item(
      reinterpret_cast<const vector_t*>(a.speed_cells.values), a.speed_cells.stride / max_vector_size, a.speed_cells.num_rows,
      a.speed_cells.num_cols / max_vector_size);

    auto vectorized_b =
      gko::batch::multi_vector::batch_item(reinterpret_cast<const vector_t*>(b.values), 1, b.num_rows, 1);
    auto vectorized_x = gko::batch::multi_vector::batch_item(reinterpret_cast<vector_t*>(x.values), 1, b.num_rows, 1);

    DGAdvection::gpu::face_kernel<fe_degree, dim>(vectorized_jac, vectorized_normal_speed_faces,
                                                  a.quadrature_data_on_face[0], a.face_weights, a.time_factor,
                                                  vectorized_b, vectorized_x);

    DGAdvection::gpu::simple::cell_apply_single_gradient<fe_degree, dim>(
      vectorized_jac, a.shape_gradients_collocation, vectorized_speed_cells, a.cell_weights, a.time_factor, a.inv_dt,
      vectorized_b, vectorized_x);
  }
}
GENERATE_LAUNCH(full_apply_vectorized_simple_reduced_shared, max_vector_size);

template<int dim, int fe_degree>
__global__ void full_apply_tensor(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                                  gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                  gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    DGAdvection::gpu::face_kernel<fe_degree, dim>(a.jac, a.normal_speed_faces, a.quadrature_data_on_face[0],
                                                  a.face_weights, a.time_factor, b, x);

    DGAdvection::gpu::tensor::cell_apply<fe_degree, dim>(a.jac, a.shape_gradients_collocation, a.speed_cells,
                                                         a.cell_weights, a.time_factor, a.inv_dt, b, x);
  }
}
GENERATE_LAUNCH(full_apply_tensor, 1);

template<int dim, int fe_degree>
__global__ void face(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                     gko::batch::multi_vector::uniform_batch<const Number> b_view,
                     gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    DGAdvection::gpu::face_kernel<fe_degree, dim>(a.jac, a.normal_speed_faces, a.quadrature_data_on_face[0],
                                                  a.face_weights, a.time_factor, b, x);
  }
}
GENERATE_LAUNCH(face, 1);

template<int dim, int fe_degree>
__global__ void vectorized_face(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, max_vector_size> a_view,
                                gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    auto vectorized_jac = gko::batch::matrix::dense::batch_item(reinterpret_cast<const vector_t*>(a.jac.values),
                                                                a.jac.stride / max_vector_size, a.jac.num_rows, a.jac.num_cols / max_vector_size);
    auto vectorized_normal_speed_faces = gko::batch::matrix::dense::batch_item(
      reinterpret_cast<const vector_t*>(a.normal_speed_faces.values), a.normal_speed_faces.stride / max_vector_size,
      a.normal_speed_faces.num_rows, a.normal_speed_faces.num_cols / max_vector_size);

    auto vectorized_b =
      gko::batch::multi_vector::batch_item(reinterpret_cast<const vector_t*>(b.values), 1, b.num_rows, 1);
    auto vectorized_x = gko::batch::multi_vector::batch_item(reinterpret_cast<vector_t*>(x.values), 1, b.num_rows, 1);

    DGAdvection::gpu::face_kernel<fe_degree, dim>(vectorized_jac, vectorized_normal_speed_faces,
                                                  a.quadrature_data_on_face[0], a.face_weights, a.time_factor,
                                                  vectorized_b, vectorized_x);
  }
}
GENERATE_LAUNCH(vectorized_face, max_vector_size);

template<int dim, int fe_degree>
__global__ void face_shmem(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                           gko::batch::multi_vector::uniform_batch<const Number> b_view,
                           gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    __shared__ Number x_sh_storage[pow(fe_degree + 1, dim)];
    __shared__ Number b_sh_storage[pow(fe_degree + 1, dim)];
    gko::batch::multi_vector::batch_item<Number> x_sh(x_sh_storage, 1, pow(fe_degree + 1, dim), 1);
    gko::batch::multi_vector::batch_item<const Number> b_sh(b_sh_storage, 1, pow(fe_degree + 1, dim), 1);

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { b_sh_storage[tid] = b[tid]; }

    DGAdvection::gpu::face_kernel<fe_degree, dim>(a.jac, a.normal_speed_faces, a.quadrature_data_on_face[0],
                                                  a.face_weights, a.time_factor, b_sh, x_sh);

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { x[tid] = x_sh_storage[tid]; }
  }
}
GENERATE_LAUNCH(face_shmem, 1);

template<int dim, int fe_degree>
__global__ void simple(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                       gko::batch::multi_vector::uniform_batch<const Number> b_view,
                       gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    DGAdvection::gpu::simple::cell_apply<fe_degree, dim>(a.jac, a.shape_gradients_collocation, a.speed_cells,
                                                         a.cell_weights, a.time_factor, a.inv_dt, b, x);
  }
}
GENERATE_LAUNCH(simple, 1);

template<int dim, int fe_degree>
__global__ void simple_reduced_shared(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                                      gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                      gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    DGAdvection::gpu::simple::cell_apply_single_gradient<fe_degree, dim>(
      a.jac, a.shape_gradients_collocation, a.speed_cells, a.cell_weights, a.time_factor, a.inv_dt, b, x);
  }
}
GENERATE_LAUNCH(simple_reduced_shared, 1);

template<int dim, int fe_degree>
__global__ void simple_reduced_shared_vectorized(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, max_vector_size> a_view,
                                                 gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                                 gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    auto vectorized_jac = gko::batch::matrix::dense::batch_item(reinterpret_cast<const vector_t*>(a.jac.values),
                                                                a.jac.stride / max_vector_size, a.jac.num_rows, a.jac.num_cols / max_vector_size);
    auto vectorized_normal_speed_faces = gko::batch::matrix::dense::batch_item(
      reinterpret_cast<const vector_t*>(a.normal_speed_faces.values), a.normal_speed_faces.stride / max_vector_size,
      a.normal_speed_faces.num_rows, a.normal_speed_faces.num_cols / max_vector_size);
    auto vectorized_speed_cells = gko::batch::matrix::dense::batch_item(
      reinterpret_cast<const vector_t*>(a.speed_cells.values), a.speed_cells.stride / max_vector_size, a.speed_cells.num_rows,
      a.speed_cells.num_cols / max_vector_size);

    auto vectorized_b =
      gko::batch::multi_vector::batch_item(reinterpret_cast<const vector_t*>(b.values), 1, b.num_rows, 1);
    auto vectorized_x = gko::batch::multi_vector::batch_item(reinterpret_cast<vector_t*>(x.values), 1, b.num_rows, 1);

    DGAdvection::gpu::simple::cell_apply_single_gradient<fe_degree, dim>(
      vectorized_jac, a.shape_gradients_collocation, vectorized_speed_cells, a.cell_weights, a.time_factor, a.inv_dt,
      vectorized_b, vectorized_x);
  }
}
GENERATE_LAUNCH(simple_reduced_shared_vectorized, max_vector_size);

template<int dim, int fe_degree>
__global__ void tensor(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                       gko::batch::multi_vector::uniform_batch<const Number> b_view,
                       gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    DGAdvection::gpu::tensor::cell_apply<fe_degree, dim>(a.jac, a.shape_gradients_collocation, a.speed_cells,
                                                         a.cell_weights, a.time_factor, a.inv_dt, b, x);
  }
}
GENERATE_LAUNCH(tensor, 1);

template<int dim, int fe_degree>
__global__ void transpose(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                          gko::batch::multi_vector::uniform_batch<const Number> b_view,
                          gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    __shared__ Number x_sh_storage[pow(fe_degree + 1, dim)];
    __shared__ Number b_sh_storage[pow(fe_degree + 1, dim)];
    col_major_matrix<Number> x_sh(x_sh_storage, pow(fe_degree + 1, dim - 1), pow(fe_degree + 1, dim - 1),
                                  fe_degree + 1);
    col_major_matrix<const Number> b_sh(b_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { b_sh_storage[tid] = b[tid]; }
    __syncthreads();

    ::transpose(b_sh, x_sh);

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { x[tid] = x_sh_storage[tid]; }
  }
}
GENERATE_LAUNCH(transpose, 1);

template<int dim, int fe_degree>
__global__ void gemm(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                     gko::batch::multi_vector::uniform_batch<const Number> b_view,
                     gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    __shared__ Number x_sh_storage[pow(fe_degree + 1, dim)];
    __shared__ Number b_sh_storage[pow(fe_degree + 1, dim)];
    col_major_matrix<Number> x_sh(x_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));
    col_major_matrix<const Number> b_sh(b_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { b_sh_storage[tid] = b[tid]; }

    ::gemm(a.shape_gradients_collocation, b_sh, x_sh);

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { x[tid] = x_sh_storage[tid]; }
  }
}
GENERATE_LAUNCH(gemm, 1);

template<int dim, int fe_degree>
__global__ void simple_d0(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                          gko::batch::multi_vector::uniform_batch<const Number> b_view,
                          gko::batch::multi_vector::uniform_batch<Number> x_view) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    __shared__ Number x_sh_storage[pow(fe_degree + 1, dim)];
    __shared__ Number b_sh_storage[pow(fe_degree + 1, dim)];
    col_major_matrix<Number> x_sh(x_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));
    col_major_matrix<const Number> b_sh(b_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));

    auto out_tensor = tensor3d{x_sh.values, tensor_size};
    auto in_tensor = tensor3d{b_sh.values, tensor_size};

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { b_sh_storage[tid] = b[tid]; }

    simple_tensor_apply<fe_degree, dim, 0, true, false>(a.shape_gradients_collocation, in_tensor, out_tensor);

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { x[tid] = x_sh_storage[tid]; }
  }
}
GENERATE_LAUNCH(simple_d0, 1);

template<int dim, int fe_degree>
__global__ void simple_d1(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                          gko::batch::multi_vector::uniform_batch<const Number> b_view,
                          gko::batch::multi_vector::uniform_batch<Number> x_view) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    __shared__ Number x_sh_storage[pow(fe_degree + 1, dim)];
    __shared__ Number b_sh_storage[pow(fe_degree + 1, dim)];
    col_major_matrix<Number> x_sh(x_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));
    col_major_matrix<const Number> b_sh(b_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));

    auto out_tensor = tensor3d{x_sh.values, tensor_size};
    auto in_tensor = tensor3d{b_sh.values, tensor_size};

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { b_sh_storage[tid] = b[tid]; }

    simple_tensor_apply<fe_degree, dim, 1, true, false>(a.shape_gradients_collocation, in_tensor, out_tensor);

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { x[tid] = x_sh_storage[tid]; }
  }
}
GENERATE_LAUNCH(simple_d1, 1);

template<int dim, int fe_degree>
__global__ void simple_d2(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                          gko::batch::multi_vector::uniform_batch<const Number> b_view,
                          gko::batch::multi_vector::uniform_batch<Number> x_view) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    __shared__ Number x_sh_storage[pow(fe_degree + 1, dim)];
    __shared__ Number b_sh_storage[pow(fe_degree + 1, dim)];
    col_major_matrix<Number> x_sh(x_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));
    col_major_matrix<const Number> b_sh(b_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));

    auto out_tensor = tensor3d{x_sh.values, tensor_size};
    auto in_tensor = tensor3d{b_sh.values, tensor_size};

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { b_sh_storage[tid] = b[tid]; }

    simple_tensor_apply<fe_degree, dim, 2, true, false>(a.shape_gradients_collocation, in_tensor, out_tensor);

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { x[tid] = x_sh_storage[tid]; }
  }
}
GENERATE_LAUNCH(simple_d2, 1);

template<int dim, int fe_degree>
__global__ void vectorized_simple_d0(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, max_vector_size> a_view,
                                     gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                     gko::batch::multi_vector::uniform_batch<Number> x_view) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});
    auto vectorized_x =
      gko::batch::multi_vector::batch_item<vector_t>(reinterpret_cast<vector_t*>(x.values), 1, x.num_rows, 1);
    auto vectorized_b =
      gko::batch::multi_vector::batch_item<const vector_t>(reinterpret_cast<const vector_t*>(b.values), 1, b.num_rows, 1);

    __shared__ vector_t x_sh_storage[pow(fe_degree + 1, dim)];
    __shared__ vector_t b_sh_storage[pow(fe_degree + 1, dim)];
    col_major_matrix<vector_t> x_sh(x_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));
    col_major_matrix<const vector_t> b_sh(b_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));

    auto out_tensor = tensor3d{x_sh.values, tensor_size};
    auto in_tensor = tensor3d{b_sh.values, tensor_size};

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) {
      b_sh_storage[tid] = vectorized_b[tid];
    }

    simple_tensor_apply<fe_degree, dim, 0, true, false>(a.shape_gradients_collocation, in_tensor, out_tensor);

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) {
      vectorized_x[tid] = x_sh_storage[tid];
    }
  }
}
GENERATE_LAUNCH(vectorized_simple_d0, max_vector_size);

template<int dim, int fe_degree>
__global__ void vectorized_simple_d1(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, max_vector_size> a_view,
                                     gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                     gko::batch::multi_vector::uniform_batch<Number> x_view) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});
    auto vectorized_x =
      gko::batch::multi_vector::batch_item<vector_t>(reinterpret_cast<vector_t*>(x.values), 1, x.num_rows, 1);
    auto vectorized_b =
      gko::batch::multi_vector::batch_item<const vector_t>(reinterpret_cast<const vector_t*>(b.values), 1, b.num_rows, 1);

    __shared__ vector_t x_sh_storage[pow(fe_degree + 1, dim)];
    __shared__ vector_t b_sh_storage[pow(fe_degree + 1, dim)];
    col_major_matrix<vector_t> x_sh(x_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));
    col_major_matrix<const vector_t> b_sh(b_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));

    auto out_tensor = tensor3d{x_sh.values, tensor_size};
    auto in_tensor = tensor3d{b_sh.values, tensor_size};

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) {
      b_sh_storage[tid] = vectorized_b[tid];
    }

    simple_tensor_apply<fe_degree, dim, 1, true, false>(a.shape_gradients_collocation, in_tensor, out_tensor);

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) {
      vectorized_x[tid] = x_sh_storage[tid];
    }
  }
}
GENERATE_LAUNCH(vectorized_simple_d1, max_vector_size);

template<int dim, int fe_degree>
__global__ void vectorized_simple_d2(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, max_vector_size> a_view,
                                     gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                     gko::batch::multi_vector::uniform_batch<Number> x_view) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});
    auto vectorized_x =
      gko::batch::multi_vector::batch_item<vector_t>(reinterpret_cast<vector_t*>(x.values), 1, x.num_rows, 1);
    auto vectorized_b =
      gko::batch::multi_vector::batch_item<const vector_t>(reinterpret_cast<const vector_t*>(b.values), 1, b.num_rows, 1);

    __shared__ vector_t x_sh_storage[pow(fe_degree + 1, dim)];
    __shared__ vector_t b_sh_storage[pow(fe_degree + 1, dim)];
    col_major_matrix<vector_t> x_sh(x_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));
    col_major_matrix<const vector_t> b_sh(b_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));

    auto out_tensor = tensor3d{x_sh.values, tensor_size};
    auto in_tensor = tensor3d{b_sh.values, tensor_size};

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) {
      b_sh_storage[tid] = vectorized_b[tid];
    }

    simple_tensor_apply<fe_degree, dim, 2, true, false>(a.shape_gradients_collocation, in_tensor, out_tensor);

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) {
      vectorized_x[tid] = x_sh_storage[tid];
    }
  }
}
GENERATE_LAUNCH(vectorized_simple_d2, max_vector_size);

template<int dim, int fe_degree>
__global__ void simple_reduction(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                                 gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                 gko::batch::multi_vector::uniform_batch<Number> x_view) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    __shared__ Number x_sh_storage[pow(fe_degree + 1, dim)];
    __shared__ Number b_sh_storage[pow(fe_degree + 1, dim)];
    col_major_matrix<Number> x_sh(x_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));
    col_major_matrix<const Number> b_sh(b_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));

    auto out_tensor = tensor3d{x_sh.values, tensor_size};
    auto in_tensor = tensor3d{b_sh.values, tensor_size};

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { b_sh_storage[tid] = b[tid]; }

    simple_reduce_tensor_apply<fe_degree, dim, 0, true, false>(a.shape_gradients_collocation, in_tensor, out_tensor);

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { x[tid] = x_sh_storage[tid]; }
  }
}
GENERATE_LAUNCH(simple_reduction, 1);

template<int dim, int fe_degree>
__global__ void copy(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                     gko::batch::multi_vector::uniform_batch<const Number> b_view,
                     gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    __shared__ Number x_sh_storage[pow(fe_degree + 1, dim)];
    __shared__ Number b_sh_storage[pow(fe_degree + 1, dim)];
    col_major_matrix<Number> x_sh(x_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));
    col_major_matrix<const Number> b_sh(b_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { b_sh_storage[tid] = b[tid]; }
    __syncthreads();

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) {
      x_sh_storage[tid] = b_sh_storage[tid];
    }
    __syncthreads();

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { x[tid] = x_sh_storage[tid]; }
  }
}
GENERATE_LAUNCH(copy, 1);

template<int dim, int fe_degree>
__global__ void shmem_overhead(DGAdvection::GkoCellwiseOperatorView<dim, fe_degree, Number, 1> a_view,
                               gko::batch::multi_vector::uniform_batch<const Number> b_view,
                               gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::extract_batch_item(a_view, bid, gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    __shared__ Number b_sh_storage[pow(fe_degree + 1, dim)];
    col_major_matrix<const Number> b_sh(b_sh_storage, fe_degree + 1, fe_degree + 1, pow(fe_degree + 1, dim - 1));

    for (auto tid = threadIdx.x; tid < pow(fe_degree + 1, dim); tid += blockDim.x) { b_sh_storage[tid] = b[tid]; }

    __syncthreads();
  }
}
GENERATE_LAUNCH(shmem_overhead, 1);

} // namespace kernel

constexpr int VectorSize = max_vector_size;

template<int dim, int fe_degree>
__host__ void gradient_test(uint32_t batches) {
  auto nan = std::numeric_limits<Number>::signaling_NaN();

  auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create(), std::make_shared<gko::CudaUnifiedAllocator>(0));

  gko::batch_dim<2> batch_op_size(batches, gko::dim<2>(pow(fe_degree + 1, dim), pow(fe_degree + 1, dim)));
  gko::batch_dim<2> batch_vector_size(batches, gko::dim<2>(pow(fe_degree + 1, dim), 1));
  gko::batch_dim<2> vectorized_batch_vector_size(batches / VectorSize,
                                                 gko::dim<2>(pow(fe_degree + 1, dim), VectorSize));

  auto gko_op = std::make_shared<DGAdvection::GkoCellwiseOperator<dim, fe_degree, Number, 1>>(exec);

  auto fill = [&](auto dense, auto val) {
    gko::make_array_view(exec, dense->get_num_stored_elements(), dense->get_values()).fill(val);
  };

  gko_op->tmp_storage_ = gko::array<Number>{exec, pow(fe_degree + 1, dim) * 3 * VectorSize * batches};
  gko_op->jac = gko::batch::matrix::Dense<Number>::create(exec, gko::batch_dim<2>{batches, gko::dim<2>(dim, dim)});
  fill(gko_op->jac, 1.9);
  gko_op->speed_cells = gko::batch::matrix::Dense<Number>::create(
    exec, gko::batch_dim<2>{batches, gko::dim<2>(pow(fe_degree + 1, dim), dim)});
  fill(gko_op->speed_cells, 1.0);
  gko_op->normal_speed_faces = gko::batch::matrix::Dense<Number>::create(
    exec, gko::batch_dim<2>{batches, gko::dim<2>(2 * dim, pow(fe_degree + 1, dim))});
  fill(gko_op->normal_speed_faces, 1.0);

  gko_op->quadrature_data_on_face = {
    gko::array<Number>{exec, pow(fe_degree + 1, dim)},
    gko::array<Number>{exec, pow(fe_degree + 1, dim)},
  };
  gko_op->quadrature_data_on_face[0].fill(1.0);
  gko_op->quadrature_data_on_face[1].fill(1.0);

  gko_op->shape_values = gko::array<Number>{exec, fe_degree + 1};
  gko_op->shape_values.fill(1.0);
  gko_op->shape_gradients_collocation = gko::array<Number>{exec, pow(fe_degree + 1, 2)};
  gko_op->shape_gradients_collocation.fill(1.0);
  gko_op->cell_weights = gko::array<Number>{exec, pow(fe_degree + 1, dim)};
  gko_op->cell_weights.fill(1.0);
  gko_op->face_weights = gko::array<Number>{exec, pow(fe_degree + 1, dim - 1)};
  gko_op->face_weights.fill(1.0);

  gko_op->inv_dt = 1.0;
  gko_op->time_factor = 1.0;

  auto vectorized_gko_op = std::make_shared<DGAdvection::GkoCellwiseOperator<dim, fe_degree, Number, VectorSize>>(exec);
  vectorized_gko_op->tmp_storage_ = gko::array<Number>{exec, pow(fe_degree + 1, dim) * 3 * VectorSize * batches};
  vectorized_gko_op->jac = gko::batch::matrix::Dense<Number>::create(
    exec, gko::batch_dim<2>{batches / VectorSize, gko::dim<2>(dim, dim * VectorSize)});
  fill(vectorized_gko_op->jac, 1.9);
  vectorized_gko_op->speed_cells = gko::batch::matrix::Dense<Number>::create(
    exec, gko::batch_dim<2>{batches / VectorSize, gko::dim<2>(pow(fe_degree + 1, dim), dim * VectorSize)});
  fill(vectorized_gko_op->speed_cells, 1.0);
  vectorized_gko_op->normal_speed_faces = gko::batch::matrix::Dense<Number>::create(
    exec, gko::batch_dim<2>{batches / VectorSize, gko::dim<2>(2 * dim, pow(fe_degree + 1, dim) * VectorSize)});
  fill(vectorized_gko_op->normal_speed_faces, 1.0);
  vectorized_gko_op->quadrature_data_on_face = {gko_op->quadrature_data_on_face[0], gko_op->quadrature_data_on_face[1]};
  vectorized_gko_op->shape_values = gko_op->shape_values;
  vectorized_gko_op->shape_gradients_collocation = gko_op->shape_gradients_collocation;
  vectorized_gko_op->cell_weights = gko_op->cell_weights;
  vectorized_gko_op->face_weights = gko_op->face_weights;

  auto src = gko::batch::MultiVector<Number>::create(exec, batch_vector_size);
  auto dst = gko::batch::MultiVector<Number>::create(exec, batch_vector_size);
  src->fill(1.0);
  dst->fill(0.0);

  auto gko_op_view = gko::batch::create_view(gko_op.get());
  auto src_view = gko::batch::to_const(gko::batch::create_view(src.get()));
  auto dst_view = gko::batch::create_view(dst.get());

  auto vectorized_gko_op_view = gko::batch::create_view(vectorized_gko_op.get());
  auto vectorized_src = gko::batch::MultiVector<Number>::create(exec, vectorized_batch_vector_size);
  auto vectorized_dst = gko::batch::MultiVector<Number>::create(exec, vectorized_batch_vector_size);
  vectorized_src->fill(1.0);
  vectorized_dst->fill(0.0);

  auto vectorized_src_view = gko::batch::to_const(gko::batch::create_view(vectorized_src.get()));
  auto vectorized_dst_view = gko::batch::create_view(vectorized_dst.get());

  auto set_counters = [&](benchmark::State& st) {
    st.counters["batches"] = batch_vector_size.get_num_batch_items();
    st.counters["n"] = fe_degree + 1;
    st.counters["dim"] = dim;
  };

  auto vectorized_set_counters = [&](benchmark::State& st) {
    st.counters["batches"] = vectorized_batch_vector_size.get_num_batch_items();
    st.counters["n"] = fe_degree + 1;
    st.counters["dim"] = dim;
  };

  auto run_gpu_full_simple = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_full_apply_simple<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_full_global_storage_simple = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time =
        kernel::launch_full_apply_force_global_storage_simple<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_full_simple_reduced_shared = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_full_apply_simple_reduced_shared<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_full_vectorized_simple = [&](benchmark::State& st) {
    vectorized_set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_full_apply_vectorized_simple<dim, fe_degree>(vectorized_gko_op_view,
                                                                              vectorized_src_view, vectorized_dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_full_vectorized_simple_reduced_shared = [&](benchmark::State& st) {
    vectorized_set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_full_apply_vectorized_simple_reduced_shared<dim, fe_degree>(
        vectorized_gko_op_view, vectorized_src_view, vectorized_dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_full_tensor = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_full_apply_tensor<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_face = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_face<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_vectorized_face = [&](benchmark::State& st) {
    vectorized_set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_vectorized_face<dim, fe_degree>(vectorized_gko_op_view, vectorized_src_view,
                                                                 vectorized_dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_face_shmem = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_face_shmem<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_simple = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_simple<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_simple_reduced_shared = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_simple_reduced_shared<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_simple_reduced_shared_vectorized = [&](benchmark::State& st) {
    vectorized_set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_simple_reduced_shared_vectorized<dim, fe_degree>(
        vectorized_gko_op_view, vectorized_src_view, vectorized_dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_tensor = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_tensor<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_transpose = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_transpose<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_gemm = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_gemm<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_simple_d0 = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_simple_d0<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_simple_d1 = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_simple_d1<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_simple_d2 = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_simple_d2<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_simple_reduction_d0 = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_simple_reduction<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_vectorized_simple_d0 = [&](benchmark::State& st) {
    vectorized_set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_vectorized_simple_d0<dim, fe_degree>(vectorized_gko_op_view, vectorized_src_view,
                                                                      vectorized_dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_vectorized_simple_d1 = [&](benchmark::State& st) {
    vectorized_set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_vectorized_simple_d1<dim, fe_degree>(vectorized_gko_op_view, vectorized_src_view,
                                                                      vectorized_dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_vectorized_simple_d2 = [&](benchmark::State& st) {
    vectorized_set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_vectorized_simple_d2<dim, fe_degree>(vectorized_gko_op_view, vectorized_src_view,
                                                                      vectorized_dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_copy = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_copy<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_overhead = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_shmem_overhead<dim, fe_degree>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };

  std::clog << "Launching GEMM kernel..." << std::endl;
  benchmark::RegisterBenchmark("gpu-full-simple", run_gpu_full_simple)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-full-global-storage-simple", run_gpu_full_global_storage_simple)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-full-simple-reduced-shared", run_gpu_full_simple_reduced_shared)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-full-vectorized-simple", run_gpu_full_vectorized_simple)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-full-vectorized-simple-reduced-shared", run_gpu_full_vectorized_simple_reduced_shared)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-full-tensor", run_gpu_full_tensor)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-face", run_gpu_face)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-vectorized-face", run_gpu_vectorized_face)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-face-shmem", run_gpu_face_shmem)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-simple", run_gpu_simple)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-simple-reduced-shared", run_gpu_simple_reduced_shared)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-simple-reduced-shared-vectorized", run_gpu_simple_reduced_shared_vectorized)
    ->UseManualTime();
  benchmark::RegisterBenchmark("gpu-tensor", run_gpu_tensor)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-transpose", run_gpu_transpose)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-gemm", run_gpu_gemm)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-simple-d0", run_gpu_simple_d0)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-simple-d1", run_gpu_simple_d1)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-simple-d2", run_gpu_simple_d2)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-simple-reduction-d0", run_gpu_simple_reduction_d0)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-vectorized-simple-d0", run_gpu_vectorized_simple_d0)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-vectorized-simple-d1", run_gpu_vectorized_simple_d1)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-vectorized-simple-d2", run_gpu_vectorized_simple_d2)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-copy", run_gpu_copy)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-shmem-overhead", run_gpu_overhead)->UseManualTime();

  benchmark::RunSpecifiedBenchmarks();

  std::clog << "Finished!" << std::endl;
}

int main(int argc, char** argv) {
  cxxopts::Options options("bench-gemm", "Simple example to test tensor product applications.");
  options.allow_unrecognised_options();
  options.add_options()("d,dim", "dimension", cxxopts::value<uint32_t>()->default_value("2"))(
    "p,degree", "Polynomial degree", cxxopts::value<uint32_t>()->default_value("5"))(
    "b,batches", "number of batches", cxxopts::value<uint32_t>()->default_value("1"));
  auto args = options.parse(argc, argv);

  benchmark::Initialize(&argc, argv);

  auto dim = args["dim"].as<uint32_t>();
  auto degree = args["degree"].as<uint32_t>();
  auto batches = args["batches"].as<uint32_t>();

  if (degree != DGAdvection::fe_degree) {
    std::cerr << "unsupported degree " << degree << std::endl;
    std::exit(1);
  }
  if (dim == 2) { gradient_test<2, DGAdvection::fe_degree>(batches); }
  else if (dim == 3) { gradient_test<3, DGAdvection::fe_degree>(batches); }
  else { std::cerr << "unsupported dimension " << dim << std::endl; }

  benchmark::Shutdown();

  return 0;
}
