// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <cstdlib>
#include <iostream>
#include <string>

#include <benchmark/benchmark.h>
#include <cxxopts.hpp>

#include <deal.II/base/utilities.h>

#include "pdexa-ext/ginkgo/backend/cuda_hip/batch_bicgstab_kernels.hpp"
#include "pdexa-ext/ginkgo/core/base/view.hpp"

#include "cellwise_operator.apply.hpp"
#include "cellwise_operator.hpp"
#include "definitions.hpp"

using Number = double;
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

#define GENERATE_LAUNCH(_func, _vs)                                                                                    \
  template<int fe_degree, int dim, typename Number>                                                                    \
  double launch_##_func(DGAdvection::CellwisePreconditionerView<dim, fe_degree, Number, _vs> a_view,                   \
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

struct matrix_noop {
  int64_t num_rows;
  int64_t num_batch_items;
};

__device__ __host__ matrix_noop extract_batch_item(matrix_noop a, gko::size_type, gko::any_kernel) { return a; }

template<typename Mat>
__device__ __host__ matrix_noop generate_batch_item(matrix_noop a, Mat, char*, gko::size_type, gko::any_kernel) {
  return a;
}

template<typename Number>
__device__ void simple_apply(matrix_noop,
                             gko::batch::multi_vector::batch_item<const Number> r,
                             gko::batch::multi_vector::batch_item<Number> z,
                             gko::cuda_hip_kernel) {
  const int max_li = r.num_rows * r.num_rhs;
  for (auto li = static_cast<int>(threadIdx.x); li < max_li; li += static_cast<int>(blockDim.x)) {
    const int row = li / r.num_rhs;
    const int col = li % r.num_rhs;

    z.at(row, col) = r.at(row, col);
  }
}

struct logger_noop {
  constexpr void log_iteration(const int64_t, const int, const double) {}

  template<int n_rhs>
  constexpr void log_iteration(const int64_t, const int, const double*) {}
};

struct stop_noop {
  using real_type = Number;
  constexpr stop_noop(const real_type, const real_type* const) {}

  constexpr bool check_converged(const real_type* const) const { return false; }

  template<int n_rhs>
  constexpr bool check_converged(const real_type* const) const {
    return false;
  }
};

constexpr int num_iters = 10;

namespace kernel {

template<int num_cols, typename Number>
double launch_bicgstab_func(std::shared_ptr<const gko::Executor> exec,
                            gko::batch::multi_vector::uniform_batch<const Number> b_view,
                            gko::batch::multi_vector::uniform_batch<Number> x_view) {
  assert(b_view.num_batch_items == x_view.num_batch_items);
  assert(b_view.num_rhs == num_cols);
  auto num_batches = b_view.num_batch_items;
  auto num_rows = b_view.num_rows;
  auto blockDim = dim3(std::max(b_view.num_rows, 32), 1, 1);
  auto gridDim = dim3(num_batches, 1, 1);

  using StopType = stop_noop;
  using LogType = logger_noop;
  using PrecType = matrix_noop;
  auto prec = PrecType{};
  auto logger = LogType{};
  auto sconf = gko::kernels::batch_bicgstab::compute_shared_storage<PrecType, Number>(0, num_rows, 0, num_cols);
  auto workspace = gko::array<double>(exec, sconf.gmem_stride_bytes * num_batches / sizeof(Number));

  auto a = matrix_noop{num_rows, num_batches};

  cudaEvent_t startEvent, stopEvent;
  CHECK_DEV_ERROR(cudaEventCreate(&startEvent));
  CHECK_DEV_ERROR(cudaEventCreate(&stopEvent));
  CHECK_DEV_ERROR(cudaEventRecord(startEvent));
  gko::kernels::cuda::batch_template::batch_single_kernels::batch_bicgstab::apply_kernel<StopType, 0, false, PrecType,
                                                                                         LogType, matrix_noop, num_cols>
    <<<gridDim, blockDim>>>(sconf, num_iters, 1.0, logger, prec, a, b_view, x_view, workspace.get_data());
  CHECK_DEV_ERROR(cudaEventRecord(stopEvent));
  auto elapsedTimeMs = 0.0f;
  CHECK_DEV_ERROR(cudaEventSynchronize(stopEvent));
  CHECK_DEV_ERROR(cudaEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
  CHECK_DEV_ERROR(cudaEventDestroy(startEvent));
  CHECK_DEV_ERROR(cudaEventDestroy(stopEvent));
  return static_cast<double>(elapsedTimeMs) / 1000.0;
}

template<int dim, int fe_degree, int num_cols, typename Number>
double
launch_bicgstab_operator_func(std::shared_ptr<DGAdvection::GkoCellwiseOperator<dim, fe_degree, Number, num_cols>> op,
                              gko::batch::multi_vector::uniform_batch<const Number> b_view,
                              gko::batch::multi_vector::uniform_batch<Number> x_view) {
  assert(b_view.num_batch_items == x_view.num_batch_items);
  assert(b_view.num_rhs == num_cols);
  auto num_batches = b_view.num_batch_items;
  auto num_rows = b_view.num_rows;
  auto blockDim = dim3(std::max(b_view.num_rows, 32), 1, 1);
  auto gridDim = dim3(num_batches, 1, 1);

  auto exec = op->get_executor();

  auto view = gko::batch::create_view(op.get());
  // need to manually add sizes, since the operator was created without size information
  view.num_batch_items = num_batches;
  view.num_rows = num_rows;

  using MatrixType = std::decay_t<decltype(view)>;
  using StopType = stop_noop;
  using LogType = logger_noop;
  using PrecType = matrix_noop;
  auto prec = PrecType{};
  auto logger = LogType{};
  auto sconf = gko::kernels::batch_bicgstab::compute_shared_storage<PrecType, Number>(0, num_rows, 0, num_cols);
  auto workspace = gko::array<double>(exec, sconf.gmem_stride_bytes * num_batches / sizeof(Number));

  cudaEvent_t startEvent, stopEvent;
  CHECK_DEV_ERROR(cudaEventCreate(&startEvent));
  CHECK_DEV_ERROR(cudaEventCreate(&stopEvent));
  CHECK_DEV_ERROR(cudaEventRecord(startEvent));
  gko::kernels::cuda::batch_template::batch_single_kernels::batch_bicgstab::apply_kernel<StopType, 0, false, PrecType,
                                                                                         LogType, MatrixType, num_cols>
    <<<gridDim, blockDim>>>(sconf, num_iters, 1.0, logger, prec, view, b_view, x_view, workspace.get_data());
  exec->synchronize();
  CHECK_DEV_ERROR(cudaEventRecord(stopEvent));
  auto elapsedTimeMs = 0.0f;
  CHECK_DEV_ERROR(cudaEventSynchronize(stopEvent));
  CHECK_DEV_ERROR(cudaEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
  CHECK_DEV_ERROR(cudaEventDestroy(startEvent));
  CHECK_DEV_ERROR(cudaEventDestroy(stopEvent));
  return static_cast<double>(elapsedTimeMs) / 1000.0;
}

} // namespace kernel

template<int dim, int fe_degree>
__host__ void gradient_test(uint32_t batches) {
  auto nan = std::numeric_limits<Number>::signaling_NaN();

  auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create(), std::make_shared<gko::CudaUnifiedAllocator>(0));

  constexpr uint32_t num_cols = 1;
  constexpr uint32_t num_rows = pow(fe_degree + 1, dim);
  gko::batch_dim<2> batch_vector_size(batches / num_cols, gko::dim<2>(num_rows, num_cols));

  auto gko_op = std::make_shared<DGAdvection::GkoCellwiseOperator<dim, fe_degree, Number, num_cols>>(exec);

  auto fill = [&](auto dense, auto val) {
    gko::make_array_view(exec, dense->get_num_stored_elements(), dense->get_values()).fill(val);
  };

  gko_op->jac = gko::batch::matrix::Dense<Number>::create(exec, gko::batch_dim<2>{batches, gko::dim<2>(dim, dim)});
  fill(gko_op->jac, 1.9);
  gko_op->speed_cells =
    gko::batch::matrix::Dense<Number>::create(exec, gko::batch_dim<2>{batches, gko::dim<2>(num_rows, dim)});
  fill(gko_op->speed_cells, 1.0);
  gko_op->normal_speed_faces =
    gko::batch::matrix::Dense<Number>::create(exec, gko::batch_dim<2>{batches, gko::dim<2>(2 * dim, num_rows)});
  fill(gko_op->normal_speed_faces, 1.0);

  gko_op->quadrature_data_on_face = {
    gko::array<Number>{exec, num_rows},
    gko::array<Number>{exec, num_rows},
  };
  gko_op->quadrature_data_on_face[0].fill(1.0);
  gko_op->quadrature_data_on_face[1].fill(1.0);

  gko_op->shape_values = gko::array<Number>{exec, fe_degree + 1};
  gko_op->shape_values.fill(1.0);
  gko_op->shape_gradients_collocation = gko::array<Number>{exec, pow(fe_degree + 1, 2)};
  gko_op->shape_gradients_collocation.fill(1.0);
  gko_op->cell_weights = gko::array<Number>{exec, num_rows};
  gko_op->cell_weights.fill(1.0);
  gko_op->face_weights = gko::array<Number>{exec, pow(fe_degree + 1, dim - 1)};
  gko_op->face_weights.fill(1.0);

  gko_op->inv_dt = 1.0;
  gko_op->time_factor = 1.0;
  auto src = gko::batch::MultiVector<Number>::create(exec, batch_vector_size);
  auto dst = gko::batch::MultiVector<Number>::create(exec, batch_vector_size);
  src->fill(1.0);
  dst->fill(0.0);

  auto src_view = gko::batch::to_const(gko::batch::create_view(src.get()));
  auto dst_view = gko::batch::create_view(dst.get());

  auto set_counters = [&](benchmark::State& st) {
    st.counters["batches"] = batch_vector_size.get_num_batch_items();
    st.counters["n"] = fe_degree + 1;
    st.counters["dim"] = dim;
    st.counters["iters"] = num_iters;
  };
  auto run_gpu_bicgstab_id = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_bicgstab_func<num_cols>(exec, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_bicgstab_operator = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_bicgstab_operator_func(gko_op, src_view, dst_view);
      exec->synchronize();
      st.SetIterationTime(time);
    }
  };

  std::clog << "Launching benchmark kernels..." << std::endl;
  benchmark::RegisterBenchmark("gpu-bicgstab-id", run_gpu_bicgstab_id)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-bicgstab-operator", run_gpu_bicgstab_operator)->UseManualTime();

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
