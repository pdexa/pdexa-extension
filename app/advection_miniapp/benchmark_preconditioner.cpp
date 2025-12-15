// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <cstdlib>
#include <iostream>
#include <string>

#include <benchmark/benchmark.h>
#include <cxxopts.hpp>

#include <deal.II/base/utilities.h>

#include "cellwise_preconditioner.apply.hpp"
#include "cellwise_preconditioner.hpp"
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

namespace kernel {

template<int dim, int fe_degree>
__global__ void full_apply_simple(DGAdvection::CellwisePreconditionerView<dim, fe_degree, Number, 1> a_view,
                                  gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                  gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::generate_batch_item(a_view, gko::batch::matrix::csr::batch_item<Number>{}, nullptr, bid,
                                             gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    DGAdvection::simple_apply<dim, fe_degree>(a, b, x, gko::cuda_kernel{});
  }
}
GENERATE_LAUNCH(full_apply_simple, 1);

template<int dim, int fe_degree>
__global__ void full_apply_vectorized_simple(DGAdvection::CellwisePreconditionerView<dim, fe_degree, Number, max_vector_size> a_view,
                                             gko::batch::multi_vector::uniform_batch<const Number> b_view,
                                             gko::batch::multi_vector::uniform_batch<Number> x_view) {
  for (auto bid = blockIdx.x; bid < b_view.num_batch_items; bid += blockDim.x) {
    auto a = gko::batch::generate_batch_item(a_view, gko::batch::matrix::csr::batch_item<Number>{}, nullptr, bid,
                                             gko::cuda_kernel{});
    auto b = gko::batch::extract_batch_item(b_view, bid, gko::cuda_kernel{});
    auto x = gko::batch::extract_batch_item(x_view, bid, gko::cuda_kernel{});

    DGAdvection::simple_apply<dim, fe_degree>(a, b, x, gko::cuda_kernel{});
  }
}
GENERATE_LAUNCH(full_apply_vectorized_simple, max_vector_size);

} // namespace kernel

template<int dim, int fe_degree>
__host__ void gradient_test(uint32_t batches) {
  auto nan = std::numeric_limits<Number>::signaling_NaN();
  constexpr int VectorSize = max_vector_size;

  auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create(), std::make_shared<gko::CudaUnifiedAllocator>(0));

  gko::batch_dim<2> batch_op_size(batches, gko::dim<2>(pow(fe_degree + 1, dim), pow(fe_degree + 1, dim)));
  gko::batch_dim<2> batch_vector_size(batches, gko::dim<2>(pow(fe_degree + 1, dim), 1));
  gko::batch_dim<2> vectorized_batch_vector_size(batches / VectorSize,
                                                 gko::dim<2>(pow(fe_degree + 1, dim), VectorSize));

  auto gko_op = std::make_shared<DGAdvection::CellwisePreconditionerFDM<dim, fe_degree, Number, 1>>(exec);

  auto fill = [&](auto dense, auto val) {
    gko::make_array_view(exec, dense->get_num_stored_elements(), dense->get_values()).fill(val);
  };

  gko_op->eigenvectors = gko::batch::matrix::Dense<std::complex<Number>>::create(
    exec, gko::batch_dim<2>{1, gko::dim<2>(2 * (fe_degree + 1), fe_degree + 1)});
  fill(gko_op->eigenvectors, 1.0);
  gko_op->inverse_eigenvectors = gko::batch::matrix::Dense<std::complex<Number>>::create(
    exec, gko::batch_dim<2>{1, gko::dim<2>(2 * (fe_degree + 1), fe_degree + 1)});
  fill(gko_op->inverse_eigenvectors, 1.0);
  gko_op->determinants = gko::array<Number>{exec, batches};
  gko_op->determinants.fill(1.0);
  gko_op->eigenvalues =
    gko::batch::matrix::Dense<std::complex<Number>>::create(exec, gko::batch_dim<2>{1, gko::dim<2>{2, fe_degree + 1}});
  fill(gko_op->eigenvalues, 1.0);
  gko_op->average_velocity =
    gko::batch::MultiVector<Number>::create(exec, gko::batch_dim<2>{batches, gko::dim<2>{dim, 1}});
  gko_op->average_velocity->fill(1.0);
  gko_op->data_array = {gko::batch::MultiVector<std::complex<Number>>::create(exec, batch_vector_size),
                        gko::batch::MultiVector<std::complex<Number>>::create(exec, batch_vector_size)};

  gko_op->inv_dt = 1.0;
  gko_op->time_factor = 1.0;

  auto src = gko::batch::MultiVector<Number>::create(exec, batch_vector_size);
  auto dst = gko::batch::MultiVector<Number>::create(exec, batch_vector_size);
  src->fill(1.0);
  dst->fill(0.0);

  auto gko_op_view = gko::batch::create_view(gko_op.get());
  auto src_view = gko::batch::to_const(gko::batch::create_view(src.get()));
  auto dst_view = gko::batch::create_view(dst.get());

  auto vectorized_gko_op =
    std::make_shared<DGAdvection::CellwisePreconditionerFDM<dim, fe_degree, Number, VectorSize>>(exec);
  vectorized_gko_op->average_velocity = gko::batch::MultiVector<Number>::create(
    exec, gko::batch_dim<2>{batches / VectorSize, gko::dim<2>{dim, 1 * VectorSize}});
  vectorized_gko_op->average_velocity->fill(1.0);
  vectorized_gko_op->data_array = {
    gko::batch::MultiVector<std::complex<Number>>::create(exec, vectorized_batch_vector_size),
    gko::batch::MultiVector<std::complex<Number>>::create(exec, vectorized_batch_vector_size)};
  vectorized_gko_op->eigenvectors = gko_op->eigenvectors;
  vectorized_gko_op->inverse_eigenvectors = gko_op->inverse_eigenvectors;
  vectorized_gko_op->determinants = gko_op->determinants;
  vectorized_gko_op->eigenvalues = gko_op->eigenvalues;
  vectorized_gko_op->inv_dt = 1.0;
  vectorized_gko_op->time_factor = 1.0;

  auto vectorized_src = gko::batch::MultiVector<Number>::create(exec, vectorized_batch_vector_size);
  auto vectorized_dst = gko::batch::MultiVector<Number>::create(exec, vectorized_batch_vector_size);
  vectorized_src->fill(1.0);
  vectorized_dst->fill(0.0);

  auto vectorized_gko_op_view = gko::batch::create_view(vectorized_gko_op.get());
  auto vectorized_src_view = gko::batch::to_const(gko::batch::create_view(vectorized_src.get()));
  auto vectorized_dst_view = gko::batch::create_view(vectorized_dst.get());

  auto set_counters = [&](benchmark::State& st) {
    st.counters["batches"] = batches;
    st.counters["n"] = fe_degree + 1;
    st.counters["dim"] = dim;
  };
  auto vectorized_set_counters = [&](benchmark::State& st) {
    st.counters["batches"] = batches / max_vector_size;
    st.counters["n"] = fe_degree + 1;
    st.counters["dim"] = dim;
  };

  auto run_gpu_full_simple = [&](benchmark::State& st) {
    set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_full_apply_simple<fe_degree, dim>(gko_op_view, src_view, dst_view);
      st.SetIterationTime(time);
    }
  };
  auto run_gpu_full_vectorized_simple = [&](benchmark::State& st) {
    vectorized_set_counters(st);
    for (auto _: st) {
      auto time = kernel::launch_full_apply_vectorized_simple<fe_degree, dim>(vectorized_gko_op_view,
                                                                              vectorized_src_view, vectorized_dst_view);
      st.SetIterationTime(time);
    }
  };

  std::clog << "Launching GEMM kernel..." << std::endl;
  benchmark::RegisterBenchmark("gpu-full-simple", run_gpu_full_simple)->UseManualTime();
  benchmark::RegisterBenchmark("gpu-full-vectorized-simple", run_gpu_full_vectorized_simple)->UseManualTime();

  benchmark::RunSpecifiedBenchmarks();

  std::clog << "Finished!" << std::endl;
}

int main(int argc, char** argv) {
  cxxopts::Options options("bench-gemm", "Simple example to test tensor product applications.");
  options.allow_unrecognised_options();
  options.add_options()("d,dim", "dimension", cxxopts::value<uint32_t>()->default_value("3"))(
    "p,degree", "Polynomial degree", cxxopts::value<uint32_t>()->default_value("7"))(
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
