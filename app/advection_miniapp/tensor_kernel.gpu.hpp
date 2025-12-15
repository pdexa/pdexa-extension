// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <cooperative_groups.h>
#include <cuda/std/complex>
#include <deal.II/base/utilities.h>

#include "definitions.hpp"
#include "pdexa-ext/ginkgo/core/matrix/batch_struct.hpp"
#include "pdexa-ext/ginkgo/backend/cuda_hip/components/reduction.hpp"
#include "tensor_helpers.hpp"
#include "wmma.hpp"

template<typename Number>
using row_major_matrix = gko::batch::matrix::dense::batch_item<Number>;

template<typename Number>
using col_major_matrix = gko::batch::matrix::dense::batch_item_colum_major<Number>;

template<typename Number>
using vector_item = gko::batch::multi_vector::batch_item<Number>;


template<int fe_degree, int dim, int dir, bool transpose, bool additive, typename Number1, typename Number2>
__device__ void simple_tensor_apply(gko::batch::matrix::dense::batch_item<const Number1> shape_data,
                                    tensor3d<const Number2> in,
                                    tensor3d<Number2> out) {
  for (uint32_t tid = threadIdx.x; tid < dealii::Utilities::pow(fe_degree + 1, dim); tid += blockDim.x) {
    auto i = tid % (fe_degree + 1);
    auto j = (tid / (fe_degree + 1)) % (fe_degree + 1);
    auto k = tid / ((fe_degree + 1) * (fe_degree + 1));

    auto dir_idx = [](auto ii, auto jj, auto kk) -> uint32_t {
      if constexpr (dir == 0) { return ii; }
      if constexpr (dir == 1) { return jj; }
      if constexpr (dir == 2) { return kk; }
      __builtin_unreachable();
    };

    auto map = [](auto ii, auto jj, auto kk) -> uint32_t {
      if constexpr (dim == 1) { return ii; }
      if constexpr (dim == 2) { return ii + jj * (fe_degree + 1); }
      if constexpr (dim == 3) { return ii + jj * (fe_degree + 1) + kk * (fe_degree + 1) * (fe_degree + 1); }
      __builtin_unreachable();
    };

    auto acc = [&](auto ii, auto jj, auto kk, auto qq) -> uint32_t {
      if constexpr (dir == 0) { return map(qq, jj, kk); }
      if constexpr (dir == 1) { return map(ii, qq, kk); }
      if constexpr (dir == 2) { return map(ii, jj, qq); }
      __builtin_unreachable();
    };

    auto sum = additive ? out(i, j, k) : gko::zero<Number2>();
    for (uint32_t q = 0; q < fe_degree + 1; ++q) {
      sum = sum + static_cast<Number1>(transpose ? shape_data(q, dir_idx(i, j, k)) : shape_data(dir_idx(i, j, k), q)) *
                    in.data[acc(i, j, k, q)];
    }
    out(i, j, k) = sum;
  }
}

template<int fe_degree, int dim, int dir, bool transpose, bool additive, typename Number1, typename Number2>
__device__ void simple_reduce_tensor_apply(gko::batch::matrix::dense::batch_item<const Number1> shape_data,
                                           tensor3d<const Number2> in,
                                           tensor3d<Number2> out) {
  namespace cg = cooperative_groups;
  auto result_group = cg::tiled_partition<fe_degree + 1>(cg::this_thread_block());
  for (uint32_t gid = result_group.meta_group_rank(); gid < dealii::Utilities::pow(fe_degree + 1, dim);
       gid += result_group.meta_group_size()) {
    auto i = gid % (fe_degree + 1);
    auto j = (gid / (fe_degree + 1)) % (fe_degree + 1);
    auto k = gid / ((fe_degree + 1) * (fe_degree + 1));

    auto dir_idx = [](auto ii, auto jj, auto kk) -> uint32_t {
      if constexpr (dir == 0) { return ii; }
      if constexpr (dir == 1) { return jj; }
      if constexpr (dir == 2) { return kk; }
      __builtin_unreachable();
    };

    auto map = [](auto ii, auto jj, auto kk) -> uint32_t {
      if constexpr (dim == 1) { return ii; }
      if constexpr (dim == 2) { return ii + jj * (fe_degree + 1); }
      if constexpr (dim == 3) { return ii + jj * (fe_degree + 1) + kk * (fe_degree + 1) * (fe_degree + 1); }
      __builtin_unreachable();
    };

    auto acc = [&](auto ii, auto jj, auto kk, auto qq) -> uint32_t {
      if constexpr (dir == 0) { return map(qq, jj, kk); }
      if constexpr (dir == 1) { return map(ii, qq, kk); }
      if constexpr (dir == 2) { return map(ii, jj, qq); }
      __builtin_unreachable();
    };

    auto val = gko::zero<Number2>();
    for (auto q = result_group.thread_rank(); q < fe_degree + 1; q += result_group.num_threads()) {
      val += static_cast<Number2>(transpose ? shape_data(q, dir_idx(i, j, k)) : shape_data(dir_idx(i, j, k), q)) *
                 static_cast<Number2>(in.data[acc(i, j, k, q)]);
    }

    val = gko::kernels::cuda::reduce(result_group, val, std::plus<>{});

    if ( result_group.thread_rank() == 0 ) {
      out(i, j, k) = (additive ? out(i, j, k) : 0) + val;
    }
  }
}

template<bool additive = false, typename TA>
__device__ void gemm(TA a, col_major_matrix<const DGAdvection::Number> b, col_major_matrix<DGAdvection::Number> c) {
  // Create frags
  auto fragA = wmma::fragment<wmma::matrix_a, wmma::M, wmma::N, wmma::K, DGAdvection::Number, type_to_layout<TA>>();
  auto fragB = wmma::fragment<wmma::matrix_b, wmma::M, wmma::N, wmma::K, DGAdvection::Number, wmma::col_major>();
  auto fragAcc = wmma::accumulator_fragment<wmma::M, wmma::N, wmma::K, DGAdvection::Number, wmma::col_major>();
  // Tile using a 1D grid
  // Target C block
  auto tid = static_cast<int>(threadIdx.x);
  auto wid = tid / wmma::WAVE_SIZE;

  if (tid >= wmma::T_BLOCK_X) { return; }

  auto m = c.num_rows;
  auto n = c.num_cols;
  auto k = a.num_cols;

  // Bounds check
  for (int cRow = 0; cRow < m; cRow += wmma::M) {
    for (int cCol = wid * wmma::N; cCol < n; cCol += wmma::N * wmma::WAVES) {
      if constexpr (!additive) { wmma::fill_fragment(fragAcc, static_cast<DGAdvection::Number>(0.0)); }
      else {
        wmma::load_matrix_sync(fragAcc, &c(cRow, cCol), c.stride,
                               runtime_layout<col_major_matrix<DGAdvection::Number>>);
      }

      for (int i = 0; i < k; i += wmma::K) {
        // Load the inputs
        wmma::load_matrix_sync(fragA, &a(cRow, i), a.stride);
        wmma::load_matrix_sync(fragB, &b(i, cCol), b.stride);

        // Matrix multiply - accumulate using MFMA units<
        wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
      }

      // Store to D
      wmma::unified_store_matrix_sync(&c(cRow, cCol), fragAcc, c.stride,
                                      runtime_layout<col_major_matrix<DGAdvection::Number>>);
    }
  }
}

template<typename T>
__device__ void transpose(col_major_matrix<const T> in, col_major_matrix<T> out) {
  auto size = in.num_rows * in.num_cols;

  for (auto tid = threadIdx.x; tid < size; tid += blockDim.x) {
    auto row = tid % in.num_rows;
    auto col = tid / in.num_rows;

    out(col, row) = in(row, col);
  }
}

template<typename T>
constexpr col_major_matrix<T> reshape(col_major_matrix<T> in, std::array<int, 2> size) {
  assert(in.num_rows * in.num_cols == size[0] * size[1]);
  return {in.values, size[0], size[0], size[1]};
}

template<typename Number>
__device__ void rotate_left(col_major_matrix<const Number> in, col_major_matrix<Number> out) {
  auto m = in.num_rows;
  auto n = in.num_cols;
  transpose(in, reshape(out, {n, m}));
}

template<typename Number>
__device__ void rotate_right(col_major_matrix<const Number> in, col_major_matrix<Number> out) {
  auto m = in.num_rows;
  auto n = in.num_cols;
  transpose(reshape(in, {n, m}), out);
}

enum class fill_mode { add, overwrite };

enum class rotation_mode : uint32_t { none = 0, all = ~uint32_t{}, in = 1 << 1, out = 1 << 2 };

constexpr bool operator&(rotation_mode a, rotation_mode b) {
  return static_cast<uint32_t>(a) & static_cast<uint32_t>(b);
}

template<int dim, fill_mode fm = fill_mode::overwrite, rotation_mode rm = rotation_mode::all, typename TA>
__device__ void tensor_apply(TA a,
                             col_major_matrix<const DGAdvection::Number> b,
                             col_major_matrix<DGAdvection::Number> c,
                             std::array<col_major_matrix<DGAdvection::Number>, 2> buffer) {
  static_assert(dim == 1 || fm != fill_mode::add || rm & rotation_mode::out,
                "Additive mode needs rotation on output for dim > 1");
  using gko::batch::matrix::to_const;

  if constexpr (dim == 1) { gemm<fm == fill_mode::add>(a, b, c); }
  else {
    col_major_matrix<const DGAdvection::Number> in = (rm & rotation_mode::in) ? to_const(buffer[0]) : b;
    col_major_matrix<DGAdvection::Number> out = (rm & rotation_mode::out || fm == fill_mode::add) ? buffer[1] : c;

    auto rotate_in = [](auto in_, auto out_) {
      if constexpr (dim == 2) {
        rotate_left(in_, out_);
        return;
      }
      if constexpr (dim == 3) {
        rotate_right(in_, out_);
        return;
      }
      __builtin_unreachable();
    };

    auto rotate_out = [](auto in_, auto out_) {
      if constexpr (dim == 2) {
        rotate_right(in_, out_);
        return;
      }
      if constexpr (dim == 3) {
        rotate_left(in_, out_);
        return;
      }
      __builtin_unreachable();
    };

    if constexpr (fm == fill_mode::add) { rotate_in(to_const(c), buffer[1]); }

    if constexpr (rm & rotation_mode::in) { rotate_in(b, buffer[0]); }

    if constexpr (fm == fill_mode::add || rm & rotation_mode::in) { __syncthreads(); }

    gemm<fm == fill_mode::add>(a, in, out);

    if constexpr (rm & rotation_mode::out) {
      __syncthreads();
      rotate_out(to_const(buffer[1]), c);
    }
  }
}
