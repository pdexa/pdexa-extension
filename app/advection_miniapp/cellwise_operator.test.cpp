// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "cellwise_operator.hpp"
#include "cellwise_operator.apply.hpp"

#include <gtest/gtest.h>

template<typename T>
using I = std::initializer_list<T>;

constexpr int fe_degree = 7;

template<int fe_degree, int dim, typename Number, typename VectorizedNumber = Number>
__global__ void face_kernel(gko::batch::matrix::dense::batch_item<const VectorizedNumber> jac,
                            gko::batch::matrix::dense::batch_item<const VectorizedNumber> normal_speed_faces,
                            const Number* shape_values,
                            const Number* weights,
                            Number time_factor,
                            gko::batch::multi_vector::batch_item<const VectorizedNumber> src,
                            gko::batch::multi_vector::batch_item<VectorizedNumber> dst) {
  DGAdvection::gpu::face_kernel<fe_degree, dim, Number, VectorizedNumber>(jac, normal_speed_faces, shape_values,
                                                                          weights, time_factor, src, dst);
}

template<int fe_degree, int dim, typename Number1, typename Number2>
__device__ void only_tensor_apply(gko::batch::matrix::dense::batch_item<const Number1> shape_gradients,
                                  gko::batch::multi_vector::batch_item<const Number2> in,
                                  gko::batch::multi_vector::batch_item<Number2> out,
                                  Number2* grad_buffer = nullptr) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  constexpr int dofs_per_component = dealii::Utilities::pow(fe_degree + 1, dim);
  std::array<tensor3d<Number2>, dim> gradients{};
  if constexpr (sizeof(Number2) <= 8) {
    __shared__ Number2 gradient_storage[dim][dofs_per_component];
    for (uint32_t i = 0; i < dim; ++i) { gradients[i] = tensor3d(gradient_storage[i], tensor_size); }
  }
  else {
    for (uint32_t i = 0; i < dim; ++i) { gradients[i] = tensor3d(grad_buffer + i * dofs_per_component, tensor_size); }
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

template<int fe_degree, int dim, typename Number, typename VectorizedNumber = Number>
__global__ void only_tensor_apply_kernel(gko::batch::matrix::dense::batch_item<const Number> shape_gradients,
                                         gko::batch::multi_vector::batch_item<const VectorizedNumber> in,
                                         gko::batch::multi_vector::batch_item<VectorizedNumber> out,
                                         VectorizedNumber* grad_buffer = nullptr) {
  only_tensor_apply<fe_degree, dim, Number, VectorizedNumber>(shape_gradients, in, out, grad_buffer);
}

template<int fe_degree, int dim, typename Number1, typename Number2>
__device__ void cell_diagonal_scaling(gko::batch::matrix::dense::batch_item<const Number2> jac,
                                      gko::batch::matrix::dense::batch_item<const Number2> speed_cells,
                                      std::array<tensor3d<Number2>, dim> gradients,
                                      const Number1* cell_weights,
                                      Number1 time_factor,
                                      Number1 inv_dt,
                                      gko::batch::multi_vector::batch_item<const Number2> in,
                                      gko::batch::multi_vector::batch_item<Number2> out) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  constexpr int dofs_per_component = dealii::Utilities::pow(fe_degree + 1, dim);

  auto in_tensor = tensor3d{in.values, tensor_size};
  auto out_tensor = tensor3d{out.values, tensor_size};

  const auto JxW = 1. / determinant<dim>(jac);
  for (unsigned int q = threadIdx.x; q < dofs_per_component; q += blockDim.x) {
    auto u = in[q];
    auto speed_gradu = gko::zero<Number2>();
    for (unsigned int d = 0; d < dim; ++d) {
      speed_gradu = speed_gradu + speed_cells(q, d) * gradients[d].data[q] * jac(d, d);
    }
    auto flux = DGAdvection::factor_skew * time_factor * speed_gradu;
    const auto result = (-1.0 + DGAdvection::factor_skew) * time_factor * u * (JxW * cell_weights[q]);
    for (unsigned int d = 0; d < dim; ++d) { gradients[d].data[q] = result * speed_cells(q, d) * jac(d, d); }

    // mass matrix part
    u = u * inv_dt;
    flux = flux + u;
    out[q] = out[q] + flux * (JxW * cell_weights[q]);
  }
}

template<int fe_degree, int dim, typename Number, typename VectorizedNumber = Number>
__global__ void diagonal_scaling_kernel(gko::batch::matrix::dense::batch_item<const VectorizedNumber> jac,
                                        gko::batch::matrix::dense::batch_item<const VectorizedNumber> speed_cells,
                                        std::array<tensor3d<VectorizedNumber>, dim> gradients,
                                        const Number* cell_weights,
                                        Number time_factor,
                                        Number inv_dt,
                                        gko::batch::multi_vector::batch_item<const VectorizedNumber> in,
                                        gko::batch::multi_vector::batch_item<VectorizedNumber> out) {
  cell_diagonal_scaling<fe_degree, dim, Number, VectorizedNumber>(jac, speed_cells, gradients, cell_weights,
                                                                  time_factor, inv_dt, in, out);
}

template<int fe_degree, int dim, typename Number, typename VectorizedNumber = Number>
__global__ void simple_cell_kernel(gko::batch::matrix::dense::batch_item<const VectorizedNumber> jac,
                                   gko::batch::matrix::dense::batch_item<const Number> shape_gradients,
                                   gko::batch::matrix::dense::batch_item<const VectorizedNumber> speed_cells,
                                   const Number* cell_weights,
                                   Number time_factor,
                                   Number inv_dt,
                                   gko::batch::multi_vector::batch_item<const VectorizedNumber> in,
                                   gko::batch::multi_vector::batch_item<VectorizedNumber> out,
                                   VectorizedNumber* grad_buffer = nullptr) {
  DGAdvection::gpu::simple::cell_apply<fe_degree, dim, false, Number, VectorizedNumber>(
    jac, shape_gradients, speed_cells, cell_weights, time_factor, inv_dt, in, out, grad_buffer);
}

template<int fe_degree, int dim, typename Number, typename VectorizedNumber = Number>
__global__ void
cell_apply_single_gradient_kernel(gko::batch::matrix::dense::batch_item<const VectorizedNumber> jac,
                                  gko::batch::matrix::dense::batch_item<const Number> shape_gradients,
                                  gko::batch::matrix::dense::batch_item<const VectorizedNumber> speed_cells,
                                  const Number* cell_weights,
                                  Number time_factor,
                                  Number inv_dt,
                                  gko::batch::multi_vector::batch_item<const VectorizedNumber> in,
                                  gko::batch::multi_vector::batch_item<VectorizedNumber> out) {
  DGAdvection::gpu::simple::cell_apply_single_gradient<fe_degree, dim, Number, VectorizedNumber>(
    jac, shape_gradients, speed_cells, cell_weights, time_factor, inv_dt, in, out);
}

namespace test_dim2 {
constexpr int dim = 2;

TEST(Operator2d, FaceKernel) {
  using namespace DGAdvection;
  auto exec =
    gko::CudaExecutor::create(0, gko::ReferenceExecutor::create(), std::make_shared<gko::CudaUnifiedAllocator>(0));
  constexpr int face_size = dealii::Utilities::pow(::fe_degree + 1, dim - 1);
  constexpr int cell_size = dealii::Utilities::pow(::fe_degree + 1, dim);
  auto jac_arr = gko::array<double>(exec, I<double>{-10, 1, 5, -5, -3, -6, 10, 0, 1, -5, -8, 7, 8, -1, -5, -5});
  auto shape_arr = gko::array<double>(exec, I<double>{-7, -4, -8, 6, 1, 7, 5, -9});
  auto normal_speed_faces_arr = gko::array<double>(
    exec, I<double>{8,   5,  -1, -6, 2,  7,  -10, -5, 2,   5,   -2,  4,   6,  10, 8,  8,   -2,  6,  -4, 5,  -7, -10,
                    4,   -1, 10, -4, 3,  7,  1,   7,  5,   1,   -9,  9,   -1, 2,  5,  -1,  -9,  -6, 8,  8,  3,  -8,
                    -10, -5, 5,  -9, 4,  4,  0,   1,  5,   -1,  3,   -9,  6,  9,  2,  10,  4,   -4, 10, -3, 10, -9,
                    -7,  1,  6,  9,  4,  -4, 10,  2,  -10, -8,  -10, -10, 10, -5, 6,  8,   -10, -9, -5, 9,  2,  -5,
                    -2,  7,  2,  6,  10, -8, 0,   -4, -3,  -10, -9,  -10, 3,  -9, 6,  -10, 5,   1,  -3, -8, 5,  -4,
                    -7,  9,  -1, -1, -4, 0,  -9,  -7, -9,  3,   -5,  -2,  1,  -1, -4, -5,  -8,  -1

          });
  auto weights_arr = gko::array<double>(exec, I<double>{-6, 4, 2, 7, -3, -7, -10, -4});
  auto src_arr = gko::array<double>(
    exec,
    I<double>{5,   9,  -7,  3,   0,   3,  -3,  -8,  -9, -1,  3,   -3,  -1, 10,  -6, 10,  -6,  -6,  9,   -9, 7,   9,
              -9,  6,  6,   0,   -9,  -3, 3,   -7,  8,  -4,  4,   3,   8,  -1,  5,  7,   -9,  8,   -8,  -6, -10, -9,
              5,   10, -2,  10,  -2,  2,  -7,  -6,  -9, -3,  -10, -9,  9,  -4,  9,  10,  -8,  -7,  2,   -8, 3,   -3,
              7,   -7, -10, -1,  -10, -4, -1,  -5,  -5, -7,  0,   2,   -7, -1,  -9, -6,  -10, -5,  -7,  6,  -4,  9,
              4,   -2, -4,  5,   -2,  -7, 7,   -7,  -9, 1,   7,   -10, -3, -10, -8, -9,  -5,  -10, 8,   -6, -9,  -4,
              -6,  4,  -10, 4,   -1,  -9, -6,  -10, -6, 3,   -10, -6,  7,  1,   5,  10,  0,   -2,  1,   -8, -1,  -1,
              2,   -6, -3,  5,   1,   4,  -8,  -8,  0,  6,   9,   5,   -5, -4,  2,  -10, -6,  5,   0,   -1, -3,  4,
              -10, -8, 4,   -10, -1,  -4, 0,   -2,  9,  -10, -2,  3,   -5, 6,   4,  -7,  3,   10,  4,   -6, 6,   -7,
              -9,  5,  4,   -10, 4,   -1, 8,   2,   0,  2,   1,   3,   10, 9,   -1, -8,  -2,  10,  -8,  0,  -3,  -9,
              7,   -6, 1,   -9,  -4,  -3, -5,  -5,  7,  -8,  -4,  -7,  -6, -6,  -1, 2,   -10, 7,   -10, -2, -2,  -9,
              -8,  -3, 5,   9,   2,   6,  -6,  0,   -5, 5,   2,   -2,  7,  4,   4,  10,  -2,  -10, -1,  9,  3,   -5,
              9,   1,  -6,  -1,  6,   -5, -10, -10, -3, -3,  -10, -8,  -4, 8});
  auto dst_arr = gko::array<double>(exec, cell_size * 4);
  auto expected_dst_arr = gko::array<double>(
    exec, I<double>{
            14679,   53343,    0,        5695.2,  4721.4,   -3222,    5232,     -2520,    5793,     -105618,  0,
            -3528,   -2882.25, 20628,    -24255,  1537.2,   308.7,    -17163,   0,        -3024,    -4851,    20718,
            1499.4,  6753.6,   -3465,    134043,  6612,     9324,     -1365,    23409,    0,        3528,     3979.5,
            24402,   0,        662.4,    370.5,   51960,    -5481.6,  0,        199.5,    30978,    0,        0,
            240.75,  -20916,   -13860,   -1134,   1124.4,   17682,    0,        0,        489,      -78078,   856.8,
            -1512,   465,      22050,    1770,    4176,     -6139.5,  31374,    0,        0,        13463.3,  9222,
            -3499.2, 5709.6,   106.8,    90552,   -6513.6,  2505.6,   -225,     32286,    2721.6,   5011.2,   -1212.75,
            -14028,  -27331.2, -5346,    -361.95, 45318,    2332.8,   -626.4,   2052.75,  -132078,  -1396.8,  -6501.6,
            1034.25, 62862,    2578.8,   5220,    -6971.25, 16026,    -2721.6,  5637.6,   -3683.25, -71295,   -18385.5,
            24055.2, 1364.4,   -97764,   -5229.3, 14313.6,  4701,     -85554,   -13933.5, 28627.2,  -4205.25, 61110,
            33799.5, -21697.2, -1535.85, -21567,  4557,     -3578.4,  -4740.75, 151809,   10138.8,  -25351.2, -3386.25,
            -8295,   4524,     -24156,   12611.2, -91665,   -21766.5, 32205.6,  -1201.5,  -53217,   -0,       -1982.7,
            -620.1,  -27864,   -4684.8,  -1120.5, 160.5,    -40578,   0,        -2216.7,  -873,     37638,    3465,
            251.1,   -278.1,   7911,     0,       221.4,    180,      46440,    -214.2,   140.4,    90,       15753,
            -1878,   338.4,    1243.5,   -62991,  0,        -2373.3,  -17742.4, 0,        4048.8,   -1159.2,  3273.68,
            -66528,  3867.6,   0,        4631.62, -5358,    -8005.2,  0,        2013.38,  -0,       25691.4,  1814.4,
            4274.55, -37044,   -4023.6,  0,       -7035,    93933,    6875.4,   2419.2,   -3517.5,  -69090,   1483.2,
            -7308,   1446.38,  0,        1864.8,  0,        1455,     281880,   -3735,    59433,    -5741.85, -204120,
            9106.8,  -19728,   -7803.75, -222954, 6414,     -23436,   -276.75,  -31320,   16377,    -14890.8, -6434.25,
            -214380, 3447,     -33783,   7781.25, 317655,   -7842,    31728.6,  3918.75,  75930,    -6435,    7425,
            11955,   219240,   -1977,    53067,   17175.8,  42336,    -17652,   2658,     -598.8,   109728,   9496.8,
            667.2,   -3483,    54132,    8244,    1334.4,   726.75,   -36288,   -27093,   586.8,    -1906.35, 41580,
            10272,   -166.8,   4640.25,  -163107, -8368.2,  949.2,    2388.75,  58590,    3060,     8562,     -6405.75,
            54432,   -15180,   1501.2});
  using VectorizedNumber = double4;
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> jac{
    reinterpret_cast<const VectorizedNumber*>(jac_arr.get_const_data()), dim, dim, dim};
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> normal_speed_faces{
    reinterpret_cast<const VectorizedNumber*>(normal_speed_faces_arr.get_const_data()), face_size, 2 * dim, face_size};
  gko::batch::multi_vector::batch_item<const VectorizedNumber> src{
    reinterpret_cast<const VectorizedNumber*>(src_arr.get_const_data()), 1, cell_size, 1};
  gko::batch::multi_vector::batch_item<VectorizedNumber> dst{reinterpret_cast<VectorizedNumber*>(dst_arr.get_data()), 1,
                                                             cell_size, 1};
  double time_factor = 1.5;

  dst_arr.fill(0.0);
  face_kernel<::fe_degree, dim><<<1, cell_size>>>(jac, normal_speed_faces, shape_arr.get_const_data(),
                                                  weights_arr.get_const_data(), time_factor, src, dst);
  exec->synchronize();

  for (int i = 0; i < cell_size * 4; ++i) {
    auto result = dst_arr.get_data()[i];
    auto expected = expected_dst_arr.get_data()[i];
    auto eps = (std::abs(result) + std::abs(expected)) * 1e-5;
    EXPECT_NEAR(result, expected, eps) << i / 4 << " " << i % 4;
  }
}

TEST(Operator2d, FaceKernelNegativeTimeFactor) {
  using namespace DGAdvection;
  auto exec =
    gko::CudaExecutor::create(0, gko::ReferenceExecutor::create(), std::make_shared<gko::CudaUnifiedAllocator>(0));
  constexpr int face_size = dealii::Utilities::pow(::fe_degree + 1, dim - 1);
  constexpr int cell_size = dealii::Utilities::pow(::fe_degree + 1, dim);
  auto jac_arr = gko::array<double>(exec, I<double>{-10, 1, 5, -5, -3, -6, 10, 0, 1, -5, -8, 7, 8, -1, -5, -5});
  auto shape_arr = gko::array<double>(exec, I<double>{-7, -4, -8, 6, 1, 7, 5, -9});
  auto normal_speed_faces_arr = gko::array<double>(
    exec, I<double>{8,   5,  -1, -6, 2,  7,  -10, -5, 2,   5,   -2,  4,   6,  10, 8,  8,   -2,  6,  -4, 5,  -7, -10,
                    4,   -1, 10, -4, 3,  7,  1,   7,  5,   1,   -9,  9,   -1, 2,  5,  -1,  -9,  -6, 8,  8,  3,  -8,
                    -10, -5, 5,  -9, 4,  4,  0,   1,  5,   -1,  3,   -9,  6,  9,  2,  10,  4,   -4, 10, -3, 10, -9,
                    -7,  1,  6,  9,  4,  -4, 10,  2,  -10, -8,  -10, -10, 10, -5, 6,  8,   -10, -9, -5, 9,  2,  -5,
                    -2,  7,  2,  6,  10, -8, 0,   -4, -3,  -10, -9,  -10, 3,  -9, 6,  -10, 5,   1,  -3, -8, 5,  -4,
                    -7,  9,  -1, -1, -4, 0,  -9,  -7, -9,  3,   -5,  -2,  1,  -1, -4, -5,  -8,  -1

          });
  auto weights_arr = gko::array<double>(exec, I<double>{-6, 4, 2, 7, -3, -7, -10, -4});
  auto src_arr = gko::array<double>(
    exec,
    I<double>{5,   9,  -7,  3,   0,   3,  -3,  -8,  -9, -1,  3,   -3,  -1, 10,  -6, 10,  -6,  -6,  9,   -9, 7,   9,
              -9,  6,  6,   0,   -9,  -3, 3,   -7,  8,  -4,  4,   3,   8,  -1,  5,  7,   -9,  8,   -8,  -6, -10, -9,
              5,   10, -2,  10,  -2,  2,  -7,  -6,  -9, -3,  -10, -9,  9,  -4,  9,  10,  -8,  -7,  2,   -8, 3,   -3,
              7,   -7, -10, -1,  -10, -4, -1,  -5,  -5, -7,  0,   2,   -7, -1,  -9, -6,  -10, -5,  -7,  6,  -4,  9,
              4,   -2, -4,  5,   -2,  -7, 7,   -7,  -9, 1,   7,   -10, -3, -10, -8, -9,  -5,  -10, 8,   -6, -9,  -4,
              -6,  4,  -10, 4,   -1,  -9, -6,  -10, -6, 3,   -10, -6,  7,  1,   5,  10,  0,   -2,  1,   -8, -1,  -1,
              2,   -6, -3,  5,   1,   4,  -8,  -8,  0,  6,   9,   5,   -5, -4,  2,  -10, -6,  5,   0,   -1, -3,  4,
              -10, -8, 4,   -10, -1,  -4, 0,   -2,  9,  -10, -2,  3,   -5, 6,   4,  -7,  3,   10,  4,   -6, 6,   -7,
              -9,  5,  4,   -10, 4,   -1, 8,   2,   0,  2,   1,   3,   10, 9,   -1, -8,  -2,  10,  -8,  0,  -3,  -9,
              7,   -6, 1,   -9,  -4,  -3, -5,  -5,  7,  -8,  -4,  -7,  -6, -6,  -1, 2,   -10, 7,   -10, -2, -2,  -9,
              -8,  -3, 5,   9,   2,   6,  -6,  0,   -5, 5,   2,   -2,  7,  4,   4,  10,  -2,  -10, -1,  9,  3,   -5,
              9,   1,  -6,  -1,  6,   -5, -10, -10, -3, -3,  -10, -8,  -4, 8});
  auto dst_arr = gko::array<double>(exec, cell_size * 4);
  auto expected_dst_arr = gko::array<double>(
    exec, I<double>{-22100.8, -202257,  28971,    -8510.4,  12453.8,  33048,    2745,     5066.4,   17435.2,  0,
                    4935,     -22920,   -890.25,  34503,    1908,     -8205.3,  14912.1,  1012.5,   -9097.2,  4217.4,
                    -7058.25, -17860.5, -16312.5, 8838.9,   -13987.5, 12420,    -2746.8,  5400,     -16787.2, 81558,
                    365.4,    -13898.4, -175.5,   71154,    -28278.6, 8300.4,   0,        -20670,   10758,    -2877.6,
                    0,        -3234,    11775.6,  12129.6,  -1932,    80598,    403.8,    -16311.6, 18,       -3334.5,
                    15964.8,  -1543.2,  -10379.3, 13618.5,  -10677.9, -5341.2,  5383.5,   -5052,    -8563.2,  -1947.6,
                    -360,     72660,    -15267.6, 10090.8,  -245.7,   79992,    -10722.6, -5598,    0,        -25704,
                    -633.6,   -4639.2,  0,        0,        -6708,    12696,    -3864,    143976,   -1725,    -20028,
                    25.2,     -787.5,   3333.6,   5284.8,   -15513.8, 13891.5,  10237.5,  -5844,    8494.5,   -9660,
                    792,      522,      -504,     132102,   -1224,    -4239.6,  -1806.97, 17068.5,  -8483.4,  -10682.1,
                    984.375,  16015.5,  0,        3558.9,   1378.12,  27562.5,  1965.6,   8334.3,   3094.88,  -66244.5,
                    -382.2,   16461.9,  1184.85,  23512.5,  3153.6,   4681.8,   -866.25,  -29515.5, 1304.1,   -7144.2,
                    -2424,    -17130,   0,        -4317.6,  -1450.12, -112576,  28.8,     -7709.1,  -1163.48, 113751,
                    -18531,   -1836,    -544.5,   -22032,   -1411.2,  -7017.6,  -1089,    0,        -4468.8,  7296,
                    1299.75,  23583,    -176.4,   2551.5,   157.725,  -675,     6044.4,   -372.6,   -9127.13, 11907,
                    10294.2,  346.5,    4826.62,  -8280,    1764,     1764,     -1657.12, -1044,    -3002.4,  -688.8,
                    -4285.39, -61060.5, 13120.2,  -23457.6, -2609.25, 47908.5,  0,        22020.3,  -5218.5,  38797.5,
                    5395.2,   8637.3,   7294.88,  -170026,  3057.6,   20903.4,  623.513,  -7920,    -4438.8,  14546.7,
                    21418.7,  -48741,   -10432.8, -20185.2, -5591.44, -13425,   0,        -13610.1, -5294.81, -67570.5,
                    -230.4,   -18662.4, 140.4,    -56685,   5067,     1224,     0,        7008,     0,        3672,
                    -0,       -15360,   3273.6,   -5788.8,  2415,     -76197,   1528.8,   12757.5,  -14.4,    2370,
                    -1652.4,  -1863,    9040.5,   5502,     -5216.4,  1732.5,   -5025,    15120,    0,        -1176,
                    288,      -99000,   -115.2,   921.6,    245.7,    -191367,  26535.6,  -158.4,   0,        33264,
                    0,        10856.4,  0,        10584,    -1087.2,  -5444.4,  -4347,    73521,    2675.4,   -22707.9,
                    -25.2,    9859.5,   -9601.2,  4887,     8552.25,  -25987.5, -9128.7,  -5163.3,  -1711.5,  3612,
                    0,        -3080.4,  504,      104562,   -201.6,   -1624.8});
  using VectorizedNumber = double4;
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> jac{
    reinterpret_cast<const VectorizedNumber*>(jac_arr.get_const_data()), dim, dim, dim};
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> normal_speed_faces{
    reinterpret_cast<const VectorizedNumber*>(normal_speed_faces_arr.get_const_data()), face_size, 2 * dim, face_size};
  gko::batch::multi_vector::batch_item<const VectorizedNumber> src{
    reinterpret_cast<const VectorizedNumber*>(src_arr.get_const_data()), 1, cell_size, 1};
  gko::batch::multi_vector::batch_item<VectorizedNumber> dst{reinterpret_cast<VectorizedNumber*>(dst_arr.get_data()), 1,
                                                             cell_size, 1};
  double time_factor = -1.5;

  dst_arr.fill(0.0);
  face_kernel<::fe_degree, dim><<<1, cell_size>>>(jac, normal_speed_faces, shape_arr.get_const_data(),
                                                  weights_arr.get_const_data(), time_factor, src, dst);
  exec->synchronize();

  for (int i = 0; i < cell_size * 4; ++i) {
    auto result = dst_arr.get_data()[i];
    auto expected = expected_dst_arr.get_data()[i];
    auto eps = (std::abs(result) + std::abs(expected)) * 1e-5;
    EXPECT_NEAR(result, expected, eps) << i / 4 << " " << i % 4;
  }
}

TEST(Operator2d, CellDiagonalScalingKernel) {
  using namespace DGAdvection;
  auto exec =
    gko::CudaExecutor::create(0, gko::ReferenceExecutor::create(), std::make_shared<gko::CudaUnifiedAllocator>(0));
  constexpr int face_size = dealii::Utilities::pow(::fe_degree + 1, dim - 1);
  constexpr int cell_size = dealii::Utilities::pow(::fe_degree + 1, dim);
  auto jac_arr = gko::array<double>(exec, I<double>{-10, 1, 5, -5, -3, -6, 10, 0, 1, -5, -8, 7, 8, -1, -5, -5});
  auto speed_cells_arr = gko::array<double>(
    exec,
    I<double>{
      10,  -4, 6,   8,   5,  1,   7,  1,   -4, -3,  -4, -1, 1,   3,   -9,  0,  -1, -10, 4,  -2, -1,  9,   -8,  -2,  -4,
      -2,  -5, -3,  4,   9,  -10, 0,  7,   1,  -8,  -2, -8, 7,   -2,  -9,  2,  -3, -5,  -2, -4, 2,   -6,  -5,  10,  -6,
      3,   0,  6,   -8,  -3, -6,  -4, -7,  10, 5,   2,  0,  0,   -3,  -4,  -2, -2, -1,  -4, -3, -8,  5,   -2,  -10, -3,
      1,   -6, 8,   9,   10, 6,   -6, 0,   -6, 8,   7,  10, -3,  3,   4,   8,  -8, 4,   10, 10, -8,  -3,  -10, 1,   -7,
      0,   -7, 6,   -5,  -6, -3,  5,  -2,  10, 2,   -1, 3,  9,   0,   -1,  -9, 5,  -7,  2,  2,  -3,  -2,  -6,  7,   0,
      3,   0,  -9,  3,   4,  -7,  6,  -6,  -9, -1,  5,  10, 1,   -10, 3,   -2, 0,  -3,  -3, 7,  7,   9,   1,   -9,  5,
      -5,  2,  -10, -1,  3,  8,   -3, 8,   10, -1,  0,  8,  -10, -10, 0,   1,  -2, 1,   3,  2,  3,   -8,  6,   6,   10,
      -1,  -9, -1,  10,  10, -2,  -4, 10,  6,  -3,  -4, -9, 9,   3,   9,   5,  7,  -7,  -1, -9, -4,  -5,  -2,  5,   5,
      -5,  -1, 6,   3,   0,  6,   7,  -5,  -6, 3,   -6, -3, -5,  3,   -8,  2,  -3, -8,  1,  -2, -2,  9,   -10, -8,  -5,
      4,   7,  1,   -10, 6,  -2,  -8, -7,  -2, -10, -5, 4,  -6,  -9,  4,   5,  -8, 8,   1,  0,  6,   10,  -4,  5,   -4,
      1,   3,  -6,  -2,  2,  2,   -5, 4,   -3, 0,   0,  -3, 5,   6,   8,   -1, -1, -4,  -7, -1, 5,   10,  7,   3,   0,
      -10, 9,  1,   -6,  5,  -3,  -1, -7,  -8, -4,  9,  5,  4,   5,   -3,  -1, 2,  3,   -6, 5,  3,   -7,  -7,  7,   -3,
      7,   9,  8,   -3,  -4, -5,  1,  6,   6,  7,   -3, -9, -6,  -4,  4,   6,  3,  6,   -5, 3,  7,   -10, -2,  -3,  8,
      2,   -5, -4,  10,  5,  -2,  -1, -8,  -9, -6,  7,  5,  -4,  -10, -10, -6, -5, 2,   9,  -2, -1,  6,   -9,  10,  -6,
      7,   1,  -6,  -10, 1,  0,   0,  -7,  8,  9,   5,  6,  0,   -4,  -6,  -9, 10, -6,  9,  8,  1,   7,   -7,  4,   -5,
      9,   -7, 10,  -9,  9,  8,   6,  6,   10, -5,  -2, -7, -8,  -1,  8,   6,  -8, -6,  8,  -5, -3,  5,   -4,  -6,  0,
      -8,  -4, 10,  -1,  -6, 3,   6,  1,   4,  -4,  3,  -2, 8,   2,   -1,  7,  -2, 10,  -5, 4,  -10, 6,   -5,  8,   -3,
      -9,  -2, 5,   9,   -4, 7,   -5, 1,   6,  -2,  5,  2,  -10, 3,   7,   5,  -6, 3,   -6, 3,  -1,  0,   4,   10,  -6,
      -8,  5,  5,   -2,  -3, 0,   6,  5,   10, -6,  3,  7,  -8,  -2,  -3,  -5, -1, -6,  3,  -8, 8,   -8,  0,   4,   7,
      -7,  -2, -9,  4,   -7, -9,  -5, -6,  8,  -9,  -9, -2, -4,  6,   -2,  3,  -8, -7,  9,  8,  7,   -1,  3,   -2,  5,
      -7,  -1, 4,   2,   -5, -2,  4,  -10, 0,  7,   4,  1

    });
  auto weights_arr = gko::array<double>(exec, I<double>{-7, 7, -1,  3,  -1, 6,  1,  -5, -2, 8,  -3, -9,  8,  3,  3, 0,
                                                        -7, 4, -10, 6,  1,  -3, -6, -4, 1,  -9, 5,  -10, -4, -7, 0, 0,
                                                        -4, 1, -7,  -6, 1,  -1, 0,  7,  -6, 8,  -3, -9,  -2, 0,  6, 9,
                                                        -4, 3, -1,  6,  7,  4,  -2, -5, 3,  10, -8, 7,   -4, -3, 9, 1});
  auto gradients_arr = gko::array<double>(
    exec,
    I<double>{
      -8,  -4, 9,  -9,  5,   -2,  3,  -9, 1,  -6,  -8,  10, 9,   -9, 3,  -5,  -1,  9,  -6,  -7, 2,  -6, -1, 3,  9,  -1,
      8,   -7, 5,  -6,  10,  -2,  5,  2,  -5, -8,  -2,  -6, -10, -6, -4, -3,  -3,  -2, -10, -5, 3,  7,  -8, -8, -2, -9,
      9,   0,  -8, 9,   7,   -1,  10, -6, -2, -10, -1,  -7, -5,  -4, 7,  10,  8,   10, 5,   10, -9, 8,  -7, 10, -8, -4,
      6,   -6, -4, 6,   -2,  7,   2,  0,  1,  -9,  -10, 9,  5,   -2, 10, 1,   3,   9,  8,   -1, -2, 1,  -6, 6,  -2, -10,
      2,   -1, 5,  -4,  10,  -4,  6,  3,  7,  2,   -4,  -5, 2,   2,  2,  4,   8,   -6, 1,   -4, 4,  2,  10, -4, 0,  0,
      10,  -5, -1, 7,   -4,  5,   -2, 8,  -2, -10, 10,  3,  9,   1,  7,  9,   3,   1,  6,   -7, 0,  -5, -2, -8, -3, -3,
      1,   9,  -2, -7,  -10, -10, -8, 6,  -3, 1,   -9,  6,  -6,  -7, 9,  1,   7,   -8, -3,  2,  9,  -6, -1, -9, 8,  -9,
      -1,  5,  10, 1,   7,   -9,  8,  6,  2,  6,   -4,  -6, -2,  -6, -6, 1,   -3,  -3, 6,   5,  -7, 0,  -5, 7,  10, -7,
      -5,  3,  -8, 8,   -9,  3,   8,  5,  1,  6,   -2,  5,  -5,  9,  5,  -7,  -9,  -4, 6,   -9, -7, 1,  -7, -7, 10, -3,
      -6,  2,  1,  -10, -3,  1,   7,  4,  -5, -3,  -10, 6,  6,   -5, 1,  -3,  -7,  -7, -5,  -2, 10, 10, -7, 4,  -1, -3,
      -6,  6,  1,  -1,  -7,  2,   -2, 1,  1,  1,   -4,  -7, 0,   10, -3, -10, -10, 0,  9,   -1, 6,  9,  -8, -5, 6,  9,
      1,   -4, 5,  6,   7,   6,   6,  3,  -9, -2,  -10, -6, 3,   6,  8,  4,   4,   -3, 6,   3,  0,  1,  3,  3,  6,  4,
      2,   -3, 9,  1,   10,  3,   4,  -9, -3, -3,  9,   0,  4,   1,  -1, -9,  8,   -9, -6,  1,  10, 8,  1,  8,  -6, 4,
      7,   8,  8,  -8,  6,   3,   8,  8,  2,  7,   -4,  -9, -3,  -3, -7, 8,   -6,  -5, -4,  4,  -7, -9, 8,  4,  10, -7,
      -2,  0,  -4, -9,  -4,  5,   2,  2,  -1, 9,   1,   4,  -6,  -2, 1,  -9,  5,   1,  -7,  4,  9,  2,  -7, 1,  0,  -4,
      -10, -5, -6, -6,  -3,  10,  -5, -1, -6, -3,  0,   -9, 7,   3,  10, -5,  1,   10, 6,   2,  -6, 3,  -9, -2, -2, 0,
      7,   -5, 6,  5,   -10, -8,  -9, 7,  10, 7,   -8,  9,  6,   6,  -9, -8,  5,   0,  5,   -8, -2, 7,  8,  -7, -4, -10,
      5,   -2, 8,  -10, 10,  8,   7,  1,  9,  9,   -4,  2,  -9,  -5, 7,  3,   -9,  2,  -1,  3,  4,  -9, 7,  -6, -3, -8,
      4,   9,  -2, 8,   5,   -3,  10, -5, -5, 4,   -3,  -6, 8,   -9, -9, 6,   -7,  -2, 7,   8,  7,  9,  7,  9,  10, 5,
      2,   -4, 3,  -7,  -10, 9,   0,  -4, 3,  -5,  -8,  -8, 5,   -6, -2, 10,  7,   8});
  auto src_arr = gko::array<double>(
    exec,
    I<double>{5,   9,  -7,  3,   0,   3,  -3,  -8,  -9, -1,  3,   -3,  -1, 10,  -6, 10,  -6,  -6,  9,   -9, 7,   9,
              -9,  6,  6,   0,   -9,  -3, 3,   -7,  8,  -4,  4,   3,   8,  -1,  5,  7,   -9,  8,   -8,  -6, -10, -9,
              5,   10, -2,  10,  -2,  2,  -7,  -6,  -9, -3,  -10, -9,  9,  -4,  9,  10,  -8,  -7,  2,   -8, 3,   -3,
              7,   -7, -10, -1,  -10, -4, -1,  -5,  -5, -7,  0,   2,   -7, -1,  -9, -6,  -10, -5,  -7,  6,  -4,  9,
              4,   -2, -4,  5,   -2,  -7, 7,   -7,  -9, 1,   7,   -10, -3, -10, -8, -9,  -5,  -10, 8,   -6, -9,  -4,
              -6,  4,  -10, 4,   -1,  -9, -6,  -10, -6, 3,   -10, -6,  7,  1,   5,  10,  0,   -2,  1,   -8, -1,  -1,
              2,   -6, -3,  5,   1,   4,  -8,  -8,  0,  6,   9,   5,   -5, -4,  2,  -10, -6,  5,   0,   -1, -3,  4,
              -10, -8, 4,   -10, -1,  -4, 0,   -2,  9,  -10, -2,  3,   -5, 6,   4,  -7,  3,   10,  4,   -6, 6,   -7,
              -9,  5,  4,   -10, 4,   -1, 8,   2,   0,  2,   1,   3,   10, 9,   -1, -8,  -2,  10,  -8,  0,  -3,  -9,
              7,   -6, 1,   -9,  -4,  -3, -5,  -5,  7,  -8,  -4,  -7,  -6, -6,  -1, 2,   -10, 7,   -10, -2, -2,  -9,
              -8,  -3, 5,   9,   2,   6,  -6,  0,   -5, 5,   2,   -2,  7,  4,   4,  10,  -2,  -10, -1,  9,  3,   -5,
              9,   1,  -6,  -1,  6,   -5, -10, -10, -3, -3,  -10, -8,  -4, 8});
  auto dst_arr = gko::array<double>(exec, cell_size * 4);
  auto expected_gradients_arr = gko::array<double>(
    exec,
    I<double>{68.1818,  12.1935,  -40.0909, -50.4,     0,         -3.04839,  -11.4545,  16.8,       1.75325,  -0.483871,
              1.63636,  -1.8,     -2.33766, -2.90323,  -12.2727,  -27,       -8.18182,  0.290323,   -9.81818, -5.4,
              -16.3636, -7.83871, -36.8182, -21.6,     -11.6883,  -0,        3.68182,   -0,         -11.6883, -11.8548,
              54.5455,  30,       -6.23377, 0.580645,  -4.36364,  -0.6,      15.5844,   -27.0968,   -29.4545, 19.2,
              -28.0519, -5.22581, -0,       -48.6,     26.2987,   -17.4194,  -19.6364,  216,        -9.35065, -7.74194,
              7.63636,  100.8,    -31.5584, 1.30645,   20.4545,   16.2,      -47.3377,  -0,         3.68182,  -81,
              -0,       0,        0,        -0,        12.2727,   4.06452,   -46.7727,  88.2,       77.9221,  -0.193548,
              -54.5455, -14.4,    -13.6364, 16.9355,   -61.3636,  21,        0,         -0.580645,  17.1818,  -14.4,
              0,        -2.32258, -13.6364, 15,        -12.2727,  -1.74194,  -4.90909,  64.8,       -42.0779, -0.580645,
              -32.7273, -90,      4.67532,  -5.41935,  -34.3636,  75.6,      -12.2727,  -0.0483871, 8.59091,  12,
              26.2987,  -4.35484, -58.9091, 72.9,      -29.2208,  -7.25806,  32.7273,   27,         52.5974,  -15.4839,
              -8.18182, 24,       38.961,   -3.09677,  -3.81818,  10.8,      57.2727,   -6.77419,   57.2727,  31.5,
              0,        0,        -0,       0,         -0,        -0,        -0,        -0,         -3.8961,  6.19355,
              1.63636,  0,        -3.11688, 0.290323,  -0.409091, -6,        9.54545,   -4.06452,   -0,       -168,
              -0,       1.74194,  -51.5455, 72,        4.87013,   0.580645,  0.272727,  -6,         8.18182,  1.69355,
              0,        -0.9,     -0,       -0,        0,         -0,        32.7273,   13.5484,    3.81818,  -50.4,
              0,        -5.80645, -14.7273, -54,       31.1688,   5.80645,   -10.9091,  -14.4,      11.6883,  -4.06452,
              -12.2727, 90,       -14.026,  -2.6129,   44.1818,   -170.1,    21.039,    4.83871,    1.09091,  0,
              -0,       -0,       -0,       -0,        -0,        4.64516,   -0.818182, 37.8,       122.727,  39.1935,
              -11.0455, -194.4,   7.79221,  3.87097,   30.5455,   0,         -10.5195,  -10.4516,   14.3182,  16.2,
              -1.55844, -1.74194, -5.45455, -0.9,      23.3766,   5.80645,   -17.1818,  28.8,       -10.9091, -23.7097,
              -28.6364, -50.4,    -2.33766, -3.48387,  -10.9091,  42,        -3.8961,   1.16129,    1.09091,  27,
              -38.961,  -4.35484, 10.2273,  81,        -11.6883,  -5.22581,  -19.6364,  0,          58.4416,  12.0968,
              -27.2727, 36,       -32.7273, 7.74194,   -4.36364,  144,       0,         -13.5484,   6.68182,  -132.3,
              -21.039,  -4.83871, -29.4545, -9.6,      -21.039,   -0.290323, 7.36364,   -36,        -17.5325, -13.0645,
              -7.36364, -40.5,    -9.74026, 0.774194,  2.18182,   -24,       -27.2727,  3.04839,    46.7727,  -6.3,
              0,        -3.04839, 25.7727,  -0,        -1.4026,   -0.435484, 3.27273,   -1.8,       -1.87013, -13.0645,
              24.5455,  0,        -7.48052, -2.03226,  2.45455,   -24.3,     -26.1818,  -5.22581,   44.1818,  -54,
              5.61039,  0,        3.68182,  5.4,       -4.67532,  -0,        -0,        -18,        4.98701,  -0.870968,
              17.4545,  3,        -37.4026, -21.6774,  -88.3636,  192,       29.9221,   -6.09677,   40.9091,  -24.3,
              -28.0519, 43.5484,  24.5455,  216,       -0,        5.41935,   -45.8182,  72,         -42.0779, 0.870968,
              4.09091,  -24.3,    21.039,   -4.06452,  7.36364,   18,        -0,        0,          0,        0,
              19.6364,  9.14516,  6.68182,  73.5,      12.4675,   0,         16.3636,   14.4,       -14.026,  -12.0968,
              -34.0909, 42,       -0,       -4.64516,  -57.2727,  1.8,       -0,        0.290323,   2.72727,  -1.5,
              19.6364,  5.22581,  16.3636,  8.1,       7.48052,   2.32258,   32.7273,   -54,        3.74026,  -12.1935,
              -19.0909, 58.8,     7.01299,  0.0967742, 4.77273,   -15,       0,         -26.129,    68.7273,  -121.5,
              19.4805,  7.25806,  -43.6364, -18,       -28.0519,  -17.4194,  -81.8182,  96,         -62.3377, 4.64516,
              -1.09091, -86.4,    26.1818,  20.3226,   -51.5455,  -25.2,     -0,        0,          0,        -0,
              -0,       0,        0,        -0,        -0,        4.64516,   2.72727,   7.2,        -2.18182, -0.290323,
              -2.04545, 15,       -9.81818, 1.35484,   -45.8182,  84,        0,         15.6774,    -36.8182, -36,
              -2.33766, -1.16129, 1.36364,  -9,        6.54545,   2.17742,   -0,        -0.9,       -0,       -0,
              0,        0,        13.0909,  20.3226,   4.77273,   -25.2,     -0,        -1.16129,   36.8182,  -72,
              19.9481,  10.4516,  32.7273,  100.8,     11.2208,   5.08065,   -2.45455,  -81,        -56.1039, 15.6774,
              -51.5455, 18.9,     0,        -3.3871,   -8.72727,  54,        -0,        -0,         0,        -0,
              -0,       -2.32258, -4.09091, 48.6,      112.208,   -23.5161,  -7.36364,  -216,       -1.24675, 15.4839,
              26.1818,  0,        -7.01299, -5.22581,  -17.1818,  -0,        0.935065,  -1.30645,   3.27273,  0.9,
              -37.4026, 2.90323,  -5.72727, -100.8,    43.6364,   14.2258,   28.6364,   -100.8,     -5.61039, 1.54839,
              -38.1818, -42,      6.23377,  1.93548,   1.63636,   37.8,      18.7013,   0.725806,   -0,       -54,
              4.67532,  1.74194,  7.36364,  0,         -23.3766,  -16.9355,  -21.8182,  12,         -26.1818, -12.3871,
              -34.9091, 192,      4.36364,  -30.4839,  -3.81818,  -132.3,    16.8312,   8.70968,    9.81818,  4.8,
              -19.6364, -1.30645, -19.6364, 31.5,      98.1818,   -4.35484,  -14.7273,  -16.2,      -0,       2.70968,
              -2.18182, 2.4});
  auto expected_dst_arr = gko::array<double>(
    exec, I<double>{
            1.81818,   8.12903,  3.56364,   -3.36,    0,         -2.70968,  -1.52727,  -8.96,     -0.467532, -0.129032,
            -0.218182, 0.48,     0.155844,  -3.87097, -1.30909,  4.8,       -0.311688, -0.774194, -0.654545, 1.44,
            -2.18182,  -6.96774, -3.92727,  5.76,     -0.311688, 0,         -0.654545, -0.48,     0.779221,  -4.51613,
            -2.90909,  3.2,      0.415584,  0.774194, -1.16364,  0.32,      -2.07792,  -7.22581,  -5.23636,  10.24,
            -1.24675,  -2.32258, 2.18182,   4.32,     2.33766,   11.6129,   1.30909,   -14.4,     0.831169,  -2.06452,
            -4.07273,  -7.68,    1.4026,    1.16129,  -2.18182,  -4.32,     -1.4026,   1.54839,   1.96364,   4.8,
            0,         0,        0,         0,        1.09091,   -2.70968,  -3.56364,  7.84,      2.07792,   0.516129,
            -2.90909,  -2.56,    -0.519481, -6.45161, 3.63636,   11.2,      0,         -1.54839,  -3.05455,  -0.96,
            0.467532,  0.774194, -0.727273, -0.8,     -1.09091,  2.32258,   0.872727,  -4.32,     1.24675,   -1.54839,
            1.74545,   -4.8,     -0.415584, -3.6129,  -2.03636,  4.48,      0.467532,  -0.129032, 0.509091,  -1.6,
            -1.4026,   -11.6129, 5.23636,   12.96,    1.2987,    6.45161,   2.90909,   -4.8,      -4.67532,  -5.16129,
            4.36364,   -6.4,     -2.07792,  2.06452,  0.290909,  5.76,      -2.18182,  -9.03226,  3.05455,   -3.36,
            0,         0,        0,         0,        0,         0,         0,         0,         0.207792,  -4.12903,
            0.290909,  0.64,     -0.103896, 0.774194, -0.218182, 0.8,       0.363636,  3.6129,    4.07273,   8.96,
            0,         4.64516,  -3.92727,  -4.8,     0.25974,   0.516129,  0.145455,  -1.6,      -0.311688, 0.645161,
            0,         0.16,     0,         0,        0,         0,         -1.45455,  9.03226,   -0.509091, -4.48,
            0,         -1.54839, -3.92727,  9.6,      0.831169,  -3.09677,  -2.90909,  7.68,      0.623377,  -2.70968,
            -0.654545, -4.8,     1.87013,   -6.96774, -3.92727,  10.08,     -0.935065, 1.29032,   -0.581818, 3.2,
            0,         0,        0,         0,        0,         -1.54839,  0.436364,  2.88,      -4.67532,  -10.4516,
            -0.654545, -11.52,   -0.415584, 5.16129,  2.32727,   0,         0.467532,  3.48387,   1.52727,   -2.88,
            0.0519481, -1.16129, 0.290909,  0.48,     1.55844,   3.87097,   3.05455,   -7.68,     1.45455,   6.32258,
            -3.05455,  -6.72,    0.207792,  -1.03226, -2.90909,  4.48,      -1.03896,  -0.516129, 0.290909,  2.88,
            -2.07792,  -1.93548, -1.81818,  -7.2,     -0.311688, -2.32258,  -1.30909,  0,         2.5974,    -6.45161,
            1.45455,   -3.2,     2.90909,   4.12903,  -2.32727,  -12.8,     0.727273,  9.03226,   -0.509091, 10.08,
            0.623377,  -2.58065, -2.61818,  -0.64,    -0.935065, -0.387097, -1.30909,  2.4,       4.67532,   11.6129,
            -1.96364,  -4.32,    0.519481,  1.03226,  -0.290909, 1.28});
  using VectorizedNumber = double4;

  gko::batch::matrix::dense::batch_item<const VectorizedNumber> jac{
    reinterpret_cast<const VectorizedNumber*>(jac_arr.get_const_data()), dim, dim, dim};
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> speed_cells{
    reinterpret_cast<const VectorizedNumber*>(speed_cells_arr.get_const_data()), dim, cell_size, dim};
  gko::batch::multi_vector::batch_item<const VectorizedNumber> src{
    reinterpret_cast<const VectorizedNumber*>(src_arr.get_const_data()), 1, cell_size, 1};
  gko::batch::multi_vector::batch_item<VectorizedNumber> dst{reinterpret_cast<VectorizedNumber*>(dst_arr.get_data()), 1,
                                                             cell_size, 1};
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? ::fe_degree + 1 : 1, dim >= 2 ? ::fe_degree + 1 : 1,
                                      dim >= 3 ? ::fe_degree + 1 : 1};
  std::array gradients = {
    tensor3d<VectorizedNumber>{reinterpret_cast<VectorizedNumber*>(gradients_arr.get_data()), tensor_size},
    tensor3d<VectorizedNumber>{reinterpret_cast<VectorizedNumber*>(gradients_arr.get_data()) + cell_size, tensor_size}};
  double time_factor = 1.5;
  double inv_dt = 4;

  dst_arr.fill(0.0);
  diagonal_scaling_kernel<::fe_degree, dim>
    <<<1, cell_size>>>(jac, speed_cells, gradients, weights_arr.get_data(), time_factor, inv_dt, src, dst);
  exec->synchronize();

  for (int i = 0; i < dst_arr.get_size(); ++i) {
    auto result = dst_arr.get_data()[i];
    auto expected = expected_dst_arr.get_data()[i];
    auto eps = (std::abs(result) + std::abs(expected)) * 1e-5;
    EXPECT_NEAR(result, expected, eps) << i / 4 << " " << i % 4;
  }
  for (int i = 0; i < gradients_arr.get_size(); ++i) {
    auto result = gradients_arr.get_data()[i];
    auto expected = expected_gradients_arr.get_data()[i];
    auto eps = (std::abs(result) + std::abs(expected)) * 1e-5;
    EXPECT_NEAR(result, expected, eps) << i / 4 << " " << i % 4;
  }
}

TEST(Operator2d, SimpleTensorApplyKernel) {
  using namespace DGAdvection;
  auto exec =
    gko::CudaExecutor::create(0, gko::ReferenceExecutor::create(), std::make_shared<gko::CudaUnifiedAllocator>(0));
  constexpr int cell_size = dealii::Utilities::pow(::fe_degree + 1, dim);
  auto shape_gradients_arr = gko::array<double>(
    exec, I<double>{1, -8, 8,   -1, -1, -1, -6, -10, -10, 10, 9,  6,  1,  6,  3,  -3, -3, -5, -3, 3, -3, -9,
                    1, -7, -1,  -4, -6, -9, 10, 1,   -9,  10, -1, 4,  -7, 1,  10, -7, 4,  -2, -1, 5, -7, 7,
                    6, -8, -10, -8, 3,  -3, -5, 0,   0,   -8, 10, -3, 6,  -5, 3,  9,  -1, -1, 9,  -1});
  auto src_arr = gko::array<double>(
    exec,
    I<double>{5,   9,  -7,  3,   0,   3,  -3,  -8,  -9, -1,  3,   -3,  -1, 10,  -6, 10,  -6,  -6,  9,   -9, 7,   9,
              -9,  6,  6,   0,   -9,  -3, 3,   -7,  8,  -4,  4,   3,   8,  -1,  5,  7,   -9,  8,   -8,  -6, -10, -9,
              5,   10, -2,  10,  -2,  2,  -7,  -6,  -9, -3,  -10, -9,  9,  -4,  9,  10,  -8,  -7,  2,   -8, 3,   -3,
              7,   -7, -10, -1,  -10, -4, -1,  -5,  -5, -7,  0,   2,   -7, -1,  -9, -6,  -10, -5,  -7,  6,  -4,  9,
              4,   -2, -4,  5,   -2,  -7, 7,   -7,  -9, 1,   7,   -10, -3, -10, -8, -9,  -5,  -10, 8,   -6, -9,  -4,
              -6,  4,  -10, 4,   -1,  -9, -6,  -10, -6, 3,   -10, -6,  7,  1,   5,  10,  0,   -2,  1,   -8, -1,  -1,
              2,   -6, -3,  5,   1,   4,  -8,  -8,  0,  6,   9,   5,   -5, -4,  2,  -10, -6,  5,   0,   -1, -3,  4,
              -10, -8, 4,   -10, -1,  -4, 0,   -2,  9,  -10, -2,  3,   -5, 6,   4,  -7,  3,   10,  4,   -6, 6,   -7,
              -9,  5,  4,   -10, 4,   -1, 8,   2,   0,  2,   1,   3,   10, 9,   -1, -8,  -2,  10,  -8,  0,  -3,  -9,
              7,   -6, 1,   -9,  -4,  -3, -5,  -5,  7,  -8,  -4,  -7,  -6, -6,  -1, 2,   -10, 7,   -10, -2, -2,  -9,
              -8,  -3, 5,   9,   2,   6,  -6,  0,   -5, 5,   2,   -2,  7,  4,   4,  10,  -2,  -10, -1,  9,  3,   -5,
              9,   1,  -6,  -1,  6,   -5, -10, -10, -3, -3,  -10, -8,  -4, 8});
  auto buffer_arr = gko::array<double>(exec, cell_size * 4 * 3);
  auto dst_arr = gko::array<double>(exec, cell_size * 4);
  auto expected_dst_arr = gko::array<double>(
    exec,
    I<double>{
      3444,  4906,  -3195, 1766,  -1017, 1645,  -304,  -6276,  -1667, -480,  684,   -642,  -565,  6416,  -4905, 8205,
      -1926, -2012, 2156,  -2535, 3103,  6399,  -3987, 3305,   2636,  -3329, -807,  -2128, 1300,  -5748, 5085,  -5079,
      2362,  -834,  3917,  -200,  3895,  7106,  -5264, 6402,   -4587, -2611, -5176, -5385, 6180,  9725,  -638,  7499,
      789,   643,   -4951, -898,  -5709, -999,  -7210, -10196, 4595,  -4109, 3923,  2021,  -4501, -7401, 1944,  -5491,
      2947,  164,   2412,  -3060, -6586, -188,  -2218, -2102,  797,   -2757, -1497, -81,   1362,  2533,  -1703, 837,
      -5739, -3466, -4968, -1623, -3865, 4643,  -4403, 5338,   226,   -2364, -4068, -1797, 340,   -5817, 4850,  -2585,
      -6251, -3400, 5914,  -6094, 2515,  -7675, -5008, -6367,  -4101, -4362, 8025,  -3799, -6496, -2034, -3990, 2883,
      -7612, 1573,  -432,  -2446, -5169, -9654, -1057, 689,    -5959, 1293,  6066,  -1540, 8241,  10618, 1597,  -2838,
      -150,  -3907, 282,   -3392, 73,    -6827, -2353, 3984,   705,   -1804, -5361, -5206, -1846, 3616,  10248, 946,
      -6299, 1357,  -1526, -8261, -5394, 2582,  219,   -1510,  -3938, 843,   -5040, -7445, 2796,  -3444, -3734, -4383,
      1674,  -3742, 6590,  -5676, -3034, 1755,  -7514, 5180,   1707,  -3348, 1595,  1842,  713,   -3990, 5759,  -5927,
      -7058, 1319,  2579,  -7348, 249,   176,   6509,  2707,   1020,  3298,  -1136, 407,   6839,  4618,  -572,  -5269,
      950,   3940,  -5650, 1569,  -1516, -3913, 4127,  -800,   1449,  -389,  -5795, 894,   -405,  -2239, 3913,  -5052,
      -3219, -6815, -3212, -3113, -1796, 1640,  -5855, 3024,   -6190, -2010, -6090, -4312, -4287, -3848, 767,   4682,
      3146,  5721,  -5952, 2538,  -2182, 6337,  3532,  -2999,  1962,  1902,  -473,  2693,  2291,  -5337, 1648,  2738,
      867,   -7083, 4490,  1642,  -1113, 2146,  4789,  -1315,  -2857, -4819, -2157, -930,  -7262, -5980, -1227, 4214});
  using VectorizedNumber = double4;
  gko::batch::matrix::dense::batch_item<const Number> shape_gradients(
    shape_gradients_arr.get_const_data(), ::fe_degree + 1, ::fe_degree + 1, ::fe_degree + 1);
  gko::batch::multi_vector::batch_item<const VectorizedNumber> src{
    reinterpret_cast<const VectorizedNumber*>(src_arr.get_const_data()), 1, cell_size, 1};
  gko::batch::multi_vector::batch_item<VectorizedNumber> dst{reinterpret_cast<VectorizedNumber*>(dst_arr.get_data()), 1,
                                                             cell_size, 1};

  dst_arr.fill(0.0);
  only_tensor_apply_kernel<::fe_degree, dim>
    <<<1, cell_size>>>(shape_gradients, src, dst, reinterpret_cast<VectorizedNumber*>(buffer_arr.get_data()));
  exec->synchronize();

  for (int i = 0; i < cell_size * 4; ++i) {
    auto result = dst_arr.get_data()[i];
    auto expected = expected_dst_arr.get_data()[i];
    auto eps = (std::abs(result) + std::abs(expected)) * 1e-5;
    EXPECT_NEAR(result, expected, eps) << i / 4 << " " << i % 4;
  }
}

TEST(Operator2d, SimpleCellKernel) {
  using namespace DGAdvection;
  auto exec =
    gko::CudaExecutor::create(0, gko::ReferenceExecutor::create(), std::make_shared<gko::CudaUnifiedAllocator>(0));
  constexpr int face_size = dealii::Utilities::pow(::fe_degree + 1, dim - 1);
  constexpr int cell_size = dealii::Utilities::pow(::fe_degree + 1, dim);
  auto jac_arr = gko::array<double>(exec, I<double>{-10, 1, 5, -5, -3, -6, 10, 0, 1, -5, -8, 7, 8, -1, -5, -5});
  auto shape_gradients_arr = gko::array<double>(
    exec, I<double>{1, -8, 8,   -1, -1, -1, -6, -10, -10, 10, 9,  6,  1,  6,  3,  -3, -3, -5, -3, 3, -3, -9,
                    1, -7, -1,  -4, -6, -9, 10, 1,   -9,  10, -1, 4,  -7, 1,  10, -7, 4,  -2, -1, 5, -7, 7,
                    6, -8, -10, -8, 3,  -3, -5, 0,   0,   -8, 10, -3, 6,  -5, 3,  9,  -1, -1, 9,  -1});
  auto speed_cells_arr = gko::array<double>(
    exec,
    I<double>{
      10,  -4, 6,   8,   5,  1,   7,  1,   -4, -3,  -4, -1, 1,   3,   -9,  0,  -1, -10, 4,  -2, -1,  9,   -8,  -2,  -4,
      -2,  -5, -3,  4,   9,  -10, 0,  7,   1,  -8,  -2, -8, 7,   -2,  -9,  2,  -3, -5,  -2, -4, 2,   -6,  -5,  10,  -6,
      3,   0,  6,   -8,  -3, -6,  -4, -7,  10, 5,   2,  0,  0,   -3,  -4,  -2, -2, -1,  -4, -3, -8,  5,   -2,  -10, -3,
      1,   -6, 8,   9,   10, 6,   -6, 0,   -6, 8,   7,  10, -3,  3,   4,   8,  -8, 4,   10, 10, -8,  -3,  -10, 1,   -7,
      0,   -7, 6,   -5,  -6, -3,  5,  -2,  10, 2,   -1, 3,  9,   0,   -1,  -9, 5,  -7,  2,  2,  -3,  -2,  -6,  7,   0,
      3,   0,  -9,  3,   4,  -7,  6,  -6,  -9, -1,  5,  10, 1,   -10, 3,   -2, 0,  -3,  -3, 7,  7,   9,   1,   -9,  5,
      -5,  2,  -10, -1,  3,  8,   -3, 8,   10, -1,  0,  8,  -10, -10, 0,   1,  -2, 1,   3,  2,  3,   -8,  6,   6,   10,
      -1,  -9, -1,  10,  10, -2,  -4, 10,  6,  -3,  -4, -9, 9,   3,   9,   5,  7,  -7,  -1, -9, -4,  -5,  -2,  5,   5,
      -5,  -1, 6,   3,   0,  6,   7,  -5,  -6, 3,   -6, -3, -5,  3,   -8,  2,  -3, -8,  1,  -2, -2,  9,   -10, -8,  -5,
      4,   7,  1,   -10, 6,  -2,  -8, -7,  -2, -10, -5, 4,  -6,  -9,  4,   5,  -8, 8,   1,  0,  6,   10,  -4,  5,   -4,
      1,   3,  -6,  -2,  2,  2,   -5, 4,   -3, 0,   0,  -3, 5,   6,   8,   -1, -1, -4,  -7, -1, 5,   10,  7,   3,   0,
      -10, 9,  1,   -6,  5,  -3,  -1, -7,  -8, -4,  9,  5,  4,   5,   -3,  -1, 2,  3,   -6, 5,  3,   -7,  -7,  7,   -3,
      7,   9,  8,   -3,  -4, -5,  1,  6,   6,  7,   -3, -9, -6,  -4,  4,   6,  3,  6,   -5, 3,  7,   -10, -2,  -3,  8,
      2,   -5, -4,  10,  5,  -2,  -1, -8,  -9, -6,  7,  5,  -4,  -10, -10, -6, -5, 2,   9,  -2, -1,  6,   -9,  10,  -6,
      7,   1,  -6,  -10, 1,  0,   0,  -7,  8,  9,   5,  6,  0,   -4,  -6,  -9, 10, -6,  9,  8,  1,   7,   -7,  4,   -5,
      9,   -7, 10,  -9,  9,  8,   6,  6,   10, -5,  -2, -7, -8,  -1,  8,   6,  -8, -6,  8,  -5, -3,  5,   -4,  -6,  0,
      -8,  -4, 10,  -1,  -6, 3,   6,  1,   4,  -4,  3,  -2, 8,   2,   -1,  7,  -2, 10,  -5, 4,  -10, 6,   -5,  8,   -3,
      -9,  -2, 5,   9,   -4, 7,   -5, 1,   6,  -2,  5,  2,  -10, 3,   7,   5,  -6, 3,   -6, 3,  -1,  0,   4,   10,  -6,
      -8,  5,  5,   -2,  -3, 0,   6,  5,   10, -6,  3,  7,  -8,  -2,  -3,  -5, -1, -6,  3,  -8, 8,   -8,  0,   4,   7,
      -7,  -2, -9,  4,   -7, -9,  -5, -6,  8,  -9,  -9, -2, -4,  6,   -2,  3,  -8, -7,  9,  8,  7,   -1,  3,   -2,  5,
      -7,  -1, 4,   2,   -5, -2,  4,  -10, 0,  7,   4,  1

    });
  auto weights_arr = gko::array<double>(
    exec, I<double>{-7, 7,  -1, 3,  -1, 6,   1,  -5, -2, 8, -3, -9, 8,  3,  3,  0,  -7, 4,  -10, 6, 1,  -3,
                    -6, -4, 1,  -9, 5,  -10, -4, -7, 0,  0, -4, 1,  -7, -6, 1,  -1, 0,  7,  -6,  8, -3, -9,
                    -2, 0,  6,  9,  -4, 3,   -1, 6,  7,  4, -2, -5, 3,  10, -8, 7,  -4, -3, 9,   1

          });
  auto src_arr = gko::array<double>(
    exec,
    I<double>{5,   9,  -7,  3,   0,   3,  -3,  -8,  -9, -1,  3,   -3,  -1, 10,  -6, 10,  -6,  -6,  9,   -9, 7,   9,
              -9,  6,  6,   0,   -9,  -3, 3,   -7,  8,  -4,  4,   3,   8,  -1,  5,  7,   -9,  8,   -8,  -6, -10, -9,
              5,   10, -2,  10,  -2,  2,  -7,  -6,  -9, -3,  -10, -9,  9,  -4,  9,  10,  -8,  -7,  2,   -8, 3,   -3,
              7,   -7, -10, -1,  -10, -4, -1,  -5,  -5, -7,  0,   2,   -7, -1,  -9, -6,  -10, -5,  -7,  6,  -4,  9,
              4,   -2, -4,  5,   -2,  -7, 7,   -7,  -9, 1,   7,   -10, -3, -10, -8, -9,  -5,  -10, 8,   -6, -9,  -4,
              -6,  4,  -10, 4,   -1,  -9, -6,  -10, -6, 3,   -10, -6,  7,  1,   5,  10,  0,   -2,  1,   -8, -1,  -1,
              2,   -6, -3,  5,   1,   4,  -8,  -8,  0,  6,   9,   5,   -5, -4,  2,  -10, -6,  5,   0,   -1, -3,  4,
              -10, -8, 4,   -10, -1,  -4, 0,   -2,  9,  -10, -2,  3,   -5, 6,   4,  -7,  3,   10,  4,   -6, 6,   -7,
              -9,  5,  4,   -10, 4,   -1, 8,   2,   0,  2,   1,   3,   10, 9,   -1, -8,  -2,  10,  -8,  0,  -3,  -9,
              7,   -6, 1,   -9,  -4,  -3, -5,  -5,  7,  -8,  -4,  -7,  -6, -6,  -1, 2,   -10, 7,   -10, -2, -2,  -9,
              -8,  -3, 5,   9,   2,   6,  -6,  0,   -5, 5,   2,   -2,  7,  4,   4,  10,  -2,  -10, -1,  9,  3,   -5,
              9,   1,  -6,  -1,  6,   -5, -10, -10, -3, -3,  -10, -8,  -4, 8});
  auto buffer_arr = gko::array<double>(exec, cell_size * 4 * 3);
  auto dst_arr = gko::array<double>(exec, cell_size * 4);
  auto expected_dst_arr = gko::array<double>(
    exec, I<double>{341.558,  139.113,  -754.891, 188.94,   -129.39,  198.968,  928.064,  -1275.26, -93,      193.516,
                    -57.0818, -1389.12, 320.104,  -223.452, 175.282,  601.5,    -446.688, -158.806, 164.345,  127.74,
                    858.039,  118.839,  263.345,  32.46,    -877.519, 231.774,  421.118,  -794.28,  112.87,   -50.0968,
                    -394.818, 359,      467.364,  282.774,  275.245,  -149.98,  -314.506, -680.968, -606.055, 2926.24,
                    810.078,  49.4516,  -237.955, -755.58,  -713.221, 861.145,  -994.282, 2934.3,   8.97403,  -78.371,
                    -648.527, 1625.82,  1359.23,  -36.4839, -1055.73, 3226.98,  -201.74,  85.4516,  306.327,  -657.6,
                    474.935,  -263.516, -255.273, 71.4,     -516.078, 168.581,  -161.745, 462.64,   455.195,  167.887,
                    -199.273, -3422.86, -282.597, 115.242,  1109.41,  -2454.2,  669.74,   -363.871, 610.445,  1718.64,
                    71.8442,  -146.419, 355.045,  -2258,    1374.04,  -15.3871, 626.918,  -621.72,  -1289.65, 17.6129,
                    142.473,  -1677.6,  -1756.04, 240.597,  -156.536, 1400.08,  -720.429, -130,     399.373,  -734.8,
                    721.792,  -46.5,    -524.809, 1818.66,  -1046.91, 12.5484,  -763.864, 1949.7,   741.935,  22.4677,
                    678.682,  -3044.2,  681.961,  -82.6613, -900.664, 1449.96,  250.519,  -270.468, -615.082, 1154.34,
                    363.273,  42.4355,  -814.909, -776.7,   141.39,   143.661,  450.273,  -330.6,   -340.273, -129.694,
                    -141.118, -791.66,  -330.26,  -287.177, -1040.81, -1033.3,  -200.208, -45.3065, -183.7,   1757.06,
                    439.247,  470.952,  870.982,  130.8,    -86.5065, 70.4355,  -95.8545, 625.1,    -547.987, -58.871,
                    -700.773, 1659.46,  -358.948, -15.9677, -144,     1304.4,   -627.636, 483.081,  -219.373, 872.42,
                    -1451.22, -710.71,  -531.245, 2699.1,   -203.403, -223.887, 192.909,  85.98,    -717.584, -109.113,
                    466.391,  -85.2,    1824.16,  746.903,  -156.109, 2266.98,  -1021.09, -87.4516, -414.855, 13.1,
                    -765.662, -162.677, 498.273,  -799.5,   -1320.7,  -89.7097, 13.5273,  220.68,   -1423.09, 254.226,
                    567.573,  714.18,   260.468,  261.613,  -177.673, -948.6,   -85.8701, -177.823, -542.018, -1126.98,
                    400.961,  257.177,  130.109,  -819.72,  -490.909, -446.129, 550.691,  -1640.58, 318.909,  -101.29,
                    276.491,  -2494.62, 370.571,  -120.5,   -505.682, -1907.12, -157.584, 155.29,   62.3364,  -80.22,
                    -423.247, 410.806,  138.909,  1171.5,   -579.117, 232.016,  645.191,  1622.4,   405.61,   -78.2581,
                    1012.59,  -1554.2,  155.481,  -54.8065, -341.736, -653.6,   -604.338, -323.581, -947.827, -783.12,
                    133.714,  10.3871,  62.7,     -3199.84, 1137.82,  189.968,  -714.355, -1737.9,  -111.039, -158.323,
                    20.6727,  -789.12,  -539.792, -332.306, -33.5636, -1411.12});
  using VectorizedNumber = double4;
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> jac{
    reinterpret_cast<const VectorizedNumber*>(jac_arr.get_const_data()), dim, dim, dim};
  gko::batch::matrix::dense::batch_item<const Number> shape_gradients(
    shape_gradients_arr.get_const_data(), ::fe_degree + 1, ::fe_degree + 1, ::fe_degree + 1);
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> speed_cells{
    reinterpret_cast<const VectorizedNumber*>(speed_cells_arr.get_const_data()), dim, cell_size, dim};
  gko::batch::multi_vector::batch_item<const VectorizedNumber> src{
    reinterpret_cast<const VectorizedNumber*>(src_arr.get_const_data()), 1, cell_size, 1};
  gko::batch::multi_vector::batch_item<VectorizedNumber> dst{reinterpret_cast<VectorizedNumber*>(dst_arr.get_data()), 1,
                                                             cell_size, 1};
  double time_factor = 1.5;
  double inv_dt = 4;

  dst_arr.fill(0.0);
  simple_cell_kernel<::fe_degree, dim><<<1, cell_size>>>(jac, shape_gradients, speed_cells,
                                                         weights_arr.get_const_data(), time_factor, inv_dt, src, dst,
                                                         reinterpret_cast<VectorizedNumber*>(buffer_arr.get_data()));
  exec->synchronize();

  for (int i = 0; i < cell_size * 4; ++i) {
    auto result = dst_arr.get_data()[i];
    auto expected = expected_dst_arr.get_data()[i];
    auto eps = (std::abs(result) + std::abs(expected)) * 1e-5;
    EXPECT_NEAR(result, expected, eps) << i / 4 << " " << i % 4;
  }
}

TEST(Operator2d, SimpleReducedSharedStorageCellKernel) {
  using namespace DGAdvection;
  auto exec =
    gko::CudaExecutor::create(0, gko::ReferenceExecutor::create(), std::make_shared<gko::CudaUnifiedAllocator>(0));
  constexpr int face_size = dealii::Utilities::pow(::fe_degree + 1, dim - 1);
  constexpr int cell_size = dealii::Utilities::pow(::fe_degree + 1, dim);
  auto jac_arr = gko::array<double>(exec, I<double>{-10, 1, 5, -5, -3, -6, 10, 0, 1, -5, -8, 7, 8, -1, -5, -5});
  auto shape_gradients_arr = gko::array<double>(
    exec, I<double>{1, -8, 8,   -1, -1, -1, -6, -10, -10, 10, 9,  6,  1,  6,  3,  -3, -3, -5, -3, 3, -3, -9,
                    1, -7, -1,  -4, -6, -9, 10, 1,   -9,  10, -1, 4,  -7, 1,  10, -7, 4,  -2, -1, 5, -7, 7,
                    6, -8, -10, -8, 3,  -3, -5, 0,   0,   -8, 10, -3, 6,  -5, 3,  9,  -1, -1, 9,  -1});
  auto speed_cells_arr = gko::array<double>(
    exec,
    I<double>{
      10,  -4, 6,   8,   5,  1,   7,  1,   -4, -3,  -4, -1, 1,   3,   -9,  0,  -1, -10, 4,  -2, -1,  9,   -8,  -2,  -4,
      -2,  -5, -3,  4,   9,  -10, 0,  7,   1,  -8,  -2, -8, 7,   -2,  -9,  2,  -3, -5,  -2, -4, 2,   -6,  -5,  10,  -6,
      3,   0,  6,   -8,  -3, -6,  -4, -7,  10, 5,   2,  0,  0,   -3,  -4,  -2, -2, -1,  -4, -3, -8,  5,   -2,  -10, -3,
      1,   -6, 8,   9,   10, 6,   -6, 0,   -6, 8,   7,  10, -3,  3,   4,   8,  -8, 4,   10, 10, -8,  -3,  -10, 1,   -7,
      0,   -7, 6,   -5,  -6, -3,  5,  -2,  10, 2,   -1, 3,  9,   0,   -1,  -9, 5,  -7,  2,  2,  -3,  -2,  -6,  7,   0,
      3,   0,  -9,  3,   4,  -7,  6,  -6,  -9, -1,  5,  10, 1,   -10, 3,   -2, 0,  -3,  -3, 7,  7,   9,   1,   -9,  5,
      -5,  2,  -10, -1,  3,  8,   -3, 8,   10, -1,  0,  8,  -10, -10, 0,   1,  -2, 1,   3,  2,  3,   -8,  6,   6,   10,
      -1,  -9, -1,  10,  10, -2,  -4, 10,  6,  -3,  -4, -9, 9,   3,   9,   5,  7,  -7,  -1, -9, -4,  -5,  -2,  5,   5,
      -5,  -1, 6,   3,   0,  6,   7,  -5,  -6, 3,   -6, -3, -5,  3,   -8,  2,  -3, -8,  1,  -2, -2,  9,   -10, -8,  -5,
      4,   7,  1,   -10, 6,  -2,  -8, -7,  -2, -10, -5, 4,  -6,  -9,  4,   5,  -8, 8,   1,  0,  6,   10,  -4,  5,   -4,
      1,   3,  -6,  -2,  2,  2,   -5, 4,   -3, 0,   0,  -3, 5,   6,   8,   -1, -1, -4,  -7, -1, 5,   10,  7,   3,   0,
      -10, 9,  1,   -6,  5,  -3,  -1, -7,  -8, -4,  9,  5,  4,   5,   -3,  -1, 2,  3,   -6, 5,  3,   -7,  -7,  7,   -3,
      7,   9,  8,   -3,  -4, -5,  1,  6,   6,  7,   -3, -9, -6,  -4,  4,   6,  3,  6,   -5, 3,  7,   -10, -2,  -3,  8,
      2,   -5, -4,  10,  5,  -2,  -1, -8,  -9, -6,  7,  5,  -4,  -10, -10, -6, -5, 2,   9,  -2, -1,  6,   -9,  10,  -6,
      7,   1,  -6,  -10, 1,  0,   0,  -7,  8,  9,   5,  6,  0,   -4,  -6,  -9, 10, -6,  9,  8,  1,   7,   -7,  4,   -5,
      9,   -7, 10,  -9,  9,  8,   6,  6,   10, -5,  -2, -7, -8,  -1,  8,   6,  -8, -6,  8,  -5, -3,  5,   -4,  -6,  0,
      -8,  -4, 10,  -1,  -6, 3,   6,  1,   4,  -4,  3,  -2, 8,   2,   -1,  7,  -2, 10,  -5, 4,  -10, 6,   -5,  8,   -3,
      -9,  -2, 5,   9,   -4, 7,   -5, 1,   6,  -2,  5,  2,  -10, 3,   7,   5,  -6, 3,   -6, 3,  -1,  0,   4,   10,  -6,
      -8,  5,  5,   -2,  -3, 0,   6,  5,   10, -6,  3,  7,  -8,  -2,  -3,  -5, -1, -6,  3,  -8, 8,   -8,  0,   4,   7,
      -7,  -2, -9,  4,   -7, -9,  -5, -6,  8,  -9,  -9, -2, -4,  6,   -2,  3,  -8, -7,  9,  8,  7,   -1,  3,   -2,  5,
      -7,  -1, 4,   2,   -5, -2,  4,  -10, 0,  7,   4,  1

    });
  auto weights_arr = gko::array<double>(
    exec, I<double>{-7, 7,  -1, 3,  -1, 6,   1,  -5, -2, 8, -3, -9, 8,  3,  3,  0,  -7, 4,  -10, 6, 1,  -3,
                    -6, -4, 1,  -9, 5,  -10, -4, -7, 0,  0, -4, 1,  -7, -6, 1,  -1, 0,  7,  -6,  8, -3, -9,
                    -2, 0,  6,  9,  -4, 3,   -1, 6,  7,  4, -2, -5, 3,  10, -8, 7,  -4, -3, 9,   1

          });
  auto src_arr = gko::array<double>(
    exec,
    I<double>{5,   9,  -7,  3,   0,   3,  -3,  -8,  -9, -1,  3,   -3,  -1, 10,  -6, 10,  -6,  -6,  9,   -9, 7,   9,
              -9,  6,  6,   0,   -9,  -3, 3,   -7,  8,  -4,  4,   3,   8,  -1,  5,  7,   -9,  8,   -8,  -6, -10, -9,
              5,   10, -2,  10,  -2,  2,  -7,  -6,  -9, -3,  -10, -9,  9,  -4,  9,  10,  -8,  -7,  2,   -8, 3,   -3,
              7,   -7, -10, -1,  -10, -4, -1,  -5,  -5, -7,  0,   2,   -7, -1,  -9, -6,  -10, -5,  -7,  6,  -4,  9,
              4,   -2, -4,  5,   -2,  -7, 7,   -7,  -9, 1,   7,   -10, -3, -10, -8, -9,  -5,  -10, 8,   -6, -9,  -4,
              -6,  4,  -10, 4,   -1,  -9, -6,  -10, -6, 3,   -10, -6,  7,  1,   5,  10,  0,   -2,  1,   -8, -1,  -1,
              2,   -6, -3,  5,   1,   4,  -8,  -8,  0,  6,   9,   5,   -5, -4,  2,  -10, -6,  5,   0,   -1, -3,  4,
              -10, -8, 4,   -10, -1,  -4, 0,   -2,  9,  -10, -2,  3,   -5, 6,   4,  -7,  3,   10,  4,   -6, 6,   -7,
              -9,  5,  4,   -10, 4,   -1, 8,   2,   0,  2,   1,   3,   10, 9,   -1, -8,  -2,  10,  -8,  0,  -3,  -9,
              7,   -6, 1,   -9,  -4,  -3, -5,  -5,  7,  -8,  -4,  -7,  -6, -6,  -1, 2,   -10, 7,   -10, -2, -2,  -9,
              -8,  -3, 5,   9,   2,   6,  -6,  0,   -5, 5,   2,   -2,  7,  4,   4,  10,  -2,  -10, -1,  9,  3,   -5,
              9,   1,  -6,  -1,  6,   -5, -10, -10, -3, -3,  -10, -8,  -4, 8});
  auto buffer_arr = gko::array<double>(exec, cell_size * 4 * 3);
  auto dst_arr = gko::array<double>(exec, cell_size * 4);
  auto expected_dst_arr = gko::array<double>(
    exec, I<double>{341.558,  139.113,  -754.891, 188.94,   -129.39,  198.968,  928.064,  -1275.26, -93,      193.516,
                    -57.0818, -1389.12, 320.104,  -223.452, 175.282,  601.5,    -446.688, -158.806, 164.345,  127.74,
                    858.039,  118.839,  263.345,  32.46,    -877.519, 231.774,  421.118,  -794.28,  112.87,   -50.0968,
                    -394.818, 359,      467.364,  282.774,  275.245,  -149.98,  -314.506, -680.968, -606.055, 2926.24,
                    810.078,  49.4516,  -237.955, -755.58,  -713.221, 861.145,  -994.282, 2934.3,   8.97403,  -78.371,
                    -648.527, 1625.82,  1359.23,  -36.4839, -1055.73, 3226.98,  -201.74,  85.4516,  306.327,  -657.6,
                    474.935,  -263.516, -255.273, 71.4,     -516.078, 168.581,  -161.745, 462.64,   455.195,  167.887,
                    -199.273, -3422.86, -282.597, 115.242,  1109.41,  -2454.2,  669.74,   -363.871, 610.445,  1718.64,
                    71.8442,  -146.419, 355.045,  -2258,    1374.04,  -15.3871, 626.918,  -621.72,  -1289.65, 17.6129,
                    142.473,  -1677.6,  -1756.04, 240.597,  -156.536, 1400.08,  -720.429, -130,     399.373,  -734.8,
                    721.792,  -46.5,    -524.809, 1818.66,  -1046.91, 12.5484,  -763.864, 1949.7,   741.935,  22.4677,
                    678.682,  -3044.2,  681.961,  -82.6613, -900.664, 1449.96,  250.519,  -270.468, -615.082, 1154.34,
                    363.273,  42.4355,  -814.909, -776.7,   141.39,   143.661,  450.273,  -330.6,   -340.273, -129.694,
                    -141.118, -791.66,  -330.26,  -287.177, -1040.81, -1033.3,  -200.208, -45.3065, -183.7,   1757.06,
                    439.247,  470.952,  870.982,  130.8,    -86.5065, 70.4355,  -95.8545, 625.1,    -547.987, -58.871,
                    -700.773, 1659.46,  -358.948, -15.9677, -144,     1304.4,   -627.636, 483.081,  -219.373, 872.42,
                    -1451.22, -710.71,  -531.245, 2699.1,   -203.403, -223.887, 192.909,  85.98,    -717.584, -109.113,
                    466.391,  -85.2,    1824.16,  746.903,  -156.109, 2266.98,  -1021.09, -87.4516, -414.855, 13.1,
                    -765.662, -162.677, 498.273,  -799.5,   -1320.7,  -89.7097, 13.5273,  220.68,   -1423.09, 254.226,
                    567.573,  714.18,   260.468,  261.613,  -177.673, -948.6,   -85.8701, -177.823, -542.018, -1126.98,
                    400.961,  257.177,  130.109,  -819.72,  -490.909, -446.129, 550.691,  -1640.58, 318.909,  -101.29,
                    276.491,  -2494.62, 370.571,  -120.5,   -505.682, -1907.12, -157.584, 155.29,   62.3364,  -80.22,
                    -423.247, 410.806,  138.909,  1171.5,   -579.117, 232.016,  645.191,  1622.4,   405.61,   -78.2581,
                    1012.59,  -1554.2,  155.481,  -54.8065, -341.736, -653.6,   -604.338, -323.581, -947.827, -783.12,
                    133.714,  10.3871,  62.7,     -3199.84, 1137.82,  189.968,  -714.355, -1737.9,  -111.039, -158.323,
                    20.6727,  -789.12,  -539.792, -332.306, -33.5636, -1411.12});
  using VectorizedNumber = double4;
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> jac{
    reinterpret_cast<const VectorizedNumber*>(jac_arr.get_const_data()), dim, dim, dim};
  gko::batch::matrix::dense::batch_item<const Number> shape_gradients(
    shape_gradients_arr.get_const_data(), ::fe_degree + 1, ::fe_degree + 1, ::fe_degree + 1);
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> speed_cells{
    reinterpret_cast<const VectorizedNumber*>(speed_cells_arr.get_const_data()), dim, cell_size, dim};
  gko::batch::multi_vector::batch_item<const VectorizedNumber> src{
    reinterpret_cast<const VectorizedNumber*>(src_arr.get_const_data()), 1, cell_size, 1};
  gko::batch::multi_vector::batch_item<VectorizedNumber> dst{reinterpret_cast<VectorizedNumber*>(dst_arr.get_data()), 1,
                                                             cell_size, 1};
  double time_factor = 1.5;
  double inv_dt = 4;

  dst_arr.fill(0.0);
  cell_apply_single_gradient_kernel<::fe_degree, dim>
    <<<1, cell_size>>>(jac, shape_gradients, speed_cells, weights_arr.get_const_data(), time_factor, inv_dt, src, dst);
  exec->synchronize();

  for (int i = 0; i < cell_size * 4; ++i) {
    auto result = dst_arr.get_data()[i];
    auto expected = expected_dst_arr.get_data()[i];
    auto eps = (std::abs(result) + std::abs(expected)) * 1e-5;
    EXPECT_NEAR(result, expected, eps) << i / 4 << " " << i % 4;
  }
}
} // namespace test_dim2

namespace test_dim3 {
constexpr int dim = 3;

TEST(Operator3d, FaceKernel) {
  using namespace DGAdvection;
  auto exec =
    gko::CudaExecutor::create(0, gko::ReferenceExecutor::create(), std::make_shared<gko::CudaUnifiedAllocator>(0));
  constexpr int face_size = dealii::Utilities::pow(::fe_degree + 1, dim - 1);
  constexpr int cell_size = dealii::Utilities::pow(::fe_degree + 1, dim);
  auto jac_arr =
    gko::array<double>(exec, I<double>{-10, 1, 5, -5, -3, -6, 10, 0,  1, -5, -8, 7,  8,  -1,  -5,  -5, -7, -4,
                                       -8,  6, 1, 7,  5,  -9, 1,  -8, 8, -1, -1, -1, -6, -10, -10, 10, 9,  6});
  auto shape_arr = gko::array<double>(exec, I<double>{1,  6,  3,   -3, -3, -5, -3, 3, -3, -9, 1,  -7, -1, -4, -6, -9,
                                                      10, 1,  -9,  10, -1, 4,  -7, 1, 10, -7, 4,  -2, -1, 5,  -7, 7,
                                                      6,  -8, -10, -8, 3,  -3, -5, 0, 0,  -8, 10, -3, 6,  -5, 3,  9,
                                                      -1, -1, 9,   -1, 10, -4, 6,  8, 5,  1,  7,  1,  -4, -3, -4, -1});
  auto normal_speed_faces_arr = gko::array<double>(
    exec,
    I<double>{
      -10, -6,  6,   -5,  -4,  9,   0,   -8,  7,   -6,  -6,  -3,  -2,  -2,  -3,  -9,  2,   0,   2,   -5,  10,  -9,  7,
      -7,  -8,  -7,  -2,  -6,  3,   1,   -9,  6,   3,   -3,  2,   -2,  8,   5,   -9,  1,   2,   -8,  8,   2,   10,  0,
      9,   10,  2,   0,   5,   -1,  0,   7,   -9,  5,   10,  4,   4,   1,   -10, 5,   -6,  8,   0,   -10, 9,   -9,  4,
      -10, -6,  -8,  -9,  -1,  8,   -2,  6,   -4,  2,   6,   -4,  9,   7,   6,   -3,  -7,  2,   -8,  -7,  5,   -9,  -10,
      10,  -8,  7,   9,   1,   -2,  3,   9,   1,   1,   5,   2,   6,   -10, -5,  -4,  -9,  -1,  -4,  -10, 1,   -7,  -7,
      6,   -10, -2,  10,  -6,  -2,  8,   -7,  -10, -3,  10,  7,   3,   -3,  -10, -9,  6,   -3,  2,   -3,  0,   1,   -1,
      -10, -3,  7,   4,   5,   -9,  -1,  9,   6,   9,   1,   -8,  -9,  -2,  3,   -4,  2,   -7,  -2,  6,   -9,  -1,  7,
      8,   0,   8,   -9,  0,   5,   9,   2,   6,   3,   4,   4,   -5,  9,   10,  -2,  1,   6,   -6,  -2,  5,   8,   -6,
      3,   8,   -3,  -8,  -9,  5,   -4,  -4,  5,   5,   -3,  -7,  -1,  1,   10,  -4,  -4,  -3,  4,   -8,  6,   4,   1,
      -3,  -8,  -10, -10, 5,   -6,  -7,  2,   -10, 5,   2,   -9,  -4,  8,   -1,  5,   6,   5,   -5,  4,   -6,  -5,  -3,
      -9,  -10, -4,  -2,  -5,  -6,  10,  -9,  -1,  3,   0,   -5,  1,   4,   1,   -2,  -6,  -5,  -10, 2,   -8,  4,   -7,
      1,   -2,  3,   9,   -6,  2,   -10, 1,   4,   5,   10,  -3,  -3,  -9,  3,   -9,  7,   -7,  7,   10,  -6,  -1,  7,
      4,   -1,  -7,  -10, 8,   -5,  5,   6,   0,   -10, 10,  -9,  3,   -4,  3,   8,   8,   8,   0,   -6,  8,   -2,  6,
      -1,  3,   -2,  -3,  -2,  -8,  7,   3,   3,   1,   0,   7,   -5,  8,   -8,  7,   3,   -6,  -7,  -1,  4,   3,   3,
      7,   6,   -4,  8,   -7,  -3,  -6,  -6,  4,   -3,  9,   7,   -3,  2,   -2,  3,   -8,  8,   6,   1,   0,   -6,  9,
      -6,  10,  5,   8,   -1,  10,  3,   7,   -4,  -7,  -3,  10,  -9,  -5,  -9,  10,  -10, 5,   9,   -7,  -8,  3,   -7,
      4,   4,   -3,  8,   2,   -4,  3,   5,   2,   10,  -9,  2,   0,   2,   6,   2,   7,   -1,  3,   -2,  -10, 7,   0,
      5,   10,  -8,  -5,  -4,  1,   1,   -9,  0,   -1,  9,   10,  4,   5,   4,   8,   10,  -6,  9,   9,   10,  -7,  -1,
      -9,  -2,  1,   -8,  -8,  -8,  -8,  -7,  -9,  -5,  3,   3,   7,   0,   -1,  9,   2,   3,   -2,  -5,  6,   -7,  -7,
      5,   -1,  7,   -10, -10, -1,  9,   -1,  10,  -7,  -8,  2,   2,   -10, 4,   6,   2,   10,  -2,  -5,  -6,  -2,  -8,
      -10, 8,   10,  10,  -9,  -3,  0,   0,   -2,  -1,  10,  4,   3,   -3,  7,   5,   -10, -1,  -4,  4,   -2,  -1,  -3,
      7,   -3,  -6,  -9,  -3,  10,  5,   -10, 5,   2,   10,  -7,  -8,  -6,  -5,  0,   1,   -9,  -9,  8,   7,   -5,  -8,
      -8,  9,   6,   0,   -10, -8,  -2,  -8,  -2,  -4,  -3,  6,   7,   6,   10,  4,   -5,  -6,  9,   8,   -1,  3,   -1,
      3,   9,   5,   -6,  10,  -6,  -2,  -1,  -9,  -3,  5,   2,   -1,  6,   0,   10,  -1,  9,   0,   10,  -5,  -2,  9,
      -7,  -3,  10,  0,   6,   1,   6,   6,   -5,  -5,  -9,  -5,  6,   -2,  3,   -10, 5,   -5,  6,   2,   -5,  -3,  1,
      -5,  -3,  8,   -2,  8,   -9,  -4,  7,   9,   -4,  9,   -10, 5,   9,   -3,  -1,  -7,  5,   -5,  -4,  2,   8,   8,
      -7,  0,   4,   5,   8,   2,   1,   4,   1,   -5,  -3,  7,   -2,  10,  8,   6,   10,  -5,  -10, -8,  -1,  2,   -8,
      6,   -4,  -6,  6,   -8,  -8,  1,   5,   2,   -7,  8,   1,   -6,  -9,  -4,  -1,  -6,  -4,  3,   5,   2,   -4,  6,
      -7,  7,   -1,  -9,  -5,  4,   -10, 1,   -1,  3,   -6,  0,   -1,  -10, -3,  -10, 9,   10,  8,   -5,  -2,  10,  0,
      -7,  -10, -9,  -4,  7,   2,   -5,  -3,  5,   7,   -10, -2,  4,   -6,  0,   10,  -1,  -10, 10,  2,   -7,  -1,  -9,
      -10, 8,   -2,  -10, -2,  3,   10,  1,   -7,  -6,  2,   -8,  -9,  -7,  0,   -4,  -5,  -10, 6,   10,  -7,  1,   -8,
      10,  -4,  -10, 8,   8,   -3,  0,   -5,  6,   4,   -1,  8,   3,   -3,  9,   -7,  -1,  -2,  -10, -8,  8,   1,   -10,
      3,   -6,  5,   -10, -5,  -1,  -7,  -10, 3,   -3,  6,   -5,  10,  -7,  2,   10,  -10, -2,  -7,  -10, -1,  -8,  -8,
      1,   8,   -5,  5,   4,   -6,  5,   -2,  3,   3,   -1,  -7,  2,   -10, 2,   -6,  5,   -4,  9,   -3,  5,   -7,  1,
      1,   9,   -5,  0,   -9,  8,   -10, -9,  9,   -10, -6,  0,   -10, 6,   7,   3,   7,   -3,  -5,  -4,  3,   -9,  -3,
      2,   -3,  9,   8,   1,   -1,  8,   -1,  -6,  4,   -10, 7,   1,   -8,  7,   5,   -2,  3,   -1,  5,   -9,  -2,  -8,
      3,   1,   -5,  -1,  -8,  8,   -3,  7,   -10, 9,   10,  0,   4,   2,   9,   9,   -2,  -8,  -1,  -9,  4,   -5,  -6,
      -3,  9,   2,   7,   -7,  0,   -6,  1,   -8,  -3,  -4,  6,   -7,  6,   -2,  -8,  -9,  7,   -1,  -4,  6,   6,   0,
      1,   -9,  -8,  5,   10,  -8,  -2,  6,   1,   2,   8,   -6,  2,   -8,  -10, 10,  0,   7,   9,   2,   -8,  7,   2,
      3,   1,   -10, 7,   2,   -7,  -6,  4,   -7,  6,   0,   3,   -6,  8,   -8,  1,   5,   3,   -3,  -7,  -5,  4,   -6,
      -6,  -6,  -3,  8,   10,  7,   0,   -6,  3,   10,  -7,  5,   7,   -2,  4,   -1,  10,  -9,  -2,  3,   8,   -1,  -3,
      8,   7,   0,   8,   0,   0,   2,   1,   9,   -8,  -10, 1,   -10, -7,  10,  0,   -10, 5,   5,   -7,  6,   -6,  -7,
      0,   9,   10,  -1,  7,   2,   -6,  8,   -7,  -3,  -1,  -4,  -2,  -5,  -4,  1,   -8,  6,   0,   -2,  -7,  4,   -2,
      -9,  8,   -5,  8,   -5,  2,   -3,  4,   7,   -1,  -4,  10,  7,   -10, 0,   5,   4,   6,   -4,  3,   -2,  -5,  -6,
      2,   10,  6,   7,   -3,  -1,  10,  2,   4,   8,   10,  10,  -8,  2,   1,   -1,  0,   5,   8,   -2,  -5,  -1,  2,
      -7,  -9,  4,   -1,  -10, -8,  -7,  -7,  8,   3,   -2,  -10, 9,   9,   1,   8,   0,   2,   -10, -10, 1,   6,   8,
      3,   3,   8,   -5,  8,   10,  9,   1,   -6,  -10, 8,   -8,  -5,  -7,  7,   7,   -2,  -3,  6,   -8,  3,   -1,  10,
      -5,  -5,  1,   10,  -6,  1,   -7,  3,   -8,  -8,  2,   -6,  7,   -10, -10, -2,  -4,  -3,  10,  8,   3,   -2,  -9,
      10,  -9,  5,   10,  6,   -4,  -3,  9,   0,   10,  2,   -6,  3,   5,   8,   -3,  -10, 6,   -10, -9,  1,   2,   1,
      -7,  10,  0,   10,  4,   10,  7,   -6,  2,   4,   1,   7,   3,   -6,  7,   6,   -2,  -8,  -6,  9,   5,   -10, 0,
      8,   -8,  -1,  2,   -1,  3,   2,   10,  9,   7,   -4,  0,   0,   6,   -8,  -5,  10,  0,   -5,  -3,  4,   8,   0,
      3,   7,   4,   -2,  9,   4,   4,   -3,  2,   8,   -9,  4,   4,   6,   0,   -9,  3,   -6,  6,   -7,  -7,  2,   4,
      -4,  -2,  10,  6,   9,   -2,  -1,  6,   -10, 6,   -2,  -9,  -5,  -2,  -6,  5,   -8,  -5,  -3,  -3,  10,  8,   4,
      10,  3,   0,   -5,  7,   -2,  -10, -10, 9,   -7,  7,   7,   -1,  -7,  -8,  -7,  4,   -7,  2,   -2,  1,   2,   2,
      10,  -10, 8,   3,   2,   -10, 6,   -4,  -10, -8,  -9,  -3,  -1,  2,   9,   -3,  -2,  -6,  10,  -4,  3,   4,   -5,
      -10, 6,   -6,  1,   10,  -2,  -3,  6,   8,   7,   -7,  4,   -6,  1,   4,   8,   -6,  -6,  -10, 0,   8,   -8,  -5,
      -3,  1,   -3,  -10, 1,   -6,  2,   5,   6,   -9,  5,   2,   7,   5,   -1,  -7,  9,   7,   -9,  -9,  0,   -1,  -8,
      8,   -3,  7,   -6,  -8,  -4,  9,   8,   1,   4,   8,   -7,  10,  7,   4,   -6,  -6,  -6,  5,   -6,  -9,  -10, -9,
      -4,  7,   2,   7,   6,   5,   6,   3,   -4,  2,   -9,  -6,  4,   -5,  0,   -10, -3,  0,   -5,  10,  -2,  -7,  -7,
      3,   -6,  -5,  6,   -8,  8,   -6,  10,  1,   9,   -1,  9,   2,   -9,  0,   1,   1,   1,   9,   -5,  0,   -1,  6,
      5,   1,   10,  -4,  8,   6,   0,   -6,  3,   6,   9,   10,  6,   8,   6,   -2,  -9,  5,   -7,  5,   -8,  -4,  1,
      -8,  2,   -9,  -4,  7,   9,   4,   -7,  0,   -7,  -10, -9,  1,   2,   -5,  10,  0,   8,   7,   10,  -1,  6,   -4,
      2,   7,   -1,  -6,  0,   5,   2,   9,   -8,  9,   -7,  9,   -9,  -10, 6,   -7,  -1,  6,   8,   3,   -5,  -6,  -2,
      -1,  1,   3,   -1,  -3,  0,   2,   5,   -1,  10,  0,   -6,  5,   -10, 10,  -6,  -3,  -3,  9,   0,   -9,  0,   -4,
      1,   3,   0,   4,   -3,  -8,  5,   3,   -1,  9,   4,   6,   0,   4,   -10, -8,  6,   -1,  -9,  4,   5,   0,   -3,
      -7,  2,   -1,  -3,  -6,  2,   -2,  2,   3,   -6,  -6,  -1,  7,   -2,  9,   5,   3,   -9,  -4,  4,   5,   -6,  1,
      -1,  3,   -2,  -5,  4,   -10, -3,  6,   10,  -3,  7,   1,   -4,  -5,  -7,  9,   1,   0

    });
  auto weights_arr = gko::array<double>(
    exec, I<double>{3,  0,  8,  5,  -1, 1,  0, -9,  -10, -7, -6,  2,  3,  -2,  6,  -8, -3, 1,  -4, -4, -3,  -1,
                    9,  -3, 10, -7, -7, -6, 2, -3,  2,   1,  9,   -8, -6, -10, -4, 4,  7,  -4, 8,  1,  -10, 6,
                    -9, 2,  -5, -5, -2, -1, 9, -10, -3,  1,  -10, -3, 5,  0,   -6, 5,  4,  -5, -5, 1});
  auto src_arr = gko::array<double>(
    exec,
    I<double>{
      2,   -4,  -6,  -9,  1,   10,  3,   -10, 8,   6,   6,   5,   4,  -10, 1,   1,   -1,  9,   -10, -1,  0,   2,   8,
      -8,  6,   -2,  2,   -9,  -10, -6,  9,   9,   2,   7,   6,   -2, 6,   -2,  -7,  -7,  -6,  8,   8,   -6,  4,   -9,
      -4,  -4,  9,   3,   -7,  -4,  6,   -2,  0,   -5,  6,   -4,  -6, -5,  -2,  10,  6,   -5,  -10, 9,   -9,  -3,  5,
      -8,  -8,  -2,  9,   -2,  -7,  2,   -9,  3,   8,   -1,  -10, -3, 6,   -10, 5,   2,   -6,  6,   -3,  -3,  -8,  8,
      -4,  4,   -9,  -5,  6,   9,   4,   -5,  -5,  -10, 8,   6,   3,  3,   -8,  -3,  -7,  4,   -7,  -5,  8,   -1,  7,
      8,   2,   6,   7,   5,   -6,  -2,  2,   2,   10,  1,   1,   -9, -1,  5,   7,   -8,  7,   -8,  -10, 0,   -5,  3,
      -9,  -5,  -2,  -5,  7,   -8,  2,   4,   -4,  -2,  -6,  -1,  2,  2,   7,   4,   -8,  -2,  -10, -6,  -8,  10,  -2,
      -3,  -8,  9,   9,   -10, 6,   -6,  3,   -7,  -3,  3,   -6,  4,  5,   1,   -8,  -6,  -5,  -2,  -5,  -10, 1,   -5,
      -3,  -1,  -9,  -7,  -6,  5,   6,   2,   -8,  -7,  -4,  1,   2,  1,   -3,  9,   9,   4,   1,   -8,  2,   -10, 8,
      10,  0,   7,   -6,  -8,  10,  4,   5,   -9,  6,   -9,  -3,  10, -2,  -7,  -4,  -4,  -2,  -7,  -1,  -10, 2,   6,
      -8,  -7,  -7,  3,   -10, 9,   1,   0,   8,   3,   -9,  -3,  -8, 7,   2,   -8,  -5,  7,   -2,  -4,  -4,  0,   2,
      2,   10,  5,   1,   8,   5,   -8,  -10, 9,   -2,  -4,  7,   2,  -10, 9,   1,   -9,  -4,  0,   -4,  -2,  -4,  2,
      2,   7,   -1,  1,   -4,  5,   -3,  -6,  -5,  1,   -10, 8,   -4, 0,   -10, 0,   -3,  -8,  -4,  -3,  3,   2,   4,
      6,   -8,  -10, -6,  -7,  -2,  -7,  -8,  7,   3,   4,   -2,  -3, -8,  -4,  -10, -3,  3,   8,   3,   9,   4,   10,
      5,   -4,  3,   10,  -4,  8,   -10, -2,  5,   -8,  -2,  4,   3,  9,   8,   -6,  4,   1,   3,   -4,  -8,  5,   -3,
      0,   7,   -3,  -8,  -9,  -10, 1,   5,   -6,  4,   2,   -1,  9,  4,   8,   2,   -7,  5,   -3,  7,   -5,  -9,  0,
      2,   -3,  -5,  -7,  9,   -6,  7,   -4,  -5,  -6,  -1,  -8,  -1, -2,  1,   6,   -3,  -8,  -5,  2,   8,   0,   -9,
      -7,  1,   7,   0,   2,   -1,  0,   -1,  -2,  -10, 9,   10,  5,  7,   5,   -8,  -7,  6,   -3,  -5,  -9,  8,   2,
      -3,  1,   -4,  -8,  4,   4,   0,   -8,  2,   2,   10,  5,   4,  1,   4,   -2,  -6,  -8,  -9,  -9,  -5,  6,   9,
      4,   7,   0,   -3,  -1,  -10, 4,   -1,  5,   0,   -10, 4,   0,  7,   -7,  3,   8,   -2,  8,   10,  -7,  -7,  3,
      -4,  -1,  2,   -7,  -1,  9,   -3,  4,   10,  7,   4,   -2,  -9, -9,  10,  10,  0,   9,   -2,  -8,  7,   5,   -6,
      -6,  -4,  -7,  -8,  -5,  4,   -4,  10,  3,   -10, -4,  -3,  4,  -6,  -5,  5,   -7,  10,  1,   -1,  8,   -2,  8,
      -9,  1,   8,   6,   6,   3,   -5,  0,   7,   1,   4,   -10, -5, -4,  -3,  6,   -3,  -4,  -3,  10,  10,  -1,  -8,
      -5,  -1,  6,   2,   0,   -2,  8,   -2,  -2,  10,  -6,  2,   3,  -3,  3,   -9,  7,   4,   -8,  1,   -4,  -6,  -6,
      6,   -7,  2,   5,   -8,  2,   -10, -6,  5,   9,   -10, -7,  -9, -7,  4,   -8,  -5,  -10, 7,   2,   9,   -2,  9,
      2,   6,   -2,  3,   -7,  -4,  -6,  1,   1,   2,   6,   9,   -8, 5,   -4,  -9,  10,  1,   -5,  -3,  -8,  9,   6,
      -2,  -4,  2,   -4,  -10, -5,  -3,  -6,  4,   3,   -9,  2,   5,  10,  -4,  2,   1,   -2,  6,   -5,  -7,  -4,  -10,
      0,   -6,  -5,  -4,  -8,  4,   4,   1,   3,   -6,  -8,  5,   -1, 1,   1,   -7,  -4,  4,   10,  -8,  0,   -9,  4,
      2,   -8,  10,  -7,  -4,  0,   8,   0,   -3,  4,   -1,  9,   6,  2,   -9,  0,   -1,  2,   4,   -2,  -5,  -9,  6,
      1,   4,   -3,  -7,  -2,  -2,  1,   1,   -3,  -5,  0,   -3,  -6, -7,  -7,  2,   4,   -9,  3,   1,   6,   4,   6,
      1,   0,   -4,  9,   2,   -5,  -5,  -5,  7,   1,   -4,  4,   1,  -1,  -10, 0,   10,  8,   -6,  5,   5,   -8,  1,
      -4,  1,   8,   6,   -6,  -2,  9,   1,   -2,  5,   10,  6,   -6, 5,   3,   -2,  9,   10,  3,   -10, 9,   -1,  0,
      6,   -6,  7,   6,   2,   4,   -5,  7,   0,   -10, -4,  5,   6,  0,   5,   2,   8,   -1,  7,   7,   -4,  -8,  5,
      -1,  -9,  6,   9,   9,   -3,  4,   -1,  -7,  -3,  -8,  -5,  -9, 4,   -6,  -1,  -9,  -3,  1,   6,   2,   7,   3,
      1,   -3,  -8,  -6,  5,   -6,  2,   -6,  -4,  8,   8,   9,   -9, -3,  -6,  -5,  1,   9,   -3,  -3,  -8,  7,   0,
      -7,  8,   5,   7,   -9,  10,  -9,  0,   10,  10,  7,   8,   -3, 10,  0,   -8,  3,   -8,  -1,  -9,  4,   4,   -3,
      8,   -5,  -4,  0,   1,   2,   8,   6,   3,   6,   1,   10,  -5, 8,   -4,  1,   -3,  0,   9,   6,   -8,  9,   2,
      -10, -9,  -2,  0,   -8,  9,   1,   7,   -3,  8,   10,  2,   -8, 3,   3,   7,   6,   -1,  -4,  -2,  9,   -2,  4,
      5,   9,   8,   -10, -9,  -9,  -6,  -10, -5,  -9,  -10, 6,   5,  9,   -6,  7,   -3,  -10, 0,   8,   2,   -1,  -10,
      -8,  -8,  5,   8,   -5,  -6,  -10, -10, 6,   3,   -5,  -1,  4,  -9,  -3,  -4,  1,   -5,  9,   2,   -3,  9,   7,
      -2,  6,   7,   2,   -4,  -6,  -1,  -10, -9,  7,   -4,  -7,  3,  -1,  5,   -5,  8,   1,   -5,  4,   0,   -9,  5,
      -2,  0,   8,   1,   -2,  9,   7,   1,   -2,  -1,  -7,  -9,  -3, 7,   -9,  10,  1,   3,   -7,  5,   -5,  9,   -9,
      4,   5,   -6,  5,   -8,  10,  -10, -3,  7,   0,   -1,  10,  9,  -10, 5,   9,   -2,  -5,  8,   -2,  -10, -8,  10,
      8,   -10, 10,  10,  2,   7,   5,   -6,  -10, 4,   5,   0,   6,  -2,  6,   -5,  7,   -10, -3,  0,   0,   -8,  -7,
      -3,  -8,  7,   -8,  -2,  -2,  10,  -6,  5,   1,   -1,  -1,  0,  3,   -10, 10,  -10, -8,  7,   4,   4,   -4,  6,
      9,   -5,  6,   -2,  -10, 5,   6,   2,   1,   -10, 6,   -7,  -7, -7,  10,  10,  -9,  -7,  4,   5,   4,   -5,  8,
      1,   3,   8,   9,   -4,  5,   -2,  -10, -8,  6,   6,   10,  -5, 2,   -9,  -5,  9,   -3,  -8,  0,   0,   9,   2,
      8,   7,   -9,  9,   1,   -1,  1,   5,   3,   6,   -4,  3,   -7, 4,   8,   6,   -6,  -9,  0,   9,   -9,  7,   7,
      7,   10,  -5,  6,   6,   5,   5,   -9,  -8,  -1,  5,   -8,  8,  1,   4,   -1,  2,   -6,  -10, 0,   5,   -5,  -1,
      1,   10,  -2,  -9,  -10, 10,  -10, -6,  -7,  7,   -4,  7,   6,  -9,  8,   -9,  -7,  3,   10,  6,   3,   -9,  -4,
      3,   5,   8,   -3,  7,   9,   -7,  -5,  -10, 9,   2,   -4,  -5, -3,  7,   -10, -4,  9,   -8,  -3,  -9,  -8,  0,
      -7,  -9,  10,  -1,  -4,  -9,  3,   4,   1,   5,   4,   10,  -3, 10,  3,   -7,  8,   3,   9,   -3,  4,   9,   2,
      -4,  -1,  7,   -3,  -8,  8,   5,   -4,  3,   8,   -8,  8,   5,  5,   -9,  2,   0,   5,   4,   -5,  -10, -4,  6,
      -8,  9,   7,   1,   -2,  -10, -3,  3,   9,   -5,  -1,  0,   -5, 7,   -7,  9,   7,   -5,  1,   4,   -5,  -7,  -1,
      1,   2,   -3,  -7,  2,   -5,  2,   -4,  8,   7,   1,   -2,  -7, 9,   -8,  10,  8,   -8,  6,   2,   10,  -8,  1,
      2,   4,   -10, 9,   7,   -3,  -4,  5,   9,   -7,  -7,  9,   -9, -1,  4,   1,   -6,  1,   2,   -3,  10,  7,   8,
      -3,  6,   -3,  -2,  1,   6,   10,  4,   -1,  -3,  5,   -5,  -9, 5,   -8,  7,   -9,  -1,  5,   -5,  -3,  -4,  -7,
      4,   -1,  -6,  2,   -10, 1,   -2,  -5,  -9,  -5,  -2,  -5,  3,  6,   0,   -4,  8,   -7,  -9,  1,   6,   5,   -5,
      5,   -9,  8,   6,   0,   1,   4,   8,   7,   8,   -1,  3,   -7, 0,   0,   9,   -8,  9,   0,   1,   1,   -1,  -4,
      9,   7,   6,   -7,  2,   10,  -1,  0,   -8,  5,   -4,  4,   3,  -9,  -10, 9,   -5,  -5,  0,   -1,  -6,  -2,  -6,
      1,   0,   -7,  3,   8,   -7,  4,   -2,  8,   -2,  -6,  8,   8,  9,   4,   -9,  -5,  3,   -5,  5,   -10, -5,  -3,
      3,   -10, -8,  -1,  -3,  -4,  6,   -3,  0,   1,   -8,  7,   -5, 8,   -1,  -5,  1,   -7,  -6,  0,   -7,  4,   9,
      9,   -1,  -1,  -10, 6,   7,   10,  5,   -9,  -10, -6,  10,  -1, -3,  4,   -2,  2,   3,   -7,  -4,  -1,  9,   -9,
      1,   8,   -4,  -7,  2,   -6,  -8,  -2,  -9,  7,   -9,  -7,  2,  -4,  -10, -8,  0,   7,   0,   3,   -10, -6,  6,
      4,   2,   -5,  1,   -5,  5,   8,   -5,  7,   -10, -4,  -3,  -6, -5,  8,   -8,  2,   -2,  3,   -4,  -9,  -9,  -6,
      5,   6,   -10, 7,   -2,  5,   3,   5,   10,  0,   -5,  -5,  7,  -7,  9,   -1,  -8,  -7,  -7,  6,   7,   8,   -10,
      6,   -6,  9,   2,   -3,  4,   3,   -8,  7,   -6,  0,   4,   4,  -7,  -8,  4,   0,   1,   -1,  -9,  2,   -1,  -3,
      -2,  5,   -3,  4,   -6,  8,   6,   -2,  -1,  7,   7,   7,   -3, 3,   0,   8,   -8,  5,   9,   6,   -4,  2,   -6,
      -5,  -7,  -7,  -4,  -5,  -4,  -2,  -7,  -1,  -10, -2,  -8,  2,  -9,  -10, -2,  -9,  -4,  3,   10,  9,   4,   -2,
      -10, -9,  -8,  -2,  -7,  8,   7,   9,   1,   1,   -4,  -4,  -3, 6,   -9,  3,   6,   -1,  6,   -4,  -5,  -6,  1,
      -9,  2,   7,   10,  -10, 0,   3,   -7,  2,   5,   -3,  0,   2,  -10, -2,  3,   -5,  7,   -1,  6,   3,   4,   -9,
      6,   -9,  8,   3,   9,   -6,  2,   -5,  6,   -10, 1,   -5,  5,  6,   7,   -10, -9,  4,   1,   -9,  5,   5,   2,
      9,   -8,  -2,  10,  5,   -1,  -3,  -4,  1,   6,   -6,  -8,  -3, 0,   1,   5,   -2,  8,   -8,  -9,  -7,  -4,  -4,
      1,   1,   -8,  -9,  -8,  -2,  -2,  5,   6,   -4,  3,   7,   3,  4,   -2,  3,   5,   7,   -3,  0,   6,   -9,  8,
      -4,  -8,  -1,  -5,  5,   9,   7,   7,   -9,  8,   6,   -1,  3,  -3,  3,   -3,  4,   5,   -7,  -6,  -5,  -9,  -2,
      -6,  -9,  6,   6,   -9,  1,   1,   -5,  -1,  5,   6,   -7,  7,  2,   2,   0,   5,   6,   -8,  -4,  -3,  -10, 0,
      -4,  5,   -8,  6,   9,   2,   4,   3,   9,   5,   -7,  8,   -8, -7,  -10, -4,  1,   3,   9,   2,   -3,  -2,  7,
      -10, 6,   7,   8,   5,   1,   6,   9,   -1,  -6,  7,   -8,  -2, 7,   7,   6,   7,   9,   -3,  -7,  7,   -6,  3,
      4,   4,   -5,  -9,  5,   2,   -6,  5,   2,   4,   6,   2,   0,  -2,  3,   4,   4,   8,   5,   2,   3,   3,   -8,
      -3,  -8,  -8,  8,   -1,  -8,  -1,  4,   -2,  -2,  2,   -6,  -5, 1,   -7,  0,   -5,  0,   5,   8,   1,   -8,  -1,
      -5,  -9,  3,   -1,  4,   9,   -3,  6,   5,   3,   -7,  -2,  7,  -3,  4,   10,  -6,  0,   10,  -3,  -10, -8,  10,
      3,   9,   -9,  7,   4,   4,   5,   -10, 0,   -10, 4,   2,   -5, 3,   3,   3,   -8,  -3,  -9,  1,   2,   2,   3,
      -10, 10,  10,  -6,  7,   7,   -4,  9,   8,   2,   -3,  -9,  9,  -2,  9,   -5,  10,  10,  6,   -8,  6,   -7,  8,
      4,   7,   -6,  10,  -8,  3,   -6,  6,   1,   -4,  4,   10,  -3, -6,  5,   2,   7,   10,  -7,  -8,  -9,  -6,  8,
      -4,  1,   1,   0,   4,   -1,  -5,  -2,  7,   -2,  4,   -3,  -7, 8,   2,   4,   -8,  -10, -5,  -3,  -10, 8,   -9,
      -10, -10, -3,  -7,  -5,  10,  -2,  8,   6,   -4,  -6,  5,   5,  10,  -3,  -3,  -6,  2,   -7,  -7,  5,   8,   6,
      9,   5,   -10, -1,  -10, -6,  -7,  -10, -1,  2,   0,   7,   2,  -8,  -9,  6,   10,  0,   6,   2,   9,   10,  10,
      2,   4,   -1,  4,   -10, 10,  6,   4,   -9,  5,   -7,  -9,  -1, -9,  4,   3,   5,   -9,  3,   -5,  3,   6,   10,
      5,   -8,  2,   -6,  -3,  -2,  1,   -4,  -8,  8,   10,  10,  -9, -7,  -4,  1,   -3,  -6,  -5,  9,   -1,  -4,  0,
      4});
  auto dst_arr = gko::array<double>(exec, cell_size * 4);
  auto expected_dst_arr = gko::array<double>(
    exec,
    I<double>{
      -0.0192857, -281.25,  -30.1125, -50.4,    33.2143,  -0,       22.125,    0,        57.8571,  61.2,     27.825,
      95.55,      220.436,  174,      3.375,    -306.25,  -34.6371, 571.05,    -46.2375, 76.5,     18.24,    -713.25,
      65.625,     683.3,    -86.0786, 207,      31.5,     -23.4,    -90.1286,  0,        76.875,   -684,     -0.495,
      -4278.75,   -34.875,  81.9,     54.6,     -2764.12, 1876.2,   -186.2,    76.3714,  1179.9,   0,        -251.55,
      -176.186,   -1836,    0,        390,      62.73,    -492.075, 96.3,      291.6,    59.7086,  972,      -0.9,
      -1227.3,    34.65,    -3845.25, 36.45,    -99.6,    350.914,  675,       192.5,    -93.6,    108.654,  0,
      -124.2,     302.7,    542.7,    131.25,   1258,     -243,     273.6,     -2433.6,  0,        -492.25,  -566.186,
      432,        -0,       -48,      -149.271, -838.35,  349.05,   -158.7,    -419.691, 1572.75,  31.3875,  -1123.55,
      -329.164,   1295.25,  -20.925,  525,      402.579,  -275.625, -98.4417,  194.4,    8.505,    247.5,    0,
      479.025,    -15.6,    742.875,  -942.113, 39.375,   253.5,    3444.45,   131.25,   -154.525, -70.7143, 5994,
      662.175,    1048.58,  28.7657,  -1383.9,  42.375,   -108.825, -63.36,    547.875,  49.95,    -394.875, 16.65,
      -522,       -8.325,   -12.85,   65.1857,  -52.5,    -208.3,   129.675,   290.248,  -4529.25, -4.125,   2.225,
      -10.4571,   1770,     -722.55,  -87.775,  -7.71429, 583.2,    -889.875,  10.0917,  204.814,  -3244.5,  713.625,
      -164.375,   -65.7771, -619.65,  143.175,  -361.775, -71.9314, 454.5,     90.975,   -563.425, 30.4071,  -6018.75,
      -4.425,     49.3,     48.3429,  381,      -312.475, -10.2083, 108.624,   204,      56.1958,  -604.8,   161.271,
      83.25,      -1785.62, 18,       95.7857,  1202.4,   -16.4375, -56.45,    115.971,  2205,     7.8125,   -429.4,
      -249.03,    425.25,   61.1375,  -5.4,     -198.591, 843.75,   43.0208,   1061,     -116.293, 207,      25.8125,
      -509.15,    6.04286,  0,        -159.312, 0,        -4.49571, -232.5,    16.2,     -40,      5.27143,  0,
      -1318.69,   23.6,     0,        -5080.72, -178.2,   1066.5,   361.8,     2605.5,   -99.75,   -195,     -49.41,
      1020.6,     -192.6,   -264.6,   -36.27,   -1942.5,  0,        1656.35,   -154.157, 339,      307.5,    -46.8,
      -84.7286,   128.25,   15.875,   106.65,   23.8436,  35.4375,  236.25,    561.45,   -56.5714, 212.625,  613.75,
      1012.5,     -141.75,  -534.488, -382.95,  443.1,    -32.5929, -1401.19,  -210.375, -443.25,  26.5757,  148.838,
      -321.45,    -360.45,  94.3757,  -58.6875, -213.75,  -791.75,  -96.8357,  -526.688, 350.438,  -514.05,  -39.0643,
      288.562,    232.096,  510.45,   14.6571,  -1687.5,  46.4917,  880,       195.193,  56.7,     -192.5,   -1009.15,
      29.5714,    -283.05,  39.45,    -1475.22, -18.5036, 3948.3,   -71.4167,  -1012.75, 80.4343,  294,      -85.4375,
      -647.5,     -86.3121, -258.75,  216.417,  959.6,    -16.7657, -96,       276.25,   1934.05,  -45.2571, 0,
      3.75,       1027.73,  -9.6,     14985.9,  -209.25,  174.125,  427.5,     2520,     -233.1,   -1488.9,  -621.771,
      -1834.01,   165,      354.075,  421.631,  257.062,  -196,     548.625,   -92.52,   2849.74,  0,        44.625,
      155.286,    220.238,  -41.4,    112.475,  -469.08,  3732.41,  -36.45,    -27.075,  -259.714, -448.462, 99,
      48.975,     -21.4714, 0,        63.45,    -134.3,   334.093,  693,       -191.25,  -450.75,  135.771,  1761.75,
      134.75,     -735.15,  329.484,  -1547.1,  34.75,    805.5,    -477.986,  2528.4,   -134.55,  -403.5,   -252.306,
      47.25,      63.8625,  178.6,    -63.4114, 13664.2,  -67.05,   -331.35,   192.857,  -3071.25, -582.675, -78.9,
      -32.4429,   2430,     -21,      -364.667, -240.557, 2187.68,  324.45,    1064.15,  -46.9286, -2513.7,  795,
      92.05,      111.105,  -3744.9,  -501.175, -1022.8,  -64.62,   -440.1,    230.85,   -324.4,   119.381,  -4198.5,
      368.7,      -664.567, 45.4114,  288,      -23.175,  -226.15,  -86.6571,  0,        -73.8,    167.2,    -284.914,
      7900.54,    53,       -179.55,  -291.921, 10357.8,  131.7,    -754.8,    43.2,     -257.513, 918,      981,
      -676.781,   3587.06,  683.75,   578.25,   -75.0343, -1793.14, 47.1,      858.45,   -23.0336, -2899.46, 364.1,
      845.55,     -60.0171, 15937.2,  -57.3,    -259.8,   20.0571,  -1614.49,  -510.6,   -24.75,   -606.986, 1218.75,
      371.3,      561.483,  -314.464, 666.45,   17.5,     966.35,   -8.35714,  -100.8,   -218.708, 160,      -456.199,
      -1275.75,   72.8333,  -635.05,  326.327,  -3571.05, -114.125, 1404.35,   48.3793,  -970.5,   18.5,     16.5833,
      -2.57143,   495.75,   -35,      1216,     1.67143,  -15.75,   -5.83333,  21.25,    27.2571,  758.7,    -217.675,
      71.7,       324.707,  -5.4,     584.062,  556.5,    457.714,  5693.62,   651.825,  -632.85,  -258.364, -3100.5,
      -264.625,   -1008,    278.126,  -2586.6,  235.875,  421.2,    -391.217,  108,      -479.875, 8.7,      -96.0171,
      -102.6,     1017.75,  321.6,    551.443,  710.1,    -25.95,   -99.15,    212.143,  -49.5,    0,        -366.833,
      192.15,     -278.1,   0,        464.95,   -694.286, -176.85,  -13.3,     757.35,   -436.686, 895.275,  57.25,
      -839,       51.24,    1971.3,   -406.8,   -320,     -22.0479, -3476.25,  462,      -825.933, 484.881,  -1540.12,
      -732.375,   -553.85,  4.62857,  -330.75,  -17.8875, 409.2,    -0.235714, -1482.45, 18.225,   -414.75,  40.0757,
      -148.5,     -223.438, -198.45,  58.1529,  -253.5,   69.0375,  512.25,    -81.36,   1869.3,   39.6625,  594.5,
      100.041,    -65.7,    -301.95,  127.55,   -56.2929, -455.55,  305.938,   471.15,   -169.521, -426.6,   1151.78,
      -117,       96.66,    -2578.5,  -196.088, 1165.35,  472.114,  14884,     -104.625, 484.5,    217.646,  4736.48,
      1040.4,     125.2,    -324.103, -4160.7,  -198,     174,      -57.24,    -1393.2,  85.4,     -294,     -43.6629,
      -1092.83,   349.2,    -145.8,   -57.0857, 43.2,     -242.7,   -264.5,    -418.693, 3882.15,  -867.15,  265.5,
      -713.854,   1459.8,   -432.9,   156,      254.636,  1180.8,   224.667,   577.1,    -50.4,    681.75,   1077,
      372,        -69.6,    2287.2,   26.3333,  -1227,    165.343,  -3303,     -40.6667, -147,     -253.414, -270,
      32.25,      -257.4,   -17.9143, 78.75,    -650.646, -613.65,  388.607,   6527.25,  -1784.78, 112.5,    -318.034,
      2512.13,    -231.542, -189,     -290.829, 1913.4,   5.83333,  -497,      142.389,  932.475,  47.4125,  -61.8,
      -178.946,   -4468.05, 213.25,   -645.8,   213.146,  -6644.7,  -1035.72,  -1649.4,  5.96571,  1805.55,  497.525,
      284.7,      -53.2286, -2381.18, 240.683,  -528.5,   21.7929,  1436.4,    -873.025, 261,      -133.616, 2620.8,
      253.3,      -423,     -730.8,   10102.2,  20.125,   373.35,   -100.183,  5176.01,  -346.65,  -207.2,   -3.18857,
      14.9625,    1324.88,  138.8,    -542.554, 7834.24,  -233.725, 15,        138.651,  491.738,  337.725,  710.3,
      36,         -1304.44, 230.425,  -120.4,   95.85,    17189.5,  -917.175,  -147.75,  -115.83,  876.038,  20.775,
      140,        -924.214, -202.65,  192.525,  534,      53.2543,  170.1,     -871.25,  -169.2,   50.1171,  487.35,
      120.75,     525,      -78.1457, -1.35,    -133.775, 276.2,    278.37,    448.65,   -562.95,  834.3,    -41.0143,
      -2478.9,    209.375,  435,      -159.879, 1892.7,   812.025,  1121.25,   323.679,  -2569.95, 396.375,  0,
      -380.357,   -766.8,   -373.275, -911.75,  281.494,  -1074.6,  -106.312,  -350.55,  448.02,   9957.38,  1396.12,
      -321.15,    52.38,    -2508,    -48.175,  270.75,   216.54,   634.5,     -276.525, 596.25,   -249.45,  612,
      -301.875,   777.3,    -331.671, 1273.5,   1691.55,  -382.5,   651.484,   -5120.55, 97.65,    -219,     494.614,
      -115.2,     109.875,  -792.25,  -196.663, -57.6,    20.25,    -39.525,   -614.794, -189.6,   -392.375, 465,
      -510.249,   -46.875,  49.5333,  -285,     -117.103, -270,     -334.125,  -108.9,   183.171,  -2925,    1070.62,
      123,        718.071,  -1578.38, 397.987,  -21.375,  12.4586,  -1160.55,  -363.013, 40.35,    197.946,  700.5,
      4.8375,     870,      -262.311, -0.3,     -98.65,   -812.4,   -381.806,  -198,     -256.725, -382.5,   -183.536,
      -2133.9,    -55.875,  753.75,   -161.743, 180,      -360.562, 280.2,     542.28,   -1059.75, 93.075,   771.6,
      452.786,    1920,     73.125,   382.5,    150.986,  -306,     -174.458,  337.1,    914.7,    900.562,  125.083,
      -1529.5,    -961.74,  2549.7,   480.55,   66.3,     -1194.06, -3062.81,  56.875,   444.75,   -542.893, -6311.59,
      117.125,    5.25,     -571.95,  1560.94,  375.125,  5.25,     399.784,   -4628.81, -264.892, -59.95,   1624.82,
      -863.438,   -93.325,  131.25,   -40.05,   -475.312, 8.375,    388.35,    -332.711, -125.25,  240.45,   -1446.97,
      -71.82,     -394.65,  -101.617, -575.325, 377.64,   1620,     -189.583,  -349.875, 578.057,  -5042.7,  176.25,
      -433.125,   638.1,    540,      440.2,    -248.625, -586.029, -2355.75,  -4.0375,  -3.225,   -414.386, -11087.2,
      290.275,    866.25,   -394.307, 1741.88,  394.392,  180.975,  -139.725,  -915.75,  49.875,   -1341,    38.52,
      -1392.97,   29.4125,  142.2,    -266.94,  -2598.75, -474.375, 121.8,     -231.043, -2980.8,  -536.55,  -1200.6,
      11.0571,    317.25,   405,      7.2,      -330.48,  3945.38,  -15.075,   -364.5,   -72,      -1152,    132.075,
      -127.5,     -93.9857, 306,      151.525,  -292.8,   -430.511, 3953.7,    1.5,      -749.85,  107.949,  -5338.8,
      -135.4,     31.2,     84.7457,  -49.5,    877.5,    -904.3,   -376.843,  6511.5,   -917.25,  88.5,     118.929,
      -569.7,     351.9,    181.5,    -331.466, 3985.2,   -124.65,  105.9,     -104.271, 2025.9,   50.4,     409.35,
      -79.9286,   306.9,    410.8,    -338.5,   -172.189, -287.25,  -202.087,  1994.78,  53.4857,  -288.75,  133.062,
      113.625,    71.7429,  0,        156.188,  -271.625, -18.5143, 4765.5,    -170.062, 96.225,   75.0214,  -900,
      -613.688,   -782.175, 348.994,  1955.25,  -30.9375, 240.375,  -82.9714,  4224,     -153,     462,      -80.8286,
      -306,       57.6042,  -226.875, -166.468, 267.45,   -51.6,    2153.95,   4.06286,  257.85,   -10.1125, 165.2,
      10.9886,    6006.98,  178.2,    -1112.97, -30.3429, 2480.85,  -48.75,    51.5,     -105.943, -725.85,  -864,
      83.9,       716.053,  2190.75,  86.4,     307.45,   22.1143,  5913.9,    -922.5,   -103,     -188.529, -836.4,
      -104.675,   -273.817, 33.6536,  -240,     -59.4792, -241,     -36.72,    14.85,    115.883,  -23.45,   285.583,
      -5.25,      80.0125,  284.167,  356.786,  -2395.58, -73.4375, 233.5,     -7.2,     515.25,   102.688,  -36.5,
      112.731,    -1919.25, 314.246,  -41.5,    86.4643,  654.375,  256.25,    -82.25,   66.6857,  -267,     -90.2,
      184.067,    -322.65,  797.85,   99.45,    347.65,   252.045,  105.3,     -62.4375, 2341.35,  534.375,  -198,
      -287.663,   1100.4,   285.429,  -1380.6,  -10.4375, -650.15,  312.801,   108,      -163.725, -1195.8,  -250.258,
      -258.75,    26.4375,  -1962,    -558.9,   -113.4,   86.075,   -1093.5,   -69.1586, 0,        20.8125,  1763.1,
      27,         1571.25,  142.425,  250.6,    -163.395, 2134.12,  277.2,     -88.15,   -58.4357, -3814.2,  0,
      -0.266667,  104.657,  -1777.5,  -7,       -383.2,   -5.58,    -236.475,  99.6,     -295,     -14.0207, 522.3,
      38.7,       -155.3,   200.571,  3985.65,  -269.25,  716,      -272.52,   -512.1,   -49.5,    191.933,  351.321,
      -137.7,     143.1,    123.15,   -325.382, -569.25,  0,        -257.25,   -717.654, 1620,     0,        392.7,
      -352.671,   -2367,    -11.6667, -130.9,   -352.329, -180,     -0.05,     184.5,    271.189,  47.25,    28.6125,
      -417.15,    812.7,    -6435.45, -53.625,  0,        106.907,  1231.88,   254.475,  -113.4,   52.2857,  -1055.17,
      -29.3167,   -460.65,  -233.081, -2382.3,  102.088,  -154.35,  -626.518,  -3209.62, -425,     101.1,    -255.729,
      -5195.02,   -637.925, -1268.1,  2.39143,  973.125,  152.225,  7.2,       172.751,  2302.5,   -61.7667, 96.3,
      254.571,    999.45,   182.975,  -127.5,   -31.2771, -320.625, -25.85,    -142.8,   -291.6,   3906.9,   107.6,
      -74,        97.3479,  -5143.95, 27.9,     453.65,   -47.7321, 36.45,     802,      -241.167, -407.829, 3839.85,
      -1282.75,   78.5,     103.963,  -371.25,  -165,     239,      -59.535,   1340.55,  -222.55,  -78.8,    -32.4,
      3450.6,     274.8,    -367.15,  44.28,    197.55,   400.3,    -60.6667,  220.457,  -355.5,   42.1,     629.05,
      -215.891,   -211.95,  -268,     -19.65,   -249.268, -45,      -432.042,  -390.5,   -226.286, 437.4,    -207,
      131.5,      -62.37,   81,       -499.225, -832.8,   255.821,  -2022.75,  232,      579,      371.571,  621,
      730,        563.25,   7.14857,  9,        89.3333,  40,       -36.8357,  242.55,   -33.2292, -832,     64.3757,
      182.25,     5.375,    953.95,   192.375,  5875.88,  152.387,  -96.5833,  -93.15,   -67.05,   265.562,  829.45,
      -103.59,    187.65,   -246.888, 706.15,   42.57,    23.85,    -77.9792,  -394,     -27.3214, 621.9,    -771.1,
      -1347.5,    -199.453, -375.3,   55.6375,  -331.233, -106.071, -216,      18.9,     -402.45,  -9.135,   -434.7,
      0,          51.45,    373.018,  -234.9,   -10.8,    227.7,    389.186,   361.575,  -75.2917, 287.9,    -1.74,
      270.9,      151.3,    0,        22.4164,  886.5,    438,      -280.8,    14.4643,  924.075,  249.788,  0,
      -17.0743,   -417.15,  5.7375,   -12.6,    58.2857,  1270.65,  67.3875,   104,      325.234,  -417.6,   -58.2,
      624,        205.187,  -21.9,    -313,     312,      -70.5793, -2895,     -81.6,    -547.65,  -179.717, -79.2,
      -118.312,   -507.3,   -261.086, 669.75,   -21.6,    -643.3,   -183.069,  206.1,    0,        -332.25,  231.364,
      353.7,      -3,       34.3,     45,       -13494.8, 162.133,  -169.125,  -380.28,  -2764.12, 354.95,   269.05,
      354.819,    1353.6,   18.875,   -484.875, -45.9814, 637.2,    24.725,    848.475,  -241.56,  -968.625, -272.675,
      34.875,     75.8571,  -63,      6.30833,  157.425,  726.274,  -3364.65,  24.575,   13.575,   185.507,  112.5,
      -67.625,    -555.225, -158.036, -327.375, -648.075, -21.8333, -70.3971,  150.375,  150.775,  -860,     147.201,
      -1074.38,   501.458,  190,      -14.0679, 2090.47,  702.875,  836.8,     278.563,  -928.125, 844.058,  737.5,
      57.6429,    -862.125, 5.5125,   676.617,  -239.606, -12282.1, -526.625,  512.05,   -231.386, 1870.5,   32.3667,
      -1050.6,    3.08571,  -1626.53, 2.925,    419,      -76.6457, -2040.6,   -446.588, -546,     -4.09286, 2335.72,
      -707.375,   -112,     -245.578, 3076.43,  690.9,    1027.2,   95.2371,   -451.725, 69.35,    231,      -15.4286,
      3698.1,     -175.275, 645.5,    -35.5371, 326.25,   139.05,   114.05,    28.0929,  -495.675, 3.375,    -556.2,
      291.6,      -7041.04, -259.2,   87.75,    96.4114,  -8827.73, 121.725,   1074.4,   -64.9929, 82.2375,  -512.75,
      -693,       391.494,  -4492.24, -168.9,   -406.8,   13.0371,  852.862,   389.75,   -707.4,   0,        2505.94,
      -512.4,     -756,     6.12,     -13977.3, -351.375, -39.3,    5.85,      945.638,  363,      -456,     539.314,
      -694.5,     -307.133, -503.65,  46.8,     -423,     -137,     -140.75,   -33.03,   249.9,    76.375,   -152.583,
      162.135,    117,      51.4,     -649.55,  15.6214,  1413,     336.025,   -1238.75, 0,        1077.75,  -24.9333,
      -127.45,    -43.8,    -322.65,  -2,       -954.1,   -5.85,    603,       -1,       446.917,  -136.95,  -342,
      -48.6,      -520.45,  -15.9171, 963,      -178.088, 470.15,   -47.79,    -5313.82, -192.2,   1513.95,  215.595,
      1832.4,     425.55,   54.45,    191.43,   382.5,    32,       83.25,     54,       -684,     -43.2,    -673.6,
      -362.173,   -305.4,   -1537.5,  -1014.75, -430.393, 965.25,   14.475,    835.65,   -176.786, 6.375,    5.4,
      409.5,      -2.64,    399.375,  17.4,     0,        611.241,  -21.825,   8.46667,  -207,     544.738,  -435.15,
      -141.575,   688.8,    -111.78,  -1240.88, 156.033,  0,        4.28571,   2962.88,  -457.2,   46.8,     -399.613,
      1210.42,    610.312,  20.85,    -5.80714, 667.875,  14.2625,  137.5,     100.071,  829.95,   30.7125,  58,
      74.8286,    24.3,     33.6,     -22.6,    91.6543,  -255.75,  -99.3,     -45.3667, -45.2229, -1708.2,  -90,
      -691.3,     -131.529, 882.3,    3.375,    -322.6,   -60.6643, 527.7,     31.5,     12.1,     -74.4429, -35.1,
      0,          -466,     138.514,  -0.6,     480.833,  -1360.17, 27,        -15893.3, 145.667,  -94.5,    23.85,
      -5564.48,   5.625,    277.5,    466.097,  204.675,  -814.592, -592.2,    117.977,  1486.12,  -9.375,   -109.2,
      -10.8,      5747.63,  -9.375,   134.4,    239.014,  231.075,  107.242,   -222.6,   582.429,  -7232.4,  234.15,
      343.5,      547.714,  1346.33,  -693.042, -533.7,   -9.83571, 0,         -240.9,   72.9,     197.1,    -393.75,
      145.8,      106.2,    58.32,    -5285.25, -510,     525.9,    -28.1829,  3705,     -22.5,    -182,     236.829,
      2614.5,     410.85,   408.5,    184.307,  4.5,      -27.225,  -136.35,   -166.629, -7857,    40.275,   970,
      -50.4643,   1839.38,  -656.383, -28.7,    -92.5714, -1635,    41.0417,   298.5,    -633.279, 472.5,    -950.838,
      531.3,      -16.5343, 8544.75,  -507.758, 1030.4,   197.383,  7960.5,    1338.73,  2945.7,   278.743,  -4457.25,
      -174.85,    779.1,    366.107,  1363.12,  25.9417,  -268.5,   277.714,   -1485,    271.475,  -820.5,   -285.943,
      204,        -497.308, 140.6,    583.2,    -11319.8, 0,        57.65,     78.0429,  -5310,    -149.4,   681,
      -104.606,   2045.25,  -1671.3,  -554.7,   752.297,  -9261,    427.5,     -124.2,   -197.486, -2614.5,  -45,
      -757.8,     -96.75,   1453.5,   -242.55,  -646.6,   64.8,     -18356.6,  50.4,     636.3,    -34.2,    1692,
      -161.7,     82.5,     569.829,  -612,     -179.692, -1134.3,  -213.3,    -351,     -176,     -205.5,   -97.2,
      3408.75,    647.008,  -154,     2.4,      126,      37.375,   78.9,      -292.757, -4357.5,  246.625,  -821.7,
      -161.25,    2881.5,   40.4583,  -7.5,     0,        0,        24.625,    -2010.75, -57,      0,        603.292,
      292,        -325.114, -867,     -122.975, -247.15,  111.15,   -72,       -11.8,    197.5,    404.537,  -9837,
      356.158,    2181.32,  212.297,  3640.5,   373.625,  487.15,   275.657,   -2578.5,  74.375,   -453.65,  -468.536,
      -43.5,      -11.375,  -269.8,   -1007.36, 261,      -1071.25, -1701.5,   -344.314, -420.75,  1032.71,  402.717,
      -108.129,   0,        4.6875,   742.5,    30.8571,  -0,       128.925,   308.7,    399.291,  -2045.25, -160.238,
      -48.6,      361.517,  98.25,    -91.6875, 157.9,    6.17143,  2614.5,    208.238,  -4.3,     109.607,  3327.75,
      -912.938,   -184.5,   -485.614, 371.25,   352.125,  -437,     25.7143,   364.5,    135.379,  -104.6,   343.644,
      -977.4,     376.971,  -315,     -212.349, -57.6,    -41.5667, 315,       -95.2843, 120,      85.95,    525,
      86.7214,    1828.8,   1.25,     167.1,    -146.443, -718.2,   16.8125,   531.3,    183.9,    3011.25,  27.0833,
      -269.1,     93.2143,  118.8,    -39.65,   -630,     -176.966, 27.75,     -59.25,   -217.2,   -425.115, 2627.4,
      -1021.73,   94.5,     216.72,   -863.775, -260.2,   -295.4,   -91.8771,  1660.5,   116.4,    -150,     -49.7571,
      286.2,      0,        134.4,    237.6,    309.825,  0,        646.8,     4.88571,  -2925,    -2.7,     -37.1,
      -285.429,   -499.95,  163.95,   -45,      155.366,  265.5,    92.5,      -249.6,   -531.032, 222.75,   -1065.9,
      20.85,      -41.6571, 489.75,   -27.2,    -93.75,   -147.639, -378,      159,      -528.25,  -44.6143, 396,
      0,          20,       242.614,  1197,     32.85,    45.15,    90,        -4890.75, 10.4625,  230.1,    141.943,
      6843.15,    94.325,   187.5,    137.85,   -1367.62, -173.358, -173.15,   -412.515, 1326.15,  -269.1,   196,
      -793.38,    1353.22,  35.8625,  240,      -304.29,  741.75,   465.15,    -1.8,     526.114,  2131.2,   220.725,
      269.25,     554.914,  519.45,   25.425,   -450.6,   683.229,  -4737.38,  149.85,   -138.5,   516.857,  -401.4,
      -10.275,    7.5,      -344.049, -54.75,   37.6,     105.8,    -252.315,  -170.1,   -269.683, -54.3167, -44.0229,
      5367.6,     72.3,     -628.6,   -1.50429, -162,     -222.85,  378.133,   133.243,  -1506.3,  755.5,    11.65,
      193.886,    718.2,    74.05,    -181.8,   -25.2,    -4477.5,  213.967,   391.933,  10.8,     1297.35,  -20.15,
      125.05,     15.48,    -416.25,  -239.55,  -85.4667, 246.358,  478.35,    967,      -171.675, 48.9086,  246.15,
      117.833,    40.875,   49.5643,  -270,     -19.4417, 86.125,   -47.3143,  -16.2,    18.5,     -248.175, -290.507,
      -718.2,     116.375,  160.675,  -107.357, 3842.25,  -72.5,    16.125,    -39.2143, -594,     -283.1,   81,
      23.7343,    -30.75,   -67.1667, 205.375,  811.23,   -94.8,    1234.8,    227.558,  -49.5257, 136.8,    29.6375,
      268.775,    6.21,     -2010.38, -247.2,   528.625,  -16.2,    -150.9,    -299.25,  -397.675, -442.8,   -1562.4,
      0,          -518.425, -33.45,   5644.5,   0,        -194.992, 76.2857,   -707.4,   742.2,    16.25,    162.026,
      517.5,      -106.475, 522.625,  206.376,  -45.375,  -54.3,    137.375,   -3.24,    -24.15,   -27.2,    5.25,
      -344.353,   159.525,  -10.2,    -66.375,  -354.086, 6.45,     77.625,    -196.675, -70.2,    -236.925, -192.9,
      320.775,    17.2286,  -123.375, -150,     -4.375,   87.2357,  -524.25,   -348.288, -2.625,   -4.64571, 93.525,
      -21.4958,   1.025});
  using VectorizedNumber = double4;
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> jac{
    reinterpret_cast<const VectorizedNumber*>(jac_arr.get_const_data()), dim, dim, dim};
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> normal_speed_faces{
    reinterpret_cast<const VectorizedNumber*>(normal_speed_faces_arr.get_const_data()), face_size, 2 * dim, face_size};
  gko::batch::multi_vector::batch_item<const VectorizedNumber> src{
    reinterpret_cast<const VectorizedNumber*>(src_arr.get_const_data()), 1, cell_size, 1};
  gko::batch::multi_vector::batch_item<VectorizedNumber> dst{reinterpret_cast<VectorizedNumber*>(dst_arr.get_data()), 1,
                                                             cell_size, 1};
  double time_factor = 1.5;

  dst_arr.fill(0.0);
  face_kernel<::fe_degree, dim><<<1, cell_size>>>(jac, normal_speed_faces, shape_arr.get_const_data(),
                                                  weights_arr.get_const_data(), time_factor, src, dst);
  exec->synchronize();

  for (int i = 0; i < cell_size * 4; ++i) {
    auto result = dst_arr.get_data()[i];
    auto expected = expected_dst_arr.get_data()[i];
    auto eps = (std::abs(result) + std::abs(expected)) * 1e-5;
    EXPECT_NEAR(result, expected, eps) << i / 4 << " " << i % 4;
  }
}

TEST(Operator3d, SimpleCellKernel) {
  using namespace DGAdvection;
  auto exec =
    gko::CudaExecutor::create(0, gko::ReferenceExecutor::create(), std::make_shared<gko::CudaUnifiedAllocator>(0));
  constexpr int face_size = dealii::Utilities::pow(::fe_degree + 1, dim - 1);
  constexpr int cell_size = dealii::Utilities::pow(::fe_degree + 1, dim);
  auto jac_arr =
    gko::array<double>(exec, I<double>{-10, 1, 5, -5, -3, -6, 10, 0,  1, -5, -8, 7,  8,  -1,  -5,  -5, -7, -4,
                                       -8,  6, 1, 7,  5,  -9, 1,  -8, 8, -1, -1, -1, -6, -10, -10, 10, 9,  6});
  auto shape_gradients_arr = gko::array<double>(
    exec, I<double>{1,  3,  -9, 0, -1, -10, 4, -2, -1, 9,  -8, -2, -4, -2, -5, -3, 4,  9,   -10, 0,  7,  1,
                    -8, -2, -8, 7, -2, -9,  2, -3, -5, -2, -4, 2,  -6, -5, 10, -6, 3,  0,   6,   -8, -3, -6,
                    -4, -7, 10, 5, 2,  0,   0, -3, -4, -2, -2, -1, -4, -3, -8, 5,  -2, -10, -3,  1});
  auto speed_cells_arr = gko::array<double>(
    exec,
    I<double>{
      -6,  8,   9,   10,  6,   -6,  0,   -6,  8,   7,   10,  -3,  3,   4,   8,   -8,  4,   10,  10,  -8,  -3,  -10, 1,
      -7,  0,   -7,  6,   -5,  -6,  -3,  5,   -2,  10,  2,   -1,  3,   9,   0,   -1,  -9,  5,   -7,  2,   2,   -3,  -2,
      -6,  7,   0,   3,   0,   -9,  3,   4,   -7,  6,   -6,  -9,  -1,  5,   10,  1,   -10, 3,   -2,  0,   -3,  -3,  7,
      7,   9,   1,   -9,  5,   -5,  2,   -10, -1,  3,   8,   -3,  8,   10,  -1,  0,   8,   -10, -10, 0,   1,   -2,  1,
      3,   2,   3,   -8,  6,   6,   10,  -1,  -9,  -1,  10,  10,  -2,  -4,  10,  6,   -3,  -4,  -9,  9,   3,   9,   5,
      7,   -7,  -1,  -9,  -4,  -5,  -2,  5,   5,   -5,  -1,  6,   3,   0,   6,   7,   -5,  -6,  3,   -6,  -3,  -5,  3,
      -8,  2,   -3,  -8,  1,   -2,  -2,  9,   -10, -8,  -5,  4,   7,   1,   -10, 6,   -2,  -8,  -7,  -2,  -10, -5,  4,
      -6,  -9,  4,   5,   -8,  8,   1,   0,   6,   10,  -4,  5,   -4,  1,   3,   -6,  -2,  2,   2,   -5,  4,   -3,  0,
      0,   -3,  5,   6,   8,   -1,  -1,  -4,  -7,  -1,  5,   10,  7,   3,   0,   -10, 9,   1,   -6,  5,   -3,  -1,  -7,
      -8,  -4,  9,   5,   4,   5,   -3,  -1,  2,   3,   -6,  5,   3,   -7,  -7,  7,   -3,  7,   9,   8,   -3,  -4,  -5,
      1,   6,   6,   7,   -3,  -9,  -6,  -4,  4,   6,   3,   6,   -5,  3,   7,   -10, -2,  -3,  8,   2,   -5,  -4,  10,
      5,   -2,  -1,  -8,  -9,  -6,  7,   5,   -4,  -10, -10, -6,  -5,  2,   9,   -2,  -1,  6,   -9,  10,  -6,  7,   1,
      -6,  -10, 1,   0,   0,   -7,  8,   9,   5,   6,   0,   -4,  -6,  -9,  10,  -6,  9,   8,   1,   7,   -7,  4,   -5,
      9,   -7,  10,  -9,  9,   8,   6,   6,   10,  -5,  -2,  -7,  -8,  -1,  8,   6,   -8,  -6,  8,   -5,  -3,  5,   -4,
      -6,  0,   -8,  -4,  10,  -1,  -6,  3,   6,   1,   4,   -4,  3,   -2,  8,   2,   -1,  7,   -2,  10,  -5,  4,   -10,
      6,   -5,  8,   -3,  -9,  -2,  5,   9,   -4,  7,   -5,  1,   6,   -2,  5,   2,   -10, 3,   7,   5,   -6,  3,   -6,
      3,   -1,  0,   4,   10,  -6,  -8,  5,   5,   -2,  -3,  0,   6,   5,   10,  -6,  3,   7,   -8,  -2,  -3,  -5,  -1,
      -6,  3,   -8,  8,   -8,  0,   4,   7,   -7,  -2,  -9,  4,   -7,  -9,  -5,  -6,  8,   -9,  -9,  -2,  -4,  6,   -2,
      3,   -8,  -7,  9,   8,   7,   -1,  3,   -2,  5,   -7,  -1,  4,   2,   -5,  -2,  4,   -10, 0,   7,   4,   1,   8,
      5,   -1,  -6,  2,   7,   -10, -5,  2,   5,   -2,  4,   6,   10,  8,   8,   -2,  6,   -4,  5,   -7,  -10, 4,   -1,
      10,  -4,  3,   7,   1,   7,   5,   1,   -9,  9,   -1,  2,   5,   -1,  -9,  -6,  8,   8,   3,   -8,  -10, -5,  5,
      -9,  4,   4,   0,   1,   5,   -1,  3,   -9,  6,   9,   2,   10,  4,   -4,  10,  -3,  10,  -9,  -7,  1,   6,   9,
      4,   -4,  10,  2,   -10, -8,  -10, -10, 10,  -5,  6,   8,   -10, -9,  -5,  9,   2,   -5,  -2,  7,   2,   6,   10,
      -8,  0,   -4,  -3,  -10, -9,  -10, 3,   -9,  6,   -10, 5,   1,   -3,  -8,  5,   -4,  -7,  9,   -1,  -1,  -4,  0,
      -9,  -7,  -9,  3,   -5,  -2,  1,   -1,  -4,  -5,  -8,  -1,  -7,  7,   -1,  3,   -1,  6,   1,   -5,  -2,  8,   -3,
      -9,  8,   3,   3,   0,   -7,  4,   -10, 6,   1,   -3,  -6,  -4,  1,   -9,  5,   -10, -4,  -7,  0,   0,   -4,  1,
      -7,  -6,  1,   -1,  0,   7,   -6,  8,   -3,  -9,  -2,  0,   6,   9,   -4,  3,   -1,  6,   7,   4,   -2,  -5,  3,
      10,  -8,  7,   -4,  -3,  9,   1,   -6,  4,   2,   7,   -3,  -7,  -10, -4,  5,   9,   -7,  3,   0,   3,   -3,  -8,
      -9,  -1,  3,   -3,  -1,  10,  -6,  10,  -6,  -6,  9,   -9,  7,   9,   -9,  6,   6,   0,   -9,  -3,  3,   -7,  8,
      -4,  4,   3,   8,   -1,  5,   7,   -9,  8,   -8,  -6,  -10, -9,  5,   10,  -2,  10,  -2,  2,   -7,  -6,  -9,  -3,
      -10, -9,  9,   -4,  9,   10,  -8,  -7,  2,   -8,  3,   -3,  7,   -7,  -10, -1,  -10, -4,  -1,  -5,  -5,  -7,  0,
      2,   -7,  -1,  -9,  -6,  -10, -5,  -7,  6,   -4,  9,   4,   -2,  -4,  5,   -2,  -7,  7,   -7,  -9,  1,   7,   -10,
      -3,  -10, -8,  -9,  -5,  -10, 8,   -6,  -9,  -4,  -6,  4,   -10, 4,   -1,  -9,  -6,  -10, -6,  3,   -10, -6,  7,
      1,   5,   10,  0,   -2,  1,   -8,  -1,  -1,  2,   -6,  -3,  5,   1,   4,   -8,  -8,  0,   6,   9,   5,   -5,  -4,
      2,   -10, -6,  5,   0,   -1,  -3,  4,   -10, -8,  4,   -10, -1,  -4,  0,   -2,  9,   -10, -2,  3,   -5,  6,   4,
      -7,  3,   10,  4,   -6,  6,   -7,  -9,  5,   4,   -10, 4,   -1,  8,   2,   0,   2,   1,   3,   10,  9,   -1,  -8,
      -2,  10,  -8,  0,   -3,  -9,  7,   -6,  1,   -9,  -4,  -3,  -5,  -5,  7,   -8,  -4,  -7,  -6,  -6,  -1,  2,   -10,
      7,   -10, -2,  -2,  -9,  -8,  -3,  5,   9,   2,   6,   -6,  0,   -5,  5,   2,   -2,  7,   4,   4,   10,  -2,  -10,
      -1,  9,   3,   -5,  9,   1,   -6,  -1,  6,   -5,  -10, -10, -3,  -3,  -10, -8,  -4,  8,   -8,  -4,  9,   -9,  5,
      -2,  3,   -9,  1,   -6,  -8,  10,  9,   -9,  3,   -5,  -1,  9,   -6,  -7,  2,   -6,  -1,  3,   9,   -1,  8,   -7,
      5,   -6,  10,  -2,  5,   2,   -5,  -8,  -2,  -6,  -10, -6,  -4,  -3,  -3,  -2,  -10, -5,  3,   7,   -8,  -8,  -2,
      -9,  9,   0,   -8,  9,   7,   -1,  10,  -6,  -2,  -10, -1,  -7,  -5,  -4,  7,   10,  8,   10,  5,   10,  -9,  8,
      -7,  10,  -8,  -4,  6,   -6,  -4,  6,   -2,  7,   2,   0,   1,   -9,  -10, 9,   5,   -2,  10,  1,   3,   9,   8,
      -1,  -2,  1,   -6,  6,   -2,  -10, 2,   -1,  5,   -4,  10,  -4,  6,   3,   7,   2,   -4,  -5,  2,   2,   2,   4,
      8,   -6,  1,   -4,  4,   2,   10,  -4,  0,   0,   10,  -5,  -1,  7,   -4,  5,   -2,  8,   -2,  -10, 10,  3,   9,
      1,   7,   9,   3,   1,   6,   -7,  0,   -5,  -2,  -8,  -3,  -3,  1,   9,   -2,  -7,  -10, -10, -8,  6,   -3,  1,
      -9,  6,   -6,  -7,  9,   1,   7,   -8,  -3,  2,   9,   -6,  -1,  -9,  8,   -9,  -1,  5,   10,  1,   7,   -9,  8,
      6,   2,   6,   -4,  -6,  -2,  -6,  -6,  1,   -3,  -3,  6,   5,   -7,  0,   -5,  7,   10,  -7,  -5,  3,   -8,  8,
      -9,  3,   8,   5,   1,   6,   -2,  5,   -5,  9,   5,   -7,  -9,  -4,  6,   -9,  -7,  1,   -7,  -7,  10,  -3,  -6,
      2,   1,   -10, -3,  1,   7,   4,   -5,  -3,  -10, 6,   6,   -5,  1,   -3,  -7,  -7,  -5,  -2,  10,  10,  -7,  4,
      -1,  -3,  -6,  6,   1,   -1,  -7,  2,   -2,  1,   1,   1,   -4,  -7,  0,   10,  -3,  -10, -10, 0,   9,   -1,  6,
      9,   -8,  -5,  6,   9,   1,   -4,  5,   6,   7,   6,   6,   3,   -9,  -2,  -10, -6,  3,   6,   8,   4,   4,   -3,
      6,   3,   0,   1,   3,   3,   6,   4,   2,   -3,  9,   1,   10,  3,   4,   -9,  -3,  -3,  9,   0,   4,   1,   -1,
      -9,  8,   -9,  -6,  1,   10,  8,   1,   8,   -6,  4,   7,   8,   8,   -8,  6,   3,   8,   8,   2,   7,   -4,  -9,
      -3,  -3,  -7,  8,   -6,  -5,  -4,  4,   -7,  -9,  8,   4,   10,  -7,  -2,  0,   -4,  -9,  -4,  5,   2,   2,   -1,
      9,   1,   4,   -6,  -2,  1,   -9,  5,   1,   -7,  4,   9,   2,   -7,  1,   0,   -4,  -10, -5,  -6,  -6,  -3,  10,
      -5,  -1,  -6,  -3,  0,   -9,  7,   3,   10,  -5,  1,   10,  6,   2,   -6,  3,   -9,  -2,  -2,  0,   7,   -5,  6,
      5,   -10, -8,  -9,  7,   10,  7,   -8,  9,   6,   6,   -9,  -8,  5,   0,   5,   -8,  -2,  7,   8,   -7,  -4,  -10,
      5,   -2,  8,   -10, 10,  8,   7,   1,   9,   9,   -4,  2,   -9,  -5,  7,   3,   -9,  2,   -1,  3,   4,   -9,  7,
      -6,  -3,  -8,  4,   9,   -2,  8,   5,   -3,  10,  -5,  -5,  4,   -3,  -6,  8,   -9,  -9,  6,   -7,  -2,  7,   8,
      7,   9,   7,   9,   10,  5,   2,   -4,  3,   -7,  -10, 9,   0,   -4,  3,   -5,  -8,  -8,  5,   -6,  -2,  10,  7,
      8,   -4,  10,  10,  -8,  1,   -3,  6,   6,   5,   -8,  2,   -9,  3,   -9,  1,   3,   -2,  5,   -7,  -8,  -10, 6,
      -3,  10,  8,   -1,  4,   10,  -4,  -6,  -5,  8,   -5,  -5,  1,   6,   -9,  8,   7,   -10, 3,   -9,  8,   9,   4,
      -8,  9,   -10, 0,   -10, 9,   -2,  7,   5,   5,   -3,  -4,  8,   -10, -9,  -5,  0,   10,  6,   -8,  2,   -2,  3,
      9,   -9,  4,   8,   9,   9,   -5,  -6,  8,   -1,  0,   -5,  4,   2,   8,   -9,  -3,  -8,  -6,  -2,  -7,  0,   1,
      -3,  6,   -1,  8,   -5,  -4,  5,   -5,  -5,  9,   6,   8,   9,   -7,  -9,  -2,  -6,  -5,  -9,  5,   9,   -7,  5,
      2,   -9,  -2,  7,   5,   -1,  4,   -1,  4,   -5,  3,   -6,  -7,  -10, -5,  10,  3,   -1,  -8,  -5,  -7,  5,   -8,
      9,   10,  -4,  0,   -6,  0,   1,   9,   -9,  -6,  3,   -1,  6,   -9,  -4,  7,   -1,  -4,  8,   9,   7,   4,   8,
      -7,  7,   9,   -1,  -8,  4,   2,   4,   4,   8,   -4,  7,   -9,  -3,  0,   10,  9,   -1,  9,   3,   10,  2,   -2,
      8,   6,   -4,  -4,  2,   -7,  7,   5,   -3,  10,  5,   -1,  -1,  10,  -9,  6,   -1,  -9,  1,   3,   7,   -3,  -3,
      -5,  3,   2,   3,   4,   -9,  -10, -7,  -8,  -2,  3,   9,   5,   -10, 9,   0,   -10, 3,   8,   2,   0,   -10, -4,
      -2,  -5,  5,   8,   -6,  3,   0,   9,   -10, 7,   3,   -10, -10, -2,  8,   -6,  6,   -1,  1,   2,   5,   3,   8,
      6,   -5,  4,   -9,  -5,  8,   2,   -5,  -10, 3,   -2,  -3,  2,   -9,  2,   5,   -7,  -7,  7,   -1,  6,   2,   -10,
      -9,  3,   -5,  7,   -4,  2,   4,   -9,  5,   1,   7,   -3,  6,   5,   2,   9,   -7,  7,   -8,  2,   0,   0,   0,
      9,   1,   8,   10,  2,   -6,  -9,  1,   -5,  2,   6,   1,   -4,  0,   -4,  -1,  -5,  -1,  8,   -3,  -10, -1,  -10,
      -7,  -5,  -1,  3,   -1,  6,   9,   10,  2,   8,   -8,  1,   -7,  -10, -8,  9,   1,   -8,  6,   10,  -5,  -9,  -8,
      -6,  -9,  8,   -5,  10,  -7,  3,   -3,  -2,  -8,  -8,  -7,  2,   -6,  2,   -1,  5,   -3,  2,   -4,  8,   -4,  -7,
      -7,  7,   1,   6,   8,   -4,  -4,  -10, -10, 5,   5,   -6,  -6,  -1,  8,   -10, 4,   -5,  1,   4,   1,   -3,  3,
      -4,  -4,  5,   -1,  7,   8,   9,   10,  3,   1,   7,   -1,  -4,  3,   5,   0,   -8,  9,   3,   9,   9,   -10, -1,
      -4,  -9,  -9,  9,   7,   4,   7,   8,   8,   -5,  -1,  9,   0,   8,   9,   -8,  0,   7,   -1,  6,   5,   1,   -3,
      5,   5,   8,   3,   -8,  -4,  1,   0,   4,   5,   -5,  9,   9,   -5,  9,   6,   6,   0,   0,   7,   -7,  3,   7,
      -7,  0,   2,   4,   -1,  4,   9,   8,   2,   -6,  -1,  -9,  0,   4,   -2,  -7,  7,   -3,  -7,  -1,  7,   -8,  -1,
      -8,  -6,  1,   6,   -6,  -1,  -8,  10,  1,   -9,  7,   1,   -8,  0,   -9,  -9,  5,   6,   10,  9,   1,   -2,  -8,
      -6,  -1,  -7,  -6,  4,   8,   6,   1,   -9,  -1,  4,   0,   -5,  10,  3,   5,   -9,  -1,  5,   1,   6,   7,   -2,
      4,   2,   3,   1,   8,   7,   10,  -7,  2,   -6,  -5,  -3,  8,   -9,  -10, 0,   6,   -7,  0,   7,   6,   10,  -1,
      9,   9,   2,   9,   -7,  7,   7,   0,   0,   -2,  5,   10,  -4,  -6,  -8,  -6,  -1,  -3,  -6,  8,   10,  7,   -1,
      4,   -5,  -8,  -3,  9,   -8,  8,   8,   -8,  8,   5,   7,   9,   5,   -3,  -6,  0,   3,   -6,  8,   -2,  9,   -5,
      1,   7,   4,   2,   -10, 4,   -10, -7,  6,   5,   10,  4,   -8,  -3,  -1,  0,   -4,  10,  -7,  2,   7,   -4,  -10,
      2,   -10, -8,  -3,  -7,  2,   -8,  6,   -2,  -10, 7,   -4,  -5,  3,   3,   -7,  0,   0,   -5,  -9,  -8,  0,   -2,
      -3,  -5,  7,   -9,  4,   -9,  1,   6,   5,   7,   6,   5,   -8,  -1,  7,   -8,  -5,  -1,  10,  -8,  5,   -2,  -7,
      2,   -1,  -3,  7,   -3,  -6,  4,   -2,  10,  9,   -5,  1,   -6,  0,   10,  -7,  -6,  4,   1,   7,   -4,  5,   -4,
      -6,  -8,  -10, -3,  -10, 10,  -6,  3,   -10, -9,  4,   3,   -6,  5,   3,   0,   2,   6,   10,  -4,  1,   7,   2,
      1,   -5,  2,   4,   -7,  -4,  5,   -2,  -10, -5,  -7,  -9,  -9,  -5,  -5,  -4,  6,   3,   -9,  -9,  7,   -6,  -7,
      1,   10,  -1,  7,   -7,  2,   -2,  10,  -3,  6,   10,  5,   8,   -6,  -9,  -10, -9,  1,   10,  5,   -3,  -8,  4,
      6,   6,   -6,  5,   -1,  6,   6,   -10, -10, -5,  1,   3,   8,   9,   -4,  1,   0,   5,   3,   9,   -5,  -6,  6,
      -4,  -7,  -10, -8,  -7,  -6,  -9,  3,   3,   6,   4,   -4,  7,   8,   9,   6,   8,   6,   0,   2,   -6,  -2,  -9,
      -4,  -2,  10,  0,   -3,  -8,  -7,  -4,  -8,  2,   10,  3,   8,   -8,  2,   4,   2,   4,   -7,  -8,  4,   -7,  -3,
      0,   -10, -1,  0,   -9,  -3,  9,   -8,  -7,  -7,  10,  9,   -3,  0,   6,   -3,  -3,  -1,  -3,  -10, 1,   9,   3,
      -1,  -7,  -7,  1,   -9,  8,   10,  0,   8,   7,   -3,  -3,  3,   -1,  2,   -3,  6,   7,   2,   -10, -6,  0,   -8,
      -1,  9,   10,  7,   -9,  1,   1,   -9,  8,   -10, 5,   3,   7,   8,   -5,  9,   -3,  -6,  -7,  -8,  0,   4,   -3,
      1,   7,   -6,  -2,  5,   -9,  5,   -2,  -9,  3,   -10, 2,   7,   1,   9,   -4,  -9,  -2,  -2,  -1,  -4,  -1,  -9,
      -3,  -6,  3,   0,   0,   2,   -6,  10,  8,   -1,  -4,  -7,  -9,  3,   4,   7,   6,   -7,  5,   0,   9,   8,   -6,
      -2,  2,   -7,  4,   4,   -10, 5,   7,   -5,  -1,  -1,  -10, 8,   2,   -2,  -2,  0,   10,  -2,  -9,  9,   6,   -3,
      -4,  -5,  3,   -2,  1,   -6,  4,   -1,  -5,  4,   -1,  7,   -4,  1,   1,   -4,  -8,  7,   2,   -8,  2,   10,  9,
      -9,  10,  -1,  8,   0,   -1,  -1,  8,   -5,  -2,  7,   -3,  4,   4,   6,   0,   7,   4,   -6,  3,   8,   2,   3,
      8,   5,   -6,  -5,  -3,  6,   9,   1,   -5,  1,   3,   10,  -9,  5,   -5,  -2,  -8,  7,   -9,  -7,  -8,  10,  1,
      -4,  -5,  3,   -2,  -7,  -8,  10,  4,   -10, -6,  3,   -3,  10,  -7,  0,   -9,  -4,  10,  0,   -10, -2,  7,   8,
      -10, -1,  -6,  -3,  5,   5,   1,   10,  -6,  5,   0,   9,   -8,  -2,  3,   -8,  -10, -10, 10,  -4,  -10, 9,   7,
      -8,  2,   8,   -5,  3,   5,   7,   3,   -7,  -6,  9,   8,   5,   8,   0,   -5,  10,  1,   -10, 2,   3,   -10, 2,
      6,   -9,  -9,  4,   -7,  -4,  8,   -5,  -5,  6,   -9,  0,   -8,  -1,  -8,  6,   9,   -8,  8,   8,   -1,  2,   4,
      -3,  5,   6,   0,   4,   -9,  -4,  3,   2,   0,   3,   8,   2,   5,   5,   -10, 3,   -3,  1,   -2,  10,  3,   -10,
      -7,  -5,  -2,  -6,  -8,  -2,  -1,  3,   -5,  -7,  8,   -6,  -2,  -2,  -10, -10, -9,  -7,  -9,  7,   -3,  9,   9,
      6,   -6,  -3,  -10, -6,  3,   -7,  10,  1,   -9,  4,   -8,  1,   -5,  -7,  -7,  9,   5,   3,   2,   4,   6,   -9,
      -6,  -8,  -7,  7,   -4,  5,   3,   5,   10,  6,   7,   1,   -7,  2,   5,   3,   -3,  4,   -4,  -5,  -2,  4,   -4,
      -4,  2,   8,   -1,  -9,  -6,  -1,  9,   -8,  7,   5,   -9,  8,   -4,  2,   7,   -5,  -3,  3,   3,   -9,  -9,  10,
      -10, 10,  3,   7,   10,  5,   0,   0,   4,   -6,  1,   -2,  -3,  9,   0,   -5,  0,   5,   3,   -6,  6,   8,   -10,
      1,   8,   5,   -3,  -8,  4,   4,   9,   -6,  -8,  -5,  4,   6,   -1,  -9,  -6,  -7,  -2,  6,   0,   -4,  7,   -1,
      -4,  5,   -6,  -4,  -10, 6,   7,   7,   6,   0,   7,   6,   -3,  6,   8,   4,   3,   -4,  6,   -1,  7,   7,   -2,
      10,  10,  -7,  -3,  8,   10,  -1,  -7,  8,   6,   10,  -2,  4,   -6,  9,   10,  1,   -5,  4,   10,  -5,  10,  -10,
      10,  -8,  -2,  3,   4,   5,   7,   6,   4,   -9,  -4,  -4,  -9,  2,   -3,  -9,  1,   -3,  7,   9,   5,   -10, 6,
      9,   -7,  1,   -4,  8,   10,  7,   5,   -3,  3,   -8,  -8,  2,   8,   -4,  -8,  0,   -5,  5,   7,   -2,  -7,  -5,
      -8,  4,   4,   10,  0,   0,   -8,  4,   -7,  4,   2,   -10, 2,   -1,  4,   -7,  3,   -7,  -1,  -3,  9,   -10, -9,
      -4,  -4,  4,   -2,  4,   0,   10,  -2,  -1,  -7,  2,   7,   9,   -3,  -7,  6,   -6,  -1,  -9,  -2,  5,   -3,  8,
      9,   -1,  3,   1,   -7,  -1,  -9,  -1,  -1,  8,   -1,  3,   -10, -8,  -8,  6,   -7,  2,   -1,  0,   -6,  7,   -10,
      3,   -8,  -5,  1,   0,   -1,  -3,  10,  -4,  1,   0,   7,   4,   0,   4,   2,   6,   -5,  3,   -4,  3,   9,   -1,
      -8,  0,   3,   -7,  -9,  10,  1,   0,   8,   9,   -8,  8,   -5,  -1,  -7,  -5,  8,   2,   4,   10,  -4,  -3,  3,
      -4,  -10, 9,   -6,  -1,  -10, 8,   -1,  0,   -5,  7,   -7,  -7,  -8,  8,   7,   3,   -2,  -4,  -5,  4,   -10, -2,
      -2,  0,   3,   0,   2,   -3,  1,   -10, 9,   6,   -6,  -8,  7,   5,   4,   -6,  6,   9,   1,   -1,  4,   -3,  -9,
      -7,  5,   -3,  -3,  5,   -9,  2,   6,   -7,  -6,  -7,  5,   4,   2,   2,   2,   7,   -6,  7,   -5,  10,  -2,  -9,
      10,  8,   -9,  1,   -7,  -9,  -2,  0,   -6,  -4,  0,   1,   4,   6,   -6,  7,   -3,  10,  5,   -7,  5,   -9,  -3,
      -4,  4,   2,   5,   -9,  -2,  8,   -4,  -5,  -2,  -2,  3,   1,   -4,  -5,  -10, -8,  -8,  9,   -3,  -7,  3,   0,
      -5,  -7,  -9,  -8,  -2,  -9,  -10, -1,  -8,  0,   -9,  -1,  -5,  -7,  -1,  10,  -9,  3,   -3,  -1,  7,   -5,  -8,
      1,   5,   6,   -6,  2,   -3,  -7,  0,   -2,  10,  5,   7,   5,   2,   8,   -3,  -7,  -10, 6,   -1,  2,   -5,  -2,
      7,   -3,  -4,  2,   -5,  8,   6,   -7,  -2,  6,   -10, 3,   3,   9,   -2,  3,   3,   3,   -10, 6,   -6,  9,   1,
      3,   5,   3,   -5,  0,   -7,  -2,  -10, 7,   3,   1,   9,   -9,  -2,  -4,  -8,  7,   8,   -3,  -1,  1,   6,   -2,
      5,   -4,  -9,  -3,  -1,  0,   6,   4,   6,   5,   -5,  3,   -5,  3,   -10, 8,   -5,  -6,  -10, 9,   -8,  -9,  7,
      -2,  2,   1,   10,  -7,  9,   0,   -7,  -3,  10,  9,   8,   1,   -2,  5,   10,  -8,  5,   3,   -8,  -6,  0,   3,
      8,   -2,  -6,  1,   -5,  -6,  -2,  2,   -10, -9,  -9,  -5,  -10, 1,   1,   5,   5,   -4,  -6,  3,   9,   4,   -5,
      -2,  0,   10,  -3,  -5,  -1,  8,   7,   10,  1,   6,   10,  -5,  -5,  -1,  2,   -8,  -7,  3,   -5,  -8,  -4,  -9,
      8,   -3,  -1,  -9,  -10, 0,   -2,  -2,  -7,  4,   5,   1,   -9,  -4,  -5,  -10, -5,  -4,  -4,  9,   -8,  8,   6,
      -6,  -5,  9,   -9,  -2,  -8,  -3,  5,   9,   -1,  -3,  -9,  1,   3,   6,   6,   9,   7,   7,   7,   -6,  5,   -1,
      1,   -1,  9,   5,   -1,  8,   3,   -8,  2,   5,   -8,  -4,  7,   4,   9,   -2,  3,   -5,  -9,  -4,  -6,  -5,  -4,
      2,   -5,  -3,  -1,  7,   8,   -5,  6,   -7,  6,   -9,  -9,  -8,  -1,  10,  3,   9,   -4,  6,   -10, -9,  -10, 7,
      -2,  2,   -4,  7,   8,   -9,  -8,  -6,  -10, -3,  10,  -1,  3,   9,   -8,  1,   10,  5,   8,   -4,  -7,  10,  5,
      2,   8,   -9,  4,   4,   -3,  -4,  -3,  -2,  5,   -7,  -7,  8,   -4,  -8,  9,   0,   -5,  5,   1,   8,   -5,  -2,
      -7,  -1,  0,   -7,  -7,  -2,  0,   10,  3,   3,   -5,  -4,  -2,  1,   6,   9,   7,   8,   -9,  -3,  -8,  -9,  9,
      10,  -8,  -10, -9,  -7,  -7,  -7,  4,   -2,  -1,  -7,  3,   2,   6,   5,   1,   -8,  9,   3,   -1,  -1,  1,   3,
      8,   -3,  -9,  1,   -1,  9,   3,   4,   -8,  5,   -10, 6,   -8,  3,   6,   9,   3,   3,   -5,  -2,  -3,  1,   -9,
      -7,  -3,  -6,  -1,  10,  1,   1,   10,  9,   6,   9,   -1,  0,   -5,  9,   -6,  -6,  -7,  -1,  -1,  2,   -2,  -5,
      -7,  4,   -1,  10,  -1,  7,   0,   1,   6,   -5,  7,   -10, -3,  -7,  0,   -4,  -1,  -9,  -6,  -5,  -7,  -2,  -5,
      7,   -8,  3,   6,   10,  -5,  -4,  -9,  -1,  -5,  -7,  4,   4,   -3,  -9,  -2,  -1,  7,   8,   -4,  -4,  6,   10,
      -4,  5,   3,   -8,  -5,  5,   -8,  -6,  1,   -2,  8,   2,   -2,  6,   -7,  3,   -7,  -1,  3,   7,   -6,  -10, -3,
      5,   2,   -5,  -8,  -2,  -4,  -7,  3,   4,   5,   -10, -6,  7,   1,   -4,  2,   1,   -9,  8,   8,   7,   -5,  2,
      8,   -1,  3,   -5,  -8,  5,   -1,  -8,  4,   -5,  -1,  -4,  10,  4,   6,   8,   -10, -10, 8,   6,   7,   3,   -1,
      0,   5,   10,  10,  8,   -9,  4,   2,   4,   3,   6,   -9,  5,   -10, -4,  -1,  3,   -1,  -3,  10,  -8,  6,   4,
      9,   -1,  2,   -4,  6,   -9,  9,   -5,  -1,  4,   8,   -1,  0,   6,   -3,  -1,  9,   -3,  -1,  -4,  -5,  -9,  8,
      -4,  9,   10,  8,   3,   3,   10,  4,   -7,  0,   0,   6,   5,   -1,  -1,  -8,  -4,  10,  9,   2,   2,   5,   2,
      10,  -5,  -1,  -6,  4,   -1,  0,   -4,  -8,  9,   -6,  2,   -6,  -6,  6,   -1,  4,   -4,  9,   -8,  -10, -10, 3,
      3,   -9,  2,   -7,  4,   -10, -6,  3,   2,   2,   2,   -10, -10, 1,   3,   0,   7,   -1,  -9,  -9,  9,   -3,  9,
      -2,  -7,  6,   2,   2,   9,   5,   -6,  7,   -3,  9,   -4,  1,   9,   10,  -8,  10,  -5,  9,   -3,  3,   8,   -3,
      -3,  10,  9,   8,   -5,  9,   1,   -8,  -1,  -7,  0,   5,   3,   -7,  -9,  -8,  0,   9,   -2,  2,   4,   2,   2,
      -6,  -8,  3,   -6,  2,   -2,  -7,  0,   6,   -4,  -10, 3,   -3,  7,   -8,  -4,  -7,  -2,  5,   -9,  -2,  -1,  -1,
      6,   10,  4,   -8,  8,   5,   3,   -1,  -2,  2,   -10, -1,  7,   8,   -10, 4,   3,   2,   -9,  -9,  2,   -4,  -8,
      0,   -10, 2,   -1,  -3,  3,   10,  3,   6,   6,   -9,  -7,  3,   -6,  4,   -9,  7,   -9,  -3,  10,  9,   -9,  -3,
      -1,  0,   -5,  -3,  4,   -6,  -9,  9,   3,   3,   -4,  6,   3,   3,   -8,  8,   -6,  -8,  -5,  10,  4,   -2,  -8,
      1,   -4,  -5,  -10, 10,  -2,  -6,  8,   4,   -2,  2,   5,   6,   8,   3,   9,   -2,  -2,  -10, 3,   2,   3,   -3,
      4,   -2,  5,   -7,  1,   4,   -6,  -1,  0,   8,   -8,  0,   5,   -1,  7,   0,   -5,  -3,  -5,  -9,  -5,  1,   -2,
      -4,  -2,  -4,  1,   -1,  3,   -6,  5,   -9,  7,   -9,  5,   -9,  -5,  -6,  -7,  -7,  7,   10,  -8,  1,   0,   -4,
      -6,  0,   -7,  -3,  3,   4,   -3,  -10, -10, -4,  -5,  3,   -8,  10,  6,   8,   6,   -7,  10,  -8,  10,  1,   0,
      9,   1,   10,  -5,  -5,  4,   -10, -3,  10,  5,   2,   9,   2,   3,   0,   3,   3,   -10, -4,  3,   -1,  -9,  8,
      5,   -8,  -2,  -4,  1,   2,   -7,  7,   9,   -3,  -1,  6,   -10, -3,  -6,  9,   8,   -4,  -9,  -6,  -7,  9,   -3,
      -4,  1,   1,   -1,  -3,  -10, 1,   -10, -9,  -10, 9,   -2,  0,   5,   6,   3,   5,   9,   -1,  0,   5,   3,   3,
      1,   -3,  4,   4,   9,   -5,  -7,  8,   5,   6,   -7,  0,   -6,  7,   9,   2,   -10, -3,  7,   -10, -1,  2,   -9,
      5,   -7,  0,   -8,  6,   -7,  -1,  -2,  -8,  -6,  -9,  8,   4,   -9,  -10, -4,  1,   6,   -4,  -2,  3,   9,   -9,
      0,   0,   -4,  3,   -4,  4,   4,   -3,  -2,  1,   2,   -5,  -10, -9,  -9,  -10, 1,   -2,  7,   -8,  -2,  -8,  -3,
      -5,  9,   -6,  -4,  -1,  -9,  -3,  8,   10,  -5,  3,   0,   -9,  -5,  8,   1,   -8,  -10, 6,   2,   -3,  0,   5,
      -4,  6,   10,  -1,  -2,  0,   -2,  -2,  -1,  1,   8,   -4,  -3,  4,   8,   -9,  -7,  3,   4,   -5,  -9,  9,   0,
      -8,  -5,  7,   8,   -4,  -10, -4,  2,   -5,  7,   7,   -6,  -6,  7,   -1,  4,   5,   -2,  4,   4,   -5,  -6,  -2,
      -7,  6,   -10, 7,   10,  5,   -9,  1,   10,  0,   4,   -8,  -2,  1,   0,   6,   6,   -1,  -10, 0,   -9,  -1,  -7,
      9,   9,   1,   5,   -5,  -7,  9,   5,   3,   -10, -8,  -4,  10,  -8,  -5,  6,   4,   -10, -10, 6,   -5,  -3,  9,
      0,   2,   9,   -10, -5,  7,   -6,  5,   -3,  -4,  -7,  -10, 0,   6,   7,   7,   -5,  -7,  -2,  -9,  -5,  -8,  6,
      9,   -3,  3,   -6,  2,   2,   -1,  10,  -4,  1,   -2,  9,   -3,  3,   1,   8,   2,   7,   0,   9,   -4,  -2,  -8,
      -3,  4,   7,   5,   -2,  3,   -7,  3,   6,   -5,  -2,  -3,  8,   3,   -7,  -4,  -6,  1,   5,   -4,  -5,  0,   7,
      -10, 4,   4,   0,   1,   -7,  -10, 2,   -9,  -7,  -10, 4,   -8,  10,  3,   -7,  8,   9,   -4,  8,   5,   8,   1,
      -3,  0,   -3,  7,   -2,  -4,  -1,  -4,  9,   8,   -4,  10,  5,   3,   9,   -9,  -8,  1,   6,   -1,  -6,  3,   -1,
      10,  5,   -1,  4,   -3,  -4,  -2,  3,   -6,  -10, -6,  2,   10,  -4,  -1,  -4,  10,  5,   10,  3,   8,   2,   -10,
      10,  -3,  -10, -3,  3,   -4,  6,   -7,  10,  9,   8,   -5,  -4,  -2,  7,   -9,  2,   10,  -9,  8,   3,   -10, 0,
      -6,  7,   7,   -1,  -4,  4,   1,   10,  7,   -9,  -1,  7,   -7,  -5,  -6,  -2,  1,   -9,  -9,  -2,  6,   -6,  6,
      8,   -9,  2,   10,  5,   -2,  10,  -5,  5,   1,   -2,  1,   -8,  1,   -6,  -1,  7,   7,   6,   10,  1,   -7,  -7,
      7,   6,   -7,  -7,  -4,  -7,  2,   -5,  3,   10,  -6,  10,  1,   -9,  1,   -1,  1,   0,   2,   -7,  10,  6,   -10,
      -8,  -2,  9,   9,   8,   -10, 7,   -2,  5,   2,   6,   1,   -7,  -2,  -6,  6,   9,   -6,  0,   10,  1,   4,   0,
      10,  5,   -10, -1,  1,   4,   4,   -7,  8,   3,   -10, 7,   -10, 2,   6,   -6,  4,   10,  -2,  -1,  7,   -6,  9,
      7,   0,   -1,  8,   -9,  0,   4,   6,   -3,  3,   5,   9,   7,   3,   7,   0,   2,   4,   1,   -2,  9,   -4,  7,
      1,   -2,  -6,  6,   2,   -1,  -8,  -9,  1,   7,   3,   8,   3,   -3,  -2,  -1,  -9,  9,   9,   -10, -3,  -3,  6,
      -10, -6,  9,   -10, -9,  -5,  2,   6,   2,   -6,  5,   0,   2,   5,   -7,  -9,  3,   3,   1,   -2,  4,   -7,  -10,
      -10, 1,   5,   1,   -9,  -8,  -9,  -8,  2,   -5,  2,   5,   -8,  -5,  5,   -9,  -7,  7,   8,   6,   8,   -6,  3,
      -7,  -10, -8,  -5,  4,   5,   0,   -4,  2,   -2,  6,   -9,  -10, 8,   -2,  3,   8,   -10, -1,  -3,  5,   10,  5,
      -2,  -5,  10,  -5,  -10, -9,  1,   5,   -9,  -5,  -3,  0,   -4,  -9,  4,   -1,  9,   -9,  5,   -8,  1,   -3,  -6,
      -8,  -1,  -9,  2,   -3,  6,   -2,  2,   -2,  -9,  -1,  6,   7,   -9,  6,   10,  -7,  3,   10,  3,   6,   4,   -2,
      -5,  -8,  4,   -6,  6,   -6,  4,   -2,  -6,  -2,  -4,  4,   6,   10,  -1,  3,   5,   2,   -2,  8,   -6,  3,   -10,
      4,   2,   -4,  -10, 6,   1,   2,   -6,  -7,  2,   -2,  2,   0,   2,   -10, 2,   -10, 9,   10,  -9,  5,   9,   9,
      -7,  7,   -10, 5,   -3,  2,   -2,  -2,  10,  -1,  -10, -1,  0,   9,   -2,  -9,  -10, -1,  2,   -7,  -9,  -10, -10,
      2,   0,   4,   -2,  0,   -2,  10,  -1,  -9,  8,   4,   -8,  -1,  -6,  6,   2,   2,   9,   -4,  4,   3,   -7,  0,
      8,   9,   5,   10,  -6,  -10, -3,  -3,  9,   6,   -10, -3,  -7,  7,   10,  -9,  -7,  3,   6,   -6,  0,   8,   7,
      4,   -6,  -6,  7,   5,   8,   9,   -9,  1,   1,   5,   2,   1,   -1,  -5,  9,   -2,  -7,  -5,  5,   9,   9,   2,
      1,   -10, -10, 8,   -7,  -8,  8,   -3,  1,   0,   4,   -1,  -7,  -2,  -2,  6,   8,   1,   3,   0,   8,   7,   -9,
      -1,  0,   -4,  -5,  -4,  6,   10,  -2,  -4,  -6,  -9,  3,   0,   3,   0,   -4,  -3,  1,   3,   -5,  -5,  9,   4,
      3,   -9,  -5,  8,   5,   -9,  -2,  -2,  9,   -8,  3,   -10, -5,  1,   9,   -8,  -4,  7,   -3,  -1,  -1,  -5,  -8,
      -2,  9,   5,   10,  4,   -10, -7,  2,   -10, 9,   -9,  0,   -9,  -8,  -10, 6,   -6,  -5,  -6,  -8,  -1,  -3,  -7,
      4,   -5,  7,   -6,  -8,  -5,  -3,  -3,  2,   -7,  6,   0,   -4,  1,   10,  -2,  9,   9,   -8,  -3,  -9,  10,  -5,
      4,   5,   8,   3,   1,   9,   6,   8,   -4,  7,   5,   7,   -2,  8,   -7,  3,   -3,  -8,  -4,  -5,  -6,  -10, 7,
      4,   -3,  -3,  7,   6,   -8,  2,   -3,  0,   -10, 2,   -9,  7,   6,   7,   3,   -1,  8,   6,   9,   4,   -7,  -8,
      6,   -2,  2,   -5,  -1,  -4,  -3,  -8,  3,   8,   4,   7,   5,   3,   3,   10,  -6,  -8,  4,   -2,  2,   -8,  7,
      2,   5,   -4,  -5,  2,   -8,  -2,  -3,  -4,  -4,  -7,  -2,  7,   -6,  -9,  5,   -6,  -2,  -9,  9,   6,   -10, -8,
      0,   3,   10,  6,   10,  -4,  -5,  9,   8,   5,   7,   -4,  4,   -7,  0,   0,   4,   5,   2,   -5,  -2,  6,   -5,
      -5,  1,   -1,  4,   0,   1,   9,   8,   -1,  8,   -7,  7,   1,   -10, 7,   -4,  -6,  -10, -10, 5,   6,   -9,  3,
      -1,  4,   6,   -8,  3,   -1,  7,   8,   -9,  8,   -4,  5,   3,   -6,  -10, 5,   -3,  -5,  4,   5,   -6,  -1,  3,
      -5,  -8,  2,   2,   -5,  -7,  8,   -1,  6,   9,   -7,  4,   -9,  2,   -3,  8,   3,   -1,  -6,  4,   7,   0,   5,
      0,   -3,  6,   4,   -1,  7,   1,   0,   -10, -7,  9,   4,   -9,  0,   -7,  -6,  3,   1,   8,   1,   -4,  -4,  2,
      -2,  8,   -4,  10,  4,   10,  9,   0,   -3,  4,   -10, 2,   -8,  5,   -1,  -8,  -8,  6,   -1,  -4,  4,   5,   -3,
      10,  7,   5,   1,   -1,  -7,  10,  3,   7,   5,   0,   -8,  -4,  4,   -7,  9,   6,   -1,  -1,  1,   -9,  -1,  -3,
      2,   -9,  -6,  9,   -6,  10,  10,  6,   0,   -3,  -8,  6,   -10, -5,  2,   4,   8,   9,   -3,  -10, -6,  5,   -3,
      10,  7,   7,   0,   7,   -6,  -10, -6,  -8,  -3,  -10, -3,  -2,  -7,  7,   8,   -3,  -10, 9,   -1,  -4,  -8,  4,
      1,   -10, -2,  -3,  3,   -5,  -3,  3,   2,   -8,  -8,  5,   -6,  2,   4,   -1,  5,   9,   -10, 9,   -10, -3,  6,
      8,   8,   8,   -6,  5,   -4,  0,   9,   -5,  -1,  6,   9,   -9,  -4,  -4,  5,   -4,  7,   7,   4,   -10, -1,  6,
      -2,  -7,  5,   0,   5,   -9,  -1,  9,   7,   8,   0,   -2,  -1,  10,  2,   9,   -1,  -10, -3,  -9,  -2,  -9,  -10,
      6,   4,   4,   -6,  -4,  -5,  9,   -10, 7,   -4,  6,   0,   -4,  5,   -2,  -7,  0,   -4,  1,   -4,  3,   4,   -6,
      -1,  2,   3,   -10, -9,  8,   7,   0,   10,  2,   -2,  2,   10,  9,   -6,  -6,  3,   -9,  -6,  -9,  10,  -7,  8,
      -9,  9,   2,   9,   8,   6,   -3,  1,   7,   -9,  9,   -3,  6,   4,   0,   10,  0,   4,   -4,  3,   10,  -3,  8,
      2,   0,   9,   10,  -4,  9,   8,   -7,  6,   -4,  10,  5,   4,   -7,  5,   -10, -8,  1,   -3,  -1,  3,   -4,  4,
      -7,  -10, -1,  -5,  2,   6,   1,   0,   6,   10,  -6,  9,   1,   8,   -8,  -5,  -2,  3,   9,   -10, -6,  -5,  -2,
      -9,  -2,  0,   3,   6,   -10, 10,  -8,  5,   8,   -8,  0,   -2,  10,  4,   3,   0,   7,   9,   -10, 3,   -3,  -5,
      6,   -7,  -7,  9,   -3,  -4,  10,  -4,  -5,  -8,  10,  -4,  10,  -3,  -7,  8,   9,   -3,  -5,  0,   3,   4,   0,
      -8,  6,   5,   -10, -10, -10, 6,   -9,  10,  0,   9,   3,   -2,  -1,  10,  -7,  10,  -1,  -7,  10,  -2,  2,   -5,
      -6,  -8,  -10, -8,  2,   -6,  -1,  1,   -4,  1,   2,   7,   1,   9,   10,  6,   -7,  -6,  -6,  -1,  -10, 8,   0,
      4,   -3,  7,   9,   4,   0,   6,   -1,  3,   -7,  1,   -3,  4,   4,   -9,  9,   8,   -4,  -6,  -9,  -9,  -9,  1,
      -10, -2,  -8,  4,   10,  -5,  0,   -2,  -10, -7,  -3,  9,   -7,  9,   -4,  -8,  2,   -3,  -1,  2,   -9,  9,   5,
      -2,  -2,  -2,  3,   2,   1,   5,   -3,  -8,  0,   9,   4,   8,   -1,  -10, -8,  -6,  -9,  8,   10,  8,   -2,  -4,
      -8,  -1,  -8,  -5,  -2,  -10, -3,  2,   3,   5,   5,   -3,  -1,  2,   -1,  -10, -10, 3,   -7,  7,   9,   0,   3,
      9,   -4,  -7,  0,   -5,  7,   -4,  -9,  0,   -7,  -2,  3,   -5,  -3,  10,  3,   6,   0,   10,  1,   4,   -8,  -9,
      -8,  6,   8,   -10, -2,  -9,  -2,  -2,  -8,  -6,  3,   7,   -2,  -1,  -9,  3,   -8,  4,   -2,  1,   -6,  6,   6,
      10,  -5,  9,   6,   2,   6,   7,   -9,  -6,  10,  -2,  3,   -6,  -2,  6,   -1,  -7,  4,   -8,  10,  -2,  2,   -4,
      10,  -4,  -2,  -6,  8,   -7,  10,  -1,  7,   5,   -1,  -8,  -1,  5,   -8,  -6,  10,  0,   7,   8,   -6,  -6,  -4,
      -5,  4,   -1,  -8,  1,   -10, 6,   -4,  0,   7,   10,  4,   7,   0,   3,   7,   -2,  2,   -3,  -1,  -7,  -10, -8,
      -7,  8,   1,   -5,  6,   0,   7,   -3,  9,   -4,  0,   -6,  1,   2,   0,   -6,  -8,  0,   -10, 8,   3,   8,   1,
      0,   2,   -5,  -7,  6,   -2,  5,   -2,  3,   8,   3,   -2,  -2,  -6,  7,   -1,  3,   3,   -9,  6,   -6,  -2,  -5,
      -5,  -10, -6,  10,  -7,  -9,  -3,  3,   -8,  -2,  2,   2,   3,   -1,  6,   9,   4,   0,   8,   9,   -1,  -10, 10,
      3,   -6,  8,   2,   0,   -2,  6,   -2,  10,  -8,  -3,  4,   1,   2,   -10, 1,   9,   4,   6,   -9,  -3,  -7,  -8,
      -8,  -4,  -4,  -1,  -5,  -3,  6,   -3,  -2,  0,   -1,  8,   1,   -2,  -3,  -4,  -6,  -3,  -2,  0,   -4,  -9,  -1,
      8,   3,   -7,  1,   3,   -4,  1,   0,   -4,  -10, 9,   5,   1,   1,   9,   -2,  2,   10,  -7,  -7,  -7,  -8,  5,
      5,   6,   -1,  -5,  0,   -10, 5,   -9,  9,   8,   -7,  6,   4,   -3,  7,   4,   -2,  1,   2,   6,   -6,  6,   7,
      0,   -1,  4,   -1,  -9,  -3,  5,   10,  -4,  3,   1,   0,   4,   -3,  -5,  5,   -7,  5,   8,   1,   3,   0,   -9,
      10,  -6,  -10, -8,  3,   9,   -7,  10,  6,   7,   2,   5,   -8,  -2,  2,   -7,  4,   -10, -6,  -5,  8,   6,   -10,
      4,   8,   -9,  8,   1,   7,   6,   1,   -6,  2,   2,   10,  -7,  3,   1,   10,  -5,  -5,  -6,  7,   -10, 10,  5,
      8,   0,   -9,  8,   6,   6,   8,   -9,  3,   -10, 10,  -3,  -8,  9,   2,   -1,  -1,  -8,  -8,  -7,  9,   3,   -3,
      2,   3,   -8,  6,   5,   3,   0,   8,   5,   -8,  1,   -8,  -9,  2,   -7,  -10, -6,  0,   -1,  -1,  0,   3,   2,
      10,  -1,  -1,  -2,  -10, 7,   -3,  -1,  2,   -2,  10,  7,   -3,  -9,  -9,  2,   -10, 5,   7,   10,  -8,  -3,  8,
      -6,  4,   5,   6,   -8,  -1,  7,   -2,  -3,  0,   3,   9,   -2,  -8,  -4,  3,   8,   5,   2,   -3,  9,   0,   8,
      -7,  -6,  6,   7,   -10, 2,   4,   -7,  -2,  2,   -8,  -7,  9,   -2,  -3,  4,   10,  3,   4,   -4,  3,   4,   6,
      4,   0,   0,   -2,  3,   4,   0,   10,  7,   1,   -9,  -5,  -5,  0,   -10, -10, -6,  -10, -9,  -6,  -5,  -9,  9,
      -3,  -7,  10,  -2,  -1,  -8,  3,   -1,  -9,  -6,  6,   3,   -2,  7,   -9,  -2,  -7,  3,   2,   5,   -2,  -5,  -9,
      3,   -3,  -4,  3,   7,   9,   -5,  -1,  -7,  0,   7,   -7,  1,   -5,  -8,  -8,  2,   10,  -8,  -8,  10,  9,   -9,
      9,   -5,  0,   9,   -7,  8,   -4,  7,   3,   -10, 3,   7,   -10, 6,   -2,  -10, 8,   4,   8,   5,   -5,  -2,  -10,
      9,   6,   6,   -7,  -8,  0,   -9,  9,   9,   -6,  -7,  5,   6,   5,   -10, 1,   1,   -5,  2,   -8,  4,   -8,  -9,
      -10, -7,  5,   7,   5,   -5,  7,   5,   -1,  -1,  4,   1,   -8,  8,   -5,  2,   5,   10,  -9,  -8,  3,   -4,  7,
      -9,  -8,  -5,  -3,  -3,  -7,  -6,  1,   -3,  5,   -6,  -7,  -10, 10,  9,   -5,  -5,  10,  -7,  -5,  4,   3,   -1,
      2,   -5,  -8,  7,   -5,  9,   3,   -9,  10,  0,   -9,  5,   5,   -10, 4,   4,   2,   -8,  6,   7,   5,   -9,  -8,
      3,   4,   -3

    });
  auto weights_arr = gko::array<double>(
    exec,
    I<double>{-7,  -3,  10,  -5, 4,   -7,  -2,  10,  8,  -6, 9,  7,   -3,  -9, -10, -7,  0,   -8,  -1,  -5,  1,  0,  10,
              -8,  -1,  -3,  2,  6,   -9,  8,   6,   6,  6,  -5, -9,  -10, 9,  -10, 4,   -9,  7,   4,   5,   -4, -2, -3,
              9,   5,   0,   1,  -7,  -8,  -10, -10, -5, 1,  -9, -2,  -5,  -5, -10, 7,   2,   6,   10,  3,   6,  -4, -8,
              6,   -2,  4,   -3, 5,   -8,  0,   7,   -9, 1,  -4, 10,  -3,  -8, 3,   6,   -5,  -2,  -9,  3,   -1, -9, 9,
              -9,  -6,  -6,  -2, 9,   1,   -3,  -2,  9,  6,  0,  6,   -7,  -5, -3,  4,   5,   3,   8,   7,   -1, -3, 0,
              -7,  -6,  -10, 4,  3,   8,   3,   6,   -8, -4, -8, -9,  5,   3,  -10, -6,  -4,  3,   5,   -1,  7,  0,  -4,
              3,   -3,  0,   -5, -4,  7,   -5,  0,   -6, -9, -4, -10, 10,  7,  5,   -10, 4,   0,   2,   8,   -2, 4,  7,
              -7,  0,   7,   3,  2,   -7,  -6,  -2,  -9, 7,  -5, -3,  2,   0,  -1,  3,   7,   -7,  0,   -5,  -4, -4, -3,
              1,   2,   3,   -2, -6,  2,   -8,  -9,  9,  -7, -1, 3,   2,   6,  4,   10,  3,   7,   9,   3,   5,  -5, 9,
              3,   -8,  -8,  0,  -3,  10,  -10, -2,  4,  -6, 3,  -6,  7,   -1, 7,   -1,  -4,  -6,  7,   1,   10, 9,  -3,
              6,   9,   -8,  -6, -1,  -5,  7,   2,   1,  -6, -7, -4,  9,   0,  6,   2,   -10, 9,   6,   -6,  -3, -2, 5,
              -10, 0,   -8,  -7, 0,   5,   7,   3,   4,  -4, -6, 10,  -7,  7,  5,   -6,  -5,  -8,  -8,  9,   -1, -8, -4,
              1,   4,   9,   2,  -6,  0,   9,   5,   9,  -3, 3,  4,   1,   -3, -2,  6,   5,   -10, 10,  7,   0,  -9, -1,
              2,   0,   8,   5,  -8,  9,   -5,  -4,  9,  -1, 7,  6,   -5,  -3, -6,  8,   8,   -4,  4,   9,   1,  7,  -4,
              -4,  -1,  8,   7,  4,   -10, 9,   6,   -5, 6,  8,  -10, 4,   10, 8,   -3,  -2,  0,   1,   -3,  4,  -8, 5,
              2,   -3,  -6,  0,  -5,  9,   3,   -6,  4,  6,  -9, 0,   -8,  -3, 6,   -5,  -8,  -7,  -10, -10, 7,  -9, -6,
              9,   5,   -2,  4,  5,   -8,  6,   -4,  -3, 10, 0,  -1,  5,   -6, -6,  -10, 7,   -6,  4,   -1,  -2, -9, -9,
              -9,  9,   4,   7,  1,   1,   -9,  0,   0,  3,  5,  6,   -3,  -6, 0,   10,  -4,  -1,  -9,  -8,  -9, 1,  7,
              5,   -1,  10,  9,  -5,  -10, 3,   -5,  -2, -3, 3,  -10, 7,   2,  -6,  -1,  8,   5,   -2,  3,   0,  9,  -2,
              -6,  -2,  -4,  -5, -10, 8,   3,   -9,  3,  10, -1, 2,   8,   -2, 3,   -5,  -1,  5,   6,   9,   -7, 7,  -7,
              -4,  -10, 7,   0,  1,   0,   -5,  8,   10, -1, 6,  -3,  5,   -8, -3,  7,   -10, 7,   -2,  3,   -7, -3, 0,
              0,   1,   2,   2,  2,   5,   3,   -6,  4,  8,  10, -1,  10,  1,  10,  9,   -3,  8,   7,   1,   -8, 10, -1,
              -3,  0,   10,  7,  10,  6

    });
  auto src_arr = gko::array<double>(
    exec,
    I<double>{
      2,   -4,  -6,  -9,  1,   10,  3,   -10, 8,   6,   6,   5,   4,  -10, 1,   1,   -1,  9,   -10, -1,  0,   2,   8,
      -8,  6,   -2,  2,   -9,  -10, -6,  9,   9,   2,   7,   6,   -2, 6,   -2,  -7,  -7,  -6,  8,   8,   -6,  4,   -9,
      -4,  -4,  9,   3,   -7,  -4,  6,   -2,  0,   -5,  6,   -4,  -6, -5,  -2,  10,  6,   -5,  -10, 9,   -9,  -3,  5,
      -8,  -8,  -2,  9,   -2,  -7,  2,   -9,  3,   8,   -1,  -10, -3, 6,   -10, 5,   2,   -6,  6,   -3,  -3,  -8,  8,
      -4,  4,   -9,  -5,  6,   9,   4,   -5,  -5,  -10, 8,   6,   3,  3,   -8,  -3,  -7,  4,   -7,  -5,  8,   -1,  7,
      8,   2,   6,   7,   5,   -6,  -2,  2,   2,   10,  1,   1,   -9, -1,  5,   7,   -8,  7,   -8,  -10, 0,   -5,  3,
      -9,  -5,  -2,  -5,  7,   -8,  2,   4,   -4,  -2,  -6,  -1,  2,  2,   7,   4,   -8,  -2,  -10, -6,  -8,  10,  -2,
      -3,  -8,  9,   9,   -10, 6,   -6,  3,   -7,  -3,  3,   -6,  4,  5,   1,   -8,  -6,  -5,  -2,  -5,  -10, 1,   -5,
      -3,  -1,  -9,  -7,  -6,  5,   6,   2,   -8,  -7,  -4,  1,   2,  1,   -3,  9,   9,   4,   1,   -8,  2,   -10, 8,
      10,  0,   7,   -6,  -8,  10,  4,   5,   -9,  6,   -9,  -3,  10, -2,  -7,  -4,  -4,  -2,  -7,  -1,  -10, 2,   6,
      -8,  -7,  -7,  3,   -10, 9,   1,   0,   8,   3,   -9,  -3,  -8, 7,   2,   -8,  -5,  7,   -2,  -4,  -4,  0,   2,
      2,   10,  5,   1,   8,   5,   -8,  -10, 9,   -2,  -4,  7,   2,  -10, 9,   1,   -9,  -4,  0,   -4,  -2,  -4,  2,
      2,   7,   -1,  1,   -4,  5,   -3,  -6,  -5,  1,   -10, 8,   -4, 0,   -10, 0,   -3,  -8,  -4,  -3,  3,   2,   4,
      6,   -8,  -10, -6,  -7,  -2,  -7,  -8,  7,   3,   4,   -2,  -3, -8,  -4,  -10, -3,  3,   8,   3,   9,   4,   10,
      5,   -4,  3,   10,  -4,  8,   -10, -2,  5,   -8,  -2,  4,   3,  9,   8,   -6,  4,   1,   3,   -4,  -8,  5,   -3,
      0,   7,   -3,  -8,  -9,  -10, 1,   5,   -6,  4,   2,   -1,  9,  4,   8,   2,   -7,  5,   -3,  7,   -5,  -9,  0,
      2,   -3,  -5,  -7,  9,   -6,  7,   -4,  -5,  -6,  -1,  -8,  -1, -2,  1,   6,   -3,  -8,  -5,  2,   8,   0,   -9,
      -7,  1,   7,   0,   2,   -1,  0,   -1,  -2,  -10, 9,   10,  5,  7,   5,   -8,  -7,  6,   -3,  -5,  -9,  8,   2,
      -3,  1,   -4,  -8,  4,   4,   0,   -8,  2,   2,   10,  5,   4,  1,   4,   -2,  -6,  -8,  -9,  -9,  -5,  6,   9,
      4,   7,   0,   -3,  -1,  -10, 4,   -1,  5,   0,   -10, 4,   0,  7,   -7,  3,   8,   -2,  8,   10,  -7,  -7,  3,
      -4,  -1,  2,   -7,  -1,  9,   -3,  4,   10,  7,   4,   -2,  -9, -9,  10,  10,  0,   9,   -2,  -8,  7,   5,   -6,
      -6,  -4,  -7,  -8,  -5,  4,   -4,  10,  3,   -10, -4,  -3,  4,  -6,  -5,  5,   -7,  10,  1,   -1,  8,   -2,  8,
      -9,  1,   8,   6,   6,   3,   -5,  0,   7,   1,   4,   -10, -5, -4,  -3,  6,   -3,  -4,  -3,  10,  10,  -1,  -8,
      -5,  -1,  6,   2,   0,   -2,  8,   -2,  -2,  10,  -6,  2,   3,  -3,  3,   -9,  7,   4,   -8,  1,   -4,  -6,  -6,
      6,   -7,  2,   5,   -8,  2,   -10, -6,  5,   9,   -10, -7,  -9, -7,  4,   -8,  -5,  -10, 7,   2,   9,   -2,  9,
      2,   6,   -2,  3,   -7,  -4,  -6,  1,   1,   2,   6,   9,   -8, 5,   -4,  -9,  10,  1,   -5,  -3,  -8,  9,   6,
      -2,  -4,  2,   -4,  -10, -5,  -3,  -6,  4,   3,   -9,  2,   5,  10,  -4,  2,   1,   -2,  6,   -5,  -7,  -4,  -10,
      0,   -6,  -5,  -4,  -8,  4,   4,   1,   3,   -6,  -8,  5,   -1, 1,   1,   -7,  -4,  4,   10,  -8,  0,   -9,  4,
      2,   -8,  10,  -7,  -4,  0,   8,   0,   -3,  4,   -1,  9,   6,  2,   -9,  0,   -1,  2,   4,   -2,  -5,  -9,  6,
      1,   4,   -3,  -7,  -2,  -2,  1,   1,   -3,  -5,  0,   -3,  -6, -7,  -7,  2,   4,   -9,  3,   1,   6,   4,   6,
      1,   0,   -4,  9,   2,   -5,  -5,  -5,  7,   1,   -4,  4,   1,  -1,  -10, 0,   10,  8,   -6,  5,   5,   -8,  1,
      -4,  1,   8,   6,   -6,  -2,  9,   1,   -2,  5,   10,  6,   -6, 5,   3,   -2,  9,   10,  3,   -10, 9,   -1,  0,
      6,   -6,  7,   6,   2,   4,   -5,  7,   0,   -10, -4,  5,   6,  0,   5,   2,   8,   -1,  7,   7,   -4,  -8,  5,
      -1,  -9,  6,   9,   9,   -3,  4,   -1,  -7,  -3,  -8,  -5,  -9, 4,   -6,  -1,  -9,  -3,  1,   6,   2,   7,   3,
      1,   -3,  -8,  -6,  5,   -6,  2,   -6,  -4,  8,   8,   9,   -9, -3,  -6,  -5,  1,   9,   -3,  -3,  -8,  7,   0,
      -7,  8,   5,   7,   -9,  10,  -9,  0,   10,  10,  7,   8,   -3, 10,  0,   -8,  3,   -8,  -1,  -9,  4,   4,   -3,
      8,   -5,  -4,  0,   1,   2,   8,   6,   3,   6,   1,   10,  -5, 8,   -4,  1,   -3,  0,   9,   6,   -8,  9,   2,
      -10, -9,  -2,  0,   -8,  9,   1,   7,   -3,  8,   10,  2,   -8, 3,   3,   7,   6,   -1,  -4,  -2,  9,   -2,  4,
      5,   9,   8,   -10, -9,  -9,  -6,  -10, -5,  -9,  -10, 6,   5,  9,   -6,  7,   -3,  -10, 0,   8,   2,   -1,  -10,
      -8,  -8,  5,   8,   -5,  -6,  -10, -10, 6,   3,   -5,  -1,  4,  -9,  -3,  -4,  1,   -5,  9,   2,   -3,  9,   7,
      -2,  6,   7,   2,   -4,  -6,  -1,  -10, -9,  7,   -4,  -7,  3,  -1,  5,   -5,  8,   1,   -5,  4,   0,   -9,  5,
      -2,  0,   8,   1,   -2,  9,   7,   1,   -2,  -1,  -7,  -9,  -3, 7,   -9,  10,  1,   3,   -7,  5,   -5,  9,   -9,
      4,   5,   -6,  5,   -8,  10,  -10, -3,  7,   0,   -1,  10,  9,  -10, 5,   9,   -2,  -5,  8,   -2,  -10, -8,  10,
      8,   -10, 10,  10,  2,   7,   5,   -6,  -10, 4,   5,   0,   6,  -2,  6,   -5,  7,   -10, -3,  0,   0,   -8,  -7,
      -3,  -8,  7,   -8,  -2,  -2,  10,  -6,  5,   1,   -1,  -1,  0,  3,   -10, 10,  -10, -8,  7,   4,   4,   -4,  6,
      9,   -5,  6,   -2,  -10, 5,   6,   2,   1,   -10, 6,   -7,  -7, -7,  10,  10,  -9,  -7,  4,   5,   4,   -5,  8,
      1,   3,   8,   9,   -4,  5,   -2,  -10, -8,  6,   6,   10,  -5, 2,   -9,  -5,  9,   -3,  -8,  0,   0,   9,   2,
      8,   7,   -9,  9,   1,   -1,  1,   5,   3,   6,   -4,  3,   -7, 4,   8,   6,   -6,  -9,  0,   9,   -9,  7,   7,
      7,   10,  -5,  6,   6,   5,   5,   -9,  -8,  -1,  5,   -8,  8,  1,   4,   -1,  2,   -6,  -10, 0,   5,   -5,  -1,
      1,   10,  -2,  -9,  -10, 10,  -10, -6,  -7,  7,   -4,  7,   6,  -9,  8,   -9,  -7,  3,   10,  6,   3,   -9,  -4,
      3,   5,   8,   -3,  7,   9,   -7,  -5,  -10, 9,   2,   -4,  -5, -3,  7,   -10, -4,  9,   -8,  -3,  -9,  -8,  0,
      -7,  -9,  10,  -1,  -4,  -9,  3,   4,   1,   5,   4,   10,  -3, 10,  3,   -7,  8,   3,   9,   -3,  4,   9,   2,
      -4,  -1,  7,   -3,  -8,  8,   5,   -4,  3,   8,   -8,  8,   5,  5,   -9,  2,   0,   5,   4,   -5,  -10, -4,  6,
      -8,  9,   7,   1,   -2,  -10, -3,  3,   9,   -5,  -1,  0,   -5, 7,   -7,  9,   7,   -5,  1,   4,   -5,  -7,  -1,
      1,   2,   -3,  -7,  2,   -5,  2,   -4,  8,   7,   1,   -2,  -7, 9,   -8,  10,  8,   -8,  6,   2,   10,  -8,  1,
      2,   4,   -10, 9,   7,   -3,  -4,  5,   9,   -7,  -7,  9,   -9, -1,  4,   1,   -6,  1,   2,   -3,  10,  7,   8,
      -3,  6,   -3,  -2,  1,   6,   10,  4,   -1,  -3,  5,   -5,  -9, 5,   -8,  7,   -9,  -1,  5,   -5,  -3,  -4,  -7,
      4,   -1,  -6,  2,   -10, 1,   -2,  -5,  -9,  -5,  -2,  -5,  3,  6,   0,   -4,  8,   -7,  -9,  1,   6,   5,   -5,
      5,   -9,  8,   6,   0,   1,   4,   8,   7,   8,   -1,  3,   -7, 0,   0,   9,   -8,  9,   0,   1,   1,   -1,  -4,
      9,   7,   6,   -7,  2,   10,  -1,  0,   -8,  5,   -4,  4,   3,  -9,  -10, 9,   -5,  -5,  0,   -1,  -6,  -2,  -6,
      1,   0,   -7,  3,   8,   -7,  4,   -2,  8,   -2,  -6,  8,   8,  9,   4,   -9,  -5,  3,   -5,  5,   -10, -5,  -3,
      3,   -10, -8,  -1,  -3,  -4,  6,   -3,  0,   1,   -8,  7,   -5, 8,   -1,  -5,  1,   -7,  -6,  0,   -7,  4,   9,
      9,   -1,  -1,  -10, 6,   7,   10,  5,   -9,  -10, -6,  10,  -1, -3,  4,   -2,  2,   3,   -7,  -4,  -1,  9,   -9,
      1,   8,   -4,  -7,  2,   -6,  -8,  -2,  -9,  7,   -9,  -7,  2,  -4,  -10, -8,  0,   7,   0,   3,   -10, -6,  6,
      4,   2,   -5,  1,   -5,  5,   8,   -5,  7,   -10, -4,  -3,  -6, -5,  8,   -8,  2,   -2,  3,   -4,  -9,  -9,  -6,
      5,   6,   -10, 7,   -2,  5,   3,   5,   10,  0,   -5,  -5,  7,  -7,  9,   -1,  -8,  -7,  -7,  6,   7,   8,   -10,
      6,   -6,  9,   2,   -3,  4,   3,   -8,  7,   -6,  0,   4,   4,  -7,  -8,  4,   0,   1,   -1,  -9,  2,   -1,  -3,
      -2,  5,   -3,  4,   -6,  8,   6,   -2,  -1,  7,   7,   7,   -3, 3,   0,   8,   -8,  5,   9,   6,   -4,  2,   -6,
      -5,  -7,  -7,  -4,  -5,  -4,  -2,  -7,  -1,  -10, -2,  -8,  2,  -9,  -10, -2,  -9,  -4,  3,   10,  9,   4,   -2,
      -10, -9,  -8,  -2,  -7,  8,   7,   9,   1,   1,   -4,  -4,  -3, 6,   -9,  3,   6,   -1,  6,   -4,  -5,  -6,  1,
      -9,  2,   7,   10,  -10, 0,   3,   -7,  2,   5,   -3,  0,   2,  -10, -2,  3,   -5,  7,   -1,  6,   3,   4,   -9,
      6,   -9,  8,   3,   9,   -6,  2,   -5,  6,   -10, 1,   -5,  5,  6,   7,   -10, -9,  4,   1,   -9,  5,   5,   2,
      9,   -8,  -2,  10,  5,   -1,  -3,  -4,  1,   6,   -6,  -8,  -3, 0,   1,   5,   -2,  8,   -8,  -9,  -7,  -4,  -4,
      1,   1,   -8,  -9,  -8,  -2,  -2,  5,   6,   -4,  3,   7,   3,  4,   -2,  3,   5,   7,   -3,  0,   6,   -9,  8,
      -4,  -8,  -1,  -5,  5,   9,   7,   7,   -9,  8,   6,   -1,  3,  -3,  3,   -3,  4,   5,   -7,  -6,  -5,  -9,  -2,
      -6,  -9,  6,   6,   -9,  1,   1,   -5,  -1,  5,   6,   -7,  7,  2,   2,   0,   5,   6,   -8,  -4,  -3,  -10, 0,
      -4,  5,   -8,  6,   9,   2,   4,   3,   9,   5,   -7,  8,   -8, -7,  -10, -4,  1,   3,   9,   2,   -3,  -2,  7,
      -10, 6,   7,   8,   5,   1,   6,   9,   -1,  -6,  7,   -8,  -2, 7,   7,   6,   7,   9,   -3,  -7,  7,   -6,  3,
      4,   4,   -5,  -9,  5,   2,   -6,  5,   2,   4,   6,   2,   0,  -2,  3,   4,   4,   8,   5,   2,   3,   3,   -8,
      -3,  -8,  -8,  8,   -1,  -8,  -1,  4,   -2,  -2,  2,   -6,  -5, 1,   -7,  0,   -5,  0,   5,   8,   1,   -8,  -1,
      -5,  -9,  3,   -1,  4,   9,   -3,  6,   5,   3,   -7,  -2,  7,  -3,  4,   10,  -6,  0,   10,  -3,  -10, -8,  10,
      3,   9,   -9,  7,   4,   4,   5,   -10, 0,   -10, 4,   2,   -5, 3,   3,   3,   -8,  -3,  -9,  1,   2,   2,   3,
      -10, 10,  10,  -6,  7,   7,   -4,  9,   8,   2,   -3,  -9,  9,  -2,  9,   -5,  10,  10,  6,   -8,  6,   -7,  8,
      4,   7,   -6,  10,  -8,  3,   -6,  6,   1,   -4,  4,   10,  -3, -6,  5,   2,   7,   10,  -7,  -8,  -9,  -6,  8,
      -4,  1,   1,   0,   4,   -1,  -5,  -2,  7,   -2,  4,   -3,  -7, 8,   2,   4,   -8,  -10, -5,  -3,  -10, 8,   -9,
      -10, -10, -3,  -7,  -5,  10,  -2,  8,   6,   -4,  -6,  5,   5,  10,  -3,  -3,  -6,  2,   -7,  -7,  5,   8,   6,
      9,   5,   -10, -1,  -10, -6,  -7,  -10, -1,  2,   0,   7,   2,  -8,  -9,  6,   10,  0,   6,   2,   9,   10,  10,
      2,   4,   -1,  4,   -10, 10,  6,   4,   -9,  5,   -7,  -9,  -1, -9,  4,   3,   5,   -9,  3,   -5,  3,   6,   10,
      5,   -8,  2,   -6,  -3,  -2,  1,   -4,  -8,  8,   10,  10,  -9, -7,  -4,  1,   -3,  -6,  -5,  9,   -1,  -4,  0,
      4});
  auto buffer_arr = gko::array<double>(exec, cell_size * 4 * 3);
  auto dst_arr = gko::array<double>(exec, cell_size * 4);
  auto expected_dst_arr = gko::array<double>(
    exec, I<double>{
            -4.45073,  165.636,  -442.179,  22.1873,  -31.1981,  -211.025,   666.201,  61.8852,   -27.0886,  -50.3442,
            131.946,   82.7477,  30.565,    35.8015,  143.844,   -70.5733,   -27.3936, -65.1935,  -1211.96,  -85.966,
            36.7264,   36.4698,  -245.312,  -20.2991, -2.13522,  -39.5628,   -567.487, -106.838,  26.9099,   -111.452,
            218.799,   -21.8044, 115.254,   58.8706,  -675,      -8.93807,   41.7421,  -9.05276,  -441.054,  27.3353,
            -28.1242,  -67.8354, -1043.28,  18.9267,  -3.65828,  -53.544,    -780.313, 95.4479,   112.168,   214.255,
            -669.054,  52.4366,  187.572,   15.0528,  864,       -127.606,   40.8019,  -91.5992,  -547.286,  -52.719,
            35.9633,   -14.1093, 1192.57,   74.5589,  -1.38679,  -39.309,    -1319.53, -20.4879,  -1.78512,  -127,
            -182.821,  -52.8701, 50.3962,   83.3492,  -667.161,  -53.0196,   -32.8019, -32.1671,  -398.268,  -109.719,
            -47.8653,  88.1947,  -163.594,  90.0166,  -88.6085,  -11.9774,   408.121,  36.5891,   7.85377,   -12.3053,
            -1563.79,  -102.242, -58.8857,  -15.054,  1566.54,   -21.0211,   59.1855,  37.858,    -486.42,   -112.289,
            -10.5849,  -102.32,  38.4911,   -83.2341, 97.4969,   42.6482,    807.746,  -83.6918,  107.403,   -25.1231,
            -44.9464,  26.1798,  129.769,   -18.1734, 751.674,   -61.0967,   -15.3139, 77.4347,   505.509,   -42.932,
            -68.6148,  178.093,  183.455,   33.8248,  -75.3931,  114.588,    252.054,  -18.5665,  -89.3428,  103.933,
            637.125,   -65.923,  -93.0262,  -241.611, -314.723,  8.17523,    -134.248, 43.5226,   229.982,   79.4502,
            -54.3732,  100.294,  608.286,   -14.5982, 20.0786,   -53.1709,   389.679,  -61.3369,  54.1887,   -207.363,
            141.848,   -60.4653, -58.8281,  6.21357,  -42.5982,  -65.9033,   -67.1604, -61.6281,  1278,      -1.5861,
            0.209644,  -11.397,  817.357,   97.9237,  33.4371,   16.3957,    401.036,  -72.8746,  -23.9528,  -133.045,
            -387.482,  40.4366,  57.327,    20.5188,  -59.2321,  123.975,    9.96751,  -168.84,   710.509,   -1.79532,
            -14.7531,  45.5352,  114.496,   87.5393,  -120.398,  22.4435,    -560.371, -73.5181,  91.3679,   106.819,
            160.513,   87.1261,  -99.1447,  25.9033,  -177.844,  -77.7689,   -66.9046, -36.4837,  39.5893,   62.787,
            16.6651,   -16.6269, 392.54,    -15.3731, -5.47694,  -140.539,   397.147,  81.2145,   -3.06289,  -19.8568,
            -194.277,  7.3716,   -152.851,  -280.291, -141.304,  -85.6813,   25.4874,  164.254,   203.719,   -1.38066,
            -21.6331,  138.107,  -541.451,  -7.38822, -87.0346,  84.1206,    -277.754, -6.80665,  -9.64361,  332.853,
            574.504,   131.328,  38.9429,   -135.347, 207.183,   -157.912,   -28.1111, 71.0879,   255.317,   -88.0853,
            98.0582,   -27.9724, 240.17,    -73.395,  -63.4423,  98.0339,    -172.067, 115.71,    -0.109015, -164.568,
            157.772,   56.9932,  18.9497,   -183.399, 582.228,   71.4856,    21.0666,  -357.545,  -500.054,  15.7402,
            -56.0409,  -162.264, 193.621,   114.48,   18.4324,   -53.4573,   811.821,  -35.7825,  -18.499,   -86.7588,
            638.335,   137.737,  -29.1751,  -60.7814, 422.286,   20.6314,    -1.84434, 281.744,   83.7991,   -14.1073,
            7.36583,   14.3945,  -398.625,  1.41843,  22.9738,   -31.0151,   19.7634,  -59.6843,  -83.8962,  -198.942,
            -673.821,  -5.9139,  -13.0503,  -151.555, 78.4598,   57.7432,    105.698,  283.317,   64.6518,   -54.2764,
            19.9308,   -8.30653, 252.804,   3.29003,  50.565,    -146.528,   189.259,  136.635,   -65.4654,  -140.111,
            -29.2768,  59.5876,  36.4329,   -36.9598, -508.344,  -26.148,    -66.5377, -0.879397, 183.75,    37.4003,
            54.8999,   -6.64573, -276.205,  -277.569, 64.5786,   212.653,    253.929,  -65.4358,  194.902,   -14.4095,
            1170.17,   -30.0279, 62.4057,   7.67337,  -438.067,  -61.1624,   121.226,  5.45729,   20.933,    53.7825,
            -83.8459,  -104.324, -618.772,  -68.8157, -12.9937,  -20.9548,   -1011.42, -110.907,  208.83,    159.106,
            -1144.41,  -62.1412, 88.3868,   -112.496, -614.987,  56.1775,    -9.00419, 138.807,   -398.429,  -63.1979,
            -20.2704,  -4.1495,  -88.9955,  -131.146, 69.0204,   132.2,      -561.335, -16.3822,  50.7579,   -98.7249,
            284.987,   150.757,  139.267,   232.538,  225.308,   44.5332,    18.3899,  -113.393,  -457.888,  -19.2462,
            50.6583,   92.4108,  117.5,     -33.503,  -50.6887,  -173.676,   352.179,  65.855,    -13.1939,  -192.347,
            -274.848,  -13.148,  68.3648,   -18.2299, 383.987,   -22.8761,   -8.14203, 111.023,   -349.156,  13.1344,
            205.016,   -78.4899, 412.701,   67.4683,  -134.679,  97.1872,    576.951,  -130.627,  -42.0597,  -23.4874,
            225.938,   -28.8353, 62.4182,   -85.5,    82.7679,   9.06344,    33.8795,  -64.5779,  629.741,   -45.1918,
            -18.5881,  -12.4271, -602.911,  -107,     25.9843,   -151.768,   105.054,  100.237,   -116.985,  9.68844,
            -971.464,  100.678,  23.3208,   -81.7538, 170.259,   31.8308,    35.1242,  -115.247,  570.911,   -3.35347,
            -36.0314,  -100.859, -688.107,  21.565,   -6.4392,   36.9322,    295.46,   32.7266,   65.979,    -45.4749,
            360.263,   6.55665,  -31.4937,  -66.8781, 303.067,   76.176,     -38.478,  7.84673,   199.768,   32.0846,
            36.12,     131.675,  -36.1161,  -53.7742, 76.2862,   173.951,    491.893,  74.685,    140.869,   -123.443,
            127.911,   24.6375,  -65.3774,  92.6005,  -58.8304,  -101.276,   43.6164,  84.3618,   273.455,   -2.62613,
            36.174,    57.6508,  -1430.14,  -93.716,  33.4686,   7.35678,    -255.951, 12.4215,   -7.55975,  -215.94,
            -215.571,  -27.1088, -19.532,   -20.2337, 804.951,   -120.223,   37.2516,  15.2161,   122.563,   -79.0929,
            3.35115,   15.3794,  -551.027,  -51.6473, 99.7736,   93.5201,    774.871,  -114.99,   -19.6834,  -203.457,
            -295.339,  -113.61,  38.4874,   -374.661, 481.902,   47.1367,    -28.26,   -138.785,  353.554,   5.11556,
            93.7673,   -100.613, 489.295,   -89.2477, 111.495,   74.0251,    -368.375, 51.4275,   61.1226,   -64.4849,
            -187.687,  -30.0136, -29.5372,  305.107,  -161.04,   109.674,    33.8312,  -59.1168,  222.612,   17.4879,
            19.8926,   1.81281,  129.335,   -19.5861, -11.8019,  -5.39322,   346.5,    58.6654,   -19.999,   -89.1231,
            -717.442,  50.3882,  28.5786,   -137.427, 448.54,    -49.6269,   78.8994,  7.09296,   -63.9375,  70.3323,
            2.66981,   -85.7337, -739.58,   -7.96677, -102.747,  -217.521,   34.0759,  66.1715,   44.3816,   57.3116,
            1062.16,   -37.3218, -125.934,  -29.2965, 882.763,   76.716,     76.7327,  21.4397,   -345.562,  -83.0295,
            42.8868,   333.437,  748.366,   47.1186,  -0.179245, 128.985,    638.625,  102.675,   -54.9623,  101.653,
            -1614.13,  -50.9275, -42.2788,  144.653,  223.862,   -110.317,   -84.1494, -0.535176, -487.437,  -36.9728,
            103.001,   -77.196,  -878.884,  -131.872, 104.349,   256.003,    -482.42,  21.9199,   9.18396,   -98.3266,
            -126.879,  17.4184,  -17.8029,  239.09,   -844.451,  -21.3912,   -56.2327, -162.013,  -794.054,  64.5076,
            33.577,    106.296,  315.67,    -26.1707, -107.592,  -137.995,   527.152,  84.7689,   63.4712,   82.9372,
            838.554,   29.8671,  19.8721,   -177.809, 70.7589,   34.2885,    -27.9738, 82.3543,   566.188,   5.0997,
            16.4313,   -56.2161, 1043.26,   33.7062,  -23.3438,  -54.1005,   227.179,  -27.9048,  74.173,    -15.6558,
            606.937,   -62.318,  28.0566,   18.1533,  -381.004,  107.196,    158.456,  -124.41,   899.29,    99.2402,
            -21.6363,  181.744,  -251.241,  66.8459,  -25.0273,  170.698,    -364.594, -23.0702,  -29.0535,  -173.028,
            583.888,   -85.0083, -34.9507,  8.13568,  -40.4018,  86.1934,    48.0157,  -46.9523,  -427.54,   -18.503,
            128.878,   -337.065, 188.138,   -71.8535, -40.7437,  -5.84548,   -487.268, 118.74,    -39.7484,  -54.9008,
            225.054,   86.1526,  -138.775,  78.0377,  53.6295,   43.1133,    57.2673,  -140.921,  -618.281,  9.02266,
            83.9942,   -211.348, 260.219,   142.23,   71.3695,   -44.3254,   -394.808, 113.175,   -122.447,  12.103,
            -47.3214,  111.011,  -52.2537,  -93.3178, -564.513,  49.9592,    -23.2704, 17.9058,   -44.4241,  -16.4456,
            -21.8218,  71.4673,  -494.884,  58.429,   112.289,   32.5364,    -66.2902, 44.5604,   -58.7725,  -90.4259,
            389.286,   -33.5166, 95.195,    11.4799,  63.0402,   10.5816,    -84.5676, 88.3417,   -569.259,  -145.824,
            51.1908,   -148.216, -207.768,  -52.8437, 79.9623,   -11.2387,   533.92,   4.20091,   -15.7736,  26.2613,
            -628.357,  147.261,  -89.9528,  56.9548,  -75.7098,  -99.2492,   -114.26,  -31.2085,  -873.009,  -84.1616,
            171.34,    103.877,  2093.8,    6.16994,  87.8616,   -216.264,   -255.201, -34.1624,  -16.4953,  -233.638,
            1389.42,   -36.8361, -12.6027,  125.796,  480.433,   32.1005,    96.6069,  2.8392,    297.455,   -10.3603,
            -63.9465,  -17.0992, 78.0937,   -52.0332, -50.565,   -42.1771,   69.7143,  -64.2485,  63.6447,   252.829,
            614.866,   210.308,  -10.5996,  -136.074, 332.491,   25.398,     -102.572, -89.8957,  244.183,   30.6934,
            83.6006,   -82.5603, -480.161,  -108.131, -0.735849, -174.251,   243.277,  -7.33384,  -92.2044,  477.373,
            196.835,   -111.915, -50.0314,  155.555,  643.232,   59.3248,    113.288,  52.4849,   210.866,   -45.6027,
            162.631,   -112.918, -743.607,  68.6156,  0.433962,  -229.176,   76.2054,  44.3542,   6.31761,   64.6432,
            -190.821,  -104.424, -135.155,  81.9472,  707.741,   -87.0272,   -132.252, 236.525,   629.455,   231.536,
            -21.0047,  157.975,  546.562,   18.281,   -67.6069,  60.2035,    294.67,   21.2221,   71.9581,   -102.987,
            1137.3,    -47.2447, -128.047,  -34.8342, -161.625,  -99.713,    -193.981, -34.7688,  227.321,   -1.35045,
            32.4224,   227.068,  -334.009,  -6.22961, 44.0079,   37.6131,    1259.68,  -35.0959,  1.17925,   143.774,
            -351.777,  31.713,   -48.2248,  -56.5477, -328.687,  -41.3021,   48.4822,  -213.553,  -563.638,  112.5,
            92.2516,   -52.2412, 303.759,   -36.3739, 111.029,   167.585,    551.661,  86.7349,   -101.686,  -88.5704,
            192.107,   106.862,  118.403,   177.259,  432.54,    54.8104,    34.4764,  -90.2299,  953.411,   -6.23112,
            -20.2945,  -140.677, 234.107,   -15.4003, -74.1027,  -118.621,   -510.134, 51.0763,   -53.587,   103.804,
            -435.938,  -140.091, 10.0881,   102.173,  309,       -102.548,   32.1667,  63.7952,   92.8259,   62.3769,
            25.217,    159.927,  -69.7232,  -33.358,  125.503,   250.534,    5.83929,  359.502,   16.544,    -136.143,
            940.254,   -13.2613, 141.928,   170.386,  -473.585,  -85.0876,   -22.2275, -73.9849,  -209.121,  127.53,
            26.0503,   31.4058,  -704.161,  70.932,   -2.96541,  -186.067,   -221.004, 79.9864,   -17.522,   35.1005,
            -130.625,  -3.67221, 68.108,    -26.4899, -1324.7,   -44.1571,   -33.2516, -79.4209,  310.687,   53.2432,
            -17.2233,  -110.342, -684.17,   102.606,  -16.4025,  -107.8,     -155.094, 5.48036,   -1.3805,   45.3015,
            644.411,   66.7477,  34.9182,   258.045,  -465.429,  -128.619,   -67.4843, 385.334,   199.312,   76.2326,
            9.92034,   -125.325, -143.777,  76.1813,  -119.165,  160.972,    -421.188, -148.574,  -134.063,  -58.5,
            136.929,   -118.785, 95.9104,   352.274,  773.652,   14.9116,    136.657,  63.995,    -112.473,  101.216,
            -3.37579,  13.2513,  919.313,   3.24698,  -23.5828,  91.8266,    -228.902, -24.7613,  -50.8113,  -16.9673,
            -473.562,  75.1556,  -105.286,  476.176,  507.839,   137.246,    81.7453,  179.977,   767.558,   -36.997,
            31.6992,   -73.9121, 304.71,    68.0831,  -23.1368,  161.992,    453.754,  -76.4653,  -74.3962,  52.4246,
            346.741,   -114.684, -61.0807,  -21.3141, -785.866,  39.7568,    10.1279,  5.9397,    -4.41518,  -74.7613,
            37.3601,   -96.897,  -222.268,  -33.7659, 133.426,   33.4673,    362.554,  -72.7855,  4.90776,   -188.128,
            1018.91,   -39.9245, 35.8616,   -6.84422, -368.156,  -57.3716,   205.817,  28.1847,   84.3661,   -75.8452,
            144.782,   -66.2751, 120.634,   -120.499, -70.957,   -222.006,   -1288.4,  134.361,   96.3947,   63.853,
            82.3214,   -98.8761, 87.9796,   250.236,  497.705,   1.40483,    27.1405,  58.3794,   -284.705,  107.884,
            98.88,     -111.138, 733.518,   -69.7749, -22.5487,  -17.255,    475.187,  -64.3512,  106.77,    -120.999,
            -212.826,  10.9079,  -17.5346,  -7.2902,  123.826,   22.9502,    -115.458, 81.5088,   -1152.9,   -18.1722,
            -104.198,  -1.11181, -854.362,  -136.202, 78.9769,   -158.545,   697.612,  95.9637,   -84.8491,  95.6734,
            487.777,   -8.9139,  51.8711,   -55.1156, 701.652,   -38.1798,   8.80084,  -67.3555,  -217.049,  76.0076,
            101.164,   -102.611, 1227.36,   66.0363,  71.3836,   -116.013,   555.964,  -44.9819,  -78.8255,  -135.03,
            -535.045,  322.72,   98.38,     -314.99,  130.848,   96.4683,    84.2374,  -8.33668,  1933.35,   129.288,
            80.3899,   65.8266,  -12.6429,  -15.0431, 5.55818,   34.3492,    264,      20.2727,   -13.6321,  97.8894,
            755.558,   95.756,   -4.5304,   44.1859,  51.433,    51.2651,    -31.173,  -138.942,  -479.879,  -141.052,
            7.24633,   -53.9912, -1086.12,  44.9683,  -32.6368,  -77.6156,   -253.473, 52.1057,   150.396,   -23.8266,
            -308.545,  141.787,  6.92138,   -85.9158, -521.879,  -62.2583,   -19.859,  98.8807,   752.625,   -3.71148,
            -5.50419,  89.1244,  -272.375,  193.592,  -116.25,   93.8593,    -708.442, -129.526,  -51.2767,  190.289,
            104.571,   -55.7311, -41.272,   68.2739,  70.2455,   48.3097,    -56.196,  105.349,   696.746,   99.8474,
            -13.9497,  -167.789, 37.8884,   -51.6616, -35.0818,  90.3719,    -332.884, -38.858,   -33.1064,  21.8467,
            -825.027,  -34.7432, 19.772,    12.8643,  67.317,    -123.171,   122.236,  -79.6018,  -533.973,  131.266,
            -94.9591,  -6.91332, 21.1518,   41.6979,  -16.7683,  14.4648,    -66.308,  144.585,   42.6604,   26.201,
            -138.228,  -140.642, -21.3118,  149.049,  -6.88393,  8.66767,    48.392,   -197.922,  -20.2857,  -24.9592,
            12.0786,   125.201,  -236.679,  -104.148, -73.0739,  19.9648,    -110.469, 33.2961,   -63.1352,  187.779,
            897.991,   93.2764,  -54.9277,  284.34,   182.692,   6.62991,    -127.468, 105.05,    155.268,   51.5332,
            47.1903,   47.2299,  105.545,   98.8218,  -71.9817,  -50.8656,   -368.549, 54.5861,   -36.8585,  435.447,
            292.406,   -32.0725, 45.7201,   -54.7312, 967.219,   109.101,    -69.3066, -86.5817,  262.509,   -19.4441,
            -80.9932,  468.965,  748.371,   -37.8097, -52,       51.5477,    197.821,  65.1722,   -92.1179,  -17.4573,
            -32.3571,  60.3089,  -116.628,  45.4221,  248.259,   76.5869,    19.1074,  -11.9171,  -855.129,  -123.242,
            48.3396,   -336.96,  -591.612,  -120.481, -70.956,   -59.9874,   -124.357, -74.1012,  9.28616,   -32.7487,
            163.036,   115.131,  4.99843,   231.95,   562.728,   76.7719,    51.3239,  2.68342,   5.39732,   55.4592,
            -0.269916, -3.69095, 949.478,   71.1073,  -144.009,  -82.8241,   -393.804, 83.7779,   -115.867,  381.271,
            440.312,   -75.9789, 219.807,   332.982,  145.96,    11.0211,    -98.7128, -167.337,  -859.433,  25.2387,
            -47.7683,  35.5075,  172.424,   -31.361,  -83.9727,  35.0088,    -453.437, 102.958,   34.7233,   -96.5465,
            402.402,   150.791,  60.5928,   -75.9284, 66.942,    -93.1715,   66.6132,  -69.7349,  -911.692,  -153.89,
            97.9036,   -122.339, -113.366,  11.4955,  -86.7736,  -83.7437,   467.732,  16.3142,   -48.6709,  83.5327,
            -1381.77,  33.8912,  -170.607,  -364.878, 32.5893,   64.7107,    -19.501,  250.206,   467.272,   87.5211,
            3.92872,   -341.977, 737.978,   -6.18353, 204.866,   186.667,    738.562,  -39.1518,  25.4308,   -219.418,
            -1010.44,  77.5967,  -91.5912,  166.153,  -375.134,  -110.277,   46.6515,  -166.255,  -749.246,  85.7228,
            -44.3805,  301.681,  148.112,   -31.5544, -40.3836,  17.6834,    -784.487, -21.1609,  -82.4748,  248.646,
            214.487,   -72.3399, -137.922,  -142.196, -477.638,  70.8595,    74.5472,  86.2915,   456.429,   -24.6526,
            -71.8491,  103.523,  -43.6473,  98.6148,  -212.387,  -116.653,   -474.415, -155.932,  38.3171,   -407.852,
            -217.687,  -37.6752, 5.66667,   -48.8367, 324.402,   24.2719,    -108.189, -88.9749,  -82.7812,  47.148,
            4.94549,   -63.6947, -320.844,  -81.2764, -142.398,  53.4095,    -213.795, -82.7719,  16.1918,   -5.82161,
            -136.027,  53.0997,  -0.821803, 62.1583,  883.46,    132.837,    84.9256,  -232.681,  -346.491,  -53.2976,
            73.7558,   -61.9962, -905.696,  -83.0091, -93.9623,  62.005,     951.201,  17.9819,   -135.119,  161.02,
            -658.165,  -115.749, -140.085,  -42.0339, 167.022,   36.435,     -15.5367, 29.1621,   165.446,   17.1375,
            -57.2998,  198.265,  702.29,    -56.565,  -0.493711, -354.763,   866.875,  139.157,   -64.9586,  -251.324,
            -1535.22,  -13.6979, -405.533,  -265.719, -108.991,  -31.1798,   12.9654,  -62.6005,  -272.129,  27.5166,
            -210.661,  54.0641,  -80.7143,  9.90937,  76.0063,   -311.845,   815.036,  -78.1178,  124.196,   288.222,
            -164.241,  123.749,  182.303,   -54.2563, -112.821,  -25.2009,   19.3291,  96.8166,   243.455,   22.0106,
            158.634,   -131.168, 235.996,   -123.228, 43.8192,   100.432,    -357.991, 163.586,   -78.173,   -311.608,
            -1422.03,  157.573,  -121.842,  154.173,  151.634,   -37.9048,   -19.1593, -27.1432,  278.058,   49.4894,
            55.7579,   -107.533, 397.902,   64.3686,  -50.674,   -51.1809,   -296.362, -143.232,  2.15042,   -26.1382,
            62.058,    31.9721,  -32.7605,  41.5854,  166.688,   66.3671,    -14.2484, 119.45,    111.187,   116.816,
            45.6195,   -35.495,  330.83,    -109.69,  36.2783,   58.8693,    444.348,  93.7885,   30.3931,   -39.8668,
            270.83,    32.327,   15.1268,   -169.367, 147.674,   87.2168,    -18.2348, 168.008,   200.446,   25.3112,
            23.1604,   -18.3719, 256.741,   29.7613,  37.1714,   20.9799,    113.375,  87.1254,   141.843,   126.746,
            -15.8839,  -68.9819, 34.0975,   -86.4724, -307.031,  -67.2779,   93.7877,  48.6181,   479.679,   14.7281,
            -65.3412,  73.3003,  -51.8304,  53.0076,  -78.1819,  -3.43342,   -383.768, 20.6677,   -123.164,  -75.4296,
            27.9107,   161.216,  -90.3082,  45.5352,  612.214,   -14.3792,   -29.6368, -54.0905,  112.527,   33.0816,
            74.5377,   60.196,   82.0848,   -151.817, -55.1992,  -48.0678,   -294.946, -94.9789,  -112.306,  75.8819,
            -557.625,  23.9622,  -125.085,  93.2638,  -37.6741,  2.37387,    -29.4843, -92.2274,  7.19196,   1.94411,
            -8.36897,  -17.3178, 351.786,   2.20619,  -27.5975,  20.9397,    -90.5759, -106.955,  -102.868,  34.0477,
            -95.567,   -40.6911, -19.6572,  202.433,  132.879,   -79.8459,   36.7626,  12.4422,   83.0982,   44.3074,
            -154.172,  -21.0013, 551.893,   -22.1699, 16.5823,   94.2814,    -25.7679, 40.5378,   7.42453,   -126.09,
            297.281,   5.04834,  44.543,    -4.33668, 791.723,   -23.7258,   -149.561, 79.8869,   -322.509,  -6.24849,
            -86.3553,  12.8367,  178.835,   -112.056, -53.8973,  -17.0251,   434.112,  -40.1375,  30.2196,   -49.1834,
            328.29,    7.22583,  34.0896,   -3.39196, 470.464,   -34.6767,   -27.0299, -97.6734,  347.933,   -67.602,
            -17.2091,  -73.2638, 846.018,   56.9079,  -63.782,   11,         624.826,  -64.8625,  13.9555,   6.75126,
            -644.504,  78.5408,  2.26415,   -85.4397, 274.179,   1.4864,     69.4717,  -182.691,  -385.705,  -33.4267,
            -23.9602,  -41.9849, 376.83,    12.4532,  93.1771,   146.397,    43.2321,  114.988,   -86.2657,  -75.7839,
            105.969,   -6.6148,  -37.6195,  -100.101, 23.2768,   73.9894,    -49.0692, 47.4196,   277.125,   -45.7704,
            -24.7264,  -25.9033, 135.107,   53.2704,  -15.7662,  -1.75251,   -322.652, 52.0604,   -7.05975,  -43.9296,
            -444.589,  48.3897,  -27.0865,  -21.7525, -315.049,  0.924471,   -72.7689, 18.3681,   -221.634,  15.6858,
            13.3024,   -49.0239, 237.504,   -66.0763, 153.246,   41.1445,    768.549,  115.089,   90.7264,   -94.0427,
            -941.71,   -109.9,   23.5126,   35.0126,  987.804,   -41.071,    -43.3255, 26.5477,   -177.509,  39.4169,
            -119.469,  16.4246,  292.058,   54.784,   89.7961,   -11.1985,   515.799,  -161.079,  50.577,    32.1709,
            574.692,   12.1254,  29.5503,   234.384,  -1246.67,  -78.3867,   -51.2568, -36.8103,  220.812,   8.10423,
            28.8899,   -13.5704, -627.674,  -10.1518, 40.6352,   14.4535,    1177.45,  18.1427,   -146.003,  -82.5766,
            -650.661,  74.2145,  27.1184,   174.839,  289.527,   -12.5415,   26.2809,  83.0503,   -30.5268,  18.6193,
            107.211,   -84.5239, -1854.92,  -55.3051, -13.1761,  -49.9899,   -183.683, -38.1548,  -22.8978,  112.935,
            593.955,   89.8565,  -79.9623,  102.994,  -518.21,   7.11782,    -88.7746, 62.4435,   776.406,   46.9456,
            27.2243,   250.447,  243.188,   40.0665,  -51.3622,  -89.951,    -155.813, 92.6035,   -97.7317,  -6.47111,
            -568.987,  1.85952,  -49.1289,  4.57161,  1187.8,    9.63444,    60.6447,  -139.866,  -991.295,  9.92221,
            29.673,    -131.966, -364.929,  5.62613,  133.236,   95.5704,    477.929,  -11.0196,  39.9539,   18.6809,
            725.562,   -90.8293, -18.0147,  200.045,  -128.08,   46.6556,    -76.9245, -77.9095,  -225.201,  127.403,
            -102.22,   54.5766,  -1439.12,  -74.3429, -106.66,   -68.7852,   572.92,   -0.471299, 14.174,    119.812,
            -553.612,  -17.3943, -72.8637,  225.984,  64.8036,   51.8369,    46.3349,  -20.5553,  169.955,   0.0951662,
            -216.355,  89.0641,  169.652,   97.5785,  -104.959,  83.1193,    170.009,  31.3112,   -150.147,  68.5917,
            -478.281,  71.4834,  -181.97,   162.183,  -342.295,  -3.23565,   36.0943,  100.101,   80.1027,   7.55891,
            62.5136,   -114.543, 418.795,   49.5128,  52.1447,   -84.9724,   -324.241, 0.799849,  6.09591,   -70.1457,
            -534.589,  11.003,   8.98742,   -50.4422, 636.75,    47.3248,    -50.4885, 97.8819,   94.3571,   21.926,
            -24.6153,  -241.555, 133.978,   -118.912, -46.8407,  -0.0502513, 16.2232,  22.5211,   144.775,   -6.84925,
            -822.152,  35.1843,  23.9518,   -21.7173, -361.647,  -11.4728,   93.6494,  91.6319,   315.174,   -252.07,
            147.48,    -166.006, -1441.51,  -63.6073, -4.34801,  -150.185,   370.683,  81.5808,   139.221,   0.158291,
            -43.3527,  -5.78927, 112.465,   173.116,  281.554,   -50.1224,   -99.0136, -18.6558,  1121.01,   40.642,
            104.661,   136.305,  -1167.83,  -13.6881, 47.0231,   63.3442,    1195.46,  -10.71,    31.1682,   28.3028,
            824.996,   126.978,  134.149,   -37.5603, 531.536,   124.4,      40.0881,  104.77,    406.888,   98.2183,
            49.8973,   -76.2977, 714.799,   3.62009,  -90.2558,  -114.425,   -168.679, -103.332,  68.2379,   206.804,
            1239.21,   16.8157,  134.23,    62.7148,  262.513,   10.0415,    -115.978, 94.7161,   -319.393,  54.1495,
            39.4712,   357.701,  -1.36607,  142.408,  19.3522,   -94.5565,   -34.4464, -127.974,  -45.4497,  20.8756,
            971.357,   -13.8693, -271.356,  80.5452,  -612.317,  101.802,    24.3648,  29.6771,   -1442.48,  70.7032,
            72.6352,   -119.691, 777.54,    -4.63142, 102.091,   -122.291,   -607.192, 5.4426});
  using VectorizedNumber = double4;
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> jac{
    reinterpret_cast<const VectorizedNumber*>(jac_arr.get_const_data()), dim, dim, dim};
  gko::batch::matrix::dense::batch_item<const Number> shape_gradients(
    shape_gradients_arr.get_const_data(), ::fe_degree + 1, ::fe_degree + 1, ::fe_degree + 1);
  gko::batch::matrix::dense::batch_item<const VectorizedNumber> speed_cells{
    reinterpret_cast<const VectorizedNumber*>(speed_cells_arr.get_const_data()), dim, cell_size, dim};
  gko::batch::multi_vector::batch_item<const VectorizedNumber> src{
    reinterpret_cast<const VectorizedNumber*>(src_arr.get_const_data()), 1, cell_size, 1};
  gko::batch::multi_vector::batch_item<VectorizedNumber> dst{reinterpret_cast<VectorizedNumber*>(dst_arr.get_data()), 1,
                                                             cell_size, 1};
  double time_factor = 1.5;
  double inv_dt = 4;

  dst_arr.fill(0.0);
  simple_cell_kernel<::fe_degree, dim><<<1, cell_size>>>(jac, shape_gradients, speed_cells,
                                                         weights_arr.get_const_data(), time_factor, inv_dt, src, dst,
                                                         reinterpret_cast<VectorizedNumber*>(buffer_arr.get_data()));
  exec->synchronize();

  for (int i = 0; i < cell_size * 4; ++i) {
    auto result = dst_arr.get_data()[i];
    auto expected = expected_dst_arr.get_data()[i];
    auto eps = (std::abs(result) + std::abs(expected)) * 1e-5;
    EXPECT_NEAR(result, expected, eps) << i / 4 << " " << i % 4;
  }
}
} // namespace test_dim3
