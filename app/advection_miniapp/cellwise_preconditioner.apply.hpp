// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include "cellwise_preconditioner.hpp"
#include "complex.hpp"
#include "tensor_kernel.cpu.hpp"
#include "tensor_kernel.gpu.hpp"
#include "vectorization.hpp"

#include <cuda/std/complex>
#include <thrust/complex.h>

namespace DGAdvection {

template<int dim, int fe_degree, typename Number1, typename Number2>
void simple_apply(CellwisePreconditionerItem<dim, fe_degree, Number1, Number2> a,
                  gko::batch::multi_vector::batch_item<const Number1> b,
                  gko::batch::multi_vector::batch_item<Number1> x,
                  gko::cpu_kernel) {
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  constexpr auto vector_size = static_cast<gko::int32>(sizeof(Number2) / sizeof(Number1));

  auto b_vectorized = gko::batch::multi_vector::batch_item<const Number2>{
    reinterpret_cast<const Number2*>(b.values), b.stride / vector_size, b.num_rows, b.num_rhs / vector_size};
  auto x_vectorized = gko::batch::multi_vector::batch_item<Number2>{
    reinterpret_cast<Number2*>(x.values), x.stride / vector_size, x.num_rows, x.num_rhs / vector_size};

  std::array tmp_tensor = {tensor3d{a.data_array[0].values, tensor_size},
                           tensor3d{a.data_array[1].values, tensor_size}};
  auto b_tensor = tensor3d{b_vectorized.values, tensor_size};
  auto x_tensor = tensor3d{x_vectorized.values, tensor_size};

  // copy from real to complex vector
  for (unsigned int i = 0; i < b.num_rows; ++i) { a.data_array[0][i] = b_vectorized[i]; }

  std::array<Number1, 3> blend_factor_eig;
  for (unsigned int d = 0; d < dim; ++d) {
    if (a.average_velocity[d] * a.time_factor < 0.0) { blend_factor_eig[d] = 1.0; }
    else { blend_factor_eig[d] = 0.0; }
  }

  std::complex<Number1> eigenvectors_storage[dim][(fe_degree + 1) * (fe_degree + 1)];
  std::complex<Number1> inverse_eigenvectors_storage[dim][(fe_degree + 1) * (fe_degree + 1)];
  row_major_matrix<std::complex<Number1>> eigenvectors[dim];
  row_major_matrix<std::complex<Number1>> inverse_eigenvectors[dim];
  for (unsigned int d = 0; d < dim; ++d) {
    eigenvectors[d] = row_major_matrix(&eigenvectors_storage[d][0], fe_degree + 1, fe_degree + 1, fe_degree + 1);
    inverse_eigenvectors[d] =
      row_major_matrix(&inverse_eigenvectors_storage[d][0], fe_degree + 1, fe_degree + 1, fe_degree + 1);
  }

  for (unsigned int d = 0; d < dim; ++d) {
    for (unsigned int i = 0; i < fe_degree + 1; ++i)
      for (unsigned int j = 0; j < fe_degree + 1; ++j) {
        eigenvectors[d](i, j) = (1.0 - blend_factor_eig[d]) * a.eigenvectors(i, j) +
                                blend_factor_eig[d] * a.eigenvectors(i + fe_degree + 1, j);
        inverse_eigenvectors[d](i, j) = (1.0 - blend_factor_eig[d]) * a.inverse_eigenvectors(i, j) +
                                        blend_factor_eig[d] * a.inverse_eigenvectors(i + fe_degree + 1, j);
      }
  }

  using gko::batch::matrix::to_const;
  // apply V M^{-1}
  cpu_tensor_apply<fe_degree, dim, 0, false, false>(to_const(inverse_eigenvectors[0]), tmp_tensor[0].to_const(),
                                                    tmp_tensor[1]);
  std::swap(tmp_tensor[0], tmp_tensor[1]);
  if (dim > 1) {
    cpu_tensor_apply<fe_degree, dim, 1, false, false>(to_const(inverse_eigenvectors[1]), tmp_tensor[0].to_const(),
                                                      tmp_tensor[1]);
    std::swap(tmp_tensor[0], tmp_tensor[1]);
  }
  if (dim > 2) {
    cpu_tensor_apply<fe_degree, dim, 2, false, false>(to_const(inverse_eigenvectors[2]), tmp_tensor[0].to_const(),
                                                      tmp_tensor[1]);
    std::swap(tmp_tensor[0], tmp_tensor[1]);
  }

  // apply inv(I x Lambda + Lambda x I)
  for (unsigned int i2 = 0, c = 0; i2 < tensor_size[2]; ++i2) {
    for (unsigned int i1 = 0; i1 < tensor_size[1]; ++i1) {
      for (unsigned int i0 = 0; i0 < tensor_size[0]; ++i0, ++c) {
        std::array<unsigned int, 3> indices{{i0, i1, i2}};
        std::complex<Number2> diagonal_element(0.0, 0.0);
        for (unsigned int d = 0; d < dim; ++d) {
          std::complex<Number> eig1(a.eigenvalues(0, indices[d])), eig2(a.eigenvalues(1, indices[d]));
          diagonal_element += a.average_velocity[d] * ((1.0 - blend_factor_eig[d]) * eig1 + blend_factor_eig[d] * eig2);
        }

        auto scal = a.determinant / (a.inv_dt + a.time_factor * diagonal_element);
        tmp_tensor[0].data[c] *= a.determinant / (a.inv_dt + a.time_factor * diagonal_element);
      }
    }
  }

  // apply V^{-1}
  cpu_tensor_apply<fe_degree, dim, 0, false, false>(to_const(eigenvectors[0]), tmp_tensor[0].to_const(), tmp_tensor[1]);
  std::swap(tmp_tensor[0], tmp_tensor[1]);
  if (dim > 1) {
    cpu_tensor_apply<fe_degree, dim, 1, false, false>(to_const(eigenvectors[1]), tmp_tensor[0].to_const(),
                                                      tmp_tensor[1]);
    std::swap(tmp_tensor[0], tmp_tensor[1]);
  }
  if (dim > 2) {
    cpu_tensor_apply<fe_degree, dim, 2, false, false>(to_const(eigenvectors[2]), tmp_tensor[0].to_const(),
                                                      tmp_tensor[1]);
    std::swap(tmp_tensor[0], tmp_tensor[1]);
  }

  // copy back to real vector
  for (unsigned int i = 0; i < b.num_rows; ++i) { x_vectorized[i] = tmp_tensor[0].data[i].real(); }
}

template<typename T>
struct cuda_type_s {
  using type = T;
};
template<typename T>
struct cuda_type_s<const T> {
  using type = const cuda_type_s<T>::type;
};
template<typename T>
struct cuda_type_s<T*> {
  using type = cuda_type_s<T>::type*;
};
template<typename T>
struct cuda_type_s<std::complex<T>> {
  using type = cuda::std::complex<T>;
};
template<>
struct cuda_type_s<std::complex<double4>> {
  using type = ext::complex<double4>;
};
template<>
struct cuda_type_s<std::complex<double2>> {
  using type = ext::complex<double2>;
};
template<typename T>
using cuda_type = typename cuda_type_s<T>::type;

template<typename T>
constexpr cuda_type<T>* as_cuda(T* obj) {
  return reinterpret_cast<cuda_type<T>*>(obj);
}
template<typename T>
constexpr cuda_type<T> as_cuda(T obj) {
  return *as_cuda(&obj);
}

template<typename T>
constexpr gko::batch::matrix::dense::batch_item<cuda_type<T>> as_cuda(gko::batch::matrix::dense::batch_item<T> item) {
  return gko::batch::matrix::dense::batch_item{as_cuda(item.values), item.stride, item.num_rows, item.num_cols};
}

template<int dim, int fe_degree, typename Number1, typename Number2>
__device__ void simple_apply(CellwisePreconditionerItem<dim, fe_degree, Number1, Number2> a,
                             gko::batch::multi_vector::batch_item<const Number1> b,
                             gko::batch::multi_vector::batch_item<Number1> x,
                             gko::cuda_hip_kernel) {
  using complex_t = cuda_type<std::complex<Number2>>;
  std::array<uint32_t, 3> tensor_size{dim >= 1 ? fe_degree + 1 : 1, dim >= 2 ? fe_degree + 1 : 1,
                                      dim >= 3 ? fe_degree + 1 : 1};

  constexpr auto vector_size = static_cast<gko::int32>(sizeof(Number2) / sizeof(Number1));

  auto b_vectorized = gko::batch::multi_vector::batch_item<const Number2>{
    reinterpret_cast<const Number2*>(b.values), b.stride / vector_size, b.num_rows, b.num_rhs / vector_size};
  auto x_vectorized = gko::batch::multi_vector::batch_item<Number2>{
    reinterpret_cast<Number2*>(x.values), x.stride / vector_size, x.num_rows, x.num_rhs / vector_size};

  std::array<tensor3d<complex_t>, 2> tmp_tensor;
  if constexpr (sizeof(Number2) <= 8) {
    __shared__ complex_t tmp_storage[2][dealii::Utilities::pow(fe_degree + 1, dim)];
    tmp_tensor[0] = tensor3d{tmp_storage[0], tensor_size};
    tmp_tensor[1] = tensor3d{tmp_storage[1], tensor_size};
  }
  else {
    tmp_tensor[0] = tensor3d{as_cuda(a.data_array[0].values), tensor_size};
    tmp_tensor[1] = tensor3d{as_cuda(a.data_array[1].values), tensor_size};
  }
  auto b_tensor = tensor3d{as_cuda(b_vectorized.values), tensor_size};
  auto x_tensor = tensor3d{as_cuda(x_vectorized.values), tensor_size};

  __shared__ complex_t eigenvectors_storage[dim][(fe_degree + 1) * (fe_degree + 1)];
  __shared__ complex_t inverse_eigenvectors_storage[dim][(fe_degree + 1) * (fe_degree + 1)];
  row_major_matrix<complex_t> eigenvectors[dim];
  row_major_matrix<complex_t> inverse_eigenvectors[dim];
  for (unsigned int d = 0; d < dim; ++d) {
    eigenvectors[d] = row_major_matrix(&eigenvectors_storage[d][0], fe_degree + 1, fe_degree + 1, fe_degree + 1);
    inverse_eigenvectors[d] =
      row_major_matrix(&inverse_eigenvectors_storage[d][0], fe_degree + 1, fe_degree + 1, fe_degree + 1);
  }

  std::array<Number2, dim> blend_factor_eig;
  for (unsigned int d = 0; d < dim; ++d) {
    blend_factor_eig[d] = static_cast<Number2>(a.average_velocity[d] * a.time_factor < 0.0);
  }

  for (unsigned int idx = threadIdx.x; idx < dim * (fe_degree + 1) * (fe_degree + 1); idx += blockDim.x) {
    auto j = idx % (fe_degree + 1);
    auto i = (idx / (fe_degree + 1)) % (fe_degree + 1);
    auto d = idx / (fe_degree + 1) / (fe_degree + 1);
    eigenvectors[d](i, j) = static_cast<complex_t>(1.0 - blend_factor_eig[d]) * as_cuda(a.eigenvectors(i, j)) +
                            static_cast<complex_t>(blend_factor_eig[d]) * as_cuda(a.eigenvectors(i + fe_degree + 1, j));
    inverse_eigenvectors[d](i, j) =
      static_cast<complex_t>(1.0 - blend_factor_eig[d]) * as_cuda(a.inverse_eigenvectors(i, j)) +
      static_cast<complex_t>(blend_factor_eig[d]) * as_cuda(a.inverse_eigenvectors(i + fe_degree + 1, j));
  }
  __syncthreads();

  using gko::batch::matrix::to_const;

  // copy from real to complex vector
  for (unsigned int i = threadIdx.x; i < b_vectorized.num_rows; i += blockDim.x) {
    tmp_tensor[0].data[i] = b_vectorized[i];
  }
  __syncthreads();

  // apply V M^{-1}
  simple_tensor_apply<fe_degree, dim, 0, false, false>(to_const(inverse_eigenvectors[0]), tmp_tensor[0].to_const(),
                                                       tmp_tensor[1]);
  std::swap(tmp_tensor[0], tmp_tensor[1]);
  if (dim > 1) {
    __syncthreads();
    simple_tensor_apply<fe_degree, dim, 1, false, false>(to_const(inverse_eigenvectors[1]), tmp_tensor[0].to_const(),
                                                         tmp_tensor[1]);
    std::swap(tmp_tensor[0], tmp_tensor[1]);
  }
  if (dim > 2) {
    __syncthreads();
    simple_tensor_apply<fe_degree, dim, 2, false, false>(to_const(inverse_eigenvectors[2]), tmp_tensor[0].to_const(),
                                                         tmp_tensor[1]);
    std::swap(tmp_tensor[0], tmp_tensor[1]);
  }

  // apply inv(I x Lambda + Lambda x I)
  __syncthreads();
  for (unsigned int c = threadIdx.x; c < a.data_array[0].num_rows; c += blockDim.x) {
    auto i = c % tensor_size[0];
    auto j = (c / tensor_size[0]) % tensor_size[1];
    auto k = c / (tensor_size[0] * tensor_size[1]);

    std::array indices{i, j, k};

    complex_t diagonal_element(gko::zero<Number2>(), gko::zero<Number2>());
    for (unsigned int d = 0; d < dim; ++d) {
      diagonal_element += a.average_velocity[d] *
                          (static_cast<complex_t>(1.0 - blend_factor_eig[d]) * as_cuda(a.eigenvalues(0, indices[d])) +
                           static_cast<complex_t>(blend_factor_eig[d]) * as_cuda(a.eigenvalues(1, indices[d])));
    }

    tmp_tensor[0](i, j, k) *= as_cuda(a.determinant) / (as_cuda(a.inv_dt) + as_cuda(a.time_factor) * diagonal_element);
  }
  __syncthreads();

  // apply V^{-1}
  simple_tensor_apply<fe_degree, dim, 0, false, false>(to_const(eigenvectors[0]), tmp_tensor[0].to_const(),
                                                       tmp_tensor[1]);
  std::swap(tmp_tensor[0], tmp_tensor[1]);
  if (dim > 1) {
    __syncthreads();
    simple_tensor_apply<fe_degree, dim, 1, false, false>(to_const(eigenvectors[1]), tmp_tensor[0].to_const(),
                                                         tmp_tensor[1]);
    std::swap(tmp_tensor[0], tmp_tensor[1]);
  }
  if (dim > 2) {
    __syncthreads();
    simple_tensor_apply<fe_degree, dim, 2, false, false>(to_const(eigenvectors[2]), tmp_tensor[0].to_const(),
                                                         tmp_tensor[1]);
    std::swap(tmp_tensor[0], tmp_tensor[1]);
  }

  // copy back to real vector
  __syncthreads();
  for (unsigned int i = threadIdx.x; i < b_vectorized.num_rows; i += blockDim.x) {
    x_vectorized[i] = tmp_tensor[0].data[i].real();
  }
}
} // namespace DGAdvection
