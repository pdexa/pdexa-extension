// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include "pdexa-ext/ginkgo/core/matrix/batch_struct.hpp"
#include "tensor_helpers.hpp"


template<int fe_degree, int dim, int dir, bool transpose, bool additive, typename NumberShape, typename NumberIn, typename NumberOut>
void cpu_tensor_apply(gko::batch::matrix::dense::batch_item<const NumberShape> shape_data,
                      tensor3d<const NumberIn> in,
                      tensor3d<NumberOut> out) {
  for (uint32_t k = 0; k < (dim >= 3 ? fe_degree + 1 : 1); ++k) {
    for (uint32_t j = 0; j < (dim >= 2 ? fe_degree + 1 : 1); ++j) {
      for (uint32_t i = 0; i < (dim >= 1 ? fe_degree + 1 : 1); ++i) {
        auto dir_idx = [](auto ii, auto jj, auto kk) -> uint32_t {
          if constexpr (dir == 0) {
            return ii;
          }
          if constexpr (dir == 1) {
            return jj;
          }
          if constexpr (dir == 2) {
            return kk;
          }
          __builtin_unreachable();
        };

        auto map = [](auto ii, auto jj, auto kk) -> uint32_t {
          if constexpr (dim == 1) {
            return ii;
          }
          if constexpr (dim == 2) {
            return ii + jj * (fe_degree + 1);
          }
          if constexpr (dim == 3) {
            return ii + jj * (fe_degree + 1) + kk * (fe_degree + 1) * (fe_degree + 1);
          }
          __builtin_unreachable();
        };

        auto acc = [&](auto ii, auto jj, auto kk, auto qq) -> uint32_t {
          if constexpr (dir == 0) {
            return map(qq, jj, kk);
          }
          if constexpr (dir == 1) {
            return map(ii, qq, kk);
          }
          if constexpr (dir == 2) {
            return map(ii, jj, qq);
          }
          __builtin_unreachable();
        };

        auto sum = additive ? out(i, j, k) : NumberOut{};
        for (uint32_t q = 0; q < fe_degree + 1; ++q) {
          sum += (transpose ? shape_data(q, dir_idx(i, j, k)) : shape_data(dir_idx(i, j, k), q)) *
                          in.data[acc(i, j, k, q)];
        }
        out(i, j, k) = sum;
      }
    }
  }
}
