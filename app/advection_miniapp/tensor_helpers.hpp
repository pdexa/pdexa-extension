// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <array>
#include "pdexa-ext/ginkgo/core/matrix/batch_struct.hpp"

template<typename T>
struct tensor3d {
  T* __restrict__ data = nullptr;
  std::array<uint32_t, 3> size = {};

  constexpr tensor3d<const T> to_const() const { return {data, size}; }

  constexpr T operator()(uint32_t i, uint32_t j, uint32_t k) const {
    assert(i < size[0] && j < size[1] && k < size[2]);
    return data[i + j * size[0] + k * size[0] * size[1]];
  }

  constexpr T& operator()(uint32_t i, uint32_t j, uint32_t k) {
    assert(i < size[0] && j < size[1] && k < size[2]);
    return data[i + j * size[0] + k * size[0] * size[1]];
  }
};


template<int dim, typename Number>
constexpr Number determinant(gko::batch::matrix::dense::batch_item<const Number> m) {
  if constexpr (dim == 1) {
    return m(0, 0);
  }
  if constexpr (dim == 2) {
    return m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1);
  }
  if constexpr (dim == 3) {
    const Number C0 = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
    const Number C1 = m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2);
    const Number C2 = m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);
    return m(0, 0) * C0 + m(0, 1) * C1 + m(0, 2) * C2;
  }
  __builtin_unreachable();
}