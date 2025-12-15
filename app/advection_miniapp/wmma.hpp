// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#if PDEXA_EXT_ENABLE_CUDA

#include "wmma.cuda.hpp"

#else

#error "No device backend enabled."

#endif

namespace wmma {

// Thread block
// : T_BLOCK_X must be multiple of WAVE_SIZE.
// Note: Each wave will compute one BLOCK_M x BLOCK_N output block
// Note: Workgroup will compute
//  T_BLOCK_X / WAVE_SIZE x T_BLOCK_Y output blocks
const int WAVES = 4;
const int T_BLOCK_X = WAVES * WAVE_SIZE;
const int T_BLOCK_Y = 4 / WAVES;

} // namespace wmma


namespace gko::batch::matrix::dense {
template<typename>
struct batch_item;

template<typename>
struct batch_item_colum_major;
}


namespace impl {

template<typename Tag>
struct type_to_layout;
template<typename T>
struct type_to_layout<gko::batch::matrix::dense::batch_item<T>> {
  using type = wmma::row_major;
};
template<typename T>
struct type_to_layout<gko::batch::matrix::dense::batch_item_colum_major<T>> {
  using type = wmma::col_major;
};

template<typename Layout>
struct runtime_layout;
template<typename T>
struct runtime_layout<gko::batch::matrix::dense::batch_item<T>> {
  static constexpr wmma::layout_t value = wmma::layout_t::mem_row_major;
};
template<typename T>
struct runtime_layout<gko::batch::matrix::dense::batch_item_colum_major<T>> {
  static constexpr wmma::layout_t value = wmma::layout_t::mem_col_major;
};

} // namespace impl
template<typename T>
using type_to_layout = typename impl::type_to_layout<T>::type;
template<typename Layout>
inline constexpr wmma::layout_t runtime_layout = impl::runtime_layout<Layout>::value;
