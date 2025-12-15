// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <mma.h>

namespace wmma {
using namespace nvcuda::wmma;

constexpr int M = 8;
constexpr int N = 8;
constexpr int K = 4;

constexpr int WAVE_SIZE = 32;

/* The cuda WMMA doesn't allow setting the layout parameter for the accumulator
 * fragment, so to unify it we introduce a type mapping.
 */
template <int m, int n, int k, typename T, typename Layout>
using accumulator_fragment = fragment<accumulator, m, n, k, T>;

/* The cuda WMMA store function requires the runtime layout parameter, while
 * the rocm verison doesn't. To not use the runtime dispatch in rocm a
 * dispatching function is introduced.
 */
template <typename MatrixT, int BlockM, int BlockN, int BlockK, typename DataT,
          typename DataLayoutT>
__device__ __forceinline__ void unified_store_matrix_sync(
    DataT *data,
    fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const &frag,
    uint32_t ldm, layout_t layout) {
  store_matrix_sync(data, frag, ldm, layout);
}

} // namespace wmma
