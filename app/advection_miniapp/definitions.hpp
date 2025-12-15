// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <cstddef>


namespace dealii {
template<typename, std::size_t>
class VectorizedArray;
}

namespace DGAdvection {

using Number = double;

// The polynomial degree can be selected between 0 and any reasonable number
// (around 30), depending on the dimension and the mesh size
constexpr unsigned int fe_degree = 7;

// The time step size is controlled via this parameter as
// dt = courant_number * min_h / transport_norm
constexpr double courant_number = 1;

// 0: central flux, 1: classical upwind flux (= Lax-Friedrichs)
constexpr double flux_alpha = 1.0;

// The final simulation time
constexpr double FINAL_TIME = 4;

// Frequency of output
constexpr double output_tick = 0.1;

// Whether to mesh the domain with Cartesian mesh elements or with curved
// elements (more memory transfer -> slower)
enum class MeshType { cartesian, deformed_cartesian };
constexpr MeshType mesh_type = MeshType::cartesian;

// Whether to set periodic boundary conditions on the domain (needs periodic
// solution as well)
constexpr bool periodic = true;

// Switch to change between a conservative formulation of the advection term
// (factor 0) or a skew-symmetric one (factor 0.5)
constexpr double factor_skew = 0.0;

// Switch to enable Gauss-Lobatto quadrature (true) or Gauss quadrature
// (false)
constexpr bool use_gl_quad = false;

// Switch to enable Gauss--Lobatto quadrature for the inverse mass
// matrix. If false, use Gauss quadrature
constexpr bool use_gl_quad_mass = false;

} // namespace DGAdvection
