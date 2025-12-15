// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <cuda_runtime.h>

#include <ginkgo/core/base/math.hpp>


template<int vector_size>
struct VectorizedNumber {};
template<>
struct VectorizedNumber<1> {
  using type = double;
  constexpr static int size = 1;
};
template<>
struct VectorizedNumber<2> {
  using type = double2;
  constexpr static int size = 2;
};
template<>
struct VectorizedNumber<4> {
  using type = double4;
  constexpr static int size = 4;
};
template<int vector_size>
using VectorizedNumber_t = typename VectorizedNumber<vector_size>::type;


// using references as parameters can lead to a cuda error ("too many resources requested for launch")
#define GENERATE_DOUBLE4_BINARY_OP(_op)                                                                                \
  inline __device__ double4 operator _op(double4 x, double4 y) {                                                       \
    double4 res{};                                                                                                     \
    res.x = static_cast<double>(x.x _op y.x);                                                                          \
    res.y = static_cast<double>(x.y _op y.y);                                                                          \
    res.z = static_cast<double>(x.z _op y.z);                                                                          \
    res.w = static_cast<double>(x.w _op y.w);                                                                          \
    return res;                                                                                                        \
  }                                                                                                                    \
  inline __device__ double4 operator _op(double4 x, double y) { return x _op make_double4(y, y, y, y); }               \
  inline __device__ double4 operator _op(double x, double4 y) { return make_double4(x, x, x, x) _op y; }               \
  static_assert(true, "Force semicolon")

GENERATE_DOUBLE4_BINARY_OP(*);
GENERATE_DOUBLE4_BINARY_OP(+);
GENERATE_DOUBLE4_BINARY_OP(-);
GENERATE_DOUBLE4_BINARY_OP(/);
GENERATE_DOUBLE4_BINARY_OP(<);
GENERATE_DOUBLE4_BINARY_OP(<=);
GENERATE_DOUBLE4_BINARY_OP(>);
GENERATE_DOUBLE4_BINARY_OP(>=);
GENERATE_DOUBLE4_BINARY_OP(==);
GENERATE_DOUBLE4_BINARY_OP(!=);

#undef GENERATE_DOUBLE4_BINARY_OP

#define GENERATE_DOUBLE4_ASSIGNMENT_OP(_op, _base_op)                                                                  \
  inline __device__ double4& operator _op(double4& x, const double4 y) {                                            \
    x = x _base_op y;                                                                                                  \
    return x;                                                                                                          \
  }                                                                                                                    \
  static_assert(true, "Force semicolon")

GENERATE_DOUBLE4_ASSIGNMENT_OP(+=, +);
GENERATE_DOUBLE4_ASSIGNMENT_OP(*=, *);
GENERATE_DOUBLE4_ASSIGNMENT_OP(-=, -);
GENERATE_DOUBLE4_ASSIGNMENT_OP(/=, /);

#undef GENERATE_DOUBLE4_ASSIGNMENT_OP

#define GENERATE_DOUBLE2_BINARY_OP(_op)                                                                                \
  inline __device__ double2 operator _op(double2 x, double2 y) {                                                       \
    double2 res{};                                                                                                     \
    res.x = static_cast<double>(x.x _op y.x);                                                                          \
    res.y = static_cast<double>(x.y _op y.y);                                                                          \
    return res;                                                                                                        \
  }                                                                                                                    \
  inline __device__ double2 operator _op(double2 x, double y) { return x _op make_double2(y, y); }                     \
  inline __device__ double2 operator _op(double x, double2 y) { return make_double2(x, x) _op y; }                     \
  static_assert(true, "Force semicolon")

GENERATE_DOUBLE2_BINARY_OP(*);
GENERATE_DOUBLE2_BINARY_OP(+);
GENERATE_DOUBLE2_BINARY_OP(-);
GENERATE_DOUBLE2_BINARY_OP(/);
GENERATE_DOUBLE2_BINARY_OP(<);
GENERATE_DOUBLE2_BINARY_OP(<=);
GENERATE_DOUBLE2_BINARY_OP(>);
GENERATE_DOUBLE2_BINARY_OP(>=);
GENERATE_DOUBLE2_BINARY_OP(==);
GENERATE_DOUBLE2_BINARY_OP(!=);

#undef GENERATE_DOUBLE2_BINARY_OP

#define GENERATE_DOUBLE2_ASSIGNMENT_OP(_op, _base_op)                                                                  \
inline __device__ double2& operator _op(double2& x, const double2 y) {                                            \
x = x _base_op y;                                                                                                  \
return x;                                                                                                          \
}                                                                                                                    \
static_assert(true, "Force semicolon")

GENERATE_DOUBLE2_ASSIGNMENT_OP(+=, +);
GENERATE_DOUBLE2_ASSIGNMENT_OP(*=, *);
GENERATE_DOUBLE2_ASSIGNMENT_OP(-=, -);
GENERATE_DOUBLE2_ASSIGNMENT_OP(/=, /);

#undef GENERATE_DOUBLE2_ASSIGNMENT_OP

inline __device__ double4 abs(double4 x) {
  return make_double4(std::abs(x.x), std::abs(x.y), std::abs(x.z), std::abs(x.w));
}

inline __device__ double2 abs(double2 x) {
  return make_double2(std::abs(x.x), std::abs(x.y));
}

namespace gko {
template<>
inline __device__ double4 one<double4>() {
  return make_double4(1.0, 1.0, 1.0, 1.0);
}
template<>
inline __device__ double4 zero<double4>() {
  return make_double4(0.0, 0.0, 0.0, 0.0);
}

template<>
inline __device__ double2 one<double2>() {
  return make_double2(1.0, 1.0);
}
template<>
inline __device__ double2 zero<double2>() {
  return make_double2(0.0, 0.0);
}
} // namespace gko
