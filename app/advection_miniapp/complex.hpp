// SPDX-FileCopyrightText: 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <cuda_runtime.h>
#include <cuda/std/complex>

#include "vectorization.hpp"

namespace ext {

template<typename T>
class complex;

template<>
class alignas(alignof(double4[2])) complex<double4> {
public:
  __device__ complex() = default;

  __device__ complex(const double4 re, const double4 im = make_double4(0, 0, 0, 0)) : re(re), im(im) {}

  __device__ complex(const double& re, const double& im = 0) :
      re(make_double4(re, re, re, re)), im(make_double4(im, im, im, im)) {}

  __device__ complex(const cuda::std::complex<double>& other) : complex(other.real(), other.imag()) {}

  __device__ complex(const complex& other) : re(other.re), im(other.im) {}

  __device__ complex(complex&& other) noexcept : re(other.re), im(other.im) {}

  __device__ complex& operator=(const complex& other) {
    re = other.re;
    im = other.im;
    return *this;
  }

  __device__ complex& operator=(complex&& other) noexcept {
    re = other.re;
    im = other.im;
    return *this;
  }

#define GENERATE_MIXED_BINARY_OP(_op)                                                                                  \
  template<typename T>                                                                                                 \
  __device__ friend complex operator _op(const complex x, const T y) {                                                 \
    return x _op complex(y);                                                                                           \
  }                                                                                                                    \
  template<typename T>                                                                                                 \
  __device__ friend complex operator _op(const T x, const complex y) {                                                 \
    return complex(x) _op y;                                                                                           \
  }                                                                                                                    \
  static_assert(true, "Force semicolon")

#define GENERATE_BINARY_OP(_op)                                                                                        \
  __device__ friend complex operator _op(const complex x, const complex y) {                                           \
    complex res{};                                                                                                     \
    res.re = x.re _op y.re;                                                                                            \
    res.im = x.im _op y.im;                                                                                            \
    return res;                                                                                                        \
  }                                                                                                                    \
  GENERATE_MIXED_BINARY_OP(_op)

#define GENERATE_ASSIGNMENT_OP(_op, _base)                                                                             \
  __device__ complex& operator _op(const complex x) {                                                                  \
    *this = *this _base x;                                                                                             \
    return *this;                                                                                                      \
  }                                                                                                                    \
  static_assert(true, "Force semicolon")

  GENERATE_BINARY_OP(+);
  GENERATE_ASSIGNMENT_OP(+=, +);
  GENERATE_BINARY_OP(-);
  GENERATE_ASSIGNMENT_OP(-=, -);

  __device__ friend complex operator*(const complex x, const complex y) {
    complex res{};
    res.re = x.re * y.re - x.im * y.im;
    res.im = x.re * y.im + x.im * y.re;
    return res;
  }

  GENERATE_MIXED_BINARY_OP(*);
  GENERATE_ASSIGNMENT_OP(*=, *);

  __device__ friend complex operator/(const complex x, const complex y) {
    complex res{};
    res.re = (x.re * y.re + x.im * y.im) / (y.re * y.re + y.im * y.im);
    res.im = (x.im * y.re - x.re * y.im) / (y.re * y.re + y.im * y.im);
    return res;
  }

  GENERATE_MIXED_BINARY_OP(/);
  GENERATE_ASSIGNMENT_OP(/=, /);

#undef GENERATE_BINARY_OP
#undef GENERATE_MIXED_BINARY_OP
#undef GENERATE_ASSIGNMENT_OP

  __device__ double4& real() { return re; }

  __device__ const double4& real() const { return re; }

  __device__ double4& imag() { return im; }

  __device__ const double4& imag() const { return im; }

private:
  double4 re;
  double4 im;
};

template<>
class alignas(alignof(double2[2])) complex<double2> {
public:
  __device__ complex() = default;

  __device__ complex(const double2 re, const double2 im = make_double2(0, 0)) : re(re), im(im) {}

  __device__ complex(const double& re, const double& im = 0) : re(make_double2(re, re)), im(make_double2(im, im)) {}

  __device__ complex(const cuda::std::complex<double>& other) : complex(other.real(), other.imag()) {}

  __device__ complex(const complex& other) : re(other.re), im(other.im) {}

  __device__ complex(complex&& other) noexcept : re(other.re), im(other.im) {}

  __device__ complex& operator=(const complex& other) {
    re = other.re;
    im = other.im;
    return *this;
  }

  __device__ complex& operator=(complex&& other) noexcept {
    re = other.re;
    im = other.im;
    return *this;
  }

#define GENERATE_MIXED_BINARY_OP(_op)                                                                                  \
  template<typename T>                                                                                                 \
  __device__ friend complex operator _op(const complex x, const T y) {                                                 \
    return x _op complex(y);                                                                                           \
  }                                                                                                                    \
  template<typename T>                                                                                                 \
  __device__ friend complex operator _op(const T x, const complex y) {                                                 \
    return complex(x) _op y;                                                                                           \
  }                                                                                                                    \
  static_assert(true, "Force semicolon")

#define GENERATE_BINARY_OP(_op)                                                                                        \
  __device__ friend complex operator _op(const complex x, const complex y) {                                           \
    complex res{};                                                                                                     \
    res.re = x.re _op y.re;                                                                                            \
    res.im = x.im _op y.im;                                                                                            \
    return res;                                                                                                        \
  }                                                                                                                    \
  GENERATE_MIXED_BINARY_OP(_op)

#define GENERATE_ASSIGNMENT_OP(_op, _base)                                                                             \
  __device__ complex& operator _op(const complex x) {                                                                  \
    *this = *this _base x;                                                                                             \
    return *this;                                                                                                      \
  }                                                                                                                    \
  static_assert(true, "Force semicolon")

  GENERATE_BINARY_OP(+);
  GENERATE_ASSIGNMENT_OP(+=, +);
  GENERATE_BINARY_OP(-);
  GENERATE_ASSIGNMENT_OP(-=, -);

  __device__ friend complex operator*(const complex x, const complex y) {
    complex res{};
    res.re = x.re * y.re - x.im * y.im;
    res.im = x.re * y.im + x.im * y.re;
    return res;
  }

  GENERATE_MIXED_BINARY_OP(*);
  GENERATE_ASSIGNMENT_OP(*=, *);

  __device__ friend complex operator/(const complex x, const complex y) {
    complex res{};
    res.re = (x.re * y.re + x.im * y.im) / (y.re * y.re + y.im * y.im);
    res.im = (x.im * y.re - x.re * y.im) / (y.re * y.re + y.im * y.im);
    return res;
  }

  GENERATE_MIXED_BINARY_OP(/);
  GENERATE_ASSIGNMENT_OP(/=, /);

#undef GENERATE_BINARY_OP
#undef GENERATE_MIXED_BINARY_OP
#undef GENERATE_ASSIGNMENT_OP

  __device__ double2& real() { return re; }

  __device__ const double2& real() const { return re; }

  __device__ double2& imag() { return im; }

  __device__ const double2& imag() const { return im; }

private:
  double2 re;
  double2 im;
};

} // namespace ext
