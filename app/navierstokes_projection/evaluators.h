#pragma once 
#include <deal.II/base/quadrature_lib.h>

using namespace dealii;

template <int dim, typename Number, int n_components = dim>
Tensor<1, n_components, VectorizedArray<Number>>
evaluate_function(const Function<dim>                       &function,
                  const Point<dim, VectorizedArray<Number>> &p_vectorized)
{
  AssertDimension(function.n_components, n_components);
  Tensor<1, n_components, VectorizedArray<Number>> result;
  for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = p_vectorized[d][v];
      for (unsigned int d = 0; d < n_components; ++d)
        result[d][v] = function.value(p, d);
    }
  return result;
}



template <int dim, typename Number>
VectorizedArray<Number>
evaluate_scalar_function(const Function<dim>                       &function,
                         const Point<dim, VectorizedArray<Number>> &p_vectorized)
{
  AssertDimension(function.n_components, 1);
  VectorizedArray<Number> result;
  for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = p_vectorized[d][v];
      result[v] = function.value(p);
    }
  return result;
}


template <int dim, typename number, int n_components = dim>
Tensor<2, n_components, VectorizedArray<number>>
evaluate_tensor_function(const Function<dim>                       &function,
                         const Point<dim, VectorizedArray<number>> &p_vectorized)
{
  Tensor<2, n_components, VectorizedArray<number>> result;
  for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = p_vectorized[d][v];
      for (unsigned int d = 0; d < n_components; ++d)
        {
          auto func_eval = function.gradient(p, d);
          for (unsigned int e = 0; e < dim; ++e)
            result[d][e][v] = func_eval[e];
        }
    }
  return result;
}