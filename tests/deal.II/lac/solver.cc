// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2022 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// test the Ginkgo CG solver

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.h>
#include <pdexa-ext/deal.II/lac/ginkgo_sparse_matrix.h>
#include <pdexa-ext/deal.II/lac/ginkgo_solver.h>
#include <pdexa-ext/deal.II/lac/ginkgo_preconditioner.h>

#include <ginkgo/core/preconditioner/jacobi.hpp>

#include "tests/utils/executor.h"
#include "tests/utils/testmatrix.h"

#include <gtest/gtest.h>


class Solver : public ::testing::Test {
protected:
  Solver()
      : control(200, 1e-3),
        A(exec, dim, dim),
        f(exec, dim),
        u(exec, dim) {
    FDMatrix testproblem(size, size);
    dealii::SparsityPattern structure(dim, dim, 5);
    testproblem.five_point_structure(structure);
    structure.compress();
    testproblem.five_point(A);
    A.compress();

    f = 1.0;
  }

  const unsigned int size = 32;
  unsigned int dim = (size - 1) * (size - 1);

  std::shared_ptr<gko::Executor> exec = create_executor<gko::EXEC>();

  dealii::SolverControl control;

  dealii::GinkgoWrappers::Csr<double> A;
  dealii::GinkgoWrappers::Vector<double> f;
  dealii::GinkgoWrappers::Vector<double> u;
};

TEST_F(Solver, can_solve_cg) {
  dealii::GinkgoWrappers::SolverCG<double> solver(exec, control);

  u = 0.0;
  solver.solve(A, u, f);

  EXPECT_GE(control.last_step(), 35);
  EXPECT_LE(control.last_step(), 39);
}

TEST_F(Solver, can_solve_bicgstab) {
  dealii::GinkgoWrappers::SolverBicgstab<double> solver(exec, control);

  u = 0.0;
  solver.solve(A, u, f);

  EXPECT_GE(control.last_step(), 26);
  EXPECT_LE(control.last_step(), 65);
}

TEST_F(Solver, can_solve_cgs) {
  dealii::GinkgoWrappers::SolverCGS<double> solver(exec, control);

  u = 0.0;
  solver.solve(A, u, f);

  EXPECT_GE(control.last_step(), 36);
  EXPECT_LE(control.last_step(), 79);
}

TEST_F(Solver, can_solve_fcg) {
  dealii::GinkgoWrappers::SolverFCG<double> solver(exec, control);

  u = 0.0;
  solver.solve(A, u, f);

  EXPECT_GE(control.last_step(), 33);
  EXPECT_LE(control.last_step(), 39);
}

TEST_F(Solver, can_solve_gmres) {
  dealii::GinkgoWrappers::SolverGMRES<double> solver(exec, control);

  u = 0.0;
  solver.solve(A, u, f);

  EXPECT_GE(control.last_step(), 20);
  EXPECT_LE(control.last_step(), 49);
}

TEST_F(Solver, can_solve_ir_inner_cg) {
  dealii::GinkgoWrappers::SolverIR<double> solver(exec, control);
  dealii::GinkgoWrappers::PreconditionBase<double, int> inner_cg(
      A,
      gko::solver::Cg<>::build()
          .with_criteria(gko::stop::Iteration::build().with_max_iters(45u).on(
                             exec),
                         gko::stop::ResidualNormReduction<>::build()
                             .with_reduction_factor(1e-5)
                             .on(exec))
          .on(exec));

  u = 0.0;
  solver.solve(A, u, f, inner_cg);

  EXPECT_GE(control.last_step(), 0);
  EXPECT_LE(control.last_step(), 2);
}

TEST_F(Solver, can_solve_cg_with_jacobi) {
  dealii::GinkgoWrappers::SolverCG<double> solver(exec, control);
  dealii::GinkgoWrappers::PreconditionJacobi<double, int> jacobi(A);

  u = 0.0;
  solver.solve(A, u, f, jacobi);

  EXPECT_GE(control.last_step(), 29);
  EXPECT_LE(control.last_step(), 33);
}
