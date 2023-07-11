#include <ginkgo/core/base/executor.hpp>
#include <pdexa-ext/deal.II/lac/ginkgo_sparse_matrix.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <gtest/gtest.h>

#include "tests/utils/executor.h"


auto exec = create_executor<gko::EXEC>();

TEST(Csr, can_create_with_size) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);

  EXPECT_EQ(m.m(), 3);
  EXPECT_EQ(m.n(), 5);
}

TEST(Csr, can_create_from_ginkgo_object) {
  auto csr =
      gko::initialize<gko::matrix::Csr<double>>({{1, 2, 3}, {4, 5, 6}, {6, 7, 8}},
                                                exec);
  auto orig_csr = gko::clone(exec->get_master(), csr);

  dealii::GinkgoWrappers::Csr<double> m(std::move(csr));

  auto host_m = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(m.get_gko_object()->get_size(), orig_csr->get_size());
  EXPECT_EQ(m.get_gko_object()->get_num_stored_elements(), orig_csr->get_num_stored_elements());
  for (size_t i = 0; i < orig_csr->get_size()[0]; ++i) {
    EXPECT_EQ(orig_csr->get_row_ptrs()[i], host_m->get_const_row_ptrs()[i]);
  }
  for (size_t k = 0; k < orig_csr->get_num_stored_elements(); ++k) {
    EXPECT_EQ(orig_csr->get_col_idxs()[k], host_m->get_const_col_idxs()[k]);
    EXPECT_EQ(orig_csr->get_values()[k], host_m->get_const_values()[k]);
  }
}

TEST(Csr, can_create_from_dealii_object) {
  std::vector<std::vector<unsigned int>> col_idxs{{0, 3}, {1}, {0, 2}};
  dealii::SparsityPattern pattern;
  pattern.copy_from(3, 4, col_idxs.begin(), col_idxs.end());

  dealii::SparseMatrix<double> dealii_obj(pattern);
  dealii_obj.set(0, 0, 1);
  dealii_obj.set(0, 3, 2);
  dealii_obj.set(1, 1, 3);
  dealii_obj.set(2, 0, 4);
  dealii_obj.set(2, 2, 5);

  dealii::GinkgoWrappers::Csr<double> csr(exec, dealii_obj);

  auto obj = gko::clone(exec->get_master(), csr.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 5);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 0);
  EXPECT_EQ(obj->get_const_col_idxs()[1], 3);
  EXPECT_EQ(obj->get_const_col_idxs()[2], 1);
  EXPECT_EQ(obj->get_const_col_idxs()[3], 0);
  EXPECT_EQ(obj->get_const_col_idxs()[4], 2);
  EXPECT_EQ(obj->get_const_values()[0], 1);
  EXPECT_EQ(obj->get_const_values()[1], 2);
  EXPECT_EQ(obj->get_const_values()[2], 3);
  EXPECT_EQ(obj->get_const_values()[3], 4);
  EXPECT_EQ(obj->get_const_values()[4], 5);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 3);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 5);
}

TEST(Csr, can_vmult) {
  dealii::GinkgoWrappers::Vector<double> v(exec, {1, 1, 1});
  dealii::GinkgoWrappers::Vector<double> u(exec, {1, 1, 1});
  dealii::GinkgoWrappers::Csr<double> m(
      gko::initialize<gko::matrix::Csr<double>>({{1, 2, 3}, {4, 5, 6}, {6, 7, 8}},
                                                exec));

  m.vmult(u, v);

  auto host_u = gko::clone(exec->get_master(), u.get_gko_object());
  EXPECT_EQ(host_u->at(0), 6);
  EXPECT_EQ(host_u->at(1), 15);
  EXPECT_EQ(host_u->at(2), 21);
}

TEST(Csr, can_vmult_add) {
  dealii::GinkgoWrappers::Vector<double> v(exec, {1, 1, 1});
  dealii::GinkgoWrappers::Vector<double> u(exec, {1, 1, 1});
  dealii::GinkgoWrappers::Csr<double> m(
      gko::initialize<gko::matrix::Csr<double>>({{1, 2, 3}, {4, 5, 6}, {6, 7, 8}},
                                                exec));

  m.vmult_add(u, v);

  auto host_u = gko::clone(exec->get_master(), u.get_gko_object());
  EXPECT_EQ(host_u->at(0), 7);
  EXPECT_EQ(host_u->at(1), 16);
  EXPECT_EQ(host_u->at(2), 22);
}

TEST(Csr, can_tvmult) {
  dealii::GinkgoWrappers::Vector<double> v(exec, {1, 1, 1});
  dealii::GinkgoWrappers::Vector<double> u(exec, {1, 1, 1});
  dealii::GinkgoWrappers::Csr<double> m(
      gko::initialize<gko::matrix::Csr<double>>({{1, 2, 3}, {4, 5, 6}, {6, 7, 8}},
                                                exec));

  m.Tvmult(u, v);

  auto host_u = gko::clone(exec->get_master(), u.get_gko_object());
  EXPECT_EQ(host_u->at(0), 11);
  EXPECT_EQ(host_u->at(1), 14);
  EXPECT_EQ(host_u->at(2), 17);
}

TEST(Csr, can_tvmult_add) {
  dealii::GinkgoWrappers::Vector<double> v(exec, {1, 1, 1});
  dealii::GinkgoWrappers::Vector<double> u(exec, {1, 1, 1});
  dealii::GinkgoWrappers::Csr<double> m(
      gko::initialize<gko::matrix::Csr<double>>({{1, 2, 3}, {4, 5, 6}, {6, 7, 8}},
                                                exec));

  m.Tvmult_add(u, v);

  auto host_u = gko::clone(exec->get_master(), u.get_gko_object());
  EXPECT_EQ(host_u->at(0), 12);
  EXPECT_EQ(host_u->at(1), 15);
  EXPECT_EQ(host_u->at(2), 18);
}

#if !defined(COMPILE_REFERENCE) && !defined(COMPILE_OMP)

TEST(CsrDeathTest, has_device_memory) {
  dealii::GinkgoWrappers::Csr<double> m(gko::initialize<gko::matrix::Csr<double>>({{1, 2, 3}, {4, 5, 6}, {6, 7, 8}},
                                                                                  exec));

  EXPECT_DEATH({
                 *m.get_gko_object()->get_const_values();
               }, "");
}

#endif
