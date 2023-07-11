#include <pdexa-ext/deal.II/lac/ginkgo_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>

#include <gtest/gtest.h>

#include "tests/utils/executor.h"


auto exec = create_executor<gko::EXEC>();
using size_type = dealii::GinkgoWrappers::Csr<double>::size_type;

TEST(CsrManip, can_set_single_value) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);

  m.set(2, 4, 4.4);
  m.compress();

  auto obj = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 1);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 4);
  EXPECT_EQ(obj->get_const_values()[0], 4.4);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 1);
}

TEST(CsrManip, can_set_full_matrix) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);
  dealii::FullMatrix<double> k(2, 2);
  k(0, 0) = 1;
  k(0, 1) = 2;
  k(1, 0) = 0;
  k(1, 1) = 4;
  std::vector<size_type> indices{0, 2};

  m.set(indices, k, false);
  m.compress();

  auto obj = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 4);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 0);
  EXPECT_EQ(obj->get_const_col_idxs()[1], 2);
  EXPECT_EQ(obj->get_const_col_idxs()[2], 0);
  EXPECT_EQ(obj->get_const_col_idxs()[3], 2);
  EXPECT_EQ(obj->get_const_values()[0], 1);
  EXPECT_EQ(obj->get_const_values()[1], 2);
  EXPECT_EQ(obj->get_const_values()[2], 0);
  EXPECT_EQ(obj->get_const_values()[3], 4);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 4);
}

TEST(CsrManip, can_set_full_matrix_elide) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);
  dealii::FullMatrix<double> k(2, 2);
  k(0, 0) = 1;
  k(0, 1) = 2;
  k(1, 0) = 0;
  k(1, 1) = 4;
  std::vector<size_type> indices{0, 2};

  m.set(indices, k, true);
  m.compress();

  auto obj = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 3);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 0);
  EXPECT_EQ(obj->get_const_col_idxs()[1], 2);
  EXPECT_EQ(obj->get_const_col_idxs()[2], 2);
  EXPECT_EQ(obj->get_const_values()[0], 1);
  EXPECT_EQ(obj->get_const_values()[1], 2);
  EXPECT_EQ(obj->get_const_values()[2], 4);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 3);
}

TEST(CsrManip, can_set_full_matrix_different_row_col_idxs) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);
  dealii::FullMatrix<double> k(2, 2);
  k(0, 0) = 1;
  k(0, 1) = 2;
  k(1, 0) = 0;
  k(1, 1) = 4;
  std::vector<size_type> row_indices{0, 2};
  std::vector<size_type> col_indices{3, 4};

  m.set(row_indices, col_indices, k, false);
  m.compress();

  auto obj = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 4);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 3);
  EXPECT_EQ(obj->get_const_col_idxs()[1], 4);
  EXPECT_EQ(obj->get_const_col_idxs()[2], 3);
  EXPECT_EQ(obj->get_const_col_idxs()[3], 4);
  EXPECT_EQ(obj->get_const_values()[0], 1);
  EXPECT_EQ(obj->get_const_values()[1], 2);
  EXPECT_EQ(obj->get_const_values()[2], 0);
  EXPECT_EQ(obj->get_const_values()[3], 4);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 4);
}

TEST(CsrManip, can_set_row) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);
  std::vector<double> values{1, 2};
  std::vector<size_type> col_indices{3, 4};

  m.set(1, col_indices, values, false);
  m.compress();

  auto obj = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 2);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 3);
  EXPECT_EQ(obj->get_const_col_idxs()[1], 4);
  EXPECT_EQ(obj->get_const_values()[0], 1);
  EXPECT_EQ(obj->get_const_values()[1], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 2);
}

TEST(CsrManip, can_set_row_raw) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);
  std::vector<double> values{1, 2};
  std::vector<size_type> col_indices{3, 4};

  m.set(1, col_indices.size(), col_indices.data(), values.data(), false);
  m.compress();

  auto obj = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 2);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 3);
  EXPECT_EQ(obj->get_const_col_idxs()[1], 4);
  EXPECT_EQ(obj->get_const_values()[0], 1);
  EXPECT_EQ(obj->get_const_values()[1], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 2);
}

TEST(CsrManip, can_add_single_value) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);

  m.add(2, 4, 1.1);
  m.add(2, 4, 4.4);
  m.compress();

  auto obj = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 1);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 4);
  EXPECT_EQ(obj->get_const_values()[0], 5.5);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 1);
}

TEST(CsrManip, can_add_full_matrix) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);
  dealii::FullMatrix<double> k(2, 2);
  k(0, 0) = 1;
  k(0, 1) = 2;
  k(1, 0) = 0;
  k(1, 1) = 4;

  m.add({0, 2}, k, false);
  m.add({1, 2}, k, false);
  m.compress();

  auto obj = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 7);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 0);
  EXPECT_EQ(obj->get_const_col_idxs()[1], 2);
  EXPECT_EQ(obj->get_const_col_idxs()[2], 1);
  EXPECT_EQ(obj->get_const_col_idxs()[3], 2);
  EXPECT_EQ(obj->get_const_col_idxs()[4], 0);
  EXPECT_EQ(obj->get_const_col_idxs()[5], 1);
  EXPECT_EQ(obj->get_const_col_idxs()[6], 2);
  EXPECT_EQ(obj->get_const_values()[0], 1);
  EXPECT_EQ(obj->get_const_values()[1], 2);
  EXPECT_EQ(obj->get_const_values()[2], 1);
  EXPECT_EQ(obj->get_const_values()[3], 2);
  EXPECT_EQ(obj->get_const_values()[4], 0);
  EXPECT_EQ(obj->get_const_values()[5], 0);
  EXPECT_EQ(obj->get_const_values()[6], 8);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 4);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 7);
}

TEST(CsrManip, can_add_full_matrix_elide) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);
  dealii::FullMatrix<double> k(2, 2);
  k(0, 0) = 1;
  k(0, 1) = 2;
  k(1, 0) = 0;
  k(1, 1) = 4;

  m.add({0, 2}, k, true);
  m.add({1, 2}, k, true);
  m.compress();

  auto obj = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 5);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 0);
  EXPECT_EQ(obj->get_const_col_idxs()[1], 2);
  EXPECT_EQ(obj->get_const_col_idxs()[2], 1);
  EXPECT_EQ(obj->get_const_col_idxs()[3], 2);
  EXPECT_EQ(obj->get_const_col_idxs()[4], 2);
  EXPECT_EQ(obj->get_const_values()[0], 1);
  EXPECT_EQ(obj->get_const_values()[1], 2);
  EXPECT_EQ(obj->get_const_values()[2], 1);
  EXPECT_EQ(obj->get_const_values()[3], 2);
  EXPECT_EQ(obj->get_const_values()[4], 8);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 4);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 5);
}

TEST(CsrManip, can_add_full_matrix_different_row_col_idxs) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);
  dealii::FullMatrix<double> k(2, 2);
  k(0, 0) = 1;
  k(0, 1) = 2;
  k(1, 0) = 0;
  k(1, 1) = 4;
  std::vector<size_type> row_indices{0, 2};
  std::vector<size_type> col_indices{3, 4};

  m.add(row_indices, col_indices, k, false);
  m.compress();

  auto obj = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 4);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 3);
  EXPECT_EQ(obj->get_const_col_idxs()[1], 4);
  EXPECT_EQ(obj->get_const_col_idxs()[2], 3);
  EXPECT_EQ(obj->get_const_col_idxs()[3], 4);
  EXPECT_EQ(obj->get_const_values()[0], 1);
  EXPECT_EQ(obj->get_const_values()[1], 2);
  EXPECT_EQ(obj->get_const_values()[2], 0);
  EXPECT_EQ(obj->get_const_values()[3], 4);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 4);
}

TEST(CsrManip, can_add_row) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);
  std::vector<double> values{1, 2};
  std::vector<size_type> col_indices{3, 4};

  m.add(1, col_indices, values, false);
  m.compress();

  auto obj = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 2);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 3);
  EXPECT_EQ(obj->get_const_col_idxs()[1], 4);
  EXPECT_EQ(obj->get_const_values()[0], 1);
  EXPECT_EQ(obj->get_const_values()[1], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 2);
}

TEST(CsrManip, can_add_row_raw) {
  dealii::GinkgoWrappers::Csr<double> m(exec, 3, 5);
  std::vector<double> values{1, 2};
  std::vector<size_type> col_indices{3, 4};

  m.add(1, col_indices.size(), col_indices.data(), values.data(), false);
  m.compress();

  auto obj = gko::clone(exec->get_master(), m.get_gko_object());
  EXPECT_EQ(obj->get_num_stored_elements(), 2);
  EXPECT_EQ(obj->get_const_col_idxs()[0], 3);
  EXPECT_EQ(obj->get_const_col_idxs()[1], 4);
  EXPECT_EQ(obj->get_const_values()[0], 1);
  EXPECT_EQ(obj->get_const_values()[1], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[0], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[1], 0);
  EXPECT_EQ(obj->get_const_row_ptrs()[2], 2);
  EXPECT_EQ(obj->get_const_row_ptrs()[3], 2);
}
