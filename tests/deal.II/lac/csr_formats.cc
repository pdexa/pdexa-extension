#include <pdexa-ext/deal.II/lac/ginkgo_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <gtest/gtest.h>

#include "tests/utils/executor.h"


auto exec = create_executor<gko::EXEC>();

TEST(CsrFormats, can_create_coo) {
  dealii::GinkgoWrappers::Coo<double> m(exec, 3, 5);

  m.compress();

  EXPECT_EQ(m.m(), 3);
  EXPECT_EQ(m.n(), 5);
  EXPECT_TRUE(dynamic_cast<const gko::matrix::Coo<double>*>(m.get_gko_object()));
}

TEST(CsrFormats, can_create_ell) {
  dealii::GinkgoWrappers::Ell<double> m(exec, 3, 5);

  m.compress();

  EXPECT_EQ(m.m(), 3);
  EXPECT_EQ(m.n(), 5);
  EXPECT_TRUE(dynamic_cast<const gko::matrix::Ell<double>*>(m.get_gko_object()));
}

TEST(CsrFormats, can_create_hybrid) {
  dealii::GinkgoWrappers::Hybrid<double> m(exec, 3, 5);

  m.compress();

  EXPECT_EQ(m.m(), 3);
  EXPECT_EQ(m.n(), 5);
  EXPECT_TRUE(dynamic_cast<const gko::matrix::Hybrid<double>*>(m.get_gko_object()));
}

TEST(CsrFormats, can_create_sellp) {
  dealii::GinkgoWrappers::Sellp<double> m(exec, 3, 5);

  m.compress();

  EXPECT_EQ(m.m(), 3);
  EXPECT_EQ(m.n(), 5);
  EXPECT_TRUE(dynamic_cast<const gko::matrix::Sellp<double>*>(m.get_gko_object()));

}
