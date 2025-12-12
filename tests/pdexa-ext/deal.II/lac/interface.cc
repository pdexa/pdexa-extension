// SPDX-FileCopyrightText: 2023 - 2025 Marcel Koch, KIT
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <gtest/gtest.h>

// all include files you need here
#include <deal.II/base/utilities.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <pdexa-ext/deal.II/lac/ginkgo_interface.h>

#include <Kokkos_Random.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>

using namespace dealii;

class Interface : public testing::Test {
protected:
  void SetUp() override {
    default_exec = gko::ext::kokkos::create_default_executor();
    host_exec = gko::ext::kokkos::create_default_host_executor();
  }

  void TearDown() override {
    default_exec.reset();
    host_exec.reset();
  }

  std::shared_ptr<gko::Executor> default_exec;
  std::shared_ptr<gko::Executor> host_exec;
};

template<typename ValueType, typename MemorySpaceType, typename OtherValueType>
bool check_equality(const ArrayView<ValueType, MemorySpaceType>& view, gko::matrix::Dense<OtherValueType>* vector) {
  using exec_space = typename MemorySpaceType::kokkos_space::execution_space;
  auto view_data = view.data();
  auto vector_data = vector->get_values();
  bool success = true;
  Kokkos::parallel_reduce(
    Kokkos::RangePolicy<exec_space>(0, view.size()),
    KOKKOS_LAMBDA(const int i, bool& lsuccess) { lsuccess = lsuccess && (view_data[i] == vector_data[i]); },
    Kokkos::LAnd<bool, exec_space>{success});
  return success;
}

template<typename ValueType, typename MemorySpaceType>
bool check_equality(const LinearAlgebra::distributed::Vector<ValueType, MemorySpaceType>& deal_vector,
                    gko::matrix::Dense<ValueType>* vector) {
  auto view = ArrayView<const ValueType, MemorySpaceType>(deal_vector.begin(), deal_vector.locally_owned_size());
  return check_equality(view, vector);
}

template<typename View>
void fill_view(View view) {
  using exec_space = typename View::execution_space;
  Kokkos::Random_XorShift1024_Pool<exec_space> pool(/*seed=*/12345);
  Kokkos::parallel_for(
    Kokkos::RangePolicy<exec_space>(0, view.size()), KOKKOS_LAMBDA(const int i) {
      auto generator = pool.get_state();
      view(i) = generator.drand(0., 1.);
      pool.free_state(generator);
    });
}

TEST_F(Interface, can_create_array_view_host) {
  Kokkos::View<double[5], MemorySpace::Host::kokkos_space> view{"name"};
  ArrayView<double, MemorySpace::Host> array_view(view.data(), view.size());
  fill_view(view);

  auto vector = GinkgoInterface::detail::create_vector(host_exec, array_view);

  ASSERT_EQ(vector->get_values(), view.data());
  check_equality(array_view, vector.get());
}

TEST_F(Interface, can_create_array_view_host_const) {
  using array_view_type = ArrayView<const double, MemorySpace::Host>;
  Kokkos::View<double[5], MemorySpace::Host::kokkos_space> view{"name"};
  array_view_type array_view(view.data(), view.size());
  fill_view(view);

  auto vector = GinkgoInterface::detail::create_const_vector(host_exec, array_view);

  using gko_type = typename decltype(vector)::element_type;
  ASSERT_TRUE(std::is_const_v<gko_type>);
  ASSERT_EQ(vector->get_const_values(), view.data());
}

TEST_F(Interface, can_create_array_view_default) {
  Kokkos::View<double[5], MemorySpace::Default::kokkos_space> view{"name"};
  ArrayView<double, MemorySpace::Default> array_view(view.data(), view.size());
  fill_view(view);

  auto vector = GinkgoInterface::detail::create_vector(default_exec, array_view);

  ASSERT_EQ(vector->get_values(), array_view.begin());
  check_equality(array_view, vector.get());
}

TEST_F(Interface, throws_on_incompatible_executor) {
  if (std::is_same_v<MemorySpace::Host::kokkos_space, MemorySpace::Default::kokkos_space>) { GTEST_SKIP(); }

  Kokkos::View<double[5], MemorySpace::Default::kokkos_space> view{"name"};
  ArrayView<double, MemorySpace::Default> array_view(view.data(), view.size());

  ASSERT_THROW(GinkgoInterface::detail::create_vector(host_exec, array_view), ExcMessage);
}

TEST_F(Interface, can_create_vector) {
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> deal_vector(5);
  deal_vector = 1.1;

  auto vector = GinkgoInterface::create_vector(default_exec, deal_vector);

  ASSERT_EQ(vector->get_values(), deal_vector.begin());
  check_equality(deal_vector, vector.get());
}

TEST_F(Interface, can_create_vector_const) {
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> deal_vector(5);
  deal_vector = 1.1;

  auto vector = GinkgoInterface::create_vector(
    default_exec, static_cast<const LinearAlgebra::distributed::Vector<double, MemorySpace::Default>&>(deal_vector));

  using gko_type = typename decltype(vector)::element_type;
  ASSERT_TRUE(std::is_const_v<gko_type>);
  ASSERT_EQ(vector->get_const_values(), deal_vector.begin());
}

TEST_F(Interface, can_create_csr) {
  std::vector<std::vector<unsigned int>> col_idxs{{0, 3}, {1}, {0, 2}};
  SparsityPattern pattern;
  pattern.copy_from(3, 4, col_idxs.begin(), col_idxs.end());
  SparseMatrix<double> dealii_obj(pattern);
  dealii_obj.set(0, 0, 1);
  dealii_obj.set(0, 3, 2);
  dealii_obj.set(1, 1, 3);
  dealii_obj.set(2, 0, 4);
  dealii_obj.set(2, 2, 5);

  auto obj = GinkgoInterface::create_csr_matrix(host_exec, dealii_obj);

  ASSERT_EQ(obj->get_num_stored_elements(), 5);
  ASSERT_EQ(obj->get_const_col_idxs()[0], 0);
  ASSERT_EQ(obj->get_const_col_idxs()[1], 3);
  ASSERT_EQ(obj->get_const_col_idxs()[2], 1);
  ASSERT_EQ(obj->get_const_col_idxs()[3], 0);
  ASSERT_EQ(obj->get_const_col_idxs()[4], 2);
  ASSERT_EQ(obj->get_const_values()[0], 1);
  ASSERT_EQ(obj->get_const_values()[1], 2);
  ASSERT_EQ(obj->get_const_values()[2], 3);
  ASSERT_EQ(obj->get_const_values()[3], 4);
  ASSERT_EQ(obj->get_const_values()[4], 5);
  ASSERT_EQ(obj->get_const_row_ptrs()[0], 0);
  ASSERT_EQ(obj->get_const_row_ptrs()[1], 2);
  ASSERT_EQ(obj->get_const_row_ptrs()[2], 3);
  ASSERT_EQ(obj->get_const_row_ptrs()[3], 5);
}

TEST_F(Interface, can_create_csr_with_strategy) {
  std::vector<std::vector<unsigned int>> col_idxs{{0, 3}, {1}, {0, 2}};
  SparsityPattern pattern;
  pattern.copy_from(3, 4, col_idxs.begin(), col_idxs.end());
  SparseMatrix<double> dealii_obj(pattern);
  dealii_obj.set(0, 0, 1);
  dealii_obj.set(0, 3, 2);
  dealii_obj.set(1, 1, 3);
  dealii_obj.set(2, 0, 4);
  dealii_obj.set(2, 2, 5);

  auto obj = GinkgoInterface::create_csr_matrix(host_exec, dealii_obj, GinkgoInterface::csr_strategy::merge_path);

  using csr_type = std::remove_cv_t<std::remove_reference_t<decltype(obj)>>::element_type;
  ASSERT_TRUE(std::dynamic_pointer_cast<typename csr_type::merge_path>(obj->get_strategy()));
}

TEST_F(Interface, can_create_csr_with_index_type) {
  std::vector<std::vector<unsigned int>> col_idxs{{0, 3}, {1}, {0, 2}};
  SparsityPattern pattern;
  pattern.copy_from(3, 4, col_idxs.begin(), col_idxs.end());
  SparseMatrix<double> dealii_obj(pattern);
  dealii_obj.set(0, 0, 1);
  dealii_obj.set(0, 3, 2);
  dealii_obj.set(1, 1, 3);
  dealii_obj.set(2, 0, 4);
  dealii_obj.set(2, 2, 5);

  auto obj =
    GinkgoInterface::create_csr_matrix<gko::int64>(host_exec, dealii_obj, GinkgoInterface::csr_strategy::merge_path);

  using csr_type = std::remove_cv_t<std::remove_reference_t<decltype(obj)>>::element_type;
  auto same_index_type = std::is_same_v<typename csr_type::index_type, gko::int64>;
  ASSERT_TRUE(same_index_type);
}

TEST_F(Interface, can_create_csr_with_value_index_type) {
  std::vector<std::vector<unsigned int>> col_idxs{{0, 3}, {1}, {0, 2}};
  SparsityPattern pattern;
  pattern.copy_from(3, 4, col_idxs.begin(), col_idxs.end());
  SparseMatrix<double> dealii_obj(pattern);
  dealii_obj.set(0, 0, 1);
  dealii_obj.set(0, 3, 2);
  dealii_obj.set(1, 1, 3);
  dealii_obj.set(2, 0, 4);
  dealii_obj.set(2, 2, 5);

  auto obj = GinkgoInterface::create_csr_matrix<float, gko::int64>(host_exec, dealii_obj,
                                                                   GinkgoInterface::csr_strategy::merge_path);

  using csr_type = std::remove_cv_t<std::remove_reference_t<decltype(obj)>>::element_type;
  auto same_value_type = std::is_same_v<typename csr_type::value_type, float>;
  auto same_index_type = std::is_same_v<typename csr_type::index_type, gko::int64>;
  ASSERT_TRUE(same_value_type);
  ASSERT_TRUE(same_index_type);
}

template<typename MemorySpace>
struct linop_data {
  linop_data() {
    std::vector<std::vector<unsigned int>> col_idxs{{0, 3}, {1}, {0, 2}};
    pattern.copy_from(3, 4, col_idxs.begin(), col_idxs.end());

    dealii_obj = SparseMatrix<double>(pattern);
    dealii_obj.set(0, 0, 1);
    dealii_obj.set(0, 3, 2);
    dealii_obj.set(1, 1, 3);
    dealii_obj.set(2, 0, 4);
    dealii_obj.set(2, 2, 5);

    exec = gko::ext::kokkos::create_executor(typename MemorySpace::kokkos_space::execution_space{});
    gko_obj = gko::share(GinkgoInterface::create_csr_matrix(exec, dealii_obj));
  }

  SparsityPattern pattern;
  SparseMatrix<double> dealii_obj;

  std::shared_ptr<gko::Executor> exec;
  std::shared_ptr<gko::LinOp> gko_obj;
};

template<typename MemorySpace>
struct linop_vectors {
  using Vector = LinearAlgebra::distributed::Vector<double, MemorySpace>;
  using VectorHost = LinearAlgebra::distributed::Vector<double, ::MemorySpace::Host>;

  linop_vectors(size_t n, size_t m) {
    auto init_vectors = [](auto& x, auto& x_ref, auto size) {
      auto get_view = [](auto& vector) {
        return Kokkos::View<double*, typename MemorySpace::kokkos_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
          vector.begin(), vector.size()};
      };

      x.reinit(size);
      auto view_x = get_view(x);
      fill_view(view_x);

      x_ref.reinit(size);
      auto view_x_ref = Kokkos::create_mirror(view_x);
      Kokkos::deep_copy(view_x, view_x_ref);
    };

    init_vectors(u, u_ref, n);
    init_vectors(v, v_ref, m);
  }

  Vector u;
  Vector v;
  VectorHost u_ref;
  VectorHost v_ref;
};

TEST_F(Interface, can_create_linop_host) {
  linop_data<MemorySpace::Host> data{};

  auto linop = GinkgoInterface::linear_operator(data.gko_obj);

  {
    SCOPED_TRACE("vmult");
    linop_vectors<MemorySpace::Host> vectors(data.dealii_obj.n(), data.dealii_obj.m());

    linop.vmult(vectors.v, vectors.u);
    data.dealii_obj.vmult(vectors.v_ref, vectors.u_ref);

    vectors.v_ref -= vectors.v;
    ASSERT_LT(vectors.v_ref.linfty_norm(), 1e-12);
  }
  {
    SCOPED_TRACE("vmult_add");
    linop_vectors<MemorySpace::Host> vectors(data.dealii_obj.n(), data.dealii_obj.m());

    linop.vmult_add(vectors.v, vectors.u);
    data.dealii_obj.vmult_add(vectors.v_ref, vectors.u_ref);

    vectors.v_ref -= vectors.v;
    ASSERT_LT(vectors.v_ref.linfty_norm(), 1e-12);
  }
  {
    SCOPED_TRACE("Tvmult");
    linop_vectors<MemorySpace::Host> vectors(data.dealii_obj.n(), data.dealii_obj.m());

    linop.Tvmult(vectors.u, vectors.v);
    data.dealii_obj.Tvmult(vectors.u_ref, vectors.v_ref);

    vectors.u_ref -= vectors.u;
    ASSERT_LT(vectors.u_ref.linfty_norm(), 1e-12);
  }
  {
    SCOPED_TRACE("Tvmult_add");
    linop_vectors<MemorySpace::Host> vectors(data.dealii_obj.n(), data.dealii_obj.m());

    linop.Tvmult_add(vectors.u, vectors.v);
    data.dealii_obj.Tvmult_add(vectors.u_ref, vectors.v_ref);

    vectors.u_ref -= vectors.u;
    ASSERT_LT(vectors.u_ref.linfty_norm(), 1e-12);
  }
}

TEST_F(Interface, can_create_linop_default) {
  linop_data<MemorySpace::Default> data{};
  linop_data<MemorySpace::Host> data_host{};

  auto linop =
    GinkgoInterface::linear_operator<double, double, MemorySpace::Default, MemorySpace::Default>(data.gko_obj);
  auto host_linop = GinkgoInterface::linear_operator(data_host.gko_obj);

  auto compare = [&](const auto& u, const auto& u_ref) {
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> mirror(u_ref.size());

    default_exec->copy_from(host_exec, mirror.size(), u_ref.begin(), mirror.begin());

    mirror -= u;
    ASSERT_LT(mirror.linfty_norm(), 1e-12);
  };
  {
    SCOPED_TRACE("vmult");
    linop_vectors<MemorySpace::Default> vectors(data.dealii_obj.n(), data.dealii_obj.m());

    linop.vmult(vectors.v, vectors.u);
    host_linop.vmult(vectors.v_ref, vectors.u_ref);

    compare(vectors.v, vectors.v_ref);
  }
  {
    SCOPED_TRACE("vmult_add");
    linop_vectors<MemorySpace::Default> vectors(data.dealii_obj.n(), data.dealii_obj.m());

    linop.vmult_add(vectors.v, vectors.u);
    host_linop.vmult_add(vectors.v_ref, vectors.u_ref);

    compare(vectors.v, vectors.v_ref);
  }
  {
    SCOPED_TRACE("Tvmult");
    linop_vectors<MemorySpace::Default> vectors(data.dealii_obj.n(), data.dealii_obj.m());

    linop.Tvmult(vectors.u, vectors.v);
    host_linop.Tvmult(vectors.u_ref, vectors.v_ref);

    compare(vectors.v, vectors.v_ref);
  }
  {
    SCOPED_TRACE("Tvmult_add");
    linop_vectors<MemorySpace::Default> vectors(data.dealii_obj.n(), data.dealii_obj.m());

    linop.Tvmult_add(vectors.u, vectors.v);
    host_linop.Tvmult_add(vectors.u_ref, vectors.v_ref);

    compare(vectors.v, vectors.v_ref);
  }
}

TEST_F(Interface, can_create_inverse_linop) {
  auto gko_obj = gko::share(
    gko::initialize<gko::matrix::Csr<>>({{2, -1, 0, 0}, {-1, 2, -1, 0}, {0, -1, 2, -1}, {0, 0, -1, 2}}, default_exec));
  auto linear_operator =
    GinkgoInterface::linear_operator<double, double, MemorySpace::Default, MemorySpace::Default>(gko_obj);
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> u(gko_obj->get_size()[0]);
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> v(gko_obj->get_size()[0]);
  u = 1.0;
  v = 0.0;
  linear_operator.vmult(v, u);

  auto inverse_operator = GinkgoInterface::inverse_operator<double, double, MemorySpace::Default, MemorySpace::Default>(
    gko_obj,
    gko::solver::Cg<>::build().with_criteria(gko::stop::Iteration::build().with_max_iters(4u)).on(default_exec));
  auto solution = u;
  u = 0.0;
  inverse_operator.vmult(u, v);

  solution -= u;
  ASSERT_LT(solution.linfty_norm(), 1e-12);
}

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  auto result = RUN_ALL_TESTS();
  return result;
}
