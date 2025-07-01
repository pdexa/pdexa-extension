#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/communication_pattern_base.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/memory_space_data.h>
#include <deal.II/base/mpi_stub.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/read_vector.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/lac/vector_type_traits.h>

#include <Kokkos_StdAlgorithms.hpp>

#include <iomanip>
#include <memory>

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra::distributed {

template<typename Number, typename MemorySpaceType = MemorySpace::Host>
class VectorView : public ::dealii::ReadVector<Number>, public Subscriptor {
public:
  using memory_space = MemorySpaceType;
  using kokkos_memory_space = typename memory_space::kokkos_space;
  using kokkos_execution_space = typename kokkos_memory_space::execution_space;
  using value_type = Number;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = value_type*;
  using const_iterator = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using size_type = types::global_dof_index;
  using real_type = typename numbers::NumberTraits<Number>::real_type;

  static_assert(std::is_same_v<MemorySpaceType, ::dealii::MemorySpace::Host> ||
                  std::is_same_v<MemorySpaceType, ::dealii::MemorySpace::Default>,
                "MemorySpace should be Host or Default");

  VectorView(kokkos_execution_space exec_space,
             value_type* data,
             std::shared_ptr<const Utilities::MPI::Partitioner> partitioner);

  VectorView(value_type* data, std::shared_ptr<const Utilities::MPI::Partitioner> partitioner);

  iterator begin();

  const_iterator begin() const;

  iterator end();

  const_iterator end() const;

  reference operator[](size_type global_index);

  const_reference operator[](size_type global_index) const;

  reference local_element(size_type local_index);

  const_reference local_element(size_type local_index) const;

  VectorView& operator=(Number s);

  size_type size() const override;

  void extract_subvector_to(const ArrayView<const types::global_dof_index>& indices,
                            ArrayView<Number>& elements) const override;

  bool has_ghost_elements() const;

  void update_ghost_values() const;

  void zero_out_ghost_values() const;

  void compress(VectorOperation::values operation);

  /**
   * Exception
   */
  DeclException3(ExcAccessToNonLocalElement,
                 size_type,
                 size_type,
                 size_type,
                 << "You tried to access element " << arg1 << " of a distributed vector, but this element is not "
                 << "stored on the current processor. Note: The range of "
                 << "locally owned elements is [" << arg2 << ',' << arg3 << "]."
                 << "\n\n"
                 << "A common source for this kind of problem is that you "
                 << "are passing a 'fully distributed' vector into a function "
                 << "that needs read access to vector elements that correspond "
                 << "to degrees of freedom on ghost cells (or at least to "
                 << "'locally active' degrees of freedom that are not also "
                 << "'locally owned'). You need to pass a vector that has these "
                 << "elements as ghost entries.");

private:
  // same as LinearAlgebra::distributed::Vector::operator(), except that access to ghost vectors is not allowed
  auto global_access(size_type global_index) -> reference;
  auto global_access(size_type global_index) const -> const_reference;

  kokkos_execution_space exec_space = {};

  Kokkos::View<value_type*, kokkos_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> data_ = {};

  /**
   * Shared pointer to store the parallel partitioning information. This
   * information can be shared between several vectors that have the same
   * partitioning.
   */
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
};

template<typename Number, typename MemorySpaceType>
VectorView<Number, MemorySpaceType>::VectorView(kokkos_execution_space exec_space,
                                                value_type* data,
                                                std::shared_ptr<const Utilities::MPI::Partitioner> partitioner) :
    exec_space(std::move(exec_space)), data_(data, partitioner->locally_owned_size()),
    partitioner(std::move(partitioner)) {}

template<typename Number, typename MemorySpaceType>
VectorView<Number, MemorySpaceType>::VectorView(value_type* data,
                                                std::shared_ptr<const Utilities::MPI::Partitioner> partitioner) :
    data_(data, partitioner->locally_owned_size()), partitioner(partitioner) {}

template<typename Number, typename MemorySpaceType>
typename VectorView<Number, MemorySpaceType>::iterator VectorView<Number, MemorySpaceType>::begin() {
  return data_.data();
}

template<typename Number, typename MemorySpaceType>
typename VectorView<Number, MemorySpaceType>::const_iterator VectorView<Number, MemorySpaceType>::begin() const {
  return data_.data();
}

template<typename Number, typename MemorySpaceType>
typename VectorView<Number, MemorySpaceType>::iterator VectorView<Number, MemorySpaceType>::end() {
  return data_.data() + data_.size();
}

template<typename Number, typename MemorySpaceType>
typename VectorView<Number, MemorySpaceType>::const_iterator VectorView<Number, MemorySpaceType>::end() const {
  return data_.data() + data_.size();
}

template<typename Number, typename MemorySpaceType>
typename VectorView<Number, MemorySpaceType>::reference
VectorView<Number, MemorySpaceType>::operator[](size_type global_index) {
  return global_access(global_index);
}

template<typename Number, typename MemorySpaceType>
typename VectorView<Number, MemorySpaceType>::const_reference
VectorView<Number, MemorySpaceType>::operator[](size_type global_index) const {
  return global_access(global_index);
}

template<typename Number, typename MemorySpaceType>
typename VectorView<Number, MemorySpaceType>::reference
VectorView<Number, MemorySpaceType>::local_element(size_type local_index) {
  return data_(local_index);
}

template<typename Number, typename MemorySpaceType>
typename VectorView<Number, MemorySpaceType>::const_reference
VectorView<Number, MemorySpaceType>::local_element(size_type local_index) const {
  return data_(local_index);
}

template<typename Number, typename MemorySpaceType>
VectorView<Number, MemorySpaceType>& VectorView<Number, MemorySpaceType>::operator=(Number s) {
  Kokkos::Experimental::fill_n("VectorView::operator=(Number)", exec_space, Kokkos::Experimental::begin(data_),
                               partitioner->locally_owned_size(), s);
  return *this;
}

template<typename Number, typename MemorySpaceType>
typename VectorView<Number, MemorySpaceType>::size_type VectorView<Number, MemorySpaceType>::size() const {
  // same as LinearAlgebra::distributed::Vector, this returns the global size
  return partitioner->size();
}

template<typename Number, typename MemorySpaceType>
void VectorView<Number, MemorySpaceType>::extract_subvector_to(
  const ArrayView<const types::global_dof_index, MemorySpace::Host>& indices,
  ArrayView<Number, MemorySpace::Host>& elements) const {
  for (size_type i = 0; i < indices.size(); ++i) { elements[i] = global_access(indices[i]); }
}

template<typename Number, typename MemorySpaceType>
bool VectorView<Number, MemorySpaceType>::has_ghost_elements() const {
  return false;
}

template<typename Number, typename MemorySpaceType>
void VectorView<Number, MemorySpaceType>::update_ghost_values() const {
  Assert(false, ExcMessage("This class doesn't support ghost values"));
}

template<typename Number, typename MemorySpaceType>
void VectorView<Number, MemorySpaceType>::zero_out_ghost_values() const {}

template<typename Number, typename MemorySpaceType>
void VectorView<Number, MemorySpaceType>::compress(VectorOperation::values operation) {}

template<typename Number, typename MemorySpaceType>
auto VectorView<Number, MemorySpaceType>::global_access(const size_type global_index) -> reference {
  Assert((std::is_same_v<MemorySpaceType, MemorySpace::Host>),
         ExcMessage("This function is only implemented for the Host memory space"));
  Assert(
    partitioner->in_local_range(global_index),
    ExcAccessToNonLocalElement(global_index, partitioner->local_range().first,
                               partitioner->local_range().second == 0 ? 0 : (partitioner->local_range().second - 1)));
  return data_(partitioner->global_to_local(global_index));
}

template<typename Number, typename MemorySpaceType>
auto VectorView<Number, MemorySpaceType>::global_access(size_type global_index) const -> const_reference {
  Assert((std::is_same_v<MemorySpaceType, MemorySpace::Host>),
         ExcMessage("This function is only implemented for the Host memory space"));
  Assert(
    partitioner->in_local_range(global_index),
    ExcAccessToNonLocalElement(global_index, partitioner->local_range().first,
                               partitioner->local_range().second == 0 ? 0 : (partitioner->local_range().second - 1)));
  return data_(partitioner->global_to_local(global_index));
}

} // namespace LinearAlgebra::distributed

/**
 * Declare dealii::LinearAlgebra::distributed::VectorView as distributed vector.
 */
template<typename Number, typename MemorySpace>
struct is_serial_vector<LinearAlgebra::distributed::VectorView<Number, MemorySpace>> : std::false_type {};

DEAL_II_NAMESPACE_CLOSE
