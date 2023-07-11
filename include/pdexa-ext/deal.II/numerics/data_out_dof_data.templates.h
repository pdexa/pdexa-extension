#ifndef PDEXA_INCLUDE_PDEXA_EXT_DEAL_II_NUMERICS_DATA_OUT_DOF_DATA_TEMPLATES_H
#define PDEXA_INCLUDE_PDEXA_EXT_DEAL_II_NUMERICS_DATA_OUT_DOF_DATA_TEMPLATES_H


#include <deal.II/numerics/data_out.h>
#include <pdexa-ext/deal.II/lac/ginkgo_vector.h>


namespace dealii::internal::DataOutImplementation {
namespace {

/**
 * Need to inject this overload for GinkgoWrapper::Vector.
 */
template<typename Number> void copy_locally_owned_data_from(const GinkgoWrappers::Vector<Number> &src,
                                                            LinearAlgebra::distributed::Vector<Number> &dst) {
  LinearAlgebra::ReadWriteVector<Number> temp;
  temp.reinit(src.locally_owned_elements());

  auto gko_obj = src.get_gko_object();
  GinkgoWrappers::Vector<Number> host_src(gko_obj->get_executor()->get_master());
  host_src = src;
  std::copy_n(host_src.begin(), host_src.size(), temp.begin());

  LinearAlgebra::ReadWriteVector<Number> temp2;
  temp2.reinit(temp, true);
  temp2 = temp;

  dst.import_elements(temp2, VectorOperation::insert);
}

}
}


#include <deal.II/numerics/data_out_dof_data.templates.h>


#endif //PDEXA_INCLUDE_PDEXA_EXT_DEAL_II_NUMERICS_DATA_OUT_DOF_DATA_TEMPLATES_H
