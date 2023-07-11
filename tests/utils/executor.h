#ifndef PDEXA_TESTS_UTILS_EXECUTOR_H
#define PDEXA_TESTS_UTILS_EXECUTOR_H


#include <ginkgo/core/base/executor.hpp>


template<typename Exec> std::shared_ptr<gko::Executor> create_executor();

template<> std::shared_ptr<gko::Executor> create_executor<gko::ReferenceExecutor>() {
  return gko::ReferenceExecutor::create();
}

template<> std::shared_ptr<gko::Executor> create_executor<gko::OmpExecutor>() {
  return gko::OmpExecutor::create();
}

template<> std::shared_ptr<gko::Executor> create_executor<gko::CudaExecutor>() {
  if (gko::CudaExecutor::get_num_devices() == 0) {
    throw std::runtime_error{"No suitable CUDA devices"};
  }
  return gko::CudaExecutor::create(0, gko::ReferenceExecutor::create(), false,
                                   gko::default_cuda_alloc_mode);
}

template<> std::shared_ptr<gko::Executor> create_executor<gko::HipExecutor>() {
  if (gko::HipExecutor::get_num_devices() == 0) {
    throw std::runtime_error{"No suitable HIP devices"};
  }
  return gko::HipExecutor::create(0, gko::ReferenceExecutor::create(), false,
                                  gko::default_hip_alloc_mode);
}

template<> std::shared_ptr<gko::Executor> create_executor<gko::DpcppExecutor>() {
  if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
    return gko::DpcppExecutor::create(0, gko::ReferenceExecutor::create(), "gpu");
  } else if (gko::DpcppExecutor::get_num_devices("cpu") > 0) {
    return gko::DpcppExecutor::create(0, gko::ReferenceExecutor::create(), "cpu");
  } else {
    throw std::runtime_error{"No suitable DPC++ devices"};
  }
}

#endif //PDEXA_TESTS_UTILS_EXECUTOR_H
