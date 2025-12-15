add_executable(advection_solver_implicit advection_solver_implicit.cc
        cellwise_operator.hpp
        cellwise_operator.cpp
        cellwise_preconditioner.cpp
        cellwise_preconditioner.hpp
        definitions.hpp
        apply_gko_solver.cpp)
target_link_libraries(advection_solver_implicit PUBLIC pdexa::ext cxxopts::cxxopts)
target_compile_features(advection_solver_implicit PUBLIC cxx_std_20)

add_executable(advection_solver_implicit_deal advection_solver_implicit_deal.cc)
target_link_libraries(advection_solver_implicit_deal PUBLIC pdexa::ext)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    target_compile_features(advection_solver_implicit PUBLIC cuda_std_20)

    set_source_files_properties(apply_gko_solver.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(apply_gko_solver.cpp PROPERTIES COMPILE_OPTIONS --expt-relaxed-constexpr -diag-suppress 554)
    # warning 554 is a false positive
endif ()

find_package(GTest)
if (GTest_FOUND)
    add_executable(cellwise_preconditioner.test cellwise_preconditioner.test.cpp)
    target_link_libraries(cellwise_preconditioner.test PRIVATE pdexa::ext GTest::gtest_main GTest::GTest)
    set_source_files_properties(cellwise_preconditioner.test.cpp
            PROPERTIES
            LANGUAGE CUDA
            COMPILE_OPTIONS --expt-relaxed-constexpr)
    target_compile_features(cellwise_preconditioner.test PUBLIC cxx_std_20 cuda_std_20)

    add_executable(cellwise_operator.test cellwise_operator.test.cpp)
    target_link_libraries(cellwise_operator.test PRIVATE pdexa::ext GTest::gtest_main GTest::GTest)
    set_source_files_properties(cellwise_operator.test.cpp
            PROPERTIES
            LANGUAGE CUDA
            COMPILE_OPTIONS --expt-relaxed-constexpr)
    target_compile_features(cellwise_operator.test PUBLIC cxx_std_20 cuda_std_20)
endif ()

include(FetchContent)
FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.2
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(benchmark)
add_executable(bench-operator benchmark_operator.cpp)
add_executable(bench-preconditioner benchmark_preconditioner.cpp)
add_executable(bench-bicgstab benchmark_bicgstab.cpp)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(bench-operator PRIVATE pdexa::ext cxxopts::cxxopts benchmark::benchmark CUDA::cublas)
target_link_libraries(bench-preconditioner PRIVATE pdexa::ext cxxopts::cxxopts benchmark::benchmark CUDA::cublas)
target_link_libraries(bench-bicgstab PRIVATE pdexa::ext cxxopts::cxxopts benchmark::benchmark CUDA::cublas)
if (CMAKE_CUDA_COMPILER)
    set_source_files_properties(
            benchmark_operator.cpp benchmark_preconditioner.cpp benchmark_bicgstab.cpp
            PROPERTIES
            LANGUAGE CUDA
            COMPILE_OPTIONS --expt-relaxed-constexpr -diag-suppress 554)
    # warning 554 is a false positive
    target_compile_features(bench-operator PUBLIC cxx_std_20 cuda_std_20)
    target_compile_features(bench-preconditioner PUBLIC cxx_std_20 cuda_std_20)
    target_compile_features(bench-bicgstab PUBLIC cxx_std_20 cuda_std_20)
endif ()
