add_library(testing INTERFACE)
target_link_libraries(testing INTERFACE GTest::gtest_main)
target_include_directories(
        testing INTERFACE
        ${PROJECT_SOURCE_DIR}
)

function(pdexa_create_test base_name source)
    foreach (exec Reference Omp Cuda Hip Dpcpp)
        string(TOLOWER ${exec} exec_lc)
        string(TOUPPER ${exec} exec_uc)
        set(target ${base_name}-${exec_lc})
        if (${GINKGO_BUILD_${exec_uc}})
            add_executable(${target} ${source})
            target_compile_definitions(${target} PRIVATE -DEXEC=${exec}Executor -DCOMPILE_${exec_uc})
            target_link_libraries(${target} PRIVATE pdexa::ext testing)
        endif ()
    endforeach ()
endfunction()

pdexa_create_test(csr deal.II/lac/csr.cc)
pdexa_create_test(csr_formats deal.II/lac/csr_formats.cc)
pdexa_create_test(csr_manip deal.II/lac/csr_manip.cc)

pdexa_create_test(solver deal.II/lac/solver.cc)
