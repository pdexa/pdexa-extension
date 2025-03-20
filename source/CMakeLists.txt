add_library(pdexa-ext "")
add_library(pdexa::ext ALIAS pdexa-ext)

target_sources(
        pdexa-ext
        PRIVATE
        dummy.cpp
)

target_include_directories(
        pdexa-ext PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)

target_compile_definitions(
        pdexa-ext
        PUBLIC
        PDEXA_EXT_ENABLE_REFERENCE=1
)
foreach (exec OPENMP CUDA HIP SYCL)
    if (${Kokkos_ENABLE_${exec_uc}})
        target_compile_definitions(
                pdexa-ext
                PUBLIC
                PDEXA_EXT_ENABLE_{exec}=1
        )
    endif ()
endforeach ()

target_link_libraries(pdexa-ext pdexa-options Ginkgo::ginkgo Kokkos::kokkos)  # can't use PUBLIC because of deal.ii
deal_ii_setup_target(pdexa-ext)