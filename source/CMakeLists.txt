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
    if (${Kokkos_ENABLE_${exec}})
        string(TOLOWER ${exec} exec_lc)
        target_compile_definitions(
                pdexa-ext
                PUBLIC
                PDEXA_EXT_ENABLE_${exec}=1
                GKO_DEVICE_NAMESPACE=${exec_lc}
        )
    endif ()
endforeach ()

target_link_libraries(pdexa-ext Ginkgo::ginkgo Kokkos::kokkos)  # can't use PUBLIC because of deal.ii
deal_ii_setup_target(pdexa-ext)