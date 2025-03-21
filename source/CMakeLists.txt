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

target_link_libraries(pdexa-ext pdexa-options Ginkgo::ginkgo Kokkos::kokkos)  # can't use PUBLIC because of deal.ii
deal_ii_setup_target(pdexa-ext)