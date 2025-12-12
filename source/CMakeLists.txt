target_sources(
        ext
        PRIVATE
        dummy.cpp
)

file(
        GLOB_RECURSE header_files
        LIST_DIRECTORIES false
        ${PROJECT_SOURCE_DIR}/include/*
)

target_sources(
        ext
        INTERFACE
        FILE_SET
        headers
        TYPE HEADERS
        BASE_DIRS
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        FILES
        ${header_files}
)

target_link_libraries(ext Ginkgo::ginkgo Kokkos::kokkos)  # can't use PUBLIC because of deal.ii
deal_ii_setup_target(ext)