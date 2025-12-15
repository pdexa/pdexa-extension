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

target_compile_definitions(
        ext
        PUBLIC
        PDEXA_EXT_ENABLE_REFERENCE=1
)
foreach (exec OPENMP SYCL CUDA HIP)
    if (PDEXA_EXT_ENABLE_${exec} OR Kokkos_ENABLE_${exec})
        set(PDEXA_EXT_ENABLE_${exec} ON)
        target_compile_definitions(
                ext
                PUBLIC
                PDEXA_EXT_ENABLE_${exec}=1
        )
    endif ()
endforeach ()
if (PDEXA_EXT_ENABLE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(
            ext
            PUBLIC
            GKO_DEVICE_NAMESPACE=cuda
    )
elseif (PDEXA_EXT_ENABLE_HIP)
    enable_language(HIP)
    target_compile_definitions(
            ext
            PUBLIC
            GKO_DEVICE_NAMESPACE=hip
    )
elseif (PDEXA_EXT_ENABLE_SYCL)
    target_compile_definitions(
            ext
            PUBLIC
            GKO_DEVICE_NAMESPACE=dpcpp
    )
endif ()

if(PDEXA_EXT_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(ext CUDA::cudart)
endif ()

target_link_libraries(ext Ginkgo::ginkgo Kokkos::kokkos)  # can't use PUBLIC because of deal.ii
deal_ii_setup_target(ext)