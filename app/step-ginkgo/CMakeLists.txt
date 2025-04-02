add_executable(step-ginkgo-seq step-ginkgo-seq.cc)
target_link_libraries(step-ginkgo-seq PRIVATE pdexa::ext)

if (DEAL_II_WITH_MPI)
    add_executable(step-ginkgo-dist step-ginkgo-dist.cpp)
    target_link_libraries(step-ginkgo-dist PRIVATE pdexa::ext)
endif ()
