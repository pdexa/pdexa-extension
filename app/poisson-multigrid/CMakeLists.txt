if (DEAL_II_WITH_MPI)
    add_executable(poisson-multigrid poisson-multigrid.cpp)
    target_link_libraries(poisson-multigrid PRIVATE pdexa::ext cxxopts::cxxopts)
endif ()
