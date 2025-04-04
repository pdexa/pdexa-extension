include(FetchContent)
fetchcontent_declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v3.2.0
)
fetchcontent_makeavailable(cxxopts)

add_subdirectory(batched-solver)
add_subdirectory(step-ginkgo)
add_subdirectory(poisson-multigrid)
