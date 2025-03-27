cmake_minimum_required(VERSION 3.23)

project(pdexa LANGUAGES CXX VERSION 0.1)

option(PDEXA_ENABLE_TESTS "Enable unit tests (from the `tests` subfolder)." ON)
option(PDEXA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" FALSE)

add_library(ext "")
add_library(pdexa::ext ALIAS ext)

target_compile_features(ext INTERFACE cxx_std_17)

include(cmake/BuildSettings.cmake)
include(cmake/CompilerWarnings.cmake)
include(third-party/enable_packages.cmake)

set_project_warnings(ext)

add_subdirectory(source)
add_subdirectory(app)

if (PDEXA_ENABLE_TESTS)
    add_subdirectory(tests)
endif ()

install(
        TARGETS ext
        EXPORT pdexa-targets
        FILE_SET headers
)
install(
        EXPORT pdexa-targets
        NAMESPACE pdexa::
        DESTINATION lib64/cmake/pdexa
)
export(
        EXPORT pdexa-targets
        NAMESPACE pdexa::
        FILE ${CMAKE_CURRENT_BINARY_DIR}/pdexa-targets.cmake
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/pdexa-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/pdexa-config.cmake
        INSTALL_DESTINATION lib64/cmake/pdexa
        NO_SET_AND_CHECK_MACRO
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
install(
        FILES
        ${CMAKE_CURRENT_BINARY_DIR}/pdexa-config.cmake
        DESTINATION lib64/cmake/pdexa
)

# install internal ginkgo version
if (Ginkgo_POPULATED)
    install(
            TARGETS ginkgo
            EXPORT GinkgoTargets
            INCLUDES DESTINATION include
    )
    install(
            EXPORT GinkgoTargets
            NAMESPACE Ginkgo::
    )
endif ()