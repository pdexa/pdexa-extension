cmake_minimum_required(VERSION 3.18)

project(pdexa LANGUAGES CXX VERSION 0.1)

option(PDEXA_ENABLE_TESTS "Enable unit tests (from the `tests` subfolder)." ON)
option(PDEXA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" FALSE)

add_library(pdexa-ext "")
add_library(pdexa::ext ALIAS pdexa-ext)

target_compile_features(pdexa-ext INTERFACE cxx_std_17)

include(cmake/BuildSettings.cmake)
include(cmake/CompilerWarnings.cmake)
include(third-party/enable_packages.cmake)

set_project_warnings(pdexa-ext)

add_subdirectory(source)
add_subdirectory(app)

if (PDEXA_ENABLE_TESTS)
    add_subdirectory(tests)
endif ()