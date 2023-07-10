cmake_minimum_required(VERSION 3.18)

project(pdexa LANGUAGES CXX VERSION 0.1)

add_library(pdexa-options INTERFACE)
target_compile_features(pdexa-options INTERFACE cxx_std_17)
add_library(pdexa::options ALIAS pdexa-options)

include(cmake/BuildSettings.cmake)
include(cmake/CompilerWarnings.cmake)

set_project_warnings(pdexa-options)

