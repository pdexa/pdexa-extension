message(STATUS "Enabling package Ginkgo")

find_package(Ginkgo 1.9.0 QUIET)

if(NOT Ginkgo_FOUND)
  message(STATUS "Fetching external Ginkgo")

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    cmake_policy(SET CMP0104 OLD)
  endif()

  FetchContent_Declare(
      Ginkgo
      GIT_REPOSITORY https://github.com/ginkgo-project/ginkgo.git
      GIT_TAG develop
      GIT_SHALLOW ON
  )

  set(GINKGO_BUILD_HWLOC OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(Ginkgo)
endif()
