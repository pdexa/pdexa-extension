message(STATUS "Enabling package GTest")

find_package(GTest QUIET)

if(NOT GTest_FOUND)
  message(STATUS "Fetching external GTest")
  FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.12.1
  )

  set(gtest_force_shared_crt ON CACHE INTERNAL "")
  set(INSTALL_GTEST OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable()
endif()
