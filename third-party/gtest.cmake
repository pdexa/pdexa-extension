message(STATUS "Enabling package GTest")

if(NOT GTest_FOUND)
  message(STATUS "Fetching external GTest")
  fetchcontent_declare(
          googletest
          GIT_REPOSITORY https://github.com/google/googletest.git
          GIT_TAG main
  )

  set(gtest_force_shared_crt ON CACHE INTERNAL "")
  set(INSTALL_GTEST OFF CACHE INTERNAL "")
  fetchcontent_makeavailable(googletest)

  set(GTest_FOUND ON CACHE INTERNAL "")
endif()
