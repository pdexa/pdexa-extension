include(FetchContent)

# This has to be on top, because other packages seem to mess with cmake policies
include(${CMAKE_SOURCE_DIR}/third-party/ginkgo.cmake)

include(${CMAKE_SOURCE_DIR}/third-party/kokkos.cmake)

include(${CMAKE_SOURCE_DIR}/third-party/deal.ii.cmake)

include(${CMAKE_SOURCE_DIR}/third-party/libceed.cmake)

if (PDEXA_ENABLE_TESTS)
  include(${CMAKE_SOURCE_DIR}/third-party/gtest.cmake)
endif ()
