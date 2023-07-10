include(FetchContent)

# This has to be on top, because other packages seem to mess with cmake policies
include(${CMAKE_SOURCE_DIR}/third-party/ginkgo.cmake)

include(${CMAKE_SOURCE_DIR}/third-party/deal.ii.cmake)

if(${PROJECT_NAME}_ENABLE_UNIT_TESTING)
  include(${CMAKE_SOURCE_DIR}/third-party/gtest.cmake)
endif()
