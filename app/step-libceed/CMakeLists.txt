
add_executable(step-libceed step-libceed.cc)
target_link_libraries(step-libceed PRIVATE pdexa::ext)

TARGET_INCLUDE_DIRECTORIES(step-libceed PUBLIC ${libCEED_DIR}/include)
TARGET_LINK_LIBRARIES(step-libceed PRIVATE ${libCEED_DIR}/lib/libceed.so)
