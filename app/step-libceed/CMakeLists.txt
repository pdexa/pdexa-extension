add_executable(step-libceed step-libceed.cc)

target_link_libraries(step-libceed PRIVATE pdexa::ext)
if (TARGET PkgConfig::libCEED)
    target_link_libraries(step-libceed PRIVATE PkgConfig::libCEED)
else ()
    target_include_directories(step-libceed PUBLIC ${libCEED_DIR}/include)
    target_link_libraries(step-libceed PRIVATE ${libCEED_DIR}/lib/libceed.so)
endif ()
