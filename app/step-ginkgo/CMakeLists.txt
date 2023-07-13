add_executable(step-ginkgo step-ginkgo.cc)
target_link_libraries(step-ginkgo PRIVATE pdexa::ext)

add_executable(step-ginkgo-device-rhs step-ginkgo-device-rhs.cc)
target_link_libraries(step-ginkgo-device-rhs PRIVATE pdexa::ext)