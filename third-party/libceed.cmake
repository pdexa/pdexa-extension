message(STATUS "Enabling package libCEED")

find_library(libceed libceed HINTS ${libCEED_DIR})
message("Found libCEED at ${libCEED}") 

