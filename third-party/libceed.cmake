message(STATUS "Enabling package libCEED")

include(FindPkgConfig)
pkg_search_module(
        libCEED
        QUIET
        IMPORTED_TARGET
        ceed
)

if (libCEED_FOUND)
    message(STATUS "Found libCEED at ${libCEED_LIBRARY_DIRS}")
else ()
    message(STATUS "Try finding libCEED manually")
    find_library(libceed libceed HINTS ${libCEED_DIR})
    if (libCEED)
        message(STATUS "Found libCEED at ${libCEED}")
    else ()
        message(FATAL_ERROR "Could not locate libCEED")
    endif ()
endif ()

