message(STATUS "Enabling package deal.II")

find_package(deal.II REQUIRED)

deal_ii_initialize_cached_variables()

if (DEAL_II_WITH_GINKGO)
    message(FATAL_ERROR
            "The deal.II installation has Ginkgo enabled. Please disable it to prevent conflicts with the "
            "wrappers implemented in PDExa.")
endif ()
