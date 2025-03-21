// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <ginkgo/core/solver/batch_solver_base.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>



namespace gko::batch_template::solver {

template<typename T>
class BatchSolver {
public:
  std::shared_ptr<const T> get_system_matrix() const { return this->system_matrix_; }

  double get_tolerance() const { return this->residual_tol_; }

  int get_max_iterations() const { return this->max_iterations_; }

  batch::stop::tolerance_type get_tolerance_type() const { return this->tol_type_; }

protected:
  BatchSolver() {}

  BatchSolver(std::shared_ptr<const T> system_matrix,
              const double res_tol,
              const int max_iterations,
              const batch::stop::tolerance_type tol_type) :
      system_matrix_{std::move(system_matrix)}, residual_tol_{res_tol}, max_iterations_{max_iterations},
      tol_type_{tol_type} {}

  void set_system_matrix_base(std::shared_ptr<const T> system_matrix) {
    this->system_matrix_ = std::move(system_matrix);
  }

  std::shared_ptr<const T> system_matrix_{};
  double residual_tol_{};
  int max_iterations_{};
  batch::stop::tolerance_type tol_type_{};
  mutable array<unsigned char> workspace_{};
};

} // namespace gko::batch_template::solver
