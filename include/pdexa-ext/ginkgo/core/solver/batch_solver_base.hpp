#pragma once


namespace gko::batch_template::solver {
template<typename ValueType>
class BatchSolver : public EnableAbstractPolymorphicObject<BatchSolver<ValueType>, batch::BatchLinOp> {
public:
  using value_type = ValueType;

  [[nodiscard]] double get_tolerance() const { return this->residual_tol_; }

  [[nodiscard]] int get_max_iterations() const { return this->max_iterations_; }

  [[nodiscard]] batch::stop::tolerance_type get_tolerance_type() const { return this->tol_type_; }

  void apply(ptr_param<const batch::MultiVector<value_type>> b,
             ptr_param<batch::MultiVector<value_type>> x) const {
    using real_type = remove_complex<value_type>;
    this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    auto view = workspace_.as_view();
    auto log_data_ = std::make_unique<batch::log::detail::log_data<real_type>>(exec, b->get_num_batch_items(), view);
    this->solver_apply(make_temporary_clone(exec, b).get(), make_temporary_clone(exec, x).get(), log_data_.get());
    this->template log<log::Logger::batch_solver_completed>(log_data_->iter_counts, log_data_->res_norms);
  }

protected:
  explicit BatchSolver(std::shared_ptr<const Executor> exec) : EnableAbstractPolymorphicObject<
                                                               BatchSolver, batch::BatchLinOp>(exec) {}

  BatchSolver(std::shared_ptr<const Executor> exec,
              batch_dim<2> size,
              const double res_tol,
              const int max_iterations,
              const batch::stop::tolerance_type tol_type) : EnableAbstractPolymorphicObject<
                                                            BatchSolver, batch::BatchLinOp>(exec, size)
                                                          , residual_tol_{res_tol}
                                                          , max_iterations_{max_iterations}
                                                          , tol_type_{tol_type}
                                                          , workspace_{
                                                            this->get_executor(),
                                                            this->get_num_batch_items() * 32
                                                          } {}

  virtual void solver_apply(const batch::MultiVector<value_type>* b,
                            batch::MultiVector<value_type>* x,
                            batch::log::detail::log_data<remove_complex<value_type>>* log_data) const = 0;

private:
  double residual_tol_{};
  int max_iterations_{};
  batch::stop::tolerance_type tol_type_{};
  mutable array<unsigned char> workspace_{};
};


template<typename ConcreteType, typename OpType>
class EnableBatchSolver : public EnablePolymorphicObject<ConcreteType, BatchSolver<typename OpType::value_type>> {
public:
  using op_type = OpType;
  using value_type = typename op_type::value_type;

  [[nodiscard]] std::shared_ptr<const op_type> get_system_matrix() const { return this->system_matrix_; }

protected:
  explicit EnableBatchSolver(std::shared_ptr<const Executor> exec) : EnablePolymorphicObject<
                                                                     ConcreteType, BatchSolver<value_type>>(exec) {}

  EnableBatchSolver(std::shared_ptr<const op_type> system_matrix,
                    const double res_tol,
                    const int max_iterations,
                    const batch::stop::tolerance_type tol_type) : EnablePolymorphicObject<
                                                                  ConcreteType, BatchSolver<value_type>>(
                                                                  system_matrix->get_executor(),
                                                                  system_matrix->get_size(),
                                                                  res_tol,
                                                                  max_iterations,
                                                                  tol_type)
                                                                , system_matrix_{std::move(system_matrix)} {}

private:
  std::shared_ptr<const op_type> system_matrix_{};
};
} // namespace gko::batch_template::solver
