#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
namespace LA
{
  using namespace dealii::LinearAlgebraPETSc;
} // namespace LA
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h> 
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/distributed/solution_transfer.h> // Added for mesh hirearchy
#include <deal.II/lac/petsc_block_vector.h> 
#include <petscmat.h> // Added in order to use MatDiagonalScale function
#include <fstream>
#include <iostream>
#include <string> 
#include <deal.II/base/data_out_base.h> // Needed for output


#include <deal.II/lac/sparse_matrix.h>
#include <ginkgo/core/solver/multigrid.hpp>


#include <ginkgo/ginkgo.hpp>
#include <pdexa-ext/deal.II/lac/ginkgo_interface.h>




namespace IRK
{
  using namespace dealii;

  template <int dim>
  class IRK_SP
  {
  public:
    IRK_SP();
    void run();
  private:
    void setup_system();
    void IRK_run();
    void readin_IRK();
    void Assemble();
    void compare_against_analytical();

    double u_analytical( double x, double y, double t,  double a_x, double a_y, double a_t);
    double g_testcase( double x, double y, double t,  double a_x, double a_y, double a_t);
    void assemble_u0( LA::MPI::Vector &u0 ,double t0);
    void assemble_rhs( LA::MPI::BlockVector &rhs,LA::MPI::BlockVector &rhs_scaledss ,double t);
    void output_results() const;

    unsigned int q = 2; // number of stages, q = 2, ..., 10
    unsigned int N = 1; // Number of timesteps
    double final_time = 0.5;
    double tau = final_time/N; // Timestep size
    unsigned int current_timpestep = 0;
    unsigned int number_of_refinements = 3; // How many times to refine unit square

    double t0 = 0; // initial t0
    double t1 = 0; // used to stoe next t-step

    // constants used in test problems
    double a_x = 2;
    double a_y = 2;
    double a_t = 0.5;
    
    // Krylov tolerances 
    double tol_inner = 1e-10; 
    double tol_outer = 1e-10;
    
    MPI_Comm mpi_communicator;
    //------------------------------------------------------------------------------
    // Matricies and vectors containing the IRK constants and related decompositions
    FullMatrix<double> A = FullMatrix<double>(q,q);
    FullMatrix<double> A_inv = FullMatrix<double>(q,q);
    FullMatrix<double> L = FullMatrix<double>(q,q);
    FullMatrix<double> T_mat = FullMatrix<double>(q,q);
    FullMatrix<double> T_mat_inv = FullMatrix<double>(q,q);
    std::vector<double> D_vec = std::vector<double>(q); 
    std::vector<double> b_vec = std::vector<double>(q); 
    std::vector<double> c_vec = std::vector<double>(q); 
    unsigned int n_read_in;
    unsigned int m_read_in;
    //------------------------------------------------------------------------------

    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    LA::MPI::Vector       system_rhs;

    // Adding our own sturctures
    LA::MPI::SparseMatrix K; // To store stiffness block
    LA::MPI::SparseMatrix M; // To store full mass matrix
    LA::MPI::SparseMatrix AMGblock; // To store AMG blocks
    LA::MPI::SparseMatrix Kc; // To store compensator matrix accounting for bundary values in rhs

    SparseMatrix<double> K_d2; // To store stiffness block
    SparseMatrix<double> M_d2; // To store full mass matrix
    SparseMatrix<double> AMGblock_d2; // To store AMG blocks
    SparseMatrix<double> Kc_d2; // To store compensator matrix accounting for bundary values in rhs

    SparsityPattern      sparsity_pattern;


    // Ginkgo equiv
    using Mtx = gko::matrix::Csr<double>;
    std::shared_ptr<Mtx> gko_K;
    std::shared_ptr<Mtx> gko_M;
    std::shared_ptr<Mtx> gko_AMGblock;
    std::shared_ptr<Mtx> gko_Kc;


    LA::MPI::Vector boundaryones;
    LA::MPI::Vector boundaryzeros;
    LA::MPI::Vector u0_vec;
    LA::MPI::Vector u1_vec;

    LA::MPI::Vector locally_relevant_u0;
    LA::MPI::Vector locally_relevant_u_tru;

    // Core block vectors for the IRK problem
    LA::MPI::BlockVector rhs_vec_block; // Right hand side block system
    LA::MPI::BlockVector rhs_scaled_vec_block; // Right hand side block system
    LA::MPI::BlockVector k_vec_block; // RK steps
    LA::MPI::BlockVector temp_vec_block; // RK steps

    LA::MPI::Vector temp_vec;
    LA::MPI::Vector temp_vec2;

    // To get indecies of boundary points
    std::map<types::global_dof_index, Point<dim>> current_point_map; 
    std::vector<IndexSet> locally_owned_dofs_per_proc; 

    SolverControl cn_GMRES;
    SolverControl cn;

    ConditionalOStream pcout;
  };

   // Manually pull boundary indicies
   template <int dim, int spacedim>
   void
   test_map_dof_to_boundary_indices(const DoFHandler<dim, spacedim> &     dof_handler,
                               std::set<types::global_dof_index> &mapping)
   {
     mapping.clear();
     std::vector<types::global_dof_index> dofs_on_face;
     dofs_on_face.reserve(dof_handler.get_fe_collection().max_dofs_per_face());
     typename DoFHandler<dim, spacedim>::active_cell_iterator
       cell = dof_handler.begin_active(),
       endc = dof_handler.end();
     for (; cell != endc; ++cell)
	if( cell->is_locally_owned() ){
	       for (const unsigned int f : cell->face_indices())
		 if (cell->at_boundary(f))
		   {
		     const unsigned int dofs_per_face =
		       cell->get_fe().n_dofs_per_face(f);
		     dofs_on_face.resize(dofs_per_face);
		     cell->face(f)->get_dof_indices(dofs_on_face,
		                                    cell->active_fe_index());
			for( const auto dof : dofs_on_face)
				mapping.insert(dof);
		   } 
       }
   }

  template <int dim>
  IRK_SP<dim>::IRK_SP()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , fe(1)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  {}

  template <int dim>
  void IRK_SP<dim>::readin_IRK(){
    // Reads in IRK strctures and facotrization
    std::string q_string = std::to_string(q);
    std::ifstream fin1("IRK_txt/A"+q_string+".txt");
    std::ifstream fin2("IRK_txt/A_inv"+q_string+".txt");
    std::ifstream fin3("IRK_txt/L"+q_string+".txt");
    std::ifstream fin4("IRK_txt/T"+q_string+".txt");
    std::ifstream fin5("IRK_txt/T_inv"+q_string+".txt");
    std::ifstream fin6("IRK_txt/b_vec_"+q_string+".txt");
    std::ifstream fin7("IRK_txt/c_vec_"+q_string+".txt");
    std::ifstream fin8("IRK_txt/D_vec_"+q_string+".txt");
    // We first read in matrix dimensions in case we ever need a check
    // for now we just assume the files are correct
    fin1 >> m_read_in >> n_read_in;
    fin2 >> m_read_in >> n_read_in;
    fin3 >> m_read_in >> n_read_in;
    fin4 >> m_read_in >> n_read_in;
    fin5 >> m_read_in >> n_read_in;
    // Last three are vectors
    fin6 >> m_read_in >> n_read_in;
    fin7 >> m_read_in >> n_read_in;
    fin8 >> m_read_in >> n_read_in;
    for (unsigned int i = 0; i < q; i++){
      fin6 >> b_vec[i];
      fin7 >> c_vec[i];
      fin8 >> D_vec[i];
      for (unsigned  j = 0; j < q; j++){
          fin1 >> A[i][j];
          fin2 >> A_inv[i][j];
          fin3 >> L[i][j];
          fin4 >> T_mat[i][j];
          fin5 >> T_mat_inv[i][j];
      }
    }
  }

    template <int dim>
    double  IRK_SP<dim>::u_analytical( double x, double y, double t, double a_x, double a_y, double a_t){
    // Analytical solution
    return std::sin(a_x*numbers::PI*x)*std::sin(a_y*numbers::PI*y)*( 1 + std::sin(numbers::PI*t) )*std::exp(-a_t * t) ;
    }

    template <int dim>
    double  IRK_SP<dim>::g_testcase( double x, double y, double t, double a_x, double a_y, double a_t){
    // Right-hand-side
    return  std::sin(a_x*numbers::PI*x)*std::sin(a_y*numbers::PI*y)*
            (  numbers::PI*std::cos(numbers::PI*t) - a_t*(std::sin(numbers::PI*t)+1) 
            + (a_x*a_x + a_y*a_y)*numbers::PI*numbers::PI*( std::sin(numbers::PI*t)+1 )  )*std::exp(-a_t*t); 
    }

    template <int dim>
    void IRK_SP<dim>::assemble_u0( LA::MPI::Vector &u0 ,double t0){
      for( unsigned int i:locally_owned_dofs){ 
        u0[i] = u_analytical( current_point_map[i][0] , current_point_map[i][1] ,t0, a_x, a_y, a_t);
      }
      u0.compress(VectorOperation::insert);
    }

    template <int dim>
    void IRK_SP<dim>::assemble_rhs( LA::MPI::BlockVector &rhs, LA::MPI::BlockVector &rhs_scaled ,double t){
      for( unsigned int j=0; j<q; j++){
        for( unsigned int i:locally_owned_dofs){ 
          temp_vec2[i] = g_testcase(current_point_map[i][0] , current_point_map[i][1], t+c_vec[j]*tau, a_x, a_y, a_t);
        }
        temp_vec2.compress(VectorOperation::insert);
        M.vmult(temp_vec,temp_vec2); // temp_vec = M*g_vec_j
        rhs.block(j) = temp_vec; // rhs.block(j) = M*g_vec_j
        K.vmult(temp_vec,u0_vec); // temp_vec = K*u0_vec ; note we use Ku0 repededly, maybe move this vmult out of loop and into u0 construction
        rhs.block(j).add(-1,temp_vec); // rhs.block(j) = M*g_vec_j -  K*u0_vec
      }
      // Apply scaling so rhs corresponds to the transformed system
      // rhs_scaled = apply_transformation( rhs, A_inv )
      rhs_scaled = 0;
      for( unsigned int i=0; i<q; i++){
        for( unsigned int j=0; j<q; j++){
          rhs_scaled.block(i).add(A_inv(i,j), rhs.block(j) ); 
        }
      }
    }

    template <int dim>
    void IRK_SP<dim>::compare_against_analytical(){
	// Compares solution in u0 for time t0 against analytical sol
	for( unsigned int i:locally_owned_dofs){ 
		temp_vec[i] = u_analytical( current_point_map[i][0] , current_point_map[i][1] ,t0, a_x, a_y, a_t);
	}
	temp_vec.compress(VectorOperation::insert);

	// Plot output
	locally_relevant_u0=u0_vec;
        locally_relevant_u_tru = temp_vec;
        output_results();

	temp_vec.add(-1,u0_vec);
	pcout<<"||u_tru-u_calc||_2 = "<< temp_vec.l2_norm() <<"\n";
	pcout<<"Maximal pointwise error =  "<< temp_vec.linfty_norm() <<"\n";
        }



  class IRKMatrix_Ginkgo
  : public gko::EnableLinOp<IRKMatrix_Ginkgo>
  {
     public:
      IRKMatrix_Ginkgo(std::shared_ptr<const gko::Executor> exec)
      : gko::EnableLinOp<IRKMatrix_Ginkgo>(exec) {}

      IRKMatrix_Ginkgo(const std::shared_ptr<gko::matrix::Csr<double>> &K, const std::shared_ptr<gko::matrix::Csr<double>> &M, const FullMatrix<double> &A_inv, const double tau )
        : 
        gko::EnableLinOp<IRKMatrix_Ginkgo>(K->get_executor(), gko::dim<2>(K->get_size()[0] * A_inv.n(), K->get_size()[1] * A_inv.n())),
        K(K)
        , M(M)
        , A_inv(A_inv)
        , tau(tau)
      {}

      using gko::EnableLinOp<IRKMatrix_Ginkgo>::apply;
      using Vec = gko::matrix::Dense<double>;

      void apply(std::vector<std::shared_ptr<gko::matrix::Dense<double>>> &dst, const std::vector<std::shared_ptr<gko::matrix::Dense<double>>> &src) const
      {
        //std::cout<<"vmult start"<<"\n";
        auto exec = K->get_executor();
        auto zero = gko::initialize<Vec>({0.0}, exec);
        for( unsigned int i = 0; i < A_inv.n(); ++i  ){ 
          dst[i]->fill(0.0); // set all output values to zero
          for( unsigned int j = 0; j < A_inv.m(); ++j  ){
            M->apply(gko::initialize<Vec>({A_inv(i,j)}, exec), src[j], zero, dst[i]); //M.vmult(  temp  ,  src.block(j) );
            //dst.block(i).add(A_inv(i,j),temp);
            //std::cout<<"i,j = "<<i<<","<<j<<"\n";

            if(i==j){
              K->apply(gko::initialize<Vec>({tau}, exec), src[j], zero, dst[i]); //M.vmult(  temp  ,  src.block(j) );
              //K.vmult(temp,  src.block(j) );
              //dst.block(i).add(tau,temp);
            }
          }
        }
      }

      void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override{
        auto b_dense = gko::as<const Vec>(b);
        auto x_dense = gko::as<Vec>(x);

        auto n = K->get_size()[0];
        auto exec = K->get_executor();

        auto block = [n, exec](auto* vec, int i){
          auto data = vec->get_values();
          return Vec::create(exec, gko::dim<2>{n, 1}, gko::make_array_view(exec, n, data + n * i), 1);
        };

        auto const_block = [n, exec](const auto* vec, int i){
          auto data = vec->get_const_values();
          return Vec::create_const(exec, gko::dim<2>{n, 1}, gko::make_const_array_view(exec, n, data + n * i), 1);
        };

        //std::cout<<"vmult start"<<"\n";
        //auto exec = K->get_executor();
        auto zero = gko::initialize<Vec>({0.0}, exec);
        auto one = gko::initialize<Vec>({1.0}, exec);

        for( unsigned int i = 0; i < A_inv.n(); ++i  ){ 
          block(x_dense, i)->fill(0.0); // set all output values to zero
          for( unsigned int j = 0; j < A_inv.m(); ++j  ){
            M->apply(gko::initialize<Vec>({A_inv(i,j)}, exec), const_block(b_dense, j), one, block(x_dense, i)); 
            //M.vmult(  temp  ,  src.block(j) );
            //dst.block(i).add(A_inv(i,j),temp);
            //std::cout<<"A_invi,j = "<<A_inv(i,j)<<"\n";

            //std::cout<<"i,j = "<<i<<","<<j<<"\n";

            if(i==j){
              K->apply(gko::initialize<Vec>({tau}, exec), const_block(b_dense, j), one, block(x_dense, i)); //M.vmult(  temp  ,  src.block(j) );
              //K.vmult(temp,  src.block(j) );
              //dst.block(i).add(tau,temp);
            }
          }
        }
      }
      void apply_impl(const gko::LinOp*,const gko::LinOp*,const gko::LinOp*, gko::LinOp*) const override{}

    private:
      std::shared_ptr<gko::matrix::Csr<double>> K;
      std::shared_ptr<gko::matrix::Csr<double>> M;
      FullMatrix<double> A_inv;
      double tau;

  };

    class Preconditioner_ginkgo
     : public gko::EnableLinOp<Preconditioner_ginkgo>
    {
    public:
        Preconditioner_ginkgo(std::shared_ptr<const gko::Executor> exec)
      : gko::EnableLinOp<Preconditioner_ginkgo>(exec) {}

      Preconditioner_ginkgo(std::shared_ptr<gko::matrix::Csr<double>> &K, std::shared_ptr<gko::matrix::Csr<double>> &M, 
                     const FullMatrix<double> &T_mat, const FullMatrix<double> &T_mat_inv, const std::vector<double> &D_vec, const double tau, 
		     std::shared_ptr<gko::matrix::Csr<double>> &AMGblock, std::shared_ptr<gko::LinOp> CGAMGsolver)
        : gko::EnableLinOp<Preconditioner_ginkgo>(K->get_executor(), gko::dim<2>(K->get_size()[0] * T_mat.n(), K->get_size()[1] * T_mat.n())),
         K(K)
        , M(M)
        , T_mat(T_mat)
        , T_mat_inv(T_mat_inv)
        , D_vec(D_vec)
        , tau(tau)
        , AMGblock(AMGblock)
        , CGAMGsolver(CGAMGsolver)
      {}

      using gko::EnableLinOp<Preconditioner_ginkgo>::apply;
      using Vec = gko::matrix::Dense<double>;

      void apply(std::vector<std::shared_ptr<gko::matrix::Dense<double>>> &dst, const std::vector<std::shared_ptr<gko::matrix::Dense<double>>> &src) const
      {
        //std::cout<<"vmult start"<<"\n";
        auto exec = K->get_executor();
        auto zero = gko::initialize<Vec>({0.0}, exec);
      }
      
      void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override{
        auto b_dense = gko::as<const Vec>(b);
        auto x_dense = gko::as<Vec>(x);
        auto temp_vec = gko::as<Vec>(gko::clone(x_dense));

        auto n = K->get_size()[0];
        auto exec = K->get_executor();

        auto block = [n, exec](auto* vec, int i){
          auto data = vec->get_values();
          return Vec::create(exec, gko::dim<2>{n, 1}, gko::make_array_view(exec, n, data + n * i), 1);
        };

        auto const_block = [n, exec](const auto* vec, int i){
          auto data = vec->get_const_values();
          return Vec::create_const(exec, gko::dim<2>{n, 1}, gko::make_const_array_view(exec, n, data + n * i), 1);
        };

        //std::cout<<"Precon vmult start"<<"\n";
        auto zero = gko::initialize<Vec>({0.0}, exec);
        auto one = gko::initialize<Vec>({1.0}, exec);

        for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 
            block(x_dense, i)->fill(0.0); // set all output values to zero
            for( unsigned int j = 0; j < T_mat.n(); ++j  ){ 
              if( abs(T_mat_inv(i,j)) > 1e-12){ // T,T_inv are currently triangular so this check saves some work
                block(x_dense, i)->add_scaled( gko::initialize<Vec>({T_mat_inv(i,j)}, exec), const_block(b_dense, j)); // Ginkgo vers
                //dst.block(i).add(T_mat_inv(i,j),src.block(j));
              }
            }
        }

        // function that returns approximation of P^-1 b
        // Apply transformation to right hand side

				
      // For simplicity we construct the blocks explicitly (but this can be replaced by its own class with its own matvec) 
      // ----------------------
      for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 

        //block(x_dense, i)->fill(1.0); // set all rhs values to 1
        //write(std::cout, block(x_dense, i));

        CGAMGsolver->apply(block(x_dense, i), block(temp_vec.get(),i) );
        auto logger = gko::as<gko::log::Convergence<double>>(CGAMGsolver->get_loggers().front());
        auto res = gko::as<gko::matrix::Dense<double>>(logger->get_residual_norm());
        //std::cout << i << " Converged after " << logger->get_num_iterations() << "\n";
         // std::cout << "Final residual norm sqrt(r^T r): \n";
         // write(std::cout, res);
      //    CGsolver.solve(AMGblock,temp_vec_block.block(i),dst.block(i),AMG_list[i]);
      }	

      // Apply transformation to sol
      //dst = 0;
      for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 
          for( unsigned int j = 0; j < T_mat.n(); ++j  ){ 
            if( abs(T_mat(i,j)) > 1e-12){ // T,T_inv are currently triangular so this check saves some work
             // dst.block(i).add(T_mat(i,j),temp_vec_block.block(j));
             block(x_dense, i)->add_scaled( gko::initialize<Vec>({T_mat(i,j)}, exec), block(temp_vec.get(), j)); // Ginkgo vers

            }
          }
      }

      }

      void apply_impl(const gko::LinOp*,const gko::LinOp*,const gko::LinOp*, gko::LinOp*) const override{}

      

      void
      vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const
      {
      // function that returns approximation of P^-1 b
      // Apply transformation to right hand side
      /*dst = 0;
      for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 
          for( unsigned int j = 0; j < T_mat.n(); ++j  ){ 
            if( abs(T_mat_inv(i,j)) > 1e-12){ // T,T_inv are currently triangular so this check saves some work
              dst.block(i).add(T_mat_inv(i,j),src.block(j));
            }
          }
	    }
				
      // For simplicity we construct the blocks explicitly (but this can be replaced by its own class with its own matvec) 
      // ----------------------
      for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 
        AMGblock.copy_from(K);
        AMGblock *= tau;
        AMGblock.add(D_vec[i],M);
          CGsolver.solve(AMGblock,temp_vec_block.block(i),dst.block(i),AMG_list[i]);
      }	

      // Apply transformation to sol
      dst = 0;
      for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 
          for( unsigned int j = 0; j < T_mat.n(); ++j  ){ 
            if( abs(T_mat(i,j)) > 1e-12){ // T,T_inv are currently triangular so this check saves some work
              dst.block(i).add(T_mat(i,j),temp_vec_block.block(j));
            }
          }*/
    	}


      
      private:
        std::shared_ptr<gko::matrix::Csr<double>> K;
        std::shared_ptr<gko::matrix::Csr<double>> M;
        FullMatrix<double> T_mat;
        FullMatrix<double> T_mat_inv;
        std::vector<double> D_vec;
        double tau;
        std::shared_ptr<gko::matrix::Csr<double>> AMGblock;
        std::shared_ptr<gko::LinOp> CGAMGsolver;
    };




// ------------------------------------------------------------------------

  class IRKMatrix
  {
    public:
      IRKMatrix(const LA::MPI::SparseMatrix &K,const LA::MPI::SparseMatrix &M, const FullMatrix<double> &A_inv, LA::MPI::Vector& temp , const double tau )
        : K(K)
        , M(M)
        , A_inv(A_inv)
        , temp(temp)
        , tau(tau)
      {}

      void
      vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const
      {
        dst = 0; // set all output values to zero
        for( unsigned int i = 0; i < A_inv.n(); ++i  ){ 
          for( unsigned int j = 0; j < A_inv.m(); ++j  ){
            M.vmult(  temp  ,  src.block(j) );
            dst.block(i).add(A_inv(i,j),temp);
            if(i==j){
              K.vmult(temp,  src.block(j) );
              dst.block(i).add(tau,temp);
            }
          }
        }
      }
    private:
      const LA::MPI::SparseMatrix &K;
      const LA::MPI::SparseMatrix &M;
      const FullMatrix<double> &A_inv;
      LA::MPI::Vector& temp;
      const double tau;
    };

    class Preconditioner
    {

    public:
      Preconditioner(const LA::MPI::SparseMatrix &K,const LA::MPI::SparseMatrix &M, 
                     const FullMatrix<double> &T_mat, const FullMatrix<double> &T_mat_inv, const std::vector<double> &D_vec, const double tau, 
		     LA::MPI::SparseMatrix &AMGblock, LA::MPI::BlockVector &temp_vec_block, SolverControl &cn_CG, SolverCG<LA::MPI::Vector> &CGsolver )
        : K(K)
        , M(M)
        , T_mat(T_mat)
        , T_mat_inv(T_mat_inv)
        , D_vec(D_vec)
        , tau(tau)
        , AMGblock(AMGblock)
        , AMG_list(std::vector<LA::MPI::PreconditionAMG>(T_mat.n()))
        , AMGdata_list(std::vector<LA::MPI::PreconditionAMG::AdditionalData>(T_mat.n()))
        , temp_vec_block(temp_vec_block)
        , cn_CG(cn_CG)
        , CGsolver(CGsolver)
      {}


      void
      AMG_setup(){
        // Set up the q AMG blocks, call this function before using precon
        // as we rely on an AMG we need to construct each block explicitly 
        for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 
          AMGblock.copy_from(K);
          AMGblock *= tau;
          AMGblock.add(D_vec[i],M);
          AMG_list[i].initialize(AMGblock,AMGdata_list[i]); 
        }
      }


      void
      vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const
      {
      // function that returns approximation of P^-1 b
      // Apply transformation to right hand side
      dst = 0;
      for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 
          for( unsigned int j = 0; j < T_mat.n(); ++j  ){ 
            if( abs(T_mat_inv(i,j)) > 1e-12){ // T,T_inv are currently triangular so this check saves some work
              dst.block(i).add(T_mat_inv(i,j),src.block(j));
            }
          }
	    }
				
      // Solve block diagonal system:
      // For simplicity we construct the blocks explicitly (but this can be replaced by its own class with its own matvec) 
      // ----------------------
      for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 
        AMGblock.copy_from(K);
        AMGblock *= tau;
        AMGblock.add(D_vec[i],M);
          CGsolver.solve(AMGblock,temp_vec_block.block(i),dst.block(i),AMG_list[i]);
      }	

      // Apply transformation to sol
      dst = 0;
      for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 
          for( unsigned int j = 0; j < T_mat.n(); ++j  ){ 
            if( abs(T_mat(i,j)) > 1e-12){ // T,T_inv are currently triangular so this check saves some work
              dst.block(i).add(T_mat(i,j),temp_vec_block.block(j));
            }
          }
    	}


      }
      private:
        const LA::MPI::SparseMatrix &K;
        const LA::MPI::SparseMatrix &M;
        const FullMatrix<double> &T_mat;
        const FullMatrix<double> &T_mat_inv;
        const std::vector<double> &D_vec;
        const double tau;
        LA::MPI::SparseMatrix &AMGblock;
        std::vector<LA::MPI::PreconditionAMG> AMG_list;
        std::vector<LA::MPI::PreconditionAMG::AdditionalData> AMGdata_list;
        LA::MPI::BlockVector &temp_vec_block;
        SolverControl &cn_CG;
        SolverCG<LA::MPI::Vector> &CGsolver;
    };


  template <int dim>
  void IRK_SP<dim>::setup_system()
  {
    pcout<<"Start setup..."<<"\n";
    dof_handler.distribute_dofs(fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    constraints.clear();
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    constraints.clear(); // <- as we manually set BC
    DoFTools::map_dofs_to_support_points<dim>(MappingQGeneric<dim>(fe.degree + 1), dof_handler, current_point_map); 
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(),  mpi_communicator, locally_relevant_dofs);
    
    sparsity_pattern.copy_from(dsp);


    // Inititalizing PDE-OPT matricies
    K.reinit(locally_owned_dofs,locally_owned_dofs, dsp, mpi_communicator);
    Kc.reinit(K);
    M.reinit(K);
    AMGblock.reinit(K);

    K_d2.reinit(sparsity_pattern);
    Kc_d2.reinit(sparsity_pattern);
    M_d2.reinit(sparsity_pattern);
    AMGblock_d2.reinit(sparsity_pattern);


    // Inititalizing PDE-OPT vectors
    boundaryones.reinit(system_rhs);
    boundaryzeros.reinit(system_rhs);
    // Initalizing block vectors 
    rhs_vec_block.reinit(q, mpi_communicator, dof_handler.n_dofs(),  dof_handler.n_locally_owned_dofs() );
    k_vec_block.reinit(rhs_vec_block);
    temp_vec_block.reinit(rhs_vec_block);
    rhs_scaled_vec_block.reinit(rhs_vec_block);
    //--------------------------------------------------------------------
    u0_vec.reinit(system_rhs); 
    u1_vec.reinit(system_rhs); 
    temp_vec.reinit(system_rhs); 
    temp_vec2.reinit(system_rhs);     
    locally_relevant_u0.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator); // created for output
    locally_relevant_u_tru.reinit(locally_relevant_u0);
  }

  template <int dim>
  void IRK_SP<dim>::Assemble()
  {

    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_K(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_M(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_K = 0.;
          cell_M = 0.;
          cell_rhs    = 0.;
          fe_values.reinit(cell);
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j){
                    cell_K(i, j) += fe_values.shape_grad(i, q_point) *fe_values.shape_grad(j, q_point) *fe_values.JxW(q_point);
                    cell_M(i, j) += fe_values.shape_value(i, q_point) *fe_values.shape_value(j, q_point) *fe_values.JxW(q_point);
		  }
                }
            }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_K,cell_rhs,local_dof_indices,K,system_rhs); // Create K 
          constraints.distribute_local_to_global(cell_M,cell_rhs,local_dof_indices,M,system_rhs); // Create M

      for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices()){
            K_d2.add(local_dof_indices[i],local_dof_indices[j], cell_K(i, j));
            M_d2.add(local_dof_indices[i],local_dof_indices[j], cell_M(i, j));
            AMGblock_d2.add(local_dof_indices[i],local_dof_indices[j], tau*cell_K(i, j)+D_vec[0]*cell_M(i, j));
            }
        }
          
      }

    

    Kc_d2 = 0; // Currently w.o. Boundary conditions, e.g., we need no compensator matrix.

    auto exec = gko::ReferenceExecutor::create(); //Ref CPU exec
    gko_K = dealii::GinkgoInterface::create_csr_matrix(exec, K_d2);
    gko_M = dealii::GinkgoInterface::create_csr_matrix(exec, M_d2);
    gko_Kc = dealii::GinkgoInterface::create_csr_matrix(exec, Kc_d2);
    gko_AMGblock = dealii::GinkgoInterface::create_csr_matrix(exec, AMGblock_d2);

    //gko::write(std::cout, gko_K); // output matrix

    K.compress(VectorOperation::add);
    M.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    std::set< types::global_dof_index > boundary_points ;
    test_map_dof_to_boundary_indices( dof_handler ,  boundary_points ) ;

    // create diagonal matrices from boundary zeros
    auto host_boundary_points = gko::array<double>(exec->get_master(), dof_handler.n_dofs());
    host_boundary_points.fill(0.0);

    for( unsigned int k:locally_owned_dofs)
    {
   	if( locally_owned_dofs.is_element(k) ){
		if( boundary_points.find(k) != boundary_points.end() ){
			boundaryones(k) = 1;
      host_boundary_points.get_data()[k] = 1.0;
		}
		else{
			boundaryones(k) = 0;
		}
    	}
    }
    auto gko_boundary_points = gko::array<double>(exec, std::move(host_boundary_points));
    boundaryones.compress(VectorOperation::insert); 

    system_rhs=1;
    boundaryzeros = 1; // Set all elements in boundary zeros =1
    boundaryzeros.add(-1,boundaryones);
    Kc.copy_from(K);

    // Apply boundary conditions symmerically: - Note current scaling example with boundary ones,
    // for K this should be boundary zeros.  
    auto diag = gko::matrix::Diagonal<double>::create(exec, dof_handler.n_dofs(), gko_boundary_points);
    auto gko_row_scale_K = gko::clone(gko_K);
    auto gko_scaled_K = gko::clone(gko_K);
    diag->apply(gko_K, gko_row_scale_K);
    diag->rapply(gko_row_scale_K, gko_scaled_K);
    auto id = gko::matrix::Identity<double>::create(exec, dof_handler.n_dofs());
    //-----------------------------
    // Multigrid example



    using mg = gko::solver::Multigrid;
    using pgm = gko::multigrid::Pgm<double>;
    std::shared_ptr<gko::LinOpFactory> multigrid_gen = mg::build().with_mg_level(pgm::build().with_deterministic(true).on(exec))
          .with_criteria(gko::stop::Iteration::build().with_max_iters(1u).on(exec)).on(exec);

    double tolerance = tol_inner; 
    using cg = gko::solver::Cg<double>;
    auto solver_gen =  cg::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                gko::stop::ResidualNorm<double>::build()
                .with_baseline(gko::stop::mode::absolute)
                .with_reduction_factor(tolerance)
                .on(exec))
                .with_preconditioner(multigrid_gen)
                .on(exec);

   auto solver = solver_gen->generate(gko_M);

  std::shared_ptr<const gko::log::Convergence<double>> logger = gko::log::Convergence<double>::create();
  solver->add_logger(logger);

  auto b = gko::matrix::Dense<double>::create(exec, gko::dim<2>(dof_handler.n_dofs(), 1));
  b->fill(1.0);
  auto x = gko::clone(b);
  x->fill(0.0);

  solver->apply(b, x);
  auto res = gko::as<gko::matrix::Dense<double>>(logger->get_residual_norm());
  std::cout << "Final residual norm sqrt(r^T r): \n";
  write(std::cout, res);
  // ----------------------------------------------------------

  MatDiagonalScale(K,boundaryzeros,boundaryzeros); // MatDiagonalScale(Mat mat,Vec left scaling,Vec right scaling)
  MatDiagonalScale(Kc,boundaryzeros,boundaryones); // Compenstaro matrix to adjust terms in rhs
  MatDiagonalScale(M,boundaryzeros,boundaryzeros); 

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::map_dofs_to_support_points<dim>(MappingQGeneric<dim>(fe.degree + 1), dof_handler, current_point_map); 
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(),  mpi_communicator, locally_relevant_dofs);  
  sparsity_pattern.copy_from(dsp);

  // loop over nonzero elements - currently I don't understand the deal.ii iteratiorss
  // TODO
  //auto iterator = sparsity_pattern.begin();
  //while(iterator != sparsity_pattern.end()){
    //std::cout<<"test";
    //iterator.Accessor();
    //iterator++;
  //}

  // loop over all elements
  //for( unsigned int i = 0; i<dof_handler.n_dofs(), ++i){
  //}

  //M_d2.copy_from(M); // Try to set deal.ii matrix using a PETSc matrix
    // create combination D * K * D

    for( unsigned int i:locally_owned_dofs){
   	if( locally_owned_dofs.is_element(i) ){
		if( boundary_points.find(i) != boundary_points.end() ){
			K.set(i,i,1); 	
		}
		else{
			boundaryones(i) = 0;
		}
    	}
    }
    K.compress(VectorOperation::insert); 
    boundaryones.compress(VectorOperation::insert);
    AMGblock.copy_from(K); // This will be used with reversed sign but we set it to K for readability, consider changing 
  }
  
  template <int dim>
  void IRK_SP<dim>::IRK_run()
  {

    setup_system(); 
    
    pcout << "   Number of active cells:       "
	  << triangulation.n_global_active_cells() << std::endl
	  << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl
	  << "   Full system dimension: " << dof_handler.n_dofs()*q 
	  << std::endl;
	      
    Assemble();

    // Now we have the ginkgo-matricies
    //gko_K, gko_M, gko_Kc, gko_AMGblock
    // print test gko::write(std::cout, gko_K);

    // Next step is to create a preconditioner using ginko stuctures,
    // for this we initally use a AMG based gko_AMGblock= K + M
    
    // Multigrid example
    auto exec = gko::ReferenceExecutor::create(); //Ref CPU exec

    using mg = gko::solver::Multigrid;
    using pgm = gko::multigrid::Pgm<double>;
    std::shared_ptr<gko::LinOpFactory> multigrid_gen = mg::build().with_mg_level(pgm::build().with_deterministic(true).on(exec))
          .with_criteria(gko::stop::Iteration::build().with_max_iters(1u).on(exec)).on(exec);

    double tolerance = tol_inner; 
    using cg = gko::solver::Cg<double>;
    auto solver_gen =  cg::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                gko::stop::ResidualNorm<double>::build()
                .with_baseline(gko::stop::mode::absolute)
                .with_reduction_factor(tolerance)
                .on(exec))
                .with_preconditioner(multigrid_gen)
                .on(exec);

  std::shared_ptr< gko::solver::Cg<double>> CGAMGsolver = solver_gen->generate(gko_AMGblock);

  std::shared_ptr<const gko::log::Convergence<double>> logger = gko::log::Convergence<double>::create();
  CGAMGsolver->add_logger(logger);

  std::vector<std::shared_ptr<gko::matrix::Dense<double>>> ginkgo_block_vec(q);
  std::vector<std::shared_ptr<gko::matrix::Dense<double>>> ginkgo_block_vec2(q);

  /*
  auto b = gko::matrix::Dense<double>::create(exec, gko::dim<2>(dof_handler.n_dofs(), 1));
  b->fill(1.0);
  auto x = gko::clone(b);
  x->fill(0.0);
  solver->apply(b, x);
  auto res = gko::as<gko::matrix::Dense<double>>(logger->get_residual_norm());
  std::cout << "Final residual norm sqrt(r^T r): \n";
  write(std::cout, re);
  */


 // Attempt to construct block-vectors with Ginkgo with vectors

  auto ginkgosystem = std::make_shared<IRKMatrix_Ginkgo>(gko_K,gko_M,A_inv,tau);

  auto ginginkgoprecon= std::make_shared<Preconditioner_ginkgo>(gko_K,gko_M,T_mat,T_mat_inv,D_vec,tau,gko_AMGblock,CGAMGsolver);

    using gmres = gko::solver::Gmres<double>;
    auto solver_gmres =  gmres::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                gko::stop::ResidualNorm<double>::build()
                .with_baseline(gko::stop::mode::absolute)
                .with_reduction_factor(tol_outer)
                .on(exec))
                .with_generated_preconditioner(ginginkgoprecon)
                .with_flexible(true)
                .on(exec)->generate(ginkgosystem);

  std::shared_ptr<const gko::log::Convergence<double>> loggergmres = gko::log::Convergence<double>::create();
  solver_gmres->add_logger(loggergmres);


  //    Preconditioner_ginkgo(s
  //                   const FullMatrix<double> &T_mat, const FullMatrix<double> &T_mat_inv, const std::vector<double> &D_vec, const double tau, 
	//	     std::shared_ptr<gko::matrix::Csr<double>> &AMGblock, std::shared_ptr<gko::LinOp> &CGAMGsolver)

  for( unsigned int i = 0; i<q; ++i){
    ginkgo_block_vec[i]=gko::matrix::Dense<double>::create(exec, gko::dim<2>(dof_handler.n_dofs(), 1));
    ginkgo_block_vec2[i]=gko::matrix::Dense<double>::create(exec, gko::dim<2>(dof_handler.n_dofs(), 1));
    ginkgo_block_vec[i]->fill(1.0);
    ginkgo_block_vec2[i]->fill(1.0);   
  }
    // Attempt to construct block-vectors with Ginkgo by full length vectors:
  auto ginkgo_block_vec_fl = gko::matrix::Dense<double>::create(exec, gko::dim<2>(q*dof_handler.n_dofs(), 1));
  auto ginkgo_block_vec_fl2 = gko::matrix::Dense<double>::create(exec, gko::dim<2>(q*dof_handler.n_dofs(), 1));

  ginkgo_block_vec_fl->fill(1.0);
  ginkgo_block_vec_fl2->fill(1.0);

  solver_gmres->apply(ginkgo_block_vec_fl , ginkgo_block_vec_fl2);
  auto res = gko::as<gko::matrix::Dense<double>>(loggergmres->get_residual_norm());
  std::cout << "Final residual norm sqrt(r^T r): \n";
  std::cout << "Number of iterations = ";
  std::cout<< loggergmres->get_num_iterations();
  write(std::cout, res);

  //ginkosystem.apply(ginkgo_block_vec,ginkgo_block_vec2);

  //ginkosystem.apply(ginkgo_block_vec_fl,ginkgo_block_vec_fl2);
  //ginginkgoprecon->apply(ginkgo_block_vec_fl,ginkgo_block_vec_fl2);


  // Ginkgo test
  //ginkgo_block_vec_fl->fill(1.0);
  //ginkgo_block_vec_fl2->fill(1.0);
  //ginkgosystem->apply(ginkgo_block_vec_fl,ginkgo_block_vec_fl2);
  //std::cout<<" Ginkgo matvec all ones res "<<"\n";
  //write(std::cout, ginkgo_block_vec_fl2);

  // PETSc test
      // Set up our class structures
    IRKMatrix oursystem(K,M,A_inv,temp_vec,tau); // Main matrix 

    //rhs_vec_block = 1;
    //rhs_scaled_vec_block = 1;

    //oursystem.vmult(rhs_vec_block,rhs_scaled_vec_block);
    //std::cout<<" PETSc matvec all ones res "<<"\n";

    //rhs_vec_block.print(std::cout);


 
     // inner solver and preconditioner 
    SolverControl cn_CG;
    cn_CG.set_tolerance(tol_inner); // Inner solver tolerance
    SolverCG<LA::MPI::Vector> CGsolver(cn_CG);
    Preconditioner ourprecon(K,M,T_mat,T_mat_inv, D_vec, tau , AMGblock , temp_vec_block, cn_CG, CGsolver );
    ourprecon.AMG_setup();

    // Outer Solver
    SolverFGMRES<LA::MPI::BlockVector> FGMRESsolver(cn_GMRES);
    cn_GMRES.set_tolerance(tol_outer); 

    // Below we take N timesteps with RK    
    assemble_u0( u0_vec , t0); // assemble initial state
    for( unsigned int k=0; k<N; ++k){
    	pcout<<"============================"<<"\n";
	    current_timpestep += 1;
    	pcout<<"Running for t = "<< t0 <<"\n";
    	assemble_rhs( rhs_vec_block, rhs_scaled_vec_block , t0);
    	try{ FGMRESsolver.solve(oursystem,k_vec_block,rhs_scaled_vec_block,ourprecon); }catch(...){
	  	pcout<<"Did not converge :("<<"\n";
	}
  pcout<<" Solved in "<< cn_GMRES.last_step() << " iterations." << "\n"; 
	temp_vec = 0; // used to sum RK terms
	for( unsigned int i=0; i<q; i++){
		temp_vec.add(b_vec[i],k_vec_block.block(i));
	}
	u1_vec = u0_vec;
	u1_vec.add(tau,temp_vec);
	t1 = t0+tau;
	
	// Set t0=t1, u0=u1 and take next step
	t0 = t1;
	u0_vec = u1_vec;
	compare_against_analytical(); // Evavluates analytical sol and prints difference, also saves sol to file
    	pcout<<"============================"<<"\n";
    }


  }

  template <int dim>
  void IRK_SP<dim>::output_results() const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_u0, "u_cal");
    data_out.add_data_vector(locally_relevant_u_tru, "u_tru");
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();
    data_out.set_flags(DataOutBase::VtkFlags(t0, current_timpestep));
    
    // The next step is to write this data to disk. We write up to 8 VTU files
    // in parallel with the help of MPI-IO. Additionally a PVTU record is
    // generated, which groups the written VTU files.
    const std::string filename =  "solution-" + Utilities::int_to_string(current_timpestep, 3) + ".vtk";
    data_out.write_vtu_with_pvtu_record( "./", filename, current_timpestep, mpi_communicator, 2, 8);
  }

  template <int dim>
  void IRK_SP<dim>::run()
  {
    	pcout << "Running with " << "PETSc"
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

        GridGenerator::hyper_cube(triangulation, 0, 1);
        triangulation.refine_global(number_of_refinements);
        pcout << "Number of IRK stages " << q << std::endl;
        pcout << "Expected order " << 2*q-1 << std::endl;
	readin_IRK(); // read in relevant IRK structures and factorizations
	IRK_run(); // Solve coarse mesh problem 
      
  }
} 

int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace IRK;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      IRK_SP<2> IRK_Example;
      IRK_Example.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

