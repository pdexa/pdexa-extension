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

#include <chrono>



namespace IRK
{
  using namespace dealii;


  void dealiimatscaleadd( SparseMatrix<double> & Mat, LA::MPI::Vector & left_scaling, LA::MPI::Vector & right_scaling, LA::MPI::Vector & diag_add ){
    // This is a helper function to use the PETSc vectors from the old code to manually impose B.C. and to generate the right AMG-blocks. 
    for (auto &entry : Mat){
          //std::cout<<"row = "<<entry.row()<<"\n";
          //std::cout<<"col = "<<entry.column()<<"\n";
          //std::cout<<"Indivial test "<< Mat(entry.row(),entry.column() ) << "\n";
          // Scale entry by row/col scalings
          Mat.set(entry.row(),entry.column(), left_scaling[entry.row()]*Mat(entry.row(),entry.column())*right_scaling[entry.column()] );
          // add diagonal term
          if (entry.row() == entry.column() )
          {
            Mat.set(entry.row(),entry.column(), Mat(entry.row(),entry.column()) + diag_add[entry.row()] );
            //std::cout<<"Diag value = " << entry.value()<<"\n";
          }
          //std::cout<<"Change value of entry "<<"\n";
          //std::cout<<"Value = " << entry.value()<<"\n";
    }
  }

  void dealiimatscale( SparseMatrix<double> & Mat, LA::MPI::Vector & left_scaling, LA::MPI::Vector & right_scaling){
  for (auto &entry : Mat){
        Mat.set(entry.row(),entry.column(), left_scaling[entry.row()]*Mat(entry.row(),entry.column())*right_scaling[entry.column()] );
  }
}

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
    void compare_against_analytical_ginkgo(std::shared_ptr<gko::matrix::Dense<double>> &u0_ginkgo);

    //void dealiimatscaleadd( SparseMatrix<double> & Mat, LA::MPI::Vector & left_scaling, LA::MPI::Vector & right_scaling, LA::MPI::Vector & diag_add );


    double u_analytical( double x, double y, double t,  double a_x, double a_y, double a_t);
    double g_testcase( double x, double y, double t,  double a_x, double a_y, double a_t);

    void assemble_u0( LA::MPI::Vector &u0 ,double t0);
    void assemble_u0_ginkgo( std::shared_ptr<gko::matrix::Dense<double>> u0 ,double t0);

    void assemble_rhs( LA::MPI::BlockVector &rhs,LA::MPI::BlockVector &rhs_scaledss ,double t);
    void assemble_rhs_ginkgo( std::shared_ptr<gko::matrix::Dense<double>> &rhs, std::shared_ptr<gko::matrix::Dense<double>> &rhs_scaled, std::shared_ptr<gko::matrix::Dense<double>> &u0, std::shared_ptr<gko::matrix::Dense<double>> &tempvec, double t);


    void output_results() const;

    unsigned int q = 5; // number of stages, q = 2, ..., 10
    unsigned int N = 10; // Number of timesteps
    double final_time = 0.5;
    double tau = final_time/N; // Timestep size
    unsigned int current_timpestep = 0;
    unsigned int number_of_refinements = 8; // How many times to refine unit square

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

    std::vector<std::shared_ptr<Mtx>> AMGblocks_list =  std::vector<std::shared_ptr<Mtx>>(q);
    std::vector<std::shared_ptr<gko::LinOp>> CGAMGsolver_list = std::vector<std::shared_ptr<gko::LinOp>>(q); // List for q different AMG-multigrid solvers
    std::vector<std::shared_ptr<gko::log::Convergence<double>>> logger_list = std::vector<std::shared_ptr<gko::log::Convergence<double>>>(q);

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
    void IRK_SP<dim>::assemble_u0_ginkgo( std::shared_ptr<gko::matrix::Dense<double>> u0 ,double t0){
      for( unsigned int i:locally_owned_dofs){ 
         //u0[i] = u_analytical( current_point_map[i][0] , current_point_map[i][1] ,t0, a_x, a_y, a_t);
         u0->at(i,0) = u_analytical( current_point_map[i][0] , current_point_map[i][1] ,t0, a_x, a_y, a_t); // Set the values of u0 (only works on CPU executors?)
      }
      //u0.compress(VectorOperation::insert);
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
    void IRK_SP<dim>::assemble_rhs_ginkgo( std::shared_ptr<gko::matrix::Dense<double>> &rhs, std::shared_ptr<gko::matrix::Dense<double>> &rhs_scaled,  std::shared_ptr<gko::matrix::Dense<double>> &u0, std::shared_ptr<gko::matrix::Dense<double>> &tempvec, double t){

      using Vec = gko::matrix::Dense<double>;

      //Vec = gko::matrix::Dense<double>;
      auto n = u0->get_size()[0];
      auto exec = u0->get_executor();
      //std::cout<<"TEST n = "<<n<<"\n";

      //auto rhs_dense = gko::as<gko::matrix::Dense<double>>(rhs);

      // Quasi block vector things 
      auto block = [n, exec](auto vec, int i){
        auto data = vec->get_values();
        return gko::matrix::Dense<double>::create(exec, gko::dim<2>{n, 1}, gko::make_array_view(exec, n, data + n * i), 1);
      };

      auto const_block = [n, exec](const auto vec, int i){
        auto data = vec->get_const_values();
        return gko::matrix::Dense<double>::create_const(exec, gko::dim<2>{n, 1}, gko::make_const_array_view(exec, n, data + n * i), 1);
      };


      for( unsigned int j=0; j<q; j++){
        block(rhs, j)->fill(0.0); 
        //write(std::cout,  block(rhs, j));

        for( unsigned int i:locally_owned_dofs){ 
          //temp_vec2[i] = g_testcase(current_point_map[i][0] , current_point_map[i][1], t+c_vec[j]*tau, a_x, a_y, a_t);
          tempvec->at(i,0) =  g_testcase(current_point_map[i][0] , current_point_map[i][1], t+c_vec[j]*tau, a_x, a_y, a_t); // Only works on CPU executors? 
        }
        //M.vmult(temp_vec,temp_vec2); // temp_vec = M*g_vec_j
        //rhs.block(j) = temp_vec; // rhs.block(j) = M*g_vec_j
        gko_M->apply(gko::initialize<Vec>({1.0}, exec), tempvec, gko::initialize<Vec>({0.0}, exec), block(rhs, j)); //rhs[j] = 1*M*tempvec
        //write(std::cout,  block(rhs, j));

        //K.vmult(temp_vec,u0_vec); // temp_vec = K*u0_vec ; note we use Ku0 repededly, maybe move this vmult out of loop and into u0 construction
        //rhs.block(j).add(-1,temp_vec); // rhs.block(j) = M*g_vec_j -  K*u0_vec
        gko_K->apply(gko::initialize<Vec>({-1.0}, exec), u0, gko::initialize<Vec>({1.0}, exec), block(rhs, j));

      }

      // Apply scaling so rhs corresponds to the transformed system
      // rhs_scaled = apply_transformation( rhs, A_inv )
      //rhs_scaled = 0;
      for( unsigned int i=0; i<q; i++){
        block(rhs_scaled, i)->fill(0.0); 
        for( unsigned int j=0; j<q; j++){
          block(rhs_scaled, i)->add_scaled( gko::initialize<Vec>({A_inv(i,j)}, exec), block(rhs, j)); // Ginkgo vers
          //  rhs_scaled.block(i).add(A_inv(i,j), rhs.block(j) ); 
        }
        //write(std::cout,  block(rhs_scaled, i));
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

    template <int dim>
    void IRK_SP<dim>::compare_against_analytical_ginkgo(std::shared_ptr<gko::matrix::Dense<double>> &u0_ginkgo){
      // We still use PETSc vecs for output
      for( unsigned int i:locally_owned_dofs){ 
        temp_vec[i] = u_analytical( current_point_map[i][0] , current_point_map[i][1] ,t0, a_x, a_y, a_t);
        u0_vec[i] = (u0_ginkgo->at(i,0)); // Copy ginkgo values into petsc
      }
      temp_vec.compress(VectorOperation::insert);
      u0_vec.compress(VectorOperation::insert);

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
        std::cout<<"vmult start in apply"<<"\n";
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
        //std::cout<<"vmult start in apply_impl"<<"\n";


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
        //std::cout<<"Testing apply_impl for System matrix by Ginkgo"<<"\n";
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
		                 std::vector<std::shared_ptr<gko::LinOp>> CGAMGsolver_list, std::vector<std::shared_ptr<gko::log::Convergence<double>>> logger_list)
        : gko::EnableLinOp<Preconditioner_ginkgo>(K->get_executor(), gko::dim<2>(K->get_size()[0] * T_mat.n(), K->get_size()[1] * T_mat.n())),
         K(K)
        , M(M)
        , T_mat(T_mat)
        , T_mat_inv(T_mat_inv)
        , D_vec(D_vec)
        , tau(tau)
        , CGAMGsolver_list(CGAMGsolver_list)
        , logger_list(logger_list)
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

        //std::cout<<"Print precon apply_impl rhs before application"<<"\n";
        //write(std::cout, b_dense);

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

        // zero in separate loop during debug
        //for( unsigned int i = 0; i<T_mat.n(); i++){
        //    block(x_dense, i)->fill(0.0); // set all output values to zero
        //}

        for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 
            block(x_dense, i)->fill(0.0); // set all output values to zero
            for( unsigned int j = 0; j < T_mat.n(); ++j  ){ 
              if( abs(T_mat_inv(i,j)) > 1e-12){ // T,T_inv are currently triangular so this check saves some work
                block(x_dense, i)->add_scaled( gko::initialize<Vec>({T_mat_inv(i,j)}, exec), const_block(b_dense, j)); // Ginkgo vers
                //dst.block(i).add(T_mat_inv(i,j),src.block(j));
              }
            }


        }

      for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 
        CGAMGsolver_list[i]->apply(block(x_dense, i), block(temp_vec.get(),i) );
        //std::cout<<"Inner its for block nr "<<i<< " = "<< logger_list[i] ->get_num_iterations() << "\n";
      }	

      // Apply transformation to sol
      //dst = 0;
      // zero in separate loop during debug
      for( unsigned int i = 0; i<T_mat.n(); i++){
          block(x_dense, i)->fill(0.0); // set all output values to zero
      }
      for( unsigned int i = 0; i < T_mat.n(); ++i  ){ 
          for( unsigned int j = 0; j < T_mat.n(); ++j  ){ 
            if( abs(T_mat(i,j)) > 1e-12){ // T,T_inv are currently triangular so this check saves some work
             // dst.block(i).add(T_mat(i,j),temp_vec_block.block(j));
             //std::cout<<"T_mat(i,j) = "<<T_mat(i,j)<<"\n";
             //std::cout<<"(i,j) = "<<i<<","<<j<<"\n";
             //std::cout<<"Vector to be added "<<"\n";
            //write(std::cout,block(temp_vec.get(), j));
             block(x_dense, i)->add_scaled( gko::initialize<Vec>({T_mat(i,j)}, exec), block(temp_vec.get(), j)); // Ginkgo vers

            }
          }
      }

      //std::cout<<"After precon application (Check if x_dense is correct output)"<<"\n";
      //write(std::cout, x_dense);  
      //std::cout<<"End output precon application (Check if x_dense is correct output)"<<"\n";

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
        //std::shared_ptr<gko::matrix::Csr<double>> AMGblock;
        std::vector<std::shared_ptr<gko::LinOp>> CGAMGsolver_list;
        std::vector<std::shared_ptr<gko::log::Convergence<double>>> logger_list;
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

    //gko::write(std::cout, gko_K); // output matrix

    K.compress(VectorOperation::add);
    M.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    std::set< types::global_dof_index > boundary_points ;
    test_map_dof_to_boundary_indices( dof_handler ,  boundary_points ) ;

    // create diagonal matrices from boundary zeros
    auto exec = gko::ReferenceExecutor::create(); //Ref CPU exec

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
    //auto diag = gko::matrix::Diagonal<double>::create(exec, dof_handler.n_dofs(), gko_boundary_points);
    //auto gko_row_scale_K = gko::clone(gko_K);
    //auto gko_scaled_K = gko::clone(gko_K);
    //diag->apply(gko_K, gko_row_scale_K);
    //diag->rapply(gko_row_scale_K, gko_scaled_K);
    //auto id = gko::matrix::Identity<double>::create(exec, dof_handler.n_dofs());

  MatDiagonalScale(K,boundaryzeros,boundaryzeros); // MatDiagonalScale(Mat mat,Vec left scaling,Vec right scaling)
  MatDiagonalScale(Kc,boundaryzeros,boundaryones); // Compenstaro matrix to adjust terms in rhs
  MatDiagonalScale(M,boundaryzeros,boundaryzeros); 

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::map_dofs_to_support_points<dim>(MappingQGeneric<dim>(fe.degree + 1), dof_handler, current_point_map); 
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(),  mpi_communicator, locally_relevant_dofs);  
  sparsity_pattern.copy_from(dsp);


  // Transfer to Ginkgo structures:
  std::cout<<"Manually transering BC to deal.ii matricies..."<<"\n";
  dealiimatscaleadd(  K_d2, boundaryzeros, boundaryzeros, boundaryones);
  dealiimatscale(  M_d2 , boundaryzeros, boundaryzeros);
  std::cout<<"End Manually transering BC to deal.ii matricies..."<<"\n";

  // Fix AMG-block TODO: check the BC is correct
  //AMGblock_d2.copy_from(K_d2);
  //AMGblock_d2*=tau;
  //AMGblock_d2.add(D_vec[0],M_d2);

  gko_K = dealii::GinkgoInterface::create_csr_matrix(exec, K_d2);
  gko_M = dealii::GinkgoInterface::create_csr_matrix(exec, M_d2);
  gko_Kc = dealii::GinkgoInterface::create_csr_matrix(exec, Kc_d2);
  //gko_AMGblock = dealii::GinkgoInterface::create_csr_matrix(exec, AMGblock_d2);

  // Multigrid machenery
  using mg = gko::solver::Multigrid;
  using pgm = gko::multigrid::Pgm<double>;
  using ic = gko::preconditioner::Ic<gko::solver::LowerTrs<double>>;
  using cg = gko::solver::Cg<double>;

  auto ic_gen = gko::share(
      ic::build()
          .on(exec));

  auto smoother_gen = gko::share(
      gko::solver::build_smoother(ic_gen, 1u, static_cast<double>(0.9)));

  const gko::remove_complex<double> tolerance_amg = 1e-8;


  auto iter_stop =
      gko::share(gko::stop::Iteration::build().with_max_iters(3u).on(exec));
  auto tol_stop = gko::share(gko::stop::ResidualNorm<double>::build()
                                  .with_baseline(gko::stop::mode::absolute)
                                  .with_reduction_factor(tolerance_amg)
                                  .on(exec));
  auto exact_tol_stop =
      gko::share(gko::stop::ResidualNorm<double>::build()
                      .with_baseline(gko::stop::mode::rhs_norm)
                      .with_reduction_factor(1e-8)
                      .on(exec));

auto coarsest_gen = gko::share(cg::build()
                                  .with_preconditioner(ic_gen)
                                  .with_criteria(iter_stop, exact_tol_stop)
                                  .on(exec));

  std::shared_ptr<gko::LinOpFactory> multigrid_gen = mg::build().with_mg_level(pgm::build().with_deterministic(true).on(exec)).with_min_coarse_rows(32u).with_max_levels(10u).with_coarsest_solver(coarsest_gen).with_pre_smoother(smoother_gen)
  .with_criteria(iter_stop, tol_stop).on(exec);
  
//         .with_criteria(iter_stop, tol_stop).on(exec);
//.with_criteria(gko::stop::Iteration::build().with_max_iters(1u).on(exec)).on(exec);

  double tolerance = tol_inner; 
  auto solver_gen = cg::build().with_criteria(
              gko::stop::Iteration::build().with_max_iters(100u).on(exec),
              gko::stop::ResidualNorm<double>::build()
              .with_baseline(gko::stop::mode::absolute)
              .with_reduction_factor(tolerance)
              .on(exec))
              .with_preconditioner(multigrid_gen)
              .on(exec);

  //Create q different CG-AMG solver for the preconditioner blocks
  std::cout<<"Setting up Ginkgo block multigrid solvers..."<<"\n";
  auto t1 = std::chrono::high_resolution_clock::now();
  for( unsigned int i=0; i<q ; i++){
    AMGblock_d2.copy_from(K_d2);
    AMGblock_d2*=tau;
    AMGblock_d2.add(D_vec[i],M_d2);

    // Store all blocks in list (in case some pointer magic going on)
    AMGblocks_list[i] = dealii::GinkgoInterface::create_csr_matrix(exec, AMGblock_d2);
    CGAMGsolver_list[i] = solver_gen->generate(AMGblocks_list[i]);

    //Add loggers
    logger_list[i] =gko::share(gko::log::Convergence<double>::create());
    CGAMGsolver_list[i]->add_logger(logger_list[i]);


  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

  std::cout<<"End setting up Ginkgo block multigrid solvers..."<<"\n";
  std::cout<<" Time in milliseconds = " << duration<<"\n";


  // Test reusing AMG-block, it seems to work

  //-----------------------------------------------------------------------
  //-----------------------------
  // Multigrid example


  /*
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
*/
//auto b = gko::matrix::Dense<double>::create(exec, gko::dim<2>(dof_handler.n_dofs(), 1));
//b->fill(1.0);
//auto x = gko::clone(b);
//x->fill(0.0);

//solver->apply(b, x);
//auto res = gko::as<gko::matrix::Dense<double>>(logger->get_residual_norm());
//std::cout << "Final residual norm sqrt(r^T r): \n";
//write(std::cout, res);
// ----------------------------------------------------------


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
    AMGblock.copy_from(K); //

    std::cout<<"End assembly..."<<"\n";

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

std::cout<<"Set up ginkgo structures"<<"\n";
auto exec = gko::ReferenceExecutor::create(); //Ref CPU exec

auto ginkgosystem = std::make_shared<IRKMatrix_Ginkgo>(gko_K,gko_M,A_inv,tau);
auto ginginkgoprecon= std::make_shared<Preconditioner_ginkgo>(gko_K,gko_M,T_mat,T_mat_inv,D_vec,tau, CGAMGsolver_list, logger_list);

std::cout<<"Set up ginkgo outer solver"<<"\n";

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


std::cout<<"Set up ginkgo vectors..."<<"\n";


// Attempt to construct block-vectors with Ginkgo by full length vectors:
//auto ginkgo_block_vec_fl = gko::matrix::Dense<double>::create(exec, gko::dim<2>(q*dof_handler.n_dofs(), 1));
//auto ginkgo_block_vec_fl2 = gko::matrix::Dense<double>::create(exec, gko::dim<2>(q*dof_handler.n_dofs(), 1));
//ginkgo_block_vec_fl->fill(0.1);
//ginkgo_block_vec_fl2->fill(0.1);
//solver_gmres->apply(ginkgo_block_vec_fl , ginkgo_block_vec_fl2);
//auto res = gko::as<gko::matrix::Dense<double>>(loggergmres->get_residual_norm());
//std::cout << "Final residual norm sqrt(r^T r): \n";
//write(std::cout, res); // output residual
//std::cout << "Number of iterations = ";
//std::cout<< loggergmres->get_num_iterations()<<"\n";

// Ginkgo set up 
std::shared_ptr<gko::matrix::Dense<double>> u0_ginkgo = gko::matrix::Dense<double>::create(exec, gko::dim<2>(dof_handler.n_dofs(), 1));
std::shared_ptr<gko::matrix::Dense<double>> u1_ginkgo = gko::matrix::Dense<double>::create(exec, gko::dim<2>(dof_handler.n_dofs(), 1));

u0_ginkgo->fill(0.0);
u1_ginkgo->fill(0.0);

std::shared_ptr<gko::matrix::Dense<double>> temp_ginkgo = gko::matrix::Dense<double>::create(exec, gko::dim<2>(dof_handler.n_dofs(), 1));
temp_ginkgo->fill(0.0);

std::shared_ptr<gko::matrix::Dense<double>> rhs_block_ginko = gko::matrix::Dense<double>::create(exec, gko::dim<2>(q*dof_handler.n_dofs(), 1));
std::shared_ptr<gko::matrix::Dense<double>> rhs_block_scaled_ginko = gko::matrix::Dense<double>::create(exec, gko::dim<2>(q*dof_handler.n_dofs(), 1));
std::shared_ptr<gko::matrix::Dense<double>> k_vec_block_ginkgo = gko::matrix::Dense<double>::create(exec, gko::dim<2>(q*dof_handler.n_dofs(), 1));

rhs_block_ginko->fill(0.);
rhs_block_scaled_ginko->fill(0.);
k_vec_block_ginkgo->fill(0.);


//assemble_u0_ginkgo(u0_ginkgo,t0); // Assemble initial state Ginkgo
//assemble_rhs_ginkgo(rhs_block_ginko, rhs_block_scaled_ginko, u0_ginkgo, temp_ginkgo, t0 ); // Test assembly of RHS
//write(std::cout, rhs_block_ginko); // output residual

//------------------------------------------------------------------
std::cout<<"Set up PETSc structures..."<<"\n";

// PETSc test
// Set up our class structures
IRKMatrix oursystem(K,M,A_inv,temp_vec,tau); // Main matrix 
// inner solver and preconditioner PETSc
SolverControl cn_CG;
cn_CG.set_tolerance(tol_inner); // Inner solver tolerance
SolverCG<LA::MPI::Vector> CGsolver(cn_CG);
Preconditioner ourprecon(K,M,T_mat,T_mat_inv, D_vec, tau , AMGblock , temp_vec_block, cn_CG, CGsolver );
std::cout<<"Set up PETSc multigrid"<<"\n";
auto t1time = std::chrono::high_resolution_clock::now();
ourprecon.AMG_setup();
auto t2time = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2time - t1time ).count();
std::cout<<"End set up PETSc multigrid"<<"\n";
std::cout<<" Time in milliseconds = " << duration<<"\n";

// Outer Solver PETSc
SolverFGMRES<LA::MPI::BlockVector> FGMRESsolver(cn_GMRES);
cn_GMRES.set_tolerance(tol_outer); 


// Locally define block operations
auto n = u0_ginkgo->get_size()[0];
//auto exec = u0_ginkgo->get_executor();

auto block = [n, exec](auto vec, int i){
  auto data = vec->get_values();
  return gko::matrix::Dense<double>::create(exec, gko::dim<2>{n, 1}, gko::make_array_view(exec, n, data + n * i), 1);
};

auto const_block = [n, exec](const auto vec, int i){
  auto data = vec->get_const_values();
  return gko::matrix::Dense<double>::create_const(exec, gko::dim<2>{n, 1}, gko::make_const_array_view(exec, n, data + n * i), 1);
};




double maxdiff_petsc_ginkgo = 0;
// Below we take N timesteps with RK    
assemble_u0( u0_vec , t0); // assemble initial state PETSc
assemble_u0_ginkgo(u0_ginkgo,t0); // Assemble initial state Ginkgo
for( unsigned int k=0; k<N; ++k){
    pcout<<"============================"<<"\n";
    current_timpestep += 1;
    pcout<<"Running for t = "<< t0 <<"\n";

    // =============================
    assemble_rhs( rhs_vec_block, rhs_scaled_vec_block , t0);
    assemble_rhs_ginkgo(rhs_block_ginko, rhs_block_scaled_ginko, u0_ginkgo, temp_ginkgo, t0 ); 

    // Pointwise compare Ginkgo and PETSc rhs
    //maxdiff_petsc_ginkgo = 0;
    //for( unsigned int  i = 0; i<dof_handler.n_dofs()*q ; i++){
    //  if( std::abs( rhs_scaled_vec_block[i] - (rhs_block_scaled_ginko->at(i,0)) ) > maxdiff_petsc_ginkgo ) maxdiff_petsc_ginkgo = std::abs( rhs_scaled_vec_block[i] - (rhs_block_scaled_ginko->at(i,0)) );
    //}
    //std::cout<<"Maximal pointwise difference PETSc-Ginkgo rhs vec= "<< maxdiff_petsc_ginkgo <<"\n";


    // PETSc solve
    t1time = std::chrono::high_resolution_clock::now();
    std::cout<<"Start PETSc Solve."<<"\n";
    try{ FGMRESsolver.solve(oursystem,k_vec_block,rhs_scaled_vec_block,ourprecon); }catch(...){
    pcout<<"PETSc: Did not converge :("<<"\n";
    }
    pcout<<"PETSc: Solved in "<< cn_GMRES.last_step() << " iterations." << "\n"; 
    t2time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2time - t1time ).count();
    std::cout<<" PETSc solve Time in milliseconds = " << duration<<"\n";

    // Ginkgo solve
    std::cout<<"Start Ginkgo Solve."<<"\n";
    t1time = std::chrono::high_resolution_clock::now();
    k_vec_block_ginkgo -> fill(0.0); // Added in Debug, I dont understand why this is needed, TODO: Why does non-zero initial guess lead to wrong solution? 
    solver_gmres->apply(rhs_block_scaled_ginko , k_vec_block_ginkgo);
    t2time = std::chrono::high_resolution_clock::now();

    std::cout << "Ginkgo: Solved in ";
    std::cout<< loggergmres->get_num_iterations()<< " iterations." << "\n";
    duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2time - t1time ).count();
    std::cout<<" Ginkgo solve Time in milliseconds = " << duration<<"\n";
    // Pointwise compare Ginkgo and PETSc sol
    //maxdiff_petsc_ginkgo = 0;
    //for( unsigned int  i = 0; i<dof_handler.n_dofs()*q ; i++){
    //  if( std::abs( k_vec_block[i] - (k_vec_block_ginkgo->at(i,0)) ) > maxdiff_petsc_ginkgo ) maxdiff_petsc_ginkgo = std::abs(  k_vec_block[i] - (k_vec_block_ginkgo->at(i,0)) );
    //}
    //std::cout<<"Maximal pointwise difference PETSc-Ginkgo sol vec= "<< maxdiff_petsc_ginkgo <<"\n";


    // Petsc update
    temp_vec = 0; // used to sum RK terms
    for( unsigned int i=0; i<q; i++){
      temp_vec.add(b_vec[i],k_vec_block.block(i));
    }
    u1_vec = u0_vec;
    u1_vec.add(tau,temp_vec);

    // Ginkgo update
    temp_ginkgo->fill(0.0);
    for( unsigned int i=0; i<q; i++){
          temp_ginkgo->add_scaled( gko::initialize<gko::matrix::Dense<double>>({b_vec[i]}, exec), block(k_vec_block_ginkgo, i)); // Ginkgo vers
    }
    u1_ginkgo = u0_ginkgo;
    u1_ginkgo->add_scaled( gko::initialize<gko::matrix::Dense<double>>({tau}, exec), temp_ginkgo); 

    t1 = t0+tau; // Common Ginkgo/PETSc
    // Set t0=t1, u0=u1 and take next step


    t0 = t1; // Common Ginkgo/PETSc
    u0_vec = u1_vec;
    u0_ginkgo = u1_ginkgo;
    maxdiff_petsc_ginkgo = 0;
    // Pointwise compare Ginkgo and PETSc solutions
    for( unsigned int  i = 0; i<dof_handler.n_dofs() ; i++){
      if( std::abs( u0_vec[i] - (u0_ginkgo->at(i,0)) ) > maxdiff_petsc_ginkgo ) maxdiff_petsc_ginkgo = std::abs( u0_vec[i] - (u0_ginkgo->at(i,0)) );
    }
    std::cout<<"Maximal pointwise difference PETSc-Ginkgo = "<< maxdiff_petsc_ginkgo <<"\n";

    //compare_against_analytical(); //  Compares against analytical PETSc
    compare_against_analytical_ginkgo(u0_ginkgo); // Compares against analytical Ginkgo, NOTE: modifies PETSc vectors
    // END PETSc code =============================

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
