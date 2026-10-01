
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exception_macros.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>

#include "binary_mesh_reader.h"

template <int dim>
bool
check_tetrahedron_determinant(const std::vector<dealii::Point<dim>> &vertices,
                              const dealii::CellData<dim>           &cell)
{
  static_assert(dim == 3, "This function is only for 3D tetrahedra.");

  AssertDimension(cell.vertices.size(), 4);

  const dealii::Point<3> &v0 = vertices[cell.vertices[0]];

  const dealii::Point<3> &v1 = vertices[cell.vertices[1]];

  const dealii::Point<3> &v2 = vertices[cell.vertices[2]];

  const dealii::Point<3> &v3 = vertices[cell.vertices[3]];

  const dealii::Tensor<1, 3> a = v1 - v0;
  const dealii::Tensor<1, 3> b = v2 - v0;
  const dealii::Tensor<1, 3> c = v3 - v0;

  const double det = dealii::scalar_product(a, dealii::cross_product_3d(b, c));

  return det > 1e-12;
}

template <int dim>
double
check_prism_jacobian_determinant(
  const std::vector<dealii::Point<dim>> &vertices,
  const dealii::CellData<dim>           &cell)
{
  static_assert(dim == 3, "The prism check is implemented for dim=3.");

  AssertDimension(cell.vertices.size(), 6);

  const dealii::Point<3> &x0 = vertices[cell.vertices[1]];

  const dealii::Point<3> &x1 = vertices[cell.vertices[2]];

  const dealii::Point<3> &x2 = vertices[cell.vertices[0]];

  const dealii::Point<3> &x3 = vertices[cell.vertices[4]];

  const dealii::Point<3> &x4 = vertices[cell.vertices[5]];

  const dealii::Point<3> &x5 = vertices[cell.vertices[3]];

  const double r = 1. / 3.;
  const double s = 1. / 3.;
  const double t = 1. / 2.;

  const dealii::Tensor<1, 3> dx_dr = (1.0 - t) * (x0 - x2) + t * (x3 - x5);

  const dealii::Tensor<1, 3> dx_ds = (1.0 - t) * (x1 - x2) + t * (x4 - x5);

  const dealii::Point<3> bottom_point = r * x0 + s * x1 + (1.0 - r - s) * x2;

  const dealii::Point<3> top_point = r * x3 + s * x4 + (1.0 - r - s) * x5;

  const dealii::Tensor<1, 3> dx_dt = top_point - bottom_point;

  /*
   * det(J) = (dx/dr x dx/ds) . dx/dt
   */
  const double det =
    dealii::scalar_product(dealii::cross_product_3d(dx_dr, dx_ds), dx_dt);

  return det > 1e-12;
}

using namespace dealii;

template <int dim>
void
verify_meshes_equal(const std::vector<dealii::Point<dim>>    &original_vertices,
                    const std::vector<dealii::CellData<dim>> &original_cells,
                    const BinaryMeshData<dim>                &loaded_mesh)
{
  AssertThrow(original_vertices.size() == loaded_mesh.vertices.size(),
              dealii::ExcMessage("Vertex counts differ after binary loading."));

  AssertThrow(original_cells.size() == loaded_mesh.cells.size(),
              dealii::ExcMessage("Cell counts differ after binary loading."));

  for (std::size_t i = 0; i < original_vertices.size(); ++i)
    for (unsigned int d = 0; d < dim; ++d)
      AssertThrow(original_vertices[i][d] == loaded_mesh.vertices[i][d],
                  dealii::ExcMessage(
                    "Vertex coordinates differ after loading."));

  for (std::size_t i = 0; i < original_cells.size(); ++i)
    {
      AssertThrow(original_cells[i].vertices == loaded_mesh.cells[i].vertices,
                  dealii::ExcMessage(
                    "Cell connectivity differs after loading."));

      AssertThrow(original_cells[i].material_id ==
                    loaded_mesh.cells[i].material_id,
                  dealii::ExcMessage("Material IDs differ after loading."));

      AssertThrow(original_cells[i].manifold_id ==
                    loaded_mesh.cells[i].manifold_id,
                  dealii::ExcMessage("Manifold IDs differ after loading."));
    }
}


template <int dim>
void
prepare_grid()
{
  std::cout << "Read data" << std::endl;
  const std::string input_file = "mesh.bin";
  const Mesh        mesh       = read_mesh_binary(input_file);

  std::vector<Point<dim>> vertices(mesh.vertices.size());
  for (unsigned int i = 0; i < vertices.size(); ++i)
    for (unsigned int d = 0; d < dim; ++d)
      vertices[i][d] = mesh.vertices[i][d];

  const unsigned int number_of_vertices = vertices.size();
  const unsigned int number_of_cells    = mesh.cells.size();

  std::vector<long unsigned int> cell_numbering_inverse;

  {
    std::cout << "Reorder data" << std::endl;
    std::vector<boost::container::small_vector<unsigned int, 16>>
      vertex_to_cell(number_of_vertices);
    for (unsigned int cell_index = 0; cell_index < number_of_cells;
         ++cell_index)
      for (const auto vertex_index : mesh.cells[cell_index])
        vertex_to_cell[vertex_index].push_back(cell_index);

    dealii::DynamicSparsityPattern cell_connectivity;
    cell_connectivity.reinit(number_of_cells, number_of_cells);
    std::vector<types::global_dof_index> neighbors;
    for (unsigned int cell_index = 0; cell_index < number_of_cells;
         ++cell_index)
      {
        neighbors.clear();
        for (const auto vertex_index : mesh.cells[cell_index])
          neighbors.insert(neighbors.end(),
                           vertex_to_cell[vertex_index].begin(),
                           vertex_to_cell[vertex_index].end());
        std::sort(neighbors.begin(), neighbors.end());
        cell_connectivity.add_entries(cell_index,
                                      neighbors.begin(),
                                      std::unique(neighbors.begin(),
                                                  neighbors.end()),
                                      true);
      }

    std::vector<long unsigned int> cell_numbering;
    cell_numbering.resize(number_of_cells);
    SparsityTools::reorder_hierarchical(cell_connectivity, cell_numbering);
    cell_numbering_inverse = Utilities::invert_permutation(cell_numbering);
  }

  std::cout << "Setup reorder data" << std::endl;
  std::vector<CellData<dim>> cells(number_of_cells);

  for (unsigned int idx = 0; idx < number_of_cells; ++idx)
    {
      const unsigned int i = cell_numbering_inverse[idx];
      CellData<dim>      celldata;

      const unsigned int n_vertices = mesh.cells[i].size();
      celldata.vertices.resize(n_vertices);
      for (unsigned int v = 0; v < n_vertices; ++v)
        celldata.vertices[v] =
          boost::numeric_cast<unsigned int>(mesh.cells[i][v]);

      if (n_vertices == 4)
        {
          if (!check_tetrahedron_determinant(vertices, celldata))
            std::swap(celldata.vertices[2], celldata.vertices[3]);
          if (!check_tetrahedron_determinant(vertices, celldata))
            {
              bool continue_swapping = true;
              while (continue_swapping &&
                     std::next_permutation(celldata.vertices.begin(),
                                           celldata.vertices.end()))
                continue_swapping =
                  !check_tetrahedron_determinant(vertices, celldata);
            }
          if (!check_tetrahedron_determinant(vertices, celldata))
            Assert(false, ExcNotImplemented());
        }
      else if (n_vertices == 5)
        {
          DEAL_II_NOT_IMPLEMENTED();
        }
      else if (n_vertices == 6)
        {
          if (!check_prism_jacobian_determinant(vertices, celldata))
            {
              std::swap(celldata.vertices[1], celldata.vertices[2]);
              std::swap(celldata.vertices[4], celldata.vertices[5]);
            }
          if (!check_prism_jacobian_determinant(vertices, celldata))
            {
              bool continue_swapping = true;
              while (continue_swapping &&
                     std::next_permutation(celldata.vertices.begin(),
                                           celldata.vertices.end()))
                continue_swapping =
                  !check_prism_jacobian_determinant(vertices, celldata);
            }
          if (!check_prism_jacobian_determinant(vertices, celldata))
            Assert(false, ExcNotImplemented());
        }
      else if (n_vertices == 8)
        {
          DEAL_II_NOT_IMPLEMENTED();
        }
      else
        DEAL_II_NOT_IMPLEMENTED();

      cells[idx] = celldata;
    }

  std::cout << "Save reordered mesh" << std::endl;

  save_mesh_binary<dim>("reordered_mesh.bin", vertices, cells);


  std::cout << "Verify" << std::endl;
  BinaryMeshData<dim> mesh_loaded = load_mesh_binary<dim>("reordered_mesh.bin");

  verify_meshes_equal(vertices, cells, mesh_loaded);
  std::cout << "Verified" << std::endl;
}

int
main()
{
  prepare_grid<3>();

  return 1;
}