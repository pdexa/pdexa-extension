#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fs = std::filesystem;


template <int dim>
struct BinaryMeshData
{
  std::vector<dealii::Point<dim>>    vertices;
  std::vector<dealii::CellData<dim>> cells;
};


// =============================================================================
// Low-level binary helpers
// =============================================================================

template <typename Number>
void
write_binary_value_dealii(std::ostream &output, const Number value)
{
  static_assert(std::is_trivially_copyable_v<Number>,
                "Only trivially copyable types can be written directly.");

  output.write(reinterpret_cast<const char *>(&value),
               static_cast<std::streamsize>(sizeof(Number)));

  if (!output)
    throw std::runtime_error("Failed to write binary data.");
}


template <typename Number>
Number
read_binary_value_dealii(std::istream &input)
{
  static_assert(std::is_trivially_copyable_v<Number>,
                "Only trivially copyable types can be read directly.");

  Number value{};

  input.read(reinterpret_cast<char *>(&value),
             static_cast<std::streamsize>(sizeof(Number)));

  if (!input)
    throw std::runtime_error("Unexpected end of binary mesh file.");

  return value;
}


// =============================================================================
// Save vertices and CellData
// =============================================================================

template <int dim>
void
save_mesh_binary(const fs::path                           &filename,
                 const std::vector<dealii::Point<dim>>    &vertices,
                 const std::vector<dealii::CellData<dim>> &cells)
{
  static_assert(dim > 0, "The dimension must be positive.");

  std::ofstream output(filename, std::ios::binary | std::ios::trunc);

  if (!output)
    throw std::runtime_error("Could not create binary mesh file: " +
                             filename.string());

  /*
   * Eight-byte identifier.
   *
   * The last character denotes binary-format version 1.
   */
  constexpr std::array<char, 8> magic = {
    {'D', 'E', 'A', 'L', 'M', 'S', 'H', '1'}};

  constexpr std::uint32_t format_version = 1;

  const std::uint32_t stored_dimension = static_cast<std::uint32_t>(dim);

  const std::uint64_t number_of_vertices =
    static_cast<std::uint64_t>(vertices.size());

  const std::uint64_t number_of_cells =
    static_cast<std::uint64_t>(cells.size());

  // ---------------------------------------------------------------------------
  // Header
  // ---------------------------------------------------------------------------

  output.write(magic.data(), static_cast<std::streamsize>(magic.size()));

  if (!output)
    throw std::runtime_error("Could not write binary mesh header.");

  write_binary_value_dealii(output, format_version);
  write_binary_value_dealii(output, stored_dimension);
  write_binary_value_dealii(output, number_of_vertices);
  write_binary_value_dealii(output, number_of_cells);

  // ---------------------------------------------------------------------------
  // Vertices
  // ---------------------------------------------------------------------------

  for (const dealii::Point<dim> &vertex : vertices)
    for (unsigned int d = 0; d < dim; ++d)
      write_binary_value_dealii(output, vertex[d]);

  // ---------------------------------------------------------------------------
  // Cells
  // ---------------------------------------------------------------------------

  for (const dealii::CellData<dim> &cell : cells)
    {
      const std::uint32_t vertices_per_cell =
        static_cast<std::uint32_t>(cell.vertices.size());

      write_binary_value_dealii(output, vertices_per_cell);

      for (const unsigned int vertex_index : cell.vertices)
        {
          const std::uint64_t stored_vertex_index =
            static_cast<std::uint64_t>(vertex_index);

          if (stored_vertex_index >= number_of_vertices)
            throw std::runtime_error(
              "A cell references vertex " +
              std::to_string(stored_vertex_index) + ", but the mesh has only " +
              std::to_string(number_of_vertices) + " vertices.");

          write_binary_value_dealii(output, stored_vertex_index);
        }

      /*
       * Use fixed-width representations in the file instead of relying on
       * the sizes of deal.II's internal ID types.
       */
      const std::uint32_t material_id =
        static_cast<std::uint32_t>(cell.material_id);

      const std::uint32_t manifold_id =
        static_cast<std::uint32_t>(cell.manifold_id);

      write_binary_value_dealii(output, material_id);
      write_binary_value_dealii(output, manifold_id);
    }

  output.close();

  if (!output)
    throw std::runtime_error("Failed while closing binary mesh file: " +
                             filename.string());
}


// =============================================================================
// Load vertices and CellData
// =============================================================================

template <int dim>
BinaryMeshData<dim>
load_mesh_binary(const fs::path &filename)
{
  static_assert(dim > 0, "The dimension must be positive.");

  std::ifstream input(filename, std::ios::binary);

  if (!input)
    throw std::runtime_error("Could not open binary mesh file: " +
                             filename.string());

  constexpr std::array<char, 8> expected_magic = {
    {'D', 'E', 'A', 'L', 'M', 'S', 'H', '1'}};

  std::array<char, 8> actual_magic{};

  input.read(actual_magic.data(),
             static_cast<std::streamsize>(actual_magic.size()));

  if (!input)
    throw std::runtime_error("Could not read binary mesh header.");

  if (actual_magic != expected_magic)
    throw std::runtime_error(
      "File is not a supported deal.II binary mesh file.");

  const std::uint32_t format_version =
    read_binary_value_dealii<std::uint32_t>(input);

  const std::uint32_t stored_dimension =
    read_binary_value_dealii<std::uint32_t>(input);

  const std::uint64_t number_of_vertices =
    read_binary_value_dealii<std::uint64_t>(input);

  const std::uint64_t number_of_cells =
    read_binary_value_dealii<std::uint64_t>(input);

  if (format_version != 1)
    throw std::runtime_error("Unsupported binary mesh version: " +
                             std::to_string(format_version));

  if (stored_dimension != dim)
    throw std::runtime_error("Mesh dimension mismatch. File dimension is " +
                             std::to_string(stored_dimension) +
                             ", but the program requested dimension " +
                             std::to_string(dim) + ".");

  if (number_of_vertices >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::runtime_error("Vertex count is too large for this platform.");

  if (number_of_cells >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::runtime_error("Cell count is too large for this platform.");

  BinaryMeshData<dim> mesh;

  mesh.vertices.resize(static_cast<std::size_t>(number_of_vertices));

  mesh.cells.reserve(static_cast<std::size_t>(number_of_cells));

  // ---------------------------------------------------------------------------
  // Vertices
  // ---------------------------------------------------------------------------

  for (dealii::Point<dim> &vertex : mesh.vertices)
    for (unsigned int d = 0; d < dim; ++d)
      vertex[d] = read_binary_value_dealii<double>(input);

  // ---------------------------------------------------------------------------
  // Cells
  // ---------------------------------------------------------------------------

  for (std::uint64_t cell_index = 0; cell_index < number_of_cells; ++cell_index)
    {
      const std::uint32_t vertices_per_cell =
        read_binary_value_dealii<std::uint32_t>(input);

      if (vertices_per_cell == 0)
        throw std::runtime_error("Cell " + std::to_string(cell_index) +
                                 " has no vertices.");

      /*
       * Construct CellData with the correct number of vertices.
       *
       * This supports tetrahedra, pyramids, prisms, hexahedra, and other
       * supported reference-cell sizes.
       */
      dealii::CellData<dim> cell(vertices_per_cell);

      for (std::uint32_t local_vertex = 0; local_vertex < vertices_per_cell;
           ++local_vertex)
        {
          const std::uint64_t stored_vertex_index =
            read_binary_value_dealii<std::uint64_t>(input);

          if (stored_vertex_index >= number_of_vertices)
            throw std::runtime_error("Cell " + std::to_string(cell_index) +
                                     " contains invalid vertex index " +
                                     std::to_string(stored_vertex_index) + ".");

          if (stored_vertex_index > static_cast<std::uint64_t>(
                                      std::numeric_limits<unsigned int>::max()))
            throw std::runtime_error("Vertex index cannot be represented as "
                                     "unsigned int.");

          cell.vertices[local_vertex] =
            static_cast<unsigned int>(stored_vertex_index);
        }

      const std::uint32_t stored_material_id =
        read_binary_value_dealii<std::uint32_t>(input);

      const std::uint32_t stored_manifold_id =
        read_binary_value_dealii<std::uint32_t>(input);

      if (stored_material_id >
          static_cast<std::uint32_t>(
            std::numeric_limits<dealii::types::material_id>::max()))
        throw std::runtime_error(
          "Material ID cannot be represented by deal.II.");

      if (stored_manifold_id >
          static_cast<std::uint32_t>(
            std::numeric_limits<dealii::types::manifold_id>::max()))
        throw std::runtime_error(
          "Manifold ID cannot be represented by deal.II.");

      cell.material_id =
        static_cast<dealii::types::material_id>(stored_material_id);

      cell.manifold_id =
        static_cast<dealii::types::manifold_id>(stored_manifold_id);

      mesh.cells.push_back(std::move(cell));
    }

  /*
   * Optionally reject trailing bytes. This catches situations where the
   * file format and reader do not agree.
   */
  char trailing_byte = '\0';

  if (input.read(&trailing_byte, 1))
    throw std::runtime_error(
      "Binary mesh file contains unexpected trailing data.");

  if (!input.eof())
    throw std::runtime_error(
      "An error occurred while finishing the binary read.");

  return mesh;
}


struct Mesh
{
  std::vector<std::array<double, 3>>      vertices;
  std::vector<std::vector<std::uint64_t>> cells;
};


template <typename T>
void
read_binary_value(std::ifstream &input, T &value)
{
  static_assert(std::is_trivially_copyable_v<T>,
                "T must be trivially copyable.");

  input.read(reinterpret_cast<char *>(&value), sizeof(T));

  if (!input)
    {
      throw std::runtime_error(
        "Unexpected end of file while reading binary data.");
    }
}


Mesh
read_mesh_binary(const std::string &filename)
{
  std::ifstream input(filename, std::ios::binary);

  if (!input)
    {
      throw std::runtime_error("Could not open binary mesh file: " + filename);
    }

  /*
   * Binary layout:
   *
   * char[8]  magic = "NEKMESH1"
   * uint32_t version
   * uint64_t number_of_vertices
   * uint64_t number_of_cells
   *
   * For every vertex:
   *     double x
   *     double y
   *     double z
   *
   * For every cell:
   *     uint64_t number_of_vertices_in_cell
   *     uint64_t vertex_indices[number_of_vertices_in_cell]
   */

  constexpr std::array<char, 8> expected_magic = {
    {'N', 'E', 'K', 'M', 'E', 'S', 'H', '1'}};

  std::array<char, 8> actual_magic{};

  input.read(actual_magic.data(),
             static_cast<std::streamsize>(actual_magic.size()));

  if (!input)
    {
      throw std::runtime_error("Could not read the binary mesh header.");
    }

  if (actual_magic != expected_magic)
    {
      throw std::runtime_error("Invalid mesh file: magic number does not match "
                               "\"NEKMESH1\".");
    }

  std::uint32_t version            = 0;
  std::uint64_t number_of_vertices = 0;
  std::uint64_t number_of_cells    = 0;

  read_binary_value(input, version);
  read_binary_value(input, number_of_vertices);
  read_binary_value(input, number_of_cells);

  if (version != 1)
    {
      throw std::runtime_error("Unsupported mesh binary version: " +
                               std::to_string(version));
    }

  if (number_of_vertices >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
      throw std::runtime_error(
        "The number of vertices is too large for this system.");
    }

  if (number_of_cells >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
      throw std::runtime_error(
        "The number of cells is too large for this system.");
    }

  Mesh mesh;

  mesh.vertices.resize(static_cast<std::size_t>(number_of_vertices));

  mesh.cells.resize(static_cast<std::size_t>(number_of_cells));

  /*
   * Read vertices.
   */
  for (std::array<double, 3> &vertex : mesh.vertices)
    {
      read_binary_value(input, vertex[0]);
      read_binary_value(input, vertex[1]);
      read_binary_value(input, vertex[2]);
    }

  /*
   * Read cells.
   */
  for (std::size_t cell_index = 0; cell_index < mesh.cells.size(); ++cell_index)
    {
      std::uint64_t number_of_cell_vertices = 0;

      read_binary_value(input, number_of_cell_vertices);

      /*
       * This upper bound is appropriate for the format generated by
       * the previous writer, where a cell stores unique mesh-vertex
       * indices.
       */
      if (number_of_cell_vertices > number_of_vertices)
        {
          throw std::runtime_error(
            "Cell " + std::to_string(cell_index) +
            " claims to contain more vertices than the entire mesh.");
        }

      if (number_of_cell_vertices >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
          throw std::runtime_error("Cell " + std::to_string(cell_index) +
                                   " is too large for this system.");
        }

      std::vector<std::uint64_t> &cell = mesh.cells[cell_index];

      cell.resize(static_cast<std::size_t>(number_of_cell_vertices));

      for (std::uint64_t &vertex_index : cell)
        {
          read_binary_value(input, vertex_index);

          if (vertex_index >= number_of_vertices)
            {
              throw std::runtime_error("Cell " + std::to_string(cell_index) +
                                       " references invalid vertex index " +
                                       std::to_string(vertex_index) + ".");
            }
        }
    }

  /*
   * Check for unexpected data after the mesh. This catches a mismatch
   * between the writer and reader formats.
   */
  char extra_byte = 0;

  if (input.read(&extra_byte, 1))
    {
      throw std::runtime_error(
        "The binary file contains unexpected trailing data.");
    }

  if (!input.eof())
    {
      throw std::runtime_error("An I/O error occurred after reading the mesh.");
    }

  return mesh;
}

template <int dim>
void
create_mesh_from_file(dealii::Triangulation<dim> &tria,
                      const std::string          &filename,
                      const bool                  verbose)
{
  BinaryMeshData<dim> mesh_loaded = load_mesh_binary<dim>(filename);

  if (verbose)
    std::cout << "Create triangulation" << std::endl;

  tria.create_triangulation(mesh_loaded.vertices,
                            mesh_loaded.cells,
                            dealii::SubCellData());

  if (verbose)
    std::cout << "Created triangulation" << std::endl;
}