/*
 * mesh.cc
 *
 *  Created on: Nov 2, 2020
 *      Author: heena
 */

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/grid/manifold_lib.h>

#include <fstream>
#include <iostream>
#include <map>

using namespace dealii;

/*!{Generating output for a given mesh}
 *  The following function generates some output for any of the meshes we will
 be generating in the remainder of this program. In particular, it generates the
 following information:
 *   - Some general information about the number of space dimensions in which
   this mesh lives and its number of cells.
 *  - Some general information about the number of space dimensions in which
   this mesh lives and its number of cells.
 *   - The number of boundary faces that use each boundary indicator, so that
   it can be compared with what we expect.
 *   Finally, the function outputs the mesh in VTU format that can easily be
 visualized in Paraview or VisIt.
 */

template <int dim>
void print_mesh_info(const Triangulation<dim> &triangulation,
                     const std::string &filename)
{
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << triangulation.n_active_cells() << std::endl;

  std::ofstream out(filename);
  GridOut grid_out;
  grid_out.write_vtk(triangulation, out);
  std::cout << " written to " << filename << std::endl << std::endl;
}

template <int dim> void cube_hole()
{

  Triangulation<2> triangulation;
  Triangulation<3> out; // if this line is commented it take default values
  GridGenerator::hyper_cube_with_cylindrical_hole(
      triangulation, 0.25,
      1.0); // 0.25 and 1 are inner and outer radius of cylinder respectively
  GridGenerator::extrude_triangulation(
      triangulation, 3, 2.0, out); /* is number of slices(minimum 2) and 2 is
height to extrude if this line is commented it will take default values*/

  triangulation.refine_global(4);
  out.refine_global(4);
  print_mesh_info(triangulation, "cube_hole_2D.vtk");
  print_mesh_info(out, "cube_hole_3D.vtk");
}

struct Grid6Func {
  double trans(const double y) const { return std::tanh(2 * y) / tanh(2); }
  Point<2> operator()(const Point<2> &in) const {
    return {in(0), trans(in(1))};
  }
};

template <int dim> void subdivided_rect()
{

  Triangulation<2> triangulation;
  Triangulation<3> out; // if this line is commented it take default values
  std::vector<unsigned int> repetitions(2); // number of division
  repetitions[0] = 3;                       // 40;// division in x direction
  repetitions[1] = 2;                       // 40;// division in y direction
  GridGenerator::subdivided_hyper_rectangle(
      triangulation, repetitions,
      Point<2>(1.0, -1.0), // two diagonally opposite corner
      Point<2>(4.0, 1.0));
  GridGenerator::extrude_triangulation(triangulation, 3, 2.0, out);

  GridTools::transform(Grid6Func(), triangulation);

  triangulation.refine_global(4);
  out.refine_global(4);
  print_mesh_info(triangulation, "subdivided_rect_2D.vtk");
  print_mesh_info(out, "subdivided_rect_3D.vtk");
}

template <int dim> void merge_cube_rect()
{

  Triangulation<2> tria1;
  GridGenerator::hyper_cube_with_cylindrical_hole(tria1, 0.25, 1.0);
  Triangulation<2> tria2;
  std::vector<unsigned int> repetitions(2);
  repetitions[0] = 3;
  repetitions[1] = 2;
  GridGenerator::subdivided_hyper_rectangle(
      tria2, repetitions, Point<2>(1.0, -1.0), Point<2>(4.0, 1.0));
  Triangulation<2> triangulation;
  Triangulation<3> out;
  GridGenerator::merge_triangulations(tria1, tria2, triangulation);
  GridGenerator::extrude_triangulation(triangulation, 3, 2.0, out);
  triangulation.refine_global(4);
  out.refine_global(4);
  print_mesh_info(triangulation, "merge_cube_rect_2D.vtk");
  print_mesh_info(out, "merge_cube_rect_3D.vtk");
}

template <int dim> void shift_cube()
{

  Triangulation<2> triangulation;
  Triangulation<3> out;
  GridGenerator::hyper_cube_with_cylindrical_hole(triangulation, 0.25, 1.0);
  for (const auto &cell : triangulation.active_cell_iterators()) {
    for (unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; ++i) {
      Point<2> &v = cell->vertex(i);
      if (std::abs(v(1) - 1.0) < 1e-5)
        v(1) += 0.5;
    }
  }
  GridGenerator::extrude_triangulation(triangulation, 3, 2.0, out);
  triangulation.refine_global(2);
  out.refine_global(2);
  print_mesh_info(triangulation, "shift_cube_2D.vtk");
  print_mesh_info(out, "shift_cube_3D.vtk");
}

template <int dim> void cheese()
{

  Triangulation<2> triangulation;
  Triangulation<3> out;
  std::vector<unsigned int> repetitions(2); // define holes needed
  repetitions[0] = 3;                       // number of holes in x direction
  repetitions[1] = 2;                       // number of holes in y direction

  GridGenerator::cheese(triangulation, repetitions);
  GridGenerator::extrude_triangulation(triangulation, 3, 2.0, out);
  triangulation.refine_global(4);
  out.refine_global(4);

  print_mesh_info(triangulation, "cheese_2D.vtk");
  print_mesh_info(out, "cheese_3D.vtk");
}

int main()
{

  cube_hole<3>();
  subdivided_rect<3>();
  merge_cube_rect<3>();
  shift_cube<3>();
  cheese<3>();

  return 0;
}
