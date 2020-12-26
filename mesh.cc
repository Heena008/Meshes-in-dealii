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

template <int dim>
void cheese()
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

template <int dim>
void create_coarse_grid()
{
	 Triangulation<2> triangulation;
		 Triangulation<3> out;
		 unsigned int s =1000;

		  const std::vector<Point<2>>  vertices  = {
		    {548.0*s, 5934.0*s}, {548.0*s, 5936.0*s}, {548.0*s,5938.0*s}, {548.0*s, 5940.0*s},
			{548.0*s, 5942.0*s},
			//4
			{550.0*s, 5926.0*s},{550.0*s, 5928.0*s}, {550.0*s,5930.0*s}, {550.0*s,5932.0*s}, {550.0*s,5934.0*s},
			{550.0*s, 5936.0*s},{550.0*s,5938.0*s}, {550.0*s,5940.0*s}, {550.0*s, 5942.0*s},{550.0*s, 5944.0*s},
			//14
			{552.0*s, 5922.0*s}, {552.0*s, 5924.0*s},{552.0*s, 5926.0*s}, {552.0*s, 5928.0*s},
			{552.0*s, 5930.0*s},{552.0*s,5932.0*s},{552.0*s, 5934.0*s}, {552.0*s,5936.0*s},
			{552.0*s, 5938.0*s}, {552.0*s, 5940.0*s},{552.0*s, 5942.0*s},{552.0*s, 5944.0*s},
	//		26
			{554.0*s, 5922.0*s},{554.0*s, 5924.0*s}, {554.0*s, 5926.0*s}, {554.0*s, 5928.0*s},
			{554.0*s, 5930.0*s},{554.0*s,5932.0*s},{554.0*s, 5934.0*s},
			{554.0*s,5936.0*s}, {554.0*s, 5938.0*s},{554.0*s, 5940.0*s},{554.0*s, 5942.0*s},
	//		//37
			{556.0*s, 5920.0*s},{556.0*s, 5922.0*s},{556.0*s, 5924.0*s}, {556.0*s, 5926.0*s},
			{556.0*s, 5928.0*s},{556.0*s, 5930.0*s},{556.0*s,5932.0*s},{556.0*s, 5934.0*s},
			{556.0*s,5936.0*s}, {556.0*s, 5938.0*s},{556.0*s,5940.0*s},{556.0*s,5942.0*s},
	//		//49
			{558.0*s, 5920.0*s},{558.0*s, 5922.0*s},{558.0*s, 5924.0*s}, {558.0*s, 5926.0*s},
			{558.0*s, 5928.0*s},{558.0*s, 5930.0*s},{558.0*s,5932.0*s},{558.0*s, 5934.0*s},
			{558.0*s,5936.0*s}, {558.0*s, 5938.0*s},{558.0*s,5940.0*s},{558.0*s,5942.0*s},
			{558.0*s,5944.0*s},{558.0*s,5946.0*s},
	//	  //63
			{560.0*s,5918.0*s},
			{560.0*s, 5920.0*s},{560.0*s, 5922.0*s},{560.0*s, 5924.0*s}, {560.0*s, 5926.0*s},
			{560.0*s, 5928.0*s},{560.0*s, 5930.0*s},{560.0*s,5932.0*s},{560.0*s, 5934.0*s},
			{560.0*s,5936.0*s},{560.0*s, 5938.0*s},{560.0*s,5940.0*s},{560.0*s,5942.0*s},
			{560.0*s,5944.0*s},{560.0*s,5946.0*s},
	//	    //78
			{562.0*s,5918.0*s},{562.0*s, 5920.0*s},{562.0*s, 5922.0*s},{562.0*s, 5924.0*s},
		    {562.0*s, 5926.0*s},{562.0*s, 5928.0*s},{562.0*s, 5930.0*s},{562.0*s,5932.0*s},
			{562.0*s, 5934.0*s}, {562.0*s,5936.0*s},{562.0*s, 5938.0*s},{562.0*s,5940.0*s},
			{562.0*s,5942.0*s},{562.0*s,5944.0*s},{562.0*s,5946.0*s},
		  //93
			{564.0*s,5918.0*s},{564.0*s, 5920.0*s},{564.0*s, 5922.0*s},{564.0*s, 5924.0*s},
			{564.0*s, 5926.0*s},
			{564.0*s, 5928.0*s},{564.0*s, 5930.0*s},{564.0*s,5932.0*s},{564.0*s, 5934.0*s},
			{564.0*s,5936.0*s},
			{564.0*s, 5938.0*s},{564.0*s,5940.0*s},{564.0*s,5942.0*s},{564.0*s,5944.0*s},
			{564.0*s,5946.0*s},{564.0*s,5948.0*s},
			{564.0*s,5950.0*s},
		  //110
			{566.0*s,5918.0*s},{566.0*s, 5920.0*s},{566.0*s, 5922.0*s},{566.0*s, 5924.0*s},
			{566.0*s, 5926.0*s},
			{566.0*s, 5928.0*s},{566.0*s, 5930.0*s},{566.0*s,5932.0*s},{566.0*s, 5934.0*s},
			{566.0*s,5936.0*s},{566.0*s, 5938.0*s},{566.0*s,5940.0*s},{566.0*s,5942.0*s},
			{566.0*s,5944.0*s},{566.0*s,5946.0*s},{566.0*s,5948.0*s},{566.0*s,5950.0*s},
		  //127
			{568.0*s,5918.0*s},{568.0*s, 5920.0*s},{568.0*s, 5922.0*s},{568.0*s, 5924.0*s},
			{568.0*s, 5926.0*s},{568.0*s, 5928.0*s},{568.0*s, 5930.0*s},{568.0*s,5932.0*s},
			{568.0*s, 5934.0*s}, {568.0*s,5936.0*s},{568.0*s, 5938.0*s},{568.0*s,5940.0*s},
			{568.0*s,5942.0*s},{568.0*s,5944.0*s},{568.0*s,5946.0*s},
			{568.0*s,5948.0*s},{568.0*s,5950.0*s},
		  //144
			{570.0*s, 5920.0*s},{570.0*s, 5922.0*s},{570.0*s, 5924.0*s}, {570.0*s, 5926.0*s},{570.0*s, 5928.0*s},
			{570.0*s, 5930.0*s},{570.0*s,5932.0*s},{570.0*s, 5934.0*s}, {570.0*s,5936.0*s},
			{570.0*s, 5938.0*s},{570.0*s,5940.0*s},{570.0*s,5942.0*s},{570.0*s,5944.0*s},
			{570.0*s,5946.0*s},{570.0*s,5948.0*s},{570.0*s,5950.0*s},{570.0*s,5952.0*s},
			{570.0*s,5954.0*s},
	//	 // 162
			{572.0*s, 5920.0*s},{572.0*s, 5922.0*s},{572.0*s, 5924.0*s}, {572.0*s, 5926.0*s},
			{572.0*s, 5928.0*s},{572.0*s, 5930.0*s},{572.0*s,5932.0*s},{572.0*s, 5934.0*s},
			{572.0*s,5936.0*s},{572.0*s, 5938.0*s},{572.0*s,5940.0*s},{572.0*s,5942.0*s},
			{572.0*s,5944.0*s},{572.0*s,5946.0*s},{572.0*s,5948.0*s},
			{572.0*s,5950.0*s},{572.0*s,5952.0*s},{572.0*s,5954.0*s},
	//	  //180
			{574.0*s,5918.0*s},{574.0*s, 5920.0*s},{574.0*s, 5922.0*s},{574.0*s, 5924.0*s},
			{574.0*s, 5926.0*s},{574.0*s, 5928.0*s},{574.0*s, 5930.0*s},{574.0*s,5932.0*s},
			{574.0*s, 5934.0*s},{574.0*s,5936.0*s},{574.0*s, 5938.0*s},{574.0*s,5940.0*s},
			{574.0*s,5942.0*s},{574.0*s,5944.0*s},{574.0*s,5946.0*s},{574.0*s,5948.0*s},
			{574.0*s,5950.0*s},{574.0*s,5952.0*s},{574.0*s,5954.0*s},{574.0*s,5956.0*s},
	////		//200
			{576.0*s,5916.0*s},{576.0*s,5918.0*s},{576.0*s, 5920.0*s},{576.0*s, 5922.0*s},
			{576.0*s, 5924.0*s},{576.0*s, 5926.0*s},{576.0*s, 5928.0*s},{576.0*s, 5930.0*s},
			{576.0*s,5932.0*s},{576.0*s, 5934.0*s},{576.0*s,5936.0*s}, {576.0*s, 5938.0*s},
			{576.0*s,5940.0*s},{576.0*s,5942.0*s},{576.0*s,5944.0*s},{576.0*s,5946.0*s},
			{576.0*s,5948.0*s},{576.0*s,5950.0*s},{576.0*s,5952.0*s},{576.0*s,5954.0*s},
			{576.0*s,5956.0*s},
	//		//221
			{578.0*s, 5916.0*s},{578.0*s, 5918.0*s},{578.0*s, 5920.0*s},{578.0*s, 5922.0*s},
			{578.0*s, 5924.0*s},{578.0*s, 5926.0*s},{578.0*s, 5928.0*s},{578.0*s, 5930.0*s},
			{578.0*s, 5932.0*s},{578.0*s, 5934.0*s},{578.0*s,5936.0*s}, {578.0*s, 5938.0*s},
		    {578.0*s,5940.0*s},{578.0*s,5942.0*s},{578.0*s,5944.0*s},{578.0*s,5946.0*s},
			{578.0*s,5948.0*s},{578.0*s,5950.0*s},
			{578.0*s,5952.0*s},{578.0*s,5954.0*s},{578.0*s,5956.0*s},
		  //242
			{580.0*s, 5916.0*s},{580.0*s, 5918.0*s},{580.0*s, 5920.0*s},{580.0*s, 5922.0*s},
			{580.0*s, 5924.0*s},{580.0*s, 5926.0*s},{580.0*s, 5928.0*s},{580.0*s, 5930.0*s},
			{580.0*s, 5932.0*s},
			{580.0*s, 5936.0*s},{580.0*s, 5938.0*s},{580.0*s, 5940.0*s},{580.0*s,5942.0*s},
			{580.0*s,5944.0*s},{580.0*s, 5946.0*s},{580.0*s, 5948.0*s},{580.0*s, 5950.0*s},{580.0*s, 5952.0*s},
			{580.0*s, 5954.0*s},{580.0*s, 5956.0*s},
	//		//262
			{582.0*s, 5916.0*s},{582.0*s, 5918.0*s},{582.0*s, 5920.0*s},{582.0*s, 5922.0*s},
			{582.0*s, 5924.0*s},{582.0*s, 5926.0*s},{582.0*s, 5928.0*s},{582.0*s, 5930.0*s},{582.0*s, 5932.0*s},
			{582.0*s, 5942.0*s},{582.0*s, 5944.0*s},
	//	  //273
			{584.0*s, 5916.0*s},{584.0*s, 5918.0*s},{584.0*s, 5920.0*s},{584.0*s, 5922.0*s},{584.0*s, 5924.0*s},
			{584.0*s, 5926.0*s},{584.0*s, 5928.0*s},{584.0*s, 5930.0*s},
			//281
		  {586.0*s, 5918.0*s},{586.0*s, 5920.0*s},{586.0*s, 5922.0*s},{586.0*s, 5924.0*s},
		  {586.0*s, 5926.0*s},
			//286
		  {588.0*s, 5920.0*s},{588.0*s, 5922.0*s},{588.0*s, 5924.0*s},
		  //289
		    {590.0*s, 5922.0*s},{590.0*s, 5924.0*s}};
		 //291




	const std::vector<std::array<int, GeometryInfo<dim>::vertices_per_cell>>
		    cell_vertices = {{0, 9, 1, 10},{1, 10, 2, 11},{2, 11, 3, 12},{3,12,4,13},
					//3
					{5, 17, 6, 18},{6, 18, 7, 19},{7, 19, 8, 20},{8,20,9,21},
					{9, 21, 10, 22},{10, 22, 11, 23},{11, 23, 12, 24},{12,24,13,25},
					{13,25,14,26},
					//12
					{15, 27, 16, 28},{16, 28, 17, 29},{17,29, 18, 30},{18, 30, 19, 31},
					{19,31,20,32},{20,32,21,33},{21,33,22,34},{22,34,23,35},
					{23,35,24,36},{24,36,25,37},
					//22
					{27,39,28,40},{28, 40, 29,41},{29, 41,30, 42},{30, 42, 31, 43},
					{31,43,32,44},{32,44,33,45},{33,45,34,46},{34,46,35,47},
					{35,47,36,48},
					//31
					{38,50, 39,51},{39,51,40, 52},{40,52, 41, 53},{41,53, 42, 54},
					{42,54, 43, 55},{43,55,44,56},{44,56,45,57},{45,57,46,58},
					{46,58,47,59},{47,59,48,60},{48,60,49,61},
					//42
					{50,65,51,66},{51,66,52,67},{52, 67, 53,68},{53, 68, 54,69},
					{54, 69, 55,70},{55, 70, 56, 71},{56,71,57,72},{57,72,58,73},
					{58,73,59,74},{59,74,60,75},{60,75,61,76},{61,76,62,77},
					{62,77,63,78},
					//55
					{64,79,65,80},{65,80,66,81},{66, 81, 67,82},{67, 82, 68,83},
					{68, 83, 69,84},{69, 84, 70, 85},{70,85,71,86},{71,86,72,87},
					{72,87,73,88},{73,88,74,89},{74,89,75,90},{75,90,76,91},
					{76,91,77,92},{77,92,78,93},
					//69
					{79,94,80,95},{80,95,81,96},{81, 96,82,97},{82,97, 83,98},
					{83, 98,84,99},{84, 99, 85,100},{85,100,86,101},{86,101,87,102},
					{87,102,88,103},{88,103,89,104},{89,104,90,105},{90,105,91,106},
					{91,106,92,107},{92,107,93,108},
					//83
					{94,111,95,112},{95,112,96,113},{96, 113,97,114},{97, 114,98,115},
					{98, 115,99,116},{99, 116, 100, 117},{100,117,101,118},{101,118,102,119},
					{102,119,103,120},{103,120,104,121},{104,121,105,122},{105,122,106,123},
					{106,123,107,124},{107,124,108,125},{108,125,109,126},{109,126,110,127},
					//99
					{111,128,112,129},{112,129,113,130},{113,130,114,131},{114,131,115,132},
					{115,132,116,133},{116,133,117,134},{117,134,118,135},{118,135,119,136},
					{119,136,120,137},{120,137,121,138},{121,138,122,139},{122,139,123,140},
					{123,140,124,141},{124,141,125,142},{125,142,126,143},{126,143,127,144},
					//115
					{129,145,130,146},{130,146,131,147},{131,147,132,148},{132,148,133,149},
					{133,149,134,150},{134,150,135,151},{135,151,136,152},{136,152,137,153},
					{137,153,138,154},{138,154,139,155},{139,155,140,156},{140,156,141,157},
					{141,157,142,158},{142,158,143,159},{143,159,144,160},
					//130
					{145,163,146,164},{146,164,147,165},{147,165,148,166},{148,166,149,167},
					{149,167,150,168},{150,168,151,169},{151,169,152,170},{152,170,153,171},
					{153,171,154,172},{154,172,155,173},{155,173,156,174},{156,174,157,175},
					{157,175,158,176},{158,176,159,177},{159,177,160,178},{160,178,161,179},
					{161,179,162,180},
					//147
					{163,182,164,183},{164,183,165,184},{165,184,166,185},{166,185,167,186},
					{167,186,168,187},{168,187,169,188},{169,188,170,189},{170,189,171,190},
					{171,190,172,191},{172,191,173,192},{173,192,174,193},{174,193,175,194},
					{175,194,176,195},{176,195,177,196},{177,196,178,197},{178,197,179,198},
					{179,198,180,199},
					//164
					{181,202,182,203},{182,203,183,204},{183,204,184,205},{184,205,185,206},
					{185,206,186,207},{186,207,187,208},{187,208,188,209},{188,209,189,210},
					{189,210,190,211},{190,211,191,212},{191,212,192,213},{192,213,193,214},
					{193,214,194,215},{194,215,195,216},{195,216,196,217},{196,217,197,218},
					{197,218,198,219},{198,219,199,220},{199,220,200,221},
					//183
					{201,222,202,223},{202,223,203,224},{203,224,204,225},{204,225,205,226},
					{205,226,206,227},{206,227,207,228},{207,228,208,229},{208,229,209,230},
					{209,230,210,231},{210,231,211,232},{211,232,212,233},{212,233,213,234},
					{213,234,214,235},{214,235,215,236},{215,236,216,237},{216,237,217,238},
					{217,238,218,239},{218,239,219,240},{219,240,220,241},{220,241,221,242},
					//203
					{222,243,223,244},{223,244,224,245},{224,245,225,246},{225,246,226,247},
					{226,247,227,248},{227,248,228,249},{228,249,229,250},{229,250,230,251},
					{232,252,233,253},{233,253,234,254},{234,254,235,255},{235,255,236,256},
					{236,256,237,257},{237,257,238,258},{239,259,240,260},{240,260,241,261},
					{241,261,242,262},
					//220
					{243,263,244,264},{244,264,245,265},{245,265,246,266},{246,266,247,267},
					{247,267,248,268},{248,268,249,269},{249,269,250,270},{250,270,251,271},
					{255,272,256,273},
					//229
					{263,274,264,275},{264,275,265,276},{265,276,266,277},{266,277,267,278},
					{267,278,268,279},{268,279,269,280},{269,280,270,281},
					//236
					{275,282,276,283},{276,283,277,284},{277,284,278,285},{278,285,279,286},
					//240
					{283,287,284,288},{284,288,285,289},
					//242
					{288,290,289,291},
					//243
	               };
		  const unsigned int n_cells = cell_vertices.size();
		  std::vector<CellData<2>> cells(n_cells, CellData<2>());
		  for (unsigned int i = 0; i < n_cells; ++i)
		    {
		      for (unsigned int j = 0; j < GeometryInfo<2>::vertices_per_cell; ++j)
		        cells[i].vertices[j] = cell_vertices[i][j];
		      cells[i].material_id = 0;
		    }
		  triangulation.create_triangulation(vertices, cells, SubCellData());
		 GridGenerator::extrude_triangulation(triangulation, 3, 1500, out);

		  print_mesh_info(triangulation, "Hamburg_2D.vtk");
		  print_mesh_info(out, "Hamburg_3D.vtk");


}
int main()
{
//
//  cube_hole<3>();
//  subdivided_rect<3>();
//  merge_cube_rect<3>();
//  shift_cube<3>();
//  cheese<3>();
  create_coarse_grid<3>();

  return 0;
}
