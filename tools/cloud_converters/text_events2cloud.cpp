#include <iostream>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/parse.h>
#include <pcl/filters/voxel_grid.h>


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <fstream>

// --------------
// -----Help-----
// --------------
void
printUsage (const char* progName)
{
  std::cout << "\n\nUsage: "<<progName<<" [options]\n\n"
            << "Options:\n"
            << "-------------------------------------------\n"
            << "-h           this help\n"
            << "-s           Simple visualisation example\n"
            << "-r           RGB colour visualisation example\n"
            << "-c           Custom colour visualisation example\n"
            << "-n           Normals visualisation example\n"
            << "-a           Shapes visualisation example\n"
            << "-v           Viewports example\n"
            << "-i           Interaction Customization example\n"
            << "\n\n";
}


// --------------
// -----Main-----
// --------------
int main (int argc, char** argv)
{
  // --------------------------------------
  // -----Parse Command Line Arguments-----
  // --------------------------------------
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    printUsage (argv[0]);
    return 0;
  }
  bool simple(false), ransac(false), shrink(false);
  simple = true;
  std::cout << "Simple visualisation example\n";
  
  if (pcl::console::find_argument (argc, argv, "-d") >= 0)
  {
    shrink = true;
    std::cout << "Downsampling\n";
  }

  std::string in_file = "";
  pcl::console::parse_argument (argc, argv, "-in", in_file);
  std::cout << "Reading " << in_file << "\n";

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  std::string name = in_file.substr(0, in_file.find_last_of("."));
  std::string ext  = in_file.substr(in_file.find_last_of(".") + 1);
  Eigen::Vector3d center_of_mass(0, 0, 0);
  if (ext == "txt") {
    std::cout << "Reading as .txt...\n";
    std::ifstream f(in_file, std::ifstream::in);
    double t = 0, x = 0, y = 0, pol = 0;
    while (!f.eof()) {
      if (!(f >> t >> x >> y >> pol))                     
        continue;
      pcl::PointXYZRGB p;
      p.x = x / 1000.0; p.y = y / 1000.0; p.z = t / 1.0;
      p.r = 255; p.g = 0; p.b = 0;
      if (pol > 0.5) {
        p.r = 0;
        p.b = 255;
      }

      point_cloud_ptr->push_back(p);
      center_of_mass += Eigen::Vector3d(p.x, p.y, p.z);
    }
  } else if (ext == "pcd") {
    std::cout << "Reading as .pcd...\n";
    pcl::io::loadPCDFile(in_file, *point_cloud_ptr);  
  } else if (ext == "ply") {
    std::cout << "Reading as .ply...\n";
    pcl::io::loadPLYFile(in_file, *point_cloud_ptr);
  } else {
    std::cout << "Unsupported file format: " << ext << "\n";
    return -1;
  }
  if (ext != "ply") {
  center_of_mass /= point_cloud_ptr->size();
    for (auto &p : *point_cloud_ptr) {
      p.x -= center_of_mass.x();
      p.y -= center_of_mass.y();
      p.z -= center_of_mass.z();
    }
  }
  std::cout << "Read " << point_cloud_ptr->size() << " points\n";
  pcl::io::savePLYFileASCII (name + ".ply", *point_cloud_ptr);
  //pcl::io::savePCDFileASCII (name + "_processed.pcd", *point_cloud_ptr);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_filtered_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  sor.setInputCloud (point_cloud_ptr);
  sor.setLeafSize (0.001f, 0.001f, 0.001f);
  sor.filter (*point_cloud_filtered_ptr);

  std::cout << "Downsampled cloud: " << point_cloud_filtered_ptr->size() << " points\n";

  //pcl::io::savePCDFileASCII (name + "_small_processed.pcd", *point_cloud_filtered_ptr);
  
  if (shrink)
    pcl::io::savePLYFileASCII (name + "_small.ply", *point_cloud_filtered_ptr);
}
