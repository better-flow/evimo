/* \author Geoffrey Biggs */


#include <iostream>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

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




unsigned int text_id = 0;
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getKeySym () == "r" && event.keyDown ())
  {
    std::cout << "r was pressed => removing all text" << std::endl;

    char str[512];
    for (unsigned int i = 0; i < text_id; ++i)
    {
      sprintf (str, "text#%03d", i);
      viewer->removeShape (str);
    }
    text_id = 0;
  }
}

void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
                         void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
      event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)
  {
    std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;

    char str[512];
    sprintf (str, "text#%03d", text_id ++);
    viewer->addText ("clicked here", event.getX (), event.getY (), str);
  }
}

// RANSAC
void find_spheres (pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, bool mm = false) {
  std::cout << "Running RANSAC...\n";

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_SPHERE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100000000);

  float scale = 1.0;
  if (mm) scale = 1000;

  seg.setDistanceThreshold (0.00005 * scale);
  seg.setRadiusLimits (0.003 * scale, 0.014 * scale);
  seg.setInputCloud (cloud);

  seg.segment (*inliers, *coefficients);
  /*
  pcl::SampleConsensusModelSphere<pcl::PointXYZRGB>::Ptr
      model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZRGB> (cloud));

  pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model_s);
  ransac.setDistanceThreshold (.001);
  ransac.computeModel();
  ransac.getInliers(inliers);
    */

  for (auto &index : inliers->indices) {
    pcl::PointXYZRGB &p = cloud->at(index);
    p.r = 255;
    p.g = 0;
    p.b = 255;
  }

  std::cout << "Coefficients: (x, y, z):" << coefficients->values[0] << " "
                                          << coefficients->values[1] << " "
                                          << coefficients->values[2] << "\n";

  std::cout << "Inliers:" << inliers->indices.size() << "\n";
  std::cout << "Radius:" << coefficients->values[3] << "\n";
  std::cout << "Done.\n";
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

  if (pcl::console::find_argument (argc, argv, "-r") >= 0)
  {
    ransac = true;
    std::cout << "RANSAC enabled\n";
  }
  else if (pcl::console::find_argument (argc, argv, "-d") >= 0)
  {
    shrink = true;
    std::cout << "Downsampling\n";
  }

  bool in_mm = false;
  if (pcl::console::find_argument (argc, argv, "-mm") >= 0)
  {
    in_mm = true;
    std::cout << "In millimeters\n";
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
    double x = 0, y = 0, z = 0, r = 0, g = 0, b = 0;
    while (!f.eof()) {
      if (!(f >> x >> y >> z >> r >> g >> b))
        continue;
      pcl::PointXYZRGB p;
      p.x = x / 1000.0; p.y = y / 1000.0; p.z = z / 1000.0;
      p.r = 255; p.g = 255; p.b = 255;
      point_cloud_ptr->push_back(p);
      center_of_mass += Eigen::Vector3d(x, y, z);
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
    center_of_mass /= 1000.0 * point_cloud_ptr->size();
    for (auto &p : *point_cloud_ptr) {
      p.x -= center_of_mass.x();
      p.y -= center_of_mass.y();
      p.z -= center_of_mass.z();
    }
  }
  std::cout << "Read " << point_cloud_ptr->size() << " points\n";
  //pcl::io::savePLYFileASCII (name + ".ply", *point_cloud_ptr);
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

  // Algorithmic part
  if (ransac)
    find_spheres(point_cloud_ptr, in_mm);
}
