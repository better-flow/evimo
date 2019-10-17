#include <dataset.h>

std::shared_ptr<StaticObject> Dataset::background;
std::map<int, std::shared_ptr<ViObject>> Dataset::clouds;
std::vector<Event> Dataset::event_array;
std::vector<cv::Mat> Dataset::images;
std::vector<ros::Time> Dataset::image_ts;
Trajectory Dataset::cam_tj;
std::map<int, Trajectory> Dataset::obj_tjs;

float Dataset::fx, Dataset::fy, Dataset::cx, Dataset::cy;
unsigned int Dataset::res_x, Dataset::res_y;
float Dataset::k1 = 0, Dataset::k2 = 0, Dataset::k3 = 0, Dataset::k4 = 0;
float Dataset::p1 = 0, Dataset::p2 = 0;
float Dataset::rr0, Dataset::rp0, Dataset::ry0;
float Dataset::tx0, Dataset::ty0, Dataset::tz0;
tf::Transform Dataset::bg_E;
tf::Transform Dataset::cam_E;
float Dataset::slice_width = 0.04;
std::map<int, bool> Dataset::enabled_objects;
std::string Dataset::window_name;
int Dataset::value_rr = MAXVAL / 2, Dataset::value_rp = MAXVAL / 2, Dataset::value_ry = MAXVAL / 2;
int Dataset::value_tx = MAXVAL / 2, Dataset::value_ty = MAXVAL / 2, Dataset::value_tz = MAXVAL / 2;
bool Dataset::modified = true;
float Dataset::pose_filtering_window = 0.04;

// Time offset controls
float Dataset::image_to_event_to, Dataset::pose_to_event_to;
int Dataset::image_to_event_to_slider = MAXVAL / 2, Dataset::pose_to_event_to_slider = MAXVAL / 2;

// Folder names
std::string Dataset::dataset_folder = "";
std::string Dataset::gt_folder = "";
