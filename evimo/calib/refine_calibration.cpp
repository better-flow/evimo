#include <vector>
#include <valarray>
#include <algorithm>
#include <thread>
#include <type_traits>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <ros/package.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

// VICON
#include <vicon/Subject.h>

// DVS / DAVIS
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

// Local includes
#include <common.h>
#include <event.h>
#include <event_vis.h>
#include <dataset.h>
#include <object.h>
#include <filters.h>

// Detect Vicon Wand
#include "detect_wand.h"


int main (int argc, char** argv) {

    // Initialize ROS
    std::string node_name = "calibration_refinement";
    ros::init (argc, argv, node_name, ros::init_options::AnonymousName);
    ros::NodeHandle nh("~");

    std::string bag_name = "";
    if (!nh.getParam("bag", bag_name)) {
        std::cerr << "No bag file  specified!" << std::endl;
        return -1;
    }

    // Camera name
    std::string camera_name = "";
    if (!nh.getParam("camera_name", camera_name)) camera_name = "main_camera";

    // Read dataset configuration files
    std::string path_to_self = ros::package::getPath("evimo");
    std::string dataset_folder = path_to_self + "/config/"; // default
    if (!nh.getParam("folder", dataset_folder)) {
        std::cout << _yellow("No configuration folder specified! Using: ")
                  << dataset_folder << std::endl;
    }

    auto dataset = std::make_shared<Dataset>();
    if (!dataset->init_no_objects(dataset_folder, camera_name))
        return -1;

    float start_time_offset = 0.0, sequence_duration = -1.0;
    if (!nh.getParam("start_time_offset", start_time_offset)) start_time_offset =  0.0;
    if (!nh.getParam("sequence_duration", sequence_duration)) sequence_duration = -1.0;

    bool with_images = true;
    if (!nh.getParam("with_images", with_images)) with_images = true;
    else std::cout << _yellow("With 'with_images' option, the datased will be generated at image framerate.") << std::endl;

    // Force wand trajectory
    Dataset::obj_pose_topics[0] = "/vicon/Wand";

    // Extract topics from bag
    if (!dataset->read_bag_file(bag_name, start_time_offset, sequence_duration, with_images, false)) {
        return 0;
    }
    with_images = (Dataset::images.size() > 0);

    // Make sure there is no trajectory filtering
    Dataset::cam_tj.set_filtering_window_size(-1);
    Dataset::obj_tjs[0].set_filtering_window_size(-1);

    cv::namedWindow("img", cv::WINDOW_NORMAL);
    EventSlice2D esg(dataset->res_x, dataset->res_y, ull(0.03 * 1e9));
    for (size_t i = 0; i < Dataset::event_array.size(); ++i) {
        esg.push_back(Dataset::event_array[i]);
        if (i % 10000 == 0) {
            esg.remove_outside_window();

            cv::imshow("img", (cv::Mat)esg * 10);
            auto code = cv::waitKey(10);
            if (code == 27) break;
        }
    }


    std::cout << _green("Done!") << std::endl;
    return 0;
};
