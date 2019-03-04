#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <boost/filesystem.hpp>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <ros/package.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf/transform_broadcaster.h>
#include <sensor_msgs/Range.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/subscriber.h>
#include <visualization_msgs/MarkerArray.h>
#include <image_transport/image_transport.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vicon/Subject.h>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include "object.h"
#include "event_vis.h"


std::vector<ViObject*> objects;
StaticObject *room_scan;

// Calibration matrix
float fx, fy, cx, cy, k1, k2, k3, k4;

// Camera center to vicon
float rr0, rp0, ry0, tx0, ty0, tz0;




bool parse_config(std::string path, std::string &conf) {
    std::ifstream ifs;
    ifs.open(path, std::ifstream::in);
    if (!ifs.is_open()) {
        std::cout << "Could not open configuration file at " 
                  << path << "!" << std::endl;
        return false;
    }
    conf = "---";
    for (int i = 0; i < 3; ++i) {
        std::string line;
        std::getline(ifs, line);
        if (line.find("true") != std::string::npos) {
            std::cout << "\tEnabling object " << i << std::endl;
            conf[i] = '+';
        }
    }

    ifs.close();
    return true;
}


bool read_cam_intr(std::string path) {
    std::ifstream ifs;
    ifs.open(path, std::ifstream::in);
    if (!ifs.is_open()) {
        std::cout << "Could not open camera intrinsic calibration file at " 
                  << path << "!" << std::endl;
        return false;
    }

    ifs >> fx >> fy >> cx >> cy;
    if (!ifs.good()) {
        std::cout << "Camera calibration read error: Expected a file with a single line, containing "
                  << "fx fy cx cy {k1 k2 k3 k4} ({} are optional)" << std::endl;
        return false;
    }
    
    k1 = k2 = k3 = k4 = 0;
    ifs >> k1 >> k2 >> k3 >> k4;
    
    std::cout << "Read camera calibration: (fx fy cx cy {k1 k2 k3 k4}): "
              << fx << " " << fy << " " << cx << " " << cy << " "
              << k1 << " " << k2 << " " << k3 << " " << k4 << std::endl;
          
    ifs.close();
    return true;
}


bool read_extr(std::string path) {
    std::ifstream ifs;
    ifs.open(path, std::ifstream::in);
    if (!ifs.is_open()) {
        std::cout << "Could not open extrinsic calibration file at " 
                  << path << "!" << std::endl;
        return false;
    }

    ifs >> tx0 >> ty0 >> tz0 >> rr0 >> rp0 >> ry0;
    if (!ifs.good()) {
        std::cout << "Camera -> Vicon is suppposed to be in <x y z R P Y> format!" << std::endl;
        return false;
    }

    float bg_tx, bg_ty, bg_tz, bg_qw, bg_qx, bg_qy, bg_qz;
    ifs >> bg_tx >> bg_ty >> bg_tz >> bg_qw >> bg_qx >> bg_qy >> bg_qz;
    if (!ifs.good()) {
        std::cout << "Background -> Vicon is suppposed to be in <x y z Qw Qx Qy Qz> format!" << std::endl;
        return false;
    }

    ifs.close();


    tf::Vector3 T;
    tf::Quaternion Q(bg_qx, bg_qy, bg_qz, bg_qw);
    T.setValue(bg_tx, bg_ty, bg_tz);
    
    tf::Transform E_bg;
    E_bg.setRotation(Q);
    E_bg.setOrigin(T);

    if (room_scan != NULL)
        room_scan->transform(E_bg);

    return true;
}

// Service functions
tf::Transform vicon2tf(const vicon::Subject& p) {
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(p.position.x, p.position.y, p.position.z));
    tf::Quaternion q(p.orientation.x,
                     p.orientation.y,
                     p.orientation.z,
                     p.orientation.w); 
    transform.setRotation(q);
    return transform;
}


class Pose {
public:
    ros::Time ts;
    tf::Transform pq;
    Pose(ros::Time ts_, tf::Transform pq_):
        ts(ts_), pq(pq_) {}
};


class Trajectory {
public:
    std::vector<Pose> poses;
    void add(ros::Time ts_, tf::Transform pq_) {
        this->poses.push_back(Pose(ts_, pq_));
    }

    size_t size() {return this->poses.size(); }
    inline auto begin() {return this->poses.begin(); }
    inline auto end()   {return this->poses.end(); }
    inline auto operator [] (size_t idx) {return this->poses[idx]; }

    bool check() {
        if (this->size() == 0) return true;
        auto prev_ts = this->poses[0].ts;
        for (auto &p : this->poses) {
            if (p.ts < prev_ts) return false;
            prev_ts = p.ts;
        }
        return true;
    }

    void subtract_time(ros::Time t) {
        for (auto &p : this->poses) p.ts = ros::Time((p.ts - t).toSec());
    }
};


int main (int argc, char** argv) {

    // Initialize ROS
    std::string node_name = "event_imo_offline";
    ros::init (argc, argv, node_name);
    ros::NodeHandle nh;
 
    std::string dataset_folder = "";
    if (!nh.getParam(node_name + "/folder", dataset_folder)) {
        std::cerr << "No dataset folder specified!" << std::endl;
        return -1;
    }

    float FPS = 40.0;
    int traj_smoothing = 1;
    if (!nh.getParam(node_name + "/fps", FPS)) FPS = 40;
    if (!nh.getParam(node_name + "/smoothing", traj_smoothing)) traj_smoothing = 1;

    // -- camera / object topics
    std::string cam_pose_topic = "";
    std::vector<std::string> obj_pose_topics(32); // modify if the maximum object id is larger
    if (!nh.getParam(node_name + "/cam_pose_topic", cam_pose_topic)) cam_pose_topic = "/vicon/DVS346";
    if (!nh.getParam(node_name + "/obj_pose_topic_0", obj_pose_topics[0])) obj_pose_topics[0] = "/vicon/Object_1";
    if (!nh.getParam(node_name + "/obj_pose_topic_1", obj_pose_topics[1])) obj_pose_topics[1] = "/vicon/Object_2";
    if (!nh.getParam(node_name + "/obj_pose_topic_2", obj_pose_topics[2])) obj_pose_topics[2] = "/vicon/Object_3";

    // -- parse the dataset folder
    std::string bag_name = boost::filesystem::path(dataset_folder).stem().string();
    if (bag_name == ".") {
        bag_name = boost::filesystem::path(dataset_folder).parent_path().stem().string();
    }

    bag_name = boost::filesystem::path(dataset_folder).append(bag_name + ".bag").string();
    std::cout << _blue("Procesing bag file: ") << bag_name << std::endl;

    rosbag::Bag bag;
    bag.open(bag_name, rosbag::bagmode::Read);
    rosbag::View view(bag);
    std::vector<const rosbag::ConnectionInfo *> connection_infos = view.getConnections();
    
    std::cout << std::endl << "Topics available:" << std::endl;
    for (auto &info : connection_infos) {
        std::cout << "\t" << info->topic << std::endl;
    }

    std::string active_objects;
    if (!parse_config(dataset_folder + "/config.txt", active_objects))
        return -1;

    // Load 3D models
    std::string path_to_self = ros::package::getPath("event_imo_datagen");
    //StaticObject room(path_to_self + "/objects/room");
    //room_scan = &room;

    ViObject obj1(nh, path_to_self + "/objects/toy_car", 1);
    if (active_objects[0] == '+') {
        objects.push_back(&obj1);
    }
/*
    ViObject obj2(nh, path_to_self + "/objects/toy_plane", 2);
    if (active_objects[1] == '+') {
        objects.push_back(&obj2);
    }

    ViObject obj3(nh, path_to_self + "/objects/cup", 3);
    if (active_objects[2] == '+') {
        objects.push_back(&obj3);
    }
*/
    // Camera intrinsic calibration
    if (!read_cam_intr(dataset_folder + "/calib.txt"))
        return -1;

    // Camera center -> Vicon and Background -> Vicon
    if (!read_extr(dataset_folder + "/extrinsics.txt"))
        return -1;

    // Extract topics from bag
    Trajectory cam_tj;
    std::vector<Trajectory> obj_tjs(32);
    for (auto &m : view) {
        if (m.getTopic() == cam_pose_topic) {
            auto msg = m.instantiate<vicon::Subject>();  
            if (msg == NULL) continue;
            cam_tj.add(msg->header.stamp, vicon2tf(*msg));
            continue;
        }

        for (int i = 0; i < obj_pose_topics.size(); ++i) {
            if (m.getTopic() == obj_pose_topics[i]) {
                auto msg = m.instantiate<vicon::Subject>();
                if (msg == NULL) break;
                if (msg->occluded) break;
                obj_tjs[i].add(msg->header.stamp, vicon2tf(*msg));
                break;
            }
        }
    }

    std::cout << std::endl << _green("Read ") << cam_tj.size() << _green(" camera poses and ") << std::endl;
    for (int i = 0; i < obj_tjs.size(); ++i) {
        if (obj_tjs[i].size() == 0) continue;
        std::cout << "\t" << obj_tjs[i].size() << _blue(" poses for object ") << i << std::endl;
        if (!obj_tjs[i].check()) {
            std::cout << "\t\t" << _red("Check failed!") << std::endl;
        }
    }

    // Remove time offset from poses
    auto time_offset = cam_tj[0].ts;
    for (auto &obj_tj : obj_tjs)
        if (obj_tj.size() > 0 && obj_tj[0].ts < time_offset) time_offset = obj_tj[0].ts;
    cam_tj.subtract_time(time_offset);
    for (auto &obj_tj : obj_tjs)
        obj_tj.subtract_time(time_offset);
    std::cout << std::endl << "Removing time offset: " << _green(std::to_string(time_offset.toSec())) << std::endl;    
 
    double start_ts = 0.0;
    double dt = 1.0 / FPS;
    long int cam_tj_id = 0;
    std::vector<long int> obj_tj_ids(32, 0);
    while (true) {
        while (cam_tj_id < cam_tj.size() && cam_tj[cam_tj_id].ts.toSec() < start_ts) cam_tj_id ++;
        for (int i = 0; i < obj_tjs.size(); ++i)
            while (obj_tj_ids[i] < obj_tjs[i].size() && obj_tjs[i][obj_tj_ids[i]].ts.toSec() < start_ts) obj_tj_ids[i] ++;
        start_ts += dt;

        bool done = false;
        if (cam_tj_id >= cam_tj.size()) done = true;
        for (int i = 0; i < obj_tjs.size(); ++i)
            if (obj_tjs[i].size() > 0 && obj_tj_ids[i] >= obj_tjs[i].size()) done = true;
        if (done) break;

        double max_ts_err = 0.0;
        for (int i = 0; i < obj_tjs.size(); ++i) {
            if (obj_tjs[i].size() == 0) continue;
            double ts_err = std::fabs(cam_tj[cam_tj_id].ts.toSec() - obj_tjs[i][obj_tj_ids[i]].ts.toSec());
            if (ts_err > max_ts_err) max_ts_err = ts_err;
        }
        if (max_ts_err > 0.01) {
            std::cout << _red("Trajectory timestamp misalignment: ") << max_ts_err << " skipping..." << std::endl;
            continue;
        }

        std::cout << cam_tj_id << ": " << cam_tj[cam_tj_id].ts;
        for (int i = 0; i < obj_tjs.size(); ++i) {
            if (obj_tjs[i].size() == 0) continue;
            std::cout << " " << obj_tjs[i][obj_tj_ids[i]].ts << " (" << obj_tj_ids[i] << ")";
        }
        std::cout << std::endl;
    }


    return 0;
}
