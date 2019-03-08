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


class DatasetConfig {
public:

    // Calibration matrix
    static float fx, fy, cx, cy, k1, k2, k3, k4;

    // Camera center to vicon
    static float rr0, rp0, ry0, tx0, ty0, tz0;

    // Background to vicon
    static tf::Transform bg_init_E;

    // Other parameters
    static std::map<int, bool> enabled_objects;

    static bool init(std::string dataset_folder) {
        bool ret = DatasetConfig::parse_config(dataset_folder + "/config.txt");
        ret &= DatasetConfig::read_cam_intr(dataset_folder + "/calib.txt");
        ret &= DatasetConfig::read_extr(dataset_folder + "/extrinsics.txt");
        return ret;
    }

private:
    static bool parse_config(std::string path) {
        std::ifstream ifs;
        ifs.open(path, std::ifstream::in);
        if (!ifs.is_open()) {
            std::cout << _red("Could not open configuration file at ")
                      << path << "!" << std::endl;
            return false;
        }

        std::cout << _blue("Opening configuration file: ")
                  << path  << std::endl;
        for (int i = 0; i < 3; ++i) {
            std::string line;
            std::getline(ifs, line);
            if (line.find("true") != std::string::npos) {
                std::cout << _blue("\tEnabling object ") << i + 1 << std::endl;
                enabled_objects[i + 1] = true;
            }
        }

        ifs.close();
        return true;
    }

    static bool read_cam_intr(std::string path) {
        std::ifstream ifs;
        ifs.open(path, std::ifstream::in);
        if (!ifs.is_open()) {
            std::cout << _red("Could not open camera intrinsic calibration file at ")
                      << path << "!" << std::endl;
            return false;
        }

        ifs >> fx >> fy >> cx >> cy;
        if (!ifs.good()) {
            std::cout << _red("Camera calibration read error:") << " Expected a file with a single line, containing "
                      << "fx fy cx cy {k1 k2 k3 k4} ({} are optional)" << std::endl;
            return false;
        }
        
        k1 = k2 = k3 = k4 = 0;
        ifs >> k1 >> k2 >> k3 >> k4;
        
        std::cout << _green("Read camera calibration: (fx fy cx cy {k1 k2 k3 k4}): ")
                  << fx << " " << fy << " " << cx << " " << cy << " "
                  << k1 << " " << k2 << " " << k3 << " " << k4 << std::endl;
              
        ifs.close();
        return true;
    }

    static bool read_extr(std::string path) {
        std::ifstream ifs;
        ifs.open(path, std::ifstream::in);
        if (!ifs.is_open()) {
            std::cout << _red("Could not open extrinsic calibration file at ")
                      << path << "!" << std::endl;
            return false;
        }

        ifs >> tx0 >> ty0 >> tz0 >> rr0 >> rp0 >> ry0;
        if (!ifs.good()) {
            std::cout << _red("Camera -> Vicon is suppposed to be in <x y z R P Y> format!") << std::endl;
            return false;
        }

        float bg_tx, bg_ty, bg_tz, bg_qw, bg_qx, bg_qy, bg_qz;
        ifs >> bg_tx >> bg_ty >> bg_tz >> bg_qw >> bg_qx >> bg_qy >> bg_qz;
        if (!ifs.good()) {
            std::cout << _red("Background -> Vicon is suppposed to be in <x y z Qw Qx Qy Qz> format!") << std::endl;
            return false;
        }

        ifs.close();

        tf::Vector3 T;
        tf::Quaternion Q(bg_qx, bg_qy, bg_qz, bg_qw);
        T.setValue(bg_tx, bg_ty, bg_tz);

        bg_init_E.setRotation(Q);
        bg_init_E.setOrigin(T);

        //if (room_scan != NULL)
        //    room_scan->transform(E_bg);

        return true;
    }
};

float DatasetConfig::fx, DatasetConfig::fy, DatasetConfig::cx, DatasetConfig::cy;
float DatasetConfig::k1, DatasetConfig::k2, DatasetConfig::k3, DatasetConfig::k4;
float DatasetConfig::rr0, DatasetConfig::rp0, DatasetConfig::ry0;
float DatasetConfig::tx0, DatasetConfig::ty0, DatasetConfig::tz0;
tf::Transform DatasetConfig::bg_init_E;
std::map<int, bool> DatasetConfig::enabled_objects;


// Service functions
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


class DatasetFrame {
protected:
    static std::shared_ptr<StaticObject> background;
    static std::map<int, std::shared_ptr<ViObject>> clouds;

public:
    Pose cam_pose;
    std::map<int, Pose> obj_poses;

    cv::Mat depth;
    cv::Mat mask;

public:
    static void add_cloud(int id, std::shared_ptr<ViObject> cl) {
        DatasetFrame::clouds[id] = cl;
    }
    static void add_background(std::shared_ptr<StaticObject> bg) {
        DatasetFrame::background = bg;
    }
    static void init_cloud(int id, const vicon::Subject& subject) {
        if (DatasetFrame::clouds.find(id) == DatasetFrame::clouds.end())
            return;
        clouds[id]->init_cloud_to_vicon_tf(subject);
    }

    // ---------
    DatasetFrame(Pose cam_p) 
        : cam_pose(cam_p) {}

    void add_object_pos(int id, Pose obj_p) {
        this->obj_poses.insert(std::make_pair(id, obj_p));
    }

    // Generate frame
    void generate() {


    }
};

std::shared_ptr<StaticObject> DatasetFrame::background;
std::map<int, std::shared_ptr<ViObject>> DatasetFrame::clouds;



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

    // -- camera topic
    std::string cam_pose_topic = "";
    std::map<int, std::string> obj_pose_topics;
    if (!nh.getParam(node_name + "/cam_pose_topic", cam_pose_topic)) cam_pose_topic = "/vicon/DVS346";

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

    // Read datasset configuration files
    if (!DatasetConfig::init(dataset_folder))
        return -1;

    // Load 3D models
    std::string path_to_self = ros::package::getPath("event_imo_datagen");
    
    //DatasetFrame::add_background(std::make_shared<StaticObject>(path_to_self + "/objects/room"));
    if (DatasetConfig::enabled_objects.find(1) != DatasetConfig::enabled_objects.end()) {
        DatasetFrame::add_cloud(1, std::make_shared<ViObject>(nh, path_to_self + "/objects/toy_car", 1));
        if (!nh.getParam(node_name + "/obj_pose_topic_0", obj_pose_topics[1])) obj_pose_topics[1] = "/vicon/Object_1";
    }

    if (DatasetConfig::enabled_objects.find(2) != DatasetConfig::enabled_objects.end()) {
        DatasetFrame::add_cloud(2, std::make_shared<ViObject>(nh, path_to_self + "/objects/toy_plane", 2));
        if (!nh.getParam(node_name + "/obj_pose_topic_1", obj_pose_topics[2])) obj_pose_topics[2] = "/vicon/Object_2";
    }

    if (DatasetConfig::enabled_objects.find(3) != DatasetConfig::enabled_objects.end()) {
        DatasetFrame::add_cloud(3, std::make_shared<ViObject>(nh, path_to_self + "/objects/cup", 3));
        if (!nh.getParam(node_name + "/obj_pose_topic_2", obj_pose_topics[3])) obj_pose_topics[3] = "/vicon/Object_3";
    }

    // Extract topics from bag
    Trajectory cam_tj;
    std::map<int, Trajectory> obj_tjs;
    std::map<int, vicon::Subject> obj_cloud_to_vicon_tf;
    for (auto &m : view) {
        if (m.getTopic() == cam_pose_topic) {
            auto msg = m.instantiate<vicon::Subject>();  
            if (msg == NULL) continue;
            cam_tj.add(msg->header.stamp, ViObject::subject2tf(*msg));
            continue;
        }

        for (auto &p : obj_pose_topics) {
            if (m.getTopic() != p.second) continue;
            auto msg = m.instantiate<vicon::Subject>();
            if (msg == NULL) break;
            if (msg->occluded) break;
            obj_tjs[p.first].add(msg->header.stamp, ViObject::subject2tf(*msg));
            obj_cloud_to_vicon_tf[p.first] = *msg;
            break;
        }
    }

    std::cout << std::endl << _green("Read ") << cam_tj.size() << _green(" camera poses and ") << std::endl;
    for (auto &obj_tj : obj_tjs) {
        if (obj_tj.second.size() == 0) continue;
        std::cout << "\t" << obj_tj.second.size() << _blue(" poses for object ") << obj_tj.first << std::endl;
        if (!obj_tj.second.check()) {
            std::cout << "\t\t" << _red("Check failed!") << std::endl;
        }
        DatasetFrame::init_cloud(obj_tj.first, obj_cloud_to_vicon_tf[obj_tj.first]);
    }

    // Remove time offset from poses
    auto time_offset = cam_tj[0].ts;
    for (auto &obj_tj : obj_tjs)
        if (obj_tj.second.size() > 0 && obj_tj.second[0].ts < time_offset) time_offset = obj_tj.second[0].ts;
    cam_tj.subtract_time(time_offset);
    for (auto &obj_tj : obj_tjs)
        obj_tj.second.subtract_time(time_offset);
    std::cout << std::endl << "Removing time offset: " << _green(std::to_string(time_offset.toSec())) << std::endl;    

    // Align the timestamps
    double start_ts = 0.0;
    double dt = 1.0 / FPS;
    long int cam_tj_id = 0;
    std::map<int, long int> obj_tj_ids;
    std::vector<DatasetFrame> frames;
    while (true) {
        while (cam_tj_id < cam_tj.size() && cam_tj[cam_tj_id].ts.toSec() < start_ts) cam_tj_id ++;
        for (auto &obj_tj : obj_tjs)
            while (obj_tj_ids[obj_tj.first] < obj_tj.second.size()
                   && obj_tj.second[obj_tj_ids[obj_tj.first]].ts.toSec() < start_ts) obj_tj_ids[obj_tj.first] ++;
        start_ts += dt;

        bool done = false;
        if (cam_tj_id >= cam_tj.size()) done = true;
        for (auto &obj_tj : obj_tjs)
            if (obj_tj.second.size() > 0 && obj_tj_ids[obj_tj.first] >= obj_tj.second.size()) done = true;
        if (done) break;

        double max_ts_err = 0.0;
        for (auto &obj_tj : obj_tjs) {
            if (obj_tj.second.size() == 0) continue;
            double ts_err = std::fabs(cam_tj[cam_tj_id].ts.toSec() - obj_tj.second[obj_tj_ids[obj_tj.first]].ts.toSec());
            if (ts_err > max_ts_err) max_ts_err = ts_err;
        }
        if (max_ts_err > 0.01) {
            std::cout << _red("Trajectory timestamp misalignment: ") << max_ts_err << " skipping..." << std::endl;
            continue;
        }


        DatasetFrame frame(cam_tj[cam_tj_id]);
        std::cout << cam_tj_id << ": " << cam_tj[cam_tj_id].ts;
        for (auto &obj_tj : obj_tjs) {
            if (obj_tj.second.size() == 0) continue;
            std::cout << " " << obj_tj.second[obj_tj_ids[obj_tj.first]].ts << " (" << obj_tj_ids[obj_tj.first] << ")";
            frame.add_object_pos(obj_tj.first, obj_tj.second[obj_tj_ids[obj_tj.first]]);
        }
        std::cout << std::endl;
        frames.push_back(frame);
    }

    std::cout << _blue("\nTimestamp alignment done") << std::endl;
    std::cout << "\tDataset contains " << frames.size() << " frames" << std::endl;

    return 0;
}
