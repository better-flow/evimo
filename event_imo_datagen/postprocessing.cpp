#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include <ros/ros.h>
#include <ros/package.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/transformation_estimation_svd.h>

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

vicon::Subject last_cam_pos;
static unsigned long int numreceived = 0;
static unsigned long int epacks_received = 0;
ros::Time first_event_msg_ts;
ros::Time last_event_msg_ts;
ros::Time first_cam_pos_ts;

// Calibration matrix
float fx, fy, cx, cy, k1, k2, k3, k4;

// Camera center to vicon
float rr0, rp0, ry0, tx0, ty0, tz0;


static std::string mode = "CALIBRATION";
float FPS;
tf::Transform E;
static std::string cv_window_name = "Overlay output";
static std::string cv_window_name_bg = "Background adjustment";
static cv::Mat vis_img;
static bool vis_mode_depth = false;

ros::Publisher vis_pub, vis_pub_range;
image_transport::Publisher image_pub;

std::string dataset_folder;


// Event buffer
#define EVENT_WIDTH 30000
#define TIME_WIDTH 0.05
static ull start_timestamp = 0;
CircularArray<Event, EVENT_WIDTH, FROM_SEC(TIME_WIDTH)> ev_buffer;  
std::list<Event> all_events;
std::list<std::pair<cv::Mat, double>> all_depthmaps;
std::list<std::pair<int, vicon::Subject>> all_poses;


void process_camera();
void event_cb(const dvs_msgs::EventArray::ConstPtr& msg) {
    if ((epacks_received == 0) && (msg->events.size() > 0)) {
        first_event_msg_ts = msg->header.stamp;
        start_timestamp = msg->events[0].ts.toNSec();
        std::cout << "The first event timestamp: " << _green(start_timestamp) << std::endl;
    }

    for (uint i = 0; i < msg->events.size(); ++i) {
        ull time = msg->events[i].ts.toNSec() - start_timestamp;
        Event e(msg->events[i].y, msg->events[i].x, time, (msg->events[i].polarity ? 1 : 0));
        ev_buffer.push_back(e);
        all_events.push_back(e);
    }

    if (msg->events.size() > 0) {
        epacks_received ++;
        last_event_msg_ts = msg->header.stamp;
    }

    if (mode == "CALIBRATION") {
        process_camera();
    }
}


visualization_msgs::Marker get_generic_marker() {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "/vicon";
    marker.header.stamp = ros::Time();
    marker.ns = "vicon_markers";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 1;
    marker.pose.position.y = 1;
    marker.pose.position.z = 1;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.02;
    marker.scale.y = 0.02;
    marker.scale.z = 0.02;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    return marker;
}


void project_point(pcl::PointXYZRGB p, int &u, int &v) {
    //cv_p.x = p.z;
    //cv_p.y = p.y;
    //cv_p.z = p.x;
    u = 0; v = 0;
    if (p.x < 0.00001)
        return;

    float x_ = p.z / p.x;
    float y_ = p.y / p.x;
    float r2 = x_ * x_ + y_ * y_;
    float r4 = r2 * r2;
    float r6 = r2 * r2 * r2;
    float dist = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2);
    float x__ = x_ * dist;
    float y__ = y_ * dist;

    u = fx * x__ + cx;
    v = fy * y__ + cy;
}


void project_cloud(cv::Mat &img, pcl::PointCloud<pcl::PointXYZRGB> *cl, int oid) {
    if (cl->size() == 0)
        return;

    for (auto &p: *cl) {
        int u = 0, v = 0;
        project_point(p, u, v);

        if (u < 0 || v < 0 || v >= img.cols || u >= img.rows)
            continue;

        float rng = p.x;
        float base_rng = img.at<cv::Vec3f>(img.rows - u - 1, img.cols - v - 1)[0];

        if (base_rng > rng || base_rng < 0.001) {
            img.at<cv::Vec3f>(img.rows - u - 1, img.cols - v - 1)[0] = rng;
            img.at<cv::Vec3f>(img.rows - u - 1, img.cols - v - 1)[1] = 0;
            if (oid > 0)
                img.at<cv::Vec3f>(img.rows - u - 1, img.cols - v - 1)[2] = oid;
        }
    }
}


void update_vis_img(cv::Mat &projected) {
    vis_img = cv::Mat(projected.rows, projected.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    std::vector<cv::Mat> spl;
    cv::split(projected, spl);
    cv::Mat depth = spl[0];
    cv::Mat mask  = spl[2];

    cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX);

    cv::Mat img_pr = EventFile::projection_img(&ev_buffer, 1);
    cv::Mat img_color = EventFile::color_time_img(&ev_buffer, 1);

    int nRows = vis_img.rows;
    int nCols = vis_img.cols;
    for(int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            vis_img.at<cv::Vec3b>(i, j)[2] = img_pr.at<uchar>(i, j);

            if (vis_mode_depth) {
                vis_img.at<cv::Vec3b>(i, j)[0] = depth.at<float>(i, j);
            } else {
                int id = std::round(mask.at<float>(i, j));
                auto color = EventFile::id2rgb(id);
                vis_img.at<cv::Vec3b>(i, j) = color;
            }
        }
    }
}


void process_camera() {
    if (numreceived == 0)
        return;

    visualization_msgs::MarkerArray vis_markers;
      
    auto vis_marker = get_generic_marker();
    vis_marker.ns = "dvs_camera";
    vis_marker.pose.position = last_cam_pos.position;
    vis_marker.color.r = 1;
    vis_marker.color.g = 0;
    vis_marker.color.b = 1;

    vis_markers.markers.push_back(vis_marker);
    vis_pub.publish(vis_markers);

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(last_cam_pos.position.x, last_cam_pos.position.y, last_cam_pos.position.z));
    tf::Quaternion q(last_cam_pos.orientation.x,
                     last_cam_pos.orientation.y,
                     last_cam_pos.orientation.z,
                     last_cam_pos.orientation.w); 
    transform.setRotation(q);
    static tf::TransformBroadcaster tf_br;
    tf_br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/vicon", "/camera_markers"));

    sensor_msgs::Range cone;
    cone.field_of_view = 1;
    cone.min_range = 0;
    cone.max_range = 5;
    cone.range = 3;
    cone.header.frame_id = "/camera_center";
    vis_pub_range.publish(cone);

    // =====================
    Eigen::Matrix4f Tm;
    Tm <<  0.0,    1.0,   0.0,  0.00,
          -1.0,    0.0,   0.0,  0.00,
           0.0,    0.0,   1.0,  0.00,
             0,      0,     0,     1;
    Tm = Tm.inverse().eval();   

    auto to_camcenter = transform * ViObject::mat2tf(Tm) * E;
    tf_br.sendTransform(tf::StampedTransform(to_camcenter, ros::Time::now(), "/vicon", "/camera_center"));

    for (auto &obj : objects) {
        if (!obj->update_camera_pose(to_camcenter)) {
            return;
        }
    }

    if (epacks_received == 0)
        return;

    // Room scan transformation
    room_scan->update_camera_pose(to_camcenter);

    double et_sec = double(all_events.back().timestamp / 1000) / 1000000.0;
    double image_ts = et_sec + (last_cam_pos.header.stamp - last_event_msg_ts).toSec();

    cv::Mat projected(RES_X, RES_Y, CV_32FC3, cv::Scalar(0, 0, 0));
    
    for (auto &obj : objects) {
        project_cloud(projected, obj->get_cloud(), obj->get_id());
    }

    // Room scan projection
    project_cloud(projected, room_scan->get_cloud(), 0);

    all_depthmaps.push_back(std::make_pair(projected, image_ts));
    all_poses.push_back(std::make_pair(0, ViObject::tf2subject(to_camcenter)));
    for (auto &obj : objects) {
        all_poses.push_back(std::make_pair(obj->get_id(), obj->get_last_pos()));
    }

    // Visualization
    update_vis_img(projected);

    sensor_msgs::ImagePtr img_depth_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", vis_img).toImageMsg();                
    image_pub.publish(img_depth_msg);
}


void camera_pos_cb(const vicon::Subject& subject) {
    if (subject.occluded) 
        return;

    if (numreceived == 0)
        first_cam_pos_ts = subject.header.stamp;

    if (mode == "CALIBRATION") {
        last_cam_pos = subject;
        numreceived ++;
        return;
    }

    if ((subject.header.stamp - last_cam_pos.header.stamp).toSec() < 1.0 / FPS)
        return;

    last_cam_pos = subject;
    numreceived ++;

    process_camera();
}


float normval(int val, int maxval, int normval) {
    return float(val - maxval / 2) / float(normval);
}


static bool changed = false;
void on_trackbar(int, void*) {
    changed = true;
}


void save_data(std::string dir) {
    std::cout << "Gt frames: " << all_depthmaps.size() << "\t" << "events: " << all_events.size() << std::endl;
    std::cout << "Writing to " << dir << std::endl;

    unsigned long int i = 0;

    std::string calibfname = dir + "/calib.txt";
    std::ofstream calib_file(calibfname, std::ofstream::out);
    calib_file << fy << " " << 0  << " " << cy << std::endl;
    calib_file << 0  << " " << fx << " " << cx << std::endl;
    calib_file << 0  << " " << 0  << " " << 1  << std::endl;
    calib_file << std::endl;
    calib_file << k1 / 10 << " " << k2 / 10 << " " << k3 / 10 << " " << k4 / 10 << std::endl;
    calib_file.close();

    std::string tsfname = dir + "/ts.txt";
    std::ofstream ts_file(tsfname, std::ofstream::out);

    std::string obj_fname = dir + "/objects.txt";
    std::ofstream obj_file(obj_fname, std::ofstream::out);

    std::string cam_fname = dir + "/trajectory.txt";
    std::ofstream cam_file(cam_fname, std::ofstream::out);

    std::unordered_map<int, int> cnts = {{0, 0}};
    for (auto &obj : objects) {
        cnts.insert({obj->get_id(), 0});
    }

    for (auto &pos : all_poses) {
        cnts.at(pos.first) ++;
        auto &loc = pos.second.position;
        auto &rot = pos.second.orientation;

        if (pos.first == 0) {
            cam_file << cnts.at(pos.first) << " "
                     << loc.x << " " << loc.y << " " << loc.z << " "
                     << rot.w << " " << rot.x << " " << rot.y << " " << rot.z << std::endl;
            continue;
        }

        obj_file << cnts.at(pos.first) << " " << pos.first << " "
                 << loc.x << " " << loc.y << " " << loc.z << " "
                 << rot.w << " " << rot.x << " " << rot.y << " " << rot.z << std::endl;
    }

    i = 0;
    for (auto &pair : all_depthmaps) {
        if (i % 10 == 0) {
            std::cout << "Written " << i << " / " << all_depthmaps.size() <<  std::endl;
        }

        std::string gtfname = dir + "/gt/frame_" + std::to_string(i) + ".png";
        ts_file << gtfname << " " << pair.second << std::endl;

        cv::Mat projected_i16(RES_X, RES_Y, CV_16UC3, cv::Scalar(0, 0, 0));

        projected_i16 = pair.first * 1000;

        projected_i16.convertTo(projected_i16, CV_16UC3);


        cv::imwrite(gtfname, projected_i16);        
        i ++;
    }
    ts_file.close();
    obj_file.close();
    cam_file.close();
    std::cout << "GT written....\n";

    std::string efname = dir + "/events.txt";
    std::ofstream event_file(efname, std::ofstream::out);
    i = 0;
    for (auto &e : all_events) {
        if (i % 100000 == 0) {
            std::cout << "Written " << i << " / " << all_events.size() <<  std::endl;
        }
        
        event_file << std::fixed << std::setprecision(9) 
                   << double(e.timestamp / 1000) / 1000000.0 
                   << " " << e.fr_y << " " << e.fr_x 
                   << " " << int(e.polarity) << std::endl;
        i ++;
    }
    event_file.close();
    std::cout << "Events written... Done\n";
}


cv::Mat undistort(cv::Mat &img) {
    cv::Mat ret;
    cv::Mat K = (cv::Mat1d(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat D = (cv::Mat1d(1, 4) << k1, k2, k3, k4);
    cv::undistort(img, ret, K, D, 0.87 * K);
    return ret;
}


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

    room_scan->transform(E_bg);

    return true;
}


int main (int argc, char** argv) {
    // Initialize ROS
    ros::init (argc, argv, "event_imo_datagen");
    ros::NodeHandle nh;
    image_transport::ImageTransport it_(nh);
  
    // Create ROS subscribers / publishers
    vis_pub = nh.advertise<visualization_msgs::MarkerArray>("/ev_imo/markers", 0);
    vis_pub_range = nh.advertise<sensor_msgs::Range>("/ev_imo/markers_range",  0);

    ros::Subscriber cam_sub = nh.subscribe("/vicon/DVS346", 0, camera_pos_cb);
    ros::Subscriber event_sub = nh.subscribe("/dvs/events", 0, event_cb);
    image_pub = it_.advertise("/ev_imo/depth_raw", 1);

    if (!nh.getParam("event_imo_datagen/folder", dataset_folder)) dataset_folder = "";
    if (!nh.getParam("event_imo_datagen/fps", FPS)) FPS = 40;

    std::string path_to_self = ros::package::getPath("event_imo_datagen");

    last_cam_pos.header.stamp = ros::Time(0);
    last_event_msg_ts         = ros::Time(0);
    first_event_msg_ts        = ros::Time(0);
    first_cam_pos_ts          = ros::Time(0);

    if (dataset_folder == "") {
        std::cout << "Need to specify the dataset directory containing configuration files!" << std::endl;
        return -1;
    }

    // ==== Register objects ====
    StaticObject room(path_to_self + "/objects/room");
    room_scan = &room;

    std::string active_objects;
    if (!parse_config(dataset_folder + "/config.txt", active_objects))
        return -1;

    ViObject obj1(nh, path_to_self + "/objects/toy_car", 1);
    if (active_objects[0] == '+') {
        objects.push_back(&obj1);
    }

    ViObject obj2(nh, path_to_self + "/objects/toy_plane", 2);
    if (active_objects[1] == '+') {
        objects.push_back(&obj2);
    }

    ViObject obj3(nh, path_to_self + "/objects/cup", 3);
    if (active_objects[2] == '+') {
        objects.push_back(&obj3);
    }
    // ==========================


    // Camera intrinsic calibration
    if (!read_cam_intr(dataset_folder + "/calib.txt"))
        return -1;

    // Camera center -> Vicon and Background -> Vicon
    if (!read_extr(dataset_folder + "/extrinsics.txt"))
        return -1;

    tf::Vector3 T;
    tf::Quaternion q;
    q.setRPY(rr0, rp0, ry0);
    T.setValue(tx0, ty0, tz0);
    E.setRotation(q);
    E.setOrigin(T);

    vis_img = EventFile::color_time_img(&ev_buffer, 1);

    // Spin
    if (mode == "DEMO") {
        ros::spin();
        ros::shutdown();
        return 0;
    }

    if (mode != "CALIBRATION" && mode != "GENERATION") {
        std::cout << "Unsupported mode of operation: " << mode << std::endl;
        ros::shutdown();
        return 0;
    }

    int maxval = 1000;
    int value_rr = maxval / 2, value_rp = maxval / 2, value_ry = maxval / 2;
    int value_tx = maxval / 2, value_ty = maxval / 2, value_tz = maxval / 2;
    int value_rr_bg = maxval / 2, value_rp_bg = maxval / 2, value_ry_bg = maxval / 2;
    int value_tx_bg = maxval / 2, value_ty_bg = maxval / 2, value_tz_bg = maxval / 2;
    int value_br = 250;

    cv::namedWindow(cv_window_name, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("R", cv_window_name, &value_rr, maxval, on_trackbar);
    cv::createTrackbar("P", cv_window_name, &value_rp, maxval, on_trackbar);
    cv::createTrackbar("Y", cv_window_name, &value_ry, maxval, on_trackbar);
    cv::createTrackbar("x", cv_window_name, &value_tx, maxval, on_trackbar);
    cv::createTrackbar("y", cv_window_name, &value_ty, maxval, on_trackbar);
    cv::createTrackbar("z", cv_window_name, &value_tz, maxval, on_trackbar);

    cv::namedWindow(cv_window_name_bg, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("R", cv_window_name_bg, &value_rr_bg, maxval, on_trackbar);
    cv::createTrackbar("P", cv_window_name_bg, &value_rp_bg, maxval, on_trackbar);
    cv::createTrackbar("Y", cv_window_name_bg, &value_ry_bg, maxval, on_trackbar);
    cv::createTrackbar("x", cv_window_name_bg, &value_tx_bg, maxval, on_trackbar);
    cv::createTrackbar("y", cv_window_name_bg, &value_ty_bg, maxval, on_trackbar);
    cv::createTrackbar("z", cv_window_name_bg, &value_tz_bg, maxval, on_trackbar);
    cv::createTrackbar("+", cv_window_name_bg, &value_br,    maxval, on_trackbar);

    changed = true;
    int code = 0;
    while (ros::ok() && (code != 27)) {
        // ====== CV GUI ======
        float scale = (float(value_br) / float(maxval) * 2 + 0.5);
        cv::imshow(cv_window_name, vis_img * scale);
        cv::imshow(cv_window_name_bg, undistort(vis_img) * scale);
        code = cv::waitKey(1);
        
        if (code == 99) { // 'c'
            cv::setTrackbarPos("R", cv_window_name, maxval / 2);
            cv::setTrackbarPos("P", cv_window_name, maxval / 2);
            cv::setTrackbarPos("Y", cv_window_name, maxval / 2);
            cv::setTrackbarPos("x", cv_window_name, maxval / 2);
            cv::setTrackbarPos("y", cv_window_name, maxval / 2);
            cv::setTrackbarPos("z", cv_window_name, maxval / 2);
            cv::setTrackbarPos("R", cv_window_name_bg, maxval / 2);
            cv::setTrackbarPos("P", cv_window_name_bg, maxval / 2);
            cv::setTrackbarPos("Y", cv_window_name_bg, maxval / 2);
            cv::setTrackbarPos("x", cv_window_name_bg, maxval / 2);
            cv::setTrackbarPos("y", cv_window_name_bg, maxval / 2);
            cv::setTrackbarPos("z", cv_window_name_bg, maxval / 2);
        }

        if (code == 115) { // 's'
            save_data(dataset_folder);
        }

        if (code == 32) {
            vis_mode_depth = !vis_mode_depth;
            changed = true;
        }

        float rr = rr0 + normval(value_rr, maxval, maxval * 10);
        float rp = rp0 + normval(value_rp, maxval, maxval * 10);
        float ry = ry0 + normval(value_ry, maxval, maxval * 10);
        float tx = tx0 + normval(value_tx, maxval, maxval * 50);
        float ty = ty0 + normval(value_ty, maxval, maxval * 50);
        float tz = tz0 + normval(value_tz, maxval, maxval * 50);

        float rr_bg = normval(value_rr_bg, maxval, maxval * 10);
        float rp_bg = normval(value_rp_bg, maxval, maxval * 10);
        float ry_bg = normval(value_ry_bg, maxval, maxval * 10);
        float tx_bg = normval(value_tx_bg, maxval, maxval * 10);
        float ty_bg = normval(value_ty_bg, maxval, maxval * 10);
        float tz_bg = normval(value_tz_bg, maxval, maxval * 10);

        if (changed) {
            changed = false;

            tf::Vector3 T;
            tf::Quaternion q;
            q.setRPY(rr, rp, ry);
            T.setValue(tx, ty, tz);
            E.setRotation(q);
            E.setOrigin(T);

            tf::Transform E_bg;
            tf::Vector3 T_bg;
            tf::Quaternion q_bg;
            q_bg.setRPY(rr_bg, rp_bg, ry_bg);
            T_bg.setValue(tx_bg, ty_bg, tz_bg);
            E_bg.setRotation(q_bg);
            E_bg.setOrigin(T_bg);

            auto to_cam = room_scan->get_to_camcenter();
            room_scan->transform(room_scan->get_static() * to_cam * E_bg * to_cam.inverse());
 
            process_camera();
        }

        if (code == 105) { // 'i'
            ros::Time ros_start_time = first_cam_pos_ts;
            std::cout << "ROS first message time (offset): " << ros_start_time << std::endl;
            std::cout << "Last camera pos ts: " << last_cam_pos.header.stamp - ros_start_time << std::endl;
            std::cout << "Last event pack ts: " << last_event_msg_ts - ros_start_time << std::endl;
            
            double et_sec = double(all_events.back().timestamp / 1000) / 1000000.0;
            std::cout << "Last event ts: " << et_sec << "\t|\t" << all_events.back().timestamp << std::endl;
            std::cout << "Event pack - event: " << (last_event_msg_ts - ros_start_time).toSec() - et_sec << std::endl;
            std::cout << "Cam pos - event: " << (last_cam_pos.header.stamp - ros_start_time).toSec() - et_sec << std::endl;
            std::cout << "Image timestamp: " << et_sec + (last_cam_pos.header.stamp - last_event_msg_ts).toSec() << std::endl; 
            std::cout << "Gt frames: " << all_depthmaps.size() << "\t" << "events: " << all_events.size() << std::endl << std::endl;
            std::cout << "Transforms:" << std::endl;
            std::cout << "Vicon -> Camcenter (X Y Z R P Y):" << std::endl;
            std::cout << "\t" << tx << "\t" << ty << "\t" << tz << "\t" << rr << "\t" << rp << "\t" << ry << std::endl;
            std::cout << "Vicon -> Background (X Y Z Qw Qx Qy Qz):" << std::endl;
            auto T = room_scan->get_static().getOrigin();
            auto Q = room_scan->get_static().getRotation();
            std::cout << "\t" << T.getX() << "\t" << T.getY() << "\t" << T.getZ()
                      << "\t" << Q.getW() <<"\t" << Q.getX() << "\t" << Q.getY() << "\t" << Q.getZ() << std::endl << std::endl;
        }

        // ====================

        ros::spinOnce();
    }

    ros::shutdown();
    return 0;
};
