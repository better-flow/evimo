#include <algorithm>
#include <iostream>

#include <ros/package.h>

#include <event.h>
#include <object.h>
#include <trajectory.h>


#ifndef DATASET_H
#define DATASET_H


class Dataset {
public:
    // The 3D scanned objects
    static std::shared_ptr<StaticObject> background;
    static std::map<int, std::shared_ptr<ViObject>> clouds;
    static std::map<int, std::string> obj_pose_topics;

    // Event cloud
    static std::vector<Event> event_array;

    // Camera frames
    static std::vector<cv::Mat> images;
    static std::vector<ros::Time> image_ts;

    // Trajectories for camera and objects
    static Trajectory cam_tj;
    static std::map<int, Trajectory> obj_tjs;

    // Camera params
    static std::string dist_model;
    static std::string image_topic, event_topic, cam_pos_topic;

    // Calibration matrix
    static float fx, fy, cx, cy, k1, k2, k3, k4, p1, p2;

    // Camera resolution
    static unsigned int res_x, res_y;

    // Camera center to vicon
    static float rr0, rp0, ry0, tx0, ty0, tz0;
    static tf::Transform cam_E;

    // Background to vicon
    static tf::Transform bg_E;

    // Time offset
    static float image_to_event_to, pose_to_event_to;
    static int image_to_event_to_slider, pose_to_event_to_slider;

    // Event slice width, for visualization
    static float slice_width;

    // Pose filtering window, in seconds
    static float pose_filtering_window;

    // Other parameters
    static std::string window_name;
    static bool modified;

    // Folder names
    static std::string dataset_folder, gt_folder;

    static constexpr float MAXVAL = 1000;
    static constexpr float INT_LIN_SC = 10.0;
    static constexpr float INT_ANG_SC = 10.0;
    static constexpr float INT_TIM_SC = 5;

    static int value_rr, value_rp, value_ry;
    static int value_tx, value_ty, value_tz;

    static bool init(ros::NodeHandle &n_, std::string dataset_folder, std::string camera_name) {
        Dataset::dataset_folder = dataset_folder;

        auto gt_dir_path = boost::filesystem::path(Dataset::dataset_folder);
        gt_dir_path /= camera_name;
        gt_dir_path /= "ground_truth";
        Dataset::gt_folder = gt_dir_path.string();

        bool ret = Dataset::read_params(Dataset::dataset_folder + "/" + camera_name + "/params.txt");
        ret &= Dataset::read_cam_intr(Dataset::dataset_folder + "/" + camera_name + "/calib.txt");
        ret &= Dataset::read_extr(Dataset::dataset_folder + "/" + camera_name + "/extrinsics.txt");
        ret &= Dataset::load_objects(n_, Dataset::dataset_folder + "/objects.txt");
        return ret;
    }

    static void init_GUI() {
        Dataset::window_name = "Calibration Control";
        cv::namedWindow(Dataset::window_name, cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("R", Dataset::window_name, &value_rr, MAXVAL, on_trackbar);
        cv::createTrackbar("P", Dataset::window_name, &value_rp, MAXVAL, on_trackbar);
        cv::createTrackbar("Y", Dataset::window_name, &value_ry, MAXVAL, on_trackbar);
        cv::createTrackbar("x", Dataset::window_name, &value_tx, MAXVAL, on_trackbar);
        cv::createTrackbar("y", Dataset::window_name, &value_ty, MAXVAL, on_trackbar);
        cv::createTrackbar("z", Dataset::window_name, &value_tz, MAXVAL, on_trackbar);
        cv::createTrackbar("t_pos", Dataset::window_name, &pose_to_event_to_slider, MAXVAL, on_trackbar);
        cv::createTrackbar("t_img", Dataset::window_name, &image_to_event_to_slider, MAXVAL, on_trackbar);
    }

    static void reset_Intr_Sliders() {
        cv::setTrackbarPos("R", Dataset::window_name, MAXVAL / 2);
        cv::setTrackbarPos("P", Dataset::window_name, MAXVAL / 2);
        cv::setTrackbarPos("Y", Dataset::window_name, MAXVAL / 2);
        cv::setTrackbarPos("x", Dataset::window_name, MAXVAL / 2);
        cv::setTrackbarPos("y", Dataset::window_name, MAXVAL / 2);
        cv::setTrackbarPos("z", Dataset::window_name, MAXVAL / 2);
    }

    static void apply_Intr_Calib() {
        auto pose = Pose(ros::Time(0), Dataset::cam_E);
        auto T = pose.getT();
        auto R = pose.getR();

        tx0 = T[0]; ty0 = T[1]; tz0 = T[2];
        rr0 = R[0]; rp0 = R[1]; ry0 = R[2];

        Dataset::reset_Intr_Sliders();
        Dataset::printCalib();
    }

    static void set_sliders(float Tx, float Ty, float Tz,
                            float Rx, float Ry, float Rz) {
        Dataset::modified = true;

        value_rr = normval_inv(normval(value_rr, MAXVAL, MAXVAL * INT_ANG_SC) + Rx,
                               MAXVAL, MAXVAL * INT_ANG_SC);
        value_rp = normval_inv(normval(value_rp, MAXVAL, MAXVAL * INT_ANG_SC) + Ry,
                               MAXVAL, MAXVAL * INT_ANG_SC);
        value_ry = normval_inv(normval(value_ry, MAXVAL, MAXVAL * INT_ANG_SC) + Rz,
                               MAXVAL, MAXVAL * INT_ANG_SC);

        value_tx = normval_inv(normval(value_tx, MAXVAL, MAXVAL * INT_LIN_SC) + Tx,
                               MAXVAL, MAXVAL * INT_LIN_SC);
        value_ty = normval_inv(normval(value_ty, MAXVAL, MAXVAL * INT_LIN_SC) + Ty,
                               MAXVAL, MAXVAL * INT_LIN_SC);
        value_tz = normval_inv(normval(value_tz, MAXVAL, MAXVAL * INT_LIN_SC) + Tz,
                               MAXVAL, MAXVAL * INT_LIN_SC);

        cv::setTrackbarPos("R", Dataset::window_name, value_rr);
        cv::setTrackbarPos("P", Dataset::window_name, value_rp);
        cv::setTrackbarPos("Y", Dataset::window_name, value_ry);
        cv::setTrackbarPos("x", Dataset::window_name, value_tx);
        cv::setTrackbarPos("y", Dataset::window_name, value_ty);
        cv::setTrackbarPos("z", Dataset::window_name, value_tz);

        Dataset::update_cam_calib();
    }

    static void handle_keys(int code, uint8_t &vis_mode, const uint8_t nmodes) {
        if (code == 32) {
            vis_mode = (vis_mode + 1) % nmodes;
            Dataset::modified = true;
        }

        if (code == 49) { // '1'
            vis_mode = 0;
            Dataset::modified = true;
        }

        if (code == 50) { // '2'
            vis_mode = 1;
            Dataset::modified = true;
        }

        if (code == 51) { // '3'
            vis_mode = 2;
            Dataset::modified = true;
        }

        if (code == 52) { // '4'
            vis_mode = 3;
            Dataset::modified = true;
        }

        if (code == 91) { // '['
            Dataset::slice_width = std::max(0.0, Dataset::slice_width - 0.005);
            Dataset::modified = true;
        }

        if (code == 93) { // ']'
            Dataset::slice_width += 0.005;
            Dataset::modified = true;
        }

        if (code == 111) { // 'o'
            Dataset::pose_filtering_window = std::max(0.0, Dataset::pose_filtering_window - 0.01);
            Dataset::modified = true;
        }

        if (code == 112) { // 'p'
            Dataset::pose_filtering_window += 0.01;
            Dataset::modified = true;
        }

        if (code == 99) { // 'c'
            Dataset::reset_Intr_Sliders();
            Dataset::modified = true;
        }

        if (code == 115) { // 's'
            Dataset::apply_Intr_Calib();
            Dataset::modified = true;
        }
    }

    static void printCalib() {
        std::cout << std::endl << _blue("Transforms:") << std::endl;
        std::cout << "Vicon -> Camcenter (X Y Z R P Y):" << std::endl;
        std::cout << "\t" << tx0 << "\t" << ty0 << "\t" << tz0 << "\t" << rr0 << "\t" << rp0 << "\t" << ry0 << std::endl;
        //std::cout << "Vicon -> Background (X Y Z Qw Qx Qy Qz):" << std::endl;
        //auto T = room_scan->get_static().getOrigin();
        //auto Q = room_scan->get_static().getRotation();
        //std::cout << "\t" << T.getX() << "\t" << T.getY() << "\t" << T.getZ()
        //          << "\t" << Q.getW() <<"\t" << Q.getX() << "\t" << Q.getY() << "\t" << Q.getZ() << std::endl << std::endl;
        std::cout << "time offset pose to events:  " << get_time_offset_pose_to_event() << std::endl;
        std::cout << "time offset image to events: " << get_time_offset_image_to_event() << std::endl;
    }

    static void create_ground_truth_folder() {
        auto gt_dir_path = boost::filesystem::path(Dataset::gt_folder);
        std::cout << _blue("Removing old: " + gt_dir_path.string()) << std::endl;
        boost::filesystem::remove_all(gt_dir_path);
        std::cout << "Creating: " << _green(gt_dir_path.string()) << std::endl;
        boost::filesystem::create_directory(gt_dir_path);
    }

    static void write_eventstxt(std::string efname) {
        std::cout << std::endl << _yellow("Writing events.txt") << std::endl;
        std::stringstream ss;
        for (uint64_t i = 0; i < event_array.size(); ++i) {
            if (i % 10000 == 0 || i == event_array.size() - 1) {
                std::cout << "\tPreparing\t" << i + 1 << "\t/\t" << event_array.size() << "\t\r" << std::flush;
            }

            ss << std::fixed << std::setprecision(9)
               << event_array[i].get_ts_sec()
               << " " << event_array[i].fr_y << " " << event_array[i].fr_x
               << " " << int(event_array[i].polarity) << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl << _yellow("Writing to file...") << std::endl;
        std::ofstream event_file(efname, std::ofstream::out);
        event_file << ss.str();
        event_file.close();
    }

    static std::string meta_as_dict() {
        return "'meta': {'fx': " + std::to_string(Dataset::fx)
                    + ", 'fy': " + std::to_string(Dataset::fy)
                    + ", 'cx': " + std::to_string(Dataset::cx)
                    + ", 'cy': " + std::to_string(Dataset::cy)
                    + ", 'k1': " + std::to_string(Dataset::k1)
                    + ", 'k2': " + std::to_string(Dataset::k2)
                    + ", 'k3': " + std::to_string(Dataset::k3)
                    + ", 'k4': " + std::to_string(Dataset::k4)
                    + ", 'p1': " + std::to_string(Dataset::p1)
                    + ", 'p2': " + std::to_string(Dataset::p2)
                    + ", 'res_x': " + std::to_string(Dataset::res_y) // FIXME!
                    + ", 'res_y': " + std::to_string(Dataset::res_x)
                    + ", 'dist_model': '" + Dataset::dist_model + "'"
                    + "}";
    }

    // Time offset getters
    static float get_time_offset_image_to_host() {
        return 0.0;
    }

    static float get_time_offset_image_to_host_correction() {
        return 0.0;
    }

    static float get_time_offset_pose_to_host() {
        return get_time_offset_event_to_host() + get_time_offset_pose_to_event();
    }

    static float get_time_offset_pose_to_host_correction() {
        return get_time_offset_event_to_host_correction() + get_time_offset_pose_to_event_correction();
    }

    static float get_time_offset_event_to_host() {
        return get_time_offset_image_to_host() - get_time_offset_image_to_event();
    }

    static float get_time_offset_event_to_host_correction() {
        return get_time_offset_image_to_host_correction() - get_time_offset_image_to_event_correction();
    }

private:
    // slider-controlled:
    static float get_time_offset_image_to_event() {
        return Dataset::image_to_event_to + get_time_offset_image_to_event_correction();
    }

    static float get_time_offset_image_to_event_correction() {
        return normval(image_to_event_to_slider, MAXVAL, MAXVAL * INT_TIM_SC);
    }

    static float get_time_offset_pose_to_event() {
        return Dataset::pose_to_event_to + get_time_offset_pose_to_event_correction();
    }

    static float get_time_offset_pose_to_event_correction() {
        return normval(pose_to_event_to_slider, MAXVAL, MAXVAL * INT_TIM_SC);
    }

private:
    static bool read_params (std::string path) {
        std::ifstream ifs;
        ifs.open(path, std::ifstream::in);
        if (!ifs.is_open()) {
            std::cout << _red("Could not open camera parameter file file at ")
                      << path << "!" << std::endl;
            return false;
        }

        const std::string& delims = ":";
        while (ifs.good()) {
            std::string line;
            std::getline(ifs, line);
            line = trim(line);
            auto sep = line.find_first_of(delims);

            std::string key   = line.substr(0, sep);
            std::string value = line.substr(sep + 1);
            key = trim(key);
            value = trim(value);

            if (key == "res_x")
                Dataset::res_x = std::stoi(value);

            if (key == "res_y")
                Dataset::res_y = std::stoi(value);

            if (key == "dist_model")
                Dataset::dist_model = value;

            if (key == "ros_image_topic")
                Dataset::image_topic = value;

            if (key == "ros_event_topic")
                Dataset::event_topic = value;

            if (key == "ros_pos_topic")
                Dataset::cam_pos_topic = value;
        }
        ifs.close();

        std::cout << _green("Read camera parameters: \n")
                  << "\tres:\t" << Dataset::res_y << " x " << Dataset::res_x << "\n"
                  << "\tdistortion model:\t" << Dataset::dist_model << "\n"
                  << "\tros image topic:\t" << Dataset::image_topic << "\n"
                  << "\tros event topic:\t" << Dataset::event_topic << "\n"
                  << "\tros camera pose topic:\t" << Dataset::cam_pos_topic << "\n";
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

        k1 = k2 = k3 = k4 = p1 = p2 = 0;

        if (Dataset::dist_model == "radtan") {
            ifs >> k1 >> k2 >> p1 >> p2;
        } else if (Dataset::dist_model == "equidistant") {
            ifs >> k1 >> k2 >> k3 >> k4;
        } else {
            std::cout << _red("Unknown distortion model! ") << Dataset::dist_model << std::endl;
        }

        std::cout << _green("Read camera calibration: (fx fy cx cy {k1 k2 k3 k4} {p1 p2}): ")
                  << fx << " " << fy << " " << cx << " " << cy << " "
                  << k1 << " " << k2 << " " << k3 << " " << k4 << " "
                  << p1 << " " << p2 << std::endl;
        ifs.close();
        Dataset::update_cam_calib();
        return true;
    }

    static void on_trackbar(int, void*) {
        Dataset::modified = true;
        Dataset::update_cam_calib();
    }

    static float normval(int val, int maxval, int normval) {
        return float(val - maxval / 2) / float(normval);
    }

    static float normval_inv(float val, int maxval, int normval) {
        return val * float(normval) + float(maxval / 2);
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

        ifs >> pose_to_event_to;
        if (!ifs.good()) {
            Dataset::pose_to_event_to = 0;
            std::cout << _yellow("Time offset (pos) is not specified;") << " setting to " << Dataset::pose_to_event_to << std::endl;
        }

        ifs >> image_to_event_to;
        if (!ifs.good()) {
            Dataset::image_to_event_to = 0;
            std::cout << _yellow("Time offset (img) is not specified;") << " setting to " << Dataset::image_to_event_to << std::endl;
        }

        ifs.close();

        tf::Vector3 T;
        tf::Quaternion Q(bg_qx, bg_qy, bg_qz, bg_qw);
        T.setValue(bg_tx, bg_ty, bg_tz);

        bg_E.setRotation(Q);
        bg_E.setOrigin(T);

        // Old extrinsic format
        bool old_ext_format = false;
        if (old_ext_format) {
            Eigen::Matrix4f T1;
            T1 <<  0.0,   -1.0,   0.0,  0.00,
                   1.0,    0.0,   0.0,  0.00,
                   0.0,    0.0,   1.0,  0.00,
                     0,      0,     0,     1;

            Eigen::Matrix4f T2;
            T2 <<  0.0,    0.0,  -1.0,  0.00,
                   0.0,    1.0,   0.0,  0.00,
                   1.0,    0.0,   0.0,  0.00,
                     0,      0,     0,     1;

            tf::Transform E_;
            tf::Vector3 T_(tx0, ty0, tz0);
            tf::Quaternion q_;
            q_.setRPY(rr0, rp0, ry0);
            E_.setRotation(q_);
            E_.setOrigin(T_);

            Dataset::cam_E = ViObject::mat2tf(T1) * E_ * ViObject::mat2tf(T2);

            auto pose = Pose(ros::Time(0), Dataset::cam_E);
            auto T = pose.getT();
            auto R = pose.getR();

            tx0 = T[0]; ty0 = T[1]; tz0 = T[2];
            rr0 = R[0]; rp0 = R[1]; ry0 = R[2];
        }

        return true;
    }

    static bool load_objects(ros::NodeHandle &nh_, std::string path) {
        std::string path_to_self = ros::package::getPath("evimo");

        std::ifstream ifs;
        ifs.open(path, std::ifstream::in);
        if (!ifs.is_open()) {
            std::cout << _red("Could not open object configuration file file at ")
                      << path << "!" << std::endl;
            return false;
        }

        const std::string& delims = "#";
        while (ifs.good()) {
            std::string line;
            std::getline(ifs, line);
            auto sep = line.find_first_of(delims);

            std::string object_name = line.substr(0, sep);
            object_name = trim(object_name);
            if (object_name == "") continue;

            auto object_ = std::make_shared<ViObject>(nh_, path_to_self + "/" + object_name);
            if (object_->get_id() < 0) return false;

            Dataset::clouds[object_->get_id()] = object_;
            Dataset::obj_pose_topics[object_->get_id()] = object_->get_pose_topic();
        }
        ifs.close();
        return true;
    }

public:
    static void update_cam_calib() {
        tf::Transform E_;
        tf::Vector3 T_;
        tf::Quaternion q_;
        q_.setRPY(normval(value_rr, MAXVAL, MAXVAL * INT_ANG_SC),
                  normval(value_rp, MAXVAL, MAXVAL * INT_ANG_SC),
                  normval(value_ry, MAXVAL, MAXVAL * INT_ANG_SC));
        T_.setValue(normval(value_tx, MAXVAL, MAXVAL * INT_LIN_SC),
                    normval(value_ty, MAXVAL, MAXVAL * INT_LIN_SC),
                    normval(value_tz, MAXVAL, MAXVAL * INT_LIN_SC));
        E_.setRotation(q_);
        E_.setOrigin(T_);

        tf::Transform E0;
        tf::Vector3 T0(tx0, ty0, tz0);
        tf::Quaternion q0;
        q0.setRPY(rr0, rp0, ry0);
        E0.setRotation(q0);
        E0.setOrigin(T0);

        Dataset::cam_E = E0 * E_;
    }

    // Project 3D point in camera frame to pixel
    template<class T> static void project_point(T p, int &u, int &v) {
        if (Dataset::dist_model == "radtan") {
            Dataset::project_point_radtan(p, u, v);
        } else if (Dataset::dist_model == "equidistant") {
            Dataset::project_point_equi(p, u, v);
        } else {
            std::cout << _red("Unknown distortion model! ") << Dataset::dist_model << std::endl;
        }
    }

    template<class T> static void project_point_radtan(T p, int &u, int &v) {
        u = -1; v = -1;
        if (p.z < 0.001)
            return;

        float x_ = p.x / p.z;
        float y_ = p.y / p.z;

        float rng_th = std::max(Dataset::res_x, Dataset::res_y);
        float v__ = Dataset::fx * x_ + Dataset::cx;
        float u__ = Dataset::fy * y_ + Dataset::cy;
        if ((v__ > rng_th * 1.2) || (v__ < -rng_th * 0.2)) return;
        if ((u__ > rng_th * 1.2) || (u__ < -rng_th * 0.2)) return;

        float r2 = x_ * x_ + y_ * y_;
        float r4 = r2 * r2;
        float r6 = r2 * r2 * r2;
        float dist = (1.0 + Dataset::k1 * r2 + Dataset::k2 * r4 +
                            Dataset::k3 * r6) / (1 + Dataset::k4 * r2);
        float x__ = x_ * dist + 2.0 * Dataset::p1 * x_ * y_ + Dataset::p2 * (r2 + 2.0 * x_ * x_);
        float y__ = y_ * dist + 2.0 * Dataset::p2 * x_ * y_ + Dataset::p1 * (r2 + 2.0 * y_ * y_);

        v = Dataset::fx * x__ + Dataset::cx;
        u = Dataset::fy * y__ + Dataset::cy;
    }

    template<class T> static void project_point_equi(T p, int &u, int &v) {
        u = -1; v = -1;
        if (p.z < 0.001)
            return;

        float x_ = p.x / p.z;
        float y_ = p.y / p.z;
        float r = std::sqrt(x_ * x_ + y_ * y_);
        float th = std::atan(r);

        float th2 = th * th;
        float th4 = th2 * th2;
        float th6 = th2 * th2 * th2;
        float th8 = th4 * th4;
        float th_d = th * (1 + Dataset::k1 * th2 + Dataset::k2 * th4 + Dataset::k3 * th6 + Dataset::k4 * th8);

        float x__ = x_;
        float y__ = y_;
        if (r > 0.001) {
            x__ = (th_d / r) * x_;
            y__ = (th_d / r) * y_;
        }

        v = Dataset::fx * x__ + Dataset::cx;
        u = Dataset::fy * y__ + Dataset::cy;
    }

    static cv::Mat undistort(cv::Mat &img) {
        cv::Mat ret;
        cv::Mat K = (cv::Mat1d(3, 3) << Dataset::fx, 0, Dataset::cx, 0, Dataset::fy, Dataset::cy, 0, 0, 1);

        if (Dataset::dist_model == "radtan") {
            cv::Mat D = (cv::Mat1d(1, 4) << Dataset::k1, Dataset::k2, Dataset::p1, Dataset::p2);
            cv::undistort(img, ret, K, D, 1.0 * K);
        } else if (Dataset::dist_model == "equidistant") {
            cv::Mat D = (cv::Mat1d(1, 4) << Dataset::k1, Dataset::k2, Dataset::k3, Dataset::k4);
            cv::fisheye::undistortImage(img, ret, K, D, 1.0 * K);
        } else {
            std::cout << _red("Unknown distortion model! ") << Dataset::dist_model << std::endl;
        }

        return ret;
    }
};


#endif // DATASET_H
