#include <algorithm>
#include <iostream>

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

    // Event cloud
    static std::vector<Event> event_array;

    // Camera frames
    static std::vector<cv::Mat> images;
    static std::vector<ros::Time> image_ts;

    // Trajectories for camera and objects
    static Trajectory cam_tj;
    static std::map<int, Trajectory> obj_tjs;

    // Calibration matrix
    static float fx, fy, cx, cy, k1, k2, k3, k4;

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
    static std::map<int, bool> enabled_objects;
    static std::string window_name;
    static bool modified;

    static constexpr float MAXVAL = 1000;
    static constexpr float INT_LIN_SC = 10;
    static constexpr float INT_ANG_SC = 10;
    static constexpr float INT_TIM_SC = 10;

    static int value_rr, value_rp, value_ry;
    static int value_tx, value_ty, value_tz;

    static bool init(std::string dataset_folder, unsigned int rx, unsigned int ry) {
        Dataset::res_x = rx;
        Dataset::res_y = ry;
        bool ret = Dataset::parse_config(dataset_folder + "/config.txt");
        ret &= Dataset::read_cam_intr(dataset_folder + "/calib.txt");
        ret &= Dataset::read_extr(dataset_folder + "/extrinsics.txt");
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
        rr0 = rr0 + normval(value_rr, MAXVAL, MAXVAL * INT_LIN_SC);
        rp0 = rp0 + normval(value_rp, MAXVAL, MAXVAL * INT_LIN_SC);
        ry0 = ry0 + normval(value_ry, MAXVAL, MAXVAL * INT_LIN_SC);
        tx0 = tx0 + normval(value_tx, MAXVAL, MAXVAL * INT_ANG_SC);
        ty0 = ty0 + normval(value_ty, MAXVAL, MAXVAL * INT_ANG_SC);
        tz0 = tz0 + normval(value_tz, MAXVAL, MAXVAL * INT_ANG_SC);
        Dataset::reset_Intr_Sliders();
        Dataset::printCalib();
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
        return true;
    }

public:
    static void update_cam_calib() {
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

        float rr = rr0 + normval(value_rr, MAXVAL, MAXVAL * INT_LIN_SC);
        float rp = rp0 + normval(value_rp, MAXVAL, MAXVAL * INT_LIN_SC);
        float ry = ry0 + normval(value_ry, MAXVAL, MAXVAL * INT_LIN_SC);
        float tx = tx0 + normval(value_tx, MAXVAL, MAXVAL * INT_ANG_SC);
        float ty = ty0 + normval(value_ty, MAXVAL, MAXVAL * INT_ANG_SC);
        float tz = tz0 + normval(value_tz, MAXVAL, MAXVAL * INT_ANG_SC);

        Eigen::Matrix4f E_eig;
        tf::Transform E;
        tf::Vector3 T;
        tf::Quaternion q;
        q.setRPY(rr, rp, ry);
        T.setValue(tx, ty, tz);
        E.setRotation(q);
        E.setOrigin(T);
        pcl_ros::transformAsMatrix(E, E_eig);

        cam_E = ViObject::mat2tf(T1) * E * ViObject::mat2tf(T2);
        //Eigen::Matrix4f full_extr_calib = T1 * E_eig * T2;
        //cam_E = ViObject::mat2tf(full_extr_calib);
    }
};


#endif // DATASET_H
