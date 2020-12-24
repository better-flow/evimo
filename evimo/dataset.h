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
    std::shared_ptr<StaticObject> background;
    std::map<int, std::shared_ptr<ViObject>> clouds;
    std::map<int, std::string> obj_pose_topics;
    std::map<std::string, ImuInfo> imu_info;

    // Time offset from data to ros time
    ros::Time ros_time_offset; 

    // Event cloud
    std::vector<Event> event_array;

    // Camera frames
    std::vector<cv::Mat> images;
    std::vector<ros::Time> image_ts;

    // Trajectories for camera and objects
    Trajectory cam_tj;
    std::map<int, Trajectory> obj_tjs;

    // Camera params
    std::string dist_model;
    std::string image_topic, event_topic, cam_pos_topic;

    // Calibration matrix
    float fx, fy, cx, cy, k1, k2, k3, k4, p1, p2;

    // Camera resolution
    unsigned int res_x, res_y;

    // Camera center to vicon
    float rr0, rp0, ry0, tx0, ty0, tz0;
    tf::Transform cam_E;

    // Background to vicon
    tf::Transform bg_E;

    // Time offset
    float image_to_event_to, pose_to_event_to;
    int image_to_event_to_slider, pose_to_event_to_slider;

    // Event slice width, for visualization
    float slice_width;

    // Pose filtering window, in seconds
    float pose_filtering_window;

    // Instance counter for this class
    static uint32_t instance_id;

    // Other parameters
    std::string window_name;
    bool modified;
    bool window_initialized;

    // Folder names
    std::string dataset_folder, camera_name, gt_folder;

    static constexpr float MAXVAL = 1000;
    static constexpr float INT_LIN_SC = 1.0;
    static constexpr float INT_ANG_SC = 1.0;
    static constexpr float INT_TIM_SC = 1.0;

    int value_rr, value_rp, value_ry;
    int value_tx, value_ty, value_tz;

    // Constructor
    Dataset()
        : dist_model(""), image_topic(""), event_topic(""), cam_pos_topic("")
        , fx(0), fy(0), cx(0), cy(0), k1(0), k2(0), k3(0), k4(0), p1(0), p2(0)
        , res_x(0), res_y(0)
        , rr0(0), rp0(0), ry0(0), tx0(0), ty0(0), tz0(0) 
        , image_to_event_to(0), pose_to_event_to(0)
        , image_to_event_to_slider(Dataset::MAXVAL / 2), pose_to_event_to_slider(Dataset::MAXVAL / 2)
        , value_rr(Dataset::MAXVAL / 2), value_rp(Dataset::MAXVAL / 2), value_ry(Dataset::MAXVAL / 2)
        , value_tx(Dataset::MAXVAL / 2), value_ty(Dataset::MAXVAL / 2), value_tz(Dataset::MAXVAL / 2)
        , slice_width(0.04)
        , pose_filtering_window(-1) // pose filtering window, in seconds
        , window_name("")
        , modified(false), window_initialized(false)
        , dataset_folder(""), camera_name(""), gt_folder("") {
        this->cam_E.setIdentity();
        Dataset::instance_id++;
    }

    bool read_bag_file(std::string bag_name,
                       float start_time_offset = 0.0, float sequence_duration = -1.0,
                       bool with_images=true, bool ignore_tj=false);

    bool init(ros::NodeHandle &n_, std::string dataset_folder, std::string camera_name) {
        bool ret = this->init_no_objects(dataset_folder, camera_name);
        ret &= this->load_objects(n_, this->dataset_folder + "/objects.txt");
        return ret;
    }

    bool init_no_objects(std::string dataset_folder, std::string camera_name) {
        this->dataset_folder = dataset_folder;
        this->camera_name = camera_name;

        auto gt_dir_path = boost::filesystem::path(this->dataset_folder);
        gt_dir_path /= camera_name;
        gt_dir_path /= "ground_truth";
        this->gt_folder = gt_dir_path.string();

        bool ret = this->read_params(this->dataset_folder + "/" + camera_name + "/params.txt");
        ret &= this->read_cam_intr(this->dataset_folder + "/" + camera_name + "/calib.txt");
        ret &= this->read_extr(this->dataset_folder + "/" + camera_name + "/extrinsics.txt");
        return ret;
    }

    void init_GUI() {
        this->window_name = "Calibration Control " + std::to_string(Dataset::instance_id - 1);
        cv::namedWindow("Trajectories", cv::WINDOW_AUTOSIZE);
        cv::namedWindow(this->window_name, cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("R", this->window_name, &this->value_rr, Dataset::MAXVAL, Dataset::on_trackbar, this);
        cv::createTrackbar("P", this->window_name, &this->value_rp, Dataset::MAXVAL, Dataset::on_trackbar, this);
        cv::createTrackbar("Y", this->window_name, &this->value_ry, Dataset::MAXVAL, Dataset::on_trackbar, this);
        cv::createTrackbar("x", this->window_name, &this->value_tx, Dataset::MAXVAL, Dataset::on_trackbar, this);
        cv::createTrackbar("y", this->window_name, &this->value_ty, Dataset::MAXVAL, Dataset::on_trackbar, this);
        cv::createTrackbar("z", this->window_name, &this->value_tz, Dataset::MAXVAL, Dataset::on_trackbar, this);
        cv::createTrackbar("t_pos", this->window_name, &this->pose_to_event_to_slider, Dataset::MAXVAL, Dataset::on_trackbar, this);
        cv::createTrackbar("t_img", this->window_name, &this->image_to_event_to_slider, Dataset::MAXVAL, Dataset::on_trackbar, this);
        this->window_initialized = true;
    }

    void reset_Intr_Sliders() {
        if(!this->window_initialized) return;
        cv::setTrackbarPos("R", this->window_name, Dataset::MAXVAL / 2);
        cv::setTrackbarPos("P", this->window_name, Dataset::MAXVAL / 2);
	    cv::setTrackbarPos("Y", this->window_name, Dataset::MAXVAL / 2);
	    cv::setTrackbarPos("x", this->window_name, Dataset::MAXVAL / 2);
	    cv::setTrackbarPos("y", this->window_name, Dataset::MAXVAL / 2);
	    cv::setTrackbarPos("z", this->window_name, Dataset::MAXVAL / 2);
    }

    void apply_Intr_Calib() {
        auto pose = Pose(ros::Time(0), this->cam_E);
        auto T = pose.getT();
        auto R = pose.getR();

        this->tx0 = T[0]; this->ty0 = T[1]; this->tz0 = T[2];
        this->rr0 = R[0]; this->rp0 = R[1]; this->ry0 = R[2];

        this->reset_Intr_Sliders();
        this->printCalib();
        bool ret = this->write_extr(this->dataset_folder + "/" + camera_name + "/extrinsics.txt");
        if (!ret) {
            std::cout << _red("apply_Intr_Calib failed to save calibration") << std::endl;
        }
    }

    void set_sliders(float Tx, float Ty, float Tz,
                     float Rx, float Ry, float Rz) {
        this->modified = true;
        this->value_rr = Dataset::normval_inv(Dataset::normval(
                this->value_rr, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_ANG_SC) + Rx,
                Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_ANG_SC);
        this->value_rp = Dataset::normval_inv(Dataset::normval(
                this->value_rp, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_ANG_SC) + Ry,
                Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_ANG_SC);
        this->value_ry = Dataset::normval_inv(Dataset::normval(
                this->value_ry, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_ANG_SC) + Rz,
                Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_ANG_SC);
        this->value_tx = Dataset::normval_inv(Dataset::normval(
                this->value_tx, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_LIN_SC) + Tx,
                Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_LIN_SC);
        this->value_ty = Dataset::normval_inv(Dataset::normval(
                this->value_ty, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_LIN_SC) + Ty,
                Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_LIN_SC);
        this->value_tz = Dataset::normval_inv(Dataset::normval(
                this->value_tz, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_LIN_SC) + Tz,
                Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_LIN_SC);

        cv::setTrackbarPos("R", this->window_name, this->value_rr);
        cv::setTrackbarPos("P", this->window_name, this->value_rp);
        cv::setTrackbarPos("Y", this->window_name, this->value_ry);
        cv::setTrackbarPos("x", this->window_name, this->value_tx);
        cv::setTrackbarPos("y", this->window_name, this->value_ty);
        cv::setTrackbarPos("z", this->window_name, this->value_tz);

        this->update_cam_calib();
    }

    void handle_keys(int code, uint8_t &vis_mode, const uint8_t nmodes) {
        if (code == 32) {
            vis_mode = (vis_mode + 1) % nmodes;
            this->modified = true;
        }

        if (code == 49) { // '1'
            vis_mode = 0;
            this->modified = true;
        }

        if (code == 50) { // '2'
            vis_mode = 1;
            this->modified = true;
        }

        if (code == 51) { // '3'
            vis_mode = 2;
            this->modified = true;
        }

        if (code == 52) { // '4'
            vis_mode = 3;
            this->modified = true;
        }

        if (code == 91) { // '['
            this->slice_width = std::max(0.0, this->slice_width - 0.002);
            this->modified = true;
        }

        if (code == 93) { // ']'
            this->slice_width += 0.002;
            this->modified = true;
        }

        if (code == 111) { // 'o'
            this->pose_filtering_window = std::max(0.0, this->pose_filtering_window - 0.01);
            this->modified = true;
        }

        if (code == 112) { // 'p'
            this->pose_filtering_window += 0.01;
            this->modified = true;
        }

        if (code == 99) { // 'c'
            this->reset_Intr_Sliders();
            this->modified = true;
        }

        if (code == 115) { // 's'
            this->apply_Intr_Calib();
            this->modified = true;
        }
    }

    void printCalib() {
        std::cout << std::endl << _blue("Intrinsics ") << "(" << this->dist_model << "):" << std::endl;
        std::cout << this->fx << " " << this->fy << " " << this->cx << " " << this->cy << " ";
        if (this->dist_model == "radtan") {
            std::cout << this->k1 << " " << this->k2 << " " << this->p1 << " " << this->p2 << std::endl;
        } else if (this->dist_model == "equidistant") {
            std::cout << this->k1 << " " << this->k2 << " " << this->k3 << " " << this->k4 << std::endl;
        } else {
            std::cout << _red("Unknown distortion model! ") << this->dist_model << std::endl;
        }
        std::cout << std::endl << _blue("Transforms:") << std::endl;
        std::cout << "Vicon -> Camcenter (X Y Z R P Y):" << std::endl;
        std::cout << "\t" << this->tx0 << "\t" << this->ty0 << "\t" << this->tz0 
                  << "\t" << this->rr0 << "\t" << this->rp0 << "\t" << this->ry0 << std::endl;
        //std::cout << "Vicon -> Background (X Y Z Qw Qx Qy Qz):" << std::endl;
        //auto T = room_scan->get_static().getOrigin();
        //auto Q = room_scan->get_static().getRotation();
        //std::cout << "\t" << T.getX() << "\t" << T.getY() << "\t" << T.getZ()
        //          << "\t" << Q.getW() <<"\t" << Q.getX() << "\t" << Q.getY() << "\t" << Q.getZ() << std::endl << std::endl;
        std::cout << "time offset pose to events:  " << this->get_time_offset_pose_to_event() << std::endl;
        std::cout << "time offset image to events: " << this->get_time_offset_image_to_event() << std::endl;
    }

    void create_ground_truth_folder(std::string folder="") {
        if (folder == "") folder = this->gt_folder;
        auto gt_dir_path = boost::filesystem::path(folder);
        std::cout << _blue("Removing old: " + gt_dir_path.string()) << std::endl;
        boost::filesystem::remove_all(gt_dir_path);
        std::cout << "Creating: " << _green(gt_dir_path.string()) << std::endl;
        boost::filesystem::create_directory(gt_dir_path);
    }

    void write_eventstxt(std::string efname) {
        std::cout << std::endl << _yellow("Writing events.txt") << std::endl;
        std::stringstream ss;
        for (uint64_t i = 0; i < this->event_array.size(); ++i) {
            if (i % 10000 == 0 || i == this->event_array.size() - 1) {
                std::cout << "\tPreparing\t" << i + 1 << "\t/\t" 
                          << this->event_array.size() << "\t\r" << std::flush;
            }

            ss << std::fixed << std::setprecision(9)
               << this->event_array[i].get_ts_sec()
               << " " << this->event_array[i].fr_y << " " << this->event_array[i].fr_x
               << " " << int(this->event_array[i].polarity) << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl << _yellow("Writing to file...") << std::endl;
        std::ofstream event_file(efname, std::ofstream::out);
        event_file << ss.str();
        event_file.close();
    }

    std::string meta_as_dict() {
        return "'meta': {'fx': " + std::to_string(this->fx)
                    + ", 'fy': " + std::to_string(this->fy)
                    + ", 'cx': " + std::to_string(this->cx)
                    + ", 'cy': " + std::to_string(this->cy)
                    + ", 'k1': " + std::to_string(this->k1)
                    + ", 'k2': " + std::to_string(this->k2)
                    + ", 'k3': " + std::to_string(this->k3)
                    + ", 'k4': " + std::to_string(this->k4)
                    + ", 'p1': " + std::to_string(this->p1)
                    + ", 'p2': " + std::to_string(this->p2)
                    + ", 'res_x': " + std::to_string(this->res_y) // FIXME!
                    + ", 'res_y': " + std::to_string(this->res_x)
                    + ", 'dist_model': '" + this->dist_model + "'"
                    + ", 'ros_time_offset': " + std::to_string(this->ros_time_offset.toSec())
                    + "}";
    }

    // Time offset getters
    float get_time_offset_image_to_host() {
        return 0.0;
    }

    float get_time_offset_image_to_host_correction() {
        return 0.0;
    }

    float get_time_offset_pose_to_host() {
        return this->get_time_offset_event_to_host() + this->get_time_offset_pose_to_event();
    }

    float get_time_offset_pose_to_host_correction() {
        return this->get_time_offset_event_to_host_correction() + this->get_time_offset_pose_to_event_correction();
    }

    float get_time_offset_event_to_host() {
        return this->get_time_offset_image_to_host() - this->get_time_offset_image_to_event();
    }

    float get_time_offset_event_to_host_correction() {
        return this->get_time_offset_image_to_host_correction() - this->get_time_offset_image_to_event_correction();
    }

    static void on_trackbar(int v, void *object) {
        Dataset *instance = (Dataset*)object;
        instance->modified = true;
        instance->update_cam_calib();
    }

private:
    // slider-controlled:
    float get_time_offset_image_to_event() {
        return this->image_to_event_to + this->get_time_offset_image_to_event_correction();
    }

    float get_time_offset_image_to_event_correction() {
        return Dataset::normval(this->image_to_event_to_slider, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_TIM_SC);
    }

    float get_time_offset_pose_to_event() {
        return this->pose_to_event_to + get_time_offset_pose_to_event_correction();
    }

    float get_time_offset_pose_to_event_correction() {
        return Dataset::normval(this->pose_to_event_to_slider, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_TIM_SC);
    }

private:
    bool read_params(std::string path) {
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
                this->res_x = std::stoi(value);

            if (key == "res_y")
                this->res_y = std::stoi(value);

            if (key == "dist_model")
                this->dist_model = value;

            if (key == "ros_image_topic")
                this->image_topic = value;

            if (key == "ros_event_topic")
                this->event_topic = value;

            if (key == "ros_pos_topic")
                this->cam_pos_topic = value;
        }
        ifs.close();

        std::cout << _green("Read camera parameters: \n")
                  << "\tres:\t" << this->res_y << " x " << this->res_x << "\n"
                  << "\tdistortion model:\t" << this->dist_model << "\n"
                  << "\tros image topic:\t" << this->image_topic << "\n"
                  << "\tros event topic:\t" << this->event_topic << "\n"
                  << "\tros camera pose topic:\t" << this->cam_pos_topic << "\n";
        return true;
    }

    bool read_cam_intr(std::string path) {
        std::ifstream ifs;
        ifs.open(path, std::ifstream::in);
        if (!ifs.is_open()) {
            std::cout << _red("Could not open camera intrinsic calibration file at ")
                      << path << "!" << std::endl;
            return false;
        }

        ifs >> this->fx >> this->fy >> this->cx >> this->cy;
        if (!ifs.good()) {
            std::cout << _red("Camera calibration read error:") << " Expected a file with a single line, containing "
                      << "fx fy cx cy {k1 k2 k3 k4} ({} are optional)" << std::endl;
            return false;
        }

        this->k1 = this->k2 = this->k3 = this->k4 = this->p1 = this->p2 = 0;

        if (this->dist_model == "radtan") {
            ifs >> this->k1 >> this->k2 >> this->p1 >> this->p2;
        } else if (this->dist_model == "equidistant") {
            ifs >> this->k1 >> this->k2 >> this->k3 >> this->k4;
        } else {
            std::cout << _red("Unknown distortion model! ") << this->dist_model << std::endl;
        }

        std::cout << _green("Read camera calibration: (fx fy cx cy {k1 k2 k3 k4} {p1 p2}): ")
                  << this->fx << " " << this->fy << " " << this->cx << " " << this->cy << " "
                  << this->k1 << " " << this->k2 << " " << this->k3 << " " << this->k4 << " "
                  << this->p1 << " " << this->p2 << std::endl;
        ifs.close();
        this->update_cam_calib();
        return true;
    }

    static float normval(int val, int maxval, int normval) {
        return float(val - maxval / 2) / float(normval);
    }

    static float normval_inv(float val, int maxval, int normval) {
        return val * float(normval) + float(maxval / 2);
    }

    bool write_extr(std::string path) {
        std::ofstream ofs (path, std::ofstream::out);
        if (!ofs.is_open()) {
            std::cout << _red("Could not open camera parameter file file at ")
                      << path << _red(" for writing!") << std::endl;
            return false;
        }

        ofs << this->tx0 << "\t" << this->ty0 << "\t" << this->tz0 
            << "\t" << this->rr0 << "\t" << this->rp0 << "\t" << this->ry0 << std::endl;


        tf::Vector3 T = this->bg_E.getOrigin();
        tf::Quaternion Q(this->bg_E.getRotation());

        if (std::isnan(Q.getW())
            || std::isnan(Q.x())
            || std::isnan(Q.y())
            || std::isnan(Q.z())) {
            ofs << T.x() << "\t" << T.y() << "\t" << T.z() << "\t"
                << 0.0 << "\t" << 0.0 << "\t" << 0.0 << "\t" << 0.0 << std::endl;
        } else {
            ofs << T.x() << "\t" << T.y() << "\t" << T.z() << "\t"
                << Q.getW() << "\t" << Q.x() << "\t" << Q.y() << "\t" << Q.z() << std::endl;
        }

        ofs << this->get_time_offset_pose_to_event() << std::endl;
        ofs << this->get_time_offset_image_to_event() << std::endl;

        ofs.close();
        return true;
    }

    bool read_extr(std::string path) {
        std::ifstream ifs;
        ifs.open(path, std::ifstream::in);
        if (!ifs.is_open()) {
            std::cout << _red("Could not open extrinsic calibration file at ")
                      << path << "!" << std::endl;
            return false;
        }

        ifs >> this->tx0 >> this->ty0 >> this->tz0 >> this->rr0 >> this->rp0 >> this->ry0;
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

        ifs >> this->pose_to_event_to;
        if (!ifs.good()) {
            this->pose_to_event_to = 0;
            std::cout << _yellow("Time offset (pos) is not specified;") << " setting to " << this->pose_to_event_to << std::endl;
        }

        ifs >> this->image_to_event_to;
        if (!ifs.good()) {
            this->image_to_event_to = 0;
            std::cout << _yellow("Time offset (img) is not specified;") << " setting to " << this->image_to_event_to << std::endl;
        }

        ifs.close();

        tf::Vector3 T;
        tf::Quaternion Q(bg_qx, bg_qy, bg_qz, bg_qw);
        T.setValue(bg_tx, bg_ty, bg_tz);

        this->bg_E.setRotation(Q);
        this->bg_E.setOrigin(T);

        this->update_cam_calib();

        // Old extrinsic format (evimo1)
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
            tf::Vector3 T_(this->tx0, this->ty0, this->tz0);
            tf::Quaternion q_;
            q_.setRPY(this->rr0, this->rp0, this->ry0);
            E_.setRotation(q_);
            E_.setOrigin(T_);

            this->cam_E = ViObject::mat2tf(T1) * E_ * ViObject::mat2tf(T2);

            auto pose = Pose(ros::Time(0), this->cam_E);
            auto T = pose.getT();
            auto R = pose.getR();

            this->tx0 = T[0]; this->ty0 = T[1]; this->tz0 = T[2];
            this->rr0 = R[0]; this->rp0 = R[1]; this->ry0 = R[2];
        }

        return true;
    }

    bool load_objects(ros::NodeHandle &nh_, std::string path) {
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

            this->clouds[object_->get_id()] = object_;
            this->obj_pose_topics[object_->get_id()] = object_->get_pose_topic();
        }
        ifs.close();
        return true;
    }

public:
    void update_cam_calib() {
        tf::Transform E_;
        tf::Vector3 T_;
        tf::Quaternion q_;
        q_.setRPY(Dataset::normval(this->value_rr, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_ANG_SC),
                  Dataset::normval(this->value_rp, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_ANG_SC),
                  Dataset::normval(this->value_ry, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_ANG_SC));
        T_.setValue(Dataset::normval(this->value_tx, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_LIN_SC),
                    Dataset::normval(this->value_ty, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_LIN_SC),
                    Dataset::normval(this->value_tz, Dataset::MAXVAL, Dataset::MAXVAL * Dataset::INT_LIN_SC));
        E_.setRotation(q_);
        E_.setOrigin(T_);

        tf::Quaternion q0;
        tf::Vector3 T0(this->tx0, this->ty0, this->tz0);
        q0.setRPY(this->rr0, this->rp0, this->ry0);
        tf::Transform E0;
        E0.setRotation(q0);
        E0.setOrigin(T0);

        this->cam_E = E0 * E_;
    }

    // Project 3D point in camera frame to pixel
    template<class T, class Q> void project_point(T p, Q &u, Q &v) {
        if (this->dist_model == "radtan") {
            this->project_point_radtan(p, u, v);
        } else if (this->dist_model == "equidistant") {
            this->project_point_equi(p, u, v);
        } else {
            std::cout << _red("Unknown distortion model! ") << this->dist_model << std::endl;
        }
    }

    template<class T, class Q> void project_point_nodist(T p, Q &u, Q &v) {
        u = -1; v = -1;
        if (p.z < 0.00001)
            return;

        float x_ = p.x / p.z;
        float y_ = p.y / p.z;

        v = this->fx * x_ + this->cx;
        u = this->fy * y_ + this->cy;
    }

    template<class T, class Q> void project_point_radtan(T p, Q &u, Q &v) {
        u = -1; v = -1;
        if (p.z < 0.001)
            return;

        float x_ = p.x / p.z;
        float y_ = p.y / p.z;

        float rng_th = std::max(this->res_x, this->res_y);
        float v__ = this->fx * x_ + this->cx;
        float u__ = this->fy * y_ + this->cy;
        if ((v__ > rng_th * 1.2) || (v__ < -rng_th * 0.2)) return;
        if ((u__ > rng_th * 1.2) || (u__ < -rng_th * 0.2)) return;

        float r2 = x_ * x_ + y_ * y_;
        float r4 = r2 * r2;
        float r6 = r2 * r2 * r2;
        float dist = (1.0 + this->k1 * r2 + this->k2 * r4 +
                            this->k3 * r6) / (1 + this->k4 * r2);
        float x__ = x_ * dist + 2.0 * this->p1 * x_ * y_ + this->p2 * (r2 + 2.0 * x_ * x_);
        float y__ = y_ * dist + 2.0 * this->p2 * x_ * y_ + this->p1 * (r2 + 2.0 * y_ * y_);

        v = this->fx * x__ + this->cx;
        u = this->fy * y__ + this->cy;
    }

    template<class T, class Q> void project_point_equi(T p, Q &u, Q &v) {
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
        float th_d = th * (1 + this->k1 * th2 + this->k2 * th4 + this->k3 * th6 + this->k4 * th8);

        float x__ = x_;
        float y__ = y_;
        if (r > 0.001) {
            x__ = (th_d / r) * x_;
            y__ = (th_d / r) * y_;
        }

        v = this->fx * x__ + this->cx;
        u = this->fy * y__ + this->cy;
    }

    cv::Mat undistort(cv::Mat &img) {
        if (img.rows == 0 or img.cols == 0) {
            return img;
        }

        cv::Mat ret;
        cv::Mat K = (cv::Mat1d(3, 3) << this->fx, 0, this->cx, 0, this->fy, this->cy, 0, 0, 1);

        if (this->dist_model == "radtan") {
            cv::Mat D = (cv::Mat1d(1, 4) << this->k1, this->k2, this->p1, this->p2);
            cv::undistort(img, ret, K, D, 1.0 * K);
        } else if (this->dist_model == "equidistant") {
            cv::Mat D = (cv::Mat1d(1, 4) << this->k1, this->k2, this->k3, this->k4);
            cv::fisheye::undistortImage(img, ret, K, D, 1.0 * K);
        } else {
            std::cout << _red("Unknown distortion model! ") << this->dist_model << std::endl;
        }

        return ret;
    }
};


#endif // DATASET_H
