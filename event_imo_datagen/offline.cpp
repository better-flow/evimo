#include <vector>
#include <valarray>
#include <algorithm>
#include <thread>
#include <type_traits>
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

// DVS / DAVIS
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

// PROPHESEE
#include <prophesee_event_msgs/PropheseeEvent.h>
#include <prophesee_event_msgs/PropheseeEventBuffer.h>

#include "object.h"
#include "event_vis.h"


class DatasetConfig {
public:

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
        DatasetConfig::res_x = rx;
        DatasetConfig::res_y = ry;
        bool ret = DatasetConfig::parse_config(dataset_folder + "/config.txt");
        ret &= DatasetConfig::read_cam_intr(dataset_folder + "/calib.txt");
        ret &= DatasetConfig::read_extr(dataset_folder + "/extrinsics.txt");
        return ret;
    }

    static void init_GUI() {
        DatasetConfig::window_name = "Calibration Control";
        cv::namedWindow(DatasetConfig::window_name, cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("R", DatasetConfig::window_name, &value_rr, MAXVAL, on_trackbar);
        cv::createTrackbar("P", DatasetConfig::window_name, &value_rp, MAXVAL, on_trackbar);
        cv::createTrackbar("Y", DatasetConfig::window_name, &value_ry, MAXVAL, on_trackbar);
        cv::createTrackbar("x", DatasetConfig::window_name, &value_tx, MAXVAL, on_trackbar);
        cv::createTrackbar("y", DatasetConfig::window_name, &value_ty, MAXVAL, on_trackbar);
        cv::createTrackbar("z", DatasetConfig::window_name, &value_tz, MAXVAL, on_trackbar);
        cv::createTrackbar("t_pos", DatasetConfig::window_name, &pose_to_event_to_slider, MAXVAL, on_trackbar);
        cv::createTrackbar("t_img", DatasetConfig::window_name, &image_to_event_to_slider, MAXVAL, on_trackbar);
    }

    static void reset_Intr_Sliders() {
        cv::setTrackbarPos("R", DatasetConfig::window_name, MAXVAL / 2);
        cv::setTrackbarPos("P", DatasetConfig::window_name, MAXVAL / 2);
        cv::setTrackbarPos("Y", DatasetConfig::window_name, MAXVAL / 2);
        cv::setTrackbarPos("x", DatasetConfig::window_name, MAXVAL / 2);
        cv::setTrackbarPos("y", DatasetConfig::window_name, MAXVAL / 2);
        cv::setTrackbarPos("z", DatasetConfig::window_name, MAXVAL / 2);
    }

    static void apply_Intr_Calib() {
        rr0 = rr0 + normval(value_rr, MAXVAL, MAXVAL * INT_LIN_SC);
        rp0 = rp0 + normval(value_rp, MAXVAL, MAXVAL * INT_LIN_SC);
        ry0 = ry0 + normval(value_ry, MAXVAL, MAXVAL * INT_LIN_SC);
        tx0 = tx0 + normval(value_tx, MAXVAL, MAXVAL * INT_ANG_SC);
        ty0 = ty0 + normval(value_ty, MAXVAL, MAXVAL * INT_ANG_SC);
        tz0 = tz0 + normval(value_tz, MAXVAL, MAXVAL * INT_ANG_SC);
        DatasetConfig::reset_Intr_Sliders();
        DatasetConfig::printCalib();
    }

    static void handle_keys(int code, uint8_t &vis_mode, const uint8_t nmodes) {
        if (code == 32) {
            vis_mode = (vis_mode + 1) % nmodes;
            DatasetConfig::modified = true;
        }

        if (code == 49) { // '1'
            vis_mode = 0;
            DatasetConfig::modified = true;
        }

        if (code == 50) { // '2'
            vis_mode = 1;
            DatasetConfig::modified = true;
        }

        if (code == 51) { // '3'
            vis_mode = 2;
            DatasetConfig::modified = true;
        }

        if (code == 52) { // '4'
            vis_mode = 3;
            DatasetConfig::modified = true;
        }

        if (code == 91) { // '['
            DatasetConfig::slice_width = std::max(0.0, DatasetConfig::slice_width - 0.01);
            DatasetConfig::modified = true;
        }

        if (code == 93) { // ']'
            DatasetConfig::slice_width += 0.01;
            DatasetConfig::modified = true;
        }

        if (code == 111) { // 'o'
            DatasetConfig::pose_filtering_window = std::max(0.0, DatasetConfig::pose_filtering_window - 0.01);
            DatasetConfig::modified = true;
        }

        if (code == 112) { // 'p'
            DatasetConfig::pose_filtering_window += 0.01;
            DatasetConfig::modified = true;
        }

        if (code == 99) { // 'c'
            DatasetConfig::reset_Intr_Sliders();
            DatasetConfig::modified = true;
        }

        if (code == 115) { // 's'
            DatasetConfig::apply_Intr_Calib();
            DatasetConfig::modified = true;
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
        return DatasetConfig::image_to_event_to + get_time_offset_image_to_event_correction();
    }

    static float get_time_offset_image_to_event_correction() {
        return normval(image_to_event_to_slider, MAXVAL, MAXVAL * INT_TIM_SC);
    }

    static float get_time_offset_pose_to_event() {
        return DatasetConfig::pose_to_event_to + get_time_offset_pose_to_event_correction();
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
        DatasetConfig::update_cam_calib();
        return true;
    }

    static void on_trackbar(int, void*) {
        DatasetConfig::modified = true;
        DatasetConfig::update_cam_calib();
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
            DatasetConfig::pose_to_event_to = 0;
            std::cout << _yellow("Time offset (pos) is not specified;") << " setting to " << DatasetConfig::pose_to_event_to << std::endl;
        }

        ifs >> image_to_event_to;
        if (!ifs.good()) {
            DatasetConfig::image_to_event_to = 0;
            std::cout << _yellow("Time offset (img) is not specified;") << " setting to " << DatasetConfig::image_to_event_to << std::endl;
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

float DatasetConfig::fx, DatasetConfig::fy, DatasetConfig::cx, DatasetConfig::cy;
unsigned int DatasetConfig::res_x, DatasetConfig::res_y;
float DatasetConfig::k1, DatasetConfig::k2, DatasetConfig::k3, DatasetConfig::k4;
float DatasetConfig::rr0, DatasetConfig::rp0, DatasetConfig::ry0;
float DatasetConfig::tx0, DatasetConfig::ty0, DatasetConfig::tz0;
tf::Transform DatasetConfig::bg_E;
tf::Transform DatasetConfig::cam_E;
float DatasetConfig::slice_width = 0.04;
std::map<int, bool> DatasetConfig::enabled_objects;
std::string DatasetConfig::window_name;
int DatasetConfig::value_rr = MAXVAL / 2, DatasetConfig::value_rp = MAXVAL / 2, DatasetConfig::value_ry = MAXVAL / 2;
int DatasetConfig::value_tx = MAXVAL / 2, DatasetConfig::value_ty = MAXVAL / 2, DatasetConfig::value_tz = MAXVAL / 2;
bool DatasetConfig::modified = true;
float DatasetConfig::pose_filtering_window = 0.04;

// Time offset controls
float DatasetConfig::image_to_event_to, DatasetConfig::pose_to_event_to;
int DatasetConfig::image_to_event_to_slider = MAXVAL / 2, DatasetConfig::pose_to_event_to_slider = MAXVAL / 2;


// Service functions
template <class T> class Slice {
protected:
    T *data;
    std::pair<size_t, size_t> indices;
    size_t current_size;

protected:
    Slice(T &vec)
        : data(&vec), indices(0, 0), current_size(0) {}

    void set_indices(std::pair<size_t, size_t> p) {
        this->indices = p;
        this->current_size = p.second - p.first + 1;

        if (p.first > p.second)
            throw std::string("Attempt to create a Slice with first index bigger than the second! (" +
                              std::to_string(p.first) + " > " + std::to_string(p.second) + ")");
        if (p.second >= this->data->size())
            throw std::string("the second index in Slice is bigger than input vector size! (" +
                              std::to_string(p.second) + " >= " + std::to_string(this->data->size()) + ")");
    }

public:
    typedef T value_type;

    Slice(T &vec, std::pair<uint64_t, uint64_t> p)
        : Slice(vec) {
        this->set_indices(p);
    }

    std::pair<size_t, size_t> get_indices() {
        return this->indices;
    }

    size_t size() {return this->current_size; }
    auto begin()  {return this->data->begin() + this->indices.first; }
    auto end()    {return this->data->begin() + this->indices.second + 1; }
    auto operator [] (size_t idx) {return (*this->data)[idx + this->indices.first]; }
};


template <class T> class TimeSlice : public Slice<T> {
protected:
    std::pair<double, double> time_bounds;
    double get_ts(size_t idx) const {return (this->data->begin() + idx)->get_ts_sec(); }

public:
    TimeSlice(T &vec)
        : Slice<T>(vec) {
        if (this->data->size() == 0)
            throw std::string("TimeSlice: cannot construct on an empty container!");

        this->time_bounds.first  = this->get_ts(0);
        this->time_bounds.second = this->get_ts(this->data->size() - 1);
        this->set_indices(std::pair<uint64_t, uint64_t>(0, this->data->size() - 1));
    }

    TimeSlice(T &vec, std::pair<double, double> p, std::pair<size_t, size_t> hint)
        : Slice<T>(vec), time_bounds(p) {
        std::pair<uint64_t, uint64_t> idx_pair;
        idx_pair.first  = this->find_nearest(this->time_bounds.first,  hint.first);
        idx_pair.second = this->find_nearest(this->time_bounds.second, hint.second);
        this->set_indices(idx_pair);
    }

    TimeSlice(T &vec, std::pair<double, double> p, size_t hint = 0)
        : TimeSlice(vec, p, std::make_pair(hint, hint)) {}

    size_t find_nearest(double ts, size_t hint = 0) const {
        // Assuming data is sorted according to timestamps, in ascending order
        if (this->data->size() == 0)
            throw std::string("find_nearest: data container is empty!");

        if (hint >= this->data->size())
            throw std::string("find_nearest: hint specified is out of bounds!");

        size_t best_idx = hint;
        auto initial_ts = this->get_ts(best_idx);
        double best_error = std::fabs(initial_ts - ts);

        int8_t step = 1;
        if (ts < initial_ts) step = -1;

        int32_t idx = hint;
        while (idx >= 0 && idx < this->data->size() && step * (ts - this->get_ts(idx)) >= 0.0) {
            if (std::fabs(ts - this->get_ts(idx)) < best_error) {
                best_error = std::fabs(ts - this->get_ts(idx));
                best_idx = idx;
            }

            idx += step;
        }

        idx += step;
        if (idx >= 0 && idx < this->data->size() && std::fabs(ts - this->get_ts(idx)) < best_error) {
            best_error = std::fabs(ts - this->get_ts(idx));
            best_idx = idx;
        }

        return best_idx;
    }

    std::pair<double, double> get_time_bounds() {
        return this->time_bounds;
    }
};


class Pose : public SensorMeasurement {
public:
    ros::Time ts;
    tf::Transform pq;
    float occlusion;

    Pose()
        : ts(0), occlusion(std::numeric_limits<double>::quiet_NaN()) {this->pq.setIdentity(); }
    Pose(ros::Time ts_, tf::Transform pq_)
        : ts(ts_), pq(pq_), occlusion(std::numeric_limits<double>::quiet_NaN()) {}
    Pose(ros::Time ts_, const vicon::Subject& p)
        : ts(ts_), occlusion(0) {
        if (p.markers.size() == 0) {
            this->occlusion = 1;
            return;
        }

        this->pq = ViObject::subject2tf(p);
        for (auto &marker : p.markers)
            if (marker.occluded) this->occlusion ++;
        this->occlusion = this->occlusion / float(p.markers.size());
    }

    void setT(std::valarray<float> t) {
        tf::Vector3 T(t[0], t[1], t[2]);
        this->pq.setOrigin(T);
    }

    void setR(std::valarray<float> r) {
        tf::Quaternion q;
        q.setRPY(r[0], r[1], r[2]);
        this->pq.setRotation(q);
    }

    std::valarray<float> getT() {
        tf::Vector3 T = this->pq.getOrigin();
        return {(float)T.getX(), (float)T.getY(), (float)T.getZ()};
    }

    std::valarray<float> getR() {
        tf::Quaternion q = this->pq.getRotation();
        float w = q.getW(), x = q.getX(), y = q.getY(), z = q.getZ();
        float X = std::atan2(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y));
        float sin_val = 2.0f * (w * y - z * x);
        sin_val = (sin_val >  1.0f) ?  1.0f : sin_val;
        sin_val = (sin_val < -1.0f) ? -1.0f : sin_val;
        float Y = std::asin(sin_val);
        float Z = std::atan2(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
        return {X, Y, Z};
    }

    double get_ts_sec() {return this->ts.toSec(); }

    operator tf::Transform() const {return this->pq; }
    friend std::ostream &operator<< (std::ostream &output, const Pose &P) {
        auto loc = P.pq.getOrigin();
        auto rot = P.pq.getRotation();
        output << loc.getX() << " " << loc.getY() << " " << loc.getZ() << " "
               << rot.getW() << " " << rot.getX() << " " << rot.getY() << " "
               << rot.getZ();
        return output;
    }
};


/*! Trajectory class */
class Trajectory {
protected:
    double filtering_window_size; /**< size of trajectory filtering window, in seconds */

private:
    std::vector<Pose> poses; /**< array of poses */

public:
    Trajectory(int32_t window_size = 0)
        : filtering_window_size(window_size) {}

    void set_filtering_window_size(auto window_size) {this->filtering_window_size = window_size; }
    auto get_filtering_window_size() {return this->filtering_window_size; }

    void add(ros::Time ts_, auto pq_) {
        this->poses.push_back(Pose(ts_, pq_));
    }

    size_t size() {return this->poses.size(); }
    auto operator [] (size_t idx) {return this->get_filtered(idx); }

    virtual bool check() {
        if (this->size() == 0) return true;
        auto prev_ts = this->poses[0].ts;
        for (auto &p : this->poses) {
            if (p.ts < prev_ts) return false;
            prev_ts = p.ts;
        }
        return true;
    }

    virtual void subtract_time(ros::Time t) final {
        for (auto &p : this->poses) p.ts = ros::Time((p.ts - t).toSec());
    }

protected:
    auto begin() {return this->poses.begin(); }
    auto end()   {return this->poses.end(); }

    virtual Pose get_filtered(size_t idx) {
        auto central_ts = this->poses[idx].get_ts_sec();
        auto poses_in_window = TimeSlice<Trajectory>(*this,
             std::make_pair(central_ts - this->filtering_window_size / 2.0,
                            central_ts + this->filtering_window_size / 2.0), idx);
        Pose filtered_p;
        filtered_p.ts = ros::Time(central_ts);
        filtered_p.occlusion = this->poses[idx].occlusion;
        auto rot = filtered_p.getR();
        auto tr  = filtered_p.getT();

        for (auto &p : poses_in_window) {
            rot += p.getR();
            tr  += p.getT();
        }

        rot /= float(poses_in_window.size());
        tr  /= float(poses_in_window.size());

        filtered_p.setR(rot);
        filtered_p.setT(tr);

        if (poses_in_window.size() > 1)
            std::cout << "Filtering pose with\t" << poses_in_window.size() << "\tneighbours\n";
        return filtered_p;
    }

    friend class Slice<Trajectory>;
    friend class TimeSlice<Trajectory>;
};



class DatasetFrame {
protected:
    static std::shared_ptr<StaticObject> background;
    static std::map<int, std::shared_ptr<ViObject>> clouds;
    static std::list<DatasetFrame*> visualization_list;
    static std::vector<Event>* event_array;
    static Trajectory *cam_tj;
    static std::map<int, Trajectory> *obj_tjs;

    // Trackbar 0 parameters
    static int trackbar_0_max;
    static int value_trackbar_0;

    // Baseline timestamp
    double timestamp;

    // Thread handle
    std::thread thread_handle;
public:
    uint64_t cam_pose_id;
    std::map<int, uint64_t> obj_pose_ids;
    unsigned long int frame_id;
    std::pair<uint64_t, uint64_t> event_slice_ids;

    cv::Mat img;
    cv::Mat depth;
    cv::Mat mask;

public:
    static void add_cloud(int id, std::shared_ptr<ViObject> cl) {
        DatasetFrame::clouds[id] = cl;
    }
    static void add_background(std::shared_ptr<StaticObject> bg) {
        DatasetFrame::background = bg;
        DatasetFrame::background->transform(DatasetConfig::bg_E);
    }
    static void set_event_array(std::vector<Event>* ea) {
        DatasetFrame::event_array = ea;
    }
    static void set_camera_trajectory(Trajectory* cam_tj) {
        DatasetFrame::cam_tj = cam_tj;
    }
    static void set_object_trajectories(std::map<int, Trajectory>* obj_tjs) {
        DatasetFrame::obj_tjs = obj_tjs;
    }
    static void init_cloud(int id, const vicon::Subject& subject) {
        if (DatasetFrame::clouds.find(id) == DatasetFrame::clouds.end())
            return;
        clouds[id]->init_cloud_to_vicon_tf(subject);
    }
    static void on_trackbar(int, void*) {
        for (auto &frame_ptr : DatasetFrame::visualization_list) {
            std::string window_name = "Frame " + std::to_string(frame_ptr->frame_id);
            cv::setTrackbarPos("frame", window_name, value_trackbar_0);
        }
    }

    //static long int get_value_trackbar_0() { return DatasetFrame::value_trackbar_0; }
    //static long int get_trackbar_0_max()   { return DatasetFrame::trackbar_0_max; }

    static void visualization_spin() {
        std::map<DatasetFrame*, std::string> window_names;
        for (auto &frame_ptr : DatasetFrame::visualization_list) {
            window_names[frame_ptr] = "Frame " + std::to_string(frame_ptr->frame_id);
            cv::namedWindow(window_names[frame_ptr], cv::WINDOW_NORMAL);
            //cv::createTrackbar("frame", window_names[frame_ptr], &value_trackbar_0, trackbar_0_max, on_trackbar);
        }

        DatasetConfig::modified = true;
        DatasetConfig::init_GUI();
        const uint8_t nmodes = 3;
        uint8_t vis_mode = 0;

        int code = 0; // Key code
        while (code != 27) {
            code = cv::waitKey(1);
            DatasetConfig::handle_keys(code, vis_mode, nmodes);

            if (!DatasetConfig::modified) continue;
            DatasetConfig::modified = false;

            for (auto &window : window_names) {
                window.first->generate_async();
            }

            for (auto &window : window_names) {
                window.first->join();

                cv::Mat img;
                switch (vis_mode) {
                    default:
                    case 0: img = window.first->get_visualization_mask(true); break;
                    case 1: img = window.first->get_visualization_mask(false); break;
                    case 2: img = window.first->get_visualization_depth(true); break;
                    case 3: img = window.first->get_visualization_event_projection(true); break;
                }

                cv::imshow(window.second, img);
            }
        }

        cv::destroyAllWindows();
        DatasetFrame::visualization_list.clear();
    }

    // ---------
    DatasetFrame(uint64_t cam_p_id, double ref_ts, unsigned long int fid)
        : cam_pose_id(cam_p_id), timestamp(ref_ts), frame_id(fid), event_slice_ids(0, 0),
          depth(DatasetConfig::res_x, DatasetConfig::res_y, CV_32F, cv::Scalar(0)),
          mask(DatasetConfig::res_x, DatasetConfig::res_y, CV_8U, cv::Scalar(0)) {
        this->cam_pose_id = TimeSlice(*this->cam_tj).find_nearest(this->get_timestamp(), this->cam_pose_id);
    }

    void add_object_pos_id(int id, uint64_t obj_p_id) {
        this->obj_pose_ids.insert(std::make_pair(id, obj_p_id));
        this->obj_pose_ids[id] = TimeSlice(this->obj_tjs->at(id)).find_nearest(this->get_timestamp(), this->obj_pose_ids[id]);
    }

    void add_event_slice_ids(uint64_t event_low, uint64_t event_high) {
        this->event_slice_ids = std::make_pair(event_low, event_high);
        this->event_slice_ids = TimeSlice(*this->event_array,
            std::make_pair(this->timestamp - DatasetConfig::get_time_offset_event_to_host_correction() - DatasetConfig::slice_width / 2.0,
                           this->timestamp - DatasetConfig::get_time_offset_event_to_host_correction() + DatasetConfig::slice_width / 2.0),
            this->event_slice_ids).get_indices();
    }

    void add_img(cv::Mat &img_) {
        this->img = img_;
    }

    void show() {
        DatasetFrame::visualization_list.push_back(this);
    }

    Pose get_true_camera_pose() {
        auto cam_pose = this->_get_raw_camera_pose();
        auto cam_tf = cam_pose.pq * DatasetConfig::cam_E;
        return Pose(cam_pose.ts, cam_tf);
    }

    Pose get_true_object_pose(int id) {
        if (this->obj_pose_ids.find(id) == this->obj_pose_ids.end()) {
            std::cout << _yellow("Warning! ") << "No pose for object "
                      << id << ", frame id = " << this->frame_id << std::endl;
        }

        auto obj_pose = this->_get_raw_object_pose(id);
        auto cam_tf   = this->get_true_camera_pose();
        auto obj_tf   = DatasetFrame::clouds.at(id)->get_tf_in_camera_frame(
                                                        cam_tf, obj_pose.pq);
        return Pose(cam_tf.ts, obj_tf);
    }

    float get_timestamp() {
        return this->timestamp - DatasetConfig::get_time_offset_pose_to_host_correction();
    }

    std::string get_info() {
        std::string s;
        s += std::to_string(frame_id) + ": " + std::to_string(get_timestamp()) + "\t";
        s += std::to_string(get_true_camera_pose().ts.toSec()) + "\t";
        for (auto &obj : DatasetFrame::clouds) {
            s += std::to_string(this->_get_raw_object_pose(obj.first).ts.toSec()) + "\t";
        }
        return s;
    }

    // Generate frame
    void generate() {
        this->depth = cv::Scalar(0);
        this->mask  = cv::Scalar(0);

        DatasetConfig::update_cam_calib();
        this->cam_tj->set_filtering_window_size(DatasetConfig::pose_filtering_window);
        this->cam_pose_id = TimeSlice(*this->cam_tj).find_nearest(this->get_timestamp(), this->cam_pose_id);
        this->event_slice_ids = TimeSlice(*this->event_array,
            std::make_pair(this->timestamp - DatasetConfig::get_time_offset_event_to_host_correction() - DatasetConfig::slice_width / 2.0,
                           this->timestamp - DatasetConfig::get_time_offset_event_to_host_correction() + DatasetConfig::slice_width / 2.0),
            this->event_slice_ids).get_indices();

        auto cam_tf = this->get_true_camera_pose();
        if (DatasetFrame::background != nullptr) {
            auto cl = DatasetFrame::background->transform_to_camframe(cam_tf);
            this->project_cloud(cl, 0);
        }

        for (auto &obj : DatasetFrame::clouds) {
            auto id = obj.first;
            this->obj_tjs->at(id).set_filtering_window_size(DatasetConfig::pose_filtering_window);
            this->obj_pose_ids[id] = TimeSlice(this->obj_tjs->at(id)).find_nearest(this->get_timestamp(), this->obj_pose_ids[id]);

            if (this->obj_pose_ids.find(id) == this->obj_pose_ids.end()) {
                std::cout << _yellow("Warning! ") << "No pose for object "
                          << id << ", frame id = " << this->frame_id << std::endl;
                continue;
            }

            auto obj_pose = this->_get_raw_object_pose(id);
            auto cl = obj.second->transform_to_camframe(cam_tf, obj_pose.pq);
            this->project_cloud(cl, id);
        }
    }

    void generate_async() {
        this->thread_handle = std::thread(&DatasetFrame::generate, this);
    }

    void join() {
        this->thread_handle.join();
    }

    // Visualization pipeline
    cv::Mat get_visualization_event_projection(bool timg = false) {
        cv::Mat img;
        if (DatasetFrame::event_array != nullptr) {
            auto ev_slice = Slice<std::vector<Event>>(*DatasetFrame::event_array,
                                                      this->event_slice_ids);
            if (timg) {
                img = EventFile::color_time_img(&ev_slice, 1);
            } else {
                img = EventFile::projection_img(&ev_slice, 1);
            }
        }
        return img;
    }

    cv::Mat get_visualization_depth(bool overlay_events = true) {
        auto depth_img = this->depth;
        cv::Mat img_pr = this->get_visualization_event_projection();
        auto ret = cv::Mat(depth_img.rows, depth_img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat mask;
        cv::threshold(depth_img, mask, 0.01, 255, cv::THRESH_BINARY);
        mask.convertTo(mask, CV_8U);
        cv::normalize(depth_img, depth_img, 1, 255, cv::NORM_MINMAX, -1, mask);
        //cv::divide(8000.0, depth_img, depth_img);
        for(int i = 0; i < depth_img.rows; ++i) {
            for (int j = 0; j < depth_img.cols; ++j) {
                ret.at<cv::Vec3b>(i, j)[0] = depth_img.at<float>(i, j);
                ret.at<cv::Vec3b>(i, j)[1] = depth_img.at<float>(i, j);
                ret.at<cv::Vec3b>(i, j)[2] = depth_img.at<float>(i, j);
                if (overlay_events && DatasetFrame::event_array != nullptr)
                    ret.at<cv::Vec3b>(i, j)[2] = img_pr.at<uint8_t>(i, j);
            }
        }
        return ret;
    }

    cv::Mat get_visualization_mask(bool overlay_events = true) {
        auto mask_img = this->mask;
        auto rgb_img  = this->img;
        cv::Mat img_pr = this->get_visualization_event_projection();
        auto ret = cv::Mat(mask_img.rows, mask_img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        for(int i = 0; i < mask_img.rows; ++i) {
            for (int j = 0; j < mask_img.cols; ++j) {
                int id = std::round(mask_img.at<uint8_t>(i, j));
                auto color = EventFile::id2rgb(id);
                if (rgb_img.rows == mask_img.rows && rgb_img.cols == mask_img.cols) {
                    ret.at<cv::Vec3b>(i, j) = rgb_img.at<cv::Vec3b>(i, j);
                    if (id > 0) {
                        ret.at<cv::Vec3b>(i, j) = rgb_img.at<cv::Vec3b>(i, j) * 0.5 + color * 0.5;
                    }
                } else {
                    ret.at<cv::Vec3b>(i, j) = color;
                }
                if (overlay_events && DatasetFrame::event_array != nullptr && img_pr.at<uint8_t>(i, j) > 0)
                    ret.at<cv::Vec3b>(i, j)[2] = img_pr.at<uint8_t>(i, j);
            }
        }
        return ret;
    }

protected:
    void project_point(pcl::PointXYZRGB p, int &u, int &v) {
        u = -1; v = -1;
        if (p.z < 0.00001)
            return;

        float x_ = p.x / p.z;
        float y_ = p.y / p.z;

        float r2 = x_ * x_ + y_ * y_;
        float r4 = r2 * r2;
        float r6 = r2 * r2 * r2;
        float dist = (1.0 + DatasetConfig::k1 * r2 + DatasetConfig::k2 * r4 +
                            DatasetConfig::k3 * r6) / (1 + DatasetConfig::k4 * r2);
        float x__ = x_ * dist;
        float y__ = y_ * dist;

        u = DatasetConfig::fx * x__ + DatasetConfig::cx;
        v = DatasetConfig::fy * y__ + DatasetConfig::cy;
    }

    void project_cloud(auto cl, int oid) {
        if (cl->size() == 0)
            return;

        for (auto &p: *cl) {
            p.z = -p.z;

            float rng = p.z;
            if (rng < 0.001)
                continue;

            auto cols = this->depth.cols;
            auto rows = this->depth.rows;

            int u = -1, v = -1;
            this->project_point(p, u, v);

            if (u < 0 || v < 0 || v >= cols || u >= rows)
                continue;

            int patch_size = int(1.0 / rng);

            if (oid == 0)
                patch_size = int(5.0 / rng);

            int u_lo = std::max(u - patch_size / 2, 0);
            int u_hi = std::min(u + patch_size / 2, rows - 1);
            int v_lo = std::max(v - patch_size / 2, 0);
            int v_hi = std::min(v + patch_size / 2, cols - 1);

            for (int ii = u_lo; ii <= u_hi; ++ii) {
                for (int jj = v_lo; jj <= v_hi; ++jj) {
                    float base_rng = this->depth.at<float>(rows - ii - 1, cols - jj - 1);
                    if (base_rng > rng || base_rng < 0.001) {
                        this->depth.at<float>(rows - ii - 1, cols - jj - 1) = rng;
                        this->mask.at<uint8_t>(rows - ii - 1, cols - jj - 1) = oid;
                    }
                }
            }
        }
    }

    Pose _get_raw_camera_pose() {
        if (this->cam_pose_id >= DatasetFrame::cam_tj->size()) {
            std::cout << _yellow("Warning! ") << "Camera pose out of bounds for "
                      << " frame id " << this->frame_id << " with "
                      << DatasetFrame::cam_tj->size() << " trajectory records and "
                      << "trajectory id = " << this->cam_pose_id << std::endl;
            return (*DatasetFrame::cam_tj)[DatasetFrame::cam_tj->size() - 1];
        }
        return (*DatasetFrame::cam_tj)[this->cam_pose_id];
    }

    Pose _get_raw_object_pose(int id) {
        if (this->obj_pose_ids.find(id) == this->obj_pose_ids.end()) {
            std::cout << _yellow("Warning! ") << "No pose for object "
                      << id << ", frame id = " << this->frame_id << std::endl;
        }
        auto obj_pose_id = this->obj_pose_ids.at(id);
        auto obj_tj_size = DatasetFrame::obj_tjs->at(id).size();
        if (obj_pose_id >= obj_tj_size) {
            std::cout << _yellow("Warning! ") << "Object (" << id << ") pose "
                      << "out of bounds for frame id " << this->frame_id << " with "
                      << obj_tj_size << " trajectory records and "
                      << "trajectory id = " << obj_pose_id << std::endl;
            return DatasetFrame::obj_tjs->at(id)[obj_tj_size - 1];
        }
        return DatasetFrame::obj_tjs->at(id)[obj_pose_id];
    }
};

std::shared_ptr<StaticObject> DatasetFrame::background;
std::map<int, std::shared_ptr<ViObject>> DatasetFrame::clouds;
std::list<DatasetFrame*> DatasetFrame::visualization_list;
std::vector<Event>* DatasetFrame::event_array = nullptr;
Trajectory* DatasetFrame::cam_tj = nullptr;
std::map<int, Trajectory>* DatasetFrame::obj_tjs = nullptr;
int DatasetFrame::trackbar_0_max = 100;
int DatasetFrame::value_trackbar_0 = 50;


class FrameSequenceVisualizer {
protected:
    std::vector<DatasetFrame> *frames;
    int frame_id;

public:
    FrameSequenceVisualizer(std::vector<DatasetFrame> &frames)
        : frame_id(0) {
        this->frames = &frames;
        this->spin();
    }

    void spin() {
        cv::namedWindow("Frames", cv::WINDOW_NORMAL);
        cv::createTrackbar("frame", "Frames", &frame_id, frames->size() - 1, on_trackbar);

        DatasetConfig::modified = true;
        DatasetConfig::init_GUI();
        const uint8_t nmodes = 4;
        uint8_t vis_mode = 0;

        int code = 0; // Key code
        while (code != 27) {
            code = cv::waitKey(1);
            DatasetConfig::handle_keys(code, vis_mode, nmodes);

            if (!DatasetConfig::modified) continue;
            DatasetConfig::modified = false;

            auto &f = this->frames->at(this->frame_id);
            f.generate();

            cv::Mat img;
            switch (vis_mode) {
                default:
                case 0: img = f.get_visualization_mask(true); break;
                case 1: img = f.get_visualization_mask(false); break;
                case 2: img = f.get_visualization_depth(true); break;
                case 3: img = f.get_visualization_event_projection(true); break;
            }

            //std::cout << f.get_info() << "\n";
            cv::imshow("Frames", img);
        }

        cv::destroyAllWindows();
    }

    static void on_trackbar(int, void*) {
        DatasetConfig::modified = true;
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
    bool through_mode = false, generate = true;
    int show = -1;
    if (!nh.getParam(node_name + "/fps", FPS)) FPS = 40;
    if (!nh.getParam(node_name + "/smoothing", traj_smoothing)) traj_smoothing = 1;
    if (!nh.getParam(node_name + "/numbering", through_mode)) through_mode = false;
    if (!nh.getParam(node_name + "/generate", generate)) generate = true;
    if (!nh.getParam(node_name + "/show", show)) show = -1;

    bool no_background = false;
    if (!nh.getParam(node_name + "/no_bg", no_background)) no_background = false;

    bool with_images = false;
    if (!nh.getParam(node_name + "/with_images", with_images)) with_images = false;
    else std::cout << _yellow("With 'with_images' option, the datased will be generated at image framerate.") << std::endl;

    int res_x = 260, res_y = 346;
    if (!nh.getParam(node_name + "/res_x", res_x)) res_x = 260;
    if (!nh.getParam(node_name + "/res_y", res_y)) res_y = 346;

    // -- camera topics
    std::string cam_pose_topic = "", event_topic = "", img_topic = "";
    std::map<int, std::string> obj_pose_topics;
    if (!nh.getParam(node_name + "/cam_pose_topic", cam_pose_topic)) cam_pose_topic = "/vicon/DVS346";
    if (!nh.getParam(node_name + "/event_topic", event_topic)) event_topic = "/dvs/events";
    if (!nh.getParam(node_name + "/img_topic", img_topic)) img_topic = "/dvs/image_raw";

    // -- parse the dataset folder
    std::string bag_name = boost::filesystem::path(dataset_folder).stem().string();
    if (bag_name == ".") {
        bag_name = boost::filesystem::path(dataset_folder).parent_path().stem().string();
    }

    auto bag_name_path = boost::filesystem::path(dataset_folder);
    bag_name_path /= (bag_name + ".bag");
    bag_name = bag_name_path.string();
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
    if (!DatasetConfig::init(dataset_folder, (unsigned int)res_x, (unsigned int)res_y))
        return -1;

    // Load 3D models
    std::string path_to_self = ros::package::getPath("event_imo_datagen");

    if (!no_background)
        DatasetFrame::add_background(std::make_shared<StaticObject>(path_to_self + "/objects/room"));
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
    std::vector<cv::Mat> images;
    std::vector<ros::Time> image_ts;
    uint64_t n_events = 0;
    for (auto &m : view) {
        if (m.getTopic() == cam_pose_topic) {
            auto msg = m.instantiate<vicon::Subject>();
            if (msg == NULL) continue;
            cam_tj.add(msg->header.stamp + ros::Duration(DatasetConfig::get_time_offset_pose_to_host()), *msg);
            continue;
        }

        for (auto &p : obj_pose_topics) {
            if (m.getTopic() != p.second) continue;
            auto msg = m.instantiate<vicon::Subject>();
            if (msg == NULL) break;
            if (msg->occluded) break;
            obj_tjs[p.first].add(msg->header.stamp + ros::Duration(DatasetConfig::get_time_offset_pose_to_host()), *msg);
            obj_cloud_to_vicon_tf[p.first] = *msg;
            break;
        }

        if (m.getTopic() == event_topic) {
            // DVS / DAVIS event messages
            auto msg_dvs = m.instantiate<dvs_msgs::EventArray>();
            if (msg_dvs != NULL) {
                n_events += msg_dvs->events.size();
            }

            // PROPHESEE event messages
            auto msg_prs = m.instantiate<prophesee_event_msgs::PropheseeEventBuffer>();
            if (msg_prs != NULL) {
                n_events += msg_prs->events.size();
            }

            continue;
        }

        if (with_images && (m.getTopic() == img_topic)) {
            auto msg = m.instantiate<sensor_msgs::Image>();
            images.push_back(cv_bridge::toCvShare(msg, "bgr8")->image);
            image_ts.push_back(msg->header.stamp + ros::Duration(DatasetConfig::get_time_offset_image_to_host()));
        }
    }

    DatasetFrame::set_camera_trajectory(&cam_tj);
    DatasetFrame::set_object_trajectories(&obj_tjs);

    if (with_images && images.size() == 0) {
        std::cout << _red("No images found! Reverting 'with_images' to 'false'") << std::endl;
        with_images = false;
    }

    std::vector<Event> event_array(n_events);
    uint64_t id = 0;
    ros::Time first_event_ts;
    ros::Time first_event_message_ts;
    ros::Time last_event_ts;
    for (auto &m : view) {
        if (m.getTopic() != event_topic)
            continue;

        auto msg_dvs = m.instantiate<dvs_msgs::EventArray>();
        auto msg_prs = m.instantiate<prophesee_event_msgs::PropheseeEventBuffer>();

        auto msize = 0;
        if (msg_dvs != NULL) msize = msg_dvs->events.size();
        if (msg_prs != NULL) msize = msg_prs->events.size();

        for (uint64_t i = 0; i < msize; ++i) {
            int32_t x = 0, y = 0;
            ros::Time current_event_ts = ros::Time(0);
            int polarity = 0;

            // Sensor-specific switch:
            // DVS / DAVIS
            if (msg_dvs != NULL) {
                auto &e = msg_dvs->events[i];
                current_event_ts = e.ts;
                x = e.x; y = e.y;
                polarity = e.polarity ? 1 : 0;
            }

            // PROPHESEE
            if (msg_prs != NULL) {
                auto &e = msg_prs->events[i];
                current_event_ts = ros::Time(double(e.t) / 1000000.0);
                x = e.x; y = e.y;
                polarity = e.p ? 1 : 0;
            }

            if (id == 0) {
                first_event_ts = current_event_ts;
                last_event_ts = current_event_ts;
                first_event_message_ts = m.getTime();
            } else {
                if (current_event_ts < last_event_ts) {
                    std::cout << _red("Events are not sorted! ")
                              << id << ": " << last_event_ts << " -> "
                              << current_event_ts << std::endl;
                }
                last_event_ts = current_event_ts;
            }

            auto ts = (first_event_message_ts + (current_event_ts - first_event_ts) +
                       ros::Duration(DatasetConfig::get_time_offset_event_to_host())).toNSec();
            event_array[id] = Event(y, x, ts, polarity);
            id ++;
        }
    }

    std::cout << _green("Read ") << n_events << _green(" events") << std::endl;
    std::cout << std::endl << _green("Read ") << cam_tj.size() << _green(" camera poses and ") << std::endl;
    for (auto &obj_tj : obj_tjs) {
        if (obj_tj.second.size() == 0) continue;
        std::cout << "\t" << obj_tj.second.size() << _blue(" poses for object ") << obj_tj.first << std::endl;
        if (!obj_tj.second.check()) {
            std::cout << "\t\t" << _red("Check failed!") << std::endl;
        }
        DatasetFrame::init_cloud(obj_tj.first, obj_cloud_to_vicon_tf[obj_tj.first]);
    }
    DatasetFrame::set_event_array(&event_array);

    // Force the first timestamp of the event cloud to be 0
    // trajectories
    auto time_offset = first_event_message_ts + ros::Duration(DatasetConfig::get_time_offset_event_to_host());
    cam_tj.subtract_time(time_offset);
    for (auto &obj_tj : obj_tjs)
        obj_tj.second.subtract_time(time_offset);
    // images
    while(image_ts.size() > 0 && *image_ts.begin() < time_offset) {
        image_ts.erase(image_ts.begin());
        images.erase(images.begin());
    }
    for (uint64_t i = 0; i < image_ts.size(); ++i)
        image_ts[i] = ros::Time((image_ts[i] - time_offset).toSec() < 0 ? 0 : (image_ts[i] - time_offset).toSec());
    // events
    for (auto &e : event_array)
        e.timestamp -= time_offset.toNSec();

    std::cout << std::endl << "Removing time offset: " << _green(std::to_string(time_offset.toSec()))
              << std::endl << std::endl;

    // Align the timestamps
    double start_ts = 0.2;
    unsigned long int frame_id_through = 0, frame_id_real = 0;
    double dt = 1.0 / FPS;
    long int cam_tj_id = 0;
    std::map<int, long int> obj_tj_ids;
    std::vector<DatasetFrame> frames;
    uint64_t event_low = 0, event_high = 0;
    while (true) {
        if (with_images) {
            if (frame_id_real >= image_ts.size()) break;
            start_ts = image_ts[frame_id_real].toSec();
        }

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

        auto ref_ts = (with_images ? image_ts[frame_id_real].toSec() : cam_tj[cam_tj_id].ts.toSec());
        uint64_t ts_low  = (ref_ts < DatasetConfig::slice_width) ? 0 : (ref_ts - DatasetConfig::slice_width / 2.0) * 1000000000;
        uint64_t ts_high = (ref_ts + DatasetConfig::slice_width / 2.0) * 1000000000;
        while (event_low  < event_array.size() && event_array[event_low].timestamp  < ts_low)  event_low ++;
        while (event_high < event_array.size() && event_array[event_high].timestamp < ts_high) event_high ++;

        double max_ts_err = 0.0;
        for (auto &obj_tj : obj_tjs) {
            if (obj_tj.second.size() == 0) continue;
            double ts_err = std::fabs(ref_ts - obj_tj.second[obj_tj_ids[obj_tj.first]].ts.toSec());
            if (ts_err > max_ts_err) max_ts_err = ts_err;
        }

        if (max_ts_err > 0.005) {
            std::cout << _red("Trajectory timestamp misalignment: ") << max_ts_err << " skipping..." << std::endl;
            frame_id_real ++;
            continue;
        }

        frames.emplace_back(cam_tj_id, ref_ts, through_mode ? frame_id_through : frame_id_real);
        DatasetFrame &frame = frames.back();

        frame.add_event_slice_ids(event_low, event_high);
        if (with_images) frame.add_img(images[frame_id_real]);
        std::cout << (through_mode ? frame_id_through : frame_id_real) << ": " << cam_tj[cam_tj_id].ts
                  << " (" << cam_tj_id << "[" << cam_tj[cam_tj_id].occlusion * 100 << "%])";
        for (auto &obj_tj : obj_tjs) {
            if (obj_tj.second.size() == 0) continue;
            std::cout << " " << obj_tj.second[obj_tj_ids[obj_tj.first]].ts << " (" << obj_tj_ids[obj_tj.first]
                      << "[" << obj_tj.second[obj_tj_ids[obj_tj.first]].occlusion * 100 <<  "%])";
            frame.add_object_pos_id(obj_tj.first, obj_tj_ids[obj_tj.first]);
        }
        std::cout << std::endl;

        frame_id_real ++;
        frame_id_through ++;
    }

    std::cout << _blue("\nTimestamp alignment done") << std::endl;
    std::cout << "\tDataset contains " << frames.size() << " frames" << std::endl;

    // Visualization
    int step = std::max(int(frames.size()) / show, 1);
    for (int i = 0; i < frames.size() && show > 0; i += step) {
        frames[i].show();
    }
    if (show > 0) {
        DatasetFrame::visualization_spin();
    }

    if (show == -2)
        FrameSequenceVisualizer fsv(frames);

    // Exit if we are running in the visualization mode
    if (!generate) {
        return 0;
    }

    // Projecting the clouds and generating masks / depth maps
    std::cout << std::endl << _yellow("Generating ground truth") << std::endl;
    for (int i = 0; i < frames.size(); ++i) {
        frames[i].generate_async();
    }

    for (int i = 0; i < frames.size(); ++i) {
        frames[i].join();
        if (i % 10 == 0) {
            std::cout << "\tFrame " << i + 1 << " / " << frames.size() <<  std::endl;
        }
    }

    // Save the data
    auto gt_dir_path = boost::filesystem::path(dataset_folder);
    gt_dir_path /= "gt";

    auto img_dir_path = boost::filesystem::path(dataset_folder);
    img_dir_path /= "img";

    std::string tsfname = dataset_folder + "/ts.txt";
    std::ofstream ts_file(tsfname, std::ofstream::out);

    std::string rgbtsfname = dataset_folder + "/images.txt";
    std::ofstream rgb_ts_file(rgbtsfname, std::ofstream::out);

    std::string obj_fname = dataset_folder + "/objects.txt";
    std::ofstream obj_file(obj_fname, std::ofstream::out);

    std::string cam_fname = dataset_folder + "/trajectory.txt";
    std::ofstream cam_file(cam_fname, std::ofstream::out);

    std::string efname = dataset_folder + "/events.txt";
    std::ofstream event_file(efname, std::ofstream::out);

    std::cout << _blue("Removing old: " + gt_dir_path.string()) << std::endl;
    boost::filesystem::remove_all(gt_dir_path);
    std::cout << "Creating: " << _green(gt_dir_path.string()) << std::endl;
    boost::filesystem::create_directory(gt_dir_path);

    if (with_images) {
        std::cout << _blue("Removing old: " + img_dir_path.string()) << std::endl;
        boost::filesystem::remove_all(img_dir_path);
        std::cout << "Creating: " << _green(img_dir_path.string()) << std::endl;
        boost::filesystem::create_directory(img_dir_path);
    }

    std::cout << std::endl << _yellow("Writing depth and mask ground truth") << std::endl;
    for (int i = 0; i < frames.size(); ++i) {

        // camera pose
        auto cam_pose = frames[i].get_true_camera_pose();
        cam_file << frames[i].frame_id << " " << cam_pose << std::endl;

        // object poses
        for (auto &pair : frames[i].obj_pose_ids) {
            auto obj_pose = frames[i].get_true_object_pose(pair.first);
            obj_file << frames[i].frame_id << " " << pair.first << " " << obj_pose << std::endl;
        }

        // masks and depth
        std::string img_name = "/frame_" + std::to_string(frames[i].frame_id) + ".png";
        std::string gtfname = gt_dir_path.string() + img_name;
        cv::Mat depth, mask;
        frames[i].depth.convertTo(depth, CV_16UC1, 1000);
        frames[i].mask.convertTo(mask, CV_16UC1, 1000);
        std::vector<cv::Mat> ch = {depth, depth, mask};

        cv::Mat gt_frame_i16(depth.rows, depth.cols, CV_16UC3, cv::Scalar(0, 0, 0));
        cv::merge(ch, gt_frame_i16);

        gt_frame_i16.convertTo(gt_frame_i16, CV_16UC3);
        cv::imwrite(gtfname, gt_frame_i16);

        if (with_images) {
            cv::imwrite(img_dir_path.string() + img_name, frames[i].img);
            rgb_ts_file << frames[i].get_timestamp() << " " << "img" + img_name << std::endl;
        }

        // timestamps
        ts_file << "gt" + img_name << " " << frames[i].get_timestamp() << std::endl;

        if (i % 10 == 0) {
            std::cout << "\tWritten " << i + 1 << " / " << frames.size() <<  std::endl;
        }
    }
    ts_file.close();
    rgb_ts_file.close();
    obj_file.close();
    cam_file.close();

    std::cout << std::endl << _yellow("Writing events.txt") << std::endl;
    std::stringstream ss;
    for (uint64_t i = 0; i < event_array.size(); ++i) {
        if (i % 100000 == 0) {
            std::cout << "\tFormatting " << i + 1 << " / " << event_array.size() << std::endl;
        }

        ss << std::fixed << std::setprecision(9)
           << event_array[i].get_ts_sec()
           << " " << event_array[i].fr_y << " " << event_array[i].fr_x
           << " " << int(event_array[i].polarity) << std::endl;
    }

    event_file << ss.str();
    event_file.close();
    std::cout << std::endl << _green("Done!") << std::endl;

    return 0;
}
