#include <vector>
#include <valarray>
#include <algorithm>
#include <thread>
#include <type_traits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <boost/filesystem.hpp>
#include <X11/Xlib.h>

#include <ros/ros.h>
#include <ros/package.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

// Local includes
#include <common.h>
#include <event.h>
#include <event_vis.h>

// VICON
#include <vicon/Subject.h>

// DVS / DAVIS
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>


class Imager {
protected:
    static int uid;

    uint32_t res_x, res_y;
    ros::Subscriber sub;
    std::string root_dir, name, window_name, topic;
    bool headless;

    uint64_t cnt;
    bool accumulating;
    cv::Mat accumulated_image;

public:
    Imager(std::string root_dir_, std::string name_, std::string topic_, bool headless_=false)
        : res_x(0), res_y(0), root_dir(root_dir_), name(name_), topic(topic_)
        , headless(headless_), cnt(0), accumulating(false) {
        this->window_name = this->name + "_" + std::to_string(uid);
        Imager::uid += 1;
        if (!this->headless)
            cv::namedWindow(this->window_name, cv::WINDOW_NORMAL);
    }

    virtual ~Imager() {
        if (!this->headless) cv::destroyWindow(this->window_name);
    }

    virtual bool create_dir() {
        auto out_dir_path = boost::filesystem::path(this->root_dir + '/' + this->name);
        std::cout << "Creating: " << _green(out_dir_path.string()) << std::endl;
        boost::filesystem::remove_all(out_dir_path);
        return boost::filesystem::create_directories(out_dir_path);
    }

    virtual void start_accumulating() {this->accumulating = true;}
    virtual void save() {
        if (!this->accumulating) {
            std::cout << _red("Trying to save with no accumulation") << std::endl;
            return;
        }

        this->accumulating = false;
        std::string img_name = this->root_dir + '/' + this->name + '/' + std::to_string(this->cnt * 7 + 7) + "0000000.png";
        cv::imwrite(img_name, this->get_accumulated_image());
        this->cnt += 1;
    }
    virtual cv::Mat get_accumulated_image() {return this->accumulated_image;}
};

int Imager::uid = 0;


// Save data from Vicon
class ViconImager : public Imager {
protected:
    std::ofstream ofile;
    vicon::Subject last_pos;

public:
    ViconImager(ros::NodeHandle &nh, std::string root_dir_, std::string name_, std::string topic_)
        : Imager(root_dir_, name_, topic_, true) {
        this->sub = nh.subscribe(this->topic, 0, &ViconImager::pose_cb, this);
    }

    ~ViconImager() {
        this->ofile.close();
    }

    virtual bool create_dir() override {
        std::string fname = '/' + this->name + '_' + "poses.txt";
        auto out_dir_path = boost::filesystem::path(this->root_dir);
        std::cout << "Creating: " << _green(out_dir_path.string() + fname) << std::endl;
        boost::filesystem::create_directories(out_dir_path);
        this->ofile.open(this->root_dir + fname, std::ofstream::out);
        return true;
    }

    void pose_cb(const vicon::Subject& subject) {
        this->last_pos = subject;
    }

    void save() override {
        if (!this->accumulating) {
            std::cout << _red("Trying to save with no accumulation") << std::endl;
            return;
        }

        auto &p = this->last_pos;
        this->accumulating = false;
        this->ofile << this->cnt << " "
                    << p.position.x << " " << p.position.y << " " << p.position.z << " "
                    << p.orientation.x << " " << p.orientation.y << " " << p.orientation.z
                    << " " << p.orientation.w << "";

        auto &markers = p.markers;
        for (auto &marker : markers) {
            this->ofile << " | {'" << marker.name << "': '"
                        << marker.position.x << " " << marker.position.y << " " << marker.position.z << "'}";
        }
        this->ofile << "\n";

        this->ofile.flush();
        this->cnt += 1;
    }
};


// Visuzlize and collect frames from regular cameras
class RGBImager : public Imager {
protected:
    std::shared_ptr<image_transport::ImageTransport> it_ptr;
    image_transport::Publisher img_pub;

public:
    RGBImager(ros::NodeHandle &nh, std::string root_dir_, std::string name_, std::string topic_)
        : Imager(root_dir_, name_, topic_) {
        this->it_ptr = std::make_shared<image_transport::ImageTransport>(nh);
        this->img_pub = it_ptr->advertise(this->window_name + "/image_raw", 1);
        this->sub = nh.subscribe(this->topic, 1, &RGBImager::frame_cb, this);
    }

    void frame_cb(const sensor_msgs::ImageConstPtr& msg) {
        cv::Mat img;
        if (msg->encoding == "8UC1") {
            sensor_msgs::Image simg = *msg;
            simg.encoding = "mono8";
            img = cv_bridge::toCvCopy(simg, "bgr8")->image;
        } else {
            img = cv_bridge::toCvShare(msg, "bgr8")->image;
        }
        this->res_y = msg->height;
        this->res_x = msg->width;
        this->accumulated_image = img.clone();
        cv::imshow(this->window_name, this->accumulated_image);
        //cv::waitKey(1);

        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->accumulated_image).toImageMsg();
        this->img_pub.publish(img_msg);
    }
};


// Visualize and accumulate data from event cameras
class EventStreamImager : public Imager {
protected:
    ros::Rate r;
    std::mutex mutex;
    std::thread thread_handle;
    cv::Mat vis_img;

    // Buffer for incoming events (aka 'slice')
    // 1e8 == 0.1 sec
    CircularArray<Event, 1000000, 50000000> ev_buffer;

    std::shared_ptr<image_transport::ImageTransport> it_ptr;
    image_transport::Publisher img_pub;

public:
    EventStreamImager(ros::NodeHandle &nh, std::string root_dir_, std::string name_, std::string topic_, float fps_=40)
        : Imager(root_dir_, name_, topic_), r(fps_) {
        this->it_ptr = std::make_shared<image_transport::ImageTransport>(nh);
        this->img_pub = it_ptr->advertise(this->window_name + "/image_raw", 1);
        this->sub = nh.subscribe(this->topic, 1,
                                 &EventStreamImager::event_cb<dvs_msgs::EventArray::ConstPtr>, this);
        this->thread_handle = std::thread(&EventStreamImager::vis_spin, this);
        this->thread_handle.detach();
    }

    // Callbacks
    template<class T>
    void event_cb(const T& msg) {
        const std::lock_guard<std::mutex> lock(this->mutex);
        for (uint i = 0; i < msg->events.size(); ++i) {
            ull time = msg->events[i].ts.toNSec();
            Event e(msg->events[i].y, msg->events[i].x, time);
            this->ev_buffer.push_back(e);
            if (this->accumulating) this->accumulate_event(e);
        }
        this->res_y = msg->height;
        this->res_x = msg->width;
    }

    void vis_spin() {
        while (ros::ok()) {
            cv::Mat img;
            if (this->accumulating) {
                img = this->get_accumulated_image();
            } else {
                const std::lock_guard<std::mutex> lock(this->mutex);
                img = EventFile::color_time_img(&ev_buffer, 1, this->res_y, this->res_x);
            }
            if (img.cols == 0 || img.rows == 0)
                continue;
            this->vis_img = img.clone();

            cv::imshow(this->window_name, this->vis_img);
            //cv::waitKey(1);

            sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->vis_img).toImageMsg();
            this->img_pub.publish(img_msg);
            r.sleep();
        }
    }

    void start_accumulating() override {
        this->accumulating = true;
        this->accumulated_image = cv::Mat::zeros(this->res_y, this->res_x, CV_32S);
    }

    cv::Mat get_accumulated_image() override {
        const std::lock_guard<std::mutex> lock(this->mutex);
        double nz_avg = 0;
        uint64_t nz_avg_cnt = 0;
        auto &img = this->accumulated_image;
        int *p = (int *)img.data;
        for (int i = 0; i < img.rows * img.cols; ++i, p++) {
            if (*p <= 50) continue;
            nz_avg_cnt ++;
            nz_avg += *p;
        }
        nz_avg = (nz_avg_cnt == 0) ? 1 : nz_avg / double(nz_avg_cnt);
        double img_scale = 127.0 / nz_avg;

        cv::Mat ret;
        cv::convertScaleAbs(this->accumulated_image, ret, img_scale, 0);
        std::vector<cv::Mat> ch = {ret, ret, ret};
        cv::merge(ch, ret);
        return ret;
    }

private:
    void accumulate_event(Event &e) {
        if (e.fr_x >= this->accumulated_image.rows || e.fr_x < 0 ||
            e.fr_y >= this->accumulated_image.cols || e.fr_y < 0) return;
        accumulated_image.at<int>(e.fr_x, e.fr_y) ++;
    }
};


class FlickerPattern {
private:
    ros::Rate r;
    std::thread thread_handle;
    bool paused;

protected:
    std::string window_name;
    uint32_t res_x, res_y;
    cv::Mat pattern;

public:
    FlickerPattern(float fps, std::string n="pattern")
        : r(fps * 2), paused(false), res_x(0), res_y(0), window_name(n) {
        cv::namedWindow(this->window_name, cv::WINDOW_NORMAL);
        this->thread_handle = std::thread(&FlickerPattern::vis_spin, this);
        this->thread_handle.detach();
    }

    virtual ~FlickerPattern() {
        cv::destroyWindow(this->window_name);
    }

    virtual void stop() {
        // maybe should add a mutex here
        this->paused = true;
    }

    virtual void cnte() {this->paused = false;}

private:
    void vis_spin() {
        while (ros::ok()) {
            r.sleep();
            if (this->res_y == 0 || this->res_x == 0) {
                continue;
            }

            cv::imshow(this->window_name, this->pattern);
            cv::waitKey(1);
            r.sleep();
            if (!this->paused) {
                cv::imshow(this->window_name, cv::Mat::zeros(this->res_y, this->res_x, CV_8U));
                cv::waitKey(1);
            }
        }
    }
};


class FlickerCheckerBoard : public FlickerPattern {
public:
    FlickerCheckerBoard(int px_w_, int py_w_, int nx_, int ny_, float fps, std::string n="checkerboard")
        : FlickerPattern(fps, n) {
        this->res_x = px_w_ * (nx_ + 1) + px_w_;
        this->res_y = py_w_ * (ny_ + 1) + py_w_;
        this->pattern = cv::Mat::ones(this->res_y, this->res_x, CV_8U) * 190;
        for (int i = 0; i < nx_ + 1; ++i) {
            for (int j = 0; j < ny_ + 1; ++j) {
                if ((i + j) % 2 == 1) continue;
                for (int k = i * px_w_; k < (i + 1) * px_w_; ++k) {
                    for (int l = j * py_w_; l < (j + 1) * py_w_; ++l) {
                        this->pattern.at<uint8_t>(l + py_w_ / 2, k + px_w_ / 2) = 0;
                    }
                }
            }
        }
    }
};



int main (int argc, char** argv) {

    // Initialize ROS
    std::string node_name = "calibration_data_collector";
    ros::init (argc, argv, node_name);
    ros::NodeHandle nh("~");
    XInitThreads();

    // Where to save the result
    std::string result_path = "";
    if (!nh.getParam("dir", result_path)) {
        std::cout << _red("Output directory needs to be specified!\n");
        return -1;
    }

    // Data processors
    std::vector<std::shared_ptr<Imager>> imagers;

    // parse config
    std::string path_to_self = ros::package::getPath("evimo");
    std::string config_path = "";
    if (!nh.getParam("conf", config_path)) config_path = path_to_self + "/calib/collect.cfg";

    std::ifstream ifs;
    ifs.open(config_path, std::ifstream::in);
    if (!ifs.is_open()) {
        std::cout << _red("Could not open configuration file file at ")
                  << config_path << "!" << std::endl;
        return -1;
    }

    const std::string& delims = ":\t ";
    while (ifs.good()) {
        std::string line;
        std::getline(ifs, line);
        line = trim(line);
        if (line.size() == 0) continue;
        if (line[0] == '#') continue;

        auto sep = line.find_first_of(delims);
        std::string cam_name = trim(line.substr(0, sep));
        line = trim(line.substr(sep + 1));
        sep = line.find_first_of(delims);
        std::string cam_type = trim(line.substr(0, sep));
        std::string topic_name = trim(line.substr(sep + 1));

        if (cam_type == "event") {
            imagers.push_back(std::make_shared<EventStreamImager>(nh, result_path, cam_name, topic_name));
        } else if (cam_type == "rgb") {
            imagers.push_back(std::make_shared<RGBImager>(nh, result_path, cam_name, topic_name));
        } else if (cam_type == "vicon") {
            imagers.push_back(std::make_shared<ViconImager>(nh, result_path, cam_name, topic_name));
        } else {
            std::cout << _red("unknown camera type:\t") << cam_type << std::endl;
        }

        std::cout << _green("Read source:\t") << cam_name << ":\t" << cam_type << "\t" << topic_name << std::endl;
    }
    ifs.close();

    // Creating directories
    for (auto &i : imagers) {
        if (!i->create_dir()) return -1;
    }

    // Create a flicker pattern
    std::shared_ptr<FlickerPattern> fpattern = std::make_shared<FlickerCheckerBoard>(40,40,13,6,5);
    fpattern->stop();

    ros::Time begin, end;
    int code = 0; // Key code
    while (code != 27) {
        code = cv::waitKey(5);
        if (code == 32) {
            begin = ros::Time::now();
            fpattern->cnte();
            for (auto &i : imagers)
                i->start_accumulating();
        }

        if (code == 115) { // 's'
            end = ros::Time::now();
            fpattern->stop();
            ros::Duration(0.3).sleep();
            ros::spinOnce();
            for (auto &i : imagers)
                i->save();
        }

        ros::spinOnce();
    }

    ros::shutdown();
    return 0;
}
