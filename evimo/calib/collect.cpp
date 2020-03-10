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
#include <ros/package.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <X11/Xlib.h>

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
    std::string name, window_name, topic;

    bool accumulating;
    cv::Mat accumulated_image;

public:
    Imager(std::string name_, std::string topic_)
        : res_x(0), res_y(0), name(name_), topic(topic_), accumulating(false) {
        this->window_name = this->name + "_" + std::to_string(uid);
        Imager::uid += 1;
        cv::namedWindow(this->window_name, cv::WINDOW_NORMAL);
    }

    virtual ~Imager() {
        cv::destroyWindow(this->window_name);
    }

    virtual void start_accumulating() = 0;
    virtual void save() {
        this->accumulating = false;
    }
    virtual cv::Mat get_accumulated_image() {return this->accumulated_image;}
};

int Imager::uid = 0;


class EventStreamImager : public Imager {
protected:
    ros::Rate r;
    std::thread thread_handle;

    // Buffer for incoming events (aka 'slice')
    // 1e8 == 0.1 sec
    CircularArray<Event, 3000000, 100000000> ev_buffer;

public:
    EventStreamImager(ros::NodeHandle &nh, std::string name_, std::string topic_, float fps_=40)
        : Imager(name_, topic_), r(fps_) {
        this->sub = nh.subscribe(this->topic, 0,
                                 &EventStreamImager::event_cb<dvs_msgs::EventArray::ConstPtr>, this);
        this->thread_handle = std::thread(&EventStreamImager::vis_spin, this);
    }

    // Callbacks
    template<class T>
    void event_cb(const T& msg) {
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
                img = EventFile::color_time_img(&ev_buffer, 1, this->res_y, this->res_x);
            }
            if (img.cols == 0 || img.rows == 0)
                continue;

            cv::imshow(this->window_name, img);
            r.sleep();
        }
    }

    void start_accumulating() override {
        this->accumulating = true;
        this->accumulated_image = cv::Mat::zeros(this->res_y, this->res_x, CV_32S);
    }

    cv::Mat get_accumulated_image() override {
        double nz_avg = 0;
        uint64_t nz_avg_cnt = 0;
        auto &img = this->accumulated_image;
        int *p = (int *)img.data;
        for (int i = 0; i < img.rows * img.cols; ++i, p++) {
            if (*p == 0) continue;
            nz_avg_cnt ++;
            nz_avg += *p;
        }
        nz_avg = (nz_avg_cnt == 0) ? 1 : nz_avg / double(nz_avg_cnt);
        double img_scale = 127.0 / nz_avg;

        cv::Mat ret;
        cv::convertScaleAbs(this->accumulated_image, ret, img_scale, 0);
        return ret;
    }

private:
    void accumulate_event(Event &e) {
        if (e.fr_x >= this->accumulated_image.rows || e.fr_x < 0 ||
            e.fr_y >= this->accumulated_image.cols || e.fr_y < 0) return;
        accumulated_image.at<int>(e.fr_x, e.fr_y) ++;
    }
};


int main (int argc, char** argv) {

    // Initialize ROS
    std::string node_name = "calibration_data_collector";
    ros::init (argc, argv, node_name);
    ros::NodeHandle nh("~");
    XInitThreads();

    // parse config
    




    std::vector<std::shared_ptr<Imager>> imagers;

    imagers.push_back(std::make_shared<EventStreamImager>(nh, "cam0", "/samsung/camera/events"));
    //imagers.push_back(std::make_shared<EventStreamImager>(nh, "cam1", "/samsung/camera/events"));

    int code = 0; // Key code
    while (code != 27) {
        code = cv::waitKey(5);
        if (code == 32) {
            for (auto &i : imagers)
                i->start_accumulating();
        }

        if (code == 115) { // 's'
            for (auto &i : imagers)
                i->save();
        }

        ros::spinOnce();
    }

    ros::shutdown();
    return 0;
}
