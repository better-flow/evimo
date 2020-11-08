#include <vector>
#include <valarray>
#include <algorithm>
#include <thread>
#include <type_traits>
#include <cmath>
#include <mutex>
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

// VICON
#include <vicon/Subject.h>

// DVS / DAVIS
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

// Local includes
#include <dataset.h>
#include <object.h>
#include <trajectory.h>
#include <dataset_frame.h>



image_transport::Publisher res_img_pub;
std::shared_ptr<Dataset> dataset;

class RGBCameraVisualizer {
protected:
    DatasetFrame frame;
    ros::Rate r;
    static int uid;
    std::string window_name, topic, event_topic;

    uint64_t images_received;
    ros::Subscriber sub, event_sub;
    std::mutex mutex;

public:
    RGBCameraVisualizer(ros::NodeHandle &nh, float FPS, std::shared_ptr<Dataset> &dataset)
        : frame(dataset, 0, 0, 0), r(FPS), topic(dataset->image_topic), event_topic(dataset->event_topic), images_received(0) {
        this->window_name = "RGBFrames_" + std::to_string(uid);
        RGBCameraVisualizer::uid += 1;

        if (this->topic.find("compressed") != std::string::npos) {
            this->sub = nh.subscribe(this->topic, 0, &RGBCameraVisualizer::compressed_frame_cb, this);
        } else {
            this->sub = nh.subscribe(this->topic, 0, &RGBCameraVisualizer::frame_cb, this);
        }

        this->event_sub = nh.subscribe(this->event_topic, 0,
                                       &RGBCameraVisualizer::event_cb<dvs_msgs::EventArray::ConstPtr>, this);
        this->spin();
    }

    // Callbacks
    void frame_cb(const sensor_msgs::ImageConstPtr& msg) {
        const std::lock_guard<std::mutex> lock(this->mutex);
        dataset->images.resize(1);
        if (msg->encoding == "8UC1") {
            sensor_msgs::Image img = *msg;
            img.encoding = "mono8";
            dataset->images[0] = (cv_bridge::toCvCopy(img, "bgr8")->image).clone();
        } else {
            dataset->images[0] = (cv_bridge::toCvShare(msg, "bgr8")->image).clone();
        }

        std::cout << "Image ts = " << msg->header.stamp << "\n";
        this->images_received ++;
    }

    void compressed_frame_cb(const sensor_msgs::CompressedImageConstPtr& msg) {
        dataset->images.resize(1);
        cv::Mat img = cv::imdecode(cv::Mat(msg->data), 1).clone();
        dataset->images[0] = img;

        std::cout << "Image ts = " << msg->header.stamp << "\n";
        this->images_received ++;
    }

    template<class T>
    void event_cb(const T& msg) {
        dataset->event_array.clear();
        for (uint i = 0; i < msg->events.size(); ++i) {
            ull time = msg->events[i].ts.toNSec();
            Event e(msg->events[i].y, msg->events[i].x, time);
            dataset->event_array.push_back(e);
        }
        std::cout << "Event ts = " << msg->header.stamp << "\n";
    }

    void spin() {
        cv::namedWindow(this->window_name, cv::WINDOW_NORMAL);

        this->frame.dataset_handle->modified = true;
        this->frame.dataset_handle->init_GUI();
        const uint8_t nmodes = 4;
        uint8_t vis_mode = 0;

        int code = 0; // Key code
        while (code != 27) {
            {
            const std::lock_guard<std::mutex> lock(this->mutex);
            code = cv::waitKey(1);

            this->frame.dataset_handle->handle_keys(code, vis_mode, nmodes);
            this->frame.dataset_handle->modified = true;

            // Register data
            for (auto &cl : dataset->clouds) {
                dataset->obj_tjs[cl.first].clear();
                dataset->obj_tjs[cl.first].add(ros::Time(0), cl.second->get_last_pos());
            }

            for (auto &obj_tj : dataset->obj_tjs) {
                if (obj_tj.second.size() == 0) continue;
                this->frame.add_object_pos_id(obj_tj.first, 0);
            }

            if (this->images_received > 0) {
                this->frame.dataset_handle->res_x = dataset->images[0].rows;
                this->frame.dataset_handle->res_y = dataset->images[0].cols;
                this->frame.add_img(dataset->images[0]);
            }

            if (dataset->cam_tj.size() > 0)
                this->frame.generate();

            if (dataset->event_array.size() > 0)
                this->frame.event_slice_ids = std::make_pair(0, dataset->event_array.size() - 1);

            // Get visualization out
            cv::Mat img;
            switch (vis_mode) {
                default:
                case 0: img = this->frame.get_visualization_mask(true); break;
                case 1: img = this->frame.get_visualization_mask(false); break;
                case 2: img = this->frame.get_visualization_depth(true); break;
                case 3: img = this->frame.get_visualization_event_projection(true); break;
            }

            if (img.cols == 0 || img.rows == 0)
                continue;

            cv::resize(img, img, cv::Size(), 0.5, 0.5);
            cv::imshow(this->window_name, img);

            // Publish image
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->frame.img).toImageMsg();
            res_img_pub.publish(msg);
            } // mutex

            ros::spinOnce();
            r.sleep();
        }

        cv::destroyAllWindows();
    }
};


int RGBCameraVisualizer::uid = 0;


void camera_pos_cb(const vicon::Subject& subject) {
    dataset->cam_tj.clear();
    dataset->cam_tj.add(ros::Time(0), subject);
    std::cout << "Vicon ts = " << subject.header.stamp << "\n";
}


int main (int argc, char** argv) {

    // Initialize ROS
    std::string node_name = "datagen_online";
    ros::init (argc, argv, node_name);
    ros::NodeHandle nh("~");

    std::string dataset_folder = "";
    if (!nh.getParam("folder", dataset_folder)) {
        std::cerr << "No configuration folder specified!" << std::endl;
        return -1;
    }

    float FPS = 40.0;
    if (!nh.getParam("fps", FPS)) FPS = 40;

    bool no_background = true;
    if (!nh.getParam("no_bg", no_background)) no_background = true;

    bool with_images = true;
    if (!nh.getParam("with_images", with_images)) with_images = true;
    else std::cout << _yellow("With 'with_images' option, the datased will be visualized at image framerate.") << std::endl;

    // Camera name
    std::string camera_name = "";
    if (!nh.getParam("camera_name", camera_name)) camera_name = "main_camera";

    // Read dataset configuration files
    dataset = std::make_shared<Dataset>();
    if (!dataset->init(nh, dataset_folder, camera_name))
        return -1;

    image_transport::ImageTransport it(nh);
    res_img_pub = it.advertise("projector/image", 1);

    // Load 3D models
    std::string path_to_self = ros::package::getPath("evimo");

    if (!no_background) {
        dataset->background = std::make_shared<StaticObject>(path_to_self + "/objects/room");
        dataset->background->transform(dataset->bg_E);
    }

    ros::Subscriber cam_sub = nh.subscribe(dataset->cam_pos_topic, 0, camera_pos_cb);
    RGBCameraVisualizer(nh, FPS, dataset);

    ros::shutdown();
    return 0;
}
