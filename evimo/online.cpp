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


class RGBCameraVisualizer {
protected:
    DatasetFrame frame;
    ros::Rate r;
    static int uid;
    std::string window_name, topic;

    uint64_t images_received;
    ros::Subscriber sub;

public:
    RGBCameraVisualizer(ros::NodeHandle &nh, float FPS, std::string topic_)
        : frame(0, 0, 0), r(FPS), topic(topic_), images_received(0) {
        this->window_name = "RGBFrames_" + std::to_string(uid);
        uid += 1;


        this->sub = nh.subscribe(this->topic, 0, &RGBCameraVisualizer::sub_cb, this);
        this->spin();
    }

    void sub_cb(const sensor_msgs::ImageConstPtr& msg) {
        Dataset::images.resize(1);
        if (msg->encoding == "8UC1") {
            sensor_msgs::Image img = *msg;
            img.encoding = "mono8";

            Dataset::images[0] = cv_bridge::toCvCopy(img, "bgr8")->image;
        } else {
            Dataset::images[0] = cv_bridge::toCvShare(msg, "bgr8")->image;
        }
        this->images_received ++;
    }

    void spin() {
        cv::namedWindow(this->window_name, cv::WINDOW_NORMAL);

        Dataset::modified = true;
        Dataset::init_GUI();
        const uint8_t nmodes = 4;
        uint8_t vis_mode = 0;

        int code = 0; // Key code
        while (code != 27) {
            code = cv::waitKey(1);

            Dataset::handle_keys(code, vis_mode, nmodes);
            Dataset::modified = true;

            // Register data
            for (auto &cl : Dataset::clouds) {
                Dataset::obj_tjs[cl.first].clear();
                Dataset::obj_tjs[cl.first].add(ros::Time(0), cl.second->get_last_pos());
            }

            for (auto &obj_tj : Dataset::obj_tjs) {
                if (obj_tj.second.size() == 0) continue;
                frame.add_object_pos_id(obj_tj.first, 0);
            }

            if (this->images_received > 0) {
                Dataset::res_x = Dataset::images[0].rows;
                Dataset::res_y = Dataset::images[0].cols;
                this->frame.add_img(Dataset::images[0]);
            }

            if (Dataset::cam_tj.size() > 0)
                this->frame.generate();

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

            cv::imshow(this->window_name, img);

            // Publish image
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->frame.img).toImageMsg();
            res_img_pub.publish(msg);

            ros::spinOnce();
            r.sleep();
        }

        cv::destroyAllWindows();
    }
};

int RGBCameraVisualizer::uid = 0;


void camera_pos_cb(const vicon::Subject& subject) {
    Dataset::cam_tj.clear();
    Dataset::cam_tj.add(ros::Time(0), subject);
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
    if (!Dataset::init(nh, dataset_folder, camera_name))
        return -1;

    image_transport::ImageTransport it(nh);
    res_img_pub = it.advertise("projector/image", 1);

    // Load 3D models
    std::string path_to_self = ros::package::getPath("evimo");

    if (!no_background) {
        Dataset::background = std::make_shared<StaticObject>(path_to_self + "/objects/room");
        Dataset::background->transform(Dataset::bg_E);
    }

    /*
    if (Dataset::enabled_objects.find(1) != Dataset::enabled_objects.end()) {
        Dataset::clouds[1] = std::make_shared<ViObject>(nh, path_to_self + "/objects/toy_car", 1);
    }

    if (Dataset::enabled_objects.find(2) != Dataset::enabled_objects.end()) {
        Dataset::clouds[2] = std::make_shared<ViObject>(nh, path_to_self + "/objects/toy_plane", 2);
    }

    if (Dataset::enabled_objects.find(3) != Dataset::enabled_objects.end()) {
        Dataset::clouds[3] = std::make_shared<ViObject>(nh, path_to_self + "/objects/cup", 3);
    }
    */
    //Dataset::clouds[4] = std::make_shared<ViObject>(nh, path_to_self + "/objects/Wand", 4, "Wand", true);

    ros::Subscriber cam_sub = nh.subscribe(Dataset::cam_pos_topic, 0, camera_pos_cb);
    RGBCameraVisualizer(nh, FPS, Dataset::image_topic);

    ros::shutdown();
    return 0;
}
