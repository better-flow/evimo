#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iomanip>
#include <iostream>

#include <ros/ros.h>
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


#include "running_average.h"


#ifndef OBJECT_H
#define OBJECT_H


// Main class
class StaticObject {
protected:
    std::string folder, name, cloud_fname;

    pcl::PointCloud<pcl::PointXYZRGB> *obj_cloud;
    pcl::PointCloud<pcl::PointXYZRGB> *obj_cloud_transformed;
    pcl::PointCloud<pcl::PointXYZRGB> *obj_cloud_camera_frame;

    tf::Transform s_transform, last_to_camcenter;

public:
    StaticObject (std::string folder_) :
        folder(folder_),
        obj_cloud(new pcl::PointCloud<pcl::PointXYZRGB>),
        obj_cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>),
        obj_cloud_camera_frame(new pcl::PointCloud<pcl::PointXYZRGB>) {

        std::cout << "Initializing static object..." << std::endl;

        this->cloud_fname  = this->folder + "/model.ply";
        this->s_transform.setIdentity();
        this->last_to_camcenter.setIdentity();

        std::string fname = this->cloud_fname.substr(0, this->cloud_fname.find_last_of("."));
        std::string fext  = this->cloud_fname.substr(this->cloud_fname.find_last_of(".") + 1);

        if (fext == "pcd") {
            std::cout << "Reading as .pcd...\n";
            pcl::io::loadPCDFile(this->cloud_fname, *(this->obj_cloud));
        } else if (fext == "ply") {
            std::cout << "Reading as .ply...\n";
            pcl::io::loadPLYFile(this->cloud_fname, *(this->obj_cloud));
        } else {
            std::cout << "Unsupported file format: " << fext << "\n";
            return;
        }
        std::cout << "Read " << obj_cloud->size() << " points\n";

        obj_cloud_camera_frame->header.frame_id = "/camera_center";
        obj_cloud->header.frame_id = "/map";

        std::cout << "=====================================" << std::endl << std::endl;
    }

    // Camera pose update
    bool update_camera_pose(tf::Transform &to_camcenter) {
        this->last_to_camcenter = to_camcenter;
        auto to_cs_inv = this->last_to_camcenter.inverse() * this->s_transform;
        pcl_ros::transformPointCloud(*(this->obj_cloud), *(this->obj_cloud_camera_frame), to_cs_inv);
        return true;
    }

    // A separate method, for offline prcessing
    auto transform_to_camframe(const tf::Transform &cam_tf) {
        this->last_to_camcenter = cam_tf;
        auto full_tf = this->get_tf_in_camera_frame(cam_tf);
        auto out_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        pcl_ros::transformPointCloud(*(this->obj_cloud), *(out_cloud), full_tf);
        return out_cloud;
    }

    tf::Transform get_tf_in_camera_frame(const tf::Transform &cam_tf) {
        return this->last_to_camcenter.inverse() * this->s_transform;
    }

    void transform(tf::Transform s_transform) {
        this->s_transform = s_transform;
    }

    pcl::PointCloud<pcl::PointXYZRGB>* get_cloud() {
        return this->obj_cloud_camera_frame;
    }

    tf::Transform get_to_camcenter() {
        return this->last_to_camcenter;
    }

    tf::Transform get_static() {
        return this->s_transform;
    }
};


class ViObject {
protected:
    ros::NodeHandle n_;
    image_transport::ImageTransport it_;

    std::string folder, name, pos_topic = "", pointcloud_unit;
    int id;

    bool no_mesh; // This object has no pointcloud

    ros::Publisher obj_pub, vis_pub;
    ros::Subscriber obj_sub;

    pcl::PointCloud<pcl::PointXYZRGB> *obj_cloud;
    pcl::PointCloud<pcl::PointXYZRGB> *obj_cloud_transformed;
    pcl::PointCloud<pcl::PointXYZRGB> *obj_cloud_camera_frame;
    pcl::PointCloud<pcl::PointXYZRGB> *obj_markerpos;

    vicon::Subject last_pos;
    long int poses_received;

public:
    ViObject (ros::NodeHandle n_, std::string folder_) :
        n_(n_), it_(n_), folder(folder_), name(""), id(255), no_mesh(true),
        obj_cloud(new pcl::PointCloud<pcl::PointXYZRGB>),
        obj_cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>),
        obj_cloud_camera_frame(new pcl::PointCloud<pcl::PointXYZRGB>),
        obj_markerpos(new pcl::PointCloud<pcl::PointXYZRGB>),
        poses_received(0) {

        auto config_fname   = this->folder + "/config.txt";
        auto settings_fname = this->folder + "/settings.txt";
        auto cloud_fname    = std::string("");
        this->pointcloud_unit = "m"; // everything in meters by default

        std::ifstream settings;
        settings.open(settings_fname, std::ifstream::in);
        if (!settings.is_open()) {
            std::cout << _red("Could not open object setitngs file file at ")
                      << settings_fname << "!" << std::endl;
            this->id = -1;
            return;
        }

        // Read settings file
        const std::string& delims = ":";
        while (settings.good()) {
            std::string line;
            std::getline(settings, line);
            line = trim(line);
            auto sep = line.find_first_of(delims);

            std::string key   = line.substr(0, sep);
            std::string value = line.substr(sep + 1);
            key = trim(key);
            value = trim(value);

            if (key == "id")
                this->id = std::stoi(value);

            if (key == "name")
                this->name = value;

            if (key == "mesh" && value != "none" && value != "")
                cloud_fname = this->folder + "/" + value;

            if (key == "ros_pos_topic")
                this->pos_topic = value;

            if (key == "unit")
                this->pointcloud_unit = value;
        }
        float metric_scale = 1.0;
        if (this->pointcloud_unit == "mm")
            metric_scale = 0.001;
        if (this->pointcloud_unit == "cm")
            metric_scale = 0.01;

        if (this->pos_topic == "") {
            std::cout << _red("No ros pose topic specified for object in ")
                      << settings_fname << "!" << std::endl;
            this->id = -1;
            return;
        }

        if (this->id <= 0 || this->id >= 255) {
            std::cout << _red("No object id (in range 1..254) specified for object in ")
                      << settings_fname << "!" << std::endl;
            this->id = -1;
            return;
        }

        if (this->name == "") {
            this->name = "noname_" + std::to_string(this->id);
        }

        if (cloud_fname != "") {
            this->no_mesh = false;
        }

        std::cout << "Initializing " << this->name << "; id = " << this->id << std::endl;
        std::cout << "All coordinates are assumed to be in " << _yellow(this->pointcloud_unit) << std::endl;

        this->vis_pub = n_.advertise<visualization_msgs::MarkerArray>("/ev_imo/markers", 0);
        this->obj_sub = n_.subscribe(this->pos_topic, 0, &ViObject::vicon_pos_cb, this);

        if (no_mesh) {
            std::cout << "\tconfigured as 'no mesh'" << std::endl;
            std::cout << "=====================================" << std::endl << std::endl;
            return;
        }

        this->obj_pub = n_.advertise<pcl::PointCloud<pcl::PointXYZRGB>> ("/ev_imo/" + this->name, 100);

        std::string fname = cloud_fname.substr(0, cloud_fname.find_last_of("."));
        std::string fext  = cloud_fname.substr(cloud_fname.find_last_of(".") + 1);

        if (fext == "pcd") {
            std::cout << "Reading as .pcd...\n";
            pcl::io::loadPCDFile(cloud_fname, *(this->obj_cloud));
        } else if (fext == "ply") {
            std::cout << "Reading as .ply...\n";
            pcl::io::loadPLYFile(cloud_fname, *(this->obj_cloud));
        } else {
            std::cout << _red("Unsupported file format: ") << fext << "!" << std::endl;
            this->id = -1;
            return;
        }
        std::cout << "Read " << obj_cloud->size() << " points\n";
        if (obj_cloud->size() == 0) {
            std::cout << _red("Read zero points from ") << cloud_fname << "!" << std::endl;
            this->id = -1;
            return;
        }

        for (auto &p : *(this->obj_cloud)) {p.x *= metric_scale; p.y *= metric_scale; p.z *= metric_scale;} // scale the cloud

        this->obj_cloud_camera_frame->header.frame_id = "/camera_center";
        this->obj_cloud_transformed->header.frame_id = "/camera_center";
        this->obj_cloud->header.frame_id = "/map";

        // Read marker config file
        std::ifstream cfg(config_fname, std::ifstream::in);
        int marker_id = -1;
        double x = 0, y = 0, z = 0;
        while (cfg.good()) {
            if (!(cfg >> marker_id >> x >> y >> z))
                continue;
            pcl::PointXYZRGB p;
            p.x = x * metric_scale; p.y = y * metric_scale; p.z = z * metric_scale;
            p.r = 255; p.g = 255; p.b = 255;
            this->obj_markerpos->push_back(p);
        }

        std::cout << this->name << " initialized with " << this->obj_markerpos->size() << " markers:" << std::endl;
        std::cout << " ID\t|\t\tcoordinate" << std::endl;
        marker_id = 0;
        for (auto &p : *(this->obj_markerpos)) {
            std::cout << " " << marker_id << "\t|\t" << p.x << "\t" << p.y << "\t" << p.z << std::endl;
            marker_id ++;
        }
        std::cout << "=====================================" << std::endl << std::endl;

        cfg.close();
        settings.close();
    }

    // Callbacks
    void vicon_pos_cb(const vicon::Subject& subject) {
        this->last_pos = subject;

        if (this->poses_received % 200 == 0)
            this->convert_to_vicon_tf(subject);

        this->poses_received ++;
        if (this->poses_received % 20 != 0)
            return;

        auto &markers = subject.markers;
        visualization_msgs::MarkerArray vis_markers;

        int i = 0;
        for (auto &marker : markers) {
            auto vis_marker = this->get_generic_marker(i);
            vis_marker.ns = marker.name;
            vis_marker.pose.position = marker.position;
            vis_markers.markers.push_back(vis_marker);
            ++i;
        }

        this->vis_pub.publish(vis_markers);
    }

    void convert_to_vicon_tf(const vicon::Subject& subject) {
        if (subject.occluded)
            std::cout << "Computing cloud_to_vicon_tf on occluded vicon track!" << std::endl;

        //if (this->no_mesh)
        //    return;

        auto &markers = subject.markers;
        if (this->no_mesh) {
            this->obj_markerpos->clear();
            this->obj_cloud->clear();
            for (auto &marker : markers) {
                pcl::PointXYZRGB p;
                p.x = marker.position.x;
                p.y = marker.position.y;
                p.z = marker.position.z;
                this->obj_markerpos->push_back(p);
                this->obj_cloud->push_back(p);
            }
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr target(new pcl::PointCloud<pcl::PointXYZRGB>);
        int idx__ = 0;
        for (auto &marker : markers) {
            pcl::PointXYZRGB p;
            p.x = marker.position.x;
            p.y = marker.position.y;
            p.z = marker.position.z;
            target->push_back(p);
            idx__ += 1;

            // there might be unregistered trackable points on the object
            if (idx__ >= this->obj_markerpos->size()) break;
        }

        if (target->size() != this->obj_markerpos->size()) {
            std::cout << "Marker to gt size mismatch!" << std::endl;
            std::cout << "\tMarkers from vicon: " << target->size()
                      << " vs. in db: " << this->obj_markerpos->size() << std::endl;
        }

        Eigen::Matrix4f SVD;
        const pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB, pcl::PointXYZRGB> trans_est_svd;
        trans_est_svd.estimateRigidTransformation(*(this->obj_markerpos), *target, SVD);
        auto svd_tf = ViObject::mat2tf(SVD);
        auto p = ViObject::subject2tf(subject);
        auto inv_p = p.inverse();

        /*
        std::cout << "Estimating marker2vicon svd error:\n";
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr markers_vicon(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl_ros::transformPointCloud(*(this->obj_markerpos), *markers_vicon, svd_tf);
        for (int i = 0; i < this->obj_markerpos->size(); ++i) {
            auto &p0 = markers_vicon->at(i);
            auto &p1 = target->at(i);
            std::cout << "\t" << i << ":\t" << p0.x << " " << p0.y << " " << p0.z << "\t->\t"
                      << p1.x << " " << p1.y << " " << p1.z << "\t("
                      << p0.x - p1.x << " " << p0.y - p1.y << " " << p0.z - p1.z << ")\n";
        }
        */

        //                                    in                      out                tf
        pcl_ros::transformPointCloud(*(this->obj_markerpos), *(this->obj_markerpos), inv_p * svd_tf);
        pcl_ros::transformPointCloud(*(this->obj_cloud), *(this->obj_cloud), inv_p * svd_tf);
    }

    // Camera pose update
    bool update_camera_pose(tf::Transform &to_camcenter) {
        if (this->poses_received == 0)
            return false;

        if (this->last_pos.occluded)
            return false;

        if (this->no_mesh)
            return true;

        auto to_cs_inv = to_camcenter.inverse();
        auto full_tf = to_cs_inv * subject2tf(last_pos);

        pcl_ros::transformPointCloud(*(this->obj_cloud), *(this->obj_cloud_transformed), subject2tf(last_pos));
        obj_pub.publish(this->obj_cloud_transformed->makeShared());

        pcl_ros::transformPointCloud(*(this->obj_cloud), *(this->obj_cloud_camera_frame), full_tf);
        return true;
    }

    // A separate method, for offline processing
    auto transform_to_camframe(const tf::Transform &cam_tf, const tf::Transform &obj_tf) {
        auto full_tf = this->get_tf_in_camera_frame(cam_tf, obj_tf);
        auto out_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        pcl_ros::transformPointCloud(*(this->obj_cloud), *(out_cloud), full_tf);
        return out_cloud;
    }

    auto marker_cl_in_camframe(const tf::Transform &cam_tf) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr target(new pcl::PointCloud<pcl::PointXYZRGB>);
        auto &markers = this->last_pos.markers;
        for (auto &marker : markers) {
            pcl::PointXYZRGB p;
            p.x = marker.position.x;
            p.y = marker.position.y;
            p.z = marker.position.z;
            target->push_back(p);
        }

        auto full_tf = cam_tf.inverse();
        auto out_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        pcl_ros::transformPointCloud(*(target), *(out_cloud), full_tf);
        return out_cloud;
    }

    tf::Transform get_tf_in_camera_frame(const tf::Transform &cam_tf, const tf::Transform &obj_tf) {
        auto inv_cam = cam_tf.inverse();
        auto full_tf = inv_cam * obj_tf;
        return full_tf;
    }

    float get_visibility() {
        float visibility = 0;
        for (auto &m : this->last_pos.markers)
            if (!m.occluded) visibility += 1.0;
        visibility /= this->last_pos.markers.size();
        visibility *= 100.0;
        return visibility;
    }

    // Helpers
    visualization_msgs::Marker get_generic_marker(int id_ = 0) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "/map";
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

        marker.color.r = float((id_ + 0) % 3) / 2.0;
        marker.color.g = float((id_ + 1) % 3) / 2.0;
        marker.color.b = float((id_ + 2) % 3) / 2.0;

        return marker;
    }

    pcl::PointCloud<pcl::PointXYZRGB>* get_cloud() {
        return this->obj_cloud_camera_frame;
    }

    int get_id() {
        return this->id;
    }

    std::string get_pose_topic() {
        return this->pos_topic;
    }

    vicon::Subject get_last_pos() {
        return this->last_pos;
    }

    bool has_no_mesh() {
        return this->no_mesh;
    }

    static vicon::Subject tf2subject(tf::Transform &transform) {
        vicon::Subject ret;
        auto q = transform.getRotation();
        auto T = transform.getOrigin();
        ret.orientation.w = q.getW();
        ret.orientation.x = q.getX();
        ret.orientation.y = q.getY();
        ret.orientation.z = q.getZ();
        ret.position.x = T.getX();
        ret.position.y = T.getY();
        ret.position.z = T.getZ();
        return ret;
    }

    static tf::Transform mat2tf(Eigen::Matrix4f &Tm) {
        tf::Vector3 origin;
        origin.setValue(static_cast<double>(Tm(0,3)),static_cast<double>(Tm(1,3)),static_cast<double>(Tm(2,3)));
        tf::Matrix3x3 tf3d;
        tf3d.setValue(static_cast<double>(Tm(0,0)), static_cast<double>(Tm(0,1)), static_cast<double>(Tm(0,2)),
                      static_cast<double>(Tm(1,0)), static_cast<double>(Tm(1,1)), static_cast<double>(Tm(1,2)),
                      static_cast<double>(Tm(2,0)), static_cast<double>(Tm(2,1)), static_cast<double>(Tm(2,2)));
        return tf::Transform (tf3d, origin);
    }

    static tf::Transform subject2tf(const vicon::Subject& p) {
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(p.position.x, p.position.y, p.position.z));
        tf::Quaternion q(p.orientation.x,
                         p.orientation.y,
                         p.orientation.z,
                         p.orientation.w);
        transform.setRotation(q);
        return transform;
    }
};

#endif // OBJECT_H
