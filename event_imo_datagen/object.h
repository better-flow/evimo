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
        obj_cloud->header.frame_id = "/vicon";

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

    std::string folder, name, cloud_fname, config_fname;
    int id;

    ros::Publisher obj_pub, vis_pub;
    ros::Subscriber obj_sub;

    pcl::PointCloud<pcl::PointXYZRGB> *obj_cloud;
    pcl::PointCloud<pcl::PointXYZRGB> *obj_cloud_transformed;
    pcl::PointCloud<pcl::PointXYZRGB> *obj_cloud_camera_frame;
    pcl::PointCloud<pcl::PointXYZRGB> *obj_markerpos;

    //Eigen::Matrix4f LAST_SVD;
    vicon::Subject last_pos;
    long int poses_received;

    PoseManager pose_manager;

public:
    ViObject (ros::NodeHandle n_, std::string folder_, int id_) :
        n_(n_), it_(n_), folder(folder_), id(id_),
        obj_cloud(new pcl::PointCloud<pcl::PointXYZRGB>),
        obj_cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>),
        obj_cloud_camera_frame(new pcl::PointCloud<pcl::PointXYZRGB>),
        obj_markerpos(new pcl::PointCloud<pcl::PointXYZRGB>),
        poses_received(0) {

        this->name = "Object_" + std::to_string(this->id);
        std::cout << "Initializing " << this->name << std::endl;

        this->obj_pub = n_.advertise<pcl::PointCloud<pcl::PointXYZRGB>> ("/ev_imo/" + this->name, 100);
        this->vis_pub = n_.advertise<visualization_msgs::MarkerArray>("/ev_imo/markers", 0);
        this->obj_sub = n_.subscribe("/vicon/" + this->name, 0, &ViObject::vicon_pos_cb, this);

        this->cloud_fname  = this->folder + "/model.ply";
        this->config_fname = this->folder + "/config.txt";

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

        this->obj_cloud_camera_frame->header.frame_id = "/camera_center";
        this->obj_cloud_transformed->header.frame_id = "/camera_center";
        this->obj_cloud->header.frame_id = "/vicon";
        //this->LAST_SVD = Eigen::MatrixXf::Identity(4, 4);

        std::ifstream cfg(this->config_fname, std::ifstream::in);
        int id = -1;
        double x = 0, y = 0, z = 0;
        while (!cfg.eof()) {
            if (!(cfg >> id >> x >> y >> z))
                continue;
            pcl::PointXYZRGB p;
            p.x = x; p.y = y; p.z = z;
            p.r = 255; p.g = 255; p.b = 255;
            this->obj_markerpos->push_back(p);
        }

        std::cout << this->name << " initialized with " << this->obj_markerpos->size() << " markers:" << std::endl;
        std::cout << " ID\t|\t\tcoordinate" << std::endl;
        id = 0;
        for (auto &p : *(this->obj_markerpos)) {
            std::cout << " " << id << "\t|\t" << p.x << "\t" << p.y << "\t" << p.z << std::endl;
            id ++;
        }
        std::cout << "=====================================" << std::endl << std::endl;
    }

    // Callbacks
    void vicon_pos_cb(const vicon::Subject& subject) {
        this->last_pos = subject;

        if (this->poses_received == 0)
            this->convert_to_vicon_tf(subject);

        this->poses_received ++;
        this->pose_manager.push_back(subject, subject);

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

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr target(new pcl::PointCloud<pcl::PointXYZRGB>);
        auto &markers = subject.markers;
        for (auto &marker : markers) {
            pcl::PointXYZRGB p;
            p.x = marker.position.x;
            p.y = marker.position.y;
            p.z = marker.position.z;
            target->push_back(p);
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

        pcl_ros::transformPointCloud(*(this->obj_cloud), *(this->obj_cloud), inv_p * svd_tf);
    }

    // Camera pose update
    bool update_camera_pose(tf::Transform &to_camcenter) {
        if (this->poses_received == 0)
            return false;

        if (this->last_pos.occluded)
            return false;

        auto to_cs_inv = to_camcenter.inverse();
        auto full_tf = to_cs_inv * subject2tf(last_pos);

        pcl_ros::transformPointCloud(*(this->obj_cloud), *(this->obj_cloud_transformed), subject2tf(last_pos));
        obj_pub.publish(this->obj_cloud_transformed->makeShared());

        pcl_ros::transformPointCloud(*(this->obj_cloud), *(this->obj_cloud_camera_frame), full_tf);

        return true;
    }

    // A separate method, for offline prcessing
    auto transform_to_camframe(const tf::Transform &cam_tf, const tf::Transform &obj_tf) {
        auto full_tf = this->get_tf_in_camera_frame(cam_tf, obj_tf);
        auto out_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        pcl_ros::transformPointCloud(*(this->obj_cloud), *(out_cloud), full_tf);
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
    visualization_msgs::Marker get_generic_marker(int id = 0) {
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

        marker.color.r = float((id + 0) % 3) / 2.0;
        marker.color.g = float((id + 1) % 3) / 2.0;
        marker.color.b = float((id + 2) % 3) / 2.0;

        return marker;
    }

    pcl::PointCloud<pcl::PointXYZRGB>* get_cloud() {
        return this->obj_cloud_camera_frame;
    }

    int get_id() {
        return this->id;
    }

    vicon::Subject get_last_pos() {
        return this->last_pos;
    }

    auto &get_pm() {
        return this->pose_manager;
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
