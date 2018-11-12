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
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>
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


pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined;
static int received = 0;

ros::Publisher combined_pub, vis_pub, vis_pub_range;
ros::Subscriber cloud_pos_sub;
ros::Subscriber cloud_sub;

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;


// Last known pose
vicon::Subject last_pos;

void cloud_pos_cb(const vicon::Subject& subject) {
    last_pos = subject;
}


visualization_msgs::Marker get_generic_marker() {
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
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    return marker;
}


void cloud_cb(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    received += 1;

    if (last_pos.occluded) {
        std::cout << "Pos unavailable!\n";
        return;
    }

    // Time sync
    auto cl_time  = pcl_conversions::fromPCL(cloud->header.stamp);
    auto pos_time = last_pos.header.stamp;

    visualization_msgs::MarkerArray vis_markers;

    auto vis_marker = get_generic_marker();
    vis_marker.ns = "Xtion";
    vis_marker.pose.position = last_pos.position;
    vis_marker.color.r = 1;
    vis_marker.color.g = 0;
    vis_marker.color.b = 1;

    vis_markers.markers.push_back(vis_marker);
    vis_pub.publish(vis_markers);

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(last_pos.position.x, last_pos.position.y, last_pos.position.z));
    tf::Quaternion q(last_pos.orientation.x,
                     last_pos.orientation.y,
                     last_pos.orientation.z,
                     last_pos.orientation.w); 
    transform.setRotation(q);
    static tf::TransformBroadcaster tf_br;
    tf_br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/vicon", "/Xtion"));
    auto inv_transform = transform.inverse();

    sensor_msgs::Range cone;
    cone.field_of_view = 1;
    cone.min_range = 0;
    cone.max_range = 5;
    cone.range = 3;
    cone.header.frame_id = "/Xtion";
    vis_pub_range.publish(cone);
    // ============================================


    // Optical frame to vicon frame
    for (auto &p: *cloud) {
        pcl::PointXYZ p_(p.z, -p.x, -p.y);
        p.x = p_.x;
        p.y = p_.y;
        p.z = p_.z;
    }
    pcl_ros::transformPointCloud(*cloud, *cloud, transform);
    cloud->header.frame_id = "/vicon";
        
    combined_pub.publish(cloud);

    if (received % 10 == 0) {
        std::string fname = "/home/ncos/Desktop/frames/frame_" + std::to_string(received / 10) + ".ply";
        std::vector<int> ind_;
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ocloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::removeNaNFromPointCloud(*cloud, *ocloud, ind_);

        pcl::io::savePLYFileBinary(fname, *ocloud);
        //clouds.push_back(cloud);
    }

    combined->clear();

    if (false) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
        *tmp += *combined;
        *tmp += *cloud;
        combined->clear();

        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud (tmp);
        sor.setLeafSize (0.01f, 0.01f, 0.01f);
        sor.filter (*combined);
    } else {
        *combined += *cloud;
    }

    std::cout << "\tCurrent cloud: " << combined->size() << " points;"
              << " cloud time: " << cl_time
              << " vicon time: " << pos_time
              << " diff: " << cl_time - pos_time << std::endl;
}


void save_result() {
    // Apply voxel grid filter
    std::cout << "Downsampling the cloud..." << std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud (combined);
    sor.setLeafSize (0.01f, 0.01f, 0.01f);
    sor.filter (*filtered);

    std::string fname = "/home/ncos/Desktop/room.ply";

    std::cout << "Saving file in: " << fname << std::endl;
    pcl::io::savePLYFileASCII(fname, *filtered);

    int i = 0;
    for (auto &cl : clouds) {
        fname = "/home/ncos/Desktop/frames/frame_" + std::to_string(i) + ".ply";
        pcl::io::savePLYFile(fname, *cl);
        i ++;
    }
}


int main (int argc, char** argv) {
    // Initialize ROS
    ros::init (argc, argv, "vrec");
    ros::NodeHandle nh;
  
    // Create ROS subscribers / publishers
    vis_pub = nh.advertise<visualization_msgs::MarkerArray>("/vrec/markers", 0);
    vis_pub_range = nh.advertise<sensor_msgs::Range>("/vrec/markers_range",  0);
    combined_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>> ("/vrec/points_combined", 0);

    cloud_pos_sub = nh.subscribe("/vicon/Xtion", 0, cloud_pos_cb);
    cloud_sub = nh.subscribe("/camera/depth_registered/points", 0, cloud_cb);

    combined = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    combined->header.frame_id = "/vicon";

    // Spin
    ros::spin();
    //save_result();
    //ros::spin();

    /*
    ros::Rate r(200);
    while (ros::ok()) {
        ros::spinOnce();
        r.sleep();

        if (received == 800) {
            save_result();            
            //return 0;
        }
    }*/

    return 0;
}
