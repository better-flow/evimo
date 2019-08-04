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

// PROPHESEE
#include <prophesee_event_msgs/PropheseeEvent.h>
#include <prophesee_event_msgs/PropheseeEventBuffer.h>

// Local includes
#include <dataset.h>
#include <object.h>
#include <trajectory.h>
#include <dataset_frame.h>
#include <annotation_backprojector.h>

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

    void set_slider(int id) {
        this->frame_id = id < this->frames->size() - 1 ? id : this->frames->size() - 1;
        this->frame_id = this->frame_id < 0 ? 0 : this->frame_id;
        cv::setTrackbarPos("frame", "Frames", this->frame_id);
        Dataset::modified = true;
    }

    void spin() {
        cv::namedWindow("Frames", cv::WINDOW_NORMAL);
        cv::createTrackbar("frame", "Frames", &frame_id, frames->size() - 1, on_trackbar);

        Dataset::modified = true;
        Dataset::init_GUI();
        const uint8_t nmodes = 4;
        uint8_t vis_mode = 0;

        bool enable_3D = false;
        std::shared_ptr<Backprojector> bp;

        int code = 0; // Key code
        while (code != 27) {
            code = cv::waitKey(1);
            if (bp) bp->maybeViewerSpinOnce();

            Dataset::handle_keys(code, vis_mode, nmodes);

            if (code == 39) { // '''
                this->set_slider(this->frame_id + 1);
            }

            if (code == 59) { // ';'
                this->set_slider(this->frame_id - 1);
            }

            /*
            if (code == 99) { // 'c'
                Dataset::modified = true;
            }
            */

            if (!Dataset::modified) continue;
            Dataset::modified = false;

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

            cv::imshow("Frames", img);

            if (!bp) {
                //bp = std::make_shared<Backprojector>(f.get_timestamp(), 5, 10);
                //bp->initViewer();
            }

/*
            if (code == 99) { // 'c'
                enable_3D = !enable_3D;
                if (enable_3D) {
                    bp = std::make_shared<Backprojector>(f.get_timestamp(), 0.1, 100);
                    bp->initViewer();
                }
            }
            */

            //bp.visualize_parallel();

            if (bp) bp->generate();
        }

        cv::destroyAllWindows();
    }

    static void on_trackbar(int, void*) {
        Dataset::modified = true;
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
    bool through_mode = false, generate = true;
    int show = -1;
    if (!nh.getParam(node_name + "/fps", FPS)) FPS = 40;
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
    if (!Dataset::init(dataset_folder, (unsigned int)res_x, (unsigned int)res_y))
        return -1;

    // Load 3D models
    std::string path_to_self = ros::package::getPath("event_imo_datagen");

    if (!no_background) {
        Dataset::background = std::make_shared<StaticObject>(path_to_self + "/objects/room");
        Dataset::background->transform(Dataset::bg_E);
    }

    if (Dataset::enabled_objects.find(1) != Dataset::enabled_objects.end()) {
        Dataset::clouds[1] = std::make_shared<ViObject>(nh, path_to_self + "/objects/toy_car", 1);
        if (!nh.getParam(node_name + "/obj_pose_topic_0", obj_pose_topics[1])) obj_pose_topics[1] = "/vicon/Object_1";
    }

    if (Dataset::enabled_objects.find(2) != Dataset::enabled_objects.end()) {
        Dataset::clouds[2] = std::make_shared<ViObject>(nh, path_to_self + "/objects/toy_plane", 2);
        if (!nh.getParam(node_name + "/obj_pose_topic_1", obj_pose_topics[2])) obj_pose_topics[2] = "/vicon/Object_2";
    }

    if (Dataset::enabled_objects.find(3) != Dataset::enabled_objects.end()) {
        Dataset::clouds[3] = std::make_shared<ViObject>(nh, path_to_self + "/objects/cup", 3);
        if (!nh.getParam(node_name + "/obj_pose_topic_2", obj_pose_topics[3])) obj_pose_topics[3] = "/vicon/Object_3";
    }

    // Extract topics from bag
    auto &cam_tj  = Dataset::cam_tj;
    auto &obj_tjs = Dataset::obj_tjs;
    auto &images = Dataset::images;
    auto &image_ts = Dataset::image_ts;
    std::map<int, vicon::Subject> obj_cloud_to_vicon_tf;

    uint64_t n_events = 0;
    for (auto &m : view) {
        if (m.getTopic() == cam_pose_topic) {
            auto msg = m.instantiate<vicon::Subject>();
            if (msg == NULL) continue;
            cam_tj.add(msg->header.stamp + ros::Duration(Dataset::get_time_offset_pose_to_host()), *msg);
            continue;
        }

        for (auto &p : obj_pose_topics) {
            if (m.getTopic() != p.second) continue;
            auto msg = m.instantiate<vicon::Subject>();
            if (msg == NULL) break;
            if (msg->occluded) break;
            obj_tjs[p.first].add(msg->header.stamp + ros::Duration(Dataset::get_time_offset_pose_to_host()), *msg);
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
            image_ts.push_back(msg->header.stamp + ros::Duration(Dataset::get_time_offset_image_to_host()));
        }
    }

    if (with_images && images.size() == 0) {
        std::cout << _red("No images found! Reverting 'with_images' to 'false'") << std::endl;
        with_images = false;
    }

    auto &event_array = Dataset::event_array;
    event_array.resize(n_events);

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
                       ros::Duration(Dataset::get_time_offset_event_to_host())).toNSec();
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
        if (Dataset::clouds.find(obj_tj.first) == Dataset::clouds.end()) {
            std::cout << "\t\t" << _red("No pointcloud for trajectory! ") << "oid = " << obj_tj.first << std::endl;
            continue;
        }
        Dataset::clouds[obj_tj.first]->init_cloud_to_vicon_tf(obj_cloud_to_vicon_tf[obj_tj.first]);
    }

    // Force the first timestamp of the event cloud to be 0
    // trajectories
    auto time_offset = first_event_message_ts + ros::Duration(Dataset::get_time_offset_event_to_host());
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
        uint64_t ts_low  = (ref_ts < Dataset::slice_width) ? 0 : (ref_ts - Dataset::slice_width / 2.0) * 1000000000;
        uint64_t ts_high = (ref_ts + Dataset::slice_width / 2.0) * 1000000000;
        while (event_low  < event_array.size() - 1 && event_array[event_low].timestamp  < ts_low)  event_low ++;
        while (event_high < event_array.size() - 1 && event_array[event_high].timestamp < ts_high) event_high ++;

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
        auto &frame = frames.back();

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
