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
#include <annotation_backprojector.h>

class FrameSequenceVisualizer {
protected:
    std::vector<DatasetFrame> *frames;
    int frame_id;

    TjPlot plotter; // plot trajectories

public:
    FrameSequenceVisualizer(std::vector<DatasetFrame> &frames)
        : frame_id(0), plotter("Trajectories", 4000, 800) {
        std::cout << "Frame Sequence Visuzlizer...\n";
        this->frames = &frames;
        //this->frame_id = this->frames->size() / 2;

        this->plotter.add_trajectory_plot(Dataset::cam_tj);
        for (auto &tj : Dataset::obj_tjs)
            this->plotter.add_trajectory_plot(tj.second);

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
        bool nodist = true;

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

            if (code == 99) { // 'c'
                Dataset::modified = true;
                enable_3D = !enable_3D;
            }

            if (!Dataset::modified) continue;
            Dataset::modified = false;

            auto &f = this->frames->at(this->frame_id);
            f.generate(nodist);

            cv::Mat img;
            switch (vis_mode) {
                default:
                case 0: img = f.get_visualization_mask(true, nodist); break;
                case 1: img = f.get_visualization_mask(false, nodist); break;
                case 2: img = f.get_visualization_depth(true, nodist); break;
                case 3: img = f.get_visualization_event_projection(true, nodist); break;
            }

            cv::imshow("Frames", img);

            this->plotter.add_vertical(f.get_timestamp());
            this->plotter.show();

            if (enable_3D && !bp) {
                bp = std::make_shared<Backprojector>(f.get_timestamp(), 0.4, 200);
                bp->initViewer();
            }

            if (!enable_3D) {
                bp = nullptr;
            }

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
    ros::init (argc, argv, node_name, ros::init_options::AnonymousName);
    ros::NodeHandle nh("~");

    std::string dataset_folder = "";
    if (!nh.getParam("folder", dataset_folder)) {
        std::cerr << "No dataset folder specified!" << std::endl;
        return -1;
    }

    float FPS = 40.0;
    bool generate = false;
    int show = -1;
    bool save_3d = false;
    bool ignore_tj = false;
    if (!nh.getParam("fps", FPS)) FPS = 40;
    if (!nh.getParam("generate", generate)) generate = false;
    if (!nh.getParam("show", show)) show = -1;
    if (!nh.getParam("save_3d", save_3d)) save_3d = false;
    if (!nh.getParam("ignore_trajectories", ignore_tj)) ignore_tj = false;

    float start_time_offset = 0.0, sequence_duration = -1.0;
    if (!nh.getParam("start_time_offset", start_time_offset)) start_time_offset =  0.0;
    if (!nh.getParam("sequence_duration", sequence_duration)) sequence_duration = -1.0;

    bool no_background = true;
    if (!nh.getParam("no_bg", no_background)) no_background = true;

    bool with_images = true;
    if (!nh.getParam("with_images", with_images)) with_images = true;
    else std::cout << _yellow("With 'with_images' option, the datased will be generated at image framerate.") << std::endl;

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

    // Camera name
    std::string camera_name = "";
    if (!nh.getParam("camera_name", camera_name)) camera_name = "main_camera";

    // Read datasset configuration files
    if (!Dataset::init(nh, dataset_folder, camera_name))
        return -1;

    // Load 3D models
    std::string path_to_self = ros::package::getPath("evimo");
    if (!no_background) {
        Dataset::background = std::make_shared<StaticObject>(path_to_self + "/objects/room");
        Dataset::background->transform(Dataset::bg_E);
    }

    // Extract topics from bag
    auto &cam_tj  = Dataset::cam_tj;
    auto &obj_tjs = Dataset::obj_tjs;
    auto &images = Dataset::images;
    auto &image_ts = Dataset::image_ts;
    std::map<int, vicon::Subject> obj_cloud_to_vicon_tf;

    ros::Time bag_start_ts = view.begin()->getTime();
    uint64_t n_events = 0;
    for (auto &m : view) {
        if (m.getTopic() == Dataset::cam_pos_topic) {
            auto msg = m.instantiate<vicon::Subject>();
            if (msg == NULL) continue;
            auto timestamp = msg->header.stamp + ros::Duration(Dataset::get_time_offset_pose_to_host());

            if (start_time_offset > 0 && timestamp < bag_start_ts + ros::Duration(start_time_offset)) continue;
            if (sequence_duration > 0 && timestamp > bag_start_ts + ros::Duration(start_time_offset + sequence_duration)) continue;

            cam_tj.add(timestamp, *msg);
            continue;
        }

        for (auto &p : Dataset::obj_pose_topics) {
            if (m.getTopic() != p.second) continue;
            auto msg = m.instantiate<vicon::Subject>();
            if (msg == NULL) break;
            if (msg->occluded) break;
            auto timestamp = msg->header.stamp + ros::Duration(Dataset::get_time_offset_pose_to_host());

            if (start_time_offset > 0 && timestamp < bag_start_ts + ros::Duration(start_time_offset)) continue;
            if (sequence_duration > 0 && timestamp > bag_start_ts + ros::Duration(start_time_offset + sequence_duration)) continue;

            obj_tjs[p.first].add(timestamp, *msg);
            obj_cloud_to_vicon_tf[p.first] = *msg;
            break;
        }

        if (m.getTopic() == Dataset::event_topic) {
            auto msg = m.instantiate<dvs_msgs::EventArray>();
            auto timestamp = msg->header.stamp;

            if (start_time_offset > 0 && timestamp < bag_start_ts + ros::Duration(start_time_offset)) continue;
            if (sequence_duration > 0 && timestamp > bag_start_ts + ros::Duration(start_time_offset + sequence_duration)) continue;

            if (msg != NULL) {
                n_events += msg->events.size();
            }
            continue;
        }

        if (with_images && (m.getTopic() == Dataset::image_topic)) {
            auto msg = m.instantiate<sensor_msgs::Image>();
            auto timestamp = msg->header.stamp + ros::Duration(Dataset::get_time_offset_image_to_host());

            if (start_time_offset > 0 && timestamp < bag_start_ts + ros::Duration(start_time_offset)) continue;
            if (sequence_duration > 0 && timestamp > bag_start_ts + ros::Duration(start_time_offset + sequence_duration)) continue;

            images.push_back((cv_bridge::toCvShare(msg, "bgr8")->image).clone());
            image_ts.push_back(timestamp);
        }
    }

    if (with_images && images.size() == 0) {
        std::cout << _red("No images found! Reverting 'with_images' to 'false'") << std::endl;
        with_images = false;
    }

    auto &event_array = Dataset::event_array;
    event_array.reserve(n_events);

    uint64_t id = 0;
    ros::Time first_event_ts = ros::Time(0);
    ros::Time first_event_message_ts = ros::Time(0);
    ros::Time last_event_ts = ros::Time(0);
    for (auto &m : view) {
        if (m.getTopic() != Dataset::event_topic)
            continue;

        auto msize = 0;
        auto msg = m.instantiate<dvs_msgs::EventArray>();
        if (msg != NULL) msize = msg->events.size();

        auto timestamp = msg->header.stamp;

        if (start_time_offset > 0 && timestamp < bag_start_ts + ros::Duration(start_time_offset)) continue;
        if (sequence_duration > 0 && timestamp > bag_start_ts + ros::Duration(start_time_offset + sequence_duration)) continue;

        for (uint64_t i = 0; i < msize; ++i) {
            int32_t x = 0, y = 0;
            ros::Time current_event_ts = ros::Time(0);
            int polarity = 0;

            if (msg != NULL) {
                auto &e = msg->events[i];
                current_event_ts = e.ts;
                x = e.x; y = e.y;
                polarity = e.polarity ? 1 : 0;
            }

            if (id == 0) {
                first_event_ts = current_event_ts;
                last_event_ts = current_event_ts;
                first_event_message_ts = msg->header.stamp;// m.getTime();
            } else {
                if (current_event_ts < last_event_ts) {
                    std::cout << _red("Events are not sorted! ")
                              << id << ": " << last_event_ts << " -> "
                              << current_event_ts << std::endl;
                }
                last_event_ts = current_event_ts;
            }

            //auto ts = (first_event_message_ts + (current_event_ts - first_event_ts) +
            //           ros::Duration(Dataset::get_time_offset_event_to_host())).toNSec();

            auto ts = (current_event_ts + ros::Duration(Dataset::get_time_offset_event_to_host())).toNSec();

            event_array.push_back(Event(y, x, ts, polarity));
            id ++;
        }
    }
    n_events = event_array.size();

    std::cout << _green("Read ") << n_events << _green(" events") << std::endl;
    if (ignore_tj) {
        auto t0 = ros::Time(0);
        t0.fromNSec(event_array[0].timestamp);
        auto t1 = ros::Time(0);
        t1.fromNSec(event_array[event_array.size() - 1].timestamp);
        auto npos = int((t1 - t0).toSec() * 100);
        std::cout << npos << "\t" << t0 << " - " << t1 << "\n";

        for (size_t i = 0; i < npos; ++i)
            cam_tj.add(t0 + ros::Duration(float(i) / 100.0), tf::Transform());
    }

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
        Dataset::clouds[obj_tj.first]->convert_to_vicon_tf(obj_cloud_to_vicon_tf[obj_tj.first]);
    }

    // Force the first timestamp of the event cloud to be 0
    // trajectories
    auto time_offset = ros::Time(0);
    if (n_events == 0) {
        time_offset = cam_tj[0].ts;
    } else {
        //time_offset = first_event_message_ts + ros::Duration(Dataset::get_time_offset_event_to_host());
        time_offset.fromNSec(event_array[0].timestamp);
    }

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
    double start_ts = 0.001;
    unsigned long int frame_id_real = 0;
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

        if (event_array.size() > 0) {
            while (event_low  < event_array.size() - 1 && event_array[event_low].timestamp  < ts_low)  event_low ++;
            while (event_high < event_array.size() - 1 && event_array[event_high].timestamp < ts_high) event_high ++;
        }

        double max_ts_err = std::fabs(cam_tj[cam_tj_id].ts.toSec() - ref_ts);
        for (auto &obj_tj : obj_tjs) {
            if (obj_tj.second.size() == 0) continue;
            double ts_err = std::fabs(ref_ts - obj_tj.second[obj_tj_ids[obj_tj.first]].ts.toSec());
            if (ts_err > max_ts_err) max_ts_err = ts_err;
        }

        double max_p2p_err = 0.0;
        for (auto &obj_tj : obj_tjs) {
            if (obj_tj.second.size() == 0) continue;
            double ts_err = std::fabs(cam_tj[cam_tj_id].ts.toSec() - obj_tj.second[obj_tj_ids[obj_tj.first]].ts.toSec());
            if (ts_err > max_p2p_err) max_p2p_err = ts_err;
        }

        if (max_ts_err > 0.005 || max_p2p_err > 0.001) {
            std::cout << _red("Trajectory timestamp misalignment: ") << max_ts_err << " / " << max_p2p_err << " skipping..." << std::endl;
            frame_id_real ++;
            continue;
        }

        if (frames.size() > 0 && std::fabs(frames.back().get_timestamp() - ref_ts) < 1e-6) {
            std::cout << _red("Duplicate frame encountered at: ") << ref_ts << " sec. skipping..." << std::endl;
            continue;
        }

        frames.emplace_back(cam_tj_id, ref_ts, frame_id_real);
        auto &frame = frames.back();

        if (event_array.size() > 0) {
            frame.add_event_slice_ids(event_low, event_high);
        }

        if (with_images) frame.add_img(images[frame_id_real]);
        std::cout << frame_id_real << ": " << cam_tj[cam_tj_id].ts
                  << " (" << cam_tj_id << "[" << cam_tj[cam_tj_id].occlusion * 100 << "%])";
        for (auto &obj_tj : obj_tjs) {
            if (obj_tj.second.size() == 0) continue;
            std::cout << " " << obj_tj.second[obj_tj_ids[obj_tj.first]].ts << " (" << obj_tj_ids[obj_tj.first]
                      << "[" << obj_tj.second[obj_tj_ids[obj_tj.first]].occlusion * 100 <<  "%])";
            frame.add_object_pos_id(obj_tj.first, obj_tj_ids[obj_tj.first]);
        }
        std::cout << std::endl;

        frame_id_real ++;
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

    if (save_3d) {
        std::string save_dir = Dataset::dataset_folder + '/' + Dataset::camera_name + "/3D_data/";
        Dataset::create_ground_truth_folder(save_dir);
        Backprojector bp(-1, -1, -1);
        bp.save_clouds(save_dir);
    }

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
            std::cout << "\r\tFrame\t" << i + 1 << "\t/\t" << frames.size() << "\t" << std::flush;
        }
    }
    std::cout << std::endl;

    // Create / clear ground truth folder
    Dataset::create_ground_truth_folder();

    // Save ground truth
    std::cout << std::endl << _yellow("Writing depth and mask ground truth") << std::endl;
    std::string meta_fname = Dataset::gt_folder + "/meta.txt";
    std::ofstream meta_file(meta_fname, std::ofstream::out);
    meta_file << "{\n";
    meta_file << Dataset::meta_as_dict() + "\n";
    meta_file << ", 'frames': [\n";
    for (uint64_t i = 0; i < frames.size(); ++i) {
        frames[i].save_gt_images();
        meta_file << frames[i].as_dict() << ",\n\n";

        if (i % 10 == 0) {
            std::cout << "\r\tWritten " << i + 1 << "\t/\t" << frames.size() << "\t" << std::flush;
        }
    }
    meta_file << "]\n";
    std::cout << std::endl;

    std::cout << std::endl << _yellow("Writing full trajectory") << std::endl;
    meta_file << ", 'full_trajectory': [\n";
    for (uint64_t i = 0; i < Dataset::cam_tj.size(); ++i) {
        DatasetFrame frame(i, Dataset::cam_tj[i].ts.toSec(), -1);

        for (auto &obj_tj : Dataset::obj_tjs) {
            if (obj_tj.second.size() == 0) continue;
            frame.add_object_pos_id(obj_tj.first, std::min(i, obj_tj.second.size() - 1));
        }

        meta_file << frame.as_dict() << ",\n\n";

        if (i % 10 == 0) {
            std::cout << "\r\tWritten " << i + 1 << "\t/\t" << Dataset::cam_tj.size() << "\t" << std::flush;
        }
    }
    meta_file << "]\n";
    std::cout << std::endl;

    meta_file << "\n}\n";
    meta_file.close();

    // Save events.txt
    Dataset::write_eventstxt(Dataset::gt_folder + "/events.txt");
    std::cout << _green("Done!") << std::endl;
    return 0;
}
