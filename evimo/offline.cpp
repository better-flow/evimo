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
    float plot_t_offset;
    std::shared_ptr<Dataset> dataset;

    TjPlot plotter; // plot trajectories

public:
    FrameSequenceVisualizer(std::vector<DatasetFrame> &frames, std::shared_ptr<Dataset> &dataset)
        : frame_id(0), dataset(dataset), plotter("Trajectories", 2000, 400) {
        std::cout << "Frame Sequence Visuzlizer...\n";
        this->frames = &frames;
        //this->frame_id = this->frames->size() / 2;

        this->plot_t_offset = this->dataset->cam_tj[0].get_ts_sec();
        this->plotter.add_trajectory_plot(this->dataset->cam_tj, plot_t_offset);
        for (auto &tj : this->dataset->obj_tjs)
            this->plotter.add_trajectory_plot(tj.second, plot_t_offset);

        this->spin();
    }

    void set_slider(int id) {
        this->frame_id = id < this->frames->size() - 1 ? id : this->frames->size() - 1;
        this->frame_id = this->frame_id < 0 ? 0 : this->frame_id;
        cv::setTrackbarPos("frame", "Frames", this->frame_id);
        auto &f = this->frames->at(this->frame_id);
        f.dataset_handle->modified = true;
    }

    void spin() {
        cv::namedWindow("Frames", cv::WINDOW_NORMAL);
        cv::createTrackbar("frame", "Frames", &frame_id, frames->size() - 1, on_trackbar, this);
        auto &f = this->frames->at(this->frame_id);
        cv::createTrackbar("t_pos", "Frames", &f.dataset_handle->pose_to_event_to_slider,
                           Dataset::MAXVAL, Dataset::on_trackbar, f.dataset_handle.get());

        f.dataset_handle->modified = true;
        //f.dataset_handle->init_GUI(); // calibration control
        const uint8_t nmodes = 4;
        uint8_t vis_mode = 0;
        bool nodist = true;

        bool enable_3D = false;
        std::shared_ptr<Backprojector> bp;

        int code = 0; // Key code
        while (code != 27) {
            code = cv::waitKey(1);
            if (bp) bp->maybeViewerSpinOnce();

            auto &f = this->frames->at(this->frame_id);
            f.dataset_handle->handle_keys(code, vis_mode, nmodes);

            if (code == 39) { // '''
                this->set_slider(this->frame_id + 1);
            }

            if (code == 59) { // ';'
                this->set_slider(this->frame_id - 1);
            }

            if (code == 99) { // 'c'
                f.dataset_handle->modified = true;
                enable_3D = !enable_3D;
            }

            if (!f.dataset_handle->modified) continue;
            f.dataset_handle->modified = false;

            f.generate(nodist);

            cv::Mat img;
            switch (vis_mode) {
                default:
                case 0: img = f.get_visualization_mask(true, nodist); break;
                case 1: img = f.get_visualization_mask(false, nodist); break;
                case 2: img = f.get_visualization_depth(true, nodist); break;
                case 3: img = f.get_visualization_event_projection(true, nodist); break;
            }

            // rescale
            float scale = 1280.0 / float(img.cols);
            if (scale < 1.1) cv::resize(img, img, cv::Size(), scale, scale);
            cv::imshow("Frames", img);

            //this->plotter.add_vertical(f.get_timestamp() - plot_t_offset);
            //this->plotter.show();

            if (enable_3D && !bp) {
                bp = std::make_shared<Backprojector>(f.dataset_handle, f.get_timestamp(), 0.4, 200);
                bp->initViewer();
            }

            if (!enable_3D) {
                bp = nullptr;
            }

            if (bp) bp->generate();
        }

        cv::destroyAllWindows();
    }

    static void on_trackbar(int v, void *object) {
        FrameSequenceVisualizer *instance = (FrameSequenceVisualizer*)object;
        auto &f = instance->frames->at(instance->frame_id);
        f.dataset_handle->modified = true;
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

    float max_event_gap = 0.01; // in seconds
    if (!nh.getParam("max_event_gap", max_event_gap)) max_event_gap = 0.01;

    // -- parse the dataset folder
    std::string bag_name = boost::filesystem::path(dataset_folder).stem().string();
    if (bag_name == ".") {
        bag_name = boost::filesystem::path(dataset_folder).parent_path().stem().string();
    }

    auto bag_name_path = boost::filesystem::path(dataset_folder);
    bag_name_path /= (bag_name + ".bag");
    bag_name = bag_name_path.string();

    // Camera name
    std::string camera_name = "";
    if (!nh.getParam("camera_name", camera_name)) camera_name = "main_camera";

    // Read dataset configuration files
    auto dataset = std::make_shared<Dataset>();
    if (!dataset->init(nh, dataset_folder, camera_name))
        return -1;

    // Load 3D models (legacy, for evimo1)
    std::string path_to_self = ros::package::getPath("evimo");
    if (!no_background) {
        dataset->background = std::make_shared<StaticObject>(path_to_self + "/objects/room");
        dataset->background->transform(dataset->bg_E);
    }

    // Extract topics from bag
    if (!dataset->read_bag_file(bag_name, start_time_offset, sequence_duration, with_images, ignore_tj)) {
        return 0;
    }
    with_images = (dataset->images.size() > 0);

    // Align the timestamps
    double start_ts = 0.0;
    if (dataset->cam_tj.size() > 0) start_ts = dataset->cam_tj[0].ts.toSec();
    if (dataset->event_array.size() > 0) start_ts = std::max(double(dataset->event_array.front().timestamp) * 1e-9, start_ts);
    for (auto &obj_tj : dataset->obj_tjs)
        start_ts = std::max(start_ts, obj_tj.second[0].ts.toSec());
    start_ts += 1e-3; // ensure the first pose is surrounded by events
    std::cout << "Actual start timestamp: " << start_ts << "\n";

    unsigned long int frame_id_real = 0;
    double dt = 1.0 / FPS;
    long int cam_tj_id = 0;
    std::map<int, long int> obj_tj_ids;
    std::vector<DatasetFrame> frames;
    uint64_t event_low = 0, event_high = 0;
    while (true) {
        if (with_images) {
            if (frame_id_real >= dataset->image_ts.size()) break;
            start_ts = dataset->image_ts[frame_id_real].toSec();
        }

        while (cam_tj_id < dataset->cam_tj.size() && dataset->cam_tj[cam_tj_id].ts.toSec() < start_ts) cam_tj_id ++;
        for (auto &obj_tj : dataset->obj_tjs)
            while (obj_tj_ids[obj_tj.first] < obj_tj.second.size()
                   && obj_tj.second[obj_tj_ids[obj_tj.first]].ts.toSec() < start_ts) obj_tj_ids[obj_tj.first] ++;

        start_ts += dt;

        bool done = false;
        if (cam_tj_id >= dataset->cam_tj.size()) done = true;
        for (auto &obj_tj : dataset->obj_tjs)
            if (obj_tj.second.size() > 0 && obj_tj_ids[obj_tj.first] >= obj_tj.second.size()) done = true;
        if (done) break;

        auto ref_ts = (with_images ? dataset->image_ts[frame_id_real].toSec() : dataset->cam_tj[cam_tj_id].ts.toSec());
        uint64_t ts_low  = (ref_ts < dataset->slice_width) ? 0 : (ref_ts - dataset->slice_width / 2.0) * 1000000000;
        uint64_t ts_high = (ref_ts + dataset->slice_width / 2.0) * 1000000000;

        if (dataset->event_array.size() > 0) {
            while (event_low  < dataset->event_array.size() - 1 && dataset->event_array[event_low].timestamp  < ts_low)  event_low ++;
            while (event_high < dataset->event_array.size() - 1 && dataset->event_array[event_high].timestamp < ts_high) event_high ++;
        }

        double max_ts_err = std::fabs(dataset->cam_tj[cam_tj_id].ts.toSec() - ref_ts);
        for (auto &obj_tj : dataset->obj_tjs) {
            if (obj_tj.second.size() == 0) continue;
            double ts_err = std::fabs(ref_ts - obj_tj.second[obj_tj_ids[obj_tj.first]].ts.toSec());
            if (ts_err > max_ts_err) max_ts_err = ts_err;
        }

        double max_p2p_err = 0.0;
        for (auto &obj_tj : dataset->obj_tjs) {
            if (obj_tj.second.size() == 0) continue;
            double ts_err = std::fabs(dataset->cam_tj[cam_tj_id].ts.toSec() - obj_tj.second[obj_tj_ids[obj_tj.first]].ts.toSec());
            if (ts_err > max_p2p_err) max_p2p_err = ts_err;
        }

        if (max_ts_err > 0.005 || max_p2p_err > 0.001) {
            std::cout << _red("Trajectory timestamp misalignment: ") << max_ts_err << " / " << max_p2p_err << " skipping..." << std::endl;
            frame_id_real ++;
            continue;
        }

        if (dataset->event_array.size() > 0) {
            bool to_skip = false;
            auto de_low  = ts_low  > dataset->event_array[event_low ].timestamp ? ts_low  - dataset->event_array[event_low].timestamp :
                                                                      dataset->event_array[event_low].timestamp - ts_low;
            auto de_high = ts_high > dataset->event_array[event_high].timestamp ? ts_high - dataset->event_array[event_high].timestamp :
                                                                      dataset->event_array[event_high].timestamp - ts_high;
            if (de_low > max_event_gap * 1e9 || de_high > max_event_gap * 1e9) {
                std::cout << _red("Gap in event window boundaries: ") << double(de_low) * 1e-9 << " / "
                          << double(de_high) * 1e-9 << " sec. skipping..." << std::endl;
                frame_id_real ++;
                continue;
            }

            for (size_t event_id = event_low + 1; event_id <= event_high; event_id++) {
                auto event_gap = dataset->event_array[event_id].timestamp - dataset->event_array[event_id - 1].timestamp;
                if (event_gap > max_event_gap * 1e9) {
                    std::cout << _red("Gap in events ") << event_id - 1 << " -> " << event_id << _red(" detected: ") 
                              << double(event_gap) * 1e-9 << " sec. skipping..." << std::endl;
                    to_skip = true;
                    break;
                }
            }
             
            if (to_skip) {
                frame_id_real ++;
                continue;
            }
        }

        if (dataset->event_array.size() > 0 && event_high - event_low < 10) {
            std::cout << _red("No events at: ") << ref_ts << " sec. skipping..." << std::endl;
            frame_id_real ++;
            continue;
        }

        if (frames.size() > 0 && std::fabs(frames.back().get_timestamp() - ref_ts) < 1e-4) {
            std::cout << _red("Duplicate frame encountered at: ") << ref_ts << " sec. skipping..." << std::endl;
            continue;
        }

        frames.emplace_back(dataset, cam_tj_id, ref_ts, frame_id_real);
        auto &frame = frames.back();

        if (dataset->event_array.size() > 0) {
            frame.add_event_slice_ids(event_low, event_high);
        }

        if (with_images) frame.add_img(dataset->images[frame_id_real]);
        std::cout << frame_id_real << ": " << dataset->cam_tj[cam_tj_id].ts
                  << " (" << cam_tj_id << "[" << dataset->cam_tj[cam_tj_id].occlusion * 100 << "%])";
        for (auto &obj_tj : dataset->obj_tjs) {
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
    if (frames.size() == 0) {
        return 0;
    }

    // Visualization
    int step = std::max(int(frames.size()) / show, 1);
    for (int i = 0; i < frames.size() && show > 0; i += step) {
        frames[i].show();
    }
    if (show > 0) {
        DatasetFrame::visualization_spin();
    }

    if (show == -2)
        FrameSequenceVisualizer fsv(frames, dataset);

    if (save_3d) {
        std::string save_dir = dataset->dataset_folder + '/' + dataset->camera_name + "/3D_data/";
        dataset->create_ground_truth_folder(save_dir);
        Backprojector bp(dataset, -1, -1, -1);
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
    dataset->create_ground_truth_folder();

    // Save ground truth
    std::cout << std::endl << _yellow("Writing depth and mask ground truth") << std::endl;
    std::string meta_fname = dataset->gt_folder + "/meta.txt";
    std::ofstream meta_file(meta_fname, std::ofstream::out);
    meta_file << "{\n";
    meta_file << dataset->meta_as_dict() + "\n";
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
    for (uint64_t i = 0; i < dataset->cam_tj.size(); ++i) {
        DatasetFrame frame(dataset, i, dataset->cam_tj[i].ts.toSec(), -1);

        for (auto &obj_tj : dataset->obj_tjs) {
            if (obj_tj.second.size() == 0) continue;
            frame.add_object_pos_id(obj_tj.first, std::min(i, obj_tj.second.size() - 1));
        }

        meta_file << frame.as_dict() << ",\n\n";

        if (i % 10 == 0) {
            std::cout << "\r\tWritten " << i + 1 << "\t/\t" << dataset->cam_tj.size() << "\t" << std::flush;
        }
    }
    meta_file << "]\n";
    std::cout << std::endl;

    std::cout << std::endl << _yellow("Writing imu") << std::endl;
    meta_file << ", 'imu': {";
    for (auto &imu : dataset->imu_info) {
        meta_file << "'" + imu.first + "': [\n";
        for (uint64_t i = 0; i < imu.second.size(); ++i) {
            meta_file << imu.second[i].as_dict() << ",\n";
        }
        meta_file << "],\n";
    }
    meta_file << "\n}";

    meta_file << "\n}\n";
    meta_file.close();

    // Save events.txt
    dataset->write_eventstxt(dataset->gt_folder + "/events.txt");
    std::cout << _green("Done!") << std::endl;
    return 0;
}
