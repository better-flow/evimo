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
    with_images = (dataset->images.size() > 0); // We are using a classical camera
    std::cout << "Found " << dataset->images.size() << " camera images\n";

    // Align the timestamps
    double start_ts = 0.0;
    if (dataset->cam_tj.size() > 0) start_ts = dataset->cam_tj[0].ts.toSec();
    if (dataset->event_array.size() > 0) start_ts = std::max(double(dataset->event_array.front().timestamp) * 1e-9, start_ts);
    for (auto &obj_tj : dataset->obj_tjs)
        start_ts = std::max(start_ts, obj_tj.second[0].ts.toSec());
    start_ts += 1e-3; // ensure the first pose is surrounded by events
    std::cout << "Actual start timestamp: " << start_ts << "\n";
    std::cout << "Vicon occlusion info per ground truth frame:" << std::endl;

    // Make a list of GT frames to generate
    std::vector<DatasetFrame> frames;
    {
        unsigned long int frame_id_real = 0;
        double dt = 1.0 / FPS;
        long int cam_tj_id = 0;
        std::map<int, long int> obj_tj_ids;
        uint64_t event_low = 0, event_high = 0;
        start_ts -= dt;
        while (true) {
            // Increment start_ts forward by one dt
            // It has to be done here so that the continue statements
            // below do not stop start_ts from updating
            // it is decremented just before this loop to fix the off by one error
            start_ts += dt;

            // If using a classical camera
            if (with_images) {
                if (frame_id_real >= dataset->image_ts.size()) break; // If classical camera, but no frames, just exit the loop
                start_ts = dataset->image_ts[frame_id_real].toSec(); // Otherwise force the staring timestamp to the first frame

                // If the last generated GT frame is too close to start_ts
                // then there are frames from the conventional camera that are timestamped incorrectly
                if (frames.size() > 0 && std::fabs(frames.back().get_timestamp() - start_ts) < 1e-4) {
                    std::cout << _red("Duplicate frame encountered at: ") << start_ts << " sec. exiting..." << std::endl;
                    return -1;
                }
            }

            // Find the trajectory id's with the closest timestamp greater than or equal to start_ts
            // Find the right id for the camera
            while (cam_tj_id < dataset->cam_tj.size() && dataset->cam_tj[cam_tj_id].ts.toSec() < start_ts) {
                cam_tj_id ++;
            }
            // Find the right id for every object
            for (auto &obj_tj : dataset->obj_tjs) {
                while (obj_tj_ids[obj_tj.first] < obj_tj.second.size()
                       && obj_tj.second[obj_tj_ids[obj_tj.first]].ts.toSec() < start_ts) {
                    obj_tj_ids[obj_tj.first] ++;
                }
            }

            // If we are out of trajectory data for the camera or any of the objects, exit generation
            bool done = false;
            if (cam_tj_id >= dataset->cam_tj.size()) done = true;
            for (auto &obj_tj : dataset->obj_tjs)
                if (obj_tj.second.size() > 0 && obj_tj_ids[obj_tj.first] >= obj_tj.second.size()) done = true;
            if (done) break;

            // Check that the left ids are valid
            bool left_side_undefined = false;
            if (cam_tj_id - 1 < 0) {
                left_side_undefined = true;
            }
            for (auto &obj_tj : dataset->obj_tjs) {
                if (obj_tj_ids[obj_tj.first] - 1 < 0) {
                    left_side_undefined = true;
                }
            }

            // Check that difference between right and left times is not too big
            bool left_right_time_diff_too_big = false;
            double left_right_time_diff_thresh = 0.02; // Vicon runs at 200 Hz so this is reasonable
            if (dataset->cam_tj[cam_tj_id].ts.toSec() - dataset->cam_tj[cam_tj_id-1].ts.toSec() > left_right_time_diff_thresh) {
                left_right_time_diff_too_big = true;
            }
            for (auto &obj_tj : dataset->obj_tjs) {
                if (obj_tj.second[obj_tj_ids[obj_tj.first]].ts.toSec() - obj_tj.second[obj_tj_ids[obj_tj.first]-1].ts.toSec() > left_right_time_diff_thresh) {
                    left_right_time_diff_too_big = true;
                }
            }

            if (left_side_undefined || left_right_time_diff_too_big) {

            }
            else {
                // All checks have passed, schedule creation of a GT frame
                frames.emplace_back(dataset, cam_tj_id, start_ts, frame_id_real);

                // Add the event_slice times to the GT frame
                auto &frame = frames.back();
                if (dataset->event_array.size() > 0) {
                    // Find a slice of events that lie within the interval ts_low to ts_high
                    uint64_t ts_low  = (start_ts < dataset->slice_width) ? 0 : (start_ts - dataset->slice_width / 2.0) * 1000000000;
                    uint64_t ts_high = (start_ts + dataset->slice_width / 2.0) * 1000000000;
                    while (event_low  < dataset->event_array.size() - 1 && dataset->event_array[event_low].timestamp  < ts_low)  event_low ++;
                    while (event_high < dataset->event_array.size() - 1 && dataset->event_array[event_high].timestamp < ts_high) event_high ++;

                    // If ts_high is outside the interval, bump it back by one, this gauruntees the event is within the interval
                    // because the loop above exits when dataset->event_array[event_high].timestamp >= ts_high
                    if (dataset->event_array[event_high].timestamp >= ts_high) {
                        event_high--;
                    }

                    // If the lowest event is outside the interval, then the interval has no events
                    // Make sure the interval is not empty
                    if (event_low <= event_high && dataset->event_array[event_low].timestamp <= ts_high) {
                        frame.add_event_slice_ids(event_low, event_high);
                    } 
                }

                // Add a classical image to the GT frame
                if (with_images) frame.add_img(dataset->images[frame_id_real]);

                // Print information about the frame
                std::cout << frame_id_real << ": cam "
                          << std::setprecision(2) << dataset->cam_tj[cam_tj_id].ts.toSec()
                          << " (" << cam_tj_id << "[" << std::setprecision(0) << dataset->cam_tj[cam_tj_id].occlusion * 100 << "%])";
                for (auto &obj_tj : dataset->obj_tjs) {
                    if (obj_tj.second.size() != 0) {
                        std::cout << " obj_" << obj_tj.first << " "
                                  << std::setprecision(2) << obj_tj.second[obj_tj_ids[obj_tj.first]].ts.toSec()
                                  << " (" << obj_tj_ids[obj_tj.first] << "[" << std::setprecision(0) << obj_tj.second[obj_tj_ids[obj_tj.first]].occlusion * 100 <<  "%])";
                        frame.add_object_pos_id(obj_tj.first, obj_tj_ids[obj_tj.first]);
                    }
                }
                std::cout << std::endl;
            }

            // We have generated (or skipped) one gt frame so increment frame_id_real
            frame_id_real ++;
        }
    }

    std::cout << _blue("\nTimestamp alignment done") << std::endl;
    std::cout << "\tDataset contains " << frames.size() << " ground truth frames" << std::endl;
    if (frames.size() == 0) {
        return 0;
    }

    // Visualize in some way if appropriate flags are set
    // closes program if generate not set
    if (show > 0 || show == -2 || save_3d) {
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
    }

    // Generate the GT frames from the list frames
    // Projecting the clouds and generating masks / depth maps
    {
        auto processor_count = std::thread::hardware_concurrency();
        // If not able to detect hardware concurrency
        if (processor_count == 0) processor_count = 1; 

        for (uint64_t i = 0; i < frames.size(); i+=processor_count) {
            std::cout << "\r\tGenerating Frame \t" << i + 1 << "\t/\t" << frames.size() << "\t" << std::flush;
            // Quick and easy concurrency without spawning hundreds of threads simultaneously (results in big speedup)
            for (uint64_t j = 0; j < processor_count; j++) {
                if (i+j < frames.size()) frames[i+j].generate_async();
            }

            for (uint64_t j = 0; j < processor_count; j++) {
                if (i+j < frames.size()) frames[i+j].join();
            }
        }
        std::cout << std::endl;
    }

    // Save the GT frames to disk
    {
        // Create / clear ground truth folder
        dataset->create_ground_truth_folder();

        auto processor_count = std::thread::hardware_concurrency();
        // If not able to detect hardware concurrency
        if (processor_count == 0) processor_count = 1; 

        std::cout << std::endl << _yellow("Writing depth and mask ground truth") << std::endl;
        for (uint64_t i = 0; i < frames.size(); i+=processor_count) {
            std::cout << "\r\tWriting Frame \t" << i + 1 << "\t/\t" << frames.size() << "\t" << std::flush;
            // Quick and easy concurrency without spawning hundreds of threads simultaneously (results in big speedup)
            for (uint64_t j = 0; j < processor_count; j++) {
                if (i+j < frames.size()) frames[i+j].save_gt_images_async();
            }

            for (uint64_t j = 0; j < processor_count; j++) {
                if (i+j < frames.size()) frames[i+j].join();
            }
        }
    }

    // Save meta.txt
    {
        std::cout << std::endl << _yellow("Writing meta") << std::endl;
        std::string meta_fname = dataset->gt_folder + "/meta.txt";
        std::ofstream meta_file(meta_fname, std::ofstream::out);
        meta_file << "{\n";
        meta_file << dataset->meta_as_dict() + "\n";
        meta_file << ", 'frames': [\n";
        for (uint64_t i = 0; i < frames.size(); ++i) {
            meta_file << frames[i].as_dict() << ",\n\n";

            if (i % 10 == 0) {
                std::cout << "\r\tWritten meta " << i + 1 << "\t/\t" << frames.size() << "\t" << std::flush;
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
    }

    // Save events.txt
    dataset->write_eventstxt(dataset->gt_folder + "/events.txt");
    std::cout << _green("Done!") << std::endl;
    return 0;
}
