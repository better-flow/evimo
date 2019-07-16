#ifndef DATASET_FRAME_H
#define DATASET_FRAME_H

#include <vector>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <dataset.h>
#include <object.h>
#include <trajectory.h>

class DatasetFrame {
protected:
    static std::list<DatasetFrame*> visualization_list;

    // Baseline timestamp
    double timestamp;

    // Thread handle
    std::thread thread_handle;
public:
    uint64_t cam_pose_id;
    std::map<int, uint64_t> obj_pose_ids;
    unsigned long int frame_id;
    std::pair<uint64_t, uint64_t> event_slice_ids;

    cv::Mat img;
    cv::Mat depth;
    cv::Mat mask;

public:
    static void on_trackbar(int, void*) {
        for (auto &frame_ptr : DatasetFrame::visualization_list) {
            std::string window_name = "Frame " + std::to_string(frame_ptr->frame_id);
        }
    }

    static void visualization_spin() {
        std::map<DatasetFrame*, std::string> window_names;
        for (auto &frame_ptr : DatasetFrame::visualization_list) {
            window_names[frame_ptr] = "Frame " + std::to_string(frame_ptr->frame_id);
            cv::namedWindow(window_names[frame_ptr], cv::WINDOW_NORMAL);
        }

        Dataset::modified = true;
        Dataset::init_GUI();
        const uint8_t nmodes = 3;
        uint8_t vis_mode = 0;

        int code = 0; // Key code
        while (code != 27) {
            code = cv::waitKey(1);
            Dataset::handle_keys(code, vis_mode, nmodes);

            if (!Dataset::modified) continue;
            Dataset::modified = false;

            for (auto &window : window_names) {
                window.first->generate_async();
            }

            for (auto &window : window_names) {
                window.first->join();

                cv::Mat img;
                switch (vis_mode) {
                    default:
                    case 0: img = window.first->get_visualization_mask(true); break;
                    case 1: img = window.first->get_visualization_mask(false); break;
                    case 2: img = window.first->get_visualization_depth(true); break;
                    case 3: img = window.first->get_visualization_event_projection(true); break;
                }

                cv::imshow(window.second, img);
            }
        }

        cv::destroyAllWindows();
        DatasetFrame::visualization_list.clear();
    }

    // ---------
    DatasetFrame(uint64_t cam_p_id, double ref_ts, unsigned long int fid)
        : cam_pose_id(cam_p_id), timestamp(ref_ts), frame_id(fid), event_slice_ids(0, 0),
          depth(Dataset::res_x, Dataset::res_y, CV_32F, cv::Scalar(0)),
          mask(Dataset::res_x, Dataset::res_y, CV_8U, cv::Scalar(0)) {
        this->cam_pose_id = TimeSlice(Dataset::cam_tj).find_nearest(this->get_timestamp(), this->cam_pose_id);
    }

    void add_object_pos_id(int id, uint64_t obj_p_id) {
        this->obj_pose_ids.insert(std::make_pair(id, obj_p_id));
        this->obj_pose_ids[id] = TimeSlice(Dataset::obj_tjs.at(id)).find_nearest(this->get_timestamp(), this->obj_pose_ids[id]);
    }

    void add_event_slice_ids(uint64_t event_low, uint64_t event_high) {
        this->event_slice_ids = std::make_pair(event_low, event_high);
        this->event_slice_ids = TimeSlice(Dataset::event_array,
            std::make_pair(this->timestamp - Dataset::get_time_offset_event_to_host_correction() - Dataset::slice_width / 2.0,
                           this->timestamp - Dataset::get_time_offset_event_to_host_correction() + Dataset::slice_width / 2.0),
            this->event_slice_ids).get_indices();
    }

    void add_img(cv::Mat &img_) {
        this->img = img_;
    }

    void show() {
        DatasetFrame::visualization_list.push_back(this);
    }

    Pose get_true_camera_pose() {
        auto cam_pose = this->_get_raw_camera_pose();
        auto cam_tf = cam_pose.pq * Dataset::cam_E;
        return Pose(cam_pose.ts, cam_tf);
    }

    Pose get_true_object_pose(int id) {
        if (this->obj_pose_ids.find(id) == this->obj_pose_ids.end()) {
            std::cout << _yellow("Warning! ") << "No pose for object "
                      << id << ", frame id = " << this->frame_id << std::endl;
        }

        auto obj_pose = this->_get_raw_object_pose(id);
        auto cam_tf   = this->get_true_camera_pose();
        auto obj_tf   = Dataset::clouds.at(id)->get_tf_in_camera_frame(
                                                      cam_tf, obj_pose.pq);
        return Pose(cam_tf.ts, obj_tf);
    }

    float get_timestamp() {
        return this->timestamp - Dataset::get_time_offset_pose_to_host_correction();
    }

    std::string get_info() {
        std::string s;
        s += std::to_string(frame_id) + ": " + std::to_string(get_timestamp()) + "\t";
        s += std::to_string(get_true_camera_pose().ts.toSec()) + "\t";
        for (auto &obj : Dataset::clouds) {
            s += std::to_string(this->_get_raw_object_pose(obj.first).ts.toSec()) + "\t";
        }
        return s;
    }

    // Generate frame
    void generate() {
        this->depth = cv::Scalar(0);
        this->mask  = cv::Scalar(0);

        Dataset::update_cam_calib();
        Dataset::cam_tj.set_filtering_window_size(Dataset::pose_filtering_window);
        this->cam_pose_id = TimeSlice(Dataset::cam_tj).find_nearest(this->get_timestamp(), this->cam_pose_id);
        this->event_slice_ids = TimeSlice(Dataset::event_array,
            std::make_pair(this->timestamp - Dataset::get_time_offset_event_to_host_correction() - Dataset::slice_width / 2.0,
                           this->timestamp - Dataset::get_time_offset_event_to_host_correction() + Dataset::slice_width / 2.0),
            this->event_slice_ids).get_indices();

        auto cam_tf = this->get_true_camera_pose();
        if (Dataset::background != nullptr) {
            auto cl = Dataset::background->transform_to_camframe(cam_tf);
            this->project_cloud(cl, 0);
        }

        for (auto &obj : Dataset::clouds) {
            auto id = obj.first;
            Dataset::obj_tjs.at(id).set_filtering_window_size(Dataset::pose_filtering_window);
            this->obj_pose_ids[id] = TimeSlice(Dataset::obj_tjs.at(id)).find_nearest(this->get_timestamp(), this->obj_pose_ids[id]);

            if (this->obj_pose_ids.find(id) == this->obj_pose_ids.end()) {
                std::cout << _yellow("Warning! ") << "No pose for object "
                          << id << ", frame id = " << this->frame_id << std::endl;
                continue;
            }

            auto obj_pose = this->_get_raw_object_pose(id);
            auto cl = obj.second->transform_to_camframe(cam_tf, obj_pose.pq);
            this->project_cloud(cl, id);
        }
    }

    void generate_async() {
        this->thread_handle = std::thread(&DatasetFrame::generate, this);
    }

    void join() {
        this->thread_handle.join();
    }

    // Visualization helpers
    cv::Mat get_visualization_event_projection(bool timg = false);
    cv::Mat get_visualization_depth(bool overlay_events = true);
    cv::Mat get_visualization_mask(bool overlay_events = true);

protected:
    void project_point(pcl::PointXYZRGB p, int &u, int &v) {
        u = -1; v = -1;
        if (p.z < 0.00001)
            return;

        float x_ = p.x / p.z;
        float y_ = p.y / p.z;

        float r2 = x_ * x_ + y_ * y_;
        float r4 = r2 * r2;
        float r6 = r2 * r2 * r2;
        float dist = (1.0 + Dataset::k1 * r2 + Dataset::k2 * r4 +
                            Dataset::k3 * r6) / (1 + Dataset::k4 * r2);
        float x__ = x_ * dist;
        float y__ = y_ * dist;

        u = Dataset::fx * x__ + Dataset::cx;
        v = Dataset::fy * y__ + Dataset::cy;
    }

    void project_cloud(auto cl, int oid) {
        if (cl->size() == 0)
            return;

        for (auto &p: *cl) {
            p.z = -p.z;

            float rng = p.z;
            if (rng < 0.001)
                continue;

            auto cols = this->depth.cols;
            auto rows = this->depth.rows;

            int u = -1, v = -1;
            this->project_point(p, u, v);

            if (u < 0 || v < 0 || v >= cols || u >= rows)
                continue;

            int patch_size = int(1.0 / rng);

            if (oid == 0)
                patch_size = int(5.0 / rng);

            int u_lo = std::max(u - patch_size / 2, 0);
            int u_hi = std::min(u + patch_size / 2, rows - 1);
            int v_lo = std::max(v - patch_size / 2, 0);
            int v_hi = std::min(v + patch_size / 2, cols - 1);

            for (int ii = u_lo; ii <= u_hi; ++ii) {
                for (int jj = v_lo; jj <= v_hi; ++jj) {
                    float base_rng = this->depth.at<float>(rows - ii - 1, cols - jj - 1);
                    if (base_rng > rng || base_rng < 0.001) {
                        this->depth.at<float>(rows - ii - 1, cols - jj - 1) = rng;
                        this->mask.at<uint8_t>(rows - ii - 1, cols - jj - 1) = oid;
                    }
                }
            }
        }
    }

    Pose _get_raw_camera_pose() {
        if (this->cam_pose_id >= Dataset::cam_tj.size()) {
            std::cout << _yellow("Warning! ") << "Camera pose out of bounds for "
                      << " frame id " << this->frame_id << " with "
                      << Dataset::cam_tj.size() << " trajectory records and "
                      << "trajectory id = " << this->cam_pose_id << std::endl;
            return Dataset::cam_tj[Dataset::cam_tj.size() - 1];
        }
        return Dataset::cam_tj[this->cam_pose_id];
    }

    Pose _get_raw_object_pose(int id) {
        if (this->obj_pose_ids.find(id) == this->obj_pose_ids.end()) {
            std::cout << _yellow("Warning! ") << "No pose for object "
                      << id << ", frame id = " << this->frame_id << std::endl;
        }
        auto obj_pose_id = this->obj_pose_ids.at(id);
        auto obj_tj_size = Dataset::obj_tjs.at(id).size();
        if (obj_pose_id >= obj_tj_size) {
            std::cout << _yellow("Warning! ") << "Object (" << id << ") pose "
                      << "out of bounds for frame id " << this->frame_id << " with "
                      << obj_tj_size << " trajectory records and "
                      << "trajectory id = " << obj_pose_id << std::endl;
            return Dataset::obj_tjs.at(id)[obj_tj_size - 1];
        }
        return Dataset::obj_tjs.at(id)[obj_pose_id];
    }
};

#endif // DATASET_FRAME_H
