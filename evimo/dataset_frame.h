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
#include <plot.h>

class DatasetFrame {
protected:
    static std::list<DatasetFrame*> visualization_list;

    // Baseline timestamp
    double timestamp;

    // Thread handle
    std::thread thread_handle;
    bool async_gen_in_progress;
    bool no_gt_depth_mask;
public:
    std::shared_ptr<Dataset> dataset_handle;

    uint64_t cam_pose_id;
    std::map<int, uint64_t> obj_pose_ids;
    unsigned long int frame_id;
    std::pair<uint64_t, uint64_t> event_slice_ids;

    cv::Mat img;
    cv::Mat depth;
    cv::Mat mask;

    std::string gt_img_name;
    std::string rgb_img_name;

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

        if (window_names.size() == 0) return;

        TjPlot plotter("Trajectories", 1000, 100);

        std::shared_ptr<Dataset> local_dataset;
        for (auto &window : window_names) {
            local_dataset = window.first->dataset_handle;
            break;
        }

        float plot_t_offset = local_dataset->cam_tj[0].get_ts_sec();
        plotter.add_trajectory_plot(local_dataset->cam_tj, plot_t_offset);

        for (auto &tj : local_dataset->obj_tjs)
            plotter.add_trajectory_plot(tj.second, plot_t_offset);

        for (auto &window : window_names) {
            window.first->dataset_handle->modified = true;
            window.first->dataset_handle->init_GUI();
        }

        const uint8_t nmodes = 3;
        uint8_t vis_mode = 0;

        bool nodist = true;
        int code = 0; // Key code
        while (code != 27) {
            code = cv::waitKey(1);

            size_t threads_running = 0;
            for (auto &window : window_names) {
                window.first->dataset_handle->handle_keys(code, vis_mode, nmodes);
                if (!window.first->dataset_handle->modified) continue;

                threads_running ++;
                window.first->generate_async(nodist);
            }

            if (threads_running == 0) continue;

            for (auto &window : window_names) {
                window.first->join();
                window.first->dataset_handle->modified = false;

                cv::Mat img;
                switch (vis_mode) {
                    default:
                    case 0: img = window.first->get_visualization_mask(true, nodist); break;
                    case 1: img = window.first->get_visualization_mask(false, nodist); break;
                    case 2: img = window.first->get_visualization_depth(true, nodist); break;
                    case 3: img = window.first->get_visualization_event_projection(true, nodist); break;
                }

                plotter.add_vertical(window.first->get_timestamp() - plot_t_offset);

                // rescale
                float scale = 1280.0 / float(img.cols);
                if (scale < 1.1) cv::resize(img, img, cv::Size(), scale, scale);
                cv::imshow(window.second, img);
            }

            plotter.show();
        }

        for (auto &frame_ptr : DatasetFrame::visualization_list) {
            cv::destroyWindow(window_names[frame_ptr]);
        }

        DatasetFrame::visualization_list.clear();
    }

    // ---------
    DatasetFrame(std::shared_ptr<Dataset> &dataset_handle, uint64_t cam_p_id, double ref_ts, unsigned long int fid, bool no_gt_depth_mask=false)
        : dataset_handle(dataset_handle), async_gen_in_progress(false)
        , cam_pose_id(cam_p_id), timestamp(ref_ts), frame_id(fid), event_slice_ids(0, 0)
        , depth(dataset_handle->res_x, dataset_handle->res_y, CV_32F, cv::Scalar(0))
        , mask(dataset_handle->res_x, dataset_handle->res_y, CV_8U, cv::Scalar(0))
        , no_gt_depth_mask(no_gt_depth_mask) {
        
        // if (this->dataset_handle->cam_tj.size() > 0)
        //     this->cam_pose_id = TimeSlice(this->dataset_handle->cam_tj).find_nearest(this->get_timestamp(), this->cam_pose_id);
        
        this->gt_img_name  = "depth_mask_" + std::to_string(this->frame_id) + ".png";
        this->rgb_img_name = "img_" + std::to_string(this->frame_id) + ".png";
    }

    void add_object_pos_id(int id, uint64_t obj_p_id) {
        this->obj_pose_ids.insert(std::make_pair(id, obj_p_id));
        //this->obj_pose_ids[id] = TimeSlice(this->dataset_handle->obj_tjs.at(id)).find_nearest(this->get_timestamp(), this->obj_pose_ids[id]);
    }

    void add_event_slice_ids(uint64_t event_low, uint64_t event_high) {
        this->event_slice_ids = std::make_pair(event_low, event_high);
        this->event_slice_ids = TimeSlice(this->dataset_handle->event_array,
            std::make_pair(this->timestamp - this->dataset_handle->get_time_offset_event_to_host_correction() - this->dataset_handle->slice_width / 2.0,
                           this->timestamp - this->dataset_handle->get_time_offset_event_to_host_correction() + this->dataset_handle->slice_width / 2.0),
            this->event_slice_ids).get_indices();
    }

    void add_img(cv::Mat &img_) {
        this->img = img_;
        //this->img = this->dataset_handle->undistort(img_);
        this->depth = cv::Mat(this->img.rows, this->img.cols, CV_32F, cv::Scalar(0));
        this->mask  = cv::Mat(this->img.rows, this->img.cols, CV_8U, cv::Scalar(0));
    }

    void show() {
        DatasetFrame::visualization_list.push_back(this);
    }

    Pose get_true_camera_pose() {
        auto cam_pose = this->_get_raw_camera_pose();
        auto cam_tf = cam_pose.pq * this->dataset_handle->cam_E;
        return Pose(cam_pose.ts, cam_tf);
    }

    Pose get_camera_velocity() {
        auto vel = this->dataset_handle->cam_tj.get_velocity(this->cam_pose_id);
        vel.pq = this->dataset_handle->cam_E.inverse() * vel.pq * this->dataset_handle->cam_E;
        return vel;
    }

    Pose get_object_pose_cam_frame(int id) {
        if (this->obj_pose_ids.find(id) == this->obj_pose_ids.end()) {
            std::cout << _yellow("Warning! ") << "No pose for object "
                      << id << ", frame id = " << this->frame_id << std::endl;
            std::terminate();
        }

        auto obj_pose = this->_get_raw_object_pose(id);
        auto cam_tf   = this->get_true_camera_pose();
        auto obj_tf   = this->dataset_handle->clouds.at(id)->get_tf_in_camera_frame(
                                                      cam_tf, obj_pose.pq);
        return Pose(cam_tf.ts, obj_tf);
    }

    float get_timestamp() {
        return this->timestamp - this->dataset_handle->get_time_offset_pose_to_host_correction();
    }

    // Generate frame (eiter with distortion or without)
    void generate(bool nodist=false) {
        this->depth = cv::Mat(this->depth.rows, this->depth.cols, CV_32F, cv::Scalar(0));
        this->mask  = cv::Mat(this->mask.rows, this->mask.cols, CV_8U, cv::Scalar(0));

        this->dataset_handle->update_cam_calib();
        this->dataset_handle->cam_tj.set_filtering_window_size(this->dataset_handle->pose_filtering_window);
        //this->cam_pose_id = TimeSlice(this->dataset_handle->cam_tj).find_nearest(this->get_timestamp(), this->cam_pose_id);
        auto cam_tf = this->get_true_camera_pose();
        // FIXME
        auto corrected_ts = cam_tf.ts.toSec() - this->get_timestamp();

        if (this->dataset_handle->event_array.size() > 0)
            this->event_slice_ids = TimeSlice(this->dataset_handle->event_array,
                std::make_pair(this->timestamp + corrected_ts - this->dataset_handle->get_time_offset_event_to_host_correction() - this->dataset_handle->slice_width / 2.0,
                               this->timestamp + corrected_ts - this->dataset_handle->get_time_offset_event_to_host_correction() + this->dataset_handle->slice_width / 2.0),
                this->event_slice_ids).get_indices();

        if (this->dataset_handle->background != nullptr) {
            auto cl = this->dataset_handle->background->transform_to_camframe(cam_tf);
            this->project_cloud(cl, pcl::PolygonMesh::Ptr(nullptr), 0, nodist);
        }

        if (!this->no_gt_depth_mask) {
            // Generate mask and depth
            for (auto &obj : this->dataset_handle->clouds) {
                //if (obj.second->has_no_mesh()) continue;
                auto id = obj.first;
                this->dataset_handle->obj_tjs.at(id).set_filtering_window_size(this->dataset_handle->pose_filtering_window);
                //this->obj_pose_ids[id] = TimeSlice(this->dataset_handle->obj_tjs.at(id)).find_nearest(this->get_timestamp(), this->obj_pose_ids[id]);

                if (this->obj_pose_ids.find(id) == this->obj_pose_ids.end()) {
                    std::cout << _yellow("Warning! ") << "No pose for object "
                              << id << ", frame id = " << this->frame_id << std::endl;
                    continue;
                }

                auto obj_pose = this->_get_raw_object_pose(id);
                auto cl = obj.second->transform_to_camframe(cam_tf, obj_pose.pq);
                this->project_cloud(cl, obj.second->get_mesh(), id, nodist);
            }

            // Add marker labels
            for (auto &obj : this->dataset_handle->clouds) {
                auto markerpos = obj.second->marker_cl_in_camframe(cam_tf);
                this->add_marker_labels(markerpos, nodist);
            }
        }
    }

    void generate_async(bool nodist=false) {
        this->thread_handle = std::thread(&DatasetFrame::generate, this, nodist);
        this->async_gen_in_progress = true;
    }

    void join() {
        if (!this->async_gen_in_progress) return;
        this->thread_handle.join();
        this->async_gen_in_progress = false;
    }

    // This cannot be called before generate_async is complete
    void save_gt_images_async(void) {
        this->thread_handle = std::thread(&DatasetFrame::save_gt_images, this);
        this->async_gen_in_progress = true;
    }

    // Visualization helpers
    cv::Mat get_visualization_event_projection(bool timg = false, bool nodist = false);
    cv::Mat get_visualization_depth(bool overlay_events = true, bool nodist = false);
    cv::Mat get_visualization_mask(bool overlay_events = true, bool nodist = false);

    // Writeout functions
    std::string as_dict() {
        std::string ret = "{\n";
        ret += "'id': " + std::to_string(this->frame_id) + ",\t\t";
        ret += "'ts': " + std::to_string(this->get_timestamp());

        if (!this->no_gt_depth_mask) {
            ret += ",\n'cam': {\n";
            //ret += "\t'vel': " + this->get_camera_velocity().as_dict() + ",\n";

            // Represent camera poses in the frame of the initial camera pose (p0)
            auto p0 = this->dataset_handle->cam_tj[0];
            auto cam_pose = this->_get_raw_camera_pose();
            auto cam_tf = this->dataset_handle->cam_E.inverse() * (cam_pose - p0).pq * this->dataset_handle->cam_E;
            ret += "\t'pos': " + Pose(cam_pose.ts, cam_tf).as_dict() + ",\n";
            ret += "\t'ts': " + std::to_string(this->get_true_camera_pose().ts.toSec()) + "},\n";

            // object poses
            for (auto &pair : this->obj_pose_ids) {
                ret += "'" + std::to_string(pair.first) + "': {\n";
                //ret += "\t'vel': " + this->get_object_velocity(pair.first).as_dict() + ",\n";
                ret += "\t'pos': " + this->get_object_pose_cam_frame(pair.first).as_dict() + ",\n";
                ret += "\t'ts': " + std::to_string(this->get_object_pose_cam_frame(pair.first).ts.toSec()) + "},\n";
            }

            ret += "'gt_frame': '" + this->gt_img_name + "'";
        }

        // image paths
        if (this->img.rows == this->mask.rows && this->img.cols == this->mask.cols) {
            ret += ",\n'classical_frame': '" + this->rgb_img_name + "'";
        }

        ret += "\n}";
        return ret;
    }

    void save_gt_images() {
        if (!this->no_gt_depth_mask) {
            cv::Mat _depth, _mask;
            this->depth.convertTo(_depth, CV_16UC1, 1000);
            this->mask.convertTo(_mask, CV_16UC1, 1000);
            std::vector<cv::Mat> ch = {_depth, _depth, _mask};
            cv::Mat gt_frame_i16(this->mask.rows, this->mask.cols, CV_16UC3, cv::Scalar(0, 0, 0));
            cv::merge(ch, gt_frame_i16);

            gt_frame_i16.convertTo(gt_frame_i16, CV_16UC3);
            cv::imwrite(this->dataset_handle->gt_folder + "/" + this->gt_img_name, gt_frame_i16);
        }

        if (this->img.rows == this->mask.rows && this->img.cols == this->mask.cols) {
            cv::imwrite(this->dataset_handle->gt_folder + "/" + this->rgb_img_name, this->img);
        }
    }


protected:
    template<class T> void add_marker_labels(T cl, bool nodist=false) {
        if (cl->size() == 0)
            return;

        for (auto &p: *cl) {
            float rng = p.z;
            if (rng < 0.001) {
                continue;
            }

            auto cols = this->depth.cols;
            auto rows = this->depth.rows;

            int u = -1, v = -1;
            if (nodist) {
                this->dataset_handle->project_point_nodist(p, u, v);
            } else {
                this->dataset_handle->project_point(p, u, v);
            }

            if (u < 0 || v < 0 || v >= cols || u >= rows) {
                continue;
            }

            this->mask.at<uint8_t>(u, v) = 255;
        }
    }


    template<class T> void project_cloud(T cl, pcl::PolygonMesh::Ptr mesh, int oid, bool nodist=false) {
        if (cl->size() == 0)
            return;

        std::vector<int32_t> pt_u(cl->size());
        std::vector<int32_t> pt_v(cl->size());
        std::vector<float> rng(cl->size());

        auto cols = this->depth.cols;
        auto rows = this->depth.rows;

        // project points to 2d
        for (uint64_t i = 0; i < cl->size(); ++i) {
            auto &p = (*cl)[i];
            int32_t u = -1, v = -1;
            if (nodist) {
                this->dataset_handle->project_point_nodist(p, u, v);
            } else {
                this->dataset_handle->project_point(p, u, v);
            }
            pt_u.at(i) = (int32_t)u;
            pt_v.at(i) = (int32_t)v;
            rng.at(i) = (float)p.z;
        }

        const bool only_points = false;

        // project triangles
        if (mesh && mesh->polygons.size() > 0 && !only_points) {
            for (int i = 0; i < mesh->polygons.size(); ++i) {
                auto u0 = pt_u[mesh->polygons[i].vertices[0]];
                auto v0 = pt_v[mesh->polygons[i].vertices[0]];
                auto z0 = rng[mesh->polygons[i].vertices[0]];
                auto u1 = pt_u[mesh->polygons[i].vertices[1]];
                auto v1 = pt_v[mesh->polygons[i].vertices[1]];
                auto z1 = rng[mesh->polygons[i].vertices[1]];
                auto u2 = pt_u[mesh->polygons[i].vertices[2]];
                auto v2 = pt_v[mesh->polygons[i].vertices[2]];
                auto z2 = rng[mesh->polygons[i].vertices[2]];

                auto u_min = std::min(std::min(u0, u1), u2);
                auto v_min = std::min(std::min(v0, v1), v2);
                auto u_max = std::max(std::max(u0, u1), u2);
                auto v_max = std::max(std::max(v0, v1), v2);
                auto z_max = std::max(std::max(z0, z1), z2);

                if ((u0 == -1 && v0 == -1) || (u1 == -1 && v1 == -1) || (u2 == -1 && v2 == -1)) continue;
                if (u_max < 0 || v_max < 0 || u_min >= rows || v_min >= cols) continue;
                if (z_max < 0.001) continue;

                float x0 = u1 - u0, x1 = u2 - u0, y0 = v1 - v0, y1 = v2 - v0;
                for (int uu = u_min; uu <= u_max; ++uu) {
                    for (int vv = v_min; vv<= v_max; ++vv) {
                        if (uu < 0 || vv < 0 || vv >= cols || uu >= rows) continue;

                        // is it in triange?
                        auto d1 = (uu - u1) * (v0 - v1) - (vv - v1) * (u0 - u1);
                        auto d2 = (uu - u2) * (v1 - v2) - (vv - v2) * (u1 - u2);
                        auto d3 = (uu - u0) * (v2 - v0) - (vv - v0) * (u2 - u0);
                        if (((d1 < 0) || (d2 < 0) || (d3 < 0)) && ((d1 > 0) || (d2 > 0) || (d3 > 0))) continue;

                        // compute depth
                        float x = uu - u0, y = vv - v0;
                        float denom = x0 * y1 - x1 * y0;
                        if (std::fabs(denom) < 1e-5) continue;

                        float alpha = (x * y1 - x1 * y) / denom;
                        float beta  = (x0 * y - x * y0) / denom;
                        float z = z0 + alpha * (z1 - z0) + beta * (z2 - z0);

                        // update masks / depth
                        float base_rng = this->depth.at<float>(uu, vv);
                        if (base_rng > z || base_rng < 0.001) {
                            this->depth.at<float>(uu, vv) = z;
                            this->mask.at<uint8_t>(uu, vv) = oid;
                        }
                    }
                }
            }
        } else { // only project points
            for (uint64_t i = 0; i < rng.size(); ++i) {
                if (rng[i] < 0.001)
                    continue;

                if (pt_u[i] < 0 || pt_v[i] < 0 || pt_v[i] >= cols || pt_u[i] >= rows)
                    continue;

                float base_rng = this->depth.at<float>(pt_u[i], pt_v[i]);
                if (base_rng > rng[i] || base_rng < 0.001) {
                    this->depth.at<float>(pt_u[i], pt_v[i]) = rng[i];
                    this->mask.at<uint8_t>(pt_u[i], pt_v[i]) = oid;
                }
            }
        }
    }

    Pose _get_raw_camera_pose() {
        if (this->cam_pose_id >= this->dataset_handle->cam_tj.size()) {
            std::cout << _yellow("Warning! ") << "Camera right pose out of bounds for "
                      << " frame id " << this->frame_id << " with "
                      << this->dataset_handle->cam_tj.size() << " trajectory records and "
                      << "trajectory id = " << this->cam_pose_id << std::endl;
            return this->dataset_handle->cam_tj[this->dataset_handle->cam_tj.size() - 1];
        }

        if (this->cam_pose_id == 0) {
            // std::cout << _yellow("Warning! ") << "Camera left pose out of bounds for "
            //           << " frame id " << this->frame_id << " with "
            //           << this->dataset_handle->cam_tj.size() << " trajectory records and "
            //           << "trajectory id = " << this->cam_pose_id << std::endl;
            return this->dataset_handle->cam_tj[0];
        }

        auto left_pose  = this->dataset_handle->cam_tj[this->cam_pose_id-1];
        auto right_pose = this->dataset_handle->cam_tj[this->cam_pose_id];
        auto interp_pose = Pose(left_pose, right_pose, this->timestamp);

        return interp_pose;
    }

    Pose _get_raw_object_pose(int id) {
        if (this->obj_pose_ids.find(id) == this->obj_pose_ids.end()) {
            std::cout << _yellow("Warning! ") << "No pose for object "
                      << id << ", frame id = " << this->frame_id << std::endl;
        }
        auto obj_pose_id = this->obj_pose_ids.at(id);
        auto obj_tj_size = this->dataset_handle->obj_tjs.at(id).size();
        if (obj_pose_id >= obj_tj_size) {
            std::cout << _yellow("Warning! ") << "Object (" << id << ") right pose "
                      << "out of bounds for frame id " << this->frame_id << " with "
                      << obj_tj_size << " trajectory records and "
                      << "trajectory id = " << obj_pose_id << std::endl;
            return this->dataset_handle->obj_tjs.at(id)[obj_tj_size - 1];
        }

        if (obj_pose_id == 0) {
            std::cout << _yellow("Warning! ") << "Object (" << id << ") left pose "
                      << "out of bounds for frame id " << this->frame_id << " with "
                      << obj_tj_size << " trajectory records and "
                      << "trajectory id = " << obj_pose_id << std::endl;
            return this->dataset_handle->obj_tjs.at(id)[0];
        }

        auto left_pose  = this->dataset_handle->obj_tjs.at(id)[obj_pose_id-1];
        auto right_pose = this->dataset_handle->obj_tjs.at(id)[obj_pose_id];
        auto interp_pose = Pose(left_pose, right_pose, this->timestamp);

        return interp_pose;
    }
};

#endif // DATASET_FRAME_H
