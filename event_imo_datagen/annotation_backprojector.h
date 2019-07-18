#include <thread>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <dataset.h>
#include <dataset_frame.h>

#ifndef ANNOTATION_BACKPROJECTOR_H
#define ANNOTATION_BACKPROJECTOR_H

class Backprojector {
protected:
    double timestamp;
    double window_size;
    std::vector<DatasetFrame> frames;

    pcl::visualization::PCLVisualizer::Ptr viewer;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr event_pc, mask_pc;

public:
    Backprojector(double timestamp, double window_size, double framerate)
        : timestamp(timestamp), window_size(window_size)
        , event_pc(new pcl::PointCloud<pcl::PointXYZRGB>)
        , mask_pc(new pcl::PointCloud<pcl::PointXYZRGB>) {

        uint32_t i = 0;
        uint64_t last_cam_pos_id = 0;
        std::pair<uint64_t, uint64_t> last_event_slice_ids(0, 0);
        for (double ts = std::max(0.0, timestamp - window_size / 2.0);
            ts < timestamp + window_size / 2.0; ts += 1.0 / framerate) {
            frames.emplace_back(last_cam_pos_id, ts, i);
            auto &frame = frames.back();
            last_cam_pos_id = frame.cam_pose_id;

            for (auto &obj_tj : Dataset::obj_tjs) {
                frame.add_object_pos_id(obj_tj.first, frame.cam_pose_id);
            }

            frame.add_event_slice_ids(last_event_slice_ids.first, last_event_slice_ids.second);
            last_event_slice_ids = frame.event_slice_ids;

            i += 1;
        }
    }

    void generate() {
        this->event_pc->clear();
        this->mask_pc->clear();

        // Conver timestamp to z coordinate
        auto ts_to_z = [=](double ts) { return (ts - (this->timestamp - this->window_size / 2.0)) * 3.0; };

        // Event cloud
        auto e_slice = TimeSlice(Dataset::event_array,
          std::make_pair(std::max(0.0, this->timestamp - this->window_size / 2.0), this->timestamp + this->window_size / 2.0),
          std::make_pair(this->frames.front().event_slice_ids.first, this->frames.back().event_slice_ids.second));
        for (auto &e : e_slice) {
            pcl::PointXYZRGB p;
            p.x = float(e.get_x()) / 200.0; p.y = float(e.get_y()) / 200.0f; p.z = ts_to_z(e.get_ts_sec());
            uint32_t rgb = (static_cast<uint32_t>(0) << 16 |
                            static_cast<uint32_t>(20) << 8  | 
                            static_cast<uint32_t>(255));
            p.rgb = *reinterpret_cast<float*>(&rgb);
            this->event_pc->push_back(p);
        }

        // Mask trace cloud
        for (auto &f : this->frames) f.generate_async();
        for (auto &f : this->frames) f.join();

        for (auto &f : this->frames) {
            auto cl = this->mask_to_cloud(f.mask, ts_to_z(f.get_timestamp()));
            *this->mask_pc += *cl;
        }

        if (this->viewer) {
            viewer->removeAllPointClouds();

            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> ec_rgb(this->event_pc);
            viewer->addPointCloud<pcl::PointXYZRGB>(this->event_pc, ec_rgb, "event cloud");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "event cloud");

            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> m_rgb(this->mask_pc);
            viewer->addPointCloud<pcl::PointXYZRGB>(this->mask_pc, m_rgb, "mask cloud");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "mask cloud");
        }
    }

    std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> mask_to_cloud(cv::Mat mask, double z) {
        auto cl = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

        cv::Mat boundary;
        cv::morphologyEx(mask, boundary, cv::MORPH_GRADIENT, kernel);

        auto cols = mask.cols;
        auto rows = mask.rows;

        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                if (boundary.at<uint8_t>(i, j) == 0) continue;

                pcl::PointXYZRGB p;
                p.x = float(i) / 200; p.y = float(j) / 200; p.z = z;
                uint32_t rgb = (static_cast<uint32_t>(255) << 16 |
                                static_cast<uint32_t>(0) << 8  | 
                                static_cast<uint32_t>(0));
                p.rgb = *reinterpret_cast<float*>(&rgb);
                cl->push_back(p);
            }
        }

        return cl;
    }

    // Visualization
    void visualize_parallel() {
        for (auto &frame : this->frames) {
            frame.show();
        }

        DatasetFrame::visualization_spin();
    }

    void initViewer() {
        this->viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
        this->viewer->setBackgroundColor (0.9, 0.9, 0.9);
        this->viewer->addCoordinateSystem (1.0);
        this->viewer->initCameraParameters();
        this->viewer->registerKeyboardCallback(&Backprojector::keyboard_handler, *this);

        //this->generate();

        //using namespace std::chrono_literals;
        //while (!viewer->wasStopped()) {
        //this->maybeViewerSpinOnce();
        //    std::this_thread::sleep_for(100ms);
        //}

        //this->viewer = pcl::visualization::PCLVisualizer::Ptr();
    }

    void maybeViewerSpinOnce() {
        if (!this->viewer) return;
        this->viewer->spinOnce(100);
    }

    void keyboard_handler(const pcl::visualization::KeyboardEvent &event,
                          void* viewer_void) {
        if (event.getKeySym() == "r" && event.keyDown()) {
            std::cout << "r was pressed => removing all text" << std::endl;

            this->timestamp += 0.03;
            this->generate();
        }

        if (event.getKeyCode() == 27) {
            this->viewer->close();
            //this->viewer = pcl::visualization::PCLVisualizer::Ptr();
        }
    }
};

#endif // ANNOTATION_BACKPROJECTOR_H
