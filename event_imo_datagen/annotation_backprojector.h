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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr event_pc;

public:
    Backprojector(double timestamp, double window_size, double framerate)
        : timestamp(timestamp), window_size(window_size)
        , event_pc(new pcl::PointCloud<pcl::PointXYZRGB>) {

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
        auto e_slice = TimeSlice(Dataset::event_array,
          std::make_pair(std::max(0.0, this->timestamp - this->window_size / 2.0), this->timestamp + this->window_size / 2.0),
          std::make_pair(this->frames.front().event_slice_ids.first, this->frames.back().event_slice_ids.second));
        for (auto &e : e_slice) {
            pcl::PointXYZRGB p;
            p.x = float(e.get_x()) / 200.0; p.y = float(e.get_y()) / 200.0f; p.z = e.get_ts_sec() - (this->timestamp - this->window_size / 2.0);
            p.z *= 3.0;
            uint32_t rgb = (static_cast<uint32_t>(0) << 16 |
                            static_cast<uint32_t>(20) << 8  | 
                            static_cast<uint32_t>(255));
            p.rgb = *reinterpret_cast<float*>(&rgb);
            this->event_pc->push_back(p);
        }

        if (this->viewer) {
            viewer->removeAllPointClouds();
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(this->event_pc);
            viewer->addPointCloud<pcl::PointXYZRGB>(this->event_pc, rgb, "event cloud");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "event cloud");
        }
    }

    void visualize_parallel() {
        for (auto &frame : this->frames) {
            frame.show();
        }

        DatasetFrame::visualization_spin();
    }

    void visualize_3D() {
        this->viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
        this->viewer->setBackgroundColor (0.9, 0.9, 0.9);
        this->viewer->addCoordinateSystem (1.0);
        this->viewer->initCameraParameters();
        this->viewer->registerKeyboardCallback(&Backprojector::keyboard_handler, *this);

        this->generate();

        using namespace std::chrono_literals;
        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(100ms);
        }

        this->viewer = pcl::visualization::PCLVisualizer::Ptr();
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
