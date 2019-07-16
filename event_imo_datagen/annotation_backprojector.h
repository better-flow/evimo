#include <dataset.h>
#include <dataset_frame.h>

#ifndef ANNOTATION_BACKPROJECTOR_H
#define ANNOTATION_BACKPROJECTOR_H

class Backprojector {
protected:
    double timestamp;
    double window_size;
    std::vector<DatasetFrame> frames;

public:
    Backprojector(double timestamp, double window_size, double framerate)
        : timestamp(timestamp), window_size(window_size) {

        uint32_t i = 0;
        for (double ts = std::max(0.0, timestamp - window_size / 2.0);
            ts < timestamp + window_size / 2.0; ts += 1.0 / framerate) {
            frames.emplace_back(0, ts, i);
            auto &frame = frames.back();

            frame.add_event_slice_ids(0, 0);
            for (auto &obj_tj : Dataset::obj_tjs) {
                frame.add_object_pos_id(obj_tj.first, 0);
            }

            i += 1;
        }
    }

    void visualize_parallel() {
        for (auto &frame : this->frames) {
            frame.show();
        }

        DatasetFrame::visualization_spin();
    }
};

#endif // ANNOTATION_BACKPROJECTOR_H
