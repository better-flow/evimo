#include <common.h>
#include <event.h>
#include <datastructures.h>


#ifndef FILTERS_H
#define FILTERS_H



class EventSlice1D {
protected:
    uint64_t window_size;
    uint64_t timestamp;
    std::list<Event> events;

public:
    EventSlice1D(uint64_t window_size) 
        : window_size(window_size), timestamp(0) {}

    void push_back(Event &e) {
        this->timestamp = e.timestamp;
        this->events.push_back(e);
        this->remove_outside_window();
    }

    void align_to_ts(uint64_t ts) {
        this->timestamp = ts;
        this->remove_outside_window();
    }

    size_t size() {return this->events.size(); }

protected:
    void remove_outside_window() {
        if (this->timestamp <= window_size) return;
        auto cutoff_ts = this->timestamp - window_size;

        while (this->events.size() > 0 && this->events.front().timestamp < cutoff_ts) {
            this->events.pop_front();
        }
    }
};



class EventSlice2D {
protected:
    std::vector<std::vector<EventSlice1D>> grid;
    uint32_t res_x, res_y;
    uint64_t timestamp;

public:
    EventSlice2D(uint32_t res_x, uint32_t res_y, uint64_t window_size)
        : res_x(res_x), res_y(res_y), timestamp(0) {
        this->grid.resize(this->res_y);
        for (auto &e : this->grid) {
            e.resize(this->res_y, EventSlice1D(window_size));
        }
    }

    virtual void push_back(Event &e) {
        this->timestamp = e.timestamp;
        this->grid[e.get_y()][e.get_x()].push_back(e);
    }

    virtual void remove_outside_window() {
        for (size_t i = 0; i < this->res_y; ++i) {
            for (size_t j = 0; j < this->res_x; ++j) {
                this->grid[i][j].align_to_ts(this->timestamp);
            }
        }
    }

    operator cv::Mat() {
        cv::Mat ret = cv::Mat::zeros(this->res_y, this->res_x, CV_8UC1);
        for (size_t i = 0; i < this->res_y; ++i) {
            for (size_t j = 0; j < this->res_x; ++j) {
                ret.at<uchar>(i, j) = std::min((size_t)255, this->grid[i][j].size());
            }
        }
        return ret;
    }
};


#endif // FILTERS_H
