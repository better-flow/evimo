#include <common.h>
#include <event.h>
#include <datastructures.h>


#ifndef FILTERS_H
#define FILTERS_H



class EventSlice1D {
protected:
    uint64_t window_size;
    uint64_t timestamp;
    int64_t polarity_sum;
    std::list<Event> events;

public:
    EventSlice1D(uint64_t window_size) 
        : window_size(window_size), timestamp(0), polarity_sum(0) {}

    void push_back(Event &e) {
        this->timestamp = e.timestamp;
        this->polarity_sum += (e.polarity > 0) ? 1 : -1;
        this->events.push_back(e);
        this->remove_outside_window();
    }

    void align_to_ts(uint64_t ts) {
        this->timestamp = ts;
        this->remove_outside_window();
    }

    size_t size() {return this->events.size(); }
    int64_t get_polarity_sum() {return this->polarity_sum; }

    inline auto begin() {return this->events.begin(); }
    inline auto end()   {return this->events.end(); }


protected:
    void remove_outside_window() {
        if (this->timestamp <= window_size) return;
        auto cutoff_ts = this->timestamp - window_size;
        while (this->events.size() > 0 && this->events.front().timestamp < cutoff_ts) {
            this->polarity_sum -= (events.front().polarity > 0) ? 1 : -1;
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
            e.resize(this->res_x, EventSlice1D(window_size));
        }
    }

    virtual void push_back(Event &e) {
        this->timestamp = e.timestamp;
        this->grid[e.get_x()][e.get_y()].push_back(e);
    }

    virtual void remove_outside_window() {
        for (size_t i = 0; i < this->res_y; ++i) {
            for (size_t j = 0; j < this->res_x; ++j) {
                this->grid[i][j].align_to_ts(this->timestamp);
            }
        }
    }

    virtual operator cv::Mat() {
        this->remove_outside_window();
        cv::Mat ret = cv::Mat::zeros(this->res_y, this->res_x, CV_8UC1);
        for (size_t i = 0; i < this->res_y; ++i) {
            for (size_t j = 0; j < this->res_x; ++j) {
                ret.at<uchar>(i, j) = std::min((size_t)255, this->grid[i][j].size());
            }
        }
        return ret;
    }
};


class EventSlice2DFrequencyFilter: public EventSlice2D {
protected:
    float frequency_lo, frequency_hi;

public:
    EventSlice2DFrequencyFilter(uint32_t res_x, uint32_t res_y, uint64_t window_size, float frequency_lo, float frequency_hi) 
        : EventSlice2D(res_x, res_y, window_size), frequency_lo(frequency_lo), frequency_hi(frequency_hi) {}

    virtual operator cv::Mat() {
        this->remove_outside_window();
        cv::Mat ret = cv::Mat::zeros(this->res_y, this->res_x, CV_8UC1);
        for (size_t i = 0; i < this->res_y; ++i) {
            for (size_t j = 0; j < this->res_x; ++j) {
                if (this->grid[i][j].size() <= 4) continue;

                int8_t prev_pol = -2;
                std::vector<uint64_t> rising, falling;
                for (auto &e : this->grid[i][j]) {
                    if (prev_pol < -1) prev_pol = e.polarity;
                    if (e.polarity > prev_pol) { // rising edge
                        rising.push_back(e.timestamp);
                    }
                    if (e.polarity < prev_pol) { // falling edge
                        falling.push_back(e.timestamp);
                    }
                    prev_pol = e.polarity;
                }

                uint64_t avg_ts = 0;
                for (size_t k = 1; k < rising.size(); ++k) {
                    avg_ts += rising[k] - rising[k - 1];
                }
                float rising_f = (rising.size() >= 2) ? 1e-9 * double(avg_ts) / double(rising.size() - 1) : -1;

                avg_ts = 0;
                for (size_t k = 1; k < falling.size(); ++k) {
                    avg_ts += falling[k] - falling[k - 1];
                }
                float falling_f = (falling.size() >= 2) ? 1e-9 * double(avg_ts) / double(falling.size() - 1) : -1;

                float period = rising_f > 0 ? rising_f : 0;
                period += falling_f > 0 ? falling_f : 0;
                if (rising_f > 0 && falling_f > 0) period /= 2.0;
                if (period <= 1e-5) continue;

                if (this->frequency_lo > 1.0 / period || this->frequency_hi < 1.0 / period) continue;
                ret.at<uchar>(i, j) = 255;
            }
        }
        return ret;
    }
};


#endif // FILTERS_H
