#ifndef PLOT_H
#define PLOT_H

#include <common.h>
#include <trajectory.h>


class TjPlot {
protected:
    std::string name;
    std::vector<cv::Mat> plots;
    std::vector<cv::Mat> plots_cache;
    int res_x, res_y;
    double t_rng;
    float frame;
    bool initialized;

public:
    TjPlot(std::string name, int res_x, int res_y, float frame=10)
        : name(name), res_x(res_x), res_y(res_y), t_rng(-1), frame(frame), initialized(false) {}

    void add_trajectory_plot(Trajectory &tj, float shift=0.0);

    void add_vertical(float t0, int p_id=-1) {
        t0 *= float(this->res_x) / this->t_rng;
        float t_test = t0;

        for (auto &img : this->plots_cache) {
            if(t0 >= 0) {
                cv::line(img, {(int)t0, (int)(this->frame)}, {(int)t0, this->res_y - (int)(this->frame)}, cv::Scalar(0, 0, 255));
            }
        }
    }

    cv::Mat get_plot() {
        cv::Mat plot;
        cv::vconcat(this->plots_cache.data(), this->plots_cache.size(), plot);
        this->plots_cache.clear();
        for (auto &img : this->plots)
            this->plots_cache.push_back(img.clone());
        return plot;
    }

    void show() {
        if (!this->initialized) {
            this->initialized = true;
            cv::namedWindow(this->name, cv::WINDOW_NORMAL);
        }

        auto plot = this->get_plot();
        cv::imshow(this->name, plot);
    }

protected:
    void rotation2global(std::valarray<float> &arr) {
        auto last_val = arr[0];
        for (size_t i = 1; i < arr.size(); ++i) {
            auto cnt = std::round((last_val - arr[i]) / (2 * M_PI));
            arr[i] += cnt * 2 * M_PI;
            last_val = arr[i];
        }
    }
};


#endif // PLOT_H
