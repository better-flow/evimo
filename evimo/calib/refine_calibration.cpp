#include <vector>
#include <valarray>
#include <algorithm>
#include <thread>
#include <type_traits>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <ros/package.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/plot.hpp>

// VICON
#include <vicon/Subject.h>

// DVS / DAVIS
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

// Local includes
#include <common.h>
#include <event.h>
#include <event_vis.h>
#include <dataset.h>
#include <object.h>
#include <filters.h>
#include <matplotlibcpp.h>

// Detect Vicon Wand
#include "detect_wand.h"


class KeypointTracker {
protected:
    struct KeypointDetections {
        std::vector<cv::KeyPoint> raw_keypoints;
        std::vector<int32_t> next_in_chain_ids;
        std::vector<bool> is_first_in_chain;
        std::vector<bool> is_noise;
        std::vector<std::map<int, size_t>> labels;

        KeypointDetections(std::vector<cv::KeyPoint> &k)
            : raw_keypoints(k) {
            this->next_in_chain_ids.resize(this->raw_keypoints.size(), -1);
            this->is_first_in_chain.resize(this->raw_keypoints.size(), true);
            this->is_noise.resize(this->raw_keypoints.size(), false);
            this->labels.resize(this->raw_keypoints.size());
        }

        // nearest neighbour mather - *modifies 'k'!* - connects nearest points between this and k
        void match(KeypointDetections &k, float distance_th) {
            for (size_t i = 0; i < this->raw_keypoints.size(); ++i) {
                float min_d = -1;
                size_t min_j = k.raw_keypoints.size();
                for (size_t j = 0; j < k.raw_keypoints.size(); ++j) {
                    auto &p0 = this->raw_keypoints[i].pt;
                    auto &p1 = k.raw_keypoints[j].pt;
                    float d = std::hypot(float(p0.x - p1.x), float(p0.y - p1.y));
                    if (k.is_first_in_chain[j] && (min_d < 0 || d < min_d)) {
                        min_d = d;
                        min_j = j;
                    }
                }

                if (min_d >= 0 && min_d <= distance_th) {
                    this->next_in_chain_ids[i] = min_j;
                    k.is_first_in_chain[min_j] = false;
                }
            }
        }

        // increment lablel count for 'label' of 'j' by 1
        void add_label(size_t j, int label) {
            assert(j < this->raw_keypoints.size());
            auto search = this->labels[j].find(label);
            if (search == this->labels[j].end()) {
                this->labels[j][label] = 0;
            }
            this->labels[j][label] ++;
        }
    };

    std::vector<KeypointDetections> keypoint_detections;
    std::vector<double> detection_timestamps;
    float max_pps; 

public:
    KeypointTracker(float max_pps) : max_pps(max_pps) {};

    size_t size() {return this->keypoint_detections.size(); }

    // helper print for a given timestamp (idx)
    void print_stats(size_t idx) {
        auto &kptd = this->keypoint_detections[idx];
        if (kptd.raw_keypoints.size() == 0) return;

        std::cout << idx << ":\n";
        for (size_t j = 0; j < kptd.raw_keypoints.size(); ++j) {
            std::cout << "\t" << j << " " << (kptd.is_noise[j] ? "noise" : "  +  ") << ":\t";
            for (auto &lbl : kptd.labels[j]) {
                std::cout << "{" << lbl.first << " " << lbl.second << "}\t";
            }
            std::cout << "\n";
        }
    }

    // get keypoints, their timestamp and raw ids - since only non-noise keypoints are returned
    std::tuple<std::vector<cv::KeyPoint>, double, std::vector<size_t>, std::vector<int>> operator [] (size_t idx) {
        assert (idx < this->size());
        auto &k = this->keypoint_detections[idx];
        std::vector<cv::KeyPoint> ret;
        std::vector<size_t> ret_idx;
        std::vector<int> labels;
        for (size_t i = 0; i < k.raw_keypoints.size(); ++i) {
            if (k.is_noise[i]) continue;
            ret.push_back(k.raw_keypoints[i]);
            ret_idx.push_back(i);
            labels.push_back(k.labels[i].size() > 0 ? k.labels[i].begin()->first : -1);
        }
        return {ret, this->detection_timestamps[idx], ret_idx, labels};
    }

    void add_next(std::vector<cv::KeyPoint> &k, ull ts) {
        this->add_next(k, (double)((long double)(ts) * 1e-9));
    }

    // add some keyponts and linke them to prevously detected ones
    void add_next(std::vector<cv::KeyPoint> &k, double ts) {
        if (this->size() > 0) {
            if (this->detection_timestamps.back() > ts) {
                std::cout << _red("KeypointTracker: keypoint detections can only be added in the order of timestamps!") << std::endl;
                return;
            }
        }

        this->detection_timestamps.push_back(ts);
        this->keypoint_detections.emplace_back(k);

        if (this->size() >= 2) {
            float max_shift = this->max_pps * (this->detection_timestamps.back() - this->detection_timestamps[this->size() - 2]);
            this->keypoint_detections[this->size() - 2].match(this->keypoint_detections.back(), max_shift);
        }
    }

    // make sure at any given time no 2 labels co-exist with the same id, and all keypoints have 1 id
    void filter_track_id_majority_vote() {
        // this may break connectivity between keypoints across time
        for (size_t i = 0; i < this->size(); ++i) {
            std::map<int, size_t> max_counts;
            for (size_t j = 0; j < this->keypoint_detections[i].raw_keypoints.size(); ++j) {
                if (this->keypoint_detections[i].is_noise[j]) continue;
                if (this->keypoint_detections[i].labels.size() == 0) continue;
                if (this->keypoint_detections[i].labels.at(j).size() == 0) continue;

                auto max_lbl = this->keypoint_detections[i].labels.at(j).begin()->first;
                size_t max_cnt = this->keypoint_detections[i].labels.at(j).begin()->second;
                for (auto &lbl : this->keypoint_detections[i].labels.at(j)) {
                    if (lbl.second < max_cnt) continue;
                    max_cnt = lbl.second;
                    max_lbl = lbl.first;
                }

                if (max_counts.find(max_lbl) == max_counts.end()) {
                    max_counts[max_lbl] = max_cnt;
                }
                max_counts[max_lbl] = std::max(max_counts[max_lbl], max_cnt);
                this->keypoint_detections[i].labels.at(j) = {{max_lbl, max_cnt}};
            }

            for (size_t j = 0; j < this->keypoint_detections[i].raw_keypoints.size(); ++j) {
                if (this->keypoint_detections[i].labels.at(j).size() == 0) continue;
                auto &lbl = *(this->keypoint_detections[i].labels.at(j).begin());
                if (max_counts[lbl.first] > lbl.second) this->label_track_as_noise(i, j);
            }
        }
    }

    // uses timestamps to remove (label as noise) tracks shorter in span than min_track_len
    void remove_short_tracks(double min_track_len) {
        for (size_t i = 0; i < this->size(); ++i) {
            for (size_t j = 0; j < this->keypoint_detections[i].raw_keypoints.size(); ++j) {
                if (!this->keypoint_detections[i].is_first_in_chain[j]) continue;
                auto start_ts = this->detection_timestamps[i];
                auto end_ts   = this->get_track_end_ts(i, j);
                if (end_ts - start_ts >= min_track_len) continue;
                this->label_track_as_noise(i, j);
            }
        }
    }

    // apply label to the track; increment the label counter for all points in track (i_ j_) - any point on the track
    void propagate_track_label(size_t i_, size_t j_, int label) {
        this->apply_to_track(i_, j_, [&](size_t i, size_t j) {
            this->keypoint_detections[i].add_label(j, label);});
    }

    // find the ts on the end of the trck (i_ j_) - any point on the track
    double get_track_end_ts(size_t i_, size_t j_) {
        double ts = 0;
        this->apply_to_track_fw(i_, j_, [&](size_t i, size_t j) {
            ts = this->detection_timestamps[i];});
        return ts;
    }

    // labels track as noise *only* staring at (i_ j_) 
    void label_track_as_noise(size_t i_, size_t j_) {
        this->apply_to_track(i_, j_, [&](size_t i, size_t j) {
            this->keypoint_detections[i].is_noise[j] = true;});
    }

protected:
    template<class T>
    void apply_to_track(size_t i_, size_t j_, T func) {
        assert (i_ < this->size());
        assert (j_ < this->keypoint_detections[i_].raw_keypoints.size());
        if (i_ < this->size() - 1) {
            auto j = this->keypoint_detections[i_].next_in_chain_ids[j_];
            if (j >= 0) this->apply_to_track_fw(i_ + 1, j, func);
        }
        this->apply_to_track_bw(i_, j_, func);
    }

    template<class T>
    void apply_to_track_fw(size_t i_, size_t j_, T func) {
        assert (i_ < this->size());
        assert (j_ < this->keypoint_detections[i_].raw_keypoints.size());
        int32_t j = j_;
        for (size_t i = i_; i < this->size(); ++i) {
            func(i, j);
            j = this->keypoint_detections[i].next_in_chain_ids[j];
            if (j < 0) return;
        }
    }

    template<class T>
    void apply_to_track_bw(size_t i_, size_t j_, T func) {
        assert (i_ < this->size());
        assert (j_ < this->keypoint_detections[i_].raw_keypoints.size());
        int32_t j = j_;
        for (size_t i = i_; i >= 0; i--) {
            func(i, j);
            if (i == 0) return;
            bool found = false;
            for (size_t k = 0; k < this->keypoint_detections[i - 1].raw_keypoints.size(); ++k) {
                if (this->keypoint_detections[i - 1].next_in_chain_ids[k] == j) {
                    j = k;
                    found = true;
                    break;
                }
            }
            if (!found) return;
        }
    }
};


struct tuple_foreach {
    template <typename T, int N=std::tuple_size<T>::value>
    static void init(T &data, const typename std::tuple_element<N - 1, T>::type &val) {
        std::get<N - 1>(data) = val;
        if constexpr (N - 1 > 0) tuple_foreach::init<T, N - 1>(data, val);
    }

    template <typename T, int N=std::tuple_size<T>::value>
    static T init(const typename std::tuple_element<N - 1, T>::type &val) {
        T data;
        tuple_foreach::init(data, val);
        return data;
    }
 
    // ret = a * b + c
    template <typename T, int N=std::tuple_size<T>::value>
    static void fma(T &data, const T &a, const typename std::tuple_element<N - 1, T>::type &b, const T &c) {
        std::get<N - 1>(data) = std::get<N - 1>(a) * b + std::get<N - 1>(c);
        if constexpr (N - 1 > 0) tuple_foreach::fma<T, N - 1>(data, a, b, c);
    }

    template <typename T, int N=std::tuple_size<T>::value>
    static T fma(const T &a, const typename std::tuple_element<N - 1, T>::type &b, const T &c) {
        T data;
        tuple_foreach::fma(data, a, b, c);
        return data;
    }

    // return sum of squares in tuple
    template <typename T, int N=std::tuple_size<T>::value>
    static void sqlen(typename std::tuple_element<N - 1, T>::type &data, const T &a) {
        data += std::pow(std::get<N - 1>(a), (typename std::tuple_element<N - 1, T>::type)2.0);
        if constexpr (N - 1 > 0) tuple_foreach::sqlen<T, N - 1>(data, a);
    }

    template <typename T, int N=std::tuple_size<T>::value>
    static typename std::tuple_element<N - 1, T>::type sqlen(const T &a) {
        typename std::tuple_element<N - 1, T>::type data;
        tuple_foreach::sqlen<T, N>(data, a);
        return data;
    }

    template <typename T, int N=std::tuple_size<T>::value>
    static typename std::tuple_element<N - 1, T>::type hypot(const T &a) {
        auto data = sqlen(a);
        return std::sqrt(data);
    }
};


template <typename T> class Track {
protected:
    struct Tracklet {
        double ts;
        T pose;
        Tracklet(double ts, T p) : ts(ts), pose(p) {}
        double get_ts_sec() const {return this->ts;}
    };

    std::vector<Tracklet> track;

public:
    Track() {}

    void push_back(double ts, T pose) {
        this->track.emplace_back(ts, pose);
    }

    auto begin() {return this->track.begin(); }
    auto end()   {return this->track.end();   }
    auto size() const {return this->track.size();  }
    inline const Tracklet& operator [] (size_t idx) const {
        assert (idx < this->size());
        return this->track[idx];
    }

    void apply_tf(const tf::Transform &t) {
        if constexpr (std::tuple_size<T>::value == 3) {
            for (size_t i = 0; i < this->size(); ++i) {
                auto &p = this->track[i].pose;
                auto ret_p = t({std::get<0>(p), std::get<1>(p), std::get<2>(p)});
                std::get<0>(p) = ret_p.x();
                std::get<1>(p) = ret_p.y();
                std::get<2>(p) = ret_p.z();
            }
        }
    }

    void apply_time_offset(double to) {
        for (size_t i = 0; i < this->size(); ++i) this->track[i].ts += to;
    }

    template<typename TS_T>
    Track evaluate(TS_T timestamps) const {
        Track<T> ret;
        for (auto &request_ts : timestamps) {
            T default_val = tuple_foreach::init<T>(0);

            if (this->track.size() == 0 || request_ts < this->track[0].get_ts_sec() ||
                request_ts > this->track.back().get_ts_sec()) {
                ret.push_back(request_ts, default_val);
                continue;
            }

            auto j0_j1 = TimeSlice(this->track, {request_ts - 1e-5, request_ts + 1e-5}, {0, 0}).get_indices();
            auto j1 = j0_j1.second;
            assert(j1 < this->track.size());
            if (request_ts > this->track[j1].get_ts_sec()) j1 += 1;
            auto j0 = j1 - 1;
            assert(j0 >= 0);
            assert(j1 < this->track.size());

            double t_frac = (request_ts - this->track[j0].get_ts_sec()) / (this->track[j1].get_ts_sec() - this->track[j0].get_ts_sec());
            auto dp = tuple_foreach::fma(this->track[j0].pose, -1, this->track[j1].pose); // (j1 - j0):
            auto pp = tuple_foreach::fma(dp, t_frac, this->track[j0].pose); // j0 + (j1 - j0) * t_frac

            ret.push_back(request_ts, pp);
        }

        return ret;
    }

    Track derivative() const {
        Track<T> ret;
        if (this->track.size() == 0) return ret;
        ret.push_back(this->track[0].ts, tuple_foreach::init<T>(0));

        for (size_t i = 1; i < this->track.size() - 1; ++i) {
            auto dt = this->track[i + 1].ts - this->track[i - 1].ts;
            if (std::fabs(dt) < 1e-5) {
                ret.push_back(this->track[i].ts, tuple_foreach::init<T>(0));
                continue;
            }

            auto dp = tuple_foreach::fma(this->track[i - 1].pose, -1, this->track[i + 1].pose);
            auto dp_dt = tuple_foreach::fma(dp, 1.0 / dt, tuple_foreach::init<T>(0));
            ret.push_back(this->track[i].ts, dp_dt);
        }

        if (this->track.size() > 1) {
            ret.push_back(this->track.back().ts, ret.track.back().pose);
            ret.track[0].pose = ret.track[1].pose;
        }

        assert(this->track.size() == ret.track.size());
        return ret;
    }
};


class CrossCorrelator {
public:
    CrossCorrelator() {}

    // brute-force cross-correlator: will try all offsets in [-wsize/2, wsize/2] with a step 'step'
    template <typename T>
    static double correlate(const Track<T> &ref_track, const Track<T> &target_track,
                            double step=1e-2, double wsize=1.0, double wcenter=0.0, bool plot=false) {
        std::valarray<double> ref_ts(ref_track.size());
        std::vector<double> ref_val(ref_track.size());
        for (size_t i = 0; i < ref_track.size(); ++i) {
            ref_ts[i] = ref_track[i].get_ts_sec();
            ref_val[i] = tuple_foreach::hypot(ref_track[i].pose);
        }

        long double mean_val = 0, min_val = -1, max_val = -1;
        for (size_t i = 0; i < ref_val.size(); ++i) {
            mean_val += ref_val[i];
            if (min_val > ref_val[i] || min_val < 0) min_val = ref_val[i];
            if (max_val < ref_val[i] || max_val < 0) max_val = ref_val[i];
        }
        mean_val /= (long double)ref_track.size();

        for (size_t i = 0; i < ref_track.size(); ++i) {
            ref_val[i] = (ref_val[i] - mean_val) / (max_val - min_val < 1e-6 ? 1 : max_val - min_val);
        }

        double offset = -wsize / 2 + wcenter;
        ref_ts += offset;

        std::vector<double> errors;
        std::vector<double> e_offsets;

        double best_offset = offset;
        double best_error = -1;
        while (offset <= wsize / 2 + wcenter) {
            auto tt = target_track.evaluate(ref_ts);
            std::vector<double> tgt_val(tt.size());
            for (size_t i = 0; i < tt.size(); ++i) {
                tgt_val[i] = tuple_foreach::hypot(tt[i].pose);
                if (std::fabs(tgt_val[i]) > 1e-3)
                    tgt_val[i] = (tgt_val[i] - mean_val) / (max_val - min_val < 1e-6 ? 1 : max_val - min_val);
            }

            assert(tgt_val.size() == ref_val.size());
            long double score = 0;
            size_t count = 0;
            for (size_t i = 0; i < tt.size(); ++i) {
                long double val = (long double)(tgt_val[i]) * (long double)(ref_val[i]);
                if (std::fabs(val) < 1e-5) continue; // 0 values after interpolation means point is out of timestamp range
                count++;
                score += val;
            }
            score /= (long double)count;

            errors.push_back(score);
            e_offsets.push_back(offset);

            // largest score search
            if (best_error < 0 || best_error < score) {
                best_error = score;
                best_offset = offset;
            }

            offset += step;
            ref_ts += step;
        }

        // plot the result
        // reset timestamps to original; remove data scaling
        if (plot) {
            for (size_t i = 0; i < ref_track.size(); ++i) {
                ref_ts[i] = ref_track[i].get_ts_sec();
            }

            for (size_t i = 0; i < ref_track.size(); ++i) {
                ref_val[i] = tuple_foreach::hypot(ref_track[i].pose);
            }

            auto tt = target_track.evaluate(ref_ts);
            std::vector<double> tgt_val_orig(tt.size());
            for (size_t i = 0; i < tt.size(); ++i) {
                tgt_val_orig[i] = tuple_foreach::hypot(tt[i].pose);
            }

            ref_ts += best_offset;

            tt = target_track.evaluate(ref_ts);
            std::vector<double> tgt_val_corr(tt.size());
            for (size_t i = 0; i < tt.size(); ++i) {
                tgt_val_corr[i] = tuple_foreach::hypot(tt[i].pose);
            }

            std::vector<double> ref_ts_v;
            ref_ts_v.assign(std::begin(ref_ts), std::end(ref_ts));

            namespace plt = matplotlibcpp;
            plt::figure_size(1200, 780);
            plt::title("Timestamp offset correction result:");
            plt::subplot(2,1,1);
            plt::named_plot("reference",        ref_ts_v, ref_val, ".");
            plt::named_plot("target original",  ref_ts_v, tgt_val_orig, ".");
            plt::named_plot("target corrected", ref_ts_v, tgt_val_corr, ".");
            plt::legend();
            plt::subplot(2,1,2);
            plt::named_plot("correlation",      e_offsets, errors, ".");
            plt::legend();
            plt::show();
        }
        return best_offset;
    }
};


class ViconWandTimeAligner : public CrossCorrelator {
public:
    static double align(const std::map<int, Track<std::tuple<float, float>>> &src,
                        const std::map<int, Track<std::tuple<float, float>>> &tgt,
                        double step=1e-2, double wsize=1.0, bool plot=false) {

        int best_label = -1;
        size_t largest_len = 0;

        for (auto &label : std::vector({1,2,3,4,5})) {
            if (src.find(label) == src.end()) continue;
            if (tgt.find(label) == tgt.end()) continue;
            auto size = std::min(src.at(label).size(), tgt.at(label).size());
            if (size > largest_len) {
                largest_len = size;
                best_label = label;
            }
        }

        if (largest_len == 0) {
            std::cout << _red("time algnemnt failed! no tracks available") << std::endl;
            return 0;
        }

        std::cout << "Time-aligning label " << best_label << " with " << largest_len << " points" << std::endl;
        auto img_pt_v = src.at(best_label).derivative();
        auto wnd_pt_v = tgt.at(best_label).derivative();

        double wcenter = 0.0;
        std::cout << "\n\tfirst pass with step " << step << "; wsize " 
                  << wsize << "; offset guess " << wcenter << std::endl;
       
        wcenter = correlate(img_pt_v, wnd_pt_v, step, wsize, wcenter, plot);
        std::cout << "\t\t...offset = " << wcenter << std::endl;

        wsize = step * 5;
        step = step / 10;
        std::cout << "\n\tsecond pass with step " << step << "; wsize " 
                  << wsize << "; offset guess " << wcenter << std::endl;

        wcenter = correlate(img_pt_v, wnd_pt_v, step, wsize, wcenter, plot);
        std::cout << "\t\t...offset = " << wcenter << std::endl;
        return wcenter;
    }
};


class ViconWandSpatialCalibrator {
protected:
    std::shared_ptr<Dataset> dataset;

    // used in optimization
    std::vector<cv::Point2f> image_pts;
    std::vector<cv::Point3f> object_pts;

public:
    ViconWandSpatialCalibrator(std::shared_ptr<Dataset> &dataset) : dataset(dataset) {}

    void reset() {
        image_pts.clear();
        object_pts.clear();
    }

    void remove_outliers() {
        assert(this->image_pts.size() == this->object_pts.size());

        std::vector<float> errors;
        errors.reserve(this->image_pts.size());
        for (size_t i = 0; i < this->image_pts.size(); ++i) {
            auto pt2d = this->image_pts[i];
            auto pt3d = this->object_pts[i];

            float u = -1, v = -1;
            if (pt3d.z > 0.001) {
                this->dataset->project_point(pt3d, u, v);
            }

            if (u < 0 || v < 0) {
                errors.push_back(-1);
                continue;
            }

            errors.push_back(std::hypot(pt2d.x - v, pt2d.y - u));
        }

        assert(this->image_pts.size() == errors.size());

        long double mean_error = 0, count = 0;
        for (auto &e : errors) {
            if (e < 0) continue;
            mean_error += e;
            count += 1;
        }

        mean_error /= count;
        std::cout << "Mean error\t" << _yellow(std::to_string(mean_error)) << std::endl;
        std::cout << "# points:\t" << this->image_pts.size() << std::endl;
        std::cout << "# visible points:\t" << count << std::endl;

        std::vector<size_t> inliers;
        inliers.reserve(count);
        for (size_t i = 0; i < errors.size(); ++i) {
            if (errors[i] < 0 || errors[i] > mean_error) continue;
            inliers.push_back(i);
        }

        for (size_t i = 0; i < inliers.size(); ++i) {
            this->image_pts[i]  = this->image_pts[inliers[i]];
            this->object_pts[i] = this->object_pts[inliers[i]];
        }

        this->image_pts.resize(inliers.size());
        this->object_pts.resize(inliers.size());

        std::cout << "# points (filtered):\t" << this->image_pts.size() << std::endl;
    }


    void add_tracks(const std::map<int, Track<std::tuple<float, float>>> &src,
                    const std::map<int, Track<std::tuple<float, float, float>>> &tgt) {

        for (auto &label : std::vector({1,2,3,4,5})) {
            if (src.find(label) == src.end()) continue;

            auto n_points = src.at(label).size();
            auto &s_tt = src.at(label);

            std::vector<double> tgt_ts(n_points);
            for (size_t i = 0; i < n_points; ++i) {
                tgt_ts[i] = s_tt[i].get_ts_sec();
            }

            auto t_tt = tgt.at(label).evaluate(tgt_ts);
            assert(t_tt.size() == n_points);
            for (size_t i = 0; i < n_points; ++i) {
                if (std::get<0>(s_tt[i].pose) < 1e-3 || std::get<1>(s_tt[i].pose) < 1e-3) continue;
                if (std::fabs(std::get<0>(t_tt[i].pose)) < 1e-3 && std::fabs(std::get<1>(t_tt[i].pose)) < 1e-3 &&
                    std::fabs(std::get<2>(t_tt[i].pose)) < 1e-3) continue;
                auto point_3d = t_tt[i].pose;

                // add input points
                this->image_pts.push_back({std::get<0>(s_tt[i].pose), std::get<1>(s_tt[i].pose)});
                this->object_pts.push_back({std::get<0>(t_tt[i].pose), std::get<1>(t_tt[i].pose), std::get<2>(t_tt[i].pose)});
            }
        }
    }


    tf::Transform calibrate(bool extr_only=false, bool plot=true) {
        this->remove_outliers();

        std::vector<double> src_val_x;
        std::vector<double> src_val_y;
        std::vector<double> tgt_val_x;
        std::vector<double> tgt_val_y;
        if (plot) {
            // project points for visualization
            assert(this->image_pts.size() == this->object_pts.size());
            for (size_t i = 0; i < this->image_pts.size(); ++i) {
                auto pt2d = this->image_pts[i];
                auto pt3d = this->object_pts[i];

                float u = -1, v = -1;
                if (pt3d.z > 0.001) {
                    this->dataset->project_point(pt3d, u, v);
                }

                //if (u >= 0 && v >= 0) {
                    src_val_x.push_back(pt2d.x);
                    src_val_y.push_back(pt2d.y);
                    tgt_val_x.push_back(v);
                    tgt_val_y.push_back(u);
                //}
            }
        }

        // optimization
        std::vector<cv::Mat> rvecs, tvecs;
        int flags = cv::CALIB_USE_INTRINSIC_GUESS; 
        flags += cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5 + cv::CALIB_FIX_K6;
        if (extr_only)
            flags += cv::CALIB_FIX_PRINCIPAL_POINT + cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K1 + cv::CALIB_FIX_K2;

        cv::Mat K = (cv::Mat1d(3, 3) << dataset->fx, 0, dataset->cx, 0, dataset->fy, dataset->cy, 0, 0, 1);
        cv::Mat D;

        if (dataset->dist_model == "radtan") {
            D = (cv::Mat1d(1, 4) << dataset->k1, dataset->k2, dataset->p1, dataset->p2);

            if (extr_only) {
                cv::solvePnPGeneric(this->object_pts, this->image_pts, K, D, rvecs, tvecs);
                //cv::solvePnPRansac(this->object_pts, this->image_pts, K, D, rvecs, tvecs);
            }
            else {
                std::vector<std::vector<cv::Point2f>> imagePoints = {this->image_pts};
                std::vector<std::vector<cv::Point3f>> objectPoints = {this->object_pts};
                cv::calibrateCamera(objectPoints, imagePoints, {(int)dataset->res_x, (int)dataset->res_y}, K, D, rvecs, tvecs, flags);
            }
        } else {
            std::cout << _red("Unknown distortion model! ") << dataset->dist_model << std::endl;
        }

        auto &rvec = rvecs[0];
        auto &tvec = tvecs[0];

        assert(K.depth() == CV_64F);
        assert(D.depth() == CV_64F);
        assert(rvec.depth() == CV_64F);
        assert(tvec.depth() == CV_64F);

        // update 'dataset'
        auto E_correction = rvec_tvec2tf(rvec, tvec);
        dataset->cam_E = dataset->cam_E * E_correction.inverse();
        dataset->fx = K.at<double>(0);
        dataset->fy = K.at<double>(4);
        dataset->cx = K.at<double>(2);
        dataset->cy = K.at<double>(5);
        if (!extr_only) {
            if (dataset->dist_model == "radtan") {
                dataset->k1 = D.at<double>(0);
                dataset->k2 = D.at<double>(1);
                dataset->k3 = 0;
                dataset->k4 = 0;
                dataset->p1 = D.at<double>(2);
                dataset->p2 = D.at<double>(3);
            } else {
                std::cout << _red("Unknown distortion model! ") << dataset->dist_model << std::endl;
            }
        }
        dataset->apply_Intr_Calib();


        // Compute error
        std::vector<cv::Point2f> tgt_pts;
        cv::projectPoints(this->object_pts, rvec, tvec, K, D, tgt_pts);
        long double mean_error = 0, count = 0;
        for (size_t i = 0; i < this->image_pts.size(); ++i) {
            mean_error += std::hypot(tgt_pts[i].x - this->image_pts[i].x, tgt_pts[i].y - this->image_pts[i].y);
            count += 1;
        }
        mean_error /= count;
        std::cout << "Mean error after optimization\t" << _yellow(std::to_string(mean_error)) << std::endl;
  
        // Correct 3d point position in camera frame
        for (size_t i = 0; i < this->object_pts.size(); ++i) {
            auto &p3d = this->object_pts[i];
            auto p3d_ = E_correction({p3d.x, p3d.y, p3d.z});
            p3d.x = p3d_.x(); p3d.y = p3d_.y(); p3d.z = p3d_.z(); 
        }

        // plot
        if (plot) {
            std::vector<double> src_val_x_corrected;
            std::vector<double> src_val_y_corrected;
            std::vector<double> tgt_val_x_corrected;
            std::vector<double> tgt_val_y_corrected;
            std::vector<double> z_values;

            assert(this->image_pts.size() == this->object_pts.size());
            for (size_t i = 0; i < this->object_pts.size(); ++i) {
                auto &p3d = this->object_pts[i];

                float u = -1, v = -1;
                if (p3d.z > 0.001) {
                    //struct pXYZ {double x, y, z; };
                    //dataset->project_point(pXYZ{p3d.x(), p3d.y(), p3d.z()}, u, v);
                    dataset->project_point(p3d, u, v);
                }

                z_values.push_back(p3d.z);

                //if (u >= 0 && v >= 0) {
                    src_val_x_corrected.push_back(this->image_pts[i].x);
                    src_val_y_corrected.push_back(this->image_pts[i].y);
                    tgt_val_x_corrected.push_back(v);
                    tgt_val_y_corrected.push_back(u);
                    //tgt_val_x_corrected.push_back(tgt_pts[i].x);
                    //tgt_val_y_corrected.push_back(tgt_pts[i].y);
                //}
            }

            namespace plt = matplotlibcpp;
            plt::figure_size(1200, 780);
            plt::title("Spatial calibrator");
            plt::subplot(3,2,1);
            plt::named_plot("detections", src_val_x, ".");
            plt::named_plot("vicon",      tgt_val_x, ".");
            plt::legend();

            plt::subplot(3,2,2);
            plt::named_plot("detections", src_val_y, ".");
            plt::named_plot("vicon",      tgt_val_y, ".");
            plt::legend();

            plt::subplot(3,2,3);
            plt::named_plot("detections", src_val_x_corrected, ".");
            plt::named_plot("vicon",      tgt_val_x_corrected, ".");
            plt::legend();

            plt::subplot(3,2,4);
            plt::named_plot("detections", src_val_y_corrected, ".");
            plt::named_plot("vicon",      tgt_val_y_corrected, ".");
            plt::legend();

            plt::subplot(3,2,5);
            plt::named_hist("z distribution", z_values, 100);
            plt::legend();

            plt::subplot(3,2,6);
            plt::named_plot("pixel distribution detections", src_val_x_corrected, src_val_y_corrected, ".");
            plt::named_plot("pixel distribution wand", tgt_val_x_corrected, tgt_val_y_corrected, ".");
            plt::legend();

            plt::show();
        }

        return E_correction;
    }

protected:
    static tf::Quaternion rvec2q(cv::Mat &rvec) {
        assert(rvec.depth() == CV_64F);
        double ax = rvec.at<double>(0), ay = rvec.at<double>(1), az = rvec.at<double>(2);
        double angle = std::sqrt(ax * ax + ay * ay + az * az);
        if (angle > 1e-6) {
            ax /= angle;
            ay /= angle;
            az /= angle;
        }

        tf::Quaternion q({ax, ay, az}, angle);
        q.normalize();
        return q;
    }

    static tf::Transform rvec_tvec2tf(cv::Mat &rvec, cv::Mat &tvec) {
        assert(tvec.depth() == CV_64F);
        auto q = rvec2q(rvec);
        tf::Vector3 T(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        tf::Transform E;
        E.setRotation(q);
        E.setOrigin(T);
        return E;
    }
};



// generate tracks from a bag file
std::pair<std::map<int, Track<std::tuple<float, float>>>, std::map<int, Track<std::tuple<float, float, float>>>>
extract_tracks(ros::NodeHandle &nh, std::string bag_name, std::string camera_name, float start_time_offset = 0.0,
               float sequence_duration = -1.0, bool dilate_blobs=false, int img_th_value=-1) {
    std::string wand_topic = "/vicon/Wand";
    nh.getParam("wand_topic", wand_topic);

    double e_slice_width = 0.02;
    double e_fps = 2.0 / e_slice_width; // frame every half event slice width by default
    nh.getParam("e_slice_width", e_slice_width);
    nh.getParam("e_fps", e_fps);

    // Read dataset configuration files
    std::string path_to_self = ros::package::getPath("evimo");
    std::string dataset_folder = path_to_self + "/config/"; // default
    if (!nh.getParam("folder", dataset_folder)) {
        std::cout << _yellow("No configuration folder specified! Using: ")
                  << dataset_folder << std::endl;
    }

    std::pair<std::map<int, Track<std::tuple<float, float>>>, std::map<int, Track<std::tuple<float, float, float>>>> err_ret;

    bool with_images = true;
    if (!nh.getParam("with_images", with_images)) with_images = true;
    else std::cout << _yellow("With 'with_images' option, the datased will be generated at image framerate.") << std::endl;

    auto dataset = std::make_shared<Dataset>();
    if (!dataset->init_no_objects(dataset_folder, camera_name))
        return err_ret;

    // Force wand trajectory
    dataset->obj_pose_topics[0] = wand_topic;

    // Extract topics from bag
    if (!dataset->read_bag_file(bag_name, start_time_offset, sequence_duration, with_images, false)) {
        return err_ret;
    }
    with_images = (dataset->images.size() > 0);

    // blob tracker max blob pixel-per-second speed (here normalized by camera resolution)
    double tracker_max_pps = 1000 * std::max(dataset->res_y, dataset->res_x) / 640.0;
    nh.getParam("tracker_max_pps", tracker_max_pps);
    double tracker_min_len = 0.3;
    nh.getParam("tracker_min_len", tracker_min_len);

    // Make sure there is no trajectory filtering
    dataset->cam_tj.set_filtering_window_size(-1);
    dataset->obj_tjs[0].set_filtering_window_size(-1);

    // Convert wand markers to camera frame
    auto &cam_tj = dataset->cam_tj;
    auto &wnd_tj = dataset->obj_tjs[0];
    if (wnd_tj.size() == 0 || cam_tj.size() == 0) {
        std::cout << _red("No trajectory for either wand or sensor rig!") << std::endl;
        return err_ret;
    }

    std::map<int, Track<std::tuple<float, float, float>>> wand_red_3d;
    std::map<int, Track<std::tuple<float, float>>> wand_red_pix;
    std::map<int, Track<std::tuple<float, float, float>>> wand_ir_3d;
    std::map<int, Track<std::tuple<float, float>>> wand_ir_pix;
    {
    auto c_rig = Pose(ros::Time(0), dataset->cam_E); // camera in rig coordinates (prior)
    for (size_t i = 0; i < cam_tj.size(); ++i) {
        std::cout << "\r\tConverting vicon to camera frame: (" << i + 1 << "\t/\t" << cam_tj.size() << ")\t\t";
        auto c_world = cam_tj[i];
        double ts = c_world.get_ts_sec();

        // trajectory points need to be aligned (this returns nearest index)
        size_t j = TimeSlice(wnd_tj, {ts - 1e-3, ts + 1e-3}, {0, 0}).get_indices().first;
        auto w_world = wnd_tj[j];

        if (std::fabs(w_world.get_ts_sec() - ts) > 1e-3) continue;

        auto w_rig = w_world - c_world; // wand in rig coordinates
        auto w_cam = w_rig - c_rig; // wand in camera coordinates

        // register ir led detections
        std::vector<std::valarray<float>> tracked_wand_ir;
        std::vector<int> marker_ids;
        for (auto &m : w_cam.markers) {
            int m_id = std::stoi(std::get<3>(m).substr(std::get<3>(m).find_last_not_of("0123456789") + 1));

            float u = -1, v = -1;
            if (std::get<2>(m) > 0.001) {
                struct pXYZ {double x, y, z; };
                dataset->project_point(pXYZ{std::get<0>(m), std::get<1>(m), std::get<2>(m)}, u, v);
            }

            wand_ir_3d[m_id].push_back(ts, {std::get<0>(m), std::get<1>(m), std::get<2>(m)});
            if (u >= 0 && v >= 0) wand_ir_pix[m_id].push_back(ts, {v, u});
            tracked_wand_ir.push_back({std::get<0>(m), std::get<1>(m), std::get<2>(m)});
            marker_ids.push_back(m_id);
        }

        // we will assume that marker ids are traversed in order
        for (size_t kk = 1; kk < marker_ids.size(); ++kk) {
            if (marker_ids[kk - 1] >= marker_ids[kk]) {
                std::cout << _red("Marker id traversal not in order; terminating!") << std::endl;
                return err_ret;
            }
        }

        // convert ir detections to red led detections
        auto wand_ir2red_tf = ViObject::estimate_tf(wand::wand_ir_mapping, tracked_wand_ir);
        for (size_t kk = 0; kk < wand::wand_red_mapping.size(); ++kk) {
            auto &pt = wand::wand_red_mapping[kk];
            auto red_pt = wand_ir2red_tf({pt[0], pt[1], pt[2]});
            wand_red_3d[kk + 1].push_back(ts, {red_pt.x(), red_pt.y(), red_pt.z()});

            float u = -1, v = -1;
            if (red_pt.z() > 0.001) {
                struct pXYZ {double x, y, z; };
                dataset->project_point(pXYZ{red_pt.x(), red_pt.y(), red_pt.z()}, u, v);
            }

            if (u >= 0 && v >= 0) wand_red_pix[kk + 1].push_back(ts, {v, u});
        }
    } std::cout << std::endl;
    }

    std::vector<double> image_timestamps;

    // Generate images from events
    if (!with_images || dataset->images.size() == 0) {
        assert(dataset->event_array.size() > 0);
        dataset->images.clear();
        size_t to_reserve = size_t((dataset->event_array.back().get_ts_sec() - dataset->event_array.front().get_ts_sec()) / e_fps + 1);
        dataset->images.reserve(to_reserve);
        image_timestamps.reserve(to_reserve);

        EventSlice2DFrequencyFilter eff(dataset->res_y, dataset->res_x, ull(e_slice_width * 1e9), 150, 250);
        //EventSlice2D eff(dataset->res_y, dataset->res_x, ull(e_slice_width * 1e9));

        double start_ts = dataset->event_array.front().get_ts_sec();
        for (size_t i = 0; i < dataset->event_array.size(); ++i) {
            eff.push_back(dataset->event_array[i]);
            double current_ts = dataset->event_array[i].get_ts_sec();
            if (i % 10000 == 0 || i >= dataset->event_array.size() - 1) {
                std::cout << "\r\tConverting events to images: (" << i + 1 << "\t/\t" 
                          << dataset->event_array.size() << ")\t\t\t";
            }

            if (current_ts - start_ts >= 1.0 / e_fps) {
                auto img = (cv::Mat)eff;
                auto ts = (current_ts + start_ts) / 2.0;
                image_timestamps.push_back(ts);

                if (dilate_blobs) {
                    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                                                cv::Size(5, 5),
                                                                cv::Point(2, 2));
                    cv::dilate(img, img, element);
                }

                dataset->images.push_back(img.clone());
                start_ts = current_ts;
            }
        } std::cout << std::endl;
    } else { // ...or filter camera images
        assert(dataset->images.size() > 0);
        image_timestamps.reserve(dataset->images.size());
        for (auto &ts : dataset->image_ts) image_timestamps.push_back(ts.toSec());
        for (size_t i = 0; i < dataset->images.size(); ++i) {
            if (dataset->images[i].channels() > 1) {
                std::vector<cv::Mat> ch;
                cv::split(dataset->images[i], ch);
                dataset->images[i] = ch[2].clone();
            }

            if (dataset->images[i].elemSize1() > 1) {
                dataset->images[i].convertTo(dataset->images[i], CV_8U, 1.0 / float(dataset->images[i].elemSize1()));
            }

            if (img_th_value > 0)
                cv::threshold(dataset->images[i], dataset->images[i], img_th_value, 255, 0);
        }
    }

    std::cout << _green("Using ") << dataset->images.size() << _green(" images") << std::endl;
    if (dataset->images.size() == 0) {
        return err_ret;
    }

    // Track filtering
    std::cout << "Tracker max pps: " << tracker_max_pps << std::endl;
    KeypointTracker kpt_tracker(tracker_max_pps); // max pixel-per-second blob speed
    for (size_t i = 0; i < dataset->images.size(); ++i) {
        std::cout << "\r\tExtracting blobs: (" << i + 1 << "\t/\t" << dataset->images.size() 
                  << ")\t\t" << std::flush;
        auto keypoints = wand::get_blobs(dataset->images[i], false, 100, 110, 5);
        kpt_tracker.add_next(keypoints, image_timestamps[i]);
    } std::cout << std::endl;

    std::cout << "Removing tracks shorter than: " << tracker_min_len << std::endl;
    kpt_tracker.remove_short_tracks(tracker_min_len);


    // detect wand and assign labels to keypoints (can be multiple labels per point)
    for (size_t i = 0; i < dataset->images.size(); ++i) {
        auto img_ = dataset->images[i];
        auto keypoint_tuple = kpt_tracker[i];

        auto wand_line_idx = wand::find_all_3lines(std::get<0>(keypoint_tuple), 0.2);
        auto wand_idx = wand::detect_wand_internal_idx(std::get<0>(keypoint_tuple), wand_line_idx, wand::wand_red_mapping, 0.5, 0.5, 0.5);
        auto kpt_idx = std::get<2>(keypoint_tuple);

        for (int label = 0; label < wand_idx.size(); ++label) {
            kpt_tracker.propagate_track_label(i, kpt_idx[wand_idx[label]], label + 1); // wand labels are 1..5
        }

/*
        cv::Mat img = img_.clone();
        cv::drawKeypoints(img, std::get<0>(keypoint_tuple), img, cv::Scalar(255,0,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::imshow("img", img);
        auto code = cv::waitKey(0);
        if (code == 27) break;
*/

    }

    // pick best labels for each keypoint
    kpt_tracker.filter_track_id_majority_vote();

    // extract pixel tracks
    std::map<int, Track<std::tuple<float, float>>> wand_detected_pix;
    for (size_t i = 0; i < kpt_tracker.size(); ++i) {
        auto keypoint_tuple = kpt_tracker[i];
        kpt_tracker.print_stats(i);

        auto kpts   = std::get<0>(keypoint_tuple);
        auto ts     = std::get<1>(keypoint_tuple);
        auto labels = std::get<3>(keypoint_tuple);

        for (size_t j = 0; j < kpts.size(); ++j) {
            if (labels[j] == -1) continue;
            wand_detected_pix[labels[j]].push_back(ts, {kpts[j].pt.x, kpts[j].pt.y});
        }
    }


    double to_step = 1e-2, to_wsize = 1.0;
    bool plot = false;
    double time_offset = ViconWandTimeAligner::align(wand_detected_pix, wand_red_pix, to_step, to_wsize, plot);
    std::cout << bag_name << _green(":\tcorrecting time offset: ") << time_offset << std::endl;

    for (auto &lbl_pair : wand_red_3d) {
        lbl_pair.second.apply_time_offset(-time_offset);
    }

    return {wand_detected_pix, wand_red_3d};
}



int main (int argc, char** argv) {

    // Initialize ROS
    std::string node_name = "calibration_refinement";
    ros::init (argc, argv, node_name, ros::init_options::AnonymousName);
    ros::NodeHandle nh("~");

    // Read dataset configuration files
    std::string path_to_self = ros::package::getPath("evimo");
    std::string dataset_folder = path_to_self + "/config/"; // default
    if (!nh.getParam("folder", dataset_folder)) {
        std::cout << _yellow("No configuration folder specified! Using: ")
                  << dataset_folder << std::endl;
    }

    std::string config_path = "";
    if (!nh.getParam("conf", config_path)) {
        std::cerr << "No configuration file specified!" << std::endl;
        return -1;
    }

    std::string camera_name = "";
    std::vector<std::tuple<std::string, float, float>> bag_names;

    bool dilate_blobs = false;
    int img_th_value = -1;

    // read and parse the config file
    std::ifstream ifs;
    ifs.open(config_path, std::ifstream::in);
    if (!ifs.is_open()) {
        std::cout << _red("Could not open configuration file file at ")
                  << config_path << "!" << std::endl;
        return -1;
    }
 
    auto sep = config_path.find_last_of("/");
    std::string root_folder = trim(config_path.substr(0, sep));
    std::cout << "Folder:\t" << root_folder << "\n";

    const std::string& delims = ":\t ";
    while (ifs.good()) {
        std::string line;
        std::getline(ifs, line);
        line = trim(line);
        if (line.size() == 0) continue;
        if (line[0] == '#') continue;
        auto sep = line.find_first_of(delims);

        std::string key = trim(line.substr(0, sep));
        line = trim(line.substr(sep + 1));
        sep = line.find_first_of(delims);
        std::string value = trim(line.substr(0, sep));

        if (key == "camera_name") {
            camera_name = value;
        } else if (key == "dilate_blobs") {
            if (value == "true") dilate_blobs = true;
        } else if (key == "image_th") {
            img_th_value = std::stoi(value);
        } else if (key == "bag_file") {
            line = trim(line.substr(sep + 1));
            sep = line.find_first_of(delims);
            float st_offset = std::stof(trim(line.substr(0, sep)));

            line = trim(line.substr(sep + 1));
            sep = line.find_first_of(delims);
            float s_len = std::stof(trim(line.substr(0, sep)));

            bag_names.push_back({root_folder + "/" + value, st_offset, s_len});
            std::cout << value << "  " << st_offset << " " << s_len << "\n";
        }
    }
    ifs.close();

    // dummy dataset with intrinsics/extrinsics
    auto dataset = std::make_shared<Dataset>();
    if (!dataset->init_no_objects(dataset_folder, camera_name))
        return -1;

    ViconWandSpatialCalibrator calibrator(dataset);

    // convert events to images; detect blobs, filter and track
    for (auto &bag_name : bag_names) {
        auto tracks = extract_tracks(nh, std::get<0>(bag_name), camera_name,
                                     std::get<1>(bag_name),
                                     std::get<2>(bag_name), dilate_blobs, img_th_value);
        auto &p2d = tracks.first;
        auto &p3d = tracks.second;

        calibrator.add_tracks(p2d, p3d);
    }


    bool extr_only = false;
    bool plot = true;

    std::cout << "\n\nRough calibration:\n";
    auto E_corr = calibrator.calibrate(extr_only, false);

    std::cout << "\n\nFine calibration:\n";
    E_corr = calibrator.calibrate(extr_only, plot);
    calibrator.reset();


    return 0;

/*
    std::cout << _green("Time Offset Error: ") << time_offset << std::endl;
    for (size_t n_iter = 0; n_iter < 20; ++n_iter) {
        // compute spatial calibration
        E_corr = ViconWandSpatialCalibrator::calibrate(dataset, wand_detected_pix, wand_red_3d, true, true);

        // apply correction to wand red 3d and 2d points:
        for (auto &lbl_pair : wand_red_3d) {
            wand_red_pix[lbl_pair.first] = Track<std::tuple<float, float>>();
            lbl_pair.second.apply_tf(E_corr);
            lbl_pair.second.apply_time_offset(-time_offset);
            for (size_t i = 0; i < lbl_pair.second.size(); ++i) {
                auto &ps = lbl_pair.second[i];

                float u = -1, v = -1;
                if (std::get<2>(ps.pose) > 0.001) {
                    struct pXYZ {double x, y, z; };
                    dataset->project_point(pXYZ{std::get<0>(ps.pose), std::get<1>(ps.pose), std::get<2>(ps.pose)}, u, v);
                }

                if (u >= 0 && v >= 0) wand_red_pix[lbl_pair.first].push_back(ps.get_ts_sec(), {v, u});
            }
        }

        // align time
        time_offset = ViconWandTimeAligner::align(wand_detected_pix, wand_red_pix, to_step, to_wsize);
        std::cout << _green("Time Offset Error: ") << time_offset << std::endl;

        if (std::fabs(time_offset) < to_wsize / 4) {
            to_step /= 2;
            to_wsize /= 2;
        }

        if (std::fabs(time_offset) < 1e-6) break;
    }

    time_offset = ViconWandTimeAligner::align(wand_detected_pix, wand_red_pix, to_step / 10, to_wsize);
    std::cout << _green("\n\nTime Offset Error: ") << time_offset << std::endl;
    E_corr = ViconWandSpatialCalibrator::calibrate(dataset, wand_detected_pix, wand_red_3d, false, true);


    return 0;


*/
/*
    cv::namedWindow("img", cv::WINDOW_NORMAL);
    for (size_t i = 0; i < dataset->images.size(); ++i) {
        std::cout << i << "\n";
        auto img_ = dataset->images[i];
        auto ts = image_timestamps[i];
        cv::Mat img = img_.clone();

        auto keypoint_tuple = kpt_tracker[i];
        kpt_tracker.print_stats(i);
*/

        /*
        auto wand_line_idx = wand::find_all_3lines(keypoint_pair.first, 0.2);
        auto wand_idx = wand::detect_wand_internal_idx(keypoint_pair.first, wand_line_idx, wand::wand_red_mapping, 0.5, 0.5, 0.5);
        img = wand::draw_wand(img, keypoint_pair.first, wand_idx);
        */


        //cv::drawKeypoints(img, std::get<0>(keypoint_tuple), img, cv::Scalar(255,0,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


        /*
        for (auto &kpt_tj : wand_ir_pix) {
            auto kpt = TimeSlice(kpt_tj.second, {ts - 1e-3, ts + 1e-3}, {0, 0}).begin()->pose;
            cv::circle(img, {std::get<0>(kpt), std::get<1>(kpt)}, 1, {255, 0, 0}, -1);
        }

        for (auto &kpt_tj : wand_red_pix) {
            auto kpt = TimeSlice(kpt_tj.second, {ts - 1e-3, ts + 1e-3}, {0, 0}).begin()->pose;
            std::cout << "\t" << std::get<0>(kpt) << " " <<  std::get<1>(kpt) << "\n";
            cv::circle(img, {std::get<0>(kpt), std::get<1>(kpt)}, 1, {0, 0, 255}, -1);
        }
        */

/*
        if (dataset->res_x > 1000)
            cv::resize(img, img, cv::Size(), 0.3, 0.3);

        cv::imshow("img", img);
        auto code = cv::waitKey(0);
        if (code == 27) break;
    }
*/



    std::cout << _green("Done!") << std::endl;
    return 0;
};
