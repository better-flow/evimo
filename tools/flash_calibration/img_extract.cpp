#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
#include <ctime>
#include <new>
#include <cassert>
#include <unistd.h>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <ros/package.h>

// DVS / DAVIS (all messages will be cast to this)
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>


typedef long int lint;
typedef long long int llint;
typedef unsigned int uint;
typedef unsigned long int ulong;
typedef unsigned long long int ull;



// Input parameters
int resolution_x = 240;
int resolution_y = 180;

bool verbose = false;

std::vector<uint> x_arr;
std::vector<uint> y_arr;
std::vector<ull>  t_arr;
std::vector<bool> p_arr;
std::vector<ull>  idx;
ull discretization = 1000000;

// Range of events
ull min_actual_time = 0;
ull max_actual_time = 0;

// Event slice we are looking at (in ms * 10)
ull min_slice_time = 0; // 0.0 sec
ull width_slice_time = 1000; // 100 ms


bool read_events (std::string fname, std::string tname, double llimit, double hlimit) {
    std::cout << "Reading from file... (" << fname << ")"
              << " topic: " << tname
              << std::endl << std::flush;

    x_arr.clear();
    y_arr.clear();
    t_arr.clear();
    p_arr.clear();
    idx.clear();
    min_actual_time = 0;
    max_actual_time = 0;
    min_slice_time = 0;

    rosbag::Bag bag;
    bag.open(fname, rosbag::bagmode::Read);
    rosbag::View view(bag);
    std::vector<const rosbag::ConnectionInfo *> connection_infos = view.getConnections();

    bool topic_found = false;
    for (auto &info : connection_infos) {
        if (tname == info->topic) topic_found = true;
    }

    if (!topic_found) {
        std::cout << std::endl << "Specified topic not found!" << std::endl;
        return false;
    }

    ull cnt = 0;
    ull total_cnt = 0;
    clock_t begin = std::clock();

    ros::Time first_event_ts;
    ros::Time cutoff_lo;
    ros::Time cutoff_hi;

    if (llimit < 0) llimit = 0;
    for (auto &m : view) {
        if (m.getTopic() == tname) {
            auto msg = m.instantiate<dvs_msgs::EventArray>();
            if (msg != NULL) {
                if (total_cnt == 0) {
                    first_event_ts = msg->events[0].ts;
                    cutoff_lo = first_event_ts + ros::Duration(llimit);
                    cutoff_hi = first_event_ts + ros::Duration(hlimit);
                }
                total_cnt += msg->events.size();

                if (msg->events[0].ts < cutoff_lo)
                    continue;

                if (hlimit > 0 && msg->events[msg->events.size() - 1].ts > cutoff_hi)
                    break;

                resolution_x = msg->height;
                resolution_y = msg->width;
                cnt += msg->events.size();
            }
        }
    }

    std::cout << "Found " << cnt << " events" << std::endl;
    if (cnt == 0) {
        return false;
    }

    x_arr.reserve(cnt * 1.2);
    y_arr.reserve(cnt * 1.2);
    t_arr.reserve(cnt * 1.2);
    p_arr.reserve(cnt * 1.2);

    for (auto &m : view) {
        if (m.getTopic() == tname) {
            auto msg = m.instantiate<dvs_msgs::EventArray>();
            if (msg != NULL) {
                if (msg->events[0].ts < cutoff_lo)
                    continue;

                if (hlimit > 0 && msg->events[msg->events.size() - 1].ts > cutoff_hi)
                    break;

                for (auto &e : msg->events) {
                    x_arr.push_back(e.y);
                    y_arr.push_back(e.x);
                    t_arr.push_back((e.ts - cutoff_lo).toSec() * 1000000000.0);
                    p_arr.push_back(e.polarity ? 1 : 0);
                }
            }
        }
    }

    std::cout << "Populating the lookup table..." << std::endl;
    idx.reserve((t_arr[t_arr.size() - 1] - t_arr[0]) / discretization * 1.2);
    for (ull i = 0; i < t_arr.size(); ++i) {
        auto ts = t_arr[i];
        auto target = discretization * (idx.size() + 1);
        if (ts >= target) idx.push_back(i);
    }

    clock_t end = std::clock();

    min_actual_time = t_arr[0];
    max_actual_time = t_arr[t_arr.size() - 1];

    std::cout << "Read " << cnt << " events, finished" << std::endl << std::flush;
    std::cout << "Elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl << std::flush;
    std::cout << "Time diff: " << t_arr[t_arr.size() - 1] - t_arr[0] << std::endl << std::flush;
    std::cout << "Time diff: " << (long double)(t_arr[t_arr.size() - 1] - t_arr[0]) / 1000000000.0
              << " sec." << std::endl << std::endl << std::flush;

    bag.close();
    return true;
}

// Project grayscale
cv::Mat project_events (double nx = 0, double ny = 0, double nz = 1) {
    int scale = 1;

    cv::Mat project_img = cv::Mat::zeros(resolution_x * scale, resolution_y * scale, CV_8UC1);
    if (nz == 0) return project_img;

    double t_divider = 1;
    double xy_len = hypot(nx, ny);
    double speed = xy_len / (nz / (1000000000/(t_divider * 10000)));
    double u = (xy_len == 0) ? 0 : speed * nx / xy_len;
    double v = (xy_len == 0) ? 0 : speed * ny / xy_len;

    // Choose slice events:
    ull i = (min_slice_time / discretization == 0) ? 0 : idx[min_slice_time / discretization - 1];
    for (; i < x_arr.size(); ++i) {
        if (min_slice_time < t_arr[i])
            break;
    }

    clock_t begin = std::clock();
    double kx = nx / nz;
    double ky = ny / nz;

    for (; i < x_arr.size(); ++i) {
        if (min_slice_time + width_slice_time < t_arr[i])
            break;

        int x = scale * (x_arr[i] - double((t_arr[i] - min_slice_time) / t_divider) / 10000 * kx);
        int y = scale * (y_arr[i] - double((t_arr[i] - min_slice_time) / t_divider) / 10000 * ky);

        if ((x >= scale * resolution_x) || (x < 0) || (y >= scale * resolution_y) || (y < 0))
            continue;

        for (int jx = x; jx < x + scale; ++jx) {
            for (int jy = y; jy < y + scale; ++jy) {
                if (project_img.at<uchar>(jx, jy) < 255)
                    project_img.at<uchar>(jx, jy) ++;
            }
        }
    }

    if (scale > 1) {
        int k_size = (scale % 2 == 0) ? scale + 1 : scale;
        cv::GaussianBlur(project_img, project_img, cv::Size(k_size, k_size), 0, 0);
    }

    clock_t end = std::clock();

    if (verbose)
        std::cout << "\t Elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec.\n";

    return project_img;
}


double nonzero_average (cv::Mat img) {
    // Average of nonzero
    double nz_avg = 0;
    long int nz_avg_cnt = 0;
    uchar* p = img.data;
    for(int i = 0; i < img.rows * img.cols; ++i, p++) {
        if (*p == 0) continue;
        nz_avg_cnt ++;
        nz_avg += *p;
    }
    nz_avg = (nz_avg_cnt == 0) ? 0 : nz_avg / double(nz_avg_cnt);
    return nz_avg;
}


std::vector<ull> get_ts_array(float wsize = 0.5) {
    std::vector<ull> ret;

    ull nbins = ull(wsize * 1000000000) / discretization;
    for (ull i = nbins / 2; i < idx.size() - nbins / 2; ++i) {
        ull ecount = idx[i] - idx[i - 1];

        ull total = 0;
        for (ull j = i - nbins / 2; j < i + nbins / 2; ++j) {
            total += idx[j] - idx[j - 1];
        }

        if (ecount * nbins > total * 8)
            ret.push_back(t_arr[idx[i]]);
    }

    return ret;
}


void save_images(std::string folder, std::vector<ull> ts_arr, float toffset = 0.0) {
    ull i = 0;
    for (auto &ts : ts_arr) {
        i += 1;

        min_slice_time = llint(ts - width_slice_time / 2) + llint(toffset * 1000000000.0);
        auto img = project_events();

        double img_scale = 127.0 / nonzero_average(img);
        cv::convertScaleAbs(img, img, img_scale, 0);
        cv::imwrite(folder + "/00" + std::to_string(ts) + ".png", img);

        if (i % 10 == 0)
            std::cout << "Saving " << i << " / " << ts_arr.size() << "\t\t\t\r";
    }
    std::cout << "\n";
}


float erate_cross_correlation(float max_offset, std::vector<ull> base_idx) {
    long int nbins = ull(max_offset * 1000000000) / discretization;
    ull max_score = 0;
    long int best_offst = 0;

    for (long int offst = -nbins; offst < nbins; ++offst) {
        ull score = 0;
        for (ull i = nbins + 1; i < std::min(idx.size(), base_idx.size()) - nbins; ++i) {
            score += (idx[i + offst] - idx[i + offst - 1]) * (base_idx[i] - base_idx[i - 1]);
        }

        if (score > max_score) {
            max_score = score;
            best_offst = offst;
        }
    }

    return double(best_offst) * double(discretization) / 1000000000.0;
}


int main(int argc, char *argv[]) {
    if (argc <= 2 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <bag file name> <out_folder> <start_time> <end_time>\n";
        return 1;
    }

    std::string bag_name = argv[1];
    std::string out_folder = argv[2];

    float start_time = -1;
    float end_time = -1;
    if (argc > 3)
        start_time = atof(argv[3]);

    if (argc > 4)
        end_time = atof(argv[4]);

    if (start_time >= 0 && end_time >= 0 && end_time < start_time)
        std::swap(start_time, end_time);

    width_slice_time = 0.01 * 1000000000;

    std::vector<std::pair<std::string, std::string>> topic_db {
        {"/prophesee/left/cd_events_buffer", "cam_2"},
        {"/prophesee/right/cd_events_buffer", "cam_1"},
        {"/prophesee/hvga/cd_events_buffer", "cam_0"}
    };
    std::vector<std::pair<std::string, std::string>> available_topics;


    std::vector<const rosbag::ConnectionInfo *> connection_infos;

    rosbag::Bag bag;
    bag.open(bag_name, rosbag::bagmode::Read);
    rosbag::View view(bag);
    connection_infos = view.getConnections();

    std::cout << std::endl << "Topics available:" << std::endl;
    for (auto &info : connection_infos) {
        std::cout << "\t" << info->topic << std::endl;
    }

    for (auto &topic : topic_db) {
        for (auto &info : connection_infos) {
            if (info->topic == topic.first) {
                available_topics.push_back(topic);
                break;
            }
        }
    }

    bag.close();

    std::vector<ull> ts_arr;
    std::vector<ull> base_idx;

    float toffset = 0.0;
    int i = 0;
    for (auto &topic : available_topics) {
        if (!read_events(bag_name, topic.first, start_time, end_time))
            continue;
        if (i == 0) {
            ts_arr = get_ts_array(0.5);
            base_idx = idx;
        } else {
            toffset = erate_cross_correlation(0.5, base_idx);
            std::cout << "Offset (" << topic.second << "): " << toffset << "\n";
        }

        auto folder = out_folder + "/" + topic.second;
        mkdir(folder.c_str(), 0777);
        save_images(folder, ts_arr, toffset);

        i ++;
    }

    return 0;
}
