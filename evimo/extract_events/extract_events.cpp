#include <cstdint>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

// DVS / DAVIS
// Use for samsung as well because messages are identical
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

// cnpy (write npy files)
#include <cnpy/cnpy.h>

void delete_if_exists(std::string name) {
    if (auto file = std::fopen(name.c_str(), "r")) {
        std::fclose(file);
        std::remove(name.c_str());
    }
}

int main (int argc, char** argv) {
    if (argc < 6+1) {
        throw std::invalid_argument("Need 6 arguments");
    }

    std::string bag_file(argv[1]); // Example: "/home/levi/EVIMO/imo/train/scene10_dyn_train_00/scene10_dyn_train_00.bag"
    std::string topic(argv[2]);    // Example: "/samsung/camera/events"
    std::filesystem::path output_folder(argv[3]); // Example: "/home/levi/EVIMO/output/"

    uint32_t sec  = std::stoi(argv[4]); // Example: "1234123412"
    uint32_t nsec = std::stoi(argv[5]); // Example: "1234123412"
    ros::Time t_0(sec, nsec); // t_0 is subtracted from the events time

    double duration = std::stod(argv[6]); // 1.2345

    std::vector<std::string> topics;
    topics.push_back(topic);

    rosbag::Bag bag;
    bag.open(bag_file, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topics), t_0, t_0 + ros::Duration(duration));

    auto t_file  = output_folder / "dataset_events_t.npy";
    auto xy_file = output_folder / "dataset_events_xy.npy";
    auto p_file  = output_folder / "dataset_events_p.npy";

    delete_if_exists(t_file);
    delete_if_exists(xy_file);
    delete_if_exists(p_file);

    uint64_t n_events = 0;
    for (auto &m : view) {
        auto msg = m.instantiate<dvs_msgs::EventArray>();

        auto msize = msg->events.size();
        n_events += msize;

        std::vector<float> events_t;
        events_t.reserve(msize);

        std::vector<uint16_t> events_xy;
        events_xy.reserve(msize*2);

        std::vector<uint8_t> events_p;
        events_p.reserve(msize);

        for (uint64_t i = 0; i < msize; ++i) {
            auto &e = msg->events[i];
            auto t = (e.ts - t_0).toSec();
            if (t >= 0 && t <= duration) {
                events_t.push_back(t);
                events_xy.push_back(e.x);
                events_xy.push_back(e.y);
                events_p.push_back(e.polarity ? 1 : 0);
            }
        }

        cnpy::npy_save(t_file,  &events_t [0], {events_t .size()},  "a");
        cnpy::npy_save(xy_file, &events_xy[0], {events_xy.size() / 2, 2},  "a");
        cnpy::npy_save(p_file,  &events_p [0], {events_p .size()},  "a");
    }
    // std::cout << "N: " << n_events << std::endl;

    return 0;
}
