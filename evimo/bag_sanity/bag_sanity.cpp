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


#include <common.h>
#include <event.h>

// DVS / DAVIS
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>



int main (int argc, char** argv) {

    // Initialize ROS
    std::string node_name = "bag_samity";
    ros::init (argc, argv, node_name, ros::init_options::AnonymousName);
    ros::NodeHandle nh("~");

    std::string bag_name = "";
    if (!nh.getParam("bag", bag_name)) {
        std::cerr << "No .bag file specified!" << std::endl;
        return -1;
    }

    std::cout << _blue("Procesing bag file: ") << bag_name << std::endl;

    rosbag::Bag bag;
    bag.open(bag_name, rosbag::bagmode::Read);
    rosbag::View view(bag);
    std::vector<const rosbag::ConnectionInfo *> connection_infos = view.getConnections();

    std::cout << std::endl << "Topics available:" << std::endl;
    for (auto &info : connection_infos) {
        std::cout << "\t" << info->topic << std::endl;
    }

    std::string topic_name = "";
    if (!nh.getParam("topic", topic_name)) {
        std::cerr << "No topic name specified!" << std::endl;
        return -1;
    }

    std::cout << std::endl << "Using topic:" << topic_name << std::endl;
    

    ros::Time bag_start_ts = view.begin()->getTime();
    uint64_t n_events = 0;
    for (auto &m : view) {
        if (m.getTopic() != topic_name) continue;
        auto msg = m.instantiate<dvs_msgs::EventArray>();
        if (msg == NULL) continue;
        n_events += msg->events.size();
    }


    std::vector<Event> event_array;
    event_array.reserve(n_events);
    uint64_t critical_gap_ms = 10;

    uint64_t id = 0;
    ros::Time first_event_ts = ros::Time(0);
    ros::Time first_event_message_ts = ros::Time(0);
    ros::Time last_event_ts = ros::Time(0);
    for (auto &m : view) {
        if (m.getTopic() != topic_name) continue;
        auto msg = m.instantiate<dvs_msgs::EventArray>();
        if (msg == NULL) continue;
        auto msize = msg->events.size();

        for (uint64_t i = 0; i < msize; ++i) {
            auto &e = msg->events[i];
            ros::Time current_event_ts = e.ts;
            int32_t x = e.x; 
            int32_t y = e.y;
            int polarity = e.polarity ? 1 : 0;

            if (id == 0) {
                first_event_ts = current_event_ts;
                last_event_ts = current_event_ts;
                first_event_message_ts = msg->header.stamp;// m.getTime();
            } else {
                if (current_event_ts < last_event_ts) {
                    if (i == 0) std::cout << std::endl << "packet boundary" << std::endl;
                    std::cout << msg->header.stamp << "\t" << m.getTime() << "\t"
                              << _red("Events are not sorted! ")
                              << id << ": " << last_event_ts - first_event_ts << " -> "
                              << current_event_ts - first_event_ts << "\t("
                              << i + 1 << " / " << msize << ")" << std::endl;
                } else if ((current_event_ts - last_event_ts).toNSec() > critical_gap_ms * 1e6) {
                    if (i == 0) std::cout << std::endl << "packet boundary" << std::endl;
                    std::cout << msg->header.stamp << "\t" << m.getTime() << "\t"
                              << _yellow("Events have a gap! ")
                              << id << ": " << last_event_ts - first_event_ts << " -> "
                              << current_event_ts - first_event_ts << "\t("
                              << i + 1 << " / " << msize << ")" << std::endl;
                }
                last_event_ts = current_event_ts;
            }

            auto ts = (current_event_ts - first_event_ts).toNSec();
            event_array.push_back(Event(y, x, ts, polarity));
            id ++;
        }
    }

    n_events = event_array.size();
    std::cout << _green("Read ") << n_events << _green(" events") << std::endl;

    std::cout << _green("Done!") << std::endl;
    return 0;
}
