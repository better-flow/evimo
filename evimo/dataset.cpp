#include <dataset.h>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <ros/package.h>


uint32_t Dataset::instance_id = 0;


bool Dataset::read_bag_file(std::string bag_name,
                            float start_time_offset, float sequence_duration,
                            bool with_images, bool ignore_tj) {
    std::cout << _blue("Procesing bag file: ") << bag_name << std::endl;

    rosbag::Bag bag;
    bag.open(bag_name, rosbag::bagmode::Read);
    rosbag::View view(bag);
    std::vector<const rosbag::ConnectionInfo *> connection_infos = view.getConnections();

    std::cout << std::endl << "Topics available:" << std::endl;
    for (auto &info : connection_infos) {
        std::cout << "\t" << info->topic << "\t" << info->datatype << std::endl;
        if (info->datatype == "sensor_msgs/Imu") this->imu_info[info->topic] = ImuInfo();
    }

    std::map<int, vicon::Subject> obj_cloud_to_vicon_tf;
    for (auto &p : this->obj_pose_topics) {
        this->obj_tjs[p.first] = Trajectory();
    }

    ros::Time bag_start_ts = view.begin()->getTime();
    uint64_t n_events = 0;
    for (auto &m : view) {
        if (m.getTopic() == this->cam_pos_topic) {
            auto msg = m.instantiate<vicon::Subject>();
            if (msg == NULL) continue;
            auto timestamp = msg->header.stamp + ros::Duration(this->get_time_offset_pose_to_host());

            if (start_time_offset > 0 && timestamp < bag_start_ts + ros::Duration(start_time_offset)) continue;
            if (sequence_duration > 0 && timestamp > bag_start_ts + ros::Duration(start_time_offset + sequence_duration)) continue;

            this->cam_tj.add(timestamp, *msg);
            continue;
        }

        for (auto &p : this->obj_pose_topics) {
            if (m.getTopic() != p.second) continue;
            auto msg = m.instantiate<vicon::Subject>();
            if (msg == NULL) break;
            if (msg->occluded) break;
            auto timestamp = msg->header.stamp + ros::Duration(this->get_time_offset_pose_to_host());

            if (start_time_offset > 0 && timestamp < bag_start_ts + ros::Duration(start_time_offset)) continue;
            if (sequence_duration > 0 && timestamp > bag_start_ts + ros::Duration(start_time_offset + sequence_duration)) continue;

            this->obj_tjs[p.first].add(timestamp, *msg);
            obj_cloud_to_vicon_tf[p.first] = *msg;
            break;
        }

        for (auto &p : this->imu_info) {
            if (m.getTopic() != p.first) continue;
            auto msg = m.instantiate<sensor_msgs::Imu>();
            if (msg == NULL) break;
            auto timestamp = msg->header.stamp;

            if (start_time_offset > 0 && timestamp < bag_start_ts + ros::Duration(start_time_offset)) continue;
            if (sequence_duration > 0 && timestamp > bag_start_ts + ros::Duration(start_time_offset + sequence_duration)) continue;

            this->imu_info[p.first].add(timestamp, *msg);
            break;
        }

        if (m.getTopic() == this->event_topic) {
            auto msg = m.instantiate<dvs_msgs::EventArray>();
            auto timestamp = msg->header.stamp;

            if (start_time_offset > 0 && timestamp < bag_start_ts + ros::Duration(start_time_offset)) continue;
            if (sequence_duration > 0 && timestamp > bag_start_ts + ros::Duration(start_time_offset + sequence_duration)) continue;

            if (msg != NULL) {
                n_events += msg->events.size();
            }
            continue;
        }

        if (with_images && (m.getTopic() == this->image_topic)) {
            auto msg_regular = m.instantiate<sensor_msgs::Image>();
            if (msg_regular != NULL) {
                auto timestamp = msg_regular->header.stamp + ros::Duration(this->get_time_offset_image_to_host());
                if (start_time_offset > 0 && timestamp < bag_start_ts + ros::Duration(start_time_offset)) continue;
                if (sequence_duration > 0 && timestamp > bag_start_ts + ros::Duration(start_time_offset + sequence_duration)) continue;

                this->images.push_back((cv_bridge::toCvShare(msg_regular, "bgr8")->image).clone());
                this->image_ts.push_back(timestamp);
                continue;
            }

            auto msg_compressed = m.instantiate<sensor_msgs::CompressedImage>();
            if (msg_compressed != NULL) {
                auto timestamp = msg_compressed->header.stamp + ros::Duration(this->get_time_offset_image_to_host());
                if (start_time_offset > 0 && timestamp < bag_start_ts + ros::Duration(start_time_offset)) continue;
                if (sequence_duration > 0 && timestamp > bag_start_ts + ros::Duration(start_time_offset + sequence_duration)) continue;

                this->images.push_back(cv::imdecode(cv::Mat(msg_compressed->data), 1).clone());
                this->image_ts.push_back(timestamp);
                continue;
            }
        }
    }

    {
    size_t n_invalid_objects = 0;
    for (auto &p : this->obj_pose_topics) {
        if (this->obj_tjs.find(p.first) != this->obj_tjs.end()) continue;
        std::cout << _red("Error: no valid poses for topic\t") << p.second << std::endl;
        n_invalid_objects ++;
    }
    if (n_invalid_objects > 0) {
        std::cout << _red("'objects.txt' should only contain tracked objects!") << std::endl;
        return false;
    }
    }

    if (with_images && this->images.size() == 0) {
        std::cout << _red("No images found! Reverting 'with_images' to 'false'") << std::endl;
        with_images = false;
    }

    this->event_array.reserve(n_events);

    uint64_t id = 0;
    ros::Time first_event_ts = ros::Time(0);
    ros::Time first_event_message_ts = ros::Time(0);
    ros::Time last_event_ts = ros::Time(0);
    bool sort_events = false;
    for (auto &m : view) {
        if (m.getTopic() != this->event_topic)
            continue;

        auto msize = 0;
        auto msg = m.instantiate<dvs_msgs::EventArray>();
        if (msg != NULL) msize = msg->events.size();

        auto timestamp = msg->header.stamp;

        if (start_time_offset > 0 && timestamp < bag_start_ts + ros::Duration(start_time_offset)) continue;
        if (sequence_duration > 0 && timestamp > bag_start_ts + ros::Duration(start_time_offset + sequence_duration)) continue;

        for (uint64_t i = 0; i < msize; ++i) {
            int32_t x = 0, y = 0;
            ros::Time current_event_ts = ros::Time(0);
            int polarity = 0;

            if (msg != NULL) {
                auto &e = msg->events[i];
                current_event_ts = e.ts;
                x = e.x; y = e.y;
                polarity = e.polarity ? 1 : 0;
            }

            if (id == 0) {
                first_event_ts = current_event_ts;
                last_event_ts = current_event_ts;
                first_event_message_ts = msg->header.stamp;// m.getTime();
            } else {
                if (current_event_ts < last_event_ts) {
                    sort_events = true;
                    std::cout << _yellow("Events are not sorted! ")
                              << id << ": " << last_event_ts << " -> "
                              << current_event_ts << std::endl;
                }
                last_event_ts = current_event_ts;
            }

            //auto ts = (first_event_message_ts + (current_event_ts - first_event_ts) +
            //           ros::Duration(this->get_time_offset_event_to_host())).toNSec();

            auto ts = (current_event_ts + ros::Duration(this->get_time_offset_event_to_host())).toNSec();

            this->event_array.push_back(Event(y, x, ts, polarity));
            id ++;
        }
    }
    n_events = this->event_array.size();

    // sort events
    if (sort_events) {
        std::cout << "Sorting events..." << std::endl;
        std::sort(this->event_array.begin(), this->event_array.end(),
            [](const Event &a, const Event &b) -> bool
            { return a.timestamp < b.timestamp; });
    }

    // check trajectories
    {
        std::cout << "\tCamera trajectory is... ";
        if (!this->cam_tj.is_sorted(true)) {
            std::cout << _red("not sorted") << std::endl;
            return false;
        } else {std::cout << _green("sorted") << std::endl; }
    }
    for (auto &obj_tj : this->obj_tjs) {
        std::cout << "\t" << obj_tj.first << " trajectory is... ";
        if (!obj_tj.second.is_sorted(true)) {
            std::cout << _red("not sorted") << std::endl;
            return false;
        } else {std::cout << _green("sorted") << std::endl; }
    }

    std::cout << _green("Read ") << n_events << _green(" events") << std::endl;
    if (ignore_tj) {
        auto t0 = ros::Time(0);
        t0.fromNSec(this->event_array[0].timestamp);
        auto t1 = ros::Time(0);
        t1.fromNSec(this->event_array[this->event_array.size() - 1].timestamp);
        auto npos = int((t1 - t0).toSec() * 100);
        std::cout << npos << "\t" << t0 << " - " << t1 << std::endl;

        for (size_t i = 0; i < npos; ++i)
            this->cam_tj.add(t0 + ros::Duration(float(i) / 100.0), tf::Transform());
    }

    std::cout << std::endl << _green("Read ") << this->cam_tj.size() << _green(" camera poses and ") << std::endl;
    for (auto &obj_tj : this->obj_tjs) {
        if (obj_tj.second.size() == 0) continue;
        std::cout << "\t" << obj_tj.second.size() << _blue(" poses for object ") << obj_tj.first << std::endl;
        if (!obj_tj.second.check()) {
            std::cout << "\t\t" << _red("Check failed!") << std::endl;
        }
        if (this->clouds.find(obj_tj.first) == this->clouds.end()) {
            std::cout << "\t\t" << _red("No pointcloud for trajectory! ") << "oid = " << obj_tj.first << std::endl;
            continue;
        }
        this->clouds[obj_tj.first]->convert_to_vicon_tf(obj_cloud_to_vicon_tf[obj_tj.first]);
    }

    // Force the first timestamp of the event cloud to be 0
    // trajectories
    auto time_offset = ros::Time(0);
    if (n_events == 0) {
        time_offset = this->cam_tj[0].ts;
    } else {
        //time_offset = first_event_message_ts + ros::Duration(this->get_time_offset_event_to_host());
        time_offset.fromNSec(this->event_array[0].timestamp);
        std::cout << "Event timestamp range (nsec):\t(" << this->event_array.front().timestamp 
                  << " - " << this->event_array.back().timestamp << ")" << std::endl;
    }

    if (this->cam_tj.size() == 0) {
        std::cout << _red("Camera trajectory size = 0!\n");
        return false;
    } else {
        std::cout << "Camera tj timestamp range:\t(" << this->cam_tj[0].ts
                  << " - " << this->cam_tj[this->cam_tj.size() - 1].ts << ")" << std::endl;
    }

    this->ros_time_offset = time_offset;
    this->cam_tj.subtract_time(time_offset);
    for (auto &obj_tj : this->obj_tjs) {
        if (obj_tj.second.size() == 0) continue;
        std::cout << obj_tj.first << " tj timestamp range:\t(" << obj_tj.second[0].ts
                  << " - " << obj_tj.second[obj_tj.second.size() - 1].ts << ")" << std::endl;
        obj_tj.second.subtract_time(time_offset);
    }

    for (auto &imu_inf : this->imu_info) {
        imu_inf.second.subtract_time(time_offset);
    }

    // images
    while(this->image_ts.size() > 0 && *(this->image_ts.begin()) < time_offset) {
        this->image_ts.erase(this->image_ts.begin());
        this->images.erase(this->images.begin());
    }

    for (uint64_t i = 0; i < this->image_ts.size(); ++i)
        this->image_ts[i] = ros::Time((this->image_ts[i] - time_offset).toSec() < 0 ? 0 : (this->image_ts[i] - time_offset).toSec());
    // events
    for (auto &e : this->event_array)
        e.timestamp -= time_offset.toNSec();

    std::cout << std::endl << "Removing time offset: " << _green(std::to_string(time_offset.toSec()))
              << std::endl << std::endl;

    if (n_events > 0)
        std::cout << "Event timestamp range:\t(" << double(this->event_array.front().timestamp) * 1e-9
                  << " - " << double(this->event_array.back().timestamp) * 1e-9 << ")" << std::endl;
    std::cout << "Camera tj timestamp range:\t(" << this->cam_tj[0].ts
              << " - " << this->cam_tj[this->cam_tj.size() - 1].ts << ")" << std::endl;
    auto cam_dt = this->cam_tj[this->cam_tj.size() - 1].ts - this->cam_tj[0].ts;
    for (auto &obj_tj : this->obj_tjs) {
        if (obj_tj.second.size() == 0) {
            std::cout << obj_tj.first << ":\t" << _yellow("No valid poses!\n");
            continue;
        }

        std::cout << obj_tj.first << " tj timestamp range:\t(" << obj_tj.second[0].ts
                  << " - " << obj_tj.second[obj_tj.second.size() - 1].ts << ")";
        auto dt = obj_tj.second[obj_tj.second.size() - 1].ts - obj_tj.second[0].ts;
        if (this->clouds.find(obj_tj.first) != this->clouds.end())
            std::cout << "\ttopic:\t" << this->clouds[obj_tj.first]->get_pose_topic();
        if (dt * 1.5 < cam_dt)
            std::cout << _yellow("\t<-- more than 50\% occluded!");
        std::cout << std::endl;
    }

    return true;
}
