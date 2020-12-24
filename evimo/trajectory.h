#include <valarray>
#include <common.h>

#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/Imu.h>

// VICON
#include <vicon/Subject.h>
#include <object.h>

#ifndef TRAJECTORY_H
#define TRAJECTORY_H


template <class T> class Slice {
protected:
    T *data;
    std::pair<size_t, size_t> indices;
    size_t current_size;

protected:
    Slice(T &vec)
        : data(&vec), indices(0, 0), current_size(0) {}

    void set_indices(std::pair<size_t, size_t> p) {
        this->indices = p;
        this->current_size = p.second - p.first + 1;

        if (p.first > p.second)
            throw std::runtime_error("Attempt to create a Slice with first index bigger than the second! (" +
                              std::to_string(p.first) + " > " + std::to_string(p.second) + ")");
        if (p.second >= this->data->size())
            throw std::runtime_error("the second index in Slice is bigger than input vector size! (" +
                              std::to_string(p.second) + " >= " + std::to_string(this->data->size()) + ")");
    }

public:
    typedef T value_type;

    Slice(T &vec, std::pair<uint64_t, uint64_t> p)
        : Slice(vec) {
        this->set_indices(p);
    }

    std::pair<size_t, size_t> get_indices() {
        return this->indices;
    }

    size_t size() {return this->current_size; }
    auto begin()  {return this->data->begin() + this->indices.first; }
    auto end()    {return this->data->begin() + this->indices.second + 1; }
    auto operator [] (size_t idx) {return (*this->data)[idx + this->indices.first]; }
};


template <class T> class TimeSlice : public Slice<T> {
protected:
    std::pair<double, double> time_bounds;
    double get_ts(size_t idx) const {return (this->data->begin() + idx)->get_ts_sec(); }

public:
    TimeSlice(T &vec)
        : Slice<T>(vec) {
        if (this->data->size() == 0)
            throw std::runtime_error("TimeSlice: cannot construct on an empty container!");

        this->time_bounds.first  = this->get_ts(0);
        this->time_bounds.second = this->get_ts(this->data->size() - 1);
        this->set_indices(std::pair<uint64_t, uint64_t>(0, this->data->size() - 1));
    }

    TimeSlice(T &vec, std::pair<double, double> p, std::pair<size_t, size_t> hint)
        : Slice<T>(vec), time_bounds(p) {
        std::pair<uint64_t, uint64_t> idx_pair;
        idx_pair.first  = this->find_nearest(this->time_bounds.first,  hint.first);
        idx_pair.second = this->find_nearest(this->time_bounds.second, hint.second);
        if (idx_pair.second < idx_pair.first) {
            idx_pair.first = idx_pair.second;
        }
        this->set_indices(idx_pair);
    }

    TimeSlice(T &vec, std::pair<double, double> p, size_t hint = 0)
        : TimeSlice(vec, p, std::make_pair(hint, hint)) {}

    size_t find_nearest(double ts, size_t hint = 0) const {
        // Assuming data is sorted according to timestamps, in ascending order
        if (this->data->size() == 0)
            throw std::runtime_error("find_nearest: data container is empty!");

        if (hint >= this->data->size()) {
            hint = this->data->size() - 1;
            std::cerr << "find_nearest: hint specified is out of bounds!" << std::endl;
        }

        size_t best_idx = hint;
        auto initial_ts = this->get_ts(best_idx);
        double best_error = std::fabs(initial_ts - ts);

        int8_t step = 1;
        if (ts < initial_ts) step = -1;

        int32_t idx = hint;
        while (idx >= 0 && idx < this->data->size() && step * (ts - this->get_ts(idx)) >= 0.0) {
            if (std::fabs(ts - this->get_ts(idx)) < best_error) {
                best_error = std::fabs(ts - this->get_ts(idx));
                best_idx = idx;
            }

            idx += step;
        }

        if (idx >= 0 && idx < this->data->size() && std::fabs(ts - this->get_ts(idx)) < best_error) {
            best_error = std::fabs(ts - this->get_ts(idx));
            best_idx = idx;
        }

        return best_idx;
    }

    std::pair<double, double> get_time_bounds() {
        return this->time_bounds;
    }
};


class Pose : public SensorMeasurement {
public:
    ros::Time ts;
    tf::Transform pq;
    float occlusion;
    std::vector<std::tuple<float, float, float, std::string>> markers;

    Pose()
        : ts(0), occlusion(std::numeric_limits<double>::quiet_NaN()) {this->pq.setIdentity(); }
    Pose(ros::Time ts_, tf::Transform pq_)
        : ts(ts_), pq(pq_), occlusion(std::numeric_limits<double>::quiet_NaN()) {}
    Pose(ros::Time ts_, const vicon::Subject& p)
        : ts(ts_), occlusion(0) {
        if (p.markers.size() == 0) {
            this->occlusion = 1;
            return;
        }

        this->markers.reserve(p.markers.size());
        this->pq = ViObject::subject2tf(p);
        for (auto &marker : p.markers) {
            if (marker.occluded) this->occlusion ++;
            this->markers.emplace_back(marker.position.x,
                                       marker.position.y,
                                       marker.position.z,
                                       marker.name);
        }
        this->occlusion = this->occlusion / float(p.markers.size());
    }

    void setT(std::valarray<float> t) {
        tf::Vector3 T(t[0], t[1], t[2]);
        this->pq.setOrigin(T);
    }

    void setR(std::valarray<float> r) {
        tf::Quaternion q;
        q.setRPY(r[0], r[1], r[2]);
        this->pq.setRotation(q);
    }

    std::valarray<float> getT() {
        tf::Vector3 T = this->pq.getOrigin();
        return {(float)T.getX(), (float)T.getY(), (float)T.getZ()};
    }

    std::valarray<float> getR() {
        tf::Quaternion q = this->pq.getRotation();
        float w = q.getW(), x = q.getX(), y = q.getY(), z = q.getZ();
        float X = std::atan2(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y));
        float sin_val = 2.0f * (w * y - z * x);
        sin_val = (sin_val >  1.0f) ?  1.0f : sin_val;
        sin_val = (sin_val < -1.0f) ? -1.0f : sin_val;
        float Y = std::asin(sin_val);
        float Z = std::atan2(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
        return {X, Y, Z};
    }

    double get_ts_sec() {return this->ts.toSec(); }

    operator tf::Transform() const {return this->pq; }

    Pose operator-(const Pose &p) {
        Pose ret(this->ts, this->pq);
        ret.occlusion = std::max(this->occlusion, p.occlusion);
        auto inv_tf = p.pq.inverse();
        ret.pq = inv_tf * this->pq;
        ret.markers.reserve(this->markers.size());
        for (auto &m : this->markers) {
            auto new_p = inv_tf({std::get<0>(m), std::get<1>(m), std::get<2>(m)});
            ret.markers.emplace_back(new_p.x(), new_p.y(), new_p.z(), std::get<3>(m));
        }

        return ret;
    }

    Pose operator*(const float &s) {
        auto t = this->getT();
        auto r = this->getR();

        Pose ret(this->ts, this->pq);
        ret.occlusion = this->occlusion;

        ret.setT(t * s);
        ret.setR(r * s);

        return ret;
    }

    std::string as_dict() {
        auto T = this->getT();
        auto RPY = this->getR();
        auto Q = this->pq.getRotation();

        std::string ret = "{";
        ret += "'t': {'x': " + std::to_string(T[0])
            + ", 'y': " + std::to_string(T[1])
            + ", 'z': " + std::to_string(T[2]) + "}";
        ret += ", 'rpy': {'r': " + std::to_string(RPY[0])
            + ", 'p': " + std::to_string(RPY[1])
            + ", 'y': " + std::to_string(RPY[2]) + "}";
        ret += ", 'q': {'w': " + std::to_string(Q.getW())
            + ", 'x': " + std::to_string(Q.getX())
            + ", 'y': " + std::to_string(Q.getY())
            + ", 'z': " + std::to_string(Q.getZ()) + "}}";
        return ret;
    }

    friend std::ostream &operator<< (std::ostream &output, const Pose &P) {
        auto loc = P.pq.getOrigin();
        auto rot = P.pq.getRotation();
        output << loc.getX() << " " << loc.getY() << " " << loc.getZ() << " "
               << rot.getW() << " " << rot.getX() << " " << rot.getY() << " "
               << rot.getZ();
        return output;
    }
};


/*! Trajectory class */
class Trajectory {
protected:
    double filtering_window_size; /**< size of trajectory filtering window, in seconds */

private:
    std::vector<Pose> poses; /**< array of poses */

public:
    Trajectory(int32_t window_size = -1)
        : filtering_window_size(window_size) {}

    void set_filtering_window_size(double window_size) {this->filtering_window_size = window_size; }
    auto get_filtering_window_size() {return this->filtering_window_size; }

    template<class T> void add(ros::Time ts_, T pq_) {
        if (this->poses.size() > 0 && this->poses.back().ts > ts_) {
            std::cout << _yellow("Pose is added out of order:") << "last added ts: "
                      << this->poses.back().ts << "; new ts: " << ts_ << std::endl;
        }
        this->poses.push_back(Pose(ts_, pq_));
    }

    bool is_sorted(bool verbose=false) {
        bool sorted = true;
        for (size_t i = 1; i < this->poses.size(); ++i) {
            if (this->poses[i - 1].ts < this->poses[i].ts) continue;
            sorted = false;
            if (!verbose) break;
            std::cout << _yellow("Poses are not sorted! ")
                      << i << ": " << this->poses[i - 1].ts << " -> "
                      << this->poses[i].ts << std::endl;
        }
        return sorted;
    }

    void clear() {
        this->poses.clear();
    }

    size_t size() {return this->poses.size(); }
    auto operator [] (size_t idx) {return this->get_filtered(idx); }

    virtual bool check() {
        if (this->size() == 0) return true;
        auto prev_ts = this->poses[0].ts;
        for (auto &p : this->poses) {
            if (p.ts < prev_ts) return false;
            prev_ts = p.ts;
        }
        return true;
    }

    virtual void subtract_time(ros::Time t) final {
        while ((this->poses.size() > 0) && (this->poses.begin()->ts < t))
            this->poses.erase(this->poses.begin());
        for (auto &p : this->poses) p.ts = ros::Time((p.ts - t).toSec());
    }

    Pose get_velocity(size_t idx) {
        if (idx >= this->poses.size()) {
            std::cerr << "get_velocity: index out of range!\n";
            std::terminate();
        }

        auto p0 = idx >= this->size() - 1 ? (*this)[idx] : (*this)[idx + 1];
        auto p1 = idx == 0 ? (*this)[idx] : (*this)[idx - 1];
        auto dt = (p0.ts > p1.ts ? p0.ts - p1.ts : p1.ts - p0.ts).toSec();
        dt *= (p0.ts > p1.ts) ? 1.0 : -1.0;
        auto v = (p0 - p1) * (1.0 / static_cast<float>(dt));
        return v;
    }

protected:
    auto begin() {return this->poses.begin(); }
    auto end()   {return this->poses.end(); }

    virtual Pose get_filtered(size_t idx) {
        if (this->filtering_window_size <= 0) {
            return this->poses[idx];
        }

        auto central_ts = this->poses[idx].get_ts_sec();
        auto poses_in_window = TimeSlice<Trajectory>(*this,
             std::make_pair(central_ts - this->filtering_window_size / 2.0,
                            central_ts + this->filtering_window_size / 2.0), idx);
        Pose filtered_p;
        filtered_p.ts = ros::Time(central_ts);
        filtered_p.occlusion = this->poses[idx].occlusion;
        auto rot = filtered_p.getR();
        auto tr  = filtered_p.getT();

        for (auto &p : poses_in_window) {
            rot += p.getR();
            tr  += p.getT();
        }

        rot /= float(poses_in_window.size());
        tr  /= float(poses_in_window.size());

        filtered_p.setR(rot);
        filtered_p.setT(tr);

        //if (poses_in_window.size() > 1)
        //    std::cout << "Filtering pose with\t" << poses_in_window.size() << "\tneighbours\n";
        return filtered_p;
    }

    friend class Slice<Trajectory>;
    friend class TimeSlice<Trajectory>;
};


/*! Imu info */
class ImuMeasurement : public SensorMeasurement {
public:
    ros::Time ts;
    std::tuple<float, float, float> angular_velocity;
    std::tuple<float, float, float> linear_acceleration;

    ImuMeasurement() : ts(0) {}
    ImuMeasurement(ros::Time ts_, const sensor_msgs::Imu& p) : ts(ts_) {
        this->angular_velocity = {p.angular_velocity.x, p.angular_velocity.y, p.angular_velocity.z};
        this->linear_acceleration = {p.linear_acceleration.x, p.linear_acceleration.y, p.linear_acceleration.z};
    }

    double get_ts_sec() {return this->ts.toSec(); }

    std::string as_dict() {
        std::string ret = "{";
        ret += "'ts': " + std::to_string(this->get_ts_sec()) + ",\n";
        ret += "'angular_velocity': {'x': " + std::to_string(std::get<0>(this->angular_velocity))
            + ", 'y': " + std::to_string(std::get<1>(this->angular_velocity))
            + ", 'z': " + std::to_string(std::get<2>(this->angular_velocity)) + "}, ";
        ret += "'linear_acceleration': {'x': " + std::to_string(std::get<0>(this->linear_acceleration))
            + ", 'y': " + std::to_string(std::get<1>(this->linear_acceleration))
            + ", 'z': " + std::to_string(std::get<2>(this->linear_acceleration)) + "}}";
        return ret;
    }

    friend std::ostream &operator<< (std::ostream &output, const ImuMeasurement &P) {
        output << std::get<0>(P.angular_velocity) << " "
               << std::get<1>(P.angular_velocity) << " "
               << std::get<2>(P.angular_velocity) << " "
               << std::get<0>(P.linear_acceleration) << " "
               << std::get<1>(P.linear_acceleration) << " "
               << std::get<2>(P.linear_acceleration);
        return output;
    }
};


class ImuInfo {
public:
    std::vector<ImuMeasurement> imu_mes;
    ImuInfo() {}

    template<class T> void add(ros::Time ts_, T pq_) {
        if (this->imu_mes.size() > 0 && this->imu_mes.back().ts > ts_) {
            std::cout << _yellow("IMU message added is out of order:") << "last added ts: "
                      << this->imu_mes.back().ts << "; new ts: " << ts_ << std::endl;
        }
        this->imu_mes.push_back(ImuMeasurement(ts_, pq_));
    }

    void clear() {this->imu_mes.clear(); }
    size_t size() {return this->imu_mes.size(); }
    auto operator [] (size_t idx) {return this->imu_mes[idx]; }

    virtual bool check() {
        if (this->size() == 0) return true;
        auto prev_ts = this->imu_mes[0].ts;
        for (auto &p : this->imu_mes) {
            if (p.ts < prev_ts) return false;
            prev_ts = p.ts;
        }
        return true;
    }

    virtual void subtract_time(ros::Time t) final {
        while ((this->imu_mes.size() > 0) && (this->imu_mes.begin()->ts < t))
            this->imu_mes.erase(this->imu_mes.begin());
        for (auto &p : this->imu_mes) p.ts = ros::Time((p.ts - t).toSec());
    }

protected:
    auto begin() {return this->imu_mes.begin(); }
    auto end()   {return this->imu_mes.end(); }

    friend class Slice<ImuInfo>;
    friend class TimeSlice<ImuInfo>;
};


#endif // TRAJECTORY_H
