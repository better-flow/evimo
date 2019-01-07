#ifndef RUNNING_AVERAGE_H
#define RUNNING_AVERAGE_H

#include <vector>
#include <vicon/Subject.h>

class RunningAverage final {
protected:
    std::vector<vicon::Subject> data;
    size_t max_size;
    size_t current_size;
    uint64_t total_inserted;

public:
    RunningAverage () : max_size(1), current_size(0), total_inserted(0) {
        this->resize(this->max_size);
    }

    RunningAverage (size_t sz) : max_size(sz), current_size(0), total_inserted(0) {
        this->resize(this->max_size);
    }

    inline void resize (size_t sz) {
        this->max_size = sz;
        this->data.resize(this->max_size);
    }

    inline size_t size () {
        return this->current_size;
    }

    inline void push_back (const vicon::Subject &d) {
        this->current_size += (this->current_size >= this->max_size) ? 0 : 1;
        this->data[this->total_inserted % this->max_size] = d;
        this->total_inserted ++;
    }

    inline vicon::Subject average () {
        vicon::Subject ret = this->data[this->total_inserted % this->max_size];
        float cnt = 0;
        float x = 0, y = 0, z = 0;
        float qx = 0, qy = 0, qz = 0, qw = 0;

        for (uint64 i = 0; i < this->current_size; ++i) {
            if (this->data[i].occluded) continue;

            x += this->data[i].position.x;
            y += this->data[i].position.y;
            z += this->data[i].position.z;
            
            qx += this->data[i].orientation.x;
            qy += this->data[i].orientation.y;
            qz += this->data[i].orientation.z;
            qw += this->data[i].orientation.w;
        
            cnt += 1.0;
        }

        if (cnt <= 0.1) return ret;

        x /= cnt; y /= cnt; z /= cnt;
        qx /= cnt; qy /= cnt; qz /= cnt; qw /= cnt;

        float norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw); 
        qx /= norm; qy /= norm; qz /= norm; qw /= norm;

        ret.position.x = x;
        ret.position.y = y;
        ret.position.z = z;
        ret.orientation.x = qx;
        ret.orientation.y = qy;
        ret.orientation.z = qz;
        ret.orientation.w = qw;
        return ret;
    }
};


class PoseManager {
protected:
    std::list<vicon::Subject> cam_poses;
    std::list<vicon::Subject> obj_poses;
    std::list<uint32_t> cam_checkpoints;
    std::list<uint32_t> obj_checkpoints;

    std::vector<vicon::Subject> cam_poses_smooth;
    std::vector<vicon::Subject> obj_poses_smooth;

    vicon::Subject last_cam_pos;
    vicon::Subject last_obj_pos;

    uint32_t counter;


public:
    PoseManager () : counter(0) {
        this->last_cam_pos.header.stamp = ros::Time(0);
        this->last_obj_pos.header.stamp = ros::Time(0);
    }

    void push_back(vicon::Subject cam, vicon::Subject obj) {
        this->cam_poses.push_back(cam);
        this->obj_poses.push_back(obj);
        this->counter ++;
    }

    void save_checkpoint () {
        if (this->counter == 0) {
            std::cout << "Saving checkpoint with no data!" << std::endl;
            return;
        }

        this->cam_checkpoints.push_back(this->counter - 1);
        this->obj_checkpoints.push_back(this->counter - 1);
        this->last_cam_pos = this->cam_poses.back();
        this->last_obj_pos = this->obj_poses.back();
    }

    uint32_t size() {return this->counter; }

    void smooth (uint32_t kernel) {
        this->smooth(kernel, this->cam_poses, this->cam_checkpoints, this->cam_poses_smooth);
        this->smooth(kernel, this->obj_poses, this->obj_checkpoints, this->obj_poses_smooth);
    }

    std::vector<vicon::Subject> &get_cam_poses() {
        return this->cam_poses_smooth;
    }

    std::vector<vicon::Subject> &get_obj_poses() {
        return this->obj_poses_smooth;
    }

    vicon::Subject get_last_cam_pos() {
        return this->last_cam_pos;
    }

    vicon::Subject get_last_obj_pos() {
        return this->last_obj_pos;
    }

private:
    void smooth (uint32_t kernel, std::list<vicon::Subject> &src, 
                 std::list<uint32_t> &idx, 
                 std::vector<vicon::Subject> &dst) {
        RunningAverage ra(kernel);
        dst.resize(idx.size());

        int32_t cnt = - (kernel / 2 + 1);
        uint32_t dst_id = 0;
        auto idx_it = idx.begin();
        for (auto &p : src) {
            ra.push_back(p);
            cnt ++;

            if (cnt < 0)
                continue;

            std::cout << dst_id << " " << cnt << " " << *idx_it << "\n";

            if (dst_id < idx.size() && cnt == *idx_it) {
                dst[dst_id] = ra.average();

                std::cout << "\t" << dst_id << " " << p.position.x << " -> " << dst[dst_id].position.x << "\n";
                                
                auto old_idx = *idx_it;
                ++ idx_it;
                dst_id ++;
                while(old_idx == *idx_it) {
                    dst[dst_id] = ra.average();
                    ++ idx_it;
                    dst_id ++;
                    
                    std::cout << "checkpoint index stalled!" << std::endl;
                }
            }
        }

        while (idx_it != idx.end()) {
            dst[dst_id] = ra.average();
            ++ idx_it;
            dst_id ++;
        }
    }

};


#endif // RUNNING_AVERAGE_H
