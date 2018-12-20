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

        if (cnt == 0) return ret;

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






#endif // RUNNING_AVERAGE_H
