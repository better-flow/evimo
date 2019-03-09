#ifndef EVENT_H
#define EVENT_H

#include "common.h"

class Event {
public:
    uint fr_x, fr_y;
    char polarity;

    ull timestamp;

    Event () : fr_x(UINT_MAX), fr_y(UINT_MAX), polarity(1), timestamp(LLONG_MAX) {}
    Event (uint x_, uint y_, ull t_) : fr_x(x_), fr_y(y_), polarity(1), timestamp(t_) {}
    Event (uint x_, uint y_, ull t_, char pol) : fr_x(x_), fr_y(y_), polarity(pol), timestamp(t_) {}

    inline uint get_x () const {return this->fr_x; }
    inline uint get_y () const {return this->fr_y; }

    inline sll operator- (const Event& rhs) {
        return sll(this->timestamp) - sll(rhs.timestamp);
    }
};


#endif // EVENT_H
