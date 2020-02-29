#ifndef PLOT_H
#define PLOT_H

#include <common.h>
#include <trajectory.h>

cv::Mat plot_trajectory(Trajectory &tj, int res_x=800, int res_y=200, float t0 = -1);


#endif // PLOT_H
