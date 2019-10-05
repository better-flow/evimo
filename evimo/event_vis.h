#include <common.h>
#include <event.h>

#ifndef EVENT_VIS_H
#define EVENT_VIS_H

class EventFile {
public:
    template<class T> static cv::Mat color_time_img     (T *events, int scale = 0, int res_x = RES_X, int res_y = RES_Y);
    template<class T> static cv::Mat projection_img     (T *events, int scale = 1, int res_x = RES_X, int res_y = RES_Y);
    template<class T> static cv::Mat projection_img_unopt (T *events, int scale, int res_x = RES_X, int res_y = RES_Y);
    static double nonzero_average (cv::Mat img);
    static void nonzero_norm (cv::Mat img);

    static cv::Vec3b id2rgb(unsigned int id);
};


template<class T> cv::Mat EventFile::projection_img (T *events, int scale, int res_x, int res_y) {

    int scale_img_x = res_x * scale;
    int scale_img_y = res_y * scale;

    int cnt = 0;
    cv::Mat best_project_hires_img = cv::Mat::zeros(scale_img_x, scale_img_y, CV_8UC1);
    for (auto &e : *events) {
        int x = e.fr_x * scale;
        int y = e.fr_y * scale;

        if ((x >= scale * (res_x - 1)) || (x < 0) || (y >= scale * (res_y - 1)) || (y < 0))
            continue;

        x += scale / 2;
        y += scale / 2;

        cnt ++;

        int lx = std::max(x - scale / 2, 0), ly = std::max(y - scale / 2, 0);
        int rx = std::min(x + scale / 2, scale_img_x), ry = std::min(y + scale / 2, scale_img_y);
        for (int jx = lx; jx <= rx; ++jx) {
            for (int jy = ly; jy <= ry; ++jy) {
                if (best_project_hires_img.at<uchar>(jx, jy) < 255)
                    best_project_hires_img.at<uchar>(jx, jy) ++;
            }
        }
    }

    if (scale > 1) {
        cv::GaussianBlur(best_project_hires_img, best_project_hires_img, cv::Size(scale, scale), 0, 0);
    }

    double img_scale = 127.0 / EventFile::nonzero_average(best_project_hires_img);
    cv::convertScaleAbs(best_project_hires_img, best_project_hires_img, img_scale, 0);
    return best_project_hires_img;
}


template<class T> cv::Mat EventFile::projection_img_unopt (T *events, int scale, int res_x, int res_y) {

    int scale_img_x = res_x * scale;
    int scale_img_y = res_y * scale;

    int cnt = 0;
    cv::Mat best_project_hires_img = cv::Mat::zeros(scale_img_x, scale_img_y, CV_8UC1);
    for (auto &e : *events) {
        int x = e.fr_x * scale;
        int y = e.fr_y * scale;

        if ((x >= scale * (res_x - 1)) || (x < 0) || (y >= scale * (res_y - 1)) || (y < 0))
            continue;

        x += scale / 2;
        y += scale / 2;

        cnt ++;

        int lx = std::max(x - scale / 2, 0), ly = std::max(y - scale / 2, 0);
        int rx = std::min(x + scale / 2, scale_img_x), ry = std::min(y + scale / 2, scale_img_y);
        for (int jx = lx; jx <= rx; ++jx) {
            for (int jy = ly; jy <= ry; ++jy) {
                if (best_project_hires_img.at<uchar>(jx, jy) < 255)
                    best_project_hires_img.at<uchar>(jx, jy) ++;
            }
        }
    }

    if (scale > 1) {
        cv::GaussianBlur(best_project_hires_img, best_project_hires_img, cv::Size(scale, scale), 0, 0);
    }

    double img_scale = 127.0 / EventFile::nonzero_average(best_project_hires_img);
    cv::convertScaleAbs(best_project_hires_img, best_project_hires_img, img_scale, 0);
    return best_project_hires_img;
}


template<class T> cv::Mat EventFile::color_time_img (T *events, int scale, int res_x, int res_y) {
    if (scale == 0) scale = 11;

    ull t_min = LLONG_MAX, t_max = 0;
    uint x_min = res_x, y_min = res_y, x_max = 0, y_max = 0;
    for (auto &e : *events) {
        if (e.get_x() > x_max) x_max = e.get_x();
        if (e.get_y() > y_max) y_max = e.get_y();
        if (e.get_x() < x_min) x_min = e.get_x();
        if (e.get_y() < y_min) y_min = e.get_y();
    }

    for (auto &e : *events) {
        if (e.timestamp < (sll)t_min) t_min = e.timestamp;
        if (e.timestamp > (sll)t_max) t_max = e.timestamp;
    }

    x_max = std::min(x_max, (uint)res_x);
    y_max = std::min(y_max, (uint)res_y);

    x_min = 0; y_min = 0; x_max = res_x; y_max = res_y;

    if ((x_min > x_max) || (y_min > y_max)) {
        return cv::Mat(0, 0, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    int metric_wsizex = scale * (x_max - x_min);
    int metric_wsizey = scale * (y_max - y_min);
    int scale_img_x = metric_wsizex + scale;
    int scale_img_y = metric_wsizey + scale;

    cv::Mat project_img(scale_img_x, scale_img_y, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat project_img_cnt = cv::Mat::zeros(scale_img_x, scale_img_y, CV_32FC1);
    cv::Mat project_img_avg(scale_img_x, scale_img_y, CV_8UC3, cv::Scalar(0, 0, 0));

    double x_shift = - double((x_max - x_min) / 2 + x_min) * double(scale) + double(metric_wsizex) / 2.0;
    double y_shift = - double((y_max - y_min) / 2 + y_min) * double(scale) + double(metric_wsizey) / 2.0;

    int ignored = 0, total = 0;
    for (auto &e : *events) {
        total ++;
        int x = e.fr_x * scale + x_shift;
        int y = e.fr_y * scale + y_shift;

        if ((x >= metric_wsizex) || (x < 0) || (y >= metric_wsizey) || (y < 0)) {
            ignored ++;
            continue;
        }

        float angle = 2 * 3.14 * (double(e.timestamp - t_min) / double(t_max - t_min));

        x += scale / 2;
        y += scale / 2;

        for (int jx = x - scale / 2; jx <= x + scale / 2; ++jx) {
            for (int jy = y - scale / 2; jy <= y + scale / 2; ++jy) {
                project_img.at<cv::Vec3f>(jx, jy)[0] += cos(angle);
                project_img.at<cv::Vec3f>(jx, jy)[1] += sin(angle);
                project_img_cnt.at<float>(jx, jy) += 1;
            }
        }
    }

    for (int jx = 0; jx < scale_img_x; ++jx) {
        for (int jy = 0; jy < scale_img_y; ++jy) {
            if (project_img_cnt.at<float>(jx, jy) < 1) continue;

            float vx = (project_img.at<cv::Vec3f>(jx, jy)[0] / (float)project_img_cnt.at<float>(jx, jy));
            float vy = (project_img.at<cv::Vec3f>(jx, jy)[1] / (float)project_img_cnt.at<float>(jx, jy));

            double speed = hypot(vx, vy);
            double angle = 0;
            if (speed != 0)
                angle = (atan2(vy, vx) + 3.1416) * 180 / 3.1416;

            project_img_avg.at<cv::Vec3b>(jx, jy)[0] = angle / 2;
            project_img_avg.at<cv::Vec3b>(jx, jy)[1] = speed * 255;
            project_img_avg.at<cv::Vec3b>(jx, jy)[2] = 255;
        }
    }

    cv::cvtColor(project_img_avg, project_img_avg, CV_HSV2BGR);
    return project_img_avg;
}


double EventFile::nonzero_average (cv::Mat img) {
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


void EventFile::nonzero_norm (cv::Mat img) {
    double nz_min = 10000, nz_max = 0;
    long int nz_cnt = 0;
    uchar* p = img.data;
    for(int i = 0; i < img.rows * img.cols; ++i, p++) {
        if (*p <= 0.0001) continue;
        nz_cnt ++;
        if (*p > nz_max) nz_max = *p;
        if (*p < nz_min) nz_min = *p;
    }

    if (nz_cnt == 0)
        return;

    p = img.data;
    for(int i = 0; i < img.rows * img.cols; ++i, p++) {
        if (*p == 0) continue;
        *p -= nz_min;
    }

    double rng = nz_max - nz_min;

    if (rng < 0.0001)
        return;

    img /= rng;
}


cv::Vec3b EventFile::id2rgb(unsigned int id) {
    cv::Vec3b ret;
    int COLORS[6][3] = {{255, 255, 0}, {255, 0, 255}, {0, 255, 255},
                        {255, 0, 0},   {0, 255, 0},   {0, 0, 255}};
    int len = sizeof(COLORS) / sizeof(*COLORS);
    ret[0] = COLORS[id % len][0];
    ret[1] = COLORS[id % len][1];
    ret[2] = COLORS[id % len][2];
    return ret;
}


#endif // EVENT_VIS_H
