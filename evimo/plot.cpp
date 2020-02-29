#include <plot.h>



void rotation2global(std::valarray<float> &arr) {
    auto last_val = arr[0];
    for (size_t i = 1; i < arr.size(); ++i) {
        auto cnt = std::round((last_val - arr[i]) / (2 * M_PI));
        arr[i] += cnt * 2 * M_PI;
        last_val = arr[i];
    }
}


//cv::Mat plot_erate(, int res_x, int res_y, float t0) {


cv::Mat plot_trajectory(Trajectory &tj, int res_x, int res_y, float t0) {
    auto n_pts = tj.size();
    std::valarray<float> t(n_pts);
    std::valarray<float> x(n_pts);
    std::valarray<float> y(n_pts);
    std::valarray<float> z(n_pts);
    std::valarray<float> rr(n_pts);
    std::valarray<float> rp(n_pts);
    std::valarray<float> ry(n_pts);
    for (size_t i = 0; i < n_pts; ++i) {
        auto p = tj[i];
        t[i] = p.get_ts_sec();
        auto T = p.getT();
        auto R = p.getR();
        x[i] = -T[0];
        y[i] = -T[1];
        z[i] = -T[2];
        rr[i] = -R[0];
        rp[i] = -R[1];
        ry[i] = -R[2];
    }

    auto t_rng = t[t.size() - 1] - t[0];
    t -= t[0];
    t *= float(res_x) / t_rng;
    t0 -= t[0];
    t0 *= float(res_x) / t_rng;

    x -= x[0];
    y -= y[0];
    z -= z[0];
    rr -= rr[0];
    rp -= rp[0];
    ry -= ry[0];

    rotation2global(rr);
    rotation2global(rp);
    rotation2global(ry);

    auto lo_T = std::min({x.min(), y.min(), z.min()});
    auto hi_T = std::max({x.max(), y.max(), z.max()});

    auto lo_R = std::min({rr.min(), rp.min(), ry.min()});
    auto hi_R = std::max({rr.max(), rp.max(), ry.max()});

    x -= lo_T;
    y -= lo_T;
    z -= lo_T;
    rr -= lo_R;
    rp -= lo_R;
    ry -= lo_R;

    float frame = 10;

    x *= (float(res_y / 2) - 2 * frame) / (hi_T - lo_T);
    y *= (float(res_y / 2) - 2 * frame) / (hi_T - lo_T);
    z *= (float(res_y / 2) - 2 * frame) / (hi_T - lo_T);
    rr *= (float(res_y / 2) - 2 * frame) / (hi_R - lo_R);
    rp *= (float(res_y / 2) - 2 * frame) / (hi_R - lo_R);
    ry *= (float(res_y / 2) - 2 * frame) / (hi_R - lo_R);
    x += frame;
    y += frame;
    z += frame;
    rr += frame;
    rp += frame;
    ry += frame;

    rr += res_y / 2;
    rp += res_y / 2;
    ry += res_y / 2;

    auto ret = cv::Mat(res_y, res_x, CV_8UC3, cv::Scalar(255, 255, 255));
    for (size_t i = 1; i < t.size(); ++i) {
        cv::line(ret, {t[i-1], x[i-1]}, {t[i], x[i]},   cv::Scalar(0xb4, 0x77, 0x1f), 2, CV_AA);
        cv::line(ret, {t[i-1], y[i-1]}, {t[i], y[i]},   cv::Scalar(0x0e, 0x7f, 0xff), 2, CV_AA);
        cv::line(ret, {t[i-1], z[i-1]}, {t[i], z[i]},   cv::Scalar(0x2c, 0xa0, 0x2c), 2, CV_AA);
        cv::line(ret, {t[i-1], rr[i-1]}, {t[i], rr[i]}, cv::Scalar(0x22, 0xbd, 0xbc), 2, CV_AA);
        cv::line(ret, {t[i-1], rp[i-1]}, {t[i], rp[i]}, cv::Scalar(0x7f, 0x7f, 0x7f), 2, CV_AA);
        cv::line(ret, {t[i-1], ry[i-1]}, {t[i], ry[i]}, cv::Scalar(0xcf, 0xbe, 0x17), 2, CV_AA);
    }

    cv::line(ret, {10, res_y / 2}, {res_x - 10, res_y / 2}, cv::Scalar(127, 127, 127));

    if(t0 >= 0) {
        cv::line(ret, {t0, frame}, {t0, res_y - frame}, cv::Scalar(0, 0, 255));
    }

    return ret;
}


