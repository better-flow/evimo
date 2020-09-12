#include <plot.h>



//cv::Mat plot_erate(, int res_x, int res_y, float t0) {


void TjPlot::add_trajectory_plot(Trajectory &tj) {
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

    if (this->t_rng < 0)
        this->t_rng = t[t.size() - 1];

    t *= float(this->res_x) / this->t_rng;

    x -= x[0];
    y -= y[0];
    z -= z[0];
    rr -= rr[0];
    rp -= rp[0];
    ry -= ry[0];

    this->rotation2global(rr);
    this->rotation2global(rp);
    this->rotation2global(ry);

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

    x *= (float(this->res_y / 2) - 2 * frame) / (hi_T - lo_T);
    y *= (float(this->res_y / 2) - 2 * frame) / (hi_T - lo_T);
    z *= (float(this->res_y / 2) - 2 * frame) / (hi_T - lo_T);
    rr *= (float(this->res_y / 2) - 2 * frame) / (hi_R - lo_R);
    rp *= (float(this->res_y / 2) - 2 * frame) / (hi_R - lo_R);
    ry *= (float(this->res_y / 2) - 2 * frame) / (hi_R - lo_R);
    x += frame;
    y += frame;
    z += frame;
    rr += frame;
    rp += frame;
    ry += frame;

    rr += this->res_y / 2;
    rp += this->res_y / 2;
    ry += this->res_y / 2;

    auto ret = cv::Mat(this->res_y, this->res_x, CV_8UC3, cv::Scalar(255, 255, 255));
    for (size_t i = 1; i < t.size(); ++i) {
        if (t[i] >= this->res_x) {
            std::cout << "t is out or plot range";
            continue;
        }

        #if CV_MAJOR_VERSION < 4
            auto line_type = CV_AA;
        #else
            auto line_type = cv::LINE_AA;
        #endif
        
        cv::line(ret, {(int)t[i-1], (int)x[i-1]}, {(int)t[i], (int)x[i]},   cv::Scalar(0xb4, 0x77, 0x1f), 2, line_type);
        cv::line(ret, {(int)t[i-1], (int)y[i-1]}, {(int)t[i], (int)y[i]},   cv::Scalar(0x0e, 0x7f, 0xff), 2, line_type);
        cv::line(ret, {(int)t[i-1], (int)z[i-1]}, {(int)t[i], (int)z[i]},   cv::Scalar(0x2c, 0xa0, 0x2c), 2, line_type);
        cv::line(ret, {(int)t[i-1], (int)rr[i-1]}, {(int)t[i], (int)rr[i]}, cv::Scalar(0x22, 0xbd, 0xbc), 2, line_type);
        cv::line(ret, {(int)t[i-1], (int)rp[i-1]}, {(int)t[i], (int)rp[i]}, cv::Scalar(0x7f, 0x7f, 0x7f), 2, line_type);
        cv::line(ret, {(int)t[i-1], (int)ry[i-1]}, {(int)t[i], (int)ry[i]}, cv::Scalar(0xcf, 0xbe, 0x17), 2, line_type);
    }

    cv::line(ret, {10, this->res_y / 2}, {this->res_x - 10, this->res_y / 2}, cv::Scalar(127, 127, 127));
    this->plots.push_back(ret);
    this->plots_cache.push_back(ret.clone());
}
