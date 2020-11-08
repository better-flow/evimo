#ifndef DETECT_WAND_H
#define DETECT_WAND_H

#include <vector>
#include <valarray>
#include <algorithm>
#include <type_traits>
#include <cmath>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace wand {
std::vector<cv::KeyPoint> get_blobs(cv::Mat &image_, bool filter=true, float th0=80, float th1=120, float ths=5) {
    auto px_size = image_.elemSize1();
    cv::Mat image3;
    if (px_size > 1) {
        image_.convertTo(image3, CV_8U, 1.0 / float(px_size));
    } else {
        image3 = image_;
    }

    cv::Mat image;
    std::vector<cv::Mat> ch;
    if (image_.channels() > 1) {
        cv::split(image3, ch);
        image = ch[0].clone();
    } else {
        image = image3.clone();
    }

    // FIXME: this sometimes makes it better, and sometimes worse!
    //cv::subtract(cv::Scalar::all(255), image, image);

    // Setup SimpleBlobDetector parameters.
    cv::SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold = th0;
    params.maxThreshold = th1;
    params.thresholdStep = ths;

    // Filter by Area.
    params.filterByArea = filter;
    params.minArea = 4  * float(image.rows * image.cols) / (640.0 * 480.0);
    params.maxArea = 20 * float(image.rows * image.cols) / (640.0 * 480.0);

    // Filter by Circularity
    params.filterByCircularity = filter;
    params.minCircularity = 0.7;

    // Filter by Convexity
    params.filterByConvexity = filter;
    params.minConvexity = 0.87;

    // Filter by Inertia
    params.filterByInertia = false;
    params.minInertiaRatio = 0.01;

    // Filter by Color
    params.filterByColor = false;
    params.blobColor = 255;

    std::vector<cv::KeyPoint> keypoints;
    #if CV_MAJOR_VERSION < 3
      cv::SimpleBlobDetector detector(params);
      detector.detect(image, keypoints);
    #else
      cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
      detector->detect(image, keypoints);
    #endif

    /*
    cv::Mat im_with_keypoints;
    cv::drawKeypoints(image, keypoints, im_with_keypoints, cv::Scalar(255,0,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("keypoints", im_with_keypoints);
    cv::waitKey(0);
    */

    return keypoints;
}


std::vector<std::vector<int32_t>> find_all_3lines(std::vector<cv::KeyPoint> &keypoints, float th) {
    int n_pts = keypoints.size();
    std::vector<std::vector<int32_t>> idx;
    for (int32_t i = 0; i < n_pts - 2; ++i) {
        for (int32_t j = i + 1; j < n_pts - 1; ++j) {
            float dp_x = keypoints[i].pt.x - keypoints[j].pt.x;
            float dp_y = keypoints[i].pt.y - keypoints[j].pt.y;
            float l_dp = std::sqrt(dp_x * dp_x + dp_y * dp_y);
            if (l_dp < th) continue;
            float cross = keypoints[i].pt.x * keypoints[j].pt.y - keypoints[i].pt.y * keypoints[j].pt.x;
            for (int32_t k = j + 1; k < n_pts; ++k) {
                float d = std::fabs(dp_y * keypoints[k].pt.x - dp_x * keypoints[k].pt.y + cross) / l_dp;
                if (d / l_dp >= th) continue;

                dp_x = std::fabs(dp_x);
                dp_y = std::fabs(dp_y);
                float mi = (dp_x > dp_y) ? keypoints[i].pt.x : keypoints[i].pt.y;
                float mj = (dp_x > dp_y) ? keypoints[j].pt.x : keypoints[j].pt.y;
                float mk = (dp_x > dp_y) ? keypoints[k].pt.x : keypoints[k].pt.y;

                if ((mi < mj) && (mj < mk)) idx.push_back({i, j, k});
                if ((mi < mk) && (mk < mj)) idx.push_back({i, k, j});
                if ((mj < mi) && (mi < mk)) idx.push_back({j, i, k});
                if ((mj < mk) && (mk < mi)) idx.push_back({j, k, i});
                if ((mk < mi) && (mi < mj)) idx.push_back({k, i, j});
                if ((mk < mj) && (mj < mi)) idx.push_back({k, j, i});
            }
        }
    }
    return idx;
}








std::vector<int32_t> detect_wand_internal_idx(std::vector<cv::KeyPoint> &keypoints, std::vector<std::vector<int32_t>> &idx,
                                              const std::valarray<std::valarray<float>> &wand_3d_mapping,
                                              float th_rel=0.5, float th_lin=0.5, float th_ang=0.5) {

    auto w1 = wand_3d_mapping[1] - wand_3d_mapping[0];
    auto w2 = wand_3d_mapping[1] - wand_3d_mapping[2];
    float gt_rel1 = std::hypot(w1[0], w1[1], w1[2]) / std::hypot(w2[0], w2[1], w2[2]);

    auto w3 = wand_3d_mapping[3] - wand_3d_mapping[1];
    auto w4 = wand_3d_mapping[3] - wand_3d_mapping[4];
    float gt_rel2 = std::hypot(w3[0], w3[1], w3[2]) / std::hypot(w4[0], w4[1], w4[2]);

    auto w5 = wand_3d_mapping[0] - wand_3d_mapping[2];
    auto w6 = wand_3d_mapping[1] - wand_3d_mapping[4];
    float gt_rel3 = std::hypot(w5[0], w5[1], w5[2]) / std::hypot(w6[0], w6[1], w6[2]);

    std::vector<cv::Point2f> ret;
    std::vector<int32_t> ret_idx;
    for (uint32_t i = 0; i < idx.size(); ++i) {
        for (uint32_t j = 0; j < idx.size(); ++j) {
            if (i == j) continue;
            auto &l1 = idx[i];
            auto &l2 = idx[j];
            if ((l1[1] != l2[0]) && (l1[1] != l2[2])) continue;
            if ((l1[0] == l2[1]) || (l1[2] == l2[1]) || (l1[1] == l2[1])) continue;
            auto &p2 = keypoints[l1[1]].pt;
            auto &p4 = keypoints[l2[1]].pt;
            auto &p5 = (l1[1] == l2[0]) ? keypoints[l2[2]].pt : keypoints[l2[0]].pt;

            auto &idx2 = l1[1];
            auto &idx4 = l2[1];
            auto &idx5 = (l1[1] == l2[0]) ? l2[2] : l2[0];

            float d1 = std::hypot(p2.x - keypoints[l1[0]].pt.x, p2.y - keypoints[l1[0]].pt.y);
            float d2 = std::hypot(p2.x - keypoints[l1[2]].pt.x, p2.y - keypoints[l1[2]].pt.y);
            auto &p1 = (d1 > d2) ? keypoints[l1[0]].pt : keypoints[l1[2]].pt;
            auto &p3 = (d1 > d2) ? keypoints[l1[2]].pt : keypoints[l1[0]].pt;
            auto &idx1 = (d1 > d2) ? l1[0] : l1[2];
            auto &idx3 = (d1 > d2) ? l1[2] : l1[0];

            float rel1 = (d1 > d2) ? d1 / d2 : d2 / d1;
            if (std::fabs(gt_rel1 - rel1) > th_rel) continue;

            float rel2 = std::hypot(p4.x - p2.x, p4.y - p2.y) / std::hypot(p4.x - p5.x, p4.y - p5.y);
            if (std::fabs(rel2 - gt_rel2) > th_lin) continue;

            float rel3 = std::hypot(p1.x - p3.x, p1.y - p3.y) / std::hypot(p2.x - p5.x, p2.y - p5.y);
            if (std::fabs(rel3 - gt_rel3) > th_lin) continue;

            float angle = std::fabs(std::atan2(p2.x - p1.x, p2.y - p1.y) - std::atan2(p2.x - p4.x, p2.y - p4.y));
            angle = std::min(std::min(std::fabs(angle), std::fabs(angle - float(M_PI) * 2.0f)), std::fabs(angle + float(M_PI) * 2.0f));
            if (std::fabs(angle - M_PI / 2.0) > th_ang) continue;

            if (ret.size() > 0) {
                std::cerr << "Failure: Detected multiple wands\n";
                return std::vector<int32_t>();
            }

            ret = {p1, p2, p3, p4, p5};
            ret_idx = {idx1, idx2, idx3, idx4, idx5};
        }
    }

    return ret_idx;
}


std::vector<cv::Point2f> detect_wand_internal(std::vector<cv::KeyPoint> &keypoints, std::vector<std::vector<int32_t>> &idx,
                                              const std::valarray<std::valarray<float>> &wand_3d_mapping,
                                              float th_rel=0.5, float th_lin=0.5, float th_ang=0.5) {
    auto ret_idx = detect_wand_internal_idx(keypoints, idx, wand_3d_mapping, th_rel, th_lin, th_ang);
    std::vector<cv::Point2f> ret;
    for (auto &i : ret_idx) ret.push_back(keypoints[i].pt);
    return ret;
}


cv::Mat draw_wand(cv::Mat &img_, std::vector<cv::Point2f> &wand) {
    if (wand.size() != 5) return img_;
    
    cv::Mat ret;
    if (img_.channels() == 1) {
        cv::cvtColor(img_, ret, CV_GRAY2BGR);
    } else {
        ret = img_.clone();
    }

    #if CV_MAJOR_VERSION < 4
        cv::line(ret, {int(wand[1].x), int(wand[1].y)}, {int(wand[0].x), int(wand[0].y)}, cv::Scalar(255, 0,   0),   2, CV_AA);
        cv::line(ret, {int(wand[1].x), int(wand[1].y)}, {int(wand[2].x), int(wand[2].y)}, cv::Scalar(0,   255, 0),   2, CV_AA);
        cv::line(ret, {int(wand[1].x), int(wand[1].y)}, {int(wand[3].x), int(wand[3].y)}, cv::Scalar(0,   0,   255), 2, CV_AA);
        cv::line(ret, {int(wand[3].x), int(wand[3].y)}, {int(wand[4].x), int(wand[4].y)}, cv::Scalar(255, 255, 255), 2, CV_AA);
    #else
        cv::line(ret, {int(wand[1].x), int(wand[1].y)}, {int(wand[0].x), int(wand[0].y)}, cv::Scalar(255, 0,   0),   2, cv::LINE_AA);
        cv::line(ret, {int(wand[1].x), int(wand[1].y)}, {int(wand[2].x), int(wand[2].y)}, cv::Scalar(0,   255, 0),   2, cv::LINE_AA);
        cv::line(ret, {int(wand[1].x), int(wand[1].y)}, {int(wand[3].x), int(wand[3].y)}, cv::Scalar(0,   0,   255), 2, cv::LINE_AA);
        cv::line(ret, {int(wand[3].x), int(wand[3].y)}, {int(wand[4].x), int(wand[4].y)}, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    #endif
    return ret;
}


cv::Mat draw_wand(cv::Mat &img_, std::vector<cv::KeyPoint> &keypoints, std::vector<int32_t> &idx) {
    std::vector<cv::Point2f> wand;
    for (auto &i : idx) wand.push_back(keypoints[i].pt);
    return draw_wand(img_, wand);
}



const std::valarray<std::valarray<float>> wand_red_mapping = {{ 0.030576542, -0.150730270, -0.045588951},
                                                              {-0.045614960, -0.018425253,  0.002359259},
                                                              {-0.083571022,  0.047522499,  0.026421692},
                                                              { 0.064297485,  0.039157578,  0.006395612},
                                                              { 0.169457520,  0.097256927,  0.011720362}};

const std::valarray<std::valarray<float>> wand_ir_mapping =  {{ 0.032454151, -0.153981064, -0.046729141},
                                                              {-0.043735237, -0.021440924,  0.001182336},
                                                              {-0.081761475,  0.044537903,  0.025208614},
                                                              { 0.060946213,  0.036995384,  0.006158191},
                                                              { 0.166117447,  0.095441071,  0.011552736}};

std::vector<cv::Point2f> detect_wand(cv::Mat &image, float th_rel=0.5, float th_lin=0.5, float th_ang=0.5) {
    // Coordinates of red leds:

    auto keypoints = get_blobs(image);
    auto idx = find_all_3lines(keypoints, 0.2);
    auto wand_points = detect_wand_internal(keypoints, idx, wand_red_mapping, th_rel, th_lin, th_ang);
    return wand_points;
}
} // ns wand

#endif // DETECT_WAND_H
