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
std::vector<cv::KeyPoint> get_blobs(cv::Mat &image_) {
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

    image = 255 - image;

    // Setup SimpleBlobDetector parameters.
    cv::SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold = 80;
    params.maxThreshold = 100;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = float(image.rows * image.cols) * 2e-6;
    params.maxArea = float(image.rows * image.cols) * 2e-4;

    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.7;

    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.87;

    // Filter by Inertia
    params.filterByInertia = false;
    params.minInertiaRatio = 0.01;

    std::vector<cv::KeyPoint> keypoints;
    #if CV_MAJOR_VERSION < 3
      cv::SimpleBlobDetector detector(params);
      detector.detect(image, keypoints);
    #else
      cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
      detector->detect(image, keypoints);
    #endif

    //cv::Mat im_with_keypoints;
    //cv::drawKeypoints(image, keypoints, im_with_keypoints, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    //cv::imshow("keypoints", im_with_keypoints);
    //cv::waitKey(0);

    return keypoints;
}


std::vector<std::vector<uint32_t>> find_all_3lines(std::vector<cv::KeyPoint> &keypoints, float th) {
    auto n_pts = keypoints.size();
    std::vector<std::vector<uint32_t>> idx;
    for (uint32_t i = 0; i < n_pts - 2; ++i) {
        for (uint32_t j = i + 1; j < n_pts - 1; ++j) {
            float dp_x = keypoints[i].pt.x - keypoints[j].pt.x;
            float dp_y = keypoints[i].pt.y - keypoints[j].pt.y;
            float l_dp = std::sqrt(dp_x * dp_x + dp_y * dp_y);
            if (l_dp < th) continue;
            float cross = keypoints[i].pt.x * keypoints[j].pt.y - keypoints[i].pt.y * keypoints[j].pt.x;
            for (uint32_t k = j + 1; k < n_pts; ++k) {
                float d = std::fabs(dp_y * keypoints[k].pt.x - dp_x * keypoints[k].pt.y + cross) / l_dp;
                if (d >= th) continue;

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


std::vector<cv::Point2f> detect_wand_internal(std::vector<cv::KeyPoint> &keypoints, std::vector<std::vector<uint32_t>> &idx,
                                              std::valarray<std::valarray<float>> &wand_3d_mapping,
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

            float d1 = std::hypot(p2.x - keypoints[l1[0]].pt.x, p2.y - keypoints[l1[0]].pt.y);
            float d2 = std::hypot(p2.x - keypoints[l1[2]].pt.x, p2.y - keypoints[l1[2]].pt.y);
            auto &p1 = (d1 > d2) ? keypoints[l1[0]].pt : keypoints[l1[2]].pt;
            auto &p3 = (d1 > d2) ? keypoints[l1[2]].pt : keypoints[l1[0]].pt;
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
                return std::vector<cv::Point2f>();
            }
            //std::cout << l1[0] << " " << l1[1] << " "  << l1[2] << " "
            //          << l2[0] << " " << l2[1] << " "  << l2[2] << "\n";
            ret = {p1, p2, p3, p4, p5};
        }
    }

    return ret;
}


cv::Mat draw_wand(cv::Mat &img_, std::vector<cv::Point2f> &wand) {
    if (wand.size() != 5) return img_;
    cv::Mat ret = img_.clone();
    cv::line(ret, {int(wand[1].x), int(wand[1].y)}, {int(wand[0].x), int(wand[0].y)}, cv::Scalar(255, 0,   0),   2, CV_AA);
    cv::line(ret, {int(wand[1].x), int(wand[1].y)}, {int(wand[2].x), int(wand[2].y)}, cv::Scalar(0,   255, 0),   2, CV_AA);
    cv::line(ret, {int(wand[1].x), int(wand[1].y)}, {int(wand[3].x), int(wand[3].y)}, cv::Scalar(0,   0,   255), 2, CV_AA);
    cv::line(ret, {int(wand[3].x), int(wand[3].y)}, {int(wand[4].x), int(wand[4].y)}, cv::Scalar(255, 255, 255), 2, CV_AA);
    return ret;
}


std::vector<cv::Point2f> detect_wand(cv::Mat &image, float th_rel=0.5, float th_lin=0.5, float th_ang=0.5) {
    // Coordinates of red leds:
    std::valarray<std::valarray<float>> wand_3d_mapping = {{30.576542, -150.730270, -45.588951},
                                                           {-45.614960, -18.425253, 2.359259},
                                                           {-83.571022, 47.522499, 26.421692},
                                                           {64.297485, 39.157578, 6.395612},
                                                           {169.457520, 97.256927, 11.720362}};

    auto keypoints = get_blobs(image);
    auto idx = find_all_3lines(keypoints, float(std::max(image.rows, image.cols)) * 5e-3);
    auto wand_points = detect_wand_internal(keypoints, idx, wand_3d_mapping, th_rel, th_lin, th_ang);
    return wand_points;
}
} // ns wand

#endif // DETECT_WAND_H
