#include <dataset_frame.h>
#include <event_vis.h>

cv::Mat DatasetFrame::get_visualization_event_projection(bool timg, bool nodist) {
    cv::Mat img;
    if (Dataset::event_array.size() == 0)
        return cv::Mat(this->depth.rows, this->depth.cols, CV_8U, cv::Scalar(0));

    if (Dataset::event_array.size() > 0) {
        auto ev_slice = Slice<std::vector<Event>>(Dataset::event_array,
                                                  this->event_slice_ids);
        if (timg) {
            img = EventFile::color_time_img(&ev_slice, 1, this->dataset_handle->res_x, this->dataset_handle->res_y);
        } else {
            img = EventFile::projection_img(&ev_slice, 1, this->dataset_handle->res_x, this->dataset_handle->res_y);
        }
    }

    if (nodist)
        img = this->dataset_handle->undistort(img);
    return img;
}

cv::Mat DatasetFrame::get_visualization_depth(bool overlay_events, bool nodist) {
    auto depth_img = this->depth;
    cv::Mat img_pr = this->get_visualization_event_projection(false, nodist);
    auto ret = cv::Mat(depth_img.rows, depth_img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat mask;
    cv::threshold(depth_img, mask, 0.01, 255, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);
    cv::normalize(depth_img, depth_img, 1, 255, cv::NORM_MINMAX, -1, mask);
    //cv::divide(8000.0, depth_img, depth_img);
    for(int i = 0; i < depth_img.rows; ++i) {
        for (int j = 0; j < depth_img.cols; ++j) {
            ret.at<cv::Vec3b>(i, j)[0] = depth_img.at<float>(i, j);
            ret.at<cv::Vec3b>(i, j)[1] = depth_img.at<float>(i, j);
            ret.at<cv::Vec3b>(i, j)[2] = depth_img.at<float>(i, j);
            if (overlay_events && Dataset::event_array.size() > 0)
                ret.at<cv::Vec3b>(i, j)[2] = img_pr.at<uint8_t>(i, j);
        }
    }

    return ret;
}

cv::Mat DatasetFrame::get_visualization_mask(bool overlay_events, bool nodist) {
    auto mask_img = this->mask;
    auto rgb_img  = this->img.clone();

    if (nodist)
        rgb_img = this->dataset_handle->undistort(rgb_img);

    cv::Mat img_pr = this->get_visualization_event_projection(false, nodist);
    auto ret = cv::Mat(mask_img.rows, mask_img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for(int i = 0; i < mask_img.rows; ++i) {
        for (int j = 0; j < mask_img.cols; ++j) {
            int id = std::round(mask_img.at<uint8_t>(i, j));
            auto color = EventFile::id2rgb(id);
            if (rgb_img.rows == mask_img.rows && rgb_img.cols == mask_img.cols) {
                ret.at<cv::Vec3b>(i, j) = rgb_img.at<cv::Vec3b>(i, j);
                if (id > 0) {
                    ret.at<cv::Vec3b>(i, j) = rgb_img.at<cv::Vec3b>(i, j) * 0.2 + color * 0.8;
                }
                if (id >= 250) {
                    ret.at<cv::Vec3b>(i, j) = color;
                }
            } else {
                if (id > 0)
                    ret.at<cv::Vec3b>(i, j) = color;
            }
            if (overlay_events && Dataset::event_array.size() > 0 && img_pr.at<uint8_t>(i, j) > 0)
                ret.at<cv::Vec3b>(i, j)[2] = img_pr.at<uint8_t>(i, j);
        }
    }

    return ret;
}

std::list<DatasetFrame*> DatasetFrame::visualization_list;
