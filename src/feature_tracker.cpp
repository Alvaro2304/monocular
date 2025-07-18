#include "feature_tracker.h"

void FastFeatureDetection(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) {
    static cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(50, true);
    fast->detect(image, keypoints);
}

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
TrackFeatures(const cv::Mat& prev_img, const cv::Mat& next_img, const std::vector<cv::Point2f>& prev_pts) {
    std::vector<cv::Point2f> next_pts;
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_img, next_img, prev_pts, next_pts, status, err);
    std::vector<cv::Point2f> prev_inliers, next_inliers;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            prev_inliers.push_back(prev_pts[i]);
            next_inliers.push_back(next_pts[i]);
        }
    }
    return {prev_inliers, next_inliers};
}
