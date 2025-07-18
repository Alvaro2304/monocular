#pragma once
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

void FastFeatureDetection(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
TrackFeatures(const cv::Mat& prev_img, const cv::Mat& next_img, const std::vector<cv::Point2f>& prev_pts);
