#pragma once
#include <opencv2/opencv.hpp>
#include <string>

double getAbsoluteScale(int frame_id, const std::string& gt_path);
void getCalibrationParams(std::string dataset_path, double& focal, cv::Point2d& pp);
