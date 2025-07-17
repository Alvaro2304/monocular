#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <tuple>
#include <fstream>
#include <sstream>
#include <iostream>

void FastFeatureDetection(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) {
    static cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(50, true);
    fast->detect(image, keypoints);
}

// Track features and filter by status
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

double getAbsoluteScale(int frame_id) {
    std::ifstream myfile("../../kitti_dataset/data_odometry_poses/dataset/poses/01.txt");
    if (!myfile.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return 0;
    }
    std::string line;
    double x_prev = 0, y_prev = 0, z_prev = 0, x = 0, y = 0, z = 0;
    for (int i = 0; i <= frame_id; ++i) {
        std::getline(myfile, line);
        std::istringstream in(line);
        double data[12];
        for (int j = 0; j < 12; ++j) in >> data[j];
        if (i == frame_id - 1) {
            x_prev = data[3]; y_prev = data[7]; z_prev = data[11];
        }
        if (i == frame_id) {
            x = data[3]; y = data[7]; z = data[11];
        }
    }
    myfile.close();
    return std::sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev));
}

void getCalibrationParams(std::string dataset_path, double& focal, cv::Point2d& pp){

    std::ifstream calib_file(dataset_path + "/calib.txt");
    if (!calib_file.is_open()) {
        std::cerr << "Unable to open calibration file" << std::endl;
        return;
    }
    std::string line;
    while (std::getline(calib_file, line)) {
        if (line.find("P0:") == 0) {
            std::istringstream iss(line.substr(3));
            double data[12];
            for (int i = 0; i < 12; ++i) iss >> data[i];
            focal = data[0];
            pp = cv::Point2d(data[2], data[6]);
            break;
        }
    }
    calib_file.close();
}

int main() {
    // Get calibration parameters
    double focal;
    cv::Point2d pp;
    getCalibrationParams("../../kitti_dataset/data_odometry_gray/dataset/sequences/01", focal, pp);

    // Initial pose
    cv::Mat R_f = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_f = cv::Mat::zeros(3, 1, CV_64F);

    // Load first image and detect features
    std::string base_path = "../../kitti_dataset/data_odometry_gray/dataset/sequences/01/image_0/";
    char img_name[256];
    sprintf(img_name, "%06d.png", 0);
    cv::Mat prev_img = cv::imread(base_path + img_name, cv::IMREAD_GRAYSCALE);
    if (prev_img.empty()) {
        std::cerr << "Error: Could not load first image." << std::endl;
        return 1;
    }
    std::vector<cv::KeyPoint> keypoints0;
    FastFeatureDetection(prev_img, keypoints0);
    std::vector<cv::Point2f> prev_pts;
    cv::KeyPoint::convert(keypoints0, prev_pts);

    int num_frames = 4540; // KITTI 01 has 4541 images (0-4540)
    // Trajectory visualization image
    int traj_size = 600;
    double traj_scale = 0.1; // Adjust this value as needed to fit the trajectory
    cv::Mat traj = cv::Mat::zeros(traj_size, traj_size, CV_8UC3);

    // Load ground truth poses
    std::vector<std::pair<double, double>> gt_positions(num_frames + 1);
    std::ifstream gt_file("../../kitti_dataset/data_odometry_poses/dataset/poses/01.txt");
    if (!gt_file.is_open()) {
        std::cerr << "Unable to open ground truth file for trajectory comparison." << std::endl;
    } else {
        std::string line;
        int idx = 0;
        while (std::getline(gt_file, line) && idx <= num_frames) {
            std::istringstream in(line);
            double data[12];
            for (int j = 0; j < 12; ++j) in >> data[j];
            gt_positions[idx] = {data[3], data[11]}; // x, z
            ++idx;
        }
        gt_file.close();
    }
    for (int frame_id = 1; frame_id < num_frames; ++frame_id) {
        sprintf(img_name, "%06d.png", frame_id);
        cv::Mat curr_img = cv::imread(base_path + img_name, cv::IMREAD_GRAYSCALE);
        if (curr_img.empty()) {
            std::cerr << "Error: Could not load image " << img_name << std::endl;
            break;
        }

        // Track features
        std::vector<cv::Point2f> prev_inliers, curr_inliers;
        std::tie(prev_inliers, curr_inliers) = TrackFeatures(prev_img, curr_img, prev_pts);

        // Compute Essential matrix
        cv::Mat E = cv::findEssentialMat(curr_inliers, prev_inliers, focal, pp, cv::RANSAC, 0.999, 1.0);
        cv::Mat R, t;
        int inliers = cv::recoverPose(E, curr_inliers, prev_inliers, R, t, focal, pp);

        // Get scale from ground truth
        double scale = getAbsoluteScale(frame_id);
        if (scale > 0.1 && t.at<double>(2) > t.at<double>(0) && t.at<double>(2) > t.at<double>(1)) {
            t_f = t_f + scale * (R_f * t);
            R_f = R * R_f;
        }


        // Visualization: show only one window with current image and tracks
        cv::Mat vis_img;
        cv::cvtColor(curr_img, vis_img, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < prev_inliers.size(); ++i) {
            cv::circle(vis_img, curr_inliers[i], 2, cv::Scalar(0,0,255), -1);
            cv::line(vis_img, prev_inliers[i], curr_inliers[i], cv::Scalar(255,0,0), 1);
        }
        cv::imshow("Tracked Points", vis_img);

        // Draw estimated trajectory (green)
        int x = int(t_f.at<double>(0) * traj_scale) + traj_size / 2;
        int y = int(t_f.at<double>(2) * traj_scale) + traj_size / 2;
        cv::circle(traj, cv::Point(x, y), 1, cv::Scalar(0,255,0), 2);

        // Draw ground truth trajectory (red)
        if (frame_id < gt_positions.size()) {
            int x_gt = int(gt_positions[frame_id].first * traj_scale) + traj_size / 2;
            int y_gt = int(gt_positions[frame_id].second * traj_scale) + traj_size / 2;
            cv::circle(traj, cv::Point(x_gt, y_gt), 1, cv::Scalar(0,0,255), 2);
        }
        cv::imshow("Trajectory", traj);

        if (cv::waitKey(1) == 27) break; // ESC to exit

        // Prepare for next iteration
        prev_img = curr_img;
        prev_pts = curr_inliers;

        // If too few points, re-detect
        if (prev_pts.size() < 200) {
            std::vector<cv::KeyPoint> new_kps;
            FastFeatureDetection(prev_img, new_kps);
            cv::KeyPoint::convert(new_kps, prev_pts);
        }
    }
    cv::destroyAllWindows();
    return 0;
}
