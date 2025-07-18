#include "feature_tracker.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

int main(int argc, char** argv) {
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <sequence_number>" << std::endl;
        return 1;
    }
    
    std::string sequence = argv[1];

    std::map<std::string, int> sequence_frame_count = {
        {"00", 4540},
        {"01", 1100},
        {"02", 4660},
        {"03", 800},
        {"04", 270},
        {"05", 2760},
        {"06", 1100},
        {"07", 1100},
        {"08", 4070},
        {"09", 1590},
        {"10", 1200},
        {"11", 920},
        {"12", 1060},
        {"13", 3280},
        {"14", 630},
        {"15", 1900},
        {"16", 1730},
        {"17", 490},
        {"18", 1800},
        {"19", 4980},
        {"20", 830},
        {"21", 2720}
    };

    if (sequence_frame_count.find(sequence) == sequence_frame_count.end()) {
        std::cerr << "Invalid sequence number. Available sequences are:" << std::endl;
        for (const auto& pair : sequence_frame_count) {
            std::cerr << pair.first << " ";
        }
        std::cerr << std::endl;
        return 1;
    }

    // Load calibration parameters
    double focal;
    cv::Point2d pp;
    std::string calib_path = "../../kitti_dataset/data_odometry_gray/dataset/sequences/" + sequence;
    getCalibrationParams(calib_path, focal, pp);


    cv::Mat R_f = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_f = cv::Mat::zeros(3, 1, CV_64F);

    std::string base_path = "../../kitti_dataset/data_odometry_gray/dataset/sequences/" + sequence + "/image_0/";
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

    int num_frames = sequence_frame_count[sequence];
    int traj_size = 600;
    double traj_scale = 0.1;
    cv::Mat traj = cv::Mat::zeros(traj_size, traj_size, CV_8UC3);

    // Load ground truth positions
    std::vector<std::pair<double, double>> gt_positions(num_frames + 1);
    std::string gt_path = "../../kitti_dataset/data_odometry_poses/dataset/poses/" + sequence + ".txt";
    std::ifstream gt_file(gt_path);
    if (!gt_file.is_open()) {
        std::cerr << "Unable to open ground truth file." << std::endl;
    } else {
        std::string line;
        int idx = 0;
        while (std::getline(gt_file, line) && idx <= num_frames) {
            std::istringstream in(line);
            double data[12];
            for (int j = 0; j < 12; ++j) in >> data[j];
            gt_positions[idx] = {data[3], data[11]};
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

        std::vector<cv::Point2f> prev_inliers, curr_inliers;
        std::tie(prev_inliers, curr_inliers) = TrackFeatures(prev_img, curr_img, prev_pts);

        cv::Mat E = cv::findEssentialMat(curr_inliers, prev_inliers, focal, pp, cv::RANSAC, 0.999, 1.0);
        cv::Mat R, t;
        int inliers = cv::recoverPose(E, curr_inliers, prev_inliers, R, t, focal, pp);

        double scale = getAbsoluteScale(frame_id, gt_path);

        if (scale > 0.1 && t.at<double>(2) > t.at<double>(0) && t.at<double>(2) > t.at<double>(1)) {
            t_f = t_f + scale * (R_f * t);
            R_f = R * R_f;
        }

        cv::Mat vis_img;
        cv::cvtColor(curr_img, vis_img, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < prev_inliers.size(); ++i) {
            cv::circle(vis_img, curr_inliers[i], 2, cv::Scalar(0, 0, 255), -1);
            cv::line(vis_img, prev_inliers[i], curr_inliers[i], cv::Scalar(255, 0, 0), 1);
        }
        cv::imshow("Tracked Points", vis_img);

        int x = int(t_f.at<double>(0) * traj_scale) + traj_size / 2;
        int y = int(t_f.at<double>(2) * traj_scale) + traj_size / 2;
        cv::circle(traj, cv::Point(x, y), 1, cv::Scalar(0, 255, 0), 2);

        if (frame_id < gt_positions.size()) {
            int x_gt = int(gt_positions[frame_id].first * traj_scale) + traj_size / 2;
            int y_gt = int(gt_positions[frame_id].second * traj_scale) + traj_size / 2;
            cv::circle(traj, cv::Point(x_gt, y_gt), 1, cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("Trajectory", traj);

        if (cv::waitKey(1) == 27) break;

        prev_img = curr_img;
        prev_pts = curr_inliers;

        if (prev_pts.size() < 200) {
            std::vector<cv::KeyPoint> new_kps;
            FastFeatureDetection(prev_img, new_kps);
            cv::KeyPoint::convert(new_kps, prev_pts);
        }
    }
    cv::destroyAllWindows();
    return 0;
}
