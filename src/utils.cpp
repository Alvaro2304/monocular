#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

double getAbsoluteScale(int frame_id, const std::string& gt_path) {
    std::ifstream myfile(gt_path);
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

void getCalibrationParams(std::string dataset_path, double& focal, cv::Point2d& pp) {
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
