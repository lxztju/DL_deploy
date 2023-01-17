#ifndef DETECT_UTILS_HPP
#define DETECT_UTILS_HPP

#include "yolov5.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>

namespace DetectUtils{

std::vector<std::string> loadNames(const std::string& path);


void visualizeDetection(cv::Mat& image, std::vector<DetectionResult>& detections,
                            const std::vector<std::string>& classNames);

void nmsCpu(std::vector<DetectionResult>& detections, const float& confThreshold, const float& iouThreshold, std::vector<DetectionResult>& nmsResult);

float iouBoxes(cv::Rect& x1, cv::Rect& x2);

}






#endif //DETECT_UTILS_HPP