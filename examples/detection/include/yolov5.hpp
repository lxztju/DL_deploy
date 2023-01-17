#ifndef YOLOV5_HPP
#define YOLOV5_HPP

#include <string>
#include<vector>
#include <tuple>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "backend.hpp"

class Backend;
typedef std::shared_ptr<Backend> backendPtr;


struct DetectionResult{
    cv::Rect box;
    float conf{};
    int classId{};
};



class YoloV5Detect{

public:
    YoloV5Detect(const std::string& modelPath, const std::string& backendType, int deviceId);
public:
    int runImg(
        std::vector<std::string>& imgPaths, 
        std::vector<std::vector<DetectionResult>>& res,
        const float& confThreshold,
        const float& iouThreshold
        );

private:
    int preprocessImage(cv::Mat& images);
    int postprocessImage(
        std::vector<std::vector<float>>& modelOutput,
        const std::vector<cv::Size>& originalImageShapes,
        std::vector<std::vector<DetectionResult>>& res,
        const float& confThreshold,
        const float& iouThreshold
    );


private:
    std::string modelPath;
    std::string backendType; 
    int deviceId;
    backendPtr yolov5Ptr;
    cv::Size modelInputImageShape{640, 640};

};
#endif //YOLOV5_HPP