#ifndef BACKEND_HPP
#define BACKEND_HPP

#include <string>
#include <vector>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "backendFactory.hpp"


// backend 基类
class Backend{
public:
    // Backend(){};
    // Backend(const std::string& backendType);
    // Backend(const std::string& backendType, int deviceId);

    virtual int loadModel(const std::string& modelPath, int deviceId)=0;
    virtual int inference(std::vector<cv::Mat>& imgMats, std::vector<std::vector<float>>& result)=0;


// protected:
//     const std::string backendType;
//     std::shared_ptr<Backend> inferBackendPtr;
//     int deviceId; 
//     int initStatus;
};


#endif //BACKEND_HPP
