#ifndef IMG_CLS_HPP
#define IMG_CLS_HPP

#include <string>
#include<vector>
#include <tuple>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "backend.hpp"

class Backend;
typedef std::shared_ptr<Backend> backendPtr;




class ImgCls{

public:
    ImgCls(const std::string& modelPath, const std::string& backendType, int deviceId);
public:
    int runImg(std::vector<std::string>& imgPaths, std::tuple<std::vector<int64_t>, std::vector<float>>& res);


private:
    std::string modelPath;
    std::string backendType; 
    int deviceId;
    backendPtr clsPtr;
    int clsWidth = 224;
    int clsHeight = 224;
    std::vector<float> mean_{0.485, 0.456, 0.406};
    std::vector<float> std_{0.229, 0.224, 0.225};
};

#endif //IMG_CLS_HPP