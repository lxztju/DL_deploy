#ifndef ONNXRUNTIME_BACKEND_HPP
#define ONNXRUNTIME_BACKEND_HPP

#include <string>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "backend.hpp"
#include "backendFactory.hpp"


class OnnxruntimeBackend:public Backend{
public:
    OnnxruntimeBackend();
    ~OnnxruntimeBackend();
    OnnxruntimeBackend(const std::string& model_path, int deviceId);

    int loadModel(const std::string& modelPath, int deviceId);
    int inference(std::vector<cv::Mat>& imgMats, std::vector<std::vector<float>>& result);

private:    
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    std::shared_ptr<Ort::Session> OnnxruntuimeBackendPtr;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> outputShapes;
    int initStatus;
    int deviceId;
};
#endif // ONNXRUNTIME_BACKEND_HPP