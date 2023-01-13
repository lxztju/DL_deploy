#ifndef TORCH_BACKEND_HPP
#define TORCH_BACKEND_HPP

#include "backend.hpp"
#include "backendFactory.hpp"



class TorchBackend:public Backend{
public:
    TorchBackend();
    ~TorchBackend();
    TorchBackend(const std::string& modelPath, int deviceId);


    int loadModel(const std::string& modelPath, int deviceId);
    int inference(std::vector<cv::Mat>& imgMats, std::vector<std::vector<float>>& result);

private:
    std::shared_ptr<torch::jit::script::Module> torchBackendPtr;
    torch::DeviceType deviceType;
    int initStatus;
};


#endif //TORCH_BACKEND_HPP