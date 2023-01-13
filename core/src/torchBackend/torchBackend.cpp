#include "torchBackend.hpp"

#include <iostream>


TorchBackend::TorchBackend(){

}

TorchBackend::~TorchBackend(){

}

TorchBackend::TorchBackend(const std::string& modelPath, int deviceId){
    if (deviceId < 0){
        deviceType = torch::kCPU;
        
    }
    else{
        deviceType = torch::kCUDA;
    }
    // device_ = torch::Device(device_type);
    initStatus = this->loadModel(modelPath, deviceId);
}


int TorchBackend::loadModel(const std::string& modelPath, int deviceId){
    if (deviceId < 0){
        deviceType = torch::kCPU;
        
    }
    else{
        deviceType = torch::kCUDA;
    }
    try {
        torchBackendPtr = std::make_shared<torch::jit::script::Module>(torch::jit::load(modelPath, torch::Device(deviceType)));
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model!\n";
        return -1;
        // std::exit(EXIT_FAILURE);
    }
    torchBackendPtr->to(torch::Device(deviceType));
    torchBackendPtr->eval();
    std::cout<< deviceId<<std::endl;
    return 0;
}


int TorchBackend::inference(std::vector<cv::Mat>& imgMats, std::vector<std::vector<float>>& result){
    if(0==imgMats.size()){
        return -1;
    }
    assert(imgMats.size() <= 4);
    std::vector<torch::Tensor> inputs;
    for (auto &imgMat : imgMats){
        torch::Tensor imgTensor = torch::from_blob(imgMat.data, {1, imgMat.rows, imgMat.cols, imgMat.channels()}, torch::kFloat);
        
        imgTensor = imgTensor.permute({0, 3, 1, 2}).contiguous();
        inputs.push_back(imgTensor.to(torch::Device(deviceType)));
    }
    torch::TensorList tensorList{ inputs};

    auto batchTensors = torch::cat(tensorList);
    auto output = torchBackendPtr->forward({batchTensors}).toTensor();
    // output.print();
    for (auto i=0; i < imgMats.size(); i++){
        auto out = output.index({i, "..."});
        std::vector<float> v(out.data_ptr<float>(), out.data_ptr<float>() + out.numel());
        result.push_back(v);
    }
    return 0;
}
REGISTER_BACKEND_CLASS(Torch);