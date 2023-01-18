#ifndef TENSORRT_BACKEND_HPP
#define TENSORRT_BACKEND_HPP

#include <string>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <fstream>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include "backend.hpp"
#include "backendFactory.hpp"

void print_dims(const nvinfer1::Dims& dim)
{
	for (int nIdxShape = 0; nIdxShape < dim.nbDims; ++nIdxShape)
	{
		
		printf("dim %d=%d\n", nIdxShape, dim.d[nIdxShape]);
		
	}
}



inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR: return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO: return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
    }
}

class TRTLogger: public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if (severity <= Severity::kINFO){
            printf("%s : %s \n", severity_string(severity), msg);
        }
    }
};

// enum	Severity : int32_t {
  // Severity::kINTERNAL_ERROR = 0, 
  // Severity::kERROR = 1, 
  // Severity::kWARNING = 2, 
  // Severity::kINFO = 3,
  // Severity::kVERBOSE = 4
// }

class TensorRTBackend:public Backend{
public:
    TensorRTBackend();
    ~TensorRTBackend();
    TensorRTBackend(const std::string& modelPath, int deviceId);

    int loadModel(const std::string& modelPath, int deviceId);
    int inference(std::vector<cv::Mat>& imgMats, std::vector<std::vector<float>>& result);

private: 
    TRTLogger trtLogger;
    std::shared_ptr<nvinfer1::ICudaEngine> trtEngine;
    int32_t bindingNums;
    std::vector<char const*> bindingNames;
    std::vector<nvinfer1::Dims> bindingDims;
    int initStatus;
    int deviceId;
};



#define checkRuntime(op) _check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool _check_cuda_runtime(cudaError_t code, const std::string op, const std::string file, int line){
    if (code != cudaSuccess){
        const std::string err_name = cudaGetErrorName(code);
        const std::string err_message = cudaGetErrorString(code);
        printf("Runtime Error %s: %d %s failed. \n code = %s, message = %s \n", file.c_str(), line, op.c_str(), err_name.c_str(), err_message.c_str());
        return false;
    }
    return true;
}




template<typename _T>
std::shared_ptr<_T> make_nvshared(_T* ptr){
    return std::shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

bool file_exists(const std::string path){
    std::fstream f(path, std::ios::in);
    return f.good();
}


std::vector<float> softmax_cpu(const std::vector<float>& input_data){
    std::vector<float> res(input_data.size(), 0.0);
    std::vector<float> exps(input_data.size(), 0.0);
    float sums =0;
    for(int i = 0; i <input_data.size(); i++){
        exps[i] = exp(input_data[i]);
        sums += exps[i];
    }
    for(int i = 0; i <input_data.size(); i++){
        res[i] = exps[i] / sums;
    }

    return res;
}

std::unordered_map<int, std::string> load_label_file(const std::string &path){
    std::unordered_map<int, std::string> index2label;
    std::fstream in(path, std::ios::in);
    if (!in.is_open()){
        printf("%s open is failed \n", path);
    }
    std::string line;
    while (getline(in, line)){
        std::vector<std::string> line_item;
        std::stringstream ss(line);
        std::string tmp;
        while (getline(ss, tmp, ' ')){
            line_item.push_back(tmp);
        }
        index2label.insert({std::stoi(line_item[0]), line_item[1]});
    } 
    return index2label;

}

std::vector<unsigned char> load_engine_data(const std::string& path){
    std::ifstream in(path, std::ios::in|std::ios::binary);
    if (! in.is_open()){
        printf("%s open failed \n", path);
        return {};
    }
    in.seekg(0, std::ios::end); // 对输入文件定位，第一个参数是偏移量，第二个是基地址
    int length = in.tellg(); // 返回当前定位指针的位置，表示输入流的大小。
    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, std::ios::beg);
        data.resize(length);
        in.read((char*)&data[0], length);
    }
    in.close();
    return data;

}


#endif // TENSORRT_BACKEND_HPP
