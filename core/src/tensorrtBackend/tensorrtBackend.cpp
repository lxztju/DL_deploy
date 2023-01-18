#include "tensorrtBackend.hpp"
#include "simpleLogger.hpp"

using namespace std;

TensorRTBackend::TensorRTBackend(){}
TensorRTBackend::~TensorRTBackend(){}


// TensorRTBackend(const std::string& modelPath, int deviceId);

int TensorRTBackend::loadModel(const std::string& modelPath, int deviceId){

    auto engineData = load_engine_data(modelPath);
    SLOG_INFO("engine size: {}", engineData.size());
    // cout<<"engine size: "<<engine_data.size()<<endl;
    auto runtime = make_nvshared(nvinfer1::createInferRuntime(trtLogger));
    trtEngine = make_nvshared(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if(trtEngine == nullptr){
        SLOG_INFO("Deserialize cuda engine failed.\n");
        return -1;
    }

    this->bindingNums = trtEngine->getNbBindings();
    SLOG_INFO("the binding name is {}.\n", this->bindingNums);
    char const* bindingName;
    nvinfer1::Dims bindingDim;
    for (auto i = 0; i < this->bindingNums; i++){
        bindingName = trtEngine->getBindingName(i);
        bindingDim = trtEngine->getBindingDimensions(i);
        //print_dims(bindingDim);
        this->bindingNames.push_back(bindingName);
        this->bindingDims.push_back(bindingDim);
    }
    SLOG_INFO("Deserialize cuda engine success.\n");
}


int TensorRTBackend::inference(std::vector<cv::Mat>& imgMats, std::vector<std::vector<float>>& result){
    assert(imgMats.size() < 4);
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(trtEngine->createExecutionContext());
 
    nvinfer1::Dims inputDim = bindingDims[0];
    inputDim.d[0] = imgMats.size();
    //print_dims(inputDim);
    int inputNumel = inputDim.d[0] * inputDim.d[1] * inputDim.d[2] * inputDim.d[3];
    SLOG_INFO("input Numel : {}", inputNumel);

    float* inputDataHostPtr = nullptr;
    float* inputDataDevicePtr = nullptr;
    checkRuntime(cudaMallocHost(&inputDataHostPtr, inputNumel * sizeof(float)));
    checkRuntime(cudaMalloc(&inputDataDevicePtr, inputNumel * sizeof(float)));
    
    int imageArea = imgMats[0].cols * imgMats[0].rows;
 
    float* dataPtr = inputDataHostPtr;
    for (auto& imgMat : imgMats){

        float* pimage = (float*)imgMat.data;
        float* phost_r = (float*)dataPtr+ imageArea * 0;
        float* phost_g = (float*)dataPtr + imageArea * 1;
        float* phost_b = (float*)dataPtr + imageArea * 2;
        // hwc2chw
        for(int i = 0; i < imageArea; ++i, pimage += 3){
            *phost_r++ = pimage[0];

            *phost_g++ = pimage[1];
            *phost_b++ = pimage[2];
        }
        dataPtr += imageArea * 3;
    }
        checkRuntime(cudaMemcpyAsync(inputDataDevicePtr, inputDataHostPtr, inputNumel * sizeof(float), cudaMemcpyHostToDevice, stream));

        const int outputDim = bindingDims[1].d[1] * inputDim.d[0];
        float outputDataHost[outputDim];
        float* outputDataDevicePtr = nullptr;
        checkRuntime(cudaMalloc(&outputDataDevicePtr, sizeof(outputDataHost)));
        // nvinfer1::Dims contextInputDims = execution_context->getBindingDimensions(0);
        //print_dims(contextInputDims );
        execution_context->setBindingDimensions(0, inputDim);
        float* bindings[] = {inputDataDevicePtr, outputDataDevicePtr};
        bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
        SLOG_INFO("inference success.");
        checkRuntime(cudaMemcpyAsync(outputDataHost, outputDataDevicePtr, sizeof(outputDataHost), cudaMemcpyDeviceToHost, stream));
        checkRuntime(cudaStreamSynchronize(stream));
        for (int imgIdx = 0; imgIdx < inputDim.d[0]; imgIdx++){
            std::vector<float> outputDataHostVector;
            for(int j =0; j< bindingDims[1].d[1]; j++){
                outputDataHostVector.push_back(outputDataHost[imgIdx * bindingDims[1].d[1] + j]);
            }
            
        result.push_back(outputDataHostVector);
        }

        checkRuntime(cudaStreamDestroy(stream));
        checkRuntime(cudaFreeHost(inputDataHostPtr));
        checkRuntime(cudaFree(inputDataDevicePtr));
        checkRuntime(cudaFree(outputDataDevicePtr));  
    return 0;
}
REGISTER_BACKEND_CLASS(TensorRT)



