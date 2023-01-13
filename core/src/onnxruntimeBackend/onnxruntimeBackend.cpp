#include "onnxruntimeBackend.hpp"
#include "simpleLogger.hpp"
#include <stdexcept>

OnnxruntimeBackend::OnnxruntimeBackend(){}

OnnxruntimeBackend::~OnnxruntimeBackend(){}


// OnnxruntimeBackend::OnnxruntimeBackend(const std::string& modelPath, int deviceId){
//     initStatus = this->loadModel(modelPath, deviceId);
// }


int OnnxruntimeBackend::loadModel(const std::string& modelPath, int deviceId){
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_infer");
    sessionOptions = Ort::SessionOptions();
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (deviceId < 0)
    {
        SLOG_INFO("Inference device: CPU");

    }
    else if ((cudaAvailable == availableProviders.end()))
    {
        SLOG_INFO( "GPU is not supported by your ONNXRuntime build. Fallback to CPU." );

    }
    else
    {
        SLOG_INFO("Inference device: GPU");
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }

    OnnxruntuimeBackendPtr = std::make_shared<Ort::Session>(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_input_nodes = OnnxruntuimeBackendPtr->GetInputCount();
    if (num_input_nodes != 1){
        SLOG_ERROR("Input only support one input node.");
        throw std::runtime_error("Input only support one input node.");
    }
    for (auto i =0; i< num_input_nodes; i++){
        Ort::TypeInfo inputTypeInfo = OnnxruntuimeBackendPtr->GetInputTypeInfo(i);
        std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        inputShapes.push_back(inputTensorShape);
        inputNodeNames.push_back(OnnxruntuimeBackendPtr->GetInputName(i, allocator));
    }
    SLOG_INFO("num_input_nodes: {}", num_input_nodes);

    size_t num_output_nodes = OnnxruntuimeBackendPtr->GetOutputCount();
    for (auto i =0; i< num_output_nodes; i++){
        Ort::TypeInfo OutputTypeInfo = OnnxruntuimeBackendPtr->GetOutputTypeInfo(i);
        std::vector<int64_t> OutputTensorShape = OutputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        outputShapes.push_back(OutputTensorShape);

        outputNodeNames.push_back(OnnxruntuimeBackendPtr->GetOutputName(i, allocator));
    }
    SLOG_INFO("num_output_nodes: {}", num_output_nodes);
    return 0;
}



int OnnxruntimeBackend::inference(std::vector<cv::Mat>& imgMats, std::vector<std::vector<float>>& result){
    if(0==imgMats.size()){
        return -1;
    }
    // SLOG_INFO("input images number: {}", imgMats.size());
    assert(imgMats.size() <= 4);
    inputShapes[0][0] = imgMats.size();
    const std::vector<int64_t> inputShape = inputShapes[0];
    size_t inputNumel = 1;
    for(auto i = 0; i <inputShape.size(); i ++ ){
        inputNumel *= inputShape[i];
    }
    // SLOG_INFO("inputNumel : {}", inputNumel);
    std::vector<float> inputDataHost(inputNumel);
    float* inputDataHostPtr = inputDataHost.data();
    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    int image_area = imgMats[0].cols * imgMats[0].rows;
    int oneImgLength = inputNumel / (imgMats.size());
    std::vector<int64_t> oneInputShape{1, inputShape[1], inputShape[2], inputShape[3]};
    for (auto& imgMat : imgMats){
        std::vector<cv::Mat> chw(imgMat.channels());
        for (int i = 0; i < imgMat.channels(); ++i)
        {
            chw[i] = cv::Mat(cv::Size{imgMat.cols, imgMat.rows}, CV_32FC1, inputDataHostPtr + i * image_area);
        }
        cv::split(imgMat, chw);


        // static Value Ort::Value::CreateTensor	(	const OrtMemoryInfo * 	info,
        //     T * 	p_data,
        //     size_t 	p_data_element_count,
        //     const int64_t * 	shape,
        //     size_t 	shape_len 
        //     )	

        auto inputTensor = Ort::Value::CreateTensor(memoryInfo, inputDataHost.data(), inputDataHost.size(), inputShape.data(), inputShape.size());
        inputTensors.push_back(std::move(inputTensor));
        inputDataHostPtr += imgMat.channels() * image_area;
    }

    Ort::RunOptions options{nullptr};
        // std::vector< Value > Ort::Session::Run	(	const RunOptions & 	run_options,
        // const char *const * 	input_names,
        // const Value * 	input_values,
        // size_t 	input_count,
        // const char *const * 	output_names,
        // size_t 	output_count 
        // )	
    std::vector<Ort::Value> outputTensors = OnnxruntuimeBackendPtr->Run(options, 
        inputNodeNames.data(), inputTensors.data(), inputNodeNames.size(), 
        outputNodeNames.data(), outputNodeNames.size()
    );
    // std::cout<<"outshape: "<<outputTensors.size()<<std::endl;
    
    auto outputNodeNums =outputNodeNames.size();
    result = std::vector<std::vector<float>>(imgMats.size(), std::vector<float>());
    for (auto imgIndex=0; imgIndex < imgMats.size(); imgIndex ++){
    
        for (auto i = 0; i < outputNodeNums; i ++){
            auto* rawOutput = outputTensors[i].GetTensorData<float>();  //return a pointer
            size_t count = outputTensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
            size_t oneImageOutputLength = count / (imgMats.size());
            std::vector<float> output(rawOutput, rawOutput + count);
            auto outIterationBegin = output.begin();
            // std::cout<<oneImageOutputLength<<std::endl;
            // for (auto j = oneImageOutputLength * imgIndex; j < oneImageOutputLength* (imgIndex+1); j++ ){
            //     result[imgIndex].push_back(*(outIterationBegin+ j));
            //     std::cout<<*(outIterationBegin+ j)<<" ";
            // }
            result[imgIndex] = std::vector<float>(outIterationBegin, outIterationBegin + oneImageOutputLength);
            // for (auto j=0; j < result[imgIndex].size(); j++){
            //     std::cout<<result[imgIndex][j]<<" ";
            // }
            // std::cout<<std::endl;
            outIterationBegin += oneImageOutputLength;
        }
    }
    // for (auto i=0; i < ll.size(); i++){
    //     std::cout<<ll[i]<<" ";
    // }
    // std::cout<<std::endl;
    // std::cout<<ll.size()<<std::endl;
    return 0;
}
REGISTER_BACKEND_CLASS(Onnxruntime);