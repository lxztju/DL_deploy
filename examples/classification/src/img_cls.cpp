#include "img_cls.hpp"
#include "simpleLogger.hpp"
#include "backendFactory.hpp"
// #include "torchBackend.hpp"
#include "torch/script.h"

ImgCls::ImgCls(const std::string& modelPath, const std::string& backendType, int deviceId)
        :modelPath(modelPath), 
        backendType(backendType),
        deviceId(deviceId){

            SLOG_INFO("Starting Create infer backend {}.", backendType);  

            clsPtr = BackendRegistry::CreateBackend(backendType);
            SLOG_INFO("Create infer backend {} done.", backendType);

            clsPtr->loadModel(modelPath, deviceId);
            SLOG_INFO("loaded model {} {}.", modelPath, deviceId);
        }



int ImgCls::runImg(std::vector<std::string>& imgPaths, std::tuple<std::vector<int64_t>, std::vector<float>>& res){

    std::vector<cv::Mat> images;
    std::vector<std::vector<float>> result;  
    for (auto imgPath:imgPaths){
        cv::Mat image = cv::imread(imgPath);

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        cv::resize(image, image, cv::Size(clsWidth, clsHeight));
        image.convertTo(image, CV_32FC3, 1.0 / 255);

        // hwc-> chw
        std::vector<cv::Mat> splitImg(image.channels());
        cv::split(image, splitImg);

        //normalize
        for (int i = 0; i < image.channels(); ++i)
        {
            splitImg[i].convertTo(splitImg[i], CV_32FC1, 1.0 / std_[i], (0.0 - mean_[i]) / std_[i]);

        }
        // hwc
        cv::merge(splitImg, image);
        images.push_back(image); 
    }


    clsPtr->inference(images, result);

    int64_t n = result.size();
    int64_t  cols = result[0].size();    
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor result1 = torch::zeros({ n, cols }, options);
    for(int i =0; i< n; i++){
        result1.slice(0, i, i+1) = torch::from_blob(result[i].data(), {int64_t(result[0].size())}, options).clone();
    }
    // std::cout<<result1<<std::endl;
    // result1.print();
    // std::cout<<"result: "<<result<<std::endl;
    auto output_scores = torch::softmax(result1, 1);
    // std::cout<<output_scores<<std::endl;
    
    std::tuple<torch::Tensor, torch::Tensor> max_result = torch::max(output_scores, 1);
    torch::Tensor max_score = std::get<0>(max_result);
    torch::Tensor max_index = std::get<1>(max_result).to(torch::kInt64);
    // std::cout<<max_score<<" "<<max_index<<std::endl;
    // tensor转为标准类型
    std::get<1>(res) = std::vector<float> (max_score.data_ptr<float>(), max_score.data_ptr<float>() + max_score.numel());
    std::get<0>(res) = std::vector<int64_t> (max_index.data_ptr<int64_t>(), max_index.data_ptr<int64_t>() + max_index.numel());
    return 0;
}
