#include "yolov5.hpp"
#include "simpleLogger.hpp"
#include "backendFactory.hpp"
#include "torch/script.h"
#include "detectUtils.hpp"


YoloV5Detect::YoloV5Detect(const std::string& modelPath, const std::string& backendType, int deviceId)
        :modelPath(modelPath), 
        backendType(backendType),
        deviceId(deviceId){

            SLOG_INFO("Starting Create infer backend {}.", backendType);  

            yolov5Ptr = BackendRegistry::CreateBackend(backendType);
            SLOG_INFO("Create infer backend {} done.", backendType);

            yolov5Ptr->loadModel(modelPath, deviceId);
            SLOG_INFO("loaded model {} {}.", modelPath, deviceId);
        }

int letterBox(cv::Mat& image, const cv::Size modelInputShape=cv::Size(640, 640), const cv::Scalar& color=cv::Scalar(114, 114, 114)){
    cv::Size originalImageShape = image.size();

    float r = std::min(
        static_cast<float>(modelInputShape.height) / static_cast<float>(originalImageShape.height),
        static_cast<float>(modelInputShape.width) / static_cast<float>(originalImageShape.width)
        );
    std::vector<float> ratio{r, r};
    std::vector<int> newUnpad{
        static_cast<int>(std::round(static_cast<float>(originalImageShape.width) * r)),
        static_cast<int>(std::round(static_cast<float>(originalImageShape.height) * r))
    };
    float dw = (modelInputShape.width -  newUnpad[0]) / 2.0f;
    float dh = (modelInputShape.height -  newUnpad[1]) / 2.0f;

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));
    if (newUnpad[0] != modelInputShape.width || newUnpad[1] != modelInputShape.height){
        cv::resize(image, image, cv::Size(newUnpad[0], newUnpad[1]));
    }
    cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    return 0;
}



void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                    float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }

}


void scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape)
{
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                          (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = {(int) (( (float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                  (int) (( (float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

    coords.x = (int) std::round(((float)(coords.x - pad[0]) / gain));
    coords.y = (int) std::round(((float)(coords.y - pad[1]) / gain));

    coords.width = (int) std::round(((float)coords.width / gain));
    coords.height = (int) std::round(((float)coords.height / gain));

    // // clip coords, should be modified for width and height
    // coords.x = utils::clip(coords.x, 0, imageOriginalShape.width);
    // coords.y = utils::clip(coords.y, 0, imageOriginalShape.height);
    // coords.width = utils::clip(coords.width, 0, imageOriginalShape.width);
    // coords.height = utils::clip(coords.height, 0, imageOriginalShape.height);
}


int YoloV5Detect::preprocessImage(cv::Mat& image){
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    auto re = letterBox(image, this->modelInputImageShape);
    // cv::resize(image, image, modelInputImageShape);
    image.convertTo(image, CV_32FC3, 1.0 / 255);

    return 0;
}



int YoloV5Detect::postprocessImage(
        std::vector<std::vector<float>>& modelOutput,
        const std::vector<cv::Size>& originalImageShapes,
        std::vector<std::vector<DetectionResult>>& res,
        const float& confThreshold,
        const float& iouThreshold
    ){



    for (int imgIdx = 0; imgIdx < modelOutput.size(); imgIdx++){
        std::vector<DetectionResult> detections;

        int firstLayerAnchorNum, secondLayerAnchorNum, thirdLayerAnchorNum;
        firstLayerAnchorNum = (this->modelInputImageShape.width) / 8 * (this->modelInputImageShape.height / 8) * 3;
        secondLayerAnchorNum = (this->modelInputImageShape.width) / 16 * (this->modelInputImageShape.height / 16) * 3;
        thirdLayerAnchorNum = (this->modelInputImageShape.width) / 32 * (this->modelInputImageShape.height /32) * 3;
        int numClasses = modelOutput[imgIdx].size() / (firstLayerAnchorNum + secondLayerAnchorNum + thirdLayerAnchorNum) - 5;
        int elementsOneImg = modelOutput[imgIdx].size();
        auto oneImgOutput = modelOutput[imgIdx];
        // only for batch size = 1
        for (auto it = oneImgOutput.begin(); it != oneImgOutput.begin() + elementsOneImg; it += (numClasses + 5))
        {
            float clsConf = it[4];

            if (clsConf > confThreshold)
            {
                int centerX = (int) (it[0]);
                int centerY = (int) (it[1]);
                int width = (int) (it[2]);
                int height = (int) (it[3]);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                float objConf;
                int classId;
                getBestClassInfo(it, numClasses, objConf, classId);

                float confidence = clsConf * objConf;

                DetectionResult det;
                det.box = cv::Rect({left, top, width, height});
                det.conf = confidence;
                det.classId = classId;
                detections.emplace_back(det);
            }
        }

        // cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
        // std::cout << "amount of NMS indices: " << indices.size() << std::endl;
        std::vector<DetectionResult> nmsResults;




        DetectUtils::nmsCpu(detections, confThreshold, iouThreshold, nmsResults);
        for (auto& nmsResult: nmsResults){
            // std::cout<<"ori: "<<nmsResult.box<<std::endl;
            scaleCoords(this->modelInputImageShape, nmsResult.box, originalImageShapes[imgIdx]);
            // std::cout<<"new: "<<nmsResult.box<<std::endl;
        }
        res.emplace_back(nmsResults);
    }
    return 0;
}

int YoloV5Detect::runImg(
        std::vector<std::string>& imgPaths, 
        std::vector<std::vector<DetectionResult>>& res,
        const float& confThreshold,
        const float& iouThreshold
        ){

    std::vector<cv::Mat> images;
    std::vector<cv::Size> originalImageShapes;
    std::vector<std::vector<float>> modelOutput;  
    

    for (auto imgPath:imgPaths){
        cv::Mat image = cv::imread(imgPath);
        originalImageShapes.push_back(image.size());
        auto preprocessRe = preprocessImage(image);
        images.push_back(image); 
    }

    yolov5Ptr->inference(images, modelOutput);

    std::cout<<"modelOutput: "<<modelOutput.size()<<" "<<modelOutput[0].size()<<std::endl;

    auto postprocessRe = postprocessImage(modelOutput, originalImageShapes, res, confThreshold, iouThreshold);

    // int64_t n = result.size();
    // int64_t  cols = result[0].size();    
    // auto options = torch::TensorOptions().dtype(torch::kFloat32);
    // torch::Tensor result1 = torch::zeros({ n, cols }, options);
    // for(int i =0; i< n; i++){
    //     result1.slice(0, i, i+1) = torch::from_blob(result[i].data(), {int64_t(result[0].size())}, options).clone();
    // }
    // // std::cout<<result1<<std::endl;
    // // result1.print();
    // // std::cout<<"result: "<<result<<std::endl;
    // auto output_scores = torch::softmax(result1, 1);
    // // std::cout<<output_scores<<std::endl;
    
    // std::tuple<torch::Tensor, torch::Tensor> max_result = torch::max(output_scores, 1);
    // torch::Tensor max_score = std::get<0>(max_result);
    // torch::Tensor max_index = std::get<1>(max_result).to(torch::kInt64);
    // // std::cout<<max_score<<" "<<max_index<<std::endl;
    // // tensor转为标准类型
    // std::get<1>(res) = std::vector<float> (max_score.data_ptr<float>(), max_score.data_ptr<float>() + max_score.numel());
    // std::get<0>(res) = std::vector<int64_t> (max_index.data_ptr<int64_t>(), max_index.data_ptr<int64_t>() + max_index.numel());
    return 0;
}
