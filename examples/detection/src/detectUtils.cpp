#include "detectUtils.hpp"

void DetectUtils::visualizeDetection(cv::Mat& image, std::vector<DetectionResult>& detections,
                               const std::vector<std::string>& classNames)
{
    for (const DetectionResult& detection : detections)
    {
        cv::rectangle(image, detection.box, cv::Scalar(229, 160, 21), 2);

        int x = detection.box.x;
        int y = detection.box.y;

        int conf = (int)std::round(detection.conf * 100);
        int classId = detection.classId;
        std::string text = classNames[classId] + " 0." + std::to_string(conf);

        // CV_EXPORTS_W Size getTextSize(const String& text, int fontFace,
        //                             double fontScale, int thickness,
        //                             CV_OUT int* baseLine);
        cv::Size size = cv::getTextSize(text, cv::FONT_ITALIC, 0.8, 2, nullptr);
        cv::rectangle(image,
                      cv::Point(x, y - 25), cv::Point(x + size.width, y),
                      cv::Scalar(229, 160, 21), -1);

        cv::putText(image, text,
                    cv::Point(x, y - 3), cv::FONT_ITALIC,
                    0.8, cv::Scalar(255, 255, 255), 2);
    }
}


std::vector<std::string> DetectUtils::loadNames(const std::string& path)
{
    // load class names
    std::vector<std::string> classNames;
    std::ifstream infile(path);
    if (infile.good())
    {
        std::string line;
        while (getline (infile, line))
        {
            if (line.back() == '\r')
                line.pop_back();
            classNames.emplace_back(line);
        }
        infile.close();
    }
    else
    {
        std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
    }

    return classNames;
}

float DetectUtils::iouBoxes(cv::Rect& x1, cv::Rect& x2){
    float intersectionArea = static_cast<float>((x1 & x2).area());
    float unionArea = static_cast<float> ((x1 | x2).area());
    if (intersectionArea == 0 || unionArea == 0){
        return 0.0f;
    }
    return intersectionArea / unionArea;
}




void DetectUtils::nmsCpu(std::vector<DetectionResult>& detections, const float& confThreshold, const float& iouThreshold, std::vector<DetectionResult>& nmsResult){
    
    // sorted
    std::sort(detections.begin(), detections.end(), [&](DetectionResult detection1, DetectionResult detection2){return detection1.conf > detection2.conf;});

    std::vector<bool> removeFlags(detections.size(), false);
    for (size_t i=0; i < detections.size(); i++){
        if (removeFlags[i]) continue;
        if (detections[i].conf < confThreshold) continue;
        nmsResult.push_back(detections[i]);
        for (size_t j= i+1 ;j < detections.size(); j++){
            if (removeFlags[j]) continue;
            if (detections[j].conf < confThreshold) continue;
            float iou = DetectUtils::iouBoxes(detections[i].box, detections[j].box);
            if (iou > iouThreshold){
                removeFlags[j] = true;
            }
        }
    }
}
