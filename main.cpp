#include "simpleLogger.hpp"
#include "img_cls.hpp"
#include <chrono>

using namespace std;

int main(int argc, const char** argv)
{
	std::vector<std::string>imgPaths;
    std::string img_path = "./workspace/classification/demo.jpg";
    imgPaths.push_back(img_path);
	imgPaths.push_back(img_path);
    std::string model_path = "./workspace/classification/resnet18.torchscript";
	int deiviceId = -1;
	std::string backendType = "Torch";
    ImgCls imgcls(model_path,backendType, deiviceId);
	std::tuple<std::vector<int64_t>, std::vector<float>> res;

	auto startTime = std::chrono::high_resolution_clock::now();
	for (auto i= 0; i <10; i++){
		if (-1 < imgcls.runImg(imgPaths, res)){
			// std::cout<<std::get<0>(res)<<" "<<std::get<1>(res)<<std::endl;;
		}
	}
	auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = endTime - startTime;
	std::cout<<"Torch time: " << fp_ms.count()/1000<<std::endl;


	backendType = "Onnxruntime";
	model_path = "./workspace/classification/resnet18.onnx";
    ImgCls imgcls1(model_path,backendType, deiviceId);
	startTime = std::chrono::high_resolution_clock::now();

	for (auto i= 0; i <10; i++){
		if (-1 < imgcls1.runImg(imgPaths, res)){
			// std::cout<<std::get<0>(res)<<" "<<std::get<1>(res)<<std::endl;;
		}
	}
	endTime = std::chrono::high_resolution_clock::now();
    fp_ms = endTime - startTime;
	std::cout<<"Onnx time: " << fp_ms.count() /1000<<std::endl;
	return 0;
}