cmake_minimum_required(VERSION 3.5)
project(deploy_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# add_definitions(-D _GLIBCXX_USE_CXX11_ABI=0) # libtorch cxx11


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)


set(CMAKE_BUILD_TYPE Release)


set(3RDPARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party)


# # 两个必要的依赖包
set(OpenCV_DIR ${3RDPARTY}/opencv-3.4.15/share/OpenCV)
set(Torch_DIR ${3RDPARTY}/libtorch/share/cmake/Torch)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "Torch_LIBS: ${TORCH_LIBRARIES}")


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    ${3RDPARTY}/spdlog/include
    ${3RDPARTY}/onnxruntime-linux-x64-1.10.0/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/classification/include
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include/torchBackend
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include/onnxruntimeBackend
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include/common

    )

FILE(GLOB BACKEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/torchBackend/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/onnxruntimeBackend/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/common/*.cpp

    )
message(STATUS "BACKEND_SRC: ${BACKEND_SRC}")

# add_library(Backend SHARED ${BACKEND_SRC})
# target_link_libraries(
#     Backend
#     ${OpenCV_LIBS}
#     ${TORCH_LIBRARIES}
#     ${3RDPARTY}/onnxruntime-linux-x64-1.10.0/lib/libonnxruntime.so.1.10.0
# )

add_library(onnxruntimeBackend SHARED core/src/onnxruntimeBackend/onnxruntimeBackend.cpp)
target_link_libraries(
    onnxruntimeBackend
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${3RDPARTY}/onnxruntime-linux-x64-1.10.0/lib/libonnxruntime.so.1.10.0
)

add_library(torchscriptBackend SHARED core/src/torchBackend/torchBackend.cpp)
target_link_libraries(
    torchscriptBackend
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)


add_executable(cls_test main.cpp examples/classification/src/img_cls.cpp ${BACKEND_SRC})
target_link_libraries(
    cls_test
    torchscriptBackend
    onnxruntimeBackend
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${3RDPARTY}/onnxruntime-linux-x64-1.10.0/lib/libonnxruntime.so.1.10.0
)