#include "detector2d.h"
#include <iostream>
#include <vector>
int main()
{
    GenericDetector* detector = new GenericDetector();

    const std::string modelpath = "/media/sombrali/HDD1/opencv-unity/gusto_dnn/weights/yolov7-tiny-20240821-3cls2.onnx";
    const std::string cls_names_path = "/media/sombrali/HDD1/opencv-unity/gusto_dnn/weights/cls_names.names";
    int _compile_error_code = detector->compile(640, 640, 0.5, 0.5, modelpath.c_str(), cls_names_path.c_str(), 1024);
    std::cout << "compile error code: " << _compile_error_code << std::endl;    

    // using opencv to call webcam
    cv::VideoCapture cap(1);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    float* ret_bboxes = new float[1000];
    float* ret_confidences = new float[1000];
    int* ret_classIds = new int[1000];
    int* ret_len = new int;

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        detector->detect(frame, ret_bboxes, ret_confidences, ret_classIds, ret_len);
        if (frame.empty())
            break;
        cv::imshow("Frame", frame);
        if (cv::waitKey(10) == 27)
            break;
    }
    

    return 0;
}