#include "detector2d.h"


#include <filesystem>
namespace fs = std::filesystem;

extern "C" 
{

    Net_config config;
    cv::Mat frame;
    YOLOV7* decector;

    void read_frame_buffer_from_csharp(unsigned char* bitmap, int height, int width)
    {
        frame = cv::Mat(height, width, CV_8UC4, bitmap);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::flip(frame, frame, 0);
        cv::resize(frame, frame, cv::Size(config.inpWidth, config.inpHeight));
        return ;
    }

    void load_model_config_from_csharp(char* modelpath, char* cls_names_path, int len_string, int inpHeight, int inpWidth, float confThreshold, float nmsThreshold)
    {
        config.confThreshold = confThreshold;
        config.nmsThreshold = nmsThreshold;

        config.modelpath.assign(modelpath, len_string);
        config.cls_names_path.assign(cls_names_path, len_string);

        config.inpHeight = inpHeight;
        config.inpWidth = inpWidth;
    }

    int dnn_init()
    {

        decector = new YOLOV7(config);

        return config.inpHeight;
    }
    
    void infer(float* ret_bboxes, float* ret_confidences, int* ret_classIds, int ret_len)
    {
        decector->detect(frame, ret_bboxes, ret_confidences, ret_classIds, ret_len);
    }
}