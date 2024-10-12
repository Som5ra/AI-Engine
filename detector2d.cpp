#include "detector2d.h"

// #include <ctime>
#include <chrono>

// #include <filesystem>
// namespace fs = std::filesystem;

extern "C" 
{
    #define ERR_OK                                                            0x00000000

    // General
    #define ERR_GENERAL_ERROR                                                 0x76000000
    #define ERR_GENERAL_FILE_IO                         -(ERR_GENERAL_ERROR | 0x00000001)
    #define ERR_GENERAL_IMAGE_LOAD                      -(ERR_GENERAL_ERROR | 0x00000002)
    #define ERR_GENERAL_SERIALIZATION                   -(ERR_GENERAL_ERROR | 0x00000003)
    #define ERR_GENERAL_INVALID_PARAMETER               -(ERR_GENERAL_ERROR | 0x00000004)
    #define ERR_GENERAL_NOT_SUPPORT                     -(ERR_GENERAL_ERROR | 0x00000005)

    int net_new(YOLO** net)
    {

        int32_t error = ERR_OK;
        *net = new YOLO();
        return error;
    }

    int net_compile(YOLO* detector,
                    int inpHeight, int inpWidth,
                    float confThreshold, float nmsThreshold, 
                    char* modelpath, char* cls_names_path, int len_string = 1024
    ){
        int32_t error = ERR_OK;
        int _compile_error_code = detector->compile(inpHeight, inpWidth, confThreshold, nmsThreshold, modelpath, cls_names_path, len_string);
        if (_compile_error_code != 0) {
            error = ERR_GENERAL_ERROR;
        }
        return error;
    }


    float infer(YOLO* detector, 
            unsigned char* bitmap, int height, int width,
            float* ret_bboxes, float* ret_confidences, int* ret_classIds, int* ret_len
    ){
        auto t_start = std::chrono::high_resolution_clock::now();

        // load and preprocess the frame
        cv::Mat frame = cv::Mat(height, width, CV_8UC4, bitmap);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::flip(frame, frame, 0);
        cv::resize(frame, frame, cv::Size(detector->config->inpWidth, detector->config->inpHeight));

        detector->detect(frame, ret_bboxes, ret_confidences, ret_classIds, ret_len);

        // the work...
        auto t_end = std::chrono::high_resolution_clock::now();

        float elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        return elapsed_time_ms;
    }
}