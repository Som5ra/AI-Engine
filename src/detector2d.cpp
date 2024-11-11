#include "detector2d.h"
// #include <onnxruntime_cxx_api.h>


// #include <ctime>

// #include <filesystem>
// namespace fs = std::filesystem;

extern "C"
{
    int net_new(GenericDetector** net)
    {
        int32_t error = GustoStatus::ERR_OK;
        *net = new GenericDetector();
        return error;
    }

    int net_compile(GenericDetector* detector,
                    int inpHeight, int inpWidth,
                    float confThreshold, float nmsThreshold, 
                    char* modelpath, char* cls_names_path, int len_string = 1024
    ){
        int32_t error = GustoStatus::ERR_OK;
        int _compile_error_code = detector->compile(inpHeight, inpWidth, confThreshold, nmsThreshold, modelpath, cls_names_path, len_string);
        if (_compile_error_code != 0) {
            error = GustoStatus::ERR_GENERAL_ERROR;
        }
        return error;
    }


    float infer(GenericDetector* detector, 
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