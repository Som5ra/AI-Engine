#include "detector2d_family.h"

#include <onnxruntime_cxx_api.h>

#if defined(BUILD_PLATFORM_LINUX) && defined(DEBUG)
#include <opencv2/opencv.hpp>
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif


extern "C"{


    float min_time = 1000000;
    float max_time = 0;
    float total_time = 0;
    int num_frames = 0;
    
    

    std::unique_ptr<gusto_detector2d::Detector> human_detector;
    cv::Mat frame;

    void Open_Session(
        char* _model_path, 
        int input_w, 
        int input_h
    ){
        try{
            std::unique_ptr<basic_model_config> config = gusto_detector2d::fetch_model_config("Generic Detector", _model_path, std::make_pair(input_w, input_h));
            human_detector = std::move(std::make_unique<gusto_detector2d::Detector>(config));
            std::cout << "Successfully loaded model" << std::endl;
        }catch(const std::exception& e){
            std::cerr << "Failed to open Geometry Pipeline Metadata!" << std::endl;
            std::cerr << e.what() << '\n';
        }
        return ;
    }

    void Start_Session(char* _frame_path){
        std::string frame_path;
        frame_path.assign(_frame_path);
        std::cout << "Processing Frame: " << frame_path << std::endl;
        frame = cv::imread(frame_path);
        auto start = std::chrono::high_resolution_clock::now();
        auto inter_output = human_detector->forward(frame);
        auto dets_out = human_detector->postprocess(inter_output);

        auto end = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        min_time = std::min(min_time, duration);
        max_time = std::max(max_time, duration);
        total_time += duration;
        num_frames++;
        std::cout << "\rmin_time: " << min_time << "ms  |  max_time: " << max_time << "ms  |  avg_time: " << std::ceil(total_time / num_frames * 100) / 100 << "ms " << std::flush;    
        return ;
    }

}