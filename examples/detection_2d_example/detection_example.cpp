#include "detector2d_family.h"

#include <onnxruntime_cxx_api.h>

#if defined(BUILD_PLATFORM_LINUX) && defined(DEBUG)
#include <opencv2/opencv.hpp>
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif


int main(int argc, char *argv[])
{


    bool DISPLAY = true;
    if (argc >= 2 && argv[argc - 1] == std::string("no_display")) {
        DISPLAY = false;
    }

    const std::string _model_path = "/media/sombrali/HDD1/weights_lib/human-pose/mediapipe/pose_detector.onnx";
    const std::string _model_name = "pose_detector";
    std::unique_ptr<basic_model_config> config = gusto_detector2d::fetch_model_config(_model_name, _model_path);
    std::unique_ptr<gusto_detector2d::Detector> human_detector = std::make_unique<gusto_detector2d::Detector>(config);

    float min_time = 1000000;
    float max_time = 0;
    float total_time = 0;
    int num_frames = 0;
    
    #if defined(BUILD_PLATFORM_LINUX) && defined(DEBUG)
    cv::VideoCapture cap;
    try{
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.open(0);
    }catch (const std::exception& e) {
        std::cerr << "Failed to open camera! Load Same Image Instead" << std::endl;
    }
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera! Load Same Image Instead" << std::endl;
    }
    if (DISPLAY){
        cv::namedWindow("Raw", cv::WINDOW_NORMAL);
    }
    #endif

    cv::Mat frame;

    // cv::namedWindow("cropped_face", cv::WINDOW_NORMAL);
    while (true)
    {
        #if defined(BUILD_PLATFORM_LINUX) && defined(DEBUG)
        if (!cap.isOpened()){ 
            frame = cv::imread("demo.png");            
        }else{
            cap >> frame; // so fkng slow
        }
        #else
        frame = cv::imread("demo.png");
        #endif
        auto start = std::chrono::high_resolution_clock::now();
        auto inter_output = human_detector->forward(frame);
        auto dets_out = human_detector->postprocess(inter_output);
        for(size_t i = 0; i < dets_out.size(); i++){
            std::cout << "x1: " << dets_out[i].x1 << " y1: " << dets_out[i].y1 << " x2: " << dets_out[i].x2 << " y2: " << dets_out[i].y2 << std::endl;
            cv::rectangle(frame, cv::Point(dets_out[i].x1, dets_out[i].y1), cv::Point(dets_out[i].x2, dets_out[i].y2), cv::Scalar(0, 255, 0), 2);
        }
        

        auto end = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        min_time = std::min(min_time, duration);
        max_time = std::max(max_time, duration);
        total_time += duration;
        num_frames++;
        std::cout << "\rmin_time: " << min_time << "ms  |  max_time: " << max_time << "ms  |  avg_time: " << std::ceil(total_time / num_frames * 100) / 100 << "ms " << std::flush;    

        if (DISPLAY){
            // auto painted_image = face_detector.draw_boxes(frame, boxes, scores, indices, indices_cls);
            cv::imshow("Raw", frame);
            if (cv::waitKey(25) >= 0)
                break;
        }


    }
    


    return 0;
}