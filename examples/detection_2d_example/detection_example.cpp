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




    // const std::string _model_path = "/media/sombrali/HDD1/weights_lib/human-pose/mediapipe/pose_detector.onnx";
    // const std::string _model_name = "pose_detector";

    // const std::string _model_path = "/media/sombrali/HDD1/mmlib/mmdetection/work_dirs/mbnv3_20241203/epoch_120/end2end_nonms_fp16.onnx";
    // const std::string _model_name = "mbnv3_20241203";
    // const std::pair<int, int> _input_size = std::make_pair(300, 300);

    // const std::string _model_path = "/media/sombrali/HDD1/mmlib/mmyolo/work_dirs/yolov5_nano_v7_default_optim_20241204/epoch_110/epoch_110_nonms_fp16.onnx";
    // const std::string _model_name = "yolov5_nano";
    // const std::pair<int, int> _input_size = std::make_pair(320, 320);


    // std::string _model_path = "/media/sombrali/HDD1/mmlib/mmyolo/work_dirs/rtmdet_tiny_disney_headband_v7_largesyn_20241027/best_coco_bbox_mAP_epoch_150/best_coco_bbox_mAP_epoch_150_nonms_fp16.onnx";
    std::string _model_path = "/media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/StreamingAssets/Weights/rtmdet_t_v7_20241028_preprocessor.onnx";
    std::string _model_name = "rtmdet-tiny";
    std::pair<int, int> _input_size = std::make_pair(320, 320);

    bool DISPLAY = true;
    if (argc >= 2) {
        if (argv[argc - 1] == std::string("no_display")){
            DISPLAY = false;
        }
        if (argc >= 5){
            _model_path.assign(argv[1]);
            _model_name.assign(argv[2]);
            _input_size = std::make_pair(std::stoi(argv[3]), std::stoi(argv[4]));
        }
    }


    std::unique_ptr<basic_model_config> config = gusto_detector2d::fetch_model_config(_model_name, _model_path, _input_size);
    std::unique_ptr<gusto_detector2d::Detector> human_detector;
    human_detector = std::move(std::make_unique<gusto_detector2d::Detector>(config));
    std::cout << "Successfully loaded model" << std::endl;
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
        // auto dets_out = human_detector->postprocess(inter_output);
        // for(size_t i = 0; i < dets_out.size(); i++){
            // std::cout << "x1: " << dets_out[i].x1 << " y1: " << dets_out[i].y1 << " x2: " << dets_out[i].x2 << " y2: " << dets_out[i].y2 << std::endl;
            // cv::rectangle(frame, cv::Point(dets_out[i].x1, dets_out[i].y1), cv::Point(dets_out[i].x2, dets_out[i].y2), cv::Scalar(0, 255, 0), 2);
        // }
        

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