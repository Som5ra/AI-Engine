#include "human_pose_family.h"

#include <onnxruntime_cxx_api.h>

#if defined(BUILD_PLATFORM_LINUX) && defined(DEBUG)
#include <opencv2/opencv.hpp>
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

int main(int argc, char *argv[])
{


    bool DISPLAY = true;
    if (argc >= 2 && argv[argc - 1] == std::string("no_display")) {
        DISPLAY = false;
    }

    std::string pose_detector_name = "rtmo-s";

    // const std::string segmenter_name = "selfie_segmenter_landscape";
    // const std::string segmenter_name = "deeplab_v3";
    std::unique_ptr<gusto_humanpose::humanpose_config> config = gusto_humanpose::fetch_model_config(pose_detector_name);
    std::unique_ptr<gusto_humanpose::PoseDetector> human_pose_detector = std::make_unique<gusto_humanpose::PoseDetector>(config);

    float min_time = 1000000;
    float max_time = 0;
    float total_time = 0;
    int num_frames = 0;
    
    #if defined(BUILD_PLATFORM_LINUX) && defined(DEBUG)
    // VideoCapture cap(0);
    cv::VideoCapture cap;
    try{
        // cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);
        // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        // cap.set(cv::CAP_PROP_GSTREAMER_QUEUE_LENGTH, 1);
        cap.open(0);
    }catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Failed to open camera! Load Same Image Instead" << std::endl;
    }
    if (!cap.isOpened()) {
        std::cerr << "cap.isOpened() Failed! Load Same Image Instead" << std::endl;
    }
    if (DISPLAY){
        cv::namedWindow("fb", cv::WINDOW_NORMAL);
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
            // frame = cv::imread("demo.png");
        }
        #else
        frame = cv::imread("demo.png");
        #endif
        auto start = std::chrono::high_resolution_clock::now();
        // auto inter_image = human_pose_detector->Debug_Preprocess(frame);
        auto inter_output = human_pose_detector->forward(frame);
        auto all_person_kpts = human_pose_detector->postprocess(inter_output, 0.5);
        for (auto single_person_kpts : all_person_kpts){
            frame = human_pose_detector->draw_single_person_keypoints(frame, single_person_kpts);
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
            cv::imshow("fb", frame);
            if (cv::waitKey(25) >= 0)
                break;
        }


    }
    


    return 0;
}