#include "human_seg_family.h"

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

    std::string segmenter_name = "selfie_multiclass_256x256";
    if (argc >= 2 && argv[argc - 1] == std::string("selfie_segmenter")) {
        segmenter_name = "selfie_segmenter";
    }

    // const std::string segmenter_name = "selfie_segmenter_landscape";
    // const std::string segmenter_name = "deeplab_v3";
    std::unique_ptr<gusto_humanseg::seg_config> config = gusto_humanseg::fetch_model_config(segmenter_name);
    std::unique_ptr<gusto_humanseg::Segmenter> human_segmenter = std::make_unique<gusto_humanseg::Segmenter>(config);

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
        cv::namedWindow("Mask", cv::WINDOW_NORMAL);
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
        auto inter_output = human_segmenter->forward(frame);
        auto mask_out = human_segmenter->postprocess(inter_output, std::make_pair(frame.rows, frame.cols));

        

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
            cv::imshow("Mask", mask_out);
            if (cv::waitKey(25) >= 0)
                break;
        }


    }
    


    return 0;
}