#include "wasm_api.h"


HumanPoseExtractor2D* Gusto_Human_Pose_Pipeline_Compile(
    std::string detector_path, 
    std::string detector_config_path, 
    std::string pose_model_path, 
    std::string pose_model_config_path, 
    int detect_interval
)
{
    HumanPoseExtractor2D* model_ptr = new HumanPoseExtractor2D(detector_path, detector_config_path, pose_model_path, pose_model_config_path, detect_interval);
    std::cout << "model compiled" << std::endl;
    // model_ptr->Debug();
    // std::cout << model_ptr->human_detector->_config->model_path << std::endl;
    return model_ptr;
}

GUSTO_RET Gusto_Human_Pose_Pipeline_Inference(HumanPoseExtractor2D* model_ptr, intptr_t bitmap, int height, int width, bool display_box, bool display_kpts)
{
    cv::Mat frame(height, width, CV_8UC4, reinterpret_cast<void *>(bitmap));
    cv::cvtColor(frame, frame, cv::COLOR_RGBA2RGB);
    auto start_time = std::chrono::steady_clock::now();
    auto output = model_ptr->DetectPose(frame);
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    // std::cout << "Inference time: " << duration.count() << "ms" << std::endl;
    if (display_box || display_kpts){ 
        model_ptr->Display(frame, display_box, display_kpts);
        cv::putText(frame, "Cost: " + std::to_string(duration.count()) + "ms", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);   
        cv::cvtColor(frame, frame, cv::COLOR_RGB2RGBA);
        std::memcpy(reinterpret_cast<void *>(bitmap), frame.data, height * width * 4);
    }
    return 1;
}



EMSCRIPTEN_BINDINGS(my_module) {
    class_<HumanPoseExtractor2D>("HumanPoseExtractor2D")
        .constructor<const std::string&, const std::string&, const std::string&, const std::string&, int>();
        // .function("DetectPose", &HumanPoseExtractor2D::DetectPose)
        // .function("Display", &HumanPoseExtractor2D::Display);
    function("Gusto_Human_Pose_Pipeline_Compile", &Gusto_Human_Pose_Pipeline_Compile, allow_raw_pointers());
    function("Gusto_Human_Pose_Pipeline_Inference", &Gusto_Human_Pose_Pipeline_Inference, allow_raw_pointers());
    // class_<BaseONNX>("BaseONNX");
    // function("createC", &createC, return_value_policy::take_ownership());
    // function("init", &init, allow_raw_pointers());
    // function("check_config", &check_config, allow_raw_pointers());
    // function("LoadImage", &LoadImage
}
