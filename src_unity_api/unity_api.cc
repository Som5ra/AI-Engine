#include "unity_api.h"
extern "C"
{

    using json = nlohmann::json;

    GUSTO_RET Gusto_Model_Compile(BaseONNX** model_ptr, const char* model_path, const char* config_path)
    // GUSTO_RET Gusto_Model_Compile(std::unique_ptr<std::unique_ptr<BaseONNX>> model_ptr, const char* model_path, const char* config_path)
    {
        std::string _model_path = std::string(model_path);
        std::string _config_path = std::string(config_path);
        std::unique_ptr<basic_model_config> parsed_config = std::move(BaseONNX::ParseConfig(_model_path, _config_path));
        auto result_type = parsed_config->result_type;
        switch (result_type) {
            case ResultType::DetectorResultType: {
                *model_ptr = new gusto_detector2d::Detector(std::move(parsed_config));
                break;
            }
            case ResultType::MediaPipeDetectorResultType: {
                *model_ptr = new gusto_mp_face::FaceDetector(std::move(parsed_config));
                break;
            }
            case ResultType::MediapipeFaceLandmarkResultType: {
                *model_ptr = new gusto_mp_face::FaceLandmarker(std::move(parsed_config));
                break;
            }
            case ResultType::SegmentationResultType: {
                *model_ptr = new gusto_humanseg::Segmenter(std::move(parsed_config));
                break;
            }
            // case ResultType::KeyPointResultType: {
            //     *model_ptr = new gusto_humanpose::RTMPose(std::move(parsed_config));
            //     break;
            // }
            default:
                std::cerr << "Unknown ResultType" << std::endl;
                break;
        }
        std::cout << "model compiled" << std::endl;
        return GustoStatus::ERR_OK;
    }


    GUSTO_RET GUSTO_POST_PROCESS_RESULT(std::unique_ptr<PostProcessResult> output, ResultType result_type) {
        switch (result_type) {
            case ResultType::DetectorResultType: {
                auto* result = dynamic_cast<gusto_detector2d::DetectionResult*>(output.get());
                break;
            }
            case ResultType::MediaPipeDetectorResultType: {
                auto* result = dynamic_cast<gusto_mp_face::MediaPipeDetectorResult*>(output.get());                 
                break;
            }
            case ResultType::MediapipeFaceLandmarkResultType: {
                auto* result = dynamic_cast<gusto_mp_face::MediapipeFaceLandmarkResult*>(output.get());
                break;
            }
            case ResultType::SegmentationResultType: {
                auto* result = dynamic_cast<gusto_humanseg::SegmentationResult*>(output.get());
                break;
            }
            // case ResultType::KeyPointResultType: {
            //     auto* result = dynamic_cast<gusto_humanpose::KeyPoint2DResult*>(output.get());
            //     break;
            // }
            default:
                std::cerr << "Unknown ResultType" << std::endl;
                return GustoStatus::ERR_GENERAL_NOT_SUPPORT;                
        }
        return GustoStatus::ERR_OK;
    }

    GUSTO_RET Gusto_Model_Inference_Image(BaseONNX* model_ptr, const char* image_path)
    {
        cv::Mat frame;
        frame = cv::imread(image_path);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        auto output = model_ptr->forward(frame);
        auto ResultType = model_ptr->_config->result_type; 
        return GUSTO_POST_PROCESS_RESULT(std::move(output), ResultType);
    }

    GUSTO_RET Gusto_Model_Inference(BaseONNX* model_ptr, unsigned char* bitmap, int height, int width)
    {
        cv::Mat frame = cv::Mat(height, width, CV_8UC4, bitmap);
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2RGB);
        cv::flip(frame, frame, 0);

        auto output = model_ptr->forward(frame);
        auto ResultType = model_ptr->_config->result_type; 
        return GUSTO_POST_PROCESS_RESULT(std::move(output), ResultType);
    }



    GUSTO_RET Gusto_Human_Pose_Pipeline_Compile(
        HumanPoseExtractor2D** model_ptr,
        const char* detector_path, 
        const char* detector_config_path, 
        const char* pose_model_path, 
        const char* pose_model_config_path, 
        int detect_interval
    )
    {
        std::string _detector_path = std::string(detector_path);
        std::string _config_path = std::string(detector_config_path);
        std::string _pose_path = std::string(pose_model_path);
        std::string _pose_config_path = std::string(pose_model_config_path);
        // std::unique_ptr<HumanPoseExtractor2D> extractor(new HumanPoseExtractor2D(detector_path, detector_config_path, pose_model_path, pose_model_config_path, 3));
        *model_ptr = new HumanPoseExtractor2D(_detector_path, _config_path, _pose_path, _pose_config_path, detect_interval);
        return GustoStatus::ERR_OK;
    }

    GUSTO_RET Gusto_Human_Pose_Pipeline_Inference(HumanPoseExtractor2D* model_ptr, char* bitmap, int height, int width, bool display_box, bool display_kpts)
    {
        
        cv::Mat frame = cv::Mat(height, width, CV_8UC4, bitmap);
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2RGB);
        cv::flip(frame, frame, 0);

        auto output = model_ptr->DetectPose(frame);
        if (display_box || display_kpts){ 
            model_ptr->Display(frame, display_box, display_kpts);
            cv::cvtColor(frame, frame, cv::COLOR_RGB2RGBA);
            cv::flip(frame, frame, 0);
            memcpy(bitmap, frame.data, height * width * 4);
        }
        // auto ResultType = model_ptr->_config->result_type; 
        // return GUSTO_POST_PROCESS_RESULT(std::move(output), ResultType);
        return true;
    }





}