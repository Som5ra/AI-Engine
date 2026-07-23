#include "unity_api.h"

#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#if __has_include(<opencv2/imgcodecs.hpp>)
#include <opencv2/imgcodecs.hpp>
#define AI_ENGINE_HAS_OPENCV_IMGCODECS 1
#else
#define AI_ENGINE_HAS_OPENCV_IMGCODECS 0
#endif

namespace {

template <typename Function>
CUSTOM_RET GuardUnityApi(const char* operation, Function&& function) noexcept {
    try {
        return function();
    } catch (const cv::Exception& exception) {
        std::cerr << operation << " failed: " << exception.what() << std::endl;
    } catch (const std::exception& exception) {
        std::cerr << operation << " failed: " << exception.what() << std::endl;
    } catch (...) {
        std::cerr << operation << " failed with an unknown error" << std::endl;
    }
    return CustomStatus::ERR_GENERAL_ERROR;
}

CUSTOM_RET ValidatePostProcessResult(
    const std::unique_ptr<PostProcessResult>& output,
    ResultType result_type) {
    if (!output) {
        return CustomStatus::ERR_GENERAL_ERROR;
    }

    bool valid_result = false;
    switch (result_type) {
        case ResultType::DetectorResultType:
            valid_result =
                dynamic_cast<custom_detector2d::DetectionResult*>(
                    output.get()) != nullptr;
            break;
        case ResultType::MediaPipeDetectorResultType:
            valid_result =
                dynamic_cast<custom_mp_face::MediaPipeDetectorResult*>(
                    output.get()) != nullptr;
            break;
        case ResultType::MediapipeFaceLandmarkResultType:
            valid_result =
                dynamic_cast<custom_mp_face::MediapipeFaceLandmarkResult*>(
                    output.get()) != nullptr;
            break;
        case ResultType::SegmentationResultType:
            valid_result =
                dynamic_cast<custom_humanseg::SegmentationResult*>(
                    output.get()) != nullptr;
            break;
        case ResultType::KeyPointResultType:
            valid_result =
                dynamic_cast<custom_humanpose::KeyPoint2DResult*>(
                    output.get()) != nullptr;
            break;
        default:
            return CustomStatus::ERR_GENERAL_NOT_SUPPORT;
    }

    return valid_result ? CustomStatus::ERR_OK
                        : CustomStatus::ERR_GENERAL_ERROR;
}

bool IsValidBitmap(const void* bitmap, int height, int width) {
    return bitmap != nullptr && height > 0 && width > 0;
}

}  // namespace

extern "C" {

CUSTOM_API CUSTOM_RET Custom_Model_Compile(
    BaseONNX** model_ptr,
    const char* model_path,
    const char* config_path) noexcept {
    if (model_ptr == nullptr || model_path == nullptr ||
        config_path == nullptr) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }
    *model_ptr = nullptr;

    return GuardUnityApi("Custom_Model_Compile", [&]() {
        auto parsed_config = BaseONNX::ParseConfig(model_path, config_path);
        if (!parsed_config) {
            return CustomStatus::ERR_GENERAL_SERIALIZATION;
        }

        switch (parsed_config->result_type) {
            case ResultType::DetectorResultType:
                *model_ptr =
                    new custom_detector2d::Detector(std::move(parsed_config));
                break;
            case ResultType::MediaPipeDetectorResultType:
                *model_ptr =
                    new custom_mp_face::FaceDetector(std::move(parsed_config));
                break;
            case ResultType::MediapipeFaceLandmarkResultType:
                *model_ptr =
                    new custom_mp_face::FaceLandmarker(
                        std::move(parsed_config));
                break;
            case ResultType::SegmentationResultType:
                *model_ptr =
                    new custom_humanseg::Segmenter(std::move(parsed_config));
                break;
            case ResultType::KeyPointResultType:
                *model_ptr =
                    new custom_humanpose::RTMPose(std::move(parsed_config));
                break;
            default:
                return CustomStatus::ERR_GENERAL_NOT_SUPPORT;
        }

        return *model_ptr != nullptr ? CustomStatus::ERR_OK
                                    : CustomStatus::ERR_GENERAL_ERROR;
    });
}

CUSTOM_API CUSTOM_RET Custom_Model_Inference_Image(
    BaseONNX* model_ptr,
    const char* image_path) noexcept {
    if (model_ptr == nullptr || image_path == nullptr) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

#if AI_ENGINE_HAS_OPENCV_IMGCODECS
    return GuardUnityApi("Custom_Model_Inference_Image", [&]() {
        if (!model_ptr->_config) {
            return CustomStatus::ERR_GENERAL_ERROR;
        }

        cv::Mat frame = cv::imread(image_path);
        if (frame.empty()) {
            return CustomStatus::ERR_GENERAL_IMAGE_LOAD;
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        auto output = model_ptr->forward(frame);
        return ValidatePostProcessResult(
            output, model_ptr->_config->result_type);
    });
#else
    return CustomStatus::ERR_GENERAL_NOT_SUPPORT;
#endif
}

CUSTOM_API CUSTOM_RET Custom_Model_Inference(
    BaseONNX* model_ptr,
    unsigned char* bitmap,
    int height,
    int width) noexcept {
    if (model_ptr == nullptr || !IsValidBitmap(bitmap, height, width)) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    return GuardUnityApi("Custom_Model_Inference", [&]() {
        if (!model_ptr->_config) {
            return CustomStatus::ERR_GENERAL_ERROR;
        }

        cv::Mat frame(height, width, CV_8UC4, bitmap);
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2RGB);
        cv::flip(frame, frame, 0);

        auto output = model_ptr->forward(frame);
        return ValidatePostProcessResult(
            output, model_ptr->_config->result_type);
    });
}

CUSTOM_API CUSTOM_RET Custom_Model_Destroy(BaseONNX* model_ptr) noexcept {
    delete model_ptr;
    return CustomStatus::ERR_OK;
}

CUSTOM_API CUSTOM_RET Custom_Human_Pose_Pipeline_Compile(
    HumanPoseExtractor2D** model_ptr,
    const char* detector_path,
    const char* detector_config_path,
    const char* pose_model_path,
    const char* pose_model_config_path,
    int detect_interval) noexcept {
    if (model_ptr == nullptr || detector_path == nullptr ||
        detector_config_path == nullptr || pose_model_path == nullptr ||
        pose_model_config_path == nullptr) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }
    *model_ptr = nullptr;

    return GuardUnityApi("Custom_Human_Pose_Pipeline_Compile", [&]() {
        *model_ptr = new HumanPoseExtractor2D(
            detector_path,
            detector_config_path,
            pose_model_path,
            pose_model_config_path,
            detect_interval);
        return CustomStatus::ERR_OK;
    });
}

CUSTOM_API CUSTOM_RET Custom_Human_Pose_Pipeline_Inference(
    HumanPoseExtractor2D* model_ptr,
    char* bitmap,
    int height,
    int width,
    bool display_box,
    bool display_keypoints) noexcept {
    if (model_ptr == nullptr || !IsValidBitmap(bitmap, height, width)) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    return GuardUnityApi("Custom_Human_Pose_Pipeline_Inference", [&]() {
        cv::Mat frame(height, width, CV_8UC4, bitmap);
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2RGB);
        cv::flip(frame, frame, 0);

        const CUSTOM_RET inference_status = model_ptr->DetectPose(frame);
        if (inference_status != CustomStatus::ERR_OK) {
            return inference_status;
        }

        if (display_box || display_keypoints) {
            const CUSTOM_RET display_status =
                model_ptr->Display(
                    frame, display_box, display_keypoints);
            if (display_status != CustomStatus::ERR_OK) {
                return display_status;
            }

            cv::cvtColor(frame, frame, cv::COLOR_RGB2RGBA);
            cv::flip(frame, frame, 0);
            std::memcpy(bitmap, frame.data, frame.total() * frame.elemSize());
        }

        return CustomStatus::ERR_OK;
    });
}

CUSTOM_API CUSTOM_RET Custom_Human_Pose_Pipeline_Destroy(
    HumanPoseExtractor2D* model_ptr) noexcept {
    delete model_ptr;
    return CustomStatus::ERR_OK;
}

}  // extern "C"
