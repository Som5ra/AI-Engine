#include "unity_api.h"

#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <opencv2/imgcodecs.hpp>

namespace {

template <typename Function>
GUSTO_RET GuardUnityApi(const char* operation, Function&& function) noexcept {
    try {
        return function();
    } catch (const cv::Exception& exception) {
        std::cerr << operation << " failed: " << exception.what() << std::endl;
    } catch (const std::exception& exception) {
        std::cerr << operation << " failed: " << exception.what() << std::endl;
    } catch (...) {
        std::cerr << operation << " failed with an unknown error" << std::endl;
    }
    return GustoStatus::ERR_GENERAL_ERROR;
}

GUSTO_RET ValidatePostProcessResult(
    const std::unique_ptr<PostProcessResult>& output,
    ResultType result_type) {
    if (!output) {
        return GustoStatus::ERR_GENERAL_ERROR;
    }

    bool valid_result = false;
    switch (result_type) {
        case ResultType::DetectorResultType:
            valid_result =
                dynamic_cast<gusto_detector2d::DetectionResult*>(
                    output.get()) != nullptr;
            break;
        case ResultType::MediaPipeDetectorResultType:
            valid_result =
                dynamic_cast<gusto_mp_face::MediaPipeDetectorResult*>(
                    output.get()) != nullptr;
            break;
        case ResultType::MediapipeFaceLandmarkResultType:
            valid_result =
                dynamic_cast<gusto_mp_face::MediapipeFaceLandmarkResult*>(
                    output.get()) != nullptr;
            break;
        case ResultType::SegmentationResultType:
            valid_result =
                dynamic_cast<gusto_humanseg::SegmentationResult*>(
                    output.get()) != nullptr;
            break;
        case ResultType::KeyPointResultType:
            valid_result =
                dynamic_cast<gusto_humanpose::KeyPoint2DResult*>(
                    output.get()) != nullptr;
            break;
        default:
            return GustoStatus::ERR_GENERAL_NOT_SUPPORT;
    }

    return valid_result ? GustoStatus::ERR_OK
                        : GustoStatus::ERR_GENERAL_ERROR;
}

bool IsValidBitmap(const void* bitmap, int height, int width) {
    return bitmap != nullptr && height > 0 && width > 0;
}

}  // namespace

extern "C" {

GUSTO_API GUSTO_RET Gusto_Model_Compile(
    BaseONNX** model_ptr,
    const char* model_path,
    const char* config_path) noexcept {
    if (model_ptr == nullptr || model_path == nullptr ||
        config_path == nullptr) {
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }
    *model_ptr = nullptr;

    return GuardUnityApi("Gusto_Model_Compile", [&]() {
        auto parsed_config = BaseONNX::ParseConfig(model_path, config_path);
        if (!parsed_config) {
            return GustoStatus::ERR_GENERAL_SERIALIZATION;
        }

        switch (parsed_config->result_type) {
            case ResultType::DetectorResultType:
                *model_ptr =
                    new gusto_detector2d::Detector(std::move(parsed_config));
                break;
            case ResultType::MediaPipeDetectorResultType:
                *model_ptr =
                    new gusto_mp_face::FaceDetector(std::move(parsed_config));
                break;
            case ResultType::MediapipeFaceLandmarkResultType:
                *model_ptr =
                    new gusto_mp_face::FaceLandmarker(
                        std::move(parsed_config));
                break;
            case ResultType::SegmentationResultType:
                *model_ptr =
                    new gusto_humanseg::Segmenter(std::move(parsed_config));
                break;
            case ResultType::KeyPointResultType:
                *model_ptr =
                    new gusto_humanpose::RTMPose(std::move(parsed_config));
                break;
            default:
                return GustoStatus::ERR_GENERAL_NOT_SUPPORT;
        }

        return *model_ptr != nullptr ? GustoStatus::ERR_OK
                                    : GustoStatus::ERR_GENERAL_ERROR;
    });
}

GUSTO_API GUSTO_RET Gusto_Model_Inference_Image(
    BaseONNX* model_ptr,
    const char* image_path) noexcept {
    if (model_ptr == nullptr || image_path == nullptr) {
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    return GuardUnityApi("Gusto_Model_Inference_Image", [&]() {
        if (!model_ptr->_config) {
            return GustoStatus::ERR_GENERAL_ERROR;
        }

        cv::Mat frame = cv::imread(image_path);
        if (frame.empty()) {
            return GustoStatus::ERR_GENERAL_IMAGE_LOAD;
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        auto output = model_ptr->forward(frame);
        return ValidatePostProcessResult(
            output, model_ptr->_config->result_type);
    });
}

GUSTO_API GUSTO_RET Gusto_Model_Inference(
    BaseONNX* model_ptr,
    unsigned char* bitmap,
    int height,
    int width) noexcept {
    if (model_ptr == nullptr || !IsValidBitmap(bitmap, height, width)) {
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    return GuardUnityApi("Gusto_Model_Inference", [&]() {
        if (!model_ptr->_config) {
            return GustoStatus::ERR_GENERAL_ERROR;
        }

        cv::Mat frame(height, width, CV_8UC4, bitmap);
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2RGB);
        cv::flip(frame, frame, 0);

        auto output = model_ptr->forward(frame);
        return ValidatePostProcessResult(
            output, model_ptr->_config->result_type);
    });
}

GUSTO_API GUSTO_RET Gusto_Model_Destroy(BaseONNX* model_ptr) noexcept {
    delete model_ptr;
    return GustoStatus::ERR_OK;
}

GUSTO_API GUSTO_RET Gusto_Human_Pose_Pipeline_Compile(
    HumanPoseExtractor2D** model_ptr,
    const char* detector_path,
    const char* detector_config_path,
    const char* pose_model_path,
    const char* pose_model_config_path,
    int detect_interval) noexcept {
    if (model_ptr == nullptr || detector_path == nullptr ||
        detector_config_path == nullptr || pose_model_path == nullptr ||
        pose_model_config_path == nullptr) {
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }
    *model_ptr = nullptr;

    return GuardUnityApi("Gusto_Human_Pose_Pipeline_Compile", [&]() {
        *model_ptr = new HumanPoseExtractor2D(
            detector_path,
            detector_config_path,
            pose_model_path,
            pose_model_config_path,
            detect_interval);
        return GustoStatus::ERR_OK;
    });
}

GUSTO_API GUSTO_RET Gusto_Human_Pose_Pipeline_Inference(
    HumanPoseExtractor2D* model_ptr,
    char* bitmap,
    int height,
    int width,
    bool display_box,
    bool display_keypoints) noexcept {
    if (model_ptr == nullptr || !IsValidBitmap(bitmap, height, width)) {
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    return GuardUnityApi("Gusto_Human_Pose_Pipeline_Inference", [&]() {
        cv::Mat frame(height, width, CV_8UC4, bitmap);
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2RGB);
        cv::flip(frame, frame, 0);

        const GUSTO_RET inference_status = model_ptr->DetectPose(frame);
        if (inference_status != GustoStatus::ERR_OK) {
            return inference_status;
        }

        if (display_box || display_keypoints) {
            const GUSTO_RET display_status =
                model_ptr->Display(
                    frame, display_box, display_keypoints);
            if (display_status != GustoStatus::ERR_OK) {
                return display_status;
            }

            cv::cvtColor(frame, frame, cv::COLOR_RGB2RGBA);
            cv::flip(frame, frame, 0);
            std::memcpy(bitmap, frame.data, frame.total() * frame.elemSize());
        }

        return GustoStatus::ERR_OK;
    });
}

GUSTO_API GUSTO_RET Gusto_Human_Pose_Pipeline_Destroy(
    HumanPoseExtractor2D* model_ptr) noexcept {
    delete model_ptr;
    return GustoStatus::ERR_OK;
}

}  // extern "C"
