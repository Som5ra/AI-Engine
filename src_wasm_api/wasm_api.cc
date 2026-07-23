#include "wasm_api.h"

#include <chrono>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

namespace {

template <typename Function>
CUSTOM_RET GuardWasmApi(const char* operation, Function&& function) noexcept {
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

template <typename Model, typename Function>
Model* GuardWasmFactory(const char* operation, Function&& function) noexcept {
    try {
        return function();
    } catch (const std::exception& exception) {
        std::cerr << operation << " failed: " << exception.what() << std::endl;
    } catch (...) {
        std::cerr << operation << " failed with an unknown error" << std::endl;
    }
    return nullptr;
}

bool IsValidBitmap(std::intptr_t bitmap, int height, int width) {
    return bitmap != 0 && height > 0 && width > 0;
}

}  // namespace

HumanPoseExtractor2D* Custom_Human_Pose_Pipeline_Compile(
    const std::string& detector_path,
    const std::string& detector_config_path,
    const std::string& pose_model_path,
    const std::string& pose_model_config_path,
    int detect_interval) noexcept {
    return GuardWasmFactory<HumanPoseExtractor2D>(
        "Custom_Human_Pose_Pipeline_Compile", [&]() {
            auto model = std::make_unique<HumanPoseExtractor2D>(
                detector_path,
                detector_config_path,
                pose_model_path,
                pose_model_config_path,
                detect_interval);
            return model.release();
        });
}

CUSTOM_RET Custom_Human_Pose_Pipeline_Inference(
    HumanPoseExtractor2D* model_ptr,
    std::intptr_t bitmap,
    int height,
    int width,
    bool display_box,
    bool display_keypoints) noexcept {
    if (model_ptr == nullptr || !IsValidBitmap(bitmap, height, width)) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    return GuardWasmApi("Custom_Human_Pose_Pipeline_Inference", [&]() {
        cv::Mat frame(
            height,
            width,
            CV_8UC4,
            reinterpret_cast<void*>(bitmap));
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2RGB);

        const auto start_time = std::chrono::steady_clock::now();
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

            const auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start_time);
            cv::putText(
                frame,
                "Cost: " + std::to_string(elapsed.count()) + "ms",
                cv::Point(50, 50),
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                cv::Scalar(0, 255, 0),
                2);
        }

        cv::cvtColor(frame, frame, cv::COLOR_RGB2RGBA);
        std::memcpy(
            reinterpret_cast<void*>(bitmap),
            frame.data,
            frame.total() * frame.elemSize());
        return CustomStatus::ERR_OK;
    });
}

CUSTOM_RET Custom_Human_Pose_Pipeline_Destroy(
    HumanPoseExtractor2D* model_ptr) noexcept {
    delete model_ptr;
    return CustomStatus::ERR_OK;
}

FaceGeometryTracker3D* Custom_Face_Geometry_Pipeline_Compile(
    const std::string& face_detector_path,
    const std::string& face_detector_config_path,
    const std::string& face_landmarker_path,
    const std::string& face_landmarker_config_path,
    const std::string& face_geometry_pipeline_metadata,
    int detect_interval) noexcept {
    return GuardWasmFactory<FaceGeometryTracker3D>(
        "Custom_Face_Geometry_Pipeline_Compile", [&]() {
            auto model = std::make_unique<FaceGeometryTracker3D>(
                face_detector_path,
                face_detector_config_path,
                face_landmarker_path,
                face_landmarker_config_path,
                face_geometry_pipeline_metadata,
                detect_interval);
            return model.release();
        });
}

CUSTOM_RET Custom_Face_Geometry_Pipeline_Inference(
    FaceGeometryTracker3D* model_ptr,
    std::intptr_t bitmap,
    int height,
    int width,
    bool display_keypoints,
    bool display_coordinates) noexcept {
    if (model_ptr == nullptr || !IsValidBitmap(bitmap, height, width)) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    return GuardWasmApi("Custom_Face_Geometry_Pipeline_Inference", [&]() {
        cv::Mat frame(
            height,
            width,
            CV_8UC4,
            reinterpret_cast<void*>(bitmap));
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2RGB);

        const CUSTOM_RET inference_status = model_ptr->Detect(
            frame, display_keypoints, display_coordinates);
        if (inference_status != CustomStatus::ERR_OK) {
            return inference_status;
        }

        cv::Mat rendered_frame = model_ptr->GetRenderedFrame();
        if (rendered_frame.empty()) {
            return CustomStatus::ERR_GENERAL_ERROR;
        }

        cv::cvtColor(
            rendered_frame, rendered_frame, cv::COLOR_RGB2RGBA);
        std::memcpy(
            reinterpret_cast<void*>(bitmap),
            rendered_frame.data,
            rendered_frame.total() * rendered_frame.elemSize());
        return CustomStatus::ERR_OK;
    });
}

CUSTOM_RET Custom_Face_Geometry_Pipeline_Destroy(
    FaceGeometryTracker3D* model_ptr) noexcept {
    delete model_ptr;
    return CustomStatus::ERR_OK;
}

EMSCRIPTEN_BINDINGS(custom_engine_module) {
    emscripten::class_<HumanPoseExtractor2D>("HumanPoseExtractor2D")
        .constructor<
            const std::string&,
            const std::string&,
            const std::string&,
            const std::string&,
            int>();
    emscripten::function(
        "Custom_Human_Pose_Pipeline_Compile",
        &Custom_Human_Pose_Pipeline_Compile,
        emscripten::allow_raw_pointers());
    emscripten::function(
        "Custom_Human_Pose_Pipeline_Inference",
        &Custom_Human_Pose_Pipeline_Inference,
        emscripten::allow_raw_pointers());
    emscripten::function(
        "Custom_Human_Pose_Pipeline_Destroy",
        &Custom_Human_Pose_Pipeline_Destroy,
        emscripten::allow_raw_pointers());

    emscripten::class_<FaceGeometryTracker3D>("FaceGeometryTracker3D")
        .constructor<
            const std::string&,
            const std::string&,
            const std::string&,
            const std::string&,
            const std::string&,
            int>();
    emscripten::function(
        "Custom_Face_Geometry_Pipeline_Compile",
        &Custom_Face_Geometry_Pipeline_Compile,
        emscripten::allow_raw_pointers());
    emscripten::function(
        "Custom_Face_Geometry_Pipeline_Inference",
        &Custom_Face_Geometry_Pipeline_Inference,
        emscripten::allow_raw_pointers());
    emscripten::function(
        "Custom_Face_Geometry_Pipeline_Destroy",
        &Custom_Face_Geometry_Pipeline_Destroy,
        emscripten::allow_raw_pointers());
}
