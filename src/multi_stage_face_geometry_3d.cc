#include "multi_stage_face_geometry_3d.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

namespace {

constexpr float kMinimumLandmarkScore = 0.49F;

}  // namespace

FaceGeometryTracker3D::FaceGeometryTracker3D(
    const std::string& face_detector_path,
    const std::string& face_detector_config_path,
    const std::string& face_landmarker_path,
    const std::string& face_landmarker_config_path,
    const std::string& face_geometry_pipeline_metadata,
    int detect_interval)
    : face_detector_(std::make_unique<custom_mp_face::FaceDetector>(
          face_detector_path, face_detector_config_path)),
      face_landmarker_(std::make_unique<custom_mp_face::FaceLandmarker>(
          face_landmarker_path, face_landmarker_config_path)),
      face_mesh_calculator_(
          std::make_unique<custom_face_geometry::FaceMeshCalculator>()),
      detect_interval_(detect_interval) {
    initialization_status_ =
        face_mesh_calculator_->Open(face_geometry_pipeline_metadata);
}

CUSTOM_RET FaceGeometryTracker3D::Detect(
    const cv::Mat& frame,
    bool display_keypoints,
    bool display_coord) {
    (void)display_coord;

    face_geometries_.clear();
    rendered_frame_.release();

    if (frame.empty()) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    rendered_frame_ = frame.clone();

    if (initialization_status_ != CustomStatus::ERR_OK) {
        return initialization_status_;
    }

    if (!face_detector_ || !face_landmarker_ || !face_mesh_calculator_) {
        return CustomStatus::ERR_GENERAL_ERROR;
    }

    try {
        auto detector_output = face_detector_->forward(frame);
        auto* detector_result =
            dynamic_cast<custom_mp_face::MediaPipeDetectorResult*>(
                detector_output.get());
        if (detector_result == nullptr) {
            return CustomStatus::ERR_GENERAL_ERROR;
        }

        std::vector<custom_face_geometry::NormalizedLandmarkList>
            multi_face_landmarks;
        multi_face_landmarks.reserve(detector_result->boxes.size());

        for (const auto& box : detector_result->boxes) {
            if (!std::isfinite(box.x1) || !std::isfinite(box.y1) ||
                !std::isfinite(box.x2) || !std::isfinite(box.y2)) {
                continue;
            }

            const int top = std::clamp(
                static_cast<int>(box.y1 * frame.rows), 0, frame.rows);
            const int left = std::clamp(
                static_cast<int>(box.x1 * frame.cols), 0, frame.cols);
            const int bottom = std::clamp(
                static_cast<int>(box.y2 * frame.rows), 0, frame.rows);
            const int right = std::clamp(
                static_cast<int>(box.x2 * frame.cols), 0, frame.cols);

            if (bottom <= top || right <= left) {
                continue;
            }

            const std::vector<int> crop_box = {
                top,
                left,
                bottom,
                right,
            };

            auto [cropped_face, crop_box_with_margin] =
                face_landmarker_->crop_face(frame, crop_box);
            if (cropped_face.empty() || crop_box_with_margin.size() < 2) {
                continue;
            }

            auto landmarker_output = face_landmarker_->forward(cropped_face);
            auto* landmarker_result =
                dynamic_cast<custom_mp_face::MediapipeFaceLandmarkResult*>(
                    landmarker_output.get());
            if (landmarker_result == nullptr) {
                return CustomStatus::ERR_GENERAL_ERROR;
            }

            if (landmarker_result->score < kMinimumLandmarkScore ||
                landmarker_result->points.empty()) {
                continue;
            }

            custom_face_geometry::NormalizedLandmarkList face_landmarks;
            face_landmarks.landmark.reserve(landmarker_result->points.size());

            for (const auto& point : landmarker_result->points) {
                custom_face_geometry::NormalizedLandmark landmark{};
                landmark.x =
                    (point.x + crop_box_with_margin[1]) / frame.cols;
                landmark.y =
                    (point.y + crop_box_with_margin[0]) / frame.rows;
                landmark.z = point.z / frame.cols;
                face_landmarks.landmark.push_back(landmark);
            }

            multi_face_landmarks.push_back(std::move(face_landmarks));

            if (display_keypoints) {
                rendered_frame_ = face_landmarker_->draw_points(
                    rendered_frame_,
                    landmarker_result->points,
                    cv::Point(
                        crop_box_with_margin[1],
                        crop_box_with_margin[0]));
            }
        }

        ++num_frames_;

        // No face is a valid inference result, not a pipeline failure.
        if (multi_face_landmarks.empty()) {
            return CustomStatus::ERR_OK;
        }

        auto [face_geometries, process_status] =
            face_mesh_calculator_->Process(
                std::make_pair(frame.cols, frame.rows),
                multi_face_landmarks);
        face_geometries_ = std::move(face_geometries);
        return process_status;
    } catch (const cv::Exception& exception) {
        std::cerr << "Face geometry OpenCV error: " << exception.what()
                  << std::endl;
    } catch (const std::exception& exception) {
        std::cerr << "Face geometry inference error: " << exception.what()
                  << std::endl;
    }

    face_geometries_.clear();
    rendered_frame_ = frame.clone();
    return CustomStatus::ERR_GENERAL_ERROR;
}

cv::Mat FaceGeometryTracker3D::GetRenderedFrame() const {
    return rendered_frame_;
}

const std::vector<custom_face_geometry::FaceGeometry>&
FaceGeometryTracker3D::GetFaceGeometries() const noexcept {
    return face_geometries_;
}
