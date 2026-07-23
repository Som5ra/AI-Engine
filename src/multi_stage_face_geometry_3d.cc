#include "multi_stage_face_geometry_3d.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

namespace {

constexpr float kMinimumLandmarkScore = 0.49F;
constexpr float kMinimumAxisLength = 20.0F;
constexpr float kAxisLengthRatio = 0.08F;
constexpr float kInvalidPoseValueThreshold = -9000.0F;

void DrawCoordinateAxes(
    cv::Mat& frame,
    const custom_face_geometry::FaceGeometry& geometry,
    const cv::Point2f& origin) {
    const auto& matrix = geometry.pose_transform_matrix;
    if (matrix.rows != 4 || matrix.cols != 4) {
        return;
    }

    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            const float value = matrix.at(row, column);
            if (!std::isfinite(value) ||
                value < kInvalidPoseValueThreshold) {
                return;
            }
        }
    }

    const float axis_length = std::max(
        kMinimumAxisLength,
        std::min(frame.cols, frame.rows) * kAxisLengthRatio);

    const auto endpoint = [&](std::size_t column) {
        const float x = matrix.at(0, column);
        const float y = -matrix.at(1, column);
        const float norm = std::hypot(x, y);
        if (norm <= 1e-6F) {
            return origin;
        }
        return cv::Point2f(
            origin.x + axis_length * x / norm,
            origin.y + axis_length * y / norm);
    };

    cv::circle(frame, origin, 3, cv::Scalar(255, 255, 255), -1);
    cv::line(frame, origin, endpoint(0), cv::Scalar(0, 0, 255), 2);
    cv::line(frame, origin, endpoint(1), cv::Scalar(0, 255, 0), 2);
    cv::line(frame, origin, endpoint(2), cv::Scalar(255, 0, 0), 2);
}

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
      detect_interval_(std::max(1, detect_interval)) {
    initialization_status_ =
        face_mesh_calculator_->Open(face_geometry_pipeline_metadata);
}

CUSTOM_RET FaceGeometryTracker3D::Detect(
    const cv::Mat& frame,
    bool display_keypoints,
    bool display_coord) {
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
        const bool run_detector =
            cached_face_boxes_.empty() || frame_index_ == 0;
        if (run_detector) {
            auto detector_output = face_detector_->forward(frame);
            auto* detector_result =
                dynamic_cast<custom_mp_face::MediaPipeDetectorResult*>(
                    detector_output.get());
            if (detector_result == nullptr) {
                return CustomStatus::ERR_GENERAL_ERROR;
            }

            cached_face_boxes_.clear();
            cached_face_boxes_.reserve(detector_result->boxes.size());
            for (const auto& box : detector_result->boxes) {
                if (std::isfinite(box.x1) && std::isfinite(box.y1) &&
                    std::isfinite(box.x2) && std::isfinite(box.y2) &&
                    box.x2 > box.x1 && box.y2 > box.y1) {
                    cached_face_boxes_.push_back(box);
                }
            }
        }
        frame_index_ = (frame_index_ + 1) % detect_interval_;

        std::vector<custom_face_geometry::NormalizedLandmarkList>
            multi_face_landmarks;
        multi_face_landmarks.reserve(cached_face_boxes_.size());
        std::vector<cv::Point2f> face_origins;
        face_origins.reserve(cached_face_boxes_.size());

        for (const auto& box : cached_face_boxes_) {
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
            if (cropped_face.empty() || crop_box_with_margin.size() < 4) {
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
            face_origins.emplace_back(
                0.5F * (crop_box_with_margin[1] +
                        crop_box_with_margin[3]),
                0.5F * (crop_box_with_margin[0] +
                        crop_box_with_margin[2]));

            if (display_keypoints) {
                rendered_frame_ = face_landmarker_->draw_points(
                    rendered_frame_,
                    landmarker_result->points,
                    cv::Point(
                        crop_box_with_margin[1],
                        crop_box_with_margin[0]));
            }
        }

        // No face is a valid inference result, not a pipeline failure.
        if (multi_face_landmarks.empty()) {
            return CustomStatus::ERR_OK;
        }

        auto [face_geometries, process_status] =
            face_mesh_calculator_->Process(
                std::make_pair(frame.cols, frame.rows),
                multi_face_landmarks);
        face_geometries_ = std::move(face_geometries);

        if (display_coord &&
            (process_status == CustomStatus::ERR_OK ||
             process_status == CustomStatus::ERR_PARTIAL_FAIL) &&
            face_geometries_.size() == face_origins.size()) {
            for (std::size_t index = 0; index < face_geometries_.size();
                 ++index) {
                DrawCoordinateAxes(
                    rendered_frame_, face_geometries_[index],
                    face_origins[index]);
            }
        }

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
