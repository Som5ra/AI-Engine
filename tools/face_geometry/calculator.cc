#include "Eigen/Core"
#include "tools/face_geometry/calculator.h"

#include <cstddef>
#include <exception>
#include <iostream>
#include <utility>

namespace {

constexpr std::size_t kFaceLandmarkCount = 478;
constexpr std::size_t kLandmarkCoordinateCount = 3;
constexpr std::size_t kPoseMatrixElementCount = 16;

template <typename Function>
CUSTOM_RET GuardGeometryApi(const char* operation, Function&& function) noexcept {
    try {
        return function();
    } catch (const std::exception& exception) {
        std::cerr << operation << " failed: " << exception.what() << std::endl;
    } catch (...) {
        std::cerr << operation << " failed with an unknown error" << std::endl;
    }
    return CustomStatus::ERR_GENERAL_ERROR;
}

}  // namespace

namespace custom_face_geometry {

CUSTOM_RET FaceMeshCalculator::Open(
    const std::string& face_geometry_pipeline_metadata) {
    metadata_ = GeometryPipelineMetadata{};
    const auto serialization_status =
        metadata_.serialize_json(face_geometry_pipeline_metadata);
    if (serialization_status != CustomStatus::ERR_OK) {
        return CustomStatus::ERR_GENERAL_SERIALIZATION;
    }

    if (ValidateGeometryPipelineMetadata(metadata_) != CustomStatus::ERR_OK) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    perspective_camera_.vertical_fov_degrees_ = 63.0F;
    perspective_camera_.near_ = 1.0F;
    perspective_camera_.far_ = 10000.0F;

    const Environment environment{
        OriginPointLocation::TOP_LEFT_CORNER,
        perspective_camera_,
    };
    if (ValidateEnvironment(environment) != CustomStatus::ERR_OK) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    auto [geometry_pipeline, create_status] =
        CreateGeometryPipeline(environment, metadata_);
    if (create_status != CustomStatus::ERR_OK || !geometry_pipeline) {
        return CustomStatus::ERR_GENERAL_ERROR;
    }

    geometry_pipeline_ = std::move(geometry_pipeline);
    return CustomStatus::ERR_OK;
}

std::tuple<std::vector<FaceGeometry>, CUSTOM_RET>
FaceMeshCalculator::Process(
    const std::pair<int, int>& image_size,
    const std::vector<NormalizedLandmarkList>& multi_face_landmarks) {
    if (!geometry_pipeline_) {
        return {{}, CustomStatus::ERR_GENERAL_ERROR};
    }
    if (ValidateFrameDimensions(image_size.first, image_size.second) !=
        CustomStatus::ERR_OK) {
        return {{}, CustomStatus::ERR_GENERAL_INVALID_PARAMETER};
    }
    if (multi_face_landmarks.empty()) {
        return {{}, CustomStatus::ERR_OK};
    }

    return ProcessInternal(image_size, multi_face_landmarks);
}

std::tuple<std::vector<FaceGeometry>, CUSTOM_RET>
FaceMeshCalculator::ProcessInternal(
    const std::pair<int, int>& image_size,
    const std::vector<NormalizedLandmarkList>& multi_face_landmarks) {
    auto [estimated_geometries, estimate_status] =
        geometry_pipeline_->EstimateFaceGeometry(
            multi_face_landmarks,
            image_size.first,
            image_size.second);

    if (estimate_status != CustomStatus::ERR_OK) {
        return {
            std::move(estimated_geometries),
            CustomStatus::ERR_GENERAL_ERROR,
        };
    }

    return {std::move(estimated_geometries), CustomStatus::ERR_OK};
}

}  // namespace custom_face_geometry

extern "C" {

using custom_face_geometry::FaceMeshCalculator;
using custom_face_geometry::NormalizedLandmark;
using custom_face_geometry::NormalizedLandmarkList;

CUSTOM_API CUSTOM_RET face_mesh_calculator_new(
    FaceMeshCalculator** face_mesh_calculator) {
    if (face_mesh_calculator == nullptr) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }
    *face_mesh_calculator = nullptr;

    return GuardGeometryApi("face_mesh_calculator_new", [&]() {
        *face_mesh_calculator = new FaceMeshCalculator();
        return CustomStatus::ERR_OK;
    });
}

CUSTOM_API CUSTOM_RET face_mesh_calculator_open(
    FaceMeshCalculator* face_mesh_calculator,
    const char* face_geometry_pipeline_metadata,
    int buffer_size) {
    if (face_mesh_calculator == nullptr ||
        face_geometry_pipeline_metadata == nullptr ||
        buffer_size <= 0) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    return GuardGeometryApi("face_mesh_calculator_open", [&]() {
        const std::string metadata(
            face_geometry_pipeline_metadata,
            static_cast<std::size_t>(buffer_size));
        return face_mesh_calculator->Open(metadata);
    });
}

CUSTOM_API CUSTOM_RET face_mesh_calculator_process(
    FaceMeshCalculator* face_mesh_calculator,
    int image_width,
    int image_height,
    const float* multi_face_landmarks,
    int num_faces,
    float* face_geometry_pose_mat) {
    if (face_mesh_calculator == nullptr ||
        image_width <= 0 ||
        image_height <= 0 ||
        num_faces < 0) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }
    if (num_faces == 0) {
        return CustomStatus::ERR_OK;
    }
    if (multi_face_landmarks == nullptr ||
        face_geometry_pose_mat == nullptr) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    return GuardGeometryApi("face_mesh_calculator_process", [&]() {
        std::vector<NormalizedLandmarkList> landmark_batches;
        landmark_batches.reserve(static_cast<std::size_t>(num_faces));

        for (int face_index = 0; face_index < num_faces; ++face_index) {
            NormalizedLandmarkList face_landmarks;
            face_landmarks.landmark.reserve(kFaceLandmarkCount);

            for (std::size_t landmark_index = 0;
                 landmark_index < kFaceLandmarkCount;
                 ++landmark_index) {
                const std::size_t coordinate_offset =
                    (static_cast<std::size_t>(face_index) *
                         kFaceLandmarkCount +
                     landmark_index) *
                    kLandmarkCoordinateCount;

                NormalizedLandmark landmark{};
                landmark.x = multi_face_landmarks[coordinate_offset];
                landmark.y = multi_face_landmarks[coordinate_offset + 1];
                landmark.z = multi_face_landmarks[coordinate_offset + 2];
                face_landmarks.landmark.push_back(landmark);
            }

            landmark_batches.push_back(std::move(face_landmarks));
        }

        auto [face_geometries, process_status] =
            face_mesh_calculator->Process(
                std::make_pair(image_width, image_height),
                landmark_batches);
        if (process_status != CustomStatus::ERR_OK) {
            return process_status;
        }
        if (face_geometries.size() !=
            static_cast<std::size_t>(num_faces)) {
            return CustomStatus::ERR_PARTIAL_FAIL;
        }

        for (int face_index = 0; face_index < num_faces; ++face_index) {
            for (std::size_t row = 0; row < 4; ++row) {
                for (std::size_t column = 0; column < 4; ++column) {
                    const std::size_t output_offset =
                        static_cast<std::size_t>(face_index) *
                            kPoseMatrixElementCount +
                        row * 4 +
                        column;
                    face_geometry_pose_mat[output_offset] =
                        face_geometries[face_index]
                            .pose_transform_matrix.at(row, column);
                }
            }
        }

        return CustomStatus::ERR_OK;
    });
}

CUSTOM_API CUSTOM_RET face_mesh_calculator_destroy(
    FaceMeshCalculator* face_mesh_calculator) {
    delete face_mesh_calculator;
    return CustomStatus::ERR_OK;
}

}  // extern "C"
