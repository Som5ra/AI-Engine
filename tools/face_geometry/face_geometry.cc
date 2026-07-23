#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Core"

#include "utils.h"
#include "tools/face_geometry/geometry_pipeline.h"
#include "tools/face_geometry/face_geometry.h"
#include "tools/face_geometry/procrustes_solver.h"

namespace custom_face_geometry {
    CUSTOM_RET ValidatePerspectiveCamera(
        const PerspectiveCamera& perspective_camera) {
    static constexpr float kAbsoluteErrorEps = 1e-9f;

    if (!std::isfinite(perspective_camera.near()) ||
        !std::isfinite(perspective_camera.far()) ||
        !std::isfinite(perspective_camera.vertical_fov_degrees())) {
        std::cerr << "Perspective camera values must be finite!" << std::endl;
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    if (perspective_camera.near() <= kAbsoluteErrorEps){
        std::cerr << "Near Z must be greater than 0 with a margin of 10^{-9}!" << std::endl;
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    if (perspective_camera.far() <= perspective_camera.near() + kAbsoluteErrorEps){
        std::cerr << "Far Z must be greater than Near Z with a margin of 10^{-9}!" << std::endl;
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }
    if (perspective_camera.vertical_fov_degrees() <= kAbsoluteErrorEps){
        std::cerr << "Vertical FOV must be positive with a margin of 10^{-9}!" << std::endl;
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }
    if (perspective_camera.vertical_fov_degrees() + kAbsoluteErrorEps >= 180.f){
        std::cerr << "Vertical FOV must be less than 180 degrees with a margin of 10^{-9}" << std::endl;
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }
    return CustomStatus::ERR_OK;
    }

    CUSTOM_RET ValidateEnvironment(const Environment& environment) {
        return ValidatePerspectiveCamera(environment.perspective_camera);
    }

    CUSTOM_RET ValidateMesh3d(const Mesh3d& mesh_3d) {
        const std::size_t vertex_size = mesh_3d.canonical_mesh_vertex_size;
        const std::size_t primitive_type = mesh_3d.primitive_type;
        const std::size_t position_offset =
            mesh_3d.canonical_mesh_vertex_position_offset;

        if (vertex_size == 0 || primitive_type == 0 ||
            position_offset >= vertex_size ||
            vertex_size - position_offset < 3) {
            std::cerr << "Vertex, primitive, or position layout is invalid!" << std::endl;
            return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
        }

        if (mesh_3d.vertex_buffer_size() % vertex_size != 0 ||
            mesh_3d.index_buffer_size() % primitive_type != 0) {
            std::cerr << "Invalid vertex or primitive size!" << std::endl;
            return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
        }

        const int num_vertices = mesh_3d.vertex_buffer_size() / vertex_size;
        if (num_vertices <= 0 ||
            mesh_3d.canonical_mesh_num_vertices !=
                static_cast<std::size_t>(num_vertices)) {
            std::cerr << "Canonical mesh vertex count is inconsistent!" << std::endl;
            return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
        }

        for (int idx : mesh_3d.index_buffer) {
            if (idx < 0 || idx >= num_vertices){
                std::cerr << "All mesh indices must refer to an existing vertex!" << std::endl;
                return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
            }
        }

        return CustomStatus::ERR_OK;
    }

    CUSTOM_RET ValidateFaceGeometry(const FaceGeometry& face_geometry) {
        if (ValidateMesh3d(face_geometry.mesh) != CustomStatus::ERR_OK){
            std::cerr << "Invalid mesh!" << std::endl;
            return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
        }

        const MatrixData& pose_transform_matrix = face_geometry.pose_transform_matrix;
        if (pose_transform_matrix.rows != 4 || pose_transform_matrix.cols != 4 || pose_transform_matrix.packed_data_size() != 16){
            std::cerr << "Pose transformation matrix must be a 4x4 matrix!" << std::endl;
            return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
        }
        return CustomStatus::ERR_OK;
    }

    CUSTOM_RET ValidateGeometryPipelineMetadata(const GeometryPipelineMetadata& metadata) {
        if (ValidateMesh3d(metadata.canonical_mesh) != CustomStatus::ERR_OK){
            std::cerr << "Invalid canonical mesh!" << std::endl;
            return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
        }
        if (metadata.procrustes_landmark_basis_size() <= 0){
            std::cerr << "Procrustes landmark basis must be non-empty!" << std::endl;
            return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
        }

        const int num_vertices = metadata.canonical_mesh.vertex_buffer_size() / metadata.canonical_mesh.canonical_mesh_vertex_size;
        for (const WeightedLandmarkRef& wlr : metadata.procrustes_landmark_basis) {
            if (wlr.landmark_id < 0 || wlr.landmark_id >= num_vertices){
                std::cerr << "All Procrustes basis indices must refer to an existing canonical mesh vertex!" << std::endl;
                return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
            }
            if (!std::isfinite(wlr.weight) || wlr.weight < 0.f){
                std::cerr << "All Procrustes basis landmarks must have a non-negative weight!" << std::endl;
                return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
            }
        }

        return CustomStatus::ERR_OK;
    }

    CUSTOM_RET ValidateFrameDimensions(int frame_width, int frame_height) {
        if (frame_width <= 0 || frame_height <= 0){
            std::cerr << "Frame width and height must be positive!" << std::endl;
            return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
        }
        return CustomStatus::ERR_OK;
    }

    CUSTOM_RET GeometryPipelineMetadata::CUSTOM_CHECK_CANONICAL_MESH() {
        if (ValidateMesh3d(canonical_mesh) != CustomStatus::ERR_OK ||
            canonical_mesh.canonical_mesh_num_vertices != 478 ||
            canonical_mesh.canonical_mesh_vertex_size != 5 ||
            canonical_mesh.canonical_mesh_vertex_position_offset != 0) {
            std::cerr << "Canonical mesh is not valid" << std::endl;
            return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
        }
        return CustomStatus::ERR_OK;
    }

    CUSTOM_RET GeometryPipelineMetadata::serialize_json(
        const std::string& filename) {
        try {
            std::ifstream input(filename);
            if (!input.is_open()) {
                std::cerr << "Unable to open json file: " << filename << std::endl;
                return CustomStatus::ERR_GENERAL_SERIALIZATION;
            }

            const json data = json::parse(input);
            GeometryPipelineMetadata parsed{};
            parsed.input_source =
                data.at("input_source").template get<InputSource>();

            for (const auto& item : data.at("procrustes_landmark_basis")) {
                parsed.procrustes_landmark_basis.push_back(
                    WeightedLandmarkRef{
                        item.at("landmark_id").template get<int>(),
                        item.at("weight").template get<float>(),
                    });
            }
            for (const auto& item :
                 data.at("canonical_mesh").at("vertex_buffer")) {
                parsed.canonical_mesh.vertex_buffer.push_back(
                    item.template get<float>());
            }
            for (const auto& item :
                 data.at("canonical_mesh").at("index_buffer")) {
                parsed.canonical_mesh.index_buffer.push_back(
                    item.template get<int>());
            }

            if (parsed.CUSTOM_CHECK_CANONICAL_MESH() != CustomStatus::ERR_OK ||
                ValidateGeometryPipelineMetadata(parsed) !=
                    CustomStatus::ERR_OK) {
                return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
            }

            *this = std::move(parsed);
            return CustomStatus::ERR_OK;
        } catch (const std::exception& exception) {
            std::cerr << "Error parsing json file '" << filename
                      << "': " << exception.what() << std::endl;
            return CustomStatus::ERR_GENERAL_SERIALIZATION;
        }
    }


} // namespace face_geometry
