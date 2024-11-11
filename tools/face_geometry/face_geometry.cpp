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

namespace gusto_face_geometry {
    GUSTO_RET ValidatePerspectiveCamera(
        const PerspectiveCamera& perspective_camera) {
    static constexpr float kAbsoluteErrorEps = 1e-9f;
    
    if (perspective_camera.near() <= kAbsoluteErrorEps){
        std::cerr << "Near Z must be greater than 0 with a margin of 10^{-9}!" << std::endl;
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    if (perspective_camera.far() <= perspective_camera.near() + kAbsoluteErrorEps){
        std::cerr << "Far Z must be greater than Near Z with a margin of 10^{-9}!" << std::endl;
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }
    if (perspective_camera.vertical_fov_degrees() <= kAbsoluteErrorEps){
        std::cerr << "Vertical FOV must be positive with a margin of 10^{-9}!" << std::endl;
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }
    if (perspective_camera.vertical_fov_degrees() + kAbsoluteErrorEps >= 180.f){
        std::cerr << "Vertical FOV must be less than 180 degrees with a margin of 10^{-9}" << std::endl;
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }
    return GustoStatus::ERR_OK;
    }

    GUSTO_RET ValidateEnvironment(const Environment& environment) {
        return ValidatePerspectiveCamera(environment.perspective_camera);
    }

    GUSTO_RET ValidateMesh3d(const Mesh3d& mesh_3d) {
        const std::size_t vertex_size = mesh_3d.canonical_mesh_vertex_size;
        const std::size_t primitive_type = mesh_3d.primitive_type;

        if (mesh_3d.vertex_buffer_size() % vertex_size != 0 || mesh_3d.index_buffer_size() % primitive_type != 0) {
            std::cerr << "Invalid vertex or primitive size!" << std::endl;
            return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
        }

        const int num_vertices = mesh_3d.vertex_buffer_size() / vertex_size;
        for (uint32_t idx : mesh_3d.index_buffer) {
            if (idx >= num_vertices){
                std::cerr << "All mesh indices must refer to an existing vertex!" << std::endl;
                return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
            }
        }

        return GustoStatus::ERR_OK;
    }

    GUSTO_RET ValidateFaceGeometry(const FaceGeometry& face_geometry) {
        if (ValidateMesh3d(face_geometry.mesh) != GustoStatus::ERR_OK){
            std::cerr << "Invalid mesh!" << std::endl;
            return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
        }

        const MatrixData& pose_transform_matrix = face_geometry.pose_transform_matrix;
        if (pose_transform_matrix.rows != 4 || pose_transform_matrix.cols != 4 || pose_transform_matrix.packed_data_size() != 16){
            std::cerr << "Pose transformation matrix must be a 4x4 matrix!" << std::endl;
            return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
        }
        return GustoStatus::ERR_OK;
    }

    GUSTO_RET ValidateGeometryPipelineMetadata(const GeometryPipelineMetadata& metadata) {
        if (ValidateMesh3d(metadata.canonical_mesh) != GustoStatus::ERR_OK){
            std::cerr << "Invalid canonical mesh!" << std::endl;
            return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
        }
        if (metadata.procrustes_landmark_basis_size() <= 0){
            std::cerr << "Procrustes landmark basis must be non-empty!" << std::endl;
            return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
        }

        const int num_vertices = metadata.canonical_mesh.vertex_buffer_size() / metadata.canonical_mesh.canonical_mesh_vertex_size;
        for (const WeightedLandmarkRef& wlr : metadata.procrustes_landmark_basis) {
            if (wlr.landmark_id >= num_vertices){
                std::cerr << "All Procrustes basis indices must refer to an existing canonical mesh vertex!" << std::endl;
                return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
            }
            if (wlr.weight < 0.f){
                std::cerr << "All Procrustes basis landmarks must have a non-negative weight!" << std::endl;
                return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
            }
        }

        return GustoStatus::ERR_OK;
    }

    GUSTO_RET ValidateFrameDimensions(int frame_width, int frame_height) {
        if (frame_width <= 0 || frame_height <= 0){
            std::cerr << "Frame width and height must be positive!" << std::endl;
            return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
        }
        return GustoStatus::ERR_OK;
    }

    GUSTO_RET GeometryPipelineMetadata::GUSTO_CHECK_CANONICAL_MESH(){
        // actually canonical_mesh.vertex_buffer.size() / 5 == 478
        // this->canonical_mesh_num_vertices = this->canonical_mesh.vertex_buffer.size() / 5;
        try{
            assert(this->canonical_mesh.vertex_buffer.size() / 5 == canonical_mesh_num_vertices);
            assert(this->canonical_mesh.canonical_mesh_vertex_size == 5);
            assert(this->canonical_mesh.canonical_mesh_vertex_position_offset == 0);
        }catch (const std::exception& e) {
            std::cerr << "Canonical mesh is not valid" << std::endl;
            return GustoStatus::ERR_GENERAL_ERROR;
        }
        return GustoStatus::ERR_OK;
    }

    GUSTO_RET GeometryPipelineMetadata::serialize_json(const std::string& filename)
    {
        // GustoSerializer serializer;
        // json data = serializer.load_json(filename.c_str());
        json data;
        try{
            data = json::parse(std::ifstream(filename.c_str()));
        }catch (const std::exception& e) {
            std::cerr << "Error parsing json file: " << filename << std::endl;
            return GustoStatus::ERR_GENERAL_ERROR;
        }

        this->input_source = data.template get<InputSource>();
        for(auto& it : data["procrustes_landmark_basis"]) {
            this->procrustes_landmark_basis.push_back(WeightedLandmarkRef{it["landmark_id"].get<int>(), it["weight"].get<float>()});
        }
        for(auto& it : data["canonical_mesh"]["vertex_buffer"]) { this->canonical_mesh.vertex_buffer.push_back(it.get<float>());}
        for(auto& it : data["canonical_mesh"]["index_buffer"]) { this->canonical_mesh.index_buffer.push_back(it.get<int>());}
        
        return GeometryPipelineMetadata::GUSTO_CHECK_CANONICAL_MESH();
    }



} // namespace face_geometry
