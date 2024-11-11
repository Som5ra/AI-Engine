#include "Eigen/Core"
#include "tools/face_geometry/calculator.h"
namespace gusto_face_geometry {

// Refer: https://github.com/google-ai-edge/mediapipe/blob/484ce05898708e625b71d580efaba9e9bd2d6d68/mediapipe/modules/face_geometry/geometry_pipeline_calculator.cc#L163C1-L166C59
GUSTO_RET FaceMeshCalculator::Open(std::string face_GeometryPipelineMetadata) { 
    // GeometryPipelineMetadata metadata;

    // Serialize the metadata
    // It's the same as parsing the proto file in mediapipe
    try{
        metadata.serialize_json(face_GeometryPipelineMetadata);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return GustoStatus::ERR_GENERAL_ERROR;
    }
    if(ValidateGeometryPipelineMetadata(metadata) != GustoStatus::ERR_OK){
        std::cerr << "Invalid metadata!" << std::endl;
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    // Set the perspective camera
    // PerspectiveCamera perspective_camera;
    perspective_camera.vertical_fov_degrees_ = 63.0f;
    perspective_camera.near_ = 1.0f;
    perspective_camera.far_ = 10000.0f;
    const Environment environment{OriginPointLocation::TOP_LEFT_CORNER, perspective_camera};
    if (ValidateEnvironment(environment) != GustoStatus::ERR_OK) {
        std::cerr << "Invalid environment!" << std::endl;
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    
    std::pair<std::unique_ptr<GeometryPipeline>, GUSTO_RET> geometry_pipeline_packet = CreateGeometryPipeline(environment, metadata);
    if (geometry_pipeline_packet.second != GustoStatus::ERR_OK){
        std::cerr << "Failed to create geometry pipeline!" << std::endl;
        return GustoStatus::ERR_GENERAL_ERROR;
    }
    // std::unique_ptr<GeometryPipeline> geometry_pipeline_ = std::move(geometry_pipeline_packet.first);
    geometry_pipeline_ = std::move(geometry_pipeline_packet.first);

    // Process(metadata, perspective_camera, geometry_pipeline_);
    return GustoStatus::ERR_OK;
}

std::tuple<std::vector<FaceGeometry>, GUSTO_RET> FaceMeshCalculator::Process(const std::pair<int, int>& image_size, std::vector<NormalizedLandmarkList> multi_face_landmarks) {
    return _Process(metadata, perspective_camera, geometry_pipeline_, image_size, multi_face_landmarks);
}

std::tuple<std::vector<FaceGeometry>, GUSTO_RET> FaceMeshCalculator::_Process(const GeometryPipelineMetadata& metadata, 
                        const PerspectiveCamera& perspective_camera_,
                        std::unique_ptr<GeometryPipeline>& geometry_pipeline_,
                        const std::pair<int, int>& image_size,
                        const std::vector<NormalizedLandmarkList>& multi_face_landmarks) {

    std::pair<std::vector<FaceGeometry>, GUSTO_RET> estimated_packet = geometry_pipeline_->EstimateFaceGeometry(multi_face_landmarks, image_size.first, image_size.second);
    
    if (estimated_packet.second != GustoStatus::ERR_OK){
        std::cerr << "Failed to estimate face geometry!" << std::endl;
        return {estimated_packet.first, GustoStatus::ERR_GENERAL_ERROR};
    }
    return {estimated_packet.first, GustoStatus::ERR_OK};
}

} // namespace gusto_face_geometry



// expose for C# Unity Side
extern "C" {
    using namespace gusto_face_geometry;
    int face_mesh_calculator_new(FaceMeshCalculator** face_mesh_calculator) {
        *face_mesh_calculator = new FaceMeshCalculator();
        return GustoStatus::ERR_OK;
    }

    int face_mesh_calculator_open(FaceMeshCalculator* face_mesh_calculator, const char* _face_GeometryPipelineMetadata, int buffer_size) {
        // std::string face_GeometryPipelineMetadata.assign(_face_GeometryPipelineMetadata, buffer_size);
        std::string face_GeometryPipelineMetadata(_face_GeometryPipelineMetadata, buffer_size);
        return face_mesh_calculator->Open(face_GeometryPipelineMetadata);
    }

    int face_mesh_calculator_process(FaceMeshCalculator* face_mesh_calculator, 
            int image_width, int image_height, 
            float* _multi_face_landmarks, int num_faces,
            float* face_geometry_pose_mat) {
        /*
            multi_face_landmarks: (478, 3, num_faces)
            face_geometry_pose_mat: (4, 4, num_faces)
        */
        std::vector<NormalizedLandmarkList> multi_face_landmarks;
        for (size_t face_idx = 0; face_idx < num_faces; face_idx++) {
            NormalizedLandmarkList thislandmark;
            for (size_t landmark_idx = 0; landmark_idx < 478; landmark_idx++) {
                NormalizedLandmark landmark;
                landmark.x = _multi_face_landmarks[landmark_idx * 3 + 0];
                landmark.y = _multi_face_landmarks[landmark_idx * 3 + 1];
                landmark.z = _multi_face_landmarks[landmark_idx * 3 + 2];
                thislandmark.landmark.push_back(landmark);
            }
            multi_face_landmarks.push_back(thislandmark);
        }
       auto [multi_pose_mat, process_status]  = face_mesh_calculator->Process(std::make_pair(image_width, image_height), multi_face_landmarks);
       if (process_status != GustoStatus::ERR_OK) {
           return process_status;
       }else{
           for (size_t face_idx = 0; face_idx < num_faces; face_idx++) {
               for (int i = 0; i < 4; ++i) {
                   for (int j = 0; j < 4; ++j) {
                       face_geometry_pose_mat[face_idx * 16 + i * 4 + j] = multi_pose_mat[face_idx].pose_transform_matrix.at(i, j);
                   }
               }
           }
       }
       return GustoStatus::ERR_OK;
    }
}