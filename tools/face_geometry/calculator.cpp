#include "Eigen/Core"
#include "tools/face_geometry/calculator.h"
namespace gusto_face_geometry {

    GUSTO_RET Process(const GeometryPipelineMetadata& metadata, 
                    const PerspectiveCamera& perspective_camera_,
                    std::unique_ptr<GeometryPipeline>& geometry_pipeline_) {


        const auto& image_size = std::make_pair(640, 480);
        // Receive NormalizedLandmarkList Here
        const auto& multi_face_landmarks = std::vector<NormalizedLandmarkList>();

        std::pair<std::vector<FaceGeometry>, GUSTO_RET> estimated_packet = geometry_pipeline_->EstimateFaceGeometry(multi_face_landmarks, image_size.first, image_size.second);
        
        if (estimated_packet.second != GustoStatus::ERR_OK){
            std::cerr << "Failed to estimate face geometry!" << std::endl;
            return GustoStatus::ERR_GENERAL_ERROR;
        }
        std::vector<FaceGeometry> multi_face_geometry = estimated_packet.first;

        // finish


        return GustoStatus::ERR_OK;
    }

    // Refer: https://github.com/google-ai-edge/mediapipe/blob/484ce05898708e625b71d580efaba9e9bd2d6d68/mediapipe/modules/face_geometry/geometry_pipeline_calculator.cc#L163C1-L166C59
    GUSTO_RET Open(std::string face_GeometryPipelineMetadata) { 
        GeometryPipelineMetadata metadata;
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
        PerspectiveCamera perspective_camera;
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
        std::unique_ptr<GeometryPipeline> geometry_pipeline_ = std::move(geometry_pipeline_packet.first);

        Process(metadata, perspective_camera, geometry_pipeline_);
        return GustoStatus::ERR_OK;
    }

} // namespace face_geometry
