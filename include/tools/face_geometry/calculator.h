
#ifndef GUSTO_CALCULATOR_H
#define GUSTO_CALCULATOR_H
#include "utils.h"
#include "tools/face_geometry/geometry_pipeline.h"
#include "tools/face_geometry/face_geometry.h"
#include "tools/face_geometry/procrustes_solver.h"
namespace gusto_face_geometry {

    #define GUSTO_RET int
    class FaceMeshCalculator {
        public:
            std::tuple<std::vector<FaceGeometry>, GUSTO_RET> Process(const std::pair<int, int>& image_size, std::vector<NormalizedLandmarkList> multi_face_landmarks);
            std::tuple<std::vector<FaceGeometry>, GUSTO_RET> _Process(const GeometryPipelineMetadata& metadata, 
                        const PerspectiveCamera& perspective_camera_,
                        std::unique_ptr<GeometryPipeline>& geometry_pipeline_,
                        const std::pair<int, int>& image_size,
                        const std::vector<NormalizedLandmarkList>& multi_face_landmarks);
            GUSTO_RET Open(std::string face_GeometryPipelineMetadata);
        private:
            GeometryPipelineMetadata metadata;
            PerspectiveCamera perspective_camera;
            std::unique_ptr<GeometryPipeline> geometry_pipeline_;


    };
}

#endif // GUSTO_CALCULATOR_H