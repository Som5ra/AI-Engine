#ifndef GUSTO_CALCULATOR_H
#define GUSTO_CALCULATOR_H

#include "tools/face_geometry/face_geometry.h"
#include "tools/face_geometry/geometry_pipeline.h"
#include "utils.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace gusto_face_geometry {

#define GUSTO_RET int

class FaceMeshCalculator {
public:
    GUSTO_RET Open(const std::string& face_geometry_pipeline_metadata);

    std::tuple<std::vector<FaceGeometry>, GUSTO_RET> Process(
        const std::pair<int, int>& image_size,
        const std::vector<NormalizedLandmarkList>& multi_face_landmarks);

private:
    std::tuple<std::vector<FaceGeometry>, GUSTO_RET> ProcessInternal(
        const std::pair<int, int>& image_size,
        const std::vector<NormalizedLandmarkList>& multi_face_landmarks);

    GeometryPipelineMetadata metadata_;
    PerspectiveCamera perspective_camera_;
    std::unique_ptr<GeometryPipeline> geometry_pipeline_;
};

}  // namespace gusto_face_geometry

extern "C" {

GUSTO_API GUSTO_RET face_mesh_calculator_new(
    gusto_face_geometry::FaceMeshCalculator** face_mesh_calculator);
GUSTO_API GUSTO_RET face_mesh_calculator_open(
    gusto_face_geometry::FaceMeshCalculator* face_mesh_calculator,
    const char* face_geometry_pipeline_metadata,
    int buffer_size);
GUSTO_API GUSTO_RET face_mesh_calculator_process(
    gusto_face_geometry::FaceMeshCalculator* face_mesh_calculator,
    int image_width,
    int image_height,
    const float* multi_face_landmarks,
    int num_faces,
    float* face_geometry_pose_mat);
GUSTO_API GUSTO_RET face_mesh_calculator_destroy(
    gusto_face_geometry::FaceMeshCalculator* face_mesh_calculator);

}  // extern "C"

#endif  // GUSTO_CALCULATOR_H
