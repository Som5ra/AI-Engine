#ifndef CUSTOM_CALCULATOR_H
#define CUSTOM_CALCULATOR_H

#include "tools/face_geometry/face_geometry.h"
#include "tools/face_geometry/geometry_pipeline.h"
#include "utils.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace custom_face_geometry {

#define CUSTOM_RET int

class FaceMeshCalculator {
public:
    CUSTOM_RET Open(const std::string& face_geometry_pipeline_metadata);

    std::tuple<std::vector<FaceGeometry>, CUSTOM_RET> Process(
        const std::pair<int, int>& image_size,
        const std::vector<NormalizedLandmarkList>& multi_face_landmarks);

private:
    std::tuple<std::vector<FaceGeometry>, CUSTOM_RET> ProcessInternal(
        const std::pair<int, int>& image_size,
        const std::vector<NormalizedLandmarkList>& multi_face_landmarks);

    GeometryPipelineMetadata metadata_;
    PerspectiveCamera perspective_camera_;
    std::unique_ptr<GeometryPipeline> geometry_pipeline_;
};

}  // namespace custom_face_geometry

extern "C" {

CUSTOM_API CUSTOM_RET face_mesh_calculator_new(
    custom_face_geometry::FaceMeshCalculator** face_mesh_calculator);
CUSTOM_API CUSTOM_RET face_mesh_calculator_open(
    custom_face_geometry::FaceMeshCalculator* face_mesh_calculator,
    const char* face_geometry_pipeline_metadata,
    int buffer_size);
CUSTOM_API CUSTOM_RET face_mesh_calculator_process(
    custom_face_geometry::FaceMeshCalculator* face_mesh_calculator,
    int image_width,
    int image_height,
    const float* multi_face_landmarks,
    int num_faces,
    float* face_geometry_pose_mat);
CUSTOM_API CUSTOM_RET face_mesh_calculator_destroy(
    custom_face_geometry::FaceMeshCalculator* face_mesh_calculator);

}  // extern "C"

#endif  // CUSTOM_CALCULATOR_H
