
#ifndef GUSTO_CALCULATOR_H
#define GUSTO_CALCULATOR_H
#include "utils.h"
#include "tools/face_geometry/geometry_pipeline.h"
#include "tools/face_geometry/face_geometry.h"
#include "tools/face_geometry/procrustes_solver.h"

namespace gusto_face_geometry {

    #define GUSTO_RET int
    GUSTO_RET Process(const GeometryPipelineMetadata& metadata, 
                    const PerspectiveCamera& perspective_camera_,
                    std::unique_ptr<GeometryPipeline>& geometry_pipeline_);
    GUSTO_RET Open(std::string face_GeometryPipelineMetadata);

}

#endif // GUSTO_CALCULATOR_H