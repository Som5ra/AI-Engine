#ifndef MULTI_STAGE_FACE_GEOMETRY_3D_H
#define MULTI_STAGE_FACE_GEOMETRY_3D_H

#include "face_model_family.h"
#include "tools/face_geometry/calculator.h"
#include "tools/face_geometry/face_geometry.h"
#include "utils.h"

#include <memory>
#include <string>
#include <vector>

#if defined(BUILD_PLATFORM_LINUX) && defined(DEBUG)
#include <opencv2/opencv.hpp>
#else
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#define GUSTO_RET int

class FaceGeometryTracker3D {
public:
    FaceGeometryTracker3D(
        const std::string& face_detector_path,
        const std::string& face_detector_config_path,
        const std::string& face_landmarker_path,
        const std::string& face_landmarker_config_path,
        const std::string& face_geometry_pipeline_metadata,
        int detect_interval = 0);
    ~FaceGeometryTracker3D() = default;

    GUSTO_RET Detect(
        const cv::Mat& frame,
        bool display_keypoints = true,
        bool display_coord = true);

    cv::Mat GetRenderedFrame() const;
    const std::vector<gusto_face_geometry::FaceGeometry>& GetFaceGeometries() const noexcept;

private:
    std::unique_ptr<gusto_mp_face::FaceDetector> face_detector_;
    std::unique_ptr<gusto_mp_face::FaceLandmarker> face_landmarker_;
    std::unique_ptr<gusto_face_geometry::FaceMeshCalculator> face_mesh_calculator_;

    std::vector<gusto_face_geometry::FaceGeometry> face_geometries_;
    cv::Mat rendered_frame_;

    int initialization_status_ = GustoStatus::ERR_GENERAL_ERROR;
    int detect_interval_ = 0;
    int num_frames_ = 0;
};

#endif  // MULTI_STAGE_FACE_GEOMETRY_3D_H
