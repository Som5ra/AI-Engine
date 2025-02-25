#ifndef MULTI_STAGE_FACE_GEOMETRY_3D_H
#define MULTI_STAGE_FACE_GEOMETRY_3D_H
#include "tools/face_geometry/geometry_pipeline.h"
#include "tools/face_geometry/face_geometry.h"
#include "tools/face_geometry/procrustes_solver.h"
#include "tools/face_geometry/calculator.h"
#include "tools/nms/nms.h"
#include "utils.h"
#include "face_model_family.h"

#include <onnxruntime_cxx_api.h>

#include <tuple>
#include <map>
#include <cmath>

#if defined(BUILD_PLATFORM_LINUX) && defined(DEBUG)
#include <opencv2/opencv.hpp>
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif


#define GUSTO_RET int


class FaceGeometryTracker3D {
public:
    FaceGeometryTracker3D(const std::string& face_detector_path,
                       const std::string& face_detector_config_path,
                       const std::string& face_landmarker_path,
                       const std::string& face_landmarker_config_path,
                       const std::string face_GeometryPipelineMetadata,
                       int detect_interval = 0);
    ~FaceGeometryTracker3D();

    GUSTO_RET Detect(const cv::Mat& frame, bool display_keypoints = true, bool display_coord = true);
    cv::Mat GetRenderedFrame();
private:
    std::unique_ptr<gusto_mp_face::FaceDetector> face_detector;
    std::unique_ptr<gusto_mp_face::FaceLandmarker> face_landmarker;
    std::unique_ptr<gusto_face_geometry::FaceMeshCalculator> face_mesh_calculator;
    
    std::vector<gusto_face_geometry::NormalizedLandmarkList> multi_face_landmarks;

    // for display
    cv::Mat rendered_frame;
    // std::vector<cv::Point3f> face_keypoints;
    // std::vector<cv::Point> box_to_crop_with_margin;


    int detect_interval;
    int num_frames;
    
};

#endif // MULTI_STAGE_FACE_GEOMETRY_3D_H