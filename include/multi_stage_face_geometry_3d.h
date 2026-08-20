#ifndef MULTI_STAGE_FACE_GEOMETRY_3D_H
#define MULTI_STAGE_FACE_GEOMETRY_3D_H

#include "face_model_family.h"
#include "tools/face_geometry/calculator.h"
#include "tools/face_geometry/face_geometry.h"
#include "utils.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

#if defined(BUILD_PLATFORM_LINUX) && defined(DEBUG)
#include <opencv2/opencv.hpp>
#else
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#define CUSTOM_RET int

class FaceGeometryTracker3D {
public:
    struct DetectStats {
        int n_tracks = 0;
        int n_detect = 0;
        int n_spawn = 0;
        int n_lost = 0;
        bool detector_ran = false;
        // Per live face this frame: "track" or "detect".
        std::vector<std::string> sources;
    };

    FaceGeometryTracker3D(
        const std::string& face_detector_path,
        const std::string& face_detector_config_path,
        const std::string& face_landmarker_path,
        const std::string& face_landmarker_config_path,
        const std::string& face_geometry_pipeline_metadata,
        int detect_interval = 5);
    ~FaceGeometryTracker3D() = default;

    CUSTOM_RET Detect(
        const cv::Mat& frame,
        bool display_keypoints = true,
        bool display_coord = true);

    cv::Mat GetRenderedFrame() const;
    const std::vector<custom_face_geometry::FaceGeometry>& GetFaceGeometries() const noexcept;

    // Last successful crop (face 0) for debug overlays.
    cv::Mat GetLastCropImage() const;
    cv::Mat GetLastDstToSrc() const;
    const std::vector<cv::Point3f>& GetLastCropLandmarks() const;
    const std::vector<cv::Point3f>& GetLastImageLandmarks() const;
    // Image-space landmarks for ALL live faces this frame (filtered).
    const std::vector<std::vector<cv::Point3f>>& GetImageLandmarks() const noexcept;
    bool LastCropFromTracking() const;

    const DetectStats& GetDetectStats() const noexcept;
    const std::string& GetLastDetectLog() const noexcept;

    // One Euro on image-space x,y,z. freq default 30 Hz; tests use 12.
    void SetOneEuroFrequency(float hz);
    void SetOneEuroParams(float mincutoff, float beta, float dcutoff);

    // Geometry pipeline vertical FOV is 63°.
    // fy = fx = H / (2 * tan(63° * π/360)), cx = W/2, cy = H/2.
    static void ComputeFovIntrinsics(
        int width, int height, float* fx, float* fy, float* cx, float* cy);

private:
    struct OneEuroFilter {
        float mincutoff = 1.0f;
        float beta = 0.007f;
        float dcutoff = 1.0f;
        bool initialized = false;
        float x_prev = 0.0f;
        float dx_prev = 0.0f;
        float Apply(float value, float freq);
        void Reset() { initialized = false; }
    };

    struct FaceTrack {
        std::vector<cv::Point3f> landmarks;
        std::vector<OneEuroFilter> filters;
        std::string source = "track";
    };

    void EnsureFilters(FaceTrack& track) const;
    void ApplyOneEuro(FaceTrack& track);
    void ResetOneEuro(FaceTrack& track) const;

    std::unique_ptr<custom_mp_face::FaceDetector> face_detector_;
    std::unique_ptr<custom_mp_face::FaceLandmarker> face_landmarker_;
    std::unique_ptr<custom_face_geometry::FaceMeshCalculator> face_mesh_calculator_;

    std::vector<custom_face_geometry::FaceGeometry> face_geometries_;
    cv::Mat rendered_frame_;

    int initialization_status_ = CustomStatus::ERR_GENERAL_ERROR;
    int detect_interval_ = 5;
    int frame_index_ = 0;
    std::vector<FaceTrack> tracks_;
    std::vector<std::vector<cv::Point3f>> image_landmarks_;
    DetectStats last_stats_;
    std::string last_detect_log_;

    float one_euro_freq_ = 30.0f;
    float one_euro_mincutoff_ = 1.0f;
    float one_euro_beta_ = 0.007f;
    float one_euro_dcutoff_ = 1.0f;

    cv::Mat last_crop_image_;
    cv::Mat last_dst_to_src_;
    std::vector<cv::Point3f> last_crop_landmarks_;
    std::vector<cv::Point3f> last_image_landmarks_;
    bool last_crop_from_tracking_ = false;
};

#endif  // MULTI_STAGE_FACE_GEOMETRY_3D_H
