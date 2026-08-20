#include "multi_stage_face_geometry_3d.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <tuple>
#include <utility>

namespace {

constexpr float kMinimumLandmarkScore = 0.49F;
constexpr float kMinimumAxisLength = 20.0F;
constexpr float kAxisLengthRatio = 0.08F;
constexpr float kInvalidPoseValueThreshold = -9000.0F;
constexpr float kPi = 3.14159265358979323846F;
constexpr float kDegenerateEyeDistancePx = 1e-3F;
constexpr float kVerticalFovDegrees = 63.0F;
// face_landmark_landmarks_to_roi.pbtxt
constexpr int kLandmarkRotationStart = 33;   // left side of left eye
constexpr int kLandmarkRotationEnd = 263;    // right side of right eye
constexpr float kLandmarkRoiScale = 1.5F;
// MediaPipe FaceLandmarkLandmarksToRoi uses square_long on the tight
// landmark box. Detection ROI stays anisotropic 1.5.
constexpr bool kLandmarkRoiSquareLong = true;

constexpr float kLostAabbMinSide = 20.0F;
constexpr float kLostOutsideFraction = 0.30F;
constexpr float kLostCenterJumpFrac = 0.40F;
constexpr float kMatchIouThreshold = 0.30F;
constexpr double kCropDarkMean = 8.0;

float NormalizeRadians(float angle) {
    return angle - 2.0F * kPi * std::floor((angle + kPi) / (2.0F * kPi));
}

bool HasTrackingMesh(const std::vector<cv::Point3f>& points) {
    return points.size() > static_cast<size_t>(kLandmarkRotationEnd);
}

struct Aabb {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const { return std::max(0.0F, width()) * std::max(0.0F, height()); }
    float cx() const { return 0.5F * (x1 + x2); }
    float cy() const { return 0.5F * (y1 + y2); }
};

bool ComputeLandmarkAabb(const std::vector<cv::Point3f>& points, Aabb& aabb) {
    bool any = false;
    float xmin = 0.0F;
    float ymin = 0.0F;
    float xmax = 0.0F;
    float ymax = 0.0F;
    for (const auto& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            continue;
        }
        if (!any) {
            xmin = xmax = point.x;
            ymin = ymax = point.y;
            any = true;
        } else {
            xmin = std::min(xmin, point.x);
            ymin = std::min(ymin, point.y);
            xmax = std::max(xmax, point.x);
            ymax = std::max(ymax, point.y);
        }
    }
    if (!any) {
        return false;
    }
    aabb.x1 = xmin;
    aabb.y1 = ymin;
    aabb.x2 = xmax;
    aabb.y2 = ymax;
    return true;
}

Aabb DetectionAabb(const CustomRect& box, int width, int height) {
    Aabb aabb;
    aabb.x1 = box.x1 * static_cast<float>(width);
    aabb.y1 = box.y1 * static_cast<float>(height);
    aabb.x2 = box.x2 * static_cast<float>(width);
    aabb.y2 = box.y2 * static_cast<float>(height);
    return aabb;
}

float IntersectionOverUnion(const Aabb& a, const Aabb& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float inter =
        std::max(0.0F, ix2 - ix1) * std::max(0.0F, iy2 - iy1);
    const float uni = a.area() + b.area() - inter;
    if (uni <= 1e-6F) {
        return 0.0F;
    }
    return inter / uni;
}

float FractionOutsideImage(
    const std::vector<cv::Point3f>& points, int width, int height) {
    if (points.empty()) {
        return 1.0F;
    }
    int outside = 0;
    int counted = 0;
    const float w = static_cast<float>(width);
    const float h = static_cast<float>(height);
    for (const auto& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            ++outside;
            ++counted;
            continue;
        }
        ++counted;
        if (point.x < 0.0F || point.y < 0.0F || point.x >= w || point.y >= h) {
            ++outside;
        }
    }
    if (counted == 0) {
        return 1.0F;
    }
    return static_cast<float>(outside) / static_cast<float>(counted);
}

bool CropTooDark(const cv::Mat& crop) {
    if (crop.empty()) {
        return true;
    }
    const cv::Scalar mean = cv::mean(crop);
    const double intensity =
        (mean[0] + mean[1] + mean[2]) /
        (crop.channels() > 0 ? std::min(crop.channels(), 3) : 1);
    return intensity < kCropDarkMean;
}

bool IsGeometryLost(
    const std::vector<cv::Point3f>& points,
    const std::vector<cv::Point3f>* previous,
    int width,
    int height) {
    Aabb aabb;
    if (!ComputeLandmarkAabb(points, aabb)) {
        return true;
    }
    if (aabb.area() < kLostAabbMinSide * kLostAabbMinSide) {
        return true;
    }
    if (FractionOutsideImage(points, width, height) > kLostOutsideFraction) {
        return true;
    }
    if (previous != nullptr) {
        Aabb prev_aabb;
        if (ComputeLandmarkAabb(*previous, prev_aabb)) {
            const float jump = std::hypot(
                aabb.cx() - prev_aabb.cx(), aabb.cy() - prev_aabb.cy());
            const float diag = std::hypot(
                static_cast<float>(width), static_cast<float>(height));
            if (jump > kLostCenterJumpFrac * diag) {
                return true;
            }
        }
    }
    return false;
}

// MediaPipe FaceLandmarkLandmarksToRoi:
//   1) tight AABB of all landmarks (image pixels)
//   2) rotation from 33 -> 263, target_angle_degrees = 0
//   3) RectTransformation scale 1.5, square_long
// Center is the landmark box, not the detector box.
bool LandmarksToTrackingRoi(
    const std::vector<cv::Point3f>& image_points,
    custom_mp_face::FaceRotatedRect& roi) {
    if (!HasTrackingMesh(image_points)) {
        return false;
    }

    float xmin = image_points[0].x;
    float ymin = image_points[0].y;
    float xmax = image_points[0].x;
    float ymax = image_points[0].y;
    for (const auto& point : image_points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            continue;
        }
        xmin = std::min(xmin, point.x);
        ymin = std::min(ymin, point.y);
        xmax = std::max(xmax, point.x);
        ymax = std::max(ymax, point.y);
    }
    const float box_w = xmax - xmin;
    const float box_h = ymax - ymin;
    if (!std::isfinite(box_w) || !std::isfinite(box_h) || box_w <= 1e-3F ||
        box_h <= 1e-3F) {
        return false;
    }

    const float x0 = image_points[kLandmarkRotationStart].x;
    const float y0 = image_points[kLandmarkRotationStart].y;
    const float x1 = image_points[kLandmarkRotationEnd].x;
    const float y1 = image_points[kLandmarkRotationEnd].y;
    const float eye_dist = std::hypot(x1 - x0, y1 - y0);
    if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) ||
        !std::isfinite(y1) || eye_dist <= kDegenerateEyeDistancePx) {
        return false;
    }

    roi.center_x = 0.5F * (xmin + xmax);
    roi.center_y = 0.5F * (ymin + ymax);
    roi.rotation = NormalizeRadians(0.0F - std::atan2(-(y1 - y0), x1 - x0));
    if (kLandmarkRoiSquareLong) {
        const float long_side = std::max(box_w, box_h);
        roi.width = kLandmarkRoiScale * long_side;
        roi.height = kLandmarkRoiScale * long_side;
    } else {
        roi.width = kLandmarkRoiScale * box_w;
        roi.height = kLandmarkRoiScale * box_h;
    }
    return std::isfinite(roi.center_x) && std::isfinite(roi.center_y) &&
           std::isfinite(roi.width) && std::isfinite(roi.height) &&
           std::isfinite(roi.rotation);
}

cv::Point2f ProjectCamera(
    float X,
    float Y,
    float Z,
    float fx,
    float fy,
    float cx,
    float cy,
    int cols) {
    if (!std::isfinite(X) || !std::isfinite(Y) || !std::isfinite(Z)) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    if (std::fabs(Z) < 1e-6F) {
        Z = (Z < 0.0F) ? -1e-6F : 1e-6F;
    }
    float u = fx * (X / Z) + cx;
    float v = fy * (Y / Z) + cy;
    u = static_cast<float>(cols) - u;
    return {u, v};
}

void DrawCoordinateAxes(
    cv::Mat& frame,
    const custom_face_geometry::FaceGeometry& geometry,
    const cv::Point2f& origin,
    float fx,
    float fy,
    float cx,
    float cy) {
    const auto& matrix = geometry.pose_transform_matrix;
    if (matrix.rows != 4 || matrix.cols != 4) {
        return;
    }

    float R[3][3];
    float t[3];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            const float value = matrix.at(row, column);
            if (!std::isfinite(value) ||
                value < kInvalidPoseValueThreshold) {
                return;
            }
            R[row][column] = value;
        }
        const float tv = matrix.at(row, 3);
        if (!std::isfinite(tv) || tv < kInvalidPoseValueThreshold) {
            return;
        }
        t[row] = tv;
    }

    const float pts[4][3] = {
        {10.0F, 0.0F, 0.0F},
        {0.0F, 10.0F, 0.0F},
        {0.0F, 0.0F, 10.0F},
        {0.0F, 0.0F, 0.0F},
    };
    cv::Point2f axis[4];
    for (int i = 0; i < 4; ++i) {
        const float X =
            R[0][0] * pts[i][0] + R[0][1] * pts[i][1] + R[0][2] * pts[i][2] +
            t[0];
        const float Y =
            R[1][0] * pts[i][0] + R[1][1] * pts[i][1] + R[1][2] * pts[i][2] +
            t[1];
        const float Z =
            R[2][0] * pts[i][0] + R[2][1] * pts[i][1] + R[2][2] * pts[i][2] +
            t[2];
        axis[i] = ProjectCamera(X, Y, Z, fx, fy, cx, cy, frame.cols);
        if (!std::isfinite(axis[i].x) || !std::isfinite(axis[i].y)) {
            return;
        }
    }

    const cv::Point2f x_axis = axis[0] - axis[3];
    const cv::Point2f y_axis = axis[1] - axis[3];
    const cv::Point2f z_axis = axis[2] - axis[3];
    cv::circle(frame, origin, 3, cv::Scalar(255, 255, 255), -1);
    cv::line(frame, origin, origin + x_axis, cv::Scalar(255, 0, 0), 2);
    cv::line(frame, origin, origin + y_axis, cv::Scalar(0, 255, 0), 2);
    cv::line(frame, origin, origin + z_axis, cv::Scalar(0, 0, 255), 2);
    (void)kMinimumAxisLength;
    (void)kAxisLengthRatio;
}

}  // namespace

float FaceGeometryTracker3D::OneEuroFilter::Apply(float value, float freq) {
    constexpr float kTwoPi = 2.0F * 3.14159265358979323846F;
    if (!std::isfinite(value) || freq <= 1e-6F) {
        return value;
    }
    if (!initialized) {
        initialized = true;
        x_prev = value;
        dx_prev = 0.0F;
        return value;
    }
    const auto alpha = [kTwoPi](float cutoff, float frequency) {
        const float tau = 1.0F / (kTwoPi * std::max(cutoff, 1e-6F));
        const float te = 1.0F / frequency;
        return 1.0F / (1.0F + tau / te);
    };
    const auto lowpass = [](float x, float prev, float a) {
        return a * x + (1.0F - a) * prev;
    };
    const float dx = (value - x_prev) * freq;
    const float edx = lowpass(dx, dx_prev, alpha(dcutoff, freq));
    dx_prev = edx;
    const float cutoff = mincutoff + beta * std::fabs(edx);
    const float hatx = lowpass(value, x_prev, alpha(cutoff, freq));
    x_prev = hatx;
    return hatx;
}

void FaceGeometryTracker3D::EnsureFilters(FaceTrack& track) const {
    const size_t needed = track.landmarks.size() * 3;
    if (track.filters.size() != needed) {
        track.filters.assign(needed, OneEuroFilter{});
    }
    for (auto& filter : track.filters) {
        filter.mincutoff = one_euro_mincutoff_;
        filter.beta = one_euro_beta_;
        filter.dcutoff = one_euro_dcutoff_;
    }
}

void FaceGeometryTracker3D::ResetOneEuro(FaceTrack& track) const {
    EnsureFilters(track);
    for (auto& filter : track.filters) {
        filter.Reset();
        filter.mincutoff = one_euro_mincutoff_;
        filter.beta = one_euro_beta_;
        filter.dcutoff = one_euro_dcutoff_;
    }
}

void FaceGeometryTracker3D::ApplyOneEuro(FaceTrack& track) {
    EnsureFilters(track);
    for (size_t i = 0; i < track.landmarks.size(); ++i) {
        track.landmarks[i].x =
            track.filters[i * 3 + 0].Apply(track.landmarks[i].x, one_euro_freq_);
        track.landmarks[i].y =
            track.filters[i * 3 + 1].Apply(track.landmarks[i].y, one_euro_freq_);
        track.landmarks[i].z =
            track.filters[i * 3 + 2].Apply(track.landmarks[i].z, one_euro_freq_);
    }
}

void FaceGeometryTracker3D::SetOneEuroFrequency(float hz) {
    if (std::isfinite(hz) && hz > 1e-3F) {
        one_euro_freq_ = hz;
    }
}

void FaceGeometryTracker3D::SetOneEuroParams(
    float mincutoff, float beta, float dcutoff) {
    if (std::isfinite(mincutoff) && mincutoff > 0.0F) {
        one_euro_mincutoff_ = mincutoff;
    }
    if (std::isfinite(beta) && beta >= 0.0F) {
        one_euro_beta_ = beta;
    }
    if (std::isfinite(dcutoff) && dcutoff > 0.0F) {
        one_euro_dcutoff_ = dcutoff;
    }
}

void FaceGeometryTracker3D::ComputeFovIntrinsics(
    int width, int height, float* fx, float* fy, float* cx, float* cy) {
    constexpr float kFovDeg = 63.0F;
    constexpr float kPiLocal = 3.14159265358979323846F;
    const float f =
        static_cast<float>(height) /
        (2.0F * std::tan(kFovDeg * kPiLocal / 360.0F));
    if (fx) {
        *fx = f;
    }
    if (fy) {
        *fy = f;
    }
    if (cx) {
        *cx = 0.5F * static_cast<float>(width);
    }
    if (cy) {
        *cy = 0.5F * static_cast<float>(height);
    }
}

FaceGeometryTracker3D::FaceGeometryTracker3D(
    const std::string& face_detector_path,
    const std::string& face_detector_config_path,
    const std::string& face_landmarker_path,
    const std::string& face_landmarker_config_path,
    const std::string& face_geometry_pipeline_metadata,
    int detect_interval)
    : face_detector_(std::make_unique<custom_mp_face::FaceDetector>(
          face_detector_path, face_detector_config_path)),
      face_landmarker_(std::make_unique<custom_mp_face::FaceLandmarker>(
          face_landmarker_path, face_landmarker_config_path)),
      face_mesh_calculator_(
          std::make_unique<custom_face_geometry::FaceMeshCalculator>()),
      detect_interval_(std::max(1, detect_interval)) {
    initialization_status_ =
        face_mesh_calculator_->Open(face_geometry_pipeline_metadata);
}

CUSTOM_RET FaceGeometryTracker3D::Detect(
    const cv::Mat& frame,
    bool display_keypoints,
    bool display_coord) {
    face_geometries_.clear();
    rendered_frame_.release();
    last_crop_image_.release();
    last_dst_to_src_.release();
    last_crop_landmarks_.clear();
    last_image_landmarks_.clear();
    last_crop_from_tracking_ = false;
    image_landmarks_.clear();
    last_stats_ = DetectStats{};
    last_detect_log_.clear();

    if (frame.empty()) {
        return CustomStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    rendered_frame_ = frame.clone();

    if (initialization_status_ != CustomStatus::ERR_OK) {
        return initialization_status_;
    }

    if (!face_detector_ || !face_landmarker_ || !face_mesh_calculator_) {
        return CustomStatus::ERR_GENERAL_ERROR;
    }

    try {
        const int img_w = frame.cols;
        const int img_h = frame.rows;
        last_stats_.n_tracks = static_cast<int>(tracks_.size());

        struct LandmarkRun {
            std::vector<cv::Point3f> crop_points;
            std::vector<cv::Point3f> image_points;
            custom_mp_face::FaceCropResult crop;
            bool ok = false;
        };

        auto run_landmarker =
            [&](const custom_mp_face::FaceCropResult& crop) -> LandmarkRun {
            LandmarkRun result;
            result.crop = crop;
            if (crop.image.empty() || crop.dst_to_src.empty()) {
                return result;
            }
            if (CropTooDark(crop.image)) {
                return result;
            }
            auto landmarker_output = face_landmarker_->forward(crop.image);
            auto* landmarker_result =
                dynamic_cast<custom_mp_face::MediapipeFaceLandmarkResult*>(
                    landmarker_output.get());
            if (landmarker_result == nullptr ||
                landmarker_result->score < kMinimumLandmarkScore ||
                landmarker_result->points.empty()) {
                return result;
            }
            result.crop_points = landmarker_result->points;
            result.image_points =
                custom_mp_face::FaceLandmarker::MapLandmarksToImage(
                    landmarker_result->points, crop.dst_to_src);
            result.ok = !result.image_points.empty();
            return result;
        };

        std::vector<FaceTrack> prev_tracks = std::move(tracks_);
        tracks_.clear();
        tracks_.reserve(prev_tracks.size());

        std::vector<LandmarkRun> live_runs;
        live_runs.reserve(prev_tracks.size());
        int n_lost = 0;

        for (auto& prev : prev_tracks) {
            custom_mp_face::FaceRotatedRect roi;
            if (!LandmarksToTrackingRoi(prev.landmarks, roi)) {
                ++n_lost;
                continue;
            }
            const custom_mp_face::FaceCropResult crop =
                face_landmarker_->crop_face_roi(frame, roi);
            LandmarkRun run = run_landmarker(crop);
            if (!run.ok ||
                IsGeometryLost(run.image_points, &prev.landmarks, img_w, img_h)) {
                ++n_lost;
                continue;
            }
            FaceTrack live;
            live.landmarks = std::move(run.image_points);
            live.filters = std::move(prev.filters);
            live.source = "track";
            ApplyOneEuro(live);
            run.image_points = live.landmarks;
            tracks_.push_back(std::move(live));
            live_runs.push_back(std::move(run));
        }

        last_stats_.n_lost = n_lost;

        const bool periodic =
            (frame_index_ % detect_interval_) == 0;
        const bool run_detector_now =
            tracks_.empty() || n_lost > 0 || periodic;
        ++frame_index_;

        std::vector<CustomRect> detections;
        std::vector<std::array<cv::Point2f, 6>> detection_kps;

        if (run_detector_now) {
            auto detector_output = face_detector_->forward(frame);
            auto* detector_result =
                dynamic_cast<custom_mp_face::MediaPipeDetectorResult*>(
                    detector_output.get());
            if (detector_result == nullptr) {
                return CustomStatus::ERR_GENERAL_ERROR;
            }
            last_stats_.detector_ran = true;
            detections.reserve(detector_result->boxes.size());
            detection_kps.reserve(detector_result->boxes.size());
            for (size_t i = 0; i < detector_result->boxes.size(); ++i) {
                const auto& box = detector_result->boxes[i];
                if (std::isfinite(box.x1) && std::isfinite(box.y1) &&
                    std::isfinite(box.x2) && std::isfinite(box.y2) &&
                    box.x2 > box.x1 && box.y2 > box.y1) {
                    detections.push_back(box);
                    if (i < detector_result->keypoints.size()) {
                        detection_kps.push_back(detector_result->keypoints[i]);
                    } else {
                        detection_kps.push_back({});
                    }
                }
            }
        }
        last_stats_.n_detect = static_cast<int>(detections.size());

        if (!detections.empty()) {
            std::vector<Aabb> live_aabbs(tracks_.size());
            std::vector<char> live_has_aabb(tracks_.size(), 0);
            for (size_t t = 0; t < tracks_.size(); ++t) {
                Aabb aabb;
                if (ComputeLandmarkAabb(tracks_[t].landmarks, aabb)) {
                    live_aabbs[t] = aabb;
                    live_has_aabb[t] = 1;
                }
            }
            std::vector<std::tuple<float, int, int>> pairs;
            pairs.reserve(detections.size() * tracks_.size());
            for (size_t d = 0; d < detections.size(); ++d) {
                const Aabb det_aabb = DetectionAabb(detections[d], img_w, img_h);
                for (size_t t = 0; t < tracks_.size(); ++t) {
                    if (!live_has_aabb[t]) {
                        continue;
                    }
                    const float iou = IntersectionOverUnion(det_aabb, live_aabbs[t]);
                    if (iou >= kMatchIouThreshold) {
                        pairs.emplace_back(iou, static_cast<int>(d), static_cast<int>(t));
                    }
                }
            }
            std::sort(
                pairs.begin(), pairs.end(),
                [](const auto& a, const auto& b) {
                    return std::get<0>(a) > std::get<0>(b);
                });
            std::vector<char> det_matched(detections.size(), 0);
            std::vector<char> track_matched(tracks_.size(), 0);
            for (const auto& pair : pairs) {
                const int d = std::get<1>(pair);
                const int t = std::get<2>(pair);
                if (det_matched[d] || track_matched[t]) {
                    continue;
                }
                det_matched[d] = 1;
                track_matched[t] = 1;
            }

            int n_spawn = 0;
            for (size_t d = 0; d < detections.size(); ++d) {
                if (det_matched[d]) {
                    continue;
                }
                const std::array<cv::Point2f, 6>* kps =
                    (d < detection_kps.size()) ? &detection_kps[d] : nullptr;
                const custom_mp_face::FaceCropResult crop =
                    face_landmarker_->crop_face_roi(frame, detections[d], kps);
                LandmarkRun run = run_landmarker(crop);
                if (!run.ok ||
                    IsGeometryLost(run.image_points, nullptr, img_w, img_h)) {
                    continue;
                }
                FaceTrack spawned;
                spawned.landmarks = std::move(run.image_points);
                spawned.source = "detect";
                ResetOneEuro(spawned);
                ApplyOneEuro(spawned);
                run.image_points = spawned.landmarks;
                tracks_.push_back(std::move(spawned));
                live_runs.push_back(std::move(run));
                ++n_spawn;
            }
            last_stats_.n_spawn = n_spawn;
        }

        last_stats_.sources.reserve(tracks_.size());
        image_landmarks_.reserve(tracks_.size());
        std::vector<custom_face_geometry::NormalizedLandmarkList>
            multi_face_landmarks;
        std::vector<cv::Point2f> face_origins;
        multi_face_landmarks.reserve(tracks_.size());
        face_origins.reserve(tracks_.size());

        for (size_t i = 0; i < tracks_.size(); ++i) {
            const auto& track = tracks_[i];
            last_stats_.sources.push_back(track.source);
            image_landmarks_.push_back(track.landmarks);

            custom_face_geometry::NormalizedLandmarkList face_landmarks;
            face_landmarks.landmark.reserve(track.landmarks.size());
            for (const auto& point : track.landmarks) {
                custom_face_geometry::NormalizedLandmark landmark{};
                landmark.x = point.x / static_cast<float>(img_w);
                landmark.y = point.y / static_cast<float>(img_h);
                landmark.z = point.z / static_cast<float>(img_w);
                face_landmarks.landmark.push_back(landmark);
            }
            multi_face_landmarks.push_back(std::move(face_landmarks));

            cv::Point2f origin(0.5F * static_cast<float>(img_w),
                               0.5F * static_cast<float>(img_h));
            if (track.landmarks.size() > 4 &&
                std::isfinite(track.landmarks[4].x) &&
                std::isfinite(track.landmarks[4].y)) {
                origin = cv::Point2f(track.landmarks[4].x, track.landmarks[4].y);
            } else if (i < live_runs.size() &&
                       !live_runs[i].crop.dst_to_src.empty() &&
                       !live_runs[i].crop.image.empty()) {
                origin = custom_mp_face::FaceLandmarker::MapCropToImage(
                    live_runs[i].crop.dst_to_src,
                    0.5F * static_cast<float>(live_runs[i].crop.image.cols),
                    0.5F * static_cast<float>(live_runs[i].crop.image.rows));
            }
            face_origins.push_back(origin);

            if (last_crop_image_.empty() && i < live_runs.size()) {
                last_crop_image_ = live_runs[i].crop.image;
                last_dst_to_src_ = live_runs[i].crop.dst_to_src;
                last_crop_landmarks_ = live_runs[i].crop_points;
                last_image_landmarks_ = track.landmarks;
                last_crop_from_tracking_ = (track.source == "track");
            }

            if (display_keypoints) {
                rendered_frame_ = face_landmarker_->draw_points(
                    rendered_frame_, track.landmarks, cv::Point(0, 0));
            }
        }

        if (!image_landmarks_.empty() && last_image_landmarks_.empty()) {
            last_image_landmarks_ = image_landmarks_[0];
        }

        std::ostringstream log;
        log << "n_tracks=" << last_stats_.n_tracks
            << " n_detect=" << last_stats_.n_detect
            << " n_spawn=" << last_stats_.n_spawn
            << " n_lost=" << last_stats_.n_lost
            << " detector=" << (last_stats_.detector_ran ? 1 : 0)
            << " n_faces=" << tracks_.size()
            << " source=";
        for (size_t i = 0; i < last_stats_.sources.size(); ++i) {
            if (i) {
                log << ",";
            }
            log << last_stats_.sources[i];
        }
        if (last_stats_.sources.empty()) {
            log << "-";
        }
        last_detect_log_ = log.str();

        if (multi_face_landmarks.empty()) {
            return CustomStatus::ERR_OK;
        }

        auto [face_geometries, process_status] =
            face_mesh_calculator_->Process(
                std::make_pair(frame.cols, frame.rows),
                multi_face_landmarks);
        face_geometries_ = std::move(face_geometries);

        if (display_coord &&
            (process_status == CustomStatus::ERR_OK ||
             process_status == CustomStatus::ERR_PARTIAL_FAIL) &&
            face_geometries_.size() == face_origins.size()) {
            float fx = 0.0F;
            float fy = 0.0F;
            float cx = 0.0F;
            float cy = 0.0F;
            ComputeFovIntrinsics(img_w, img_h, &fx, &fy, &cx, &cy);
            for (std::size_t index = 0; index < face_geometries_.size();
                 ++index) {
                DrawCoordinateAxes(
                    rendered_frame_, face_geometries_[index],
                    face_origins[index], fx, fy, cx, cy);
            }
        }

        return process_status;
    } catch (const cv::Exception& exception) {
        std::cerr << "Face geometry OpenCV error: " << exception.what()
                  << std::endl;
    } catch (const std::exception& exception) {
        std::cerr << "Face geometry inference error: " << exception.what()
                  << std::endl;
    }

    face_geometries_.clear();
    rendered_frame_ = frame.clone();
    tracks_.clear();
    image_landmarks_.clear();
    return CustomStatus::ERR_GENERAL_ERROR;
}

cv::Mat FaceGeometryTracker3D::GetRenderedFrame() const {
    return rendered_frame_;
}

const std::vector<custom_face_geometry::FaceGeometry>&
FaceGeometryTracker3D::GetFaceGeometries() const noexcept {
    return face_geometries_;
}

cv::Mat FaceGeometryTracker3D::GetLastCropImage() const {
    return last_crop_image_;
}

cv::Mat FaceGeometryTracker3D::GetLastDstToSrc() const {
    return last_dst_to_src_;
}

const std::vector<cv::Point3f>&
FaceGeometryTracker3D::GetLastCropLandmarks() const {
    return last_crop_landmarks_;
}

const std::vector<cv::Point3f>&
FaceGeometryTracker3D::GetLastImageLandmarks() const {
    return last_image_landmarks_;
}

const std::vector<std::vector<cv::Point3f>>&
FaceGeometryTracker3D::GetImageLandmarks() const noexcept {
    return image_landmarks_;
}

bool FaceGeometryTracker3D::LastCropFromTracking() const {
    return last_crop_from_tracking_;
}

const FaceGeometryTracker3D::DetectStats&
FaceGeometryTracker3D::GetDetectStats() const noexcept {
    return last_stats_;
}

const std::string& FaceGeometryTracker3D::GetLastDetectLog() const noexcept {
    return last_detect_log_;
}
