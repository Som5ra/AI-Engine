#ifndef FACE_MODEL_FAMILY_H
#define FACE_MODEL_FAMILY_H
#include "utils.h"
#include "BaseONNX.h"
#include "tools/nms/nms.h"
#include <array>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace custom_mp_face{
class MediaPipeDetectorResult : public PostProcessResult {
public:
    std::vector<CustomRect> boxes;
    std::vector<std::vector<float>> scores;
    // BlazeFace 6 keypoints in normalized image coords, aligned with boxes.
    // 0=left eye, 1=right eye, then nose/mouth/ear as in the 16-float box.
    std::vector<std::array<cv::Point2f, 6>> keypoints;
};

class MediapipeFaceLandmarkResult : public PostProcessResult {
public:
    std::vector<cv::Point3f> points;
    float tongueOut;
    float score;
};

// Rotated (or AABB-fallback) face crop plus the dst->src affine that maps
// landmarker crop pixels back into the original image.
struct FaceCropResult {
    cv::Mat image;
    cv::Mat dst_to_src;  // 2x3 CV_32F, OpenCV warpAffine dst->src
    bool rotated = false;
    std::vector<int> aabb_margin;  // {top, left, bottom, right} AABB fallback
};

// Pixel-space rotated sub-rect for crop_face_roi (already scaled).
// center/width/height in image pixels; rotation in radians
// (MediaPipe NormalizeRadians, target_angle=0).
struct FaceRotatedRect {
    float center_x = 0.0f;
    float center_y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float rotation = 0.0f;
};

class FaceDetector : public BaseONNX {
    public:
        FaceDetector(const std::string& model_path, const std::string& config_path);
        FaceDetector(std::unique_ptr<basic_model_config> config);

        // MediaPipe short-range BlazeFace min_score_thresh default is 0.5.
        void SetScoreThreshold(float score_thr) { score_thr_ = score_thr; }
        void SetNmsThreshold(float nms_thr) { nms_thr_ = nms_thr; }
        float GetScoreThreshold() const { return score_thr_; }
        float GetNmsThreshold() const { return nms_thr_; }

        // std::vector<float> preprocess(const cv::Mat& image);
        std::unique_ptr<PostProcessResult> postprocess(const std::vector<Ort::Value>& net_out, const cv::Mat& raw);

        std::unique_ptr<PostProcessResult> forward(const cv::Mat& raw) override;


        cv::Mat draw_boxes(cv::Mat raw, const std::vector<CustomRect>& boxes, const std::vector<std::vector<float>>& scores, const std::vector<int>& indices, const std::vector<int>& indices_cls);

    private:
        int anchor_rows, anchor_cols;
        std::vector<std::vector<float>> anchors;
        int INPUT_SIZE;
        std::map<int, std::string> class_mapper;
        float score_thr_ = 0.5f;
        float nms_thr_ = 0.5f;

        std::vector<CustomRect> decode_boxes(const float* raw_boxes, const std::vector<std::vector<float>>& anchors);
        std::vector<std::array<cv::Point2f, 6>> decode_keypoints(
            const float* raw_boxes, const std::vector<std::vector<float>>& anchors);
        std::vector<std::vector<float>> LoadBinaryFile2D(const std::string& filePath, int rows, int cols);
};


class FaceLandmarker : public BaseONNX {
    public:
        FaceLandmarker(const std::string& model_path, const std::string& config_path);
        FaceLandmarker(std::unique_ptr<basic_model_config> config);
        std::tuple<cv::Mat, std::vector<int>> crop_face(const cv::Mat& image, const std::vector<int>& box);
        // MediaPipe rotated ROI. Falls back to AABB+0.25 margin if keypoints
        // are missing or left/right eye distance is ~0.
        FaceCropResult crop_face_roi(
            const cv::Mat& image,
            const CustomRect& box,
            const std::array<cv::Point2f, 6>* keypoints = nullptr);
        // Explicit rotated rect (center, w, h, rotation). Does not use the
        // detector box. Detection path remains crop_face_roi(image, box, kps).
        FaceCropResult crop_face_roi(
            const cv::Mat& image,
            const FaceRotatedRect& roi);
        static cv::Point2f MapCropToImage(const cv::Mat& dst_to_src, float x, float y);
        static std::vector<cv::Point3f> MapLandmarksToImage(
            const std::vector<cv::Point3f>& crop_points, const cv::Mat& dst_to_src);
        // std::vector<float> preprocess(const cv::Mat& image);
        std::unique_ptr<PostProcessResult> postprocess(const std::vector<Ort::Value>& net_out);

        std::unique_ptr<PostProcessResult> forward(const cv::Mat& raw) override;


        cv::Mat draw_points(cv::Mat image, const std::vector<cv::Point3f>& points, const cv::Point& offset = cv::Point(0, 0), bool display_z = false);
};

} // //namespace custom_mp_face
#endif // FACE_MODEL_FAMILY_H
