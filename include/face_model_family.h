#ifndef FACE_MODEL_FAMILY_H
#define FACE_MODEL_FAMILY_H
#include "utils.h"
#include "BaseONNX.h"
#include "tools/nms/nms.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace gusto_mp_face{
class MediaPipeDetectorResult : public PostProcessResult {
public:
    std::vector<GustoRect> boxes;
    std::vector<std::vector<float>> scores;
};


class MediapipeFaceLandmarkResult : public PostProcessResult {
public:
    std::vector<cv::Point3f> points;
    float tongueOut;
    float score;
};


class FaceDetector : public BaseONNX {
    public:
        FaceDetector(const std::string& model_path, const std::string& config_path);
        FaceDetector(std::unique_ptr<basic_model_config> config);

        // std::vector<float> preprocess(const cv::Mat& image);
        std::unique_ptr<PostProcessResult> postprocess(const std::vector<Ort::Value>& net_out, const cv::Mat& raw);

        std::unique_ptr<PostProcessResult> forward(const cv::Mat& raw) override;


        cv::Mat draw_boxes(cv::Mat raw, const std::vector<GustoRect>& boxes, const std::vector<std::vector<float>>& scores, const std::vector<int>& indices, const std::vector<int>& indices_cls);

    private:
        int anchor_rows, anchor_cols;
        std::vector<std::vector<float>> anchors;
        int INPUT_SIZE;
        std::map<int, std::string> class_mapper;

        std::vector<GustoRect> decode_boxes(const float* raw_boxes, const std::vector<std::vector<float>>& anchors);
        std::vector<std::vector<float>> LoadBinaryFile2D(const std::string& filePath, int rows, int cols);
};


class FaceLandmarker : public BaseONNX {
    public:
        FaceLandmarker(const std::string& model_path, const std::string& config_path);
        FaceLandmarker(std::unique_ptr<basic_model_config> config);
        std::tuple<cv::Mat, std::vector<int>> crop_face(const cv::Mat& image, const std::vector<int>& box);
        // std::vector<float> preprocess(const cv::Mat& image);
        std::unique_ptr<PostProcessResult> postprocess(const std::vector<Ort::Value>& net_out);

        std::unique_ptr<PostProcessResult> forward(const cv::Mat& raw) override;


        cv::Mat draw_points(cv::Mat image, const std::vector<cv::Point3f>& points, const cv::Point& offset = cv::Point(0, 0), bool display_z = false);
};

} // //namespace gusto_mp_face
#endif // FACE_MODEL_FAMILY_H