#ifndef FACE_MODEL_FAMILY_H
#define FACE_MODEL_FAMILY_H
#include "utils.h"
#include "BaseONNX.h"
#include "tools/nms/nms.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class FaceDetector : public BaseONNX {
    public:
        FaceDetector(const std::string& model_path);

        cv::Mat preprocess_img(const cv::Mat& image);

        std::tuple<std::vector<gusto_nms::Rect>, std::vector<std::vector<float>>, std::vector<int>, std::vector<int>> forward(const cv::Mat& raw);
        

        cv::Mat draw_boxes(cv::Mat raw, const std::vector<gusto_nms::Rect>& boxes, const std::vector<std::vector<float>>& scores, const std::vector<int>& indices, const std::vector<int>& indices_cls);

    private:
        int anchor_rows, anchor_cols;
        std::vector<std::vector<float>> anchors;
        int INPUT_SIZE;
        std::map<int, std::string> class_mapper;

        std::vector<gusto_nms::Rect> decode_boxes(const float* raw_boxes, const std::vector<std::vector<float>>& anchors);
        std::vector<std::vector<float>> LoadBinaryFile2D(const std::string& filePath, int rows, int cols);
};

class FaceLandmarker : public BaseONNX {
    public:
        FaceLandmarker(const std::string& model_path);

        cv::Mat crop_face(const cv::Mat& image, const std::vector<int>& box);

        std::tuple<cv::Mat, float, float> preprocess(const cv::Mat& image);

        std::tuple<std::vector<cv::Point3f>, float, float> forward(const cv::Mat& image);
        

        cv::Mat draw_points(cv::Mat image, const std::vector<cv::Point3f>& points, const cv::Point& offset = cv::Point(0, 0), bool display_z = false);
};

#endif // FACE_MODEL_FAMILY_H