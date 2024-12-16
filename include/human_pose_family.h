
#ifndef HUMAN_POSE_FAMILY_H
#define HUMAN_POSE_FAMILY_H

#include "utils.h"
#include "BaseONNX.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace gusto_humanpose{

enum class model_lib {
    RTMO_S, // single stage
};

extern std::map<std::string, model_lib> MODEL_NAME_LIB_MAPPER;

std::unique_ptr<basic_model_config> fetch_model_config(const std::unique_ptr<basic_model_config>& _config);

class KeyPointResult : public PostProcessResult {
    public:
        std::vector<std::vector<std::tuple<int, int, int>>> keypoints;
        // std::vector<std::pair<int, int>> skeleton;
};

class PoseDetector : public BaseONNX {
    public:
        PoseDetector(const std::string& model_path, const std::string& config_path);
        PoseDetector(std::unique_ptr<basic_model_config> _config);
        // PoseDetector(std::unique_ptr<basic_model_config>& _config);
        // std::vector<float> preprocess_img(const cv::Mat& image);
        std::unique_ptr<PostProcessResult> forward(const cv::Mat& image);
        cv::Mat Debug_Preprocess(const cv::Mat& image);
        std::vector<std::vector<std::tuple<int, int, int>>>  postprocess(const std::vector<Ort::Value>& output_tensors, float threshold = 0.5);
        cv::Mat draw_single_person_keypoints(cv::Mat image, const std::vector<std::tuple<int, int, int>>& keypoints);
    private:
        std::unique_ptr<basic_model_config> _config;
        float preprocess_ratio;

        const std::vector<std::pair<std::string, cv::Vec3i>> coco17_mapper = {
            {"nose", cv::Vec3i(51, 153, 255)},
            {"left_eye", cv::Vec3i(51, 153, 255)},
            {"right_eye", cv::Vec3i(51, 153, 255)},
            {"left_ear", cv::Vec3i(51, 153, 255)},
            {"right_ear", cv::Vec3i(51, 153, 255)},
            {"left_shoulder", cv::Vec3i(0, 255, 0)},
            {"right_shoulder", cv::Vec3i(255, 128, 0)},
            {"left_elbow", cv::Vec3i(0, 255, 0)},
            {"right_elbow", cv::Vec3i(255, 128, 0)},
            {"left_wrist", cv::Vec3i(0, 255, 0)},
            {"right_wrist", cv::Vec3i(255, 128, 0)},
            {"left_hip", cv::Vec3i(0, 255, 0)},
            {"right_hip", cv::Vec3i(255, 128, 0)},
            {"left_knee", cv::Vec3i(0, 255, 0)},
            {"right_knee", cv::Vec3i(255, 128, 0)},
            {"left_ankle", cv::Vec3i(0, 255, 0)},
            {"right_ankle", cv::Vec3i(255, 128, 0)}
        };

        const std::vector<std::pair<int, int>> coco17_skeleton = {
            {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
            {5, 6}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
        };

};



}

#endif // HUMAN_POSE_FAMILY_H