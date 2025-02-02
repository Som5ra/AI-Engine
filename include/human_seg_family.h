#ifndef HUMAN_SEG_FAMILY_H
#define HUMAN_SEG_FAMILY_H
#include "utils.h"
#include "BaseONNX.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace gusto_humanseg{

enum class model_lib {
    selfie_multiclass_256x256,
    selfie_segmenter,
    selfie_segmenter_landscape,
    deeplab_v3
};

extern std::map<std::string, model_lib> MODEL_NAME_LIB_MAPPER;
extern std::map<int, cv::Vec3b> CLASS_COLOR_MAPPER;
/*
std::map<std::string, model_lib> MODEL_NAME_LIB_MAPPER = {
    {"selfie_multiclass_256x256", model_lib::selfie_multiclass_256x256},
    {"selfie_segmenter", model_lib::selfie_segmenter},
    {"selfie_segmenter_landscape", model_lib::selfie_segmenter_landscape},
    {"deeplab_v3", model_lib::deeplab_v3}
};

    // Define a color map for each class
std::map<int, cv::Vec3b> CLASS_COLOR_MAPPER = {
    {0, cv::Vec3b(0, 0, 0)},       // background
    {1, cv::Vec3b(0, 128, 0)},     // hair
    {2, cv::Vec3b(128, 0, 0)},     // body-skin
    {3, cv::Vec3b(0, 0, 128)},     // face-skin
    {4, cv::Vec3b(128, 128, 0)},   // clothes
    {5, cv::Vec3b(0, 128, 128)}    // others (accessories)
};
*/

class SegmentationResult : public PostProcessResult {
public:
    cv::Mat mask;
    // std::vector<cv::Point3f> points;
    // float tongueOut;
    // float score;
};

std::unique_ptr<basic_model_config> fetch_model_config(std::unique_ptr<basic_model_config>& base_config);


class Segmenter : public BaseONNX {
    public:
        // Segmenter(std::unique_ptr<seg_config>& _config);
        Segmenter(const std::string& model_path, const std::string& config_path);
        Segmenter(std::unique_ptr<basic_model_config> _config);
        // cv::Mat preprocess_img(const cv::Mat& image);
        // std::vector<Ort::Value> forward(const cv::Mat& raw);
        std::unique_ptr<PostProcessResult> forward(const cv::Mat& raw) override;
        cv::Mat postprocess(const std::vector<Ort::Value>& mask_out, std::pair<int, int> target_size);
    private:
        std::unique_ptr<basic_model_config> _seg_config;

};


} // //namespace gusto_humanseg
#endif // HUMAN_SEG_FAMILY_H