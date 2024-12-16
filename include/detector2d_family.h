#ifndef DETECTOR_2D_FAMILY_H
#define DETECTOR_2D_FAMILY_H
#include "utils.h"
#include "BaseONNX.h"
#include "tools/nms/nms.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace gusto_detector2d{



class DetectionResult : public PostProcessResult {
public:
    std::vector<gusto_nms::Rect> boxes;
};

// std::unique_ptr<basic_model_config> fetch_model_config(const std::string _model_name, const std::string _model_path);
// std::unique_ptr<basic_model_config> fetch_model_config(const std::string _model_name, const std::string _model_path, const std::pair<int, int> _input_size);

class Detector : public BaseONNX {
    public:
        // Detector(std::unique_ptr<basic_model_config>& _config);
        Detector(const std::string& model_path, const std::string& config_path);
        Detector(std::unique_ptr<basic_model_config> _config);
        // std::vector<float> preprocess_img(const cv::Mat& image, bool bgr2rgb = false);
        std::unique_ptr<PostProcessResult> forward(const cv::Mat& raw);
        // std::vector<Ort::Value> forward(const cv::Mat& raw);
        std::vector<gusto_nms::Rect> postprocess(const std::vector<Ort::Value>& net_out, float score_thr = 0.5, float nms_thr = 0.5) ;

};


} // //namespace gusto_humanseg
#endif // DETECTOR_2D_FAMILY_H