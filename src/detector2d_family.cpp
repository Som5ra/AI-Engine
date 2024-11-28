#include "detector2d_family.h"

namespace gusto_detector2d{


std::unique_ptr<basic_model_config> fetch_model_config(const std::string _model_name, const std::string _model_path){
    // seg_config* _config = new seg_config();
    // std::unique_ptr<seg_config> _config = std::make_unique<seg_config>();
    std::unique_ptr<basic_model_config> _config(new basic_model_config());
    _config->model_name = _model_name;
    _config->model_path = _model_path;
    _config->class_mapper = {
        {0, "dets"},
        {1, "scores"},
    };
    std::cout <<  "input model name: " << _model_name << std::endl;
    std::cout << "model path: " << _config->model_path << std::endl;
    return _config;
}


Detector::Detector(std::unique_ptr<basic_model_config>& _config)
    : BaseONNX(_config->model_path, _config->model_name) {
    this->_config = std::move(_config);
    std::cout << "input size: " << input_shape[0][2] << " " << input_shape[0][3] << std::endl;
    this->_config->input_size = std::make_pair(input_shape[0][2], input_shape[0][3]);
}

std::vector<float> Detector::preprocess_img(const cv::Mat& image) {
    cv::Mat frame;
    cv::cvtColor(image, frame, cv::COLOR_BGR2RGB);
    cv::resize(frame, frame, cv::Size(_config->input_size.second, _config->input_size.first), 0, 0, cv::INTER_LINEAR);
    frame.convertTo(frame, CV_32FC3, 1.0 / 127.5, -1); // NHWC
    
    std::vector<cv::Mat> rgbsplit;
    cv::split(frame, rgbsplit);
    std::vector<float> input_tensor_values(inputTensorSize);


    int h = rgbsplit[0].size[0];
    int w = rgbsplit[0].size[1];
    #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
    omp_set_num_threads(std::max(1, omp_get_max_threads() / 2));
    #pragma omp parallel for
    #endif
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < rgbsplit.size(); k++) {
                input_tensor_values[k * h * w + i * w + j] = static_cast<float>(rgbsplit[k].at<float>(i, j)); // NCHW
            }
        }
    }


    return input_tensor_values;
}

std::vector<Ort::Value> Detector::forward(const cv::Mat& raw) {
    std::vector<float> input_tensor_values = preprocess_img(raw);


    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), inputTensorSize, input_shape[0].data(), input_shape[0].size());

    std::vector<Ort::Value> output_tensors = ort_session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());
    return output_tensors;

}

std::vector<gusto_nms::Rect> Detector::postprocess(const std::vector<Ort::Value>& net_out, float score_thr, float nms_thr) {
    const float* _dets = net_out[1].GetTensorData<float>();
    const float* _scores = net_out[0].GetTensorData<float>();
    std::vector<gusto_nms::Rect> dets;
    std::vector<std::vector<float>> scores;
    for(size_t i = 0; i < net_out[0].GetTensorTypeAndShapeInfo().GetElementCount() / 16; i++){
        gusto_nms::Rect rect(_dets[i * 16], _dets[i * 16 + 1], _dets[i * 16 + 2], _dets[i * 16 + 3]);
        dets.push_back(rect);
        std::vector<float> score = {_scores[i]};
        scores.push_back(score);
        std::cout << "x1: " << rect.x1 << " y1: " << rect.y1 << " x2: " << rect.x2 << " y2: " << rect.y2 << " score: " << _scores[i] << std::endl;
    }
    // std::vector<float> dets(_dets, _dets + net_out[0].GetTensorTypeAndShapeInfo().GetElementCount());
    // std::vector<float> scores(_scores, _scores + net_out[1].GetTensorTypeAndShapeInfo().GetElementCount());
    auto ret = gusto_nms::multiclass_nms_class_unaware_cpu(dets, scores, score_thr, nms_thr);
    std::vector<int> indices = ret.first;
    std::vector<int> indices_cls = ret.second;

    std::vector<gusto_nms::Rect> filtered_boxes;
    for (size_t i = 0; i < indices.size(); i++) {
        filtered_boxes.push_back(dets[indices[i]]);
    }

    return filtered_boxes;
}

} //namespace gusto_humanseg