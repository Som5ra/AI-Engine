#include "detector2d_family.h"

#include <stdexcept>

namespace custom_detector2d{



Detector::Detector(const std::string& model_path, const std::string& config_path)
    : BaseONNX(model_path, config_path) {
}

Detector::Detector(std::unique_ptr<basic_model_config> _config)
    : BaseONNX(std::move(_config)) {
}


std::unique_ptr<PostProcessResult> Detector::forward(const cv::Mat& raw) {
    
    std::vector<float> input_tensor_values = preprocess(raw);
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), inputTensorSize, input_shape[0].data(), input_shape[0].size());
    std::vector<Ort::Value> output_tensors = ort_session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());
    std::unique_ptr<DetectionResult> result = std::make_unique<DetectionResult>();
    result->boxes = postprocess(output_tensors, 0.5, 0.5);
    float w_ratio = raw.cols / (float)_config->input_size.second;
    float h_ratio = raw.rows / (float)_config->input_size.first;
    for(size_t i = 0; i < result->boxes.size(); i++){
        // rescalled the boxes
        result->boxes[i].x1 *= w_ratio;
        result->boxes[i].y1 *= h_ratio;
        result->boxes[i].x2 *= w_ratio;
        result->boxes[i].y2 *= h_ratio;
    }
    return std::move(result);
}


// model with nms
std::vector<CustomRect> Detector::postprocess(
    const std::vector<Ort::Value>& net_out,
    float score_thr,
    float nms_thr) {
    (void)nms_thr;
    if (net_out.size() < 2) {
        throw std::runtime_error(
            "Detector output must contain detections and labels");
    }

    const auto detection_info = net_out[0].GetTensorTypeAndShapeInfo();
    const auto label_info = net_out[1].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> detection_shape = detection_info.GetShape();
    const std::size_t detection_count = detection_info.GetElementCount();
    const std::size_t label_count = label_info.GetElementCount();

    if (detection_shape.size() < 3 || detection_shape.back() < 5) {
        throw std::runtime_error(
            "Detector output must have shape [batch, boxes, values>=5]");
    }
    const std::size_t detection_stride =
        static_cast<std::size_t>(detection_shape.back());
    if (label_count * detection_stride != detection_count) {
        throw std::runtime_error(
            "Detector detection and label output sizes do not match");
    }

    const float* detections = net_out[0].GetTensorData<float>();
    const float* labels = net_out[1].GetTensorData<float>();
    std::vector<CustomRect> filtered_boxes;
    filtered_boxes.reserve(label_count);

    for (std::size_t index = 0; index < label_count; ++index) {
        const std::size_t offset = index * detection_stride;
        const float score = detections[offset + 4];
        if (score > score_thr) {
            filtered_boxes.emplace_back(
                detections[offset],
                detections[offset + 1],
                detections[offset + 2],
                detections[offset + 3],
                score,
                static_cast<int>(labels[index]));
        }
    }

    return filtered_boxes;
}


// model without nms
// std::vector<CustomRect> Detector::postprocess(const std::vector<Ort::Value>& net_out, float score_thr, float nms_thr) {
//     const float* _dets = net_out[0].GetTensorData<float>();
//     const float* _scores = net_out[1].GetTensorData<float>();

//     std::vector<CustomRect> dets;
//     std::vector<std::vector<float>> scores;
//     std::cout << "net_out[0].GetTensorTypeAndShapeInfo().GetElementCount(): " << net_out[0].GetTensorTypeAndShapeInfo().GetElementCount() << std::endl;
//     std::cout << "net_out[1].GetTensorTypeAndShapeInfo().GetElementCount(): " << net_out[1].GetTensorTypeAndShapeInfo().GetElementCount() << std::endl;

//     size_t proposal_num = net_out[0].GetTensorTypeAndShapeInfo().GetElementCount() / 4;


//     float max_score = 0;
//     for(size_t i = 0; i < net_out[0].GetTensorTypeAndShapeInfo().GetElementCount() / 4; i++){
//         CustomRect rect(_dets[i * 4], _dets[i * 4 + 1], _dets[i * 4 + 2], _dets[i * 4 + 3]);
//         dets.push_back(rect);
//         std::vector<float> score(this->_config->class_mapper.size());
//         #pragma omp parallel for
//         for(size_t cls_idx = 0; cls_idx < this->_config->class_mapper.size(); cls_idx++){
//             score[cls_idx] = _scores[i + cls_idx * proposal_num];
//         }
//         // std::vector<float> score = {_scores[i], _scores[i + 2100], _scores[i + 4200]};
//         scores.push_back(score);
//     }
//     auto ret = custom_nms::multiclass_nms_class_unaware_cpu(dets, scores, score_thr, nms_thr);
//     std::vector<int> indices = ret.first;
//     std::vector<int> indices_cls = ret.second;

//     std::vector<CustomRect> filtered_boxes;
//     for (size_t i = 0; i < indices.size(); i++) {
//         filtered_boxes.push_back(dets[indices[i]]);
//     }

//     return filtered_boxes;
// }


// std::vector<CustomRect> Detector::postprocess_mediapipe(const std::vector<Ort::Value>& net_out, float score_thr, float nms_thr) {
//     const float* _dets = net_out[1].GetTensorData<float>();
//     const float* _scores = net_out[0].GetTensorData<float>();
//     std::vector<CustomRect> dets;
//     std::vector<std::vector<float>> scores;
//     for(size_t i = 0; i < net_out[0].GetTensorTypeAndShapeInfo().GetElementCount() / 16; i++){
//         CustomRect rect(_dets[i * 16], _dets[i * 16 + 1], _dets[i * 16 + 2], _dets[i * 16 + 3]);
//         dets.push_back(rect);
//         std::vector<float> score = {_scores[i]};
//         scores.push_back(score);
//     }
//     // std::vector<float> dets(_dets, _dets + net_out[0].GetTensorTypeAndShapeInfo().GetElementCount());
//     // std::vector<float> scores(_scores, _scores + net_out[1].GetTensorTypeAndShapeInfo().GetElementCount());
//     auto ret = custom_nms::multiclass_nms_class_unaware_cpu(dets, scores, score_thr, nms_thr);
//     std::vector<int> indices = ret.first;
//     std::vector<int> indices_cls = ret.second;

//     std::vector<CustomRect> filtered_boxes;
//     for (size_t i = 0; i < indices.size(); i++) {
//         filtered_boxes.push_back(dets[indices[i]]);
//     }

//     return filtered_boxes;
// }

} //namespace custom_humanseg