#include "face_model_family.h"
#include <filesystem>

namespace gusto_mp_face{
FaceDetector::FaceDetector(const std::string& model_path, const std::string& config_path)
    : BaseONNX(model_path, config_path) {

    std::filesystem::path model_dir = std::filesystem::path(model_path).parent_path();
    std::string anchor_path = (model_dir / "anchor.bin").string();
    anchor_rows = 896;
    anchor_cols = 4;
    anchors = LoadBinaryFile2D(anchor_path, anchor_rows, anchor_cols);
    INPUT_SIZE = 128;
    class_mapper[0] = "Face";
}

FaceDetector::FaceDetector(std::unique_ptr<basic_model_config> config)
    : BaseONNX(std::move(config)) {

    std::filesystem::path model_dir = std::filesystem::path(this->_config->model_path).parent_path();
    std::string anchor_path = (model_dir / "anchor.bin").string();
    anchor_rows = 896;
    anchor_cols = 4;
    anchors = LoadBinaryFile2D(anchor_path, anchor_rows, anchor_cols);
    INPUT_SIZE = 128;
    class_mapper[0] = "Face";
}

std::unique_ptr<PostProcessResult> FaceDetector::postprocess(const std::vector<Ort::Value>& net_out, const cv::Mat& frame) {
    const float* raw_boxes = net_out[0].GetTensorData<float>();
    std::vector<gusto_nms::Rect> boxes = decode_boxes(raw_boxes, anchors);
    std::vector<std::vector<float>> scores;
    std::vector<float> _scores(net_out[1].GetTensorData<float>(), net_out[1].GetTensorData<float>() + net_out[1].GetTensorTypeAndShapeInfo().GetElementCount());
    // auto _scores_with_sigmoid = sigmoid(_scores);

    for (size_t i = 0; i < boxes.size(); i++) {
        std::vector<float> box_score = {sigmoid(_scores[i])};
        scores.push_back(box_score);
    }

    std::pair<std::vector<int>, std::vector<int>> res = multiclass_nms_class_unaware_cpu(boxes, scores, 0.8, 0.5);
    std::vector<int> indices = res.first;
    std::vector<int> indices_cls = res.second;

    std::unique_ptr<MediaPipeDetectorResult> ret = std::make_unique<MediaPipeDetectorResult>();
    for(size_t idx = 0; idx < indices.size(); idx++) {
        std::vector<int> box_to_crop = {
            static_cast<int>(boxes[indices[idx]].y1 * frame.size[0]),
            static_cast<int>(boxes[indices[idx]].x1 * frame.size[1]),
            static_cast<int>(boxes[indices[idx]].y2 * frame.size[0]),
            static_cast<int>(boxes[indices[idx]].x2 * frame.size[1]), 
        }; 
        ret->boxes.push_back(boxes[indices[idx]]);
        ret->scores.push_back(scores[indices[idx]]);
        
    }
    // ret->boxes = boxes;
    // ret->scores = scores;
    return std::move(ret);
}

std::unique_ptr<PostProcessResult> FaceDetector::forward(const cv::Mat& raw) {

    std::vector<float> input_tensor_values = preprocess(raw);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), inputTensorSize, input_shape[0].data(), input_shape[0].size());

    std::vector<Ort::Value> output_tensors = ort_session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());

    auto output = postprocess(output_tensors, raw);

    return std::move(output);
}




cv::Mat FaceDetector::draw_boxes(cv::Mat raw, const std::vector<gusto_nms::Rect>& boxes, const std::vector<std::vector<float>>& scores, const std::vector<int>& indices, const std::vector<int>& indices_cls) {
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Point p1(boxes[idx].x1 * raw.size[1], boxes[idx].y1 * raw.size[0]);
        cv::Point p2(boxes[idx].x2 * raw.size[1], boxes[idx].y2 * raw.size[0]);
        cv::rectangle(raw, cv::Rect(p1, p2), cv::Scalar(0, 255, 0), 2);
        cv::putText(raw, class_mapper[indices_cls[i]], p1, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
        cv::putText(raw, std::to_string(scores[idx][0]), p2, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
    }
    return raw;
}


std::vector<gusto_nms::Rect> FaceDetector::decode_boxes(const float* raw_boxes, const std::vector<std::vector<float>>& anchors) {
    std::vector<gusto_nms::Rect> boxes;
    for (int i = 0; i < anchor_rows; ++i) {
        float x_center = raw_boxes[i * 16 + 0] / 128.0 * anchors[i][2] + anchors[i][0];
        float y_center = raw_boxes[i * 16 + 1] / 128.0 * anchors[i][3] + anchors[i][1];
        float w = raw_boxes[i * 16 + 2] / 128.0 * anchors[i][2];
        float h = raw_boxes[i * 16 + 3] / 128.0 * anchors[i][3];

        boxes.push_back(gusto_nms::Rect(x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2));
    }

    return boxes;
}

std::vector<std::vector<float>> FaceDetector::LoadBinaryFile2D(const std::string& filePath, int rows, int cols) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filePath);
    }

    std::vector<char> byteArray((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    int floatSize = sizeof(float);
    if (byteArray.size() != rows * cols * floatSize) {
        throw std::runtime_error("File size does not match the expected dimensions.");
    }

    std::vector<float> floatArray(rows * cols);
    std::memcpy(floatArray.data(), byteArray.data(), byteArray.size());

    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        std::copy(floatArray.begin() + i * cols, floatArray.begin() + (i + 1) * cols, result[i].begin());
    }

    return result;
}


FaceLandmarker::FaceLandmarker(const std::string& model_path, const std::string& config_path)
    : BaseONNX(model_path, config_path) {}

FaceLandmarker::FaceLandmarker(std::unique_ptr<basic_model_config> config)
    : BaseONNX(std::move(config)) {}

std::tuple<cv::Mat, std::vector<int>>  FaceLandmarker::crop_face(const cv::Mat& image, const std::vector<int>& box) {
    int w = box[3] - box[1];
    int h = box[2] - box[0];
    float margin = 0.25;
    int x1 = std::max(0, box[0] - static_cast<int>(margin * h));
    int x2 = std::min(box[2] + static_cast<int>(margin * h), image.rows);
    int y1 = std::max(0, box[1] - static_cast<int>(margin * w));
    int y2 = std::min(box[3] + static_cast<int>(margin * w), image.cols);
    cv::Rect roi(y1, x1, y2 - y1, x2 - x1);
    std::vector<int> box_with_margin = {x1, y1, x2, y2};
    return {image(roi), box_with_margin};
}

std::unique_ptr<PostProcessResult> FaceLandmarker::forward(const cv::Mat& raw){

    std::vector<float> input_tensor_values = preprocess(raw);
    float h_ratio = static_cast<float>(raw.rows) / static_cast<float>(_config->input_size.first);
    float w_ratio = static_cast<float>(raw.cols) / static_cast<float>(_config->input_size.second);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), inputTensorSize, input_shape[0].data(), input_shape[0].size());

    auto output_tensors = ort_session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());

    const float* points_data = output_tensors[0].GetTensorData<float>();
    const float* tongueOut_data = output_tensors[1].GetTensorData<float>();
    const float* score_data = output_tensors[2].GetTensorData<float>();

    std::vector<cv::Point3f> points;
    for (size_t i = 0; i < 478; ++i) {
        points.emplace_back(points_data[i * 3] * w_ratio, points_data[i * 3 + 1] * h_ratio, points_data[i * 3 + 2]);
    }

    std::unique_ptr<MediapipeFaceLandmarkResult> ret = std::make_unique<MediapipeFaceLandmarkResult>();
    ret->points = points;
    ret->tongueOut = tongueOut_data[0];
    ret->score = sigmoid(score_data[0]);
    return std::move(ret);


}

cv::Mat FaceLandmarker::draw_points(cv::Mat image, const std::vector<cv::Point3f>& points, const cv::Point& offset, bool display_z) {
    for (const auto& point : points) {
        cv::Point2f pt(point.x + offset.x, point.y + offset.y);
        if (display_z) {
            cv::putText(image, std::to_string(point.z), pt, cv::FONT_HERSHEY_SIMPLEX, 0.2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        } else {
            cv::circle(image, pt, 1, cv::Scalar(0, 255, 0), 1);
        }
    }
    return image;
}

} // namespace gusto_mp_face