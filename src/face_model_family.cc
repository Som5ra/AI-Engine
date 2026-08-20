#include "face_model_family.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace custom_mp_face{
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegenerateEyeDistancePx = 1e-3f;

// MediaPipe NormalizeRadians: wrap to [-pi, pi]
//   angle - 2*pi*floor((angle+pi)/(2*pi))
float NormalizeRadians(float angle) {
    return angle - 2.0f * kPi * std::floor((angle + kPi) / (2.0f * kPi));
}

}  // namespace

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
    std::vector<CustomRect> boxes = decode_boxes(raw_boxes, anchors);
    std::vector<std::array<cv::Point2f, 6>> keypoints = decode_keypoints(raw_boxes, anchors);
    std::vector<std::vector<float>> scores;
    std::vector<float> _scores(net_out[1].GetTensorData<float>(), net_out[1].GetTensorData<float>() + net_out[1].GetTensorTypeAndShapeInfo().GetElementCount());
    // auto _scores_with_sigmoid = sigmoid(_scores);

    for (size_t i = 0; i < boxes.size(); i++) {
        std::vector<float> box_score = {sigmoid(_scores[i])};
        scores.push_back(box_score);
    }

    std::pair<std::vector<int>, std::vector<int>> res = custom_nms::multiclass_nms_class_unaware_cpu(boxes, scores, score_thr_, nms_thr_);
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
        (void)box_to_crop;
        ret->boxes.push_back(boxes[indices[idx]]);
        ret->scores.push_back(scores[indices[idx]]);
        ret->keypoints.push_back(keypoints[indices[idx]]);
        
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




cv::Mat FaceDetector::draw_boxes(cv::Mat raw, const std::vector<CustomRect>& boxes, const std::vector<std::vector<float>>& scores, const std::vector<int>& indices, const std::vector<int>& indices_cls) {
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


std::vector<CustomRect> FaceDetector::decode_boxes(const float* raw_boxes, const std::vector<std::vector<float>>& anchors) {
    std::vector<CustomRect> boxes;
    for (int i = 0; i < anchor_rows; ++i) {
        float x_center = raw_boxes[i * 16 + 0] / 128.0 * anchors[i][2] + anchors[i][0];
        float y_center = raw_boxes[i * 16 + 1] / 128.0 * anchors[i][3] + anchors[i][1];
        float w = raw_boxes[i * 16 + 2] / 128.0 * anchors[i][2];
        float h = raw_boxes[i * 16 + 3] / 128.0 * anchors[i][3];

        boxes.push_back(CustomRect(x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2));
    }

    return boxes;
}

std::vector<std::array<cv::Point2f, 6>> FaceDetector::decode_keypoints(
    const float* raw_boxes, const std::vector<std::vector<float>>& anchors) {
    std::vector<std::array<cv::Point2f, 6>> keypoints;
    keypoints.resize(static_cast<size_t>(anchor_rows));
    for (int i = 0; i < anchor_rows; ++i) {
        for (int k = 0; k < 6; ++k) {
            // Same decode as the box center: offset / 128 * anchor_size + anchor_center.
            const float kp_x =
                raw_boxes[i * 16 + 4 + 2 * k] / 128.0f * anchors[i][2] +
                anchors[i][0];
            const float kp_y =
                raw_boxes[i * 16 + 5 + 2 * k] / 128.0f * anchors[i][3] +
                anchors[i][1];
            keypoints[static_cast<size_t>(i)][static_cast<size_t>(k)] =
                cv::Point2f(kp_x, kp_y);
        }
    }
    return keypoints;
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

cv::Point2f FaceLandmarker::MapCropToImage(const cv::Mat& dst_to_src, float x, float y) {
    if (dst_to_src.empty() || dst_to_src.rows < 2 || dst_to_src.cols < 3) {
        return cv::Point2f(x, y);
    }
    cv::Mat m32;
    if (dst_to_src.type() == CV_32F) {
        m32 = dst_to_src;
    } else {
        dst_to_src.convertTo(m32, CV_32F);
    }
    const float m00 = m32.at<float>(0, 0);
    const float m01 = m32.at<float>(0, 1);
    const float m02 = m32.at<float>(0, 2);
    const float m10 = m32.at<float>(1, 0);
    const float m11 = m32.at<float>(1, 1);
    const float m12 = m32.at<float>(1, 2);
    return cv::Point2f(m00 * x + m01 * y + m02, m10 * x + m11 * y + m12);
}

std::vector<cv::Point3f> FaceLandmarker::MapLandmarksToImage(
    const std::vector<cv::Point3f>& crop_points, const cv::Mat& dst_to_src) {
    std::vector<cv::Point3f> image_points;
    image_points.reserve(crop_points.size());
    // Pixel scale of x (and y) under dst->src M. AABB fallback is identity
    // scale 1; rotated ROI is roi_side / crop_size.
    float z_scale = 1.0f;
    if (!dst_to_src.empty() && dst_to_src.rows >= 2 && dst_to_src.cols >= 1) {
        cv::Mat m32;
        if (dst_to_src.type() == CV_32F) {
            m32 = dst_to_src;
        } else {
            dst_to_src.convertTo(m32, CV_32F);
        }
        z_scale = std::hypot(m32.at<float>(0, 0), m32.at<float>(1, 0));
        if (!std::isfinite(z_scale) || z_scale <= 0.0f) {
            z_scale = 1.0f;
        }
    }
    for (const auto& point : crop_points) {
        const cv::Point2f mapped = MapCropToImage(dst_to_src, point.x, point.y);
        image_points.emplace_back(mapped.x, mapped.y, point.z * z_scale);
    }
    return image_points;
}

/*
 * MediaPipe-style rotated face ROI (copied, not guessed).
 *
 * DetectionsToRects (face_detection_front_detection_to_roi.pbtxt):
 *   start_keypoint_index = 0, end_keypoint_index = 1, target_angle_degrees = 0
 *   0 = left eye, 1 = right eye
 *   rotation = NormalizeRadians(target_angle - atan2(-(y1-y0), x1-x0))
 *   with x,y in pixel coords (normalized * image_size)
 *   NormalizeRadians: angle - 2*pi*floor((angle+pi)/(2*pi))
 *   Rect center/size from bounding box (USE_BOUNDING_BOX), rotation attached.
 *
 * This repo's landmarker ONNX was used with AABB*1.5 then NON-UNIFORM
 * resize to 256x256 (crop_face 0.25 margin), NOT MediaPipe square_long.
 * Keep two-eye rotation, but sub_rect is the anisotropic 1.5x box:
 *   a = 1.5 * box_w_px, b = 1.5 * box_h_px
 * Warp that rotated rectangle to 256x256 (same stretch as old cv::resize).
 *
 * Tracking path (face_landmark_landmarks_to_roi.pbtxt) passes an explicit
 * FaceRotatedRect already scaled (square_long 1.5 on the tight landmark
 * box, or anisotropic 1.5). Same warp, same WARP_INVERSE_MAP.
 *
 * ImageToTensor GetRotatedSubRectToRectTransformMatrix, flip_horizontally=false:
 *   c=cos(rotation), d=sin(rotation)
 *   a=sub_rect.width, b=sub_rect.height   // pixels
 *   e=center_x, f=center_y                // pixels
 *   x_img = a*c*(x/W-0.5) - b*d*(y/H-0.5) + e
 *   y_img = a*d*(x/W-0.5) + b*c*(y/H-0.5) + f
 * This is pixel dst(crop)->src(image). OpenCV warpAffine *without*
 * WARP_INVERSE_MAP inverts M first; with WARP_INVERSE_MAP it samples
 * dst(x,y)=src(M*[x,y,1]). Landmark remap uses the same M (not inverted).
 * MediaPipe GetRotatedSubRectToRect also maps tensor UV->image UV; we stay
 * in pixels here and do not mix the 1/image_size UV form.
 */
FaceCropResult WarpRotatedFaceRoi(
    const cv::Mat& image, const FaceRotatedRect& roi, int W, int H) {
    FaceCropResult result;
    if (image.empty() || W <= 0 || H <= 0) {
        return result;
    }
    if (!std::isfinite(roi.center_x) || !std::isfinite(roi.center_y) ||
        !std::isfinite(roi.width) || !std::isfinite(roi.height) ||
        !std::isfinite(roi.rotation) || roi.width <= 1e-3f ||
        roi.height <= 1e-3f) {
        return result;
    }

    const float a = roi.width;
    const float b = roi.height;
    const float c = std::cos(roi.rotation);
    const float d = std::sin(roi.rotation);
    const float e = roi.center_x;
    const float f = roi.center_y;

    cv::Mat M(2, 3, CV_32F);
    M.at<float>(0, 0) = a * c / static_cast<float>(W);
    M.at<float>(0, 1) = -b * d / static_cast<float>(H);
    M.at<float>(0, 2) = -0.5f * a * c + 0.5f * b * d + e;
    M.at<float>(1, 0) = a * d / static_cast<float>(W);
    M.at<float>(1, 1) = b * c / static_cast<float>(H);
    M.at<float>(1, 2) = -0.5f * a * d - 0.5f * b * c + f;

    cv::Mat cropped;
    cv::warpAffine(image, cropped, M, cv::Size(W, H),
                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    result.image = cropped;
    result.dst_to_src = M;
    result.rotated = true;
    return result;
}

FaceCropResult FaceLandmarker::crop_face_roi(
    const cv::Mat& image, const FaceRotatedRect& roi) {
    const int W = (_config && _config->input_size.second > 0)
                      ? _config->input_size.second
                      : 256;
    const int H = (_config && _config->input_size.first > 0)
                      ? _config->input_size.first
                      : 256;
    return WarpRotatedFaceRoi(image, roi, W, H);
}

FaceCropResult FaceLandmarker::crop_face_roi(
    const cv::Mat& image,
    const CustomRect& box,
    const std::array<cv::Point2f, 6>* keypoints) {
    FaceCropResult result;

    const int img_w = image.cols;
    const int img_h = image.rows;
    if (img_w <= 0 || img_h <= 0) {
        return result;
    }

    bool use_rotated = false;
    float rotation = 0.0f;
    if (keypoints != nullptr) {
        const float x0 = (*keypoints)[0].x * static_cast<float>(img_w);
        const float y0 = (*keypoints)[0].y * static_cast<float>(img_h);
        const float x1 = (*keypoints)[1].x * static_cast<float>(img_w);
        const float y1 = (*keypoints)[1].y * static_cast<float>(img_h);
        const float eye_dist = std::hypot(x1 - x0, y1 - y0);
        if (std::isfinite(x0) && std::isfinite(y0) && std::isfinite(x1) &&
            std::isfinite(y1) && eye_dist > kDegenerateEyeDistancePx) {
            constexpr float kTargetAngle = 0.0f;  // target_angle_degrees = 0
            rotation = NormalizeRadians(
                kTargetAngle - std::atan2(-(y1 - y0), x1 - x0));
            use_rotated = true;
        }
    }

    if (!use_rotated) {
        const int top = std::clamp(static_cast<int>(box.y1 * img_h), 0, img_h);
        const int left = std::clamp(static_cast<int>(box.x1 * img_w), 0, img_w);
        const int bottom = std::clamp(static_cast<int>(box.y2 * img_h), 0, img_h);
        const int right = std::clamp(static_cast<int>(box.x2 * img_w), 0, img_w);
        if (bottom <= top || right <= left) {
            return result;
        }
        auto [cropped, margin] = crop_face(image, {top, left, bottom, right});
        result.image = cropped;
        result.aabb_margin = margin;
        result.rotated = false;
        // crop (x,y) -> image (x + left, y + top); left=margin[1], top=margin[0]
        if (margin.size() >= 2) {
            result.dst_to_src = (cv::Mat_<float>(2, 3) << 1.0f, 0.0f,
                                 static_cast<float>(margin[1]), 0.0f, 1.0f,
                                 static_cast<float>(margin[0]));
        }
        return result;
    }

    const float xmin = box.x1 * static_cast<float>(img_w);
    const float ymin = box.y1 * static_cast<float>(img_h);
    const float xmax = box.x2 * static_cast<float>(img_w);
    const float ymax = box.y2 * static_cast<float>(img_h);
    const float center_x = 0.5f * (xmin + xmax);
    const float center_y = 0.5f * (ymin + ymax);
    const float box_w_px = xmax - xmin;
    const float box_h_px = ymax - ymin;
    // Anisotropic 1.5x AABB (NOT square_long): matches crop_face + resize.
    FaceRotatedRect roi;
    roi.center_x = center_x;
    roi.center_y = center_y;
    roi.width = 1.5f * box_w_px;
    roi.height = 1.5f * box_h_px;
    roi.rotation = rotation;
    return crop_face_roi(image, roi);
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

} // namespace custom_mp_face
