#include "tools/face_geometry/geometry_pipeline.h"
#include "tools/face_geometry/face_geometry.h"
#include "tools/face_geometry/procrustes_solver.h"
#include "tools/face_geometry/calculator.h"
#include "tools/nms/nms.h"
#include "utils.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <map>
#include <cmath>


class FaceDetector {
public:
    FaceDetector(const std::string& model_path) 
        : ort_env(ORT_LOGGING_LEVEL_WARNING, "FaceDetector"),
          ort_session(ort_env, model_path.c_str(), Ort::SessionOptions{nullptr}) {

        anchor_rows = 896;
        anchor_cols = 4;
        anchors = LoadBinaryFile2D("/media/sombrali/HDD1/mmlib/anchor.bin", anchor_rows, anchor_cols);
        INPUT_SIZE = 128;
        class_mapper[0] = "Face"; 
        size_t input_count = ort_session.GetInputCount();
        size_t output_count = ort_session.GetOutputCount();

        // [Sombra] -> GetInputNameAllocated(i, ort_allocator).get() is not working, GetInputNameAllocated(i, ort_allocator) first then get() is working
        for (size_t i = 0; i < input_count; ++i) {
            Ort::AllocatedStringPtr intput_name_ptr = ort_session.GetInputNameAllocated(i, ort_allocator);
            const char* input_name = strdup(intput_name_ptr.get());
            input_names.push_back(input_name);
            Ort::TypeInfo info = ort_session.GetInputTypeInfo(i);
            auto tensor_info = info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> dims = tensor_info.GetShape();
            input_shape.push_back(dims);
        }

        for (size_t i = 0; i < output_count; ++i) {
            Ort::AllocatedStringPtr output_name_ptr = ort_session.GetOutputNameAllocated(i, ort_allocator);
            const char* output_name = strdup(output_name_ptr.get());
            output_names.push_back(output_name);
        }

    }


    cv::Mat preprocess_img(const cv::Mat& image) {
        cv::Mat frame;
        cv::cvtColor(image, frame, cv::COLOR_BGR2RGB);
        cv::resize(frame, frame, cv::Size(INPUT_SIZE, INPUT_SIZE), 0, 0, cv::INTER_LINEAR);
        frame.convertTo(frame, CV_32FC3, 1.0 / 127.5, -1);
        return frame;
    }

    std::tuple<std::vector<gusto_nms::Rect>, std::vector<std::vector<float> >, std::vector<int>, std::vector<int>> forward(const cv::Mat& raw) {
        cv::Mat frame = preprocess_img(raw);

        std::vector<cv::Mat> rgbsplit;
        cv::split(frame, rgbsplit);

        // std::vector<int64_t> input_shape = {1, INPUT_SIZE, INPUT_SIZE, 3};
        // size_t inputTensorSize = 1 * INPUT_SIZE * INPUT_SIZE, 3
        size_t inputTensorSize = 1;
        for (auto dim : input_shape[0]) {
            inputTensorSize *= dim;
        }

        std::vector<float> input_tensor_values(inputTensorSize);
        #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
        omp_set_num_threads(omp_get_max_threads() / 2);
        #pragma omp parallel for
        #endif
        for(int i = 0; i < rgbsplit[0].size[0]; i++) {
            for (int j = 0; j < rgbsplit[0].size[1]; j++) {
                for (int k = 0; k < rgbsplit.size(); k++) {
                    input_tensor_values[i * rgbsplit[0].size[1] * rgbsplit.size() + j * rgbsplit.size() + k] = rgbsplit[k].at<float>(i, j);
                }
            }
        }


        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), inputTensorSize, input_shape[0].data(), input_shape[0].size());


        auto output_tensors = ort_session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 2);

        const float* raw_boxes = output_tensors[0].GetTensorData<float>();
        std::vector<gusto_nms::Rect> boxes = decode_boxes(raw_boxes, anchors);
        std::vector<std::vector<float> > scores;
        std::vector<float> _scores(output_tensors[1].GetTensorMutableData<float>(), output_tensors[1].GetTensorMutableData<float>() + output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount());
        auto _scores_with_sigmoid = sigmoid(_scores);        

        for (size_t i = 0; i < boxes.size(); i++) {
            std::vector<float> box_score = {sigmoid(_scores[i])};
            scores.push_back(box_score);
        }

        std::pair<std::vector<int>, std::vector<int>> res = multiclass_nms_class_unaware_cpu(boxes, scores, 0.8, 0.5);
        std::vector<int> indices = res.first;
        std::vector<int> indices_cls = res.second;

        return {boxes, scores, indices, indices_cls};
    }

    cv::Mat draw_boxes(cv::Mat raw, const std::vector<gusto_nms::Rect>& boxes, const std::vector<std::vector<float> >& scores, const std::vector<int>& indices, const std::vector<int>& indices_cls) {
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

private:
    Ort::Env ort_env;
    Ort::Session ort_session;
    Ort::AllocatorWithDefaultOptions ort_allocator;
    int anchor_rows, anchor_cols;
    std::vector<std::vector<float>> anchors;
    int INPUT_SIZE;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t> > input_shape;
    std::map<int, std::string> class_mapper;
    

    std::vector<gusto_nms::Rect> decode_boxes(const float* raw_boxes, const std::vector<std::vector<float> >& anchors) {
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

    static float sigmoid(float x) {
        return 1 / (1 + exp(-x));
    }

    static std::vector<float> sigmoid(const std::vector<float>& x) {
        std::vector<float> result(x.size());

        #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
        omp_set_num_threads(omp_get_max_threads() / 2);
        #pragma omp parallel for
        #endif
        for (int i = 0; i < x.size(); i++) {
            result[i] = 1 / (1 + exp(-x[i]));
        }
        return result;
    }
    static std::vector<std::vector<float>> LoadBinaryFile2D(const std::string& filePath, int rows, int cols) {
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
};


class FaceLandmarker {
public:
    FaceLandmarker(const std::string& model_path)
        : ort_env(ORT_LOGGING_LEVEL_WARNING, "FaceLandmarker"),
          ort_session(ort_env, model_path.c_str(), Ort::SessionOptions{nullptr}) {

        size_t input_count = ort_session.GetInputCount();
        size_t output_count = ort_session.GetOutputCount();

        for (size_t i = 0; i < input_count; ++i) {
            Ort::AllocatedStringPtr input_name_ptr = ort_session.GetInputNameAllocated(i, ort_allocator);
            const char* input_name = strdup(input_name_ptr.get());
            input_names.push_back(input_name);
            Ort::TypeInfo info = ort_session.GetInputTypeInfo(i);
            auto tensor_info = info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> dims = tensor_info.GetShape();
            if (dims[0] == -1) {
                dims[0] = 1;
            }
            input_shape.push_back(dims);
            std::cout << "Input Shape: " << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3] << std::endl;
        }

        for (size_t i = 0; i < output_count; ++i) {
            Ort::AllocatedStringPtr output_name_ptr = ort_session.GetOutputNameAllocated(i, ort_allocator);
            const char* output_name = strdup(output_name_ptr.get());
            output_names.push_back(output_name);
        }

        std::cout << "Input Node: " << input_names[0] << std::endl;
        std::cout << "Output Nodes: ";
        for (const auto& name : output_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl;
    }

    ~FaceLandmarker() {
        for (const char* name : input_names) {
            free(const_cast<char*>(name));
        }
        for (const char* name : output_names) {
            free(const_cast<char*>(name));
        }
    }

    cv::Mat crop_face(const cv::Mat& image, const std::vector<int>& box) {
        int w = box[3] - box[1];
        int h = box[2] - box[0];
        float margin = 0.25 / 2;
        int x1 = std::max(0, box[0] - static_cast<int>(margin * h));
        int x2 = std::min(box[2] + static_cast<int>(margin * h), image.rows);
        int y1 = std::max(0, box[1] - static_cast<int>(margin * w));
        int y2 = std::min(box[3] + static_cast<int>(margin * w), image.cols);
        cv::Rect roi(y1, x1, y2 - y1, x2 - x1);
        return image(roi);
    }

    std::tuple<cv::Mat, float, float> preprocess(const cv::Mat& image) {
        cv::Mat frame;
        cv::cvtColor(image, frame, cv::COLOR_BGR2RGB);
        float h_ratio = static_cast<float>(image.rows) / 256.0f;
        float w_ratio = static_cast<float>(image.cols) / 256.0f;
        cv::resize(frame, frame, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
        frame.convertTo(frame, CV_32F, 1.0 / 127.5, -1);
        // frame = frame.reshape(1, {1, 256, 256, 3});
        return {frame, h_ratio, w_ratio};
    }

    std::tuple<std::vector<cv::Point3f>, float, float> forward(const cv::Mat& image) {
        auto [frame, h_ratio, w_ratio] = preprocess(image);

        size_t inputTensorSize = 1 * 256 * 256 * 3;
        std::vector<float> input_tensor_values(inputTensorSize);
        // std::copy(frame.begin<float>(), frame.end<float>(), input_tensor_values.begin());

        std::vector<cv::Mat> rgbsplit;
        cv::split(frame, rgbsplit);
        #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
        omp_set_num_threads(omp_get_max_threads() / 2);
        #pragma omp parallel for
        #endif
        for(int i = 0; i < rgbsplit[0].size[0]; i++) {
            for (int j = 0; j < rgbsplit[0].size[1]; j++) {
                for (int k = 0; k < rgbsplit.size(); k++) {
                    input_tensor_values[i * rgbsplit[0].size[1] * rgbsplit.size() + j * rgbsplit.size() + k] = rgbsplit[k].at<float>(i, j);
                }
            }
        }
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), inputTensorSize, input_shape[0].data(), input_shape[0].size());

        auto output_tensors = ort_session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());

        const float* points_data = output_tensors[0].GetTensorData<float>();
        const float* tongueOut_data = output_tensors[1].GetTensorData<float>();
        const float* score_data = output_tensors[2].GetTensorData<float>();

        std::vector<cv::Point3f> points;
        for (size_t i = 0; i < 478; ++i) {
            points.emplace_back(points_data[i * 3] * w_ratio, points_data[i * 3 + 1] * h_ratio, points_data[i * 3 + 2]);
        }

        return {points, tongueOut_data[0], sigmoid(score_data[0])};
    }

    cv::Mat draw_points(cv::Mat image, const std::vector<cv::Point3f>& points, const cv::Point& offset = cv::Point(0, 0), bool display_z = false) {
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

private:
    Ort::Env ort_env;
    Ort::Session ort_session;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_shape;
    Ort::AllocatorWithDefaultOptions ort_allocator;
    static float sigmoid(float x) {
        return 1 / (1 + exp(-x));
    }

    static std::vector<float> sigmoid(const std::vector<float>& x) {
        std::vector<float> result(x.size());

        #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
        omp_set_num_threads(omp_get_max_threads() / 2);
        #pragma omp parallel for
        #endif
        for (int i = 0; i < x.size(); i++) {
            result[i] = 1 / (1 + exp(-x[i]));
        }
        return result;
    }
};


int main()
{

    std::string face_detector_path = "/media/sombrali/HDD1/facelandmark/weights/mediapipe/face_detector.onnx";
    std::string face_landmarker_path = "/media/sombrali/HDD1/facelandmark/weights/mediapipe/face_landmarks_detector.onnx";

    FaceDetector face_detector(face_detector_path);
    FaceLandmarker face_landmarker(face_landmarker_path);
    // cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    // // cv::Mat test_image = cv::imread("/media/sombrali/HDD1/opencv-unity/gusto_dnn/000000000785.jpg");
    // cv::Mat test_image = cv::imread("/media/sombrali/HDD1/opencv-unity/gusto_dnn/download.png", cv::IMREAD_COLOR);
    // auto [boxes, scores, indices, indices_cls] = face_detector.forward(test_image);
    // auto painted_image = face_detector.draw_boxes(test_image, boxes, scores, indices, indices_cls);
    // cv::imshow("Frame", painted_image);
    // cv::waitKey(0);



    float min_time = 1000000;
    float max_time = 0;
    float total_time = 0;
    int num_frames = 0;


    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.open(0);

    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera!" << std::endl;
        return 1;
    }
    cv::Mat frame;
    cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    cv::namedWindow("cropped_face", cv::WINDOW_NORMAL);
    std::cout << "==============================================================" << std::endl;
    while (true)
    {
        cap >> frame; // so fkng slow
        auto start = std::chrono::high_resolution_clock::now();
        auto [boxes, scores, indices, indices_cls] = face_detector.forward(frame);
        if (indices.size() > 0) {
            std::vector<int> box_to_crop = {
                boxes[indices[0]].y1 * frame.size[0],
                boxes[indices[0]].x1 * frame.size[1],
                boxes[indices[0]].y2 * frame.size[0],
                boxes[indices[0]].x2 * frame.size[1], 
            }; 
            cv::Mat cropped_face = face_landmarker.crop_face(frame, box_to_crop);
            auto [points, tongueOut, score] = face_landmarker.forward(cropped_face);
            if (score < 0.49) {
                continue;
            }
            // frame = face_landmarker.draw_points(frame, points);
            cropped_face = face_landmarker.draw_points(cropped_face, points);
            cv::imshow("cropped_face", cropped_face);
        }

        auto end = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        min_time = std::min(min_time, duration);
        max_time = std::max(max_time, duration);
        total_time += duration;
        num_frames++;
        std::cout << "\r|| min_time: " << min_time << "ms  |  max_time: " << max_time << "ms  |  avg_time: " << std::ceil(total_time / num_frames * 100) / 100 << "ms ||" << std::flush;    

        // auto painted_image = face_detector.draw_boxes(frame, boxes, scores, indices, indices_cls);
        cv::imshow("Frame", frame);
        if (cv::waitKey(25) >= 0)
            break;

    }
    
    // face_detector.check_names();

    const std::string face_GeometryPipelineMetadata = "/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/face_geometry/geometry_pipeline_metadata_including_iris_landmarks.json";
    GUSTO_RET open_status = gusto_face_geometry::Open(face_GeometryPipelineMetadata);
    if (open_status != GustoStatus::ERR_OK) {
        std::cerr << "Failed to open Geometry Pipeline Metadata!" << std::endl;
        return 1;
    }
    return 0;
}