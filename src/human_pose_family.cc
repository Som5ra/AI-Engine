#include "human_pose_family.h"

namespace gusto_humanpose{

std::map<std::string, model_lib> MODEL_NAME_LIB_MAPPER = {
    {"rtmo-s", model_lib::RTMO_S}
};

std::unique_ptr<basic_model_config> fetch_model_config(std::unique_ptr<basic_model_config>& base_config){
    // std::unique_ptr<base_model_config> _config(new base_model_config());
    std::cout <<  "input model name: " << base_config->model_name << std::endl;
    model_lib model_type;
    try{
        model_type = MODEL_NAME_LIB_MAPPER.at(base_config->model_name);
        // _config->model_name = _model_name;
        // _config->model_type = MODEL_NAME_LIB_MAPPER[_model_name];
    }catch(const std::exception& e){
        std::cerr << "input model name is not valid! " << std::endl;
        return nullptr;
        // return _config;
    }

    if(model_type == model_lib::RTMO_S){
        base_config->model_path = "rtmo-s.onnx";
        base_config->input_size = std::make_pair(640, 640);
        base_config->class_mapper = {
            // to be written
        };
    }

    std::cout << "model path: " << base_config->model_path << std::endl;
    return std::move(base_config);
}

PoseDetector::PoseDetector(const std::string& model_path, const std::string& config_path)
    : BaseONNX(model_path, config_path) {
    this->_config = std::move(_config);
}

PoseDetector::PoseDetector(std::unique_ptr<basic_model_config> _config)
    : BaseONNX(std::move(_config)) {
    this->_config = std::move(_config);
}


cv::Mat PoseDetector::Debug_Preprocess(const cv::Mat& image){
    cv::Mat frame;
    frame = image.clone();
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    float ratio = std::min(static_cast<float>(_config->input_size.first) / image.rows, static_cast<float>(_config->input_size.second) / image.cols);

    cv::Mat resized_img;
    cv::resize(frame, resized_img, cv::Size(static_cast<int>(image.cols * ratio), static_cast<int>(image.rows * ratio)), 0, 0, cv::INTER_LINEAR);

    cv::Rect roi(cv::Point(0, 0), resized_img.size());
    cv::Mat padded_img(_config->input_size.first, _config->input_size.second, CV_8UC3, cv::Scalar(114, 114, 114));
    resized_img.copyTo(padded_img(roi));
    return padded_img;
}

// std::vector<float> PoseDetector::preprocess_img(const cv::Mat& image) {
//     cv::Mat frame;
//     frame = image.clone();
//     cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
//     float ratio = std::min(static_cast<float>(_config->input_size.first) / image.rows, static_cast<float>(_config->input_size.second) / image.cols);
//     this->preprocess_ratio = ratio;
//     cv::Mat resized_img;
//     cv::resize(frame, resized_img, cv::Size(static_cast<int>(image.cols * ratio), static_cast<int>(image.rows * ratio)), 0, 0, cv::INTER_LINEAR);

//     cv::Rect roi(cv::Point(0, 0), resized_img.size());
//     cv::Mat padded_img(_config->input_size.first, _config->input_size.second, CV_8UC3, cv::Scalar(114, 114, 114));
//     resized_img.copyTo(padded_img(roi));
//     // cv::resize(frame, frame, cv::Size(_config->input_size.second, _config->input_size.first), 0, 0, cv::INTER_LINEAR);
//     // frame.convertTo(frame, CV_32FC3, 1.0 / 127.5, -1); 
//     // padded_img.convertTo(frame, CV_32FC3, 1.0, 0.0); 
//     std::vector<cv::Mat> rgbsplit;
//     cv::split(padded_img, rgbsplit);

//     std::vector<float> input_tensor_values(inputTensorSize);


//     int h = rgbsplit[0].size[0];
//     int w = rgbsplit[0].size[1];
//     #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
//     omp_set_num_threads(std::max(1, omp_get_max_threads() / 2));
//     #pragma omp parallel for
//     #endif
//     for (int i = 0; i < h; i++) {
//         for (int j = 0; j < w; j++) {
//             for (int k = 0; k < rgbsplit.size(); k++) {
//                 // std::cout << "i: " <<  i << " j: " << j << " k: " << k << " value: " << rgbsplit[k].at<float>(i, j)<< std::endl;
//                 // std::cout << "i: " <<  i << " j: " << j << " k: " << k << " value: " << static_cast<float>(rgbsplit[k].at<uint8_t>(i, j)) << std::endl;
//                 input_tensor_values[k * h * w + i * w + j] = static_cast<float>(rgbsplit[k].at<uint8_t>(i, j)); // CHW
//             }
//         }
//     }

//     return input_tensor_values;
// }

std::unique_ptr<PostProcessResult> PoseDetector::forward(const cv::Mat& image){
    // std::vector<float> input_tensor_values = preprocess_img(image);
    auto input_tensor_values = preprocess(image);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), inputTensorSize, input_shape[0].data(), input_shape[0].size());

    Ort::RunOptions run_options{};
    std::vector<Ort::Value> output_tensors = ort_session.Run(run_options, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());
    // std::vector<Ort::Value> output_tensors;
    std::unique_ptr<KeyPointResult> result = std::make_unique<KeyPointResult>();
    result->keypoints = postprocess(output_tensors, 0.5);
    // return output_tensors;
    return result;
}

std::vector<std::vector<std::tuple<int, int, int>>> PoseDetector::postprocess(const std::vector<Ort::Value>& output_tensors, float threshold) {
    // detector is kinda useless with rtmo-series models

    const float* prob = output_tensors[1].GetTensorData<float>();
    auto outputInfo = output_tensors[1].GetTensorTypeAndShapeInfo();
    int batch_size = outputInfo.GetShape()[0];
    int activated_person = outputInfo.GetShape()[1];
    int kpts_num = outputInfo.GetShape()[2];
    int kpts = outputInfo.GetShape()[3];
    // std::cout << "batch_size: " << batch_size << std::endl;
    // std::cout << "activated_person: " << activated_person << std::endl;
    // std::cout << "width: " << kpts_num << std::endl;
    // std::cout << "channels: " << kpts << std::endl;

    // [Sombra] TODO: Support batch size > 1
    assert(batch_size == 1);

    std::vector<std::vector<std::tuple<int, int, int>>> keypoints;
    // std::vector<float> single_person_keypoints_conf;
    for(int i = 0; i < activated_person; i++){
        // std::cout << "person id: " << i << std::endl;
        std::vector<std::tuple<int, int, int>> single_person_keypoints;

        for(int j = 0; j < kpts_num; j++){

            int x = static_cast<int>(prob[0 + kpts * (j + kpts_num * i)] / this->preprocess_ratio);
            int y = static_cast<int>(prob[1 + kpts * (j + kpts_num * i)] / this->preprocess_ratio);
            float conf = prob[2 + kpts * (j + kpts_num * i)];
            // if (conf < threshold){
            //     continue;
            // }
            // std::cout << "x: " << x << " y: " << y << " conf: " << conf << std::endl;
            int exist = conf > threshold ? 1 : 0;
            single_person_keypoints.push_back(std::make_tuple(x, y, exist));
        }
        keypoints.push_back(single_person_keypoints);
    }

    
    return keypoints;
}


cv::Mat PoseDetector::draw_single_person_keypoints(cv::Mat image, const std::vector<std::tuple<int, int, int>>& keypoints){
    for(size_t i = 0; i < keypoints.size(); i++){
        auto [x, y, exist] = keypoints[i];
        if (exist == 0){
            continue;
        }
        cv::circle(image, cv::Point(x, y), 5, coco17_mapper[i].second, 1);
        // std::cout << "keypoint: " << i << std::endl;
        // std::cout << coco17_mapper[i].first << std::endl;
        // cv::putText(image, coco17[i].first, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    for(size_t i = 0; i < coco17_skeleton.size(); i++){
        auto [start, end] = coco17_skeleton[i];
        auto [x1, y1, exist1] = keypoints[start];
        auto [x2, y2, exist2] = keypoints[end];
        if (exist1 == 0 || exist2 == 0){
            continue;
        }
        cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), coco17_mapper[i].second, 2);
    }
    return image;
}




} //namespace gusto_humanpose