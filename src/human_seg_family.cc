#include "human_seg_family.h"

namespace gusto_humanseg{

std::map<std::string, model_lib> MODEL_NAME_LIB_MAPPER = {
    {"selfie_multiclass_256x256", model_lib::selfie_multiclass_256x256},
    {"selfie_segmenter", model_lib::selfie_segmenter},
    {"selfie_segmenter_landscape", model_lib::selfie_segmenter_landscape}, // trash
    {"deeplab_v3", model_lib::deeplab_v3} // trash
};

std::map<int, cv::Vec3b> CLASS_COLOR_MAPPER = {
    {0, cv::Vec3b(0, 0, 0)},       // background
    {1, cv::Vec3b(0, 128, 0)},     // hair
    {2, cv::Vec3b(128, 0, 0)},     // body-skin
    {3, cv::Vec3b(0, 0, 128)},     // face-skin
    {4, cv::Vec3b(128, 128, 0)},   // clothes
    {5, cv::Vec3b(0, 128, 128)}    // others (accessories)
};

// std::unique_ptr<seg_config> fetch_model_config(const std::string _model_name){
std::unique_ptr<basic_model_config> fetch_model_config(std::unique_ptr<basic_model_config>& base_config){
    std::cout <<  "input model name: " << base_config->model_name << std::endl;
    model_lib _model_type;
    try{
        // _config->model_type = MODEL_NAME_LIB_MAPPER[_model_name];
        _model_type = MODEL_NAME_LIB_MAPPER[base_config->model_name];
    }catch(const std::exception& e){
        std::cerr << "input model name is not valid! " << std::endl;
        return nullptr;
    }

    if(_model_type == model_lib::selfie_multiclass_256x256){
        // _config->model_path = "selfie_multiclass_256x256.onnx";
        base_config->input_size = std::make_pair(256, 256);
        base_config->class_mapper = {
            {0, "background"},
            {1, "hair"},
            {2, "body-skin"},
            {3, "face-skin"},
            {4, "clothes"},
            {5, "others (accessories)"}
        };
    }
    if(_model_type == model_lib::selfie_segmenter){
        // base_config->model_path = "selfie_segmenter_refactor.onnx";
        base_config->input_size = std::make_pair(256, 256);
        base_config->class_mapper = {
            {0, "background"}, // [Sombra] -> added by me, easy to postprocess 
            {1, "person"}
        };
    }
    if(_model_type == model_lib::selfie_segmenter_landscape){
        // base_config->model_path = "selfie_segmentation_landscape_refactor_fixed.onnx";
        base_config->input_size = std::make_pair(256, 144),
        base_config->class_mapper = {
            {0, "background"}, // [Sombra] -> added by me, easy to postprocess 
            {1, "person"}
        };
    }
    if(_model_type == model_lib::deeplab_v3){
        // base_config->model_path = "deeplab_v3.onnx";
        base_config->input_size = std::make_pair(257, 257);
        base_config->class_mapper = {
            {0, "background"},
            {1, "person"},
            {2, "cat"},
            {3, "dog"},
            {4, "potted plant"}
        };
    }
    std::cout << "model path: " << base_config->model_path << std::endl;
    return std::move(base_config);
}


Segmenter::Segmenter(const std::string& model_path, const std::string& config_path)
    : BaseONNX(model_path, config_path) {
    _seg_config = std::move(fetch_model_config(this->_config));
}

Segmenter::Segmenter(std::unique_ptr<basic_model_config> _config)
    : BaseONNX(std::move(_config)) {
    _seg_config = std::move(fetch_model_config(this->_config));
}

// cv::Mat Segmenter::preprocess_img(const cv::Mat& image) {
//     cv::Mat frame;
//     cv::cvtColor(image, frame, cv::COLOR_BGR2RGB);
//     cv::resize(frame, frame, cv::Size(_config->input_size.second, _config->input_size.first), 0, 0, cv::INTER_LINEAR);
//     frame.convertTo(frame, CV_32FC3, 1.0 / 127.5, -1); // NHWC
//     return frame;
// }

std::unique_ptr<PostProcessResult> Segmenter::forward(const cv::Mat& raw) {
    // cv::Mat frame = preprocess_img(raw);
    std::vector<float> input_tensor_values = preprocess(raw);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), inputTensorSize, input_shape[0].data(), input_shape[0].size());

    std::vector<Ort::Value> output_tensors = ort_session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);


    std::unique_ptr<SegmentationResult> ret = std::make_unique<SegmentationResult>();
    ret->mask = postprocess(output_tensors, std::make_pair(raw.rows, raw.cols));
    return std::move(ret);
    // return output_tensors;

}

cv::Mat Segmenter::postprocess(const std::vector<Ort::Value>& mask_out, std::pair<int, int> target_size) {
    const float* prob = mask_out[0].GetTensorData<float>();
    auto outputInfo = mask_out[0].GetTensorTypeAndShapeInfo();
    int batch_size = outputInfo.GetShape()[0];
    int height = outputInfo.GetShape()[1];
    int width = outputInfo.GetShape()[2];
    int channels = outputInfo.GetShape()[3];

    std::vector<cv::Mat> channel_mats(channels);

    #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
    omp_set_num_threads(std::max(1, omp_get_max_threads() / 2));
    #pragma omp parallel for
    #endif
    for (int c = 0; c < channels; ++c) {
        channel_mats[c] = cv::Mat(height, width, CV_32F);
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                channel_mats[c].at<float>(h, w) = prob[c + channels * (w + width * h)];
            }
        }
        cv::resize(channel_mats[c], channel_mats[c], cv::Size(target_size.second, target_size.first), 0, 0, cv::INTER_LINEAR);
    }

    // Create a cv::Mat to hold the argmax results
    cv::Mat argmax_mat(target_size.first, target_size.second, CV_8U);

    if (channels == 1){
        #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
        omp_set_num_threads(std::max(1, omp_get_max_threads() / 2));
        #pragma omp parallel for
        #endif
        for (int h = 0; h < target_size.first; ++h) {
            for (int w = 0; w < target_size.second; ++w) {
                float max_val = channel_mats[0].at<float>(h, w);
                int max_idx = max_val >= 0.5 ? 1 : 0;
                argmax_mat.at<uchar>(h, w) = static_cast<uchar>(max_idx);
            }
        }
    }else{
        #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
        omp_set_num_threads(std::max(1, omp_get_max_threads() / 2));
        #pragma omp parallel for
        #endif
        for (int h = 0; h < target_size.first; ++h) {
            for (int w = 0; w < target_size.second; ++w) {
                float max_val = channel_mats[0].at<float>(h, w);
                int max_idx = 0;
                for (int c = 1; c < channels; ++c) {
                    float val = channel_mats[c].at<float>(h, w);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = c;
                    }
                }
                argmax_mat.at<uchar>(h, w) = static_cast<uchar>(max_idx);
            }
        }
    }



    cv::Mat colorized_output(target_size.first, target_size.second, CV_8UC3);
    // Create a colorized output image
    #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
    omp_set_num_threads(std::max(1, omp_get_max_threads() / 2));
    #pragma omp parallel for
    #endif
    for (int h = 0; h < target_size.first; ++h) {
        for (int w = 0; w < target_size.second; ++w) {
            int class_idx = argmax_mat.at<uchar>(h, w);
            colorized_output.at<cv::Vec3b>(h, w) = CLASS_COLOR_MAPPER[class_idx];
        }
    }

    return colorized_output;
}

} //namespace gusto_humanseg