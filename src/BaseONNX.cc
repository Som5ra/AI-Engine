
#include "BaseONNX.h"

std::unique_ptr<basic_model_config> BaseONNX::ParseConfig(const std::string& model_path, const std::string& config_path){
    std::unique_ptr<basic_model_config> _config = std::make_unique<basic_model_config>();
    nlohmann::json data;

    try{
        data = nlohmann::json::parse(std::ifstream(config_path.c_str()));
    }catch (const std::exception& e) {
        std::cerr << "Error parsing json file: " << config_path << std::endl;
    }
    try
    {
        _config->model_path = model_path;
        _config->model_name = data["model_name"].get<std::string>();
        _config->input_size = std::make_pair(data["input_size"]["width"].get<int>(), data["input_size"]["height"].get<int>());
        _config->result_type = data["result_type"].get<ResultType>();
        size_t cls_id = 0;
        for(auto& it : data["classes"]) {
            _config->class_mapper[cls_id++] = it.get<std::string>();
        }
        _config->dim_order = data["Preprocessing"]["DIM_ORDER"].template get<DimOrder>();
        _config->channel_order = data["Preprocessing"]["CHANNEL_ORDER"].template get<ChannelOrder>();


        _config->channels = data["input_size"]["channels"].get<int>();

        std::string _provider = data["execution_provider"].get<std::string>();
        
        if (_provider == "CPU") {
            _config->provider = ProviderType::CPU;
        } else if (_provider == "XNNPACK") {
            _config->provider = ProviderType::XNNPACK;
        } else if (_provider == "COREML") {
            #if defined(BUILD_PLATFORM_IOS)
                _config->provider = ProviderType::COREML;
            #else
                std::cerr << "COREML provider is not available, would use CPU provider instead" << std::endl;
            #endif
        } else if (_provider == "NNAPI") {
            #if defined(BUILD_PLATFORM_ANDROID)
                _config->provider = ProviderType::NNAPI;
            #else
                std::cerr << "NNAPI provider is not available, would use CPU provider instead" << std::endl;
            #endif
        } else {
            std::cerr << "Provider name is not correct, would use CPU provider instead" << std::endl;
            _config->provider = ProviderType::CPU;
        }

    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        throw std::runtime_error("[Sombra] -> Error parsing json file: " + config_path);
    }

    std::cout << "==================== CONFIG ====================" << std::endl;
    std::cout << "Model Name: " << _config->model_name << std::endl;
    std::cout << "Input Size: " << _config->input_size.first << "x" << _config->input_size.second << "x" << _config->channels << std::endl;
    for (auto& it : _config->class_mapper) {
        std::cout << "Class: " << it.first << " -> " << it.second << std::endl;
    }
    std::cout << "Result Type: " << _config->result_type << std::endl;
    std::cout << "Dim Order: " << _config->dim_order << std::endl;
    std::cout << "Channel Order: " << _config->channel_order << std::endl;
    std::cout << "==================== CONFIG ====================" << std::endl;
    return std::move(_config);
}


void BaseONNX::Compile(){

    Ort::SessionOptions session_options;
    
    switch (this->_config->provider)
    {
        case ProviderType::CPU:
            std::cout << "[Provider] CPU" << std::endl;
            break;
        case ProviderType::XNNPACK:
            session_options.AppendExecutionProvider("XNNPACK");
            std::cout << "[Provider] XNNPACK" << std::endl;
            break;
        case ProviderType::COREML:
            #if defined(BUILD_PLATFORM_IOS)
                uint32_t coreml_flags = 0;
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags));
                std::cout << "[Provider] COREML" << std::endl;
            #else
                std::cerr << "COREML provider is not available, would use CPU provider instead" << std::endl;
            #endif
            break;
        case ProviderType::NNAPI:
            #if defined(BUILD_PLATFORM_ANDROID)
                // Please see
                // https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html#usage
                // to enable different flags
                uint32_t nnapi_flags = 0;
                // nnapi_flags |= NNAPI_FLAG_USE_FP16;
                // nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
                OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, nnapi_flags);
                std::cout << "[Provider] NNAPI" << std::endl;

            #else
                std::cerr << "NNAPI provider is not available, would use CPU provider instead" << std::endl;
            #endif
            break;
    }
    session_options.SetInterOpNumThreads(1);
    session_options.SetIntraOpNumThreads(std::min(6, (int) std::thread::hardware_concurrency()));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    this->ort_session = Ort::Session(this->ort_env, _config->model_path.c_str(), session_options);
    std::cout << "Model Loaded: " << _config->model_path << std::endl;
    size_t input_count = ort_session.GetInputCount();
    size_t output_count = ort_session.GetOutputCount();
    std::cout << "Number of inputs: " << input_count << std::endl;
    for (size_t i = 0; i < input_count; ++i) {
        Ort::AllocatedStringPtr input_name_ptr = ort_session.GetInputNameAllocated(i, ort_allocator);
        const char* input_name = strdup(input_name_ptr.get());
        input_names.push_back(input_name);
    }
    std::cout << "Number of outputs: " << output_count << std::endl;
    for (size_t i = 0; i < output_count; ++i) {
        Ort::AllocatedStringPtr output_name_ptr = ort_session.GetOutputNameAllocated(i, ort_allocator);
        const char* output_name = strdup(output_name_ptr.get());
        output_names.push_back(output_name);
    }

    inputTensorSize = this->_config->input_size.first * this->_config->input_size.second * this->_config->channels;
    // [Sombra] -> We only accpet singular input so far, so total length of input_shape is 1
    if (_config->dim_order == DimOrder::NHWC) {
        input_shape.push_back({1, _config->input_size.first, _config->input_size.second, _config->channels});
    } else {
        input_shape.push_back({1, _config->channels, _config->input_size.first, _config->input_size.second});
    }
}

BaseONNX::BaseONNX(const std::string& model_path, const std::string& config_path)
    : ort_env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"),
        ort_session(nullptr) {
    this->_config = std::move(ParseConfig(model_path, config_path));
    this->Compile();
}

BaseONNX::BaseONNX(std::unique_ptr<basic_model_config> _config)
    : ort_env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"),
        ort_session(nullptr) {
    this->_config = std::move(_config);
    this->Compile();
}


std::vector<float> BaseONNX::preprocess(const cv::Mat& image) {
    cv::Mat frame;
    if (_config->channel_order == ChannelOrder::BGR) {
        cv::cvtColor(image, frame, cv::COLOR_RGB2BGR);
    } else {
        frame = image.clone();
    }
    cv::resize(frame, frame, cv::Size(_config->input_size.first, _config->input_size.second), 0, 0, cv::INTER_LINEAR);
    frame.convertTo(frame, CV_32FC3, 1.0 / 127.5, -1);
    
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
                if (_config->dim_order == DimOrder::NHWC) {
                    input_tensor_values[i * w * rgbsplit.size() + j * rgbsplit.size() + k] = static_cast<float>(rgbsplit[k].at<float>(i, j)); // NHWC
                } else {
                    input_tensor_values[k * h * w + i * w + j] = static_cast<float>(rgbsplit[k].at<float>(i, j)); // NCHW
                }
            }
        }
    }
    return input_tensor_values;
}