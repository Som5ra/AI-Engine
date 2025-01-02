#ifndef BASE_ONNX_H
#define BASE_ONNX_H
#include "utils.h"
#include <thread>
#include <onnxruntime_cxx_api.h>


#if defined(BUILD_PLATFORM_LINUX) && defined(DEBUG)
#include <opencv2/opencv.hpp>
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#if defined(BUILD_PLATFORM_IOS)
#include "coreml_provider_factory.h"  // NOLINT
#endif

#if defined(BUILD_PLATFORM_ANDROID)
#include "nnapi_provider_factory.h"  // NOLINT
#endif

enum DimOrder {
    NCHW,
    NHWC,
};

NLOHMANN_JSON_SERIALIZE_ENUM( DimOrder, {
    {NCHW, nullptr},
    {NCHW, "NCHW"},
    {NHWC, "NHWC"},
})

enum ChannelOrder {
    RGB,
    BGR,
};

NLOHMANN_JSON_SERIALIZE_ENUM( ChannelOrder, {
    {RGB, nullptr},
    {RGB, "RGB"},
    {BGR, "BGR"},
})


enum ResultType {
    DetectorResultType,

    SegmentationResultType,
    KeyPointResultType,

    MediaPipeDetectorResultType,
    MediapipeFaceLandmarkResultType,
};

NLOHMANN_JSON_SERIALIZE_ENUM( ResultType, {
    {DetectorResultType, nullptr},
    {DetectorResultType, "detection"},

    {SegmentationResultType, "segmentation"},
    {KeyPointResultType, "keypoint"},

    {MediaPipeDetectorResultType, "mp_detection"},
    {MediapipeFaceLandmarkResultType, "mp_keypoint"},
})

enum ProviderType {
    CPU,
    XNNPACK,
    COREML,
    NNAPI
};



struct basic_model_config{
    std::string model_name;
    std::string model_path;
    std::pair<int, int> input_size;
    int channels;
    std::map<int, std::string> class_mapper;

    ResultType result_type;
    DimOrder dim_order = DimOrder::NCHW;
    ChannelOrder channel_order = ChannelOrder::RGB;

    ProviderType provider = ProviderType::CPU;

};

class PostProcessResult {
public:
    virtual ~PostProcessResult(){};
};


class BaseONNX {
public:
    BaseONNX(const std::string& model_path, const std::string& config_path);
    BaseONNX(std::unique_ptr<basic_model_config> _config);
    virtual ~BaseONNX() {
        for (const char* name : input_names) {
            free(const_cast<char*>(name));
        }
        for (const char* name : output_names) {
            free(const_cast<char*>(name));
        }
    }

    virtual std::vector<float> preprocess(const cv::Mat& image);
    // virtual std::vector<Ort::Value> forward(const cv::Mat& raw) = 0;
    // virtual std::unique_ptr<PostProcessResult> postprocess(const std::vector<Ort::Value>& net_out) = 0;
    virtual std::unique_ptr<PostProcessResult> forward(const cv::Mat& raw) = 0;
    // virtual ResultType getResultType() const = 0;
    std::unique_ptr<basic_model_config> _config;

    static std::unique_ptr<basic_model_config> ParseConfig(const std::string& model_path, const std::string& config_path);
    void Compile();
protected:
    Ort::Env ort_env;
    Ort::Session ort_session;
    Ort::AllocatorWithDefaultOptions ort_allocator;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_shape;
    size_t inputTensorSize;

    static float sigmoid(float x) {
        return 1 / (1 + exp(-x));
    }

    static std::vector<float> sigmoid(const std::vector<float>& x) {
        std::vector<float> result(x.size());
        #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
        omp_set_num_threads(std::max(1, omp_get_max_threads() / 2));
        #pragma omp parallel for
        #endif
        for (int i = 0; i < x.size(); i++) {
            result[i] = 1 / (1 + exp(-x[i]));
        }
        return result;
    }
};

#endif // BASE_ONNX_H