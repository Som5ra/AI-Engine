#ifndef BASE_ONNX_H
#define BASE_ONNX_H
#include "utils.h"
#include <thread>
#include <onnxruntime_cxx_api.h>


struct basic_model_config{
    std::string model_name;
    std::string model_path;
    std::pair<int, int> input_size;
    std::map<int, std::string> class_mapper;
};


class BaseONNX {
public:
    BaseONNX(const std::string& model_path, const std::string& model_name);
    
    virtual ~BaseONNX() {
        for (const char* name : input_names) {
            free(const_cast<char*>(name));
        }
        for (const char* name : output_names) {
            free(const_cast<char*>(name));
        }
    }

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