
#include "BaseONNX.h"


BaseONNX::BaseONNX(const std::string& model_path, const std::string& model_name)
    : ort_env(ORT_LOGGING_LEVEL_WARNING, model_name.c_str()),
        ort_session(nullptr) {
    std::cout << "[Debug] -> " << "Before session_options" << std::endl;
    
    Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(1);
    // session_options.SetInterOpNumThreads(1); 
    std::cout << "[Debug] -> " << "Ort::Session" << std::endl;
    ort_session = Ort::Session(ort_env, model_path.c_str(), session_options);
    std::cout << "[Debug] -> " << "After session_options" << std::endl;

    size_t input_count = ort_session.GetInputCount();
    size_t output_count = ort_session.GetOutputCount();

    std::cout << "[Debug] -> " << "Before input_names" << std::endl;
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
    }
    std::cout << "[Debug] -> " << "After input_names" << std::endl;
    for (size_t i = 0; i < output_count; ++i) {
        Ort::AllocatedStringPtr output_name_ptr = ort_session.GetOutputNameAllocated(i, ort_allocator);
        const char* output_name = strdup(output_name_ptr.get());
        output_names.push_back(output_name);
    }
    std::cout << "[Debug] -> " << "Initialize End" << std::endl;

}
