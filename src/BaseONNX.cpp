
#include "BaseONNX.h"


BaseONNX::BaseONNX(const std::string& model_path, const std::string& model_name)
    : ort_env(ORT_LOGGING_LEVEL_WARNING, model_name.c_str()),
        ort_session(nullptr) {
    
    Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(1);
    // session_options.SetInterOpNumThreads(1); 
    ort_session = Ort::Session(ort_env, model_path.c_str(), session_options);

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
    }
    for (size_t i = 0; i < output_count; ++i) {
        Ort::AllocatedStringPtr output_name_ptr = ort_session.GetOutputNameAllocated(i, ort_allocator);
        const char* output_name = strdup(output_name_ptr.get());
        output_names.push_back(output_name);
    }

}
