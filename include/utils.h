#ifndef GUSTO_UTILS_H
#define GUSTO_UTILS_H

#include <iostream>
#include <optional>
#include <utility>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

#if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
#include <omp.h>
#endif

namespace GustoStatus {
    const int ERR_OK = 0x00000000;
    const int ERR_GENERAL_ERROR = 0x76000000;
    const int ERR_GENERAL_FILE_IO = -(ERR_GENERAL_ERROR | 0x00000001);
    const int ERR_GENERAL_IMAGE_LOAD = -(ERR_GENERAL_ERROR | 0x00000002);
    const int ERR_GENERAL_SERIALIZATION = -(ERR_GENERAL_ERROR | 0x00000003);
    const int ERR_GENERAL_INVALID_PARAMETER = -(ERR_GENERAL_ERROR | 0x00000004);
    const int ERR_GENERAL_NOT_SUPPORT = -(ERR_GENERAL_ERROR | 0x00000005);
    const int ERR_PARTIAL_FAIL = 0x76000001;
}

// [Sombra] -> nlohamnn::json as the return type is not valid for C# Unity Side, I guess it's some non-standard c++ feature implemented by nlohmann
// class GustoSerializer 
// {
//     public:
//         nlohmann::json load_json(const char* filename);
//         // nlohmann::json load_json(const std::string& filename);
// };


class Net_config
{
    public:
        Net_config();
        Net_config(int inpHeight, int inpWidth,
                    float confThreshold, float nmsThreshold, 
                    const char* modelpath, const char* cls_names_path, int len_string = 1024);
        Net_config(const Net_config& config);

        float confThreshold; // Confidence threshold
        float nmsThreshold;  // Non-maximum suppression threshold

        int inpHeight;
        int inpWidth;
        
        std::string cls_names_path;
        std::string modelpath;
        std::vector<std::string> class_names;
        int num_class;

};

#endif