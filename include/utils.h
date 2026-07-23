#ifndef CUSTOM_UTILS_H
#define CUSTOM_UTILS_H

#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(AI_ENGINE_HAS_OPENMP)
#include <omp.h>
#endif

#if defined(_WIN32)
#define CUSTOM_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define CUSTOM_API __attribute__((visibility("default")))
#else
#define CUSTOM_API
#endif

namespace CustomStatus {

constexpr int ERR_OK = 0x00000000;
constexpr int ERR_GENERAL_ERROR = 0x76000000;
constexpr int ERR_GENERAL_FILE_IO = -(ERR_GENERAL_ERROR | 0x00000001);
constexpr int ERR_GENERAL_IMAGE_LOAD = -(ERR_GENERAL_ERROR | 0x00000002);
constexpr int ERR_GENERAL_SERIALIZATION = -(ERR_GENERAL_ERROR | 0x00000003);
constexpr int ERR_GENERAL_INVALID_PARAMETER =
    -(ERR_GENERAL_ERROR | 0x00000004);
constexpr int ERR_GENERAL_NOT_SUPPORT =
    -(ERR_GENERAL_ERROR | 0x00000005);
constexpr int ERR_PARTIAL_FAIL = 0x76000001;

}  // namespace CustomStatus

struct CustomRect {
    float x1;
    float y1;
    float x2;
    float y2;
    float conf = 0.0F;
    int label = -1;

    CustomRect(float x1, float y1, float x2, float y2)
        : x1(x1), y1(y1), x2(x2), y2(y2) {}

    CustomRect(
        float x1,
        float y1,
        float x2,
        float y2,
        float conf,
        int label)
        : x1(x1),
          y1(y1),
          x2(x2),
          y2(y2),
          conf(conf),
          label(label) {}

    float area() const {
        return (y2 - y1 + 1.0F) * (x2 - x1 + 1.0F);
    }
};

class Net_config {
public:
    Net_config();
    Net_config(
        int inpHeight,
        int inpWidth,
        float confThreshold,
        float nmsThreshold,
        const char* modelpath,
        const char* cls_names_path,
        int len_string = 1024);
    Net_config(const Net_config& config);

    float confThreshold;
    float nmsThreshold;
    int inpHeight;
    int inpWidth;
    std::string cls_names_path;
    std::string modelpath;
    std::vector<std::string> class_names;
    int num_class;
};

#endif  // CUSTOM_UTILS_H
