#ifndef CUSTOM_UNITY_API_H
#define CUSTOM_UNITY_API_H

#include "BaseONNX.h"
#include "detector2d_family.h"
#include "face_model_family.h"
#include "human_pose_family.h"
#include "human_seg_family.h"
#include "two_stage_human_pose_extractor_2d.h"
#include "utils.h"

#include <onnxruntime_cxx_api.h>

#define CUSTOM_RET int

extern "C" {

CUSTOM_API CUSTOM_RET Custom_Model_Compile(
    BaseONNX** model_ptr,
    const char* model_path,
    const char* config_path) noexcept;
CUSTOM_API CUSTOM_RET Custom_Model_Inference_Image(
    BaseONNX* model_ptr,
    const char* image_path) noexcept;
CUSTOM_API CUSTOM_RET Custom_Model_Inference(
    BaseONNX* model_ptr,
    unsigned char* bitmap,
    int height,
    int width) noexcept;
CUSTOM_API CUSTOM_RET Custom_Model_Destroy(BaseONNX* model_ptr) noexcept;

CUSTOM_API CUSTOM_RET Custom_Human_Pose_Pipeline_Compile(
    HumanPoseExtractor2D** model_ptr,
    const char* detector_path,
    const char* detector_config_path,
    const char* pose_model_path,
    const char* pose_model_config_path,
    int detect_interval) noexcept;
CUSTOM_API CUSTOM_RET Custom_Human_Pose_Pipeline_Inference(
    HumanPoseExtractor2D* model_ptr,
    char* bitmap,
    int height,
    int width,
    bool display_box = true,
    bool display_keypoints = true) noexcept;
CUSTOM_API CUSTOM_RET Custom_Human_Pose_Pipeline_Destroy(
    HumanPoseExtractor2D* model_ptr) noexcept;

}  // extern "C"

#endif  // CUSTOM_UNITY_API_H
