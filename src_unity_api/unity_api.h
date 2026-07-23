#ifndef GUSTO_UNITY_API_H
#define GUSTO_UNITY_API_H

#include "BaseONNX.h"
#include "detector2d_family.h"
#include "face_model_family.h"
#include "human_pose_family.h"
#include "human_seg_family.h"
#include "two_stage_human_pose_extractor_2d.h"
#include "utils.h"

#include <onnxruntime_cxx_api.h>

#define GUSTO_RET int

extern "C" {

GUSTO_API GUSTO_RET Gusto_Model_Compile(
    BaseONNX** model_ptr,
    const char* model_path,
    const char* config_path) noexcept;
GUSTO_API GUSTO_RET Gusto_Model_Inference_Image(
    BaseONNX* model_ptr,
    const char* image_path) noexcept;
GUSTO_API GUSTO_RET Gusto_Model_Inference(
    BaseONNX* model_ptr,
    unsigned char* bitmap,
    int height,
    int width) noexcept;
GUSTO_API GUSTO_RET Gusto_Model_Destroy(BaseONNX* model_ptr) noexcept;

GUSTO_API GUSTO_RET Gusto_Human_Pose_Pipeline_Compile(
    HumanPoseExtractor2D** model_ptr,
    const char* detector_path,
    const char* detector_config_path,
    const char* pose_model_path,
    const char* pose_model_config_path,
    int detect_interval) noexcept;
GUSTO_API GUSTO_RET Gusto_Human_Pose_Pipeline_Inference(
    HumanPoseExtractor2D* model_ptr,
    char* bitmap,
    int height,
    int width,
    bool display_box = true,
    bool display_keypoints = true) noexcept;
GUSTO_API GUSTO_RET Gusto_Human_Pose_Pipeline_Destroy(
    HumanPoseExtractor2D* model_ptr) noexcept;

}  // extern "C"

#endif  // GUSTO_UNITY_API_H
