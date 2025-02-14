
#include "utils.h"
#include "BaseONNX.h"
#include "detector2d_family.h"
#include "human_seg_family.h"
#include "human_pose_family.h"
#include "face_model_family.h"
#include "two_stage_human_pose_extractor_2d.h"
#include <onnxruntime_cxx_api.h>

extern "C"
{

    #define GUSTO_RET int


    GUSTO_RET Gusto_Model_Compile(BaseONNX** model_ptr, const char* model_path, const char* config_path);
    
    // Image Path Inference
    GUSTO_RET Gusto_Model_Inference_Image(BaseONNX* model_ptr, const char* image_path);

    // WebCamTexture Inference
    GUSTO_RET Gusto_Model_Inference(BaseONNX* model_ptr, unsigned char* bitmap, int height, int width);


    // PIPELINES
    GUSTO_RET Gusto_Human_Pose_Pipeline_Compile(
        HumanPoseExtractor2D** model_ptr,
        const char* detector_path, 
        const char* config_path, 
        const char* pose_path, 
        const char* pose_config_path, 
        int detect_interval
    );

    GUSTO_RET Gusto_Human_Pose_Pipeline_Inference(HumanPoseExtractor2D* model_ptr, char* bitmap, int height, int width, bool display_box = true, bool display_kpts = true);




}