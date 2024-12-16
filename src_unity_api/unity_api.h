
#include "utils.h"
#include "BaseONNX.h"
#include "detector2d_family.h"
#include "human_seg_family.h"
#include "human_pose_family.h"
#include "face_model_family.h"

#include <onnxruntime_cxx_api.h>
extern "C"
{

    #define GUSTO_RET int


    GUSTO_RET Gusto_Model_Compile(BaseONNX** model_ptr, const char* model_path, const char* config_path);
    
    // Image Path Inference
    GUSTO_RET Gusto_Model_Inference_Image(BaseONNX* model_ptr, const char* image_path);

    // WebCamTexture Inference
    GUSTO_RET Gusto_Model_Inference(BaseONNX* model_ptr, unsigned char* bitmap, int height, int width);
}